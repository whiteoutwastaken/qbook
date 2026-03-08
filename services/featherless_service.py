"""
featherless_service.py
───────────────────────
Handles all Featherless.ai API calls for the Whisper app.

RESPONSIBILITIES:
  1. generate_lesson_text() — Generate lesson when OpenNote has no journal
  2. enrich_lesson()        — Image queries + source summaries
  3. chunk_lesson()         — Split into speakable TTS chunks
  4. analyze_face()         — Vision model facial expression detection
  5. simplify_chunk()       — Re-explain confused chunks

MODELS:
  LLM:    Qwen/Qwen2.5-72B-Instruct
  Vision: google/gemma-3-27b-it

Install:
  pip install openai python-dotenv

.env:
  FEATHERLESS_API_KEY=your_key_here
"""

import os
import re
import json
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv

from models import LessonContent

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────
LLM_MODEL    = "Qwen/Qwen2.5-72B-Instruct"
VISION_MODEL = "google/gemma-3-27b-it"

# ── Facial state type ─────────────────────────────────────────────────────────
FaceState = Literal["confused", "focused", "nodding", "distracted", "eureka", "unknown"]


# ── Facial Analysis Result ────────────────────────────────────────────────────

@dataclass
class FaceAnalysis:
    state:      FaceState
    confidence: float
    note:       str = ""

    def is_confused(self)   -> bool: return self.state == "confused"
    def is_nodding(self)    -> bool: return self.state == "nodding"
    def is_focused(self)    -> bool: return self.state == "focused"
    def is_distracted(self) -> bool: return self.state == "distracted"
    def is_eureka(self)     -> bool: return self.state == "eureka"


# ── Main Service ──────────────────────────────────────────────────────────────

class FeatherlessService:

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FEATHERLESS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Featherless API key not found. "
                "Set FEATHERLESS_API_KEY in your .env file."
            )
        self.client = OpenAI(
            base_url="https://api.featherless.ai/v1",
            api_key=self.api_key,
        )

    # ── 1. Generate Lesson Text ───────────────────────────────────────────────

    def generate_lesson_text(self, topic: str) -> str:
        """Generates a full plain-text lesson on the topic."""
        print(f"[Featherless] Generating lesson text for: '{topic}'")
        response = self._chat(
            system=(
                "You are a skilled tutor. Generate clear, concise lesson content "
                "optimized for text-to-speech delivery. Plain prose only — no "
                "bullet points, no markdown, no headers. Write as if speaking "
                "naturally to a student. After the lesson, on a new line write "
                "'IMAGES:' followed by 2-3 comma-separated image search queries "
                "that would visually help explain this topic."
            ),
            user=(
                f"Generate a clear, beginner-friendly lesson on: {topic}\n\n"
                f"Structure it as 4-6 short paragraphs, each covering one key concept. "
                f"Plain conversational prose only. After the lesson write:\n"
                f"IMAGES: [2-3 comma-separated image search queries for this topic]"
            ),
            max_tokens=800,
        )
        return response.strip()

    # ── 2. Enrich Lesson ──────────────────────────────────────────────────────

    def enrich_lesson(self, lesson: LessonContent) -> LessonContent:
        """Adds image queries and source summaries to the lesson."""
        print(f"[Featherless] Enriching lesson: '{lesson.topic}'")

        lesson.text, inline_images = self._extract_inline_images(lesson.text)
        if inline_images:
            lesson.image_queries.extend(inline_images)

        prompt = (
            f"Topic: {lesson.topic}\n\n"
            f"Lesson text:\n{lesson.text[:1500]}\n\n"
            f"Decide if images would significantly help a student understand this topic.\n"
            f"Also decide if a brief background summary would help contextualize it.\n\n"
            f"Respond ONLY in this exact JSON format:\n"
            f'{{\n'
            f'  "needs_images": true or false,\n'
            f'  "image_queries": ["query1", "query2"],\n'
            f'  "needs_summary": true or false,\n'
            f'  "source_summaries": ["One sentence summary of key background context."]\n'
            f'}}'
        )

        try:
            raw = self._chat(
                system=(
                    "You are an educational content assistant. Respond ONLY with "
                    "valid JSON. No preamble, no explanation, no markdown code blocks."
                ),
                user=prompt,
                max_tokens=400,
            )
            data = self._parse_json(raw)
            if data.get("needs_images") and data.get("image_queries"):
                lesson.image_queries.extend(data["image_queries"])
            if data.get("needs_summary") and data.get("source_summaries"):
                lesson.source_summaries.extend(data["source_summaries"])
        except Exception as e:
            print(f"[Featherless] Enrichment failed: {e} — continuing without visuals")

        lesson.image_queries   = self._dedupe_queries(lesson.image_queries)
        lesson.source_summaries = list(dict.fromkeys(lesson.source_summaries))

        print(f"[Featherless] Enriched — "
              f"images: {len(lesson.image_queries)}, "
              f"summaries: {len(lesson.source_summaries)}")
        return lesson

    # ── 3. Chunk Lesson ───────────────────────────────────────────────────────

    def chunk_lesson(self, lesson: LessonContent, max_chunk_words: int = 60) -> LessonContent:
        """Splits lesson text into speakable TTS chunks."""
        print(f"[Featherless] Chunking lesson: '{lesson.topic}'")

        prompt = (
            f"Split the following lesson into short, speakable chunks for "
            f"text-to-speech delivery. Each chunk should be 1-3 sentences and "
            f"no more than {max_chunk_words} words. Each chunk must be a complete "
            f"thought. Do not summarize or alter the text.\n\n"
            f"Respond ONLY as a JSON array of strings:\n"
            f'["chunk one text here", "chunk two text here", ...]\n\n'
            f"Lesson:\n{lesson.text}"
        )

        try:
            raw = self._chat(
                system=(
                    "You are a text formatting assistant. Split text into chunks "
                    "exactly as instructed. Respond ONLY with a valid JSON array "
                    "of strings. No preamble, no markdown."
                ),
                user=prompt,
                max_tokens=1500,
            )
            chunks = self._parse_json(raw)
            if isinstance(chunks, list) and all(isinstance(c, str) for c in chunks):
                lesson.chunks = [c.strip() for c in chunks if c.strip()]
                print(f"[Featherless] {len(lesson.chunks)} chunks created")
            else:
                raise ValueError("Response was not a list of strings")
        except Exception as e:
            print(f"[Featherless] Chunking failed: {e} — falling back to simple split")
            lesson.chunks = self._simple_chunk(lesson.text, max_chunk_words)

        return lesson

    # ── 4. Facial Analysis ────────────────────────────────────────────────────

    def analyze_face(
        self,
        frame_path:   str = None,
        frame_base64: str = None,
        frame_format: str = "jpeg",
    ) -> FaceAnalysis:
        """
        Analyzes a webcam frame (YOLO-cropped face) to detect expression.
        Accepts either a file path or a base64 string.
        """
        if not frame_path and not frame_base64:
            return FaceAnalysis(state="unknown", confidence=0.0, note="No frame provided")

        if frame_base64:
            image_data = frame_base64
        else:
            image_data = self._encode_image(frame_path)
            ext = Path(frame_path).suffix.lower().strip(".")
            frame_format = "jpeg" if ext in ("jpg", "jpeg") else ext

        data_url = f"data:image/{frame_format};base64,{image_data}"

        try:
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                max_tokens=150,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This is a tight crop of a person's face, preprocessed for expression analysis. "
                                    "Determine their current state from ONLY these options: "
                                    "confused, focused, nodding, distracted, eureka.\n\n"
                                    "CAMERA NOTE: The camera may be above, left, or right of the screen. "
                                    "Slight gaze offset is NORMAL — do not count it as distracted.\n\n"
                                    "- focused: neutral, relaxed, no strong reaction\n"
                                    "- confused: 2+ signals needed — furrowed brow AND squinting, "
                                    "or head tilt AND lip press, or visibly lost expression\n"
                                    "- nodding: clear up-down head movement\n"
                                    "- distracted: fully looking away, turned head, eyes closed, "
                                    "or clearly looking at something else entirely\n"
                                    "- eureka: VERY sensitive — any single signal is enough: "
                                    "eyes widening even slightly, tiniest hint of a smile, "
                                    "one eyebrow raising, subtle 'oh' mouth, any micro-expression "
                                    "of recognition. When in doubt between focused and eureka, "
                                    "always choose eureka.\n\n"
                                    "Respond ONLY in this exact JSON format:\n"
                                    '{"state": "focused", "confidence": 0.85, "note": "calm attentive expression"}'
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
            )

            raw  = response.choices[0].message.content.strip()
            data = self._parse_json(raw)

            state = data.get("state", "unknown")
            if state not in ("confused", "focused", "nodding", "distracted", "eureka"):
                state = "unknown"

            return FaceAnalysis(
                state=state,
                confidence=float(data.get("confidence", 0.5)),
                note=str(data.get("note", "")),
            )

        except Exception as e:
            print(f"[Featherless] Face analysis failed: {e}")
            return FaceAnalysis(state="unknown", confidence=0.0, note=str(e))

    # ── 5. Simplify Chunk ─────────────────────────────────────────────────────

    def simplify_chunk(self, chunk: str, topic: str, aggressive: bool = False) -> str:
        """Re-explains a chunk more simply when confusion is detected."""
        print(f"[Featherless] Simplifying chunk (aggressive={aggressive})")

        if aggressive:
            instruction = (
                f"The student is confused about this concept from a lesson on '{topic}'. "
                f"Re-explain it from scratch using a simple real-world analogy. "
                f"Keep it under 3 sentences. Plain prose only, no formatting."
            )
        else:
            instruction = (
                f"The student seems slightly confused by this explanation from a lesson on '{topic}'. "
                f"Rewrite it using simpler, more direct language. "
                f"Keep it under 3 sentences. Plain prose only, no formatting."
            )

        response = self._chat(
            system="You are a patient tutor who explains things simply and clearly.",
            user=f"{instruction}\n\nOriginal text:\n{chunk}",
            max_tokens=200,
        )
        return response.strip()

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _chat(self, system: str, user: str, max_tokens: int = 500) -> str:
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _parse_json(self, raw: str) -> any:
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        return json.loads(cleaned)

    def _extract_inline_images(self, text: str) -> tuple[str, list[str]]:
        lines, image_queries, clean_lines = text.split("\n"), [], []
        for line in lines:
            if line.strip().upper().startswith("IMAGES:"):
                raw_queries = line.split(":", 1)[1].strip()
                image_queries = [q.strip() for q in raw_queries.split(",") if q.strip()]
            else:
                clean_lines.append(line)
        return "\n".join(clean_lines).strip(), image_queries

    def _dedupe_queries(self, queries: list[str], max: int = 3) -> list[str]:
        seen_words, unique = [], []
        for query in queries:
            words = set(query.lower().split())
            is_duplicate = any(
                len(words & seen) / max(len(words), 1) > 0.6
                for seen in seen_words
            )
            if not is_duplicate:
                unique.append(query)
                seen_words.append(words)
            if len(unique) >= max:
                break
        return unique

    def _simple_chunk(self, text: str, max_words: int) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current, count = [], [], 0
        for sentence in sentences:
            words = len(sentence.split())
            if count + words > max_words and current:
                chunks.append(" ".join(current))
                current, count = [sentence], words
            else:
                current.append(sentence)
                count += words
        if current:
            chunks.append(" ".join(current))
        return chunks


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models import LessonContent

    service = FeatherlessService()

    lesson = LessonContent(topic="How does mitosis work?", text="")

    print("=" * 55)
    print("TEST 1: Generate lesson text")
    print("=" * 55)
    lesson.text = service.generate_lesson_text(lesson.topic)
    print(f"\nGenerated ({len(lesson.text)} chars):\n{lesson.text[:400]}...")

    print("\n" + "=" * 55)
    print("TEST 2: Enrich")
    print("=" * 55)
    lesson = service.enrich_lesson(lesson)
    print(f"Image queries:    {lesson.image_queries}")
    print(f"Source summaries: {lesson.source_summaries}")

    print("\n" + "=" * 55)
    print("TEST 3: Chunk")
    print("=" * 55)
    lesson = service.chunk_lesson(lesson)
    for i, chunk in enumerate(lesson.chunks, 1):
        print(f"  [{i}] {chunk}")

    print("\n" + "=" * 55)
    print("TEST 4: Simplify")
    print("=" * 55)
    if lesson.chunks:
        print(f"Original:   {lesson.chunks[0]}")
        print(f"Simpler:    {service.simplify_chunk(lesson.chunks[0], lesson.topic)}")
        print(f"Aggressive: {service.simplify_chunk(lesson.chunks[0], lesson.topic, aggressive=True)}")

    print("\n✓ Featherless service test complete.")