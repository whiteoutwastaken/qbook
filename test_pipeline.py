"""
test_pipeline.py
─────────────────
Tests the full OpenNote → Featherless pipeline.
Run from the project root:

    python test_pipeline.py

Make sure your .env has:
    OPENNOTE_API_KEY=your_key_here
    FEATHERLESS_API_KEY=your_key_here
"""

import sys
import os

# Make sure /services is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "services"))

from featherless_service import FeatherlessService
from opennote_service import OpenNoteService
from models import LessonContent


def test_pipeline(topic: str):
    print("\n" + "=" * 55)
    print(f"  WHISPER PIPELINE TEST")
    print(f"  Topic: '{topic}'")
    print("=" * 55)

    # ── Step 1: Init services ─────────────────────────────────────────────────
    print("\n[1/4] Initializing services...")
    featherless = FeatherlessService()
    opennote = OpenNoteService(featherless=featherless)
    print("      ✓ Services ready")

    # ── Step 2: Generate lesson (OpenNote → Featherless fallback) ─────────────
    print(f"\n[2/4] Generating lesson via OpenNote...")
    lesson = opennote.generate_lesson(topic)
    print(f"      ✓ Lesson received ({len(lesson.text)} chars)")
    print(f"      From journal: {lesson.from_journal}")
    print(f"\n── Lesson text ──")
    print(lesson.text)

    # ── Step 3: Enrich (Featherless decides images + summaries) ───────────────
    print(f"\n[3/4] Enriching lesson (images + summaries)...")
    lesson = featherless.enrich_lesson(lesson)
    print(f"      ✓ Enriched")
    print(f"\n── Image queries ──")
    for q in lesson.image_queries:
        print(f"  • {q}")
    print(f"\n── Source summaries ──")
    for s in lesson.source_summaries:
        print(f"  • {s}")

    # ── Step 4: Chunk (Featherless splits into speakable pieces) ──────────────
    print(f"\n[4/4] Chunking lesson for ElevenLabs...")
    lesson = featherless.chunk_lesson(lesson)
    print(f"      ✓ {len(lesson.chunks)} chunks ready")
    print(f"\n── Chunks ──")
    for i, chunk in enumerate(lesson.chunks, 1):
        print(f"  [{i}] {chunk}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Final lesson summary ──")
    print(f"  {lesson.summary()}")
    print(f"\n✓ Pipeline test complete. Ready for ElevenLabs.\n")
    return lesson


if __name__ == "__main__":
    # Change this to any topic you want to test
    topic = sys.argv[1] if len(sys.argv) > 1 else "How does photosynthesis work?"
    test_pipeline(topic)