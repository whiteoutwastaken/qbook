"""
note_saver.py
──────────────
Saves well-received lesson chunks directly to OpenNote journals.

HOW IT WORKS:
  During a lesson, lesson_manager tracks which chunks were "well received":
    - User nodded        → positive signal
    - No confusion       → understood
    - Focused throughout → engaged

  After the lesson, this service:
    1. Collects the well-received chunks
    2. Uses Featherless to format them into clean markdown notes
    3. Saves to /data/notes/ locally (always, as backup)
    4. Creates a real OpenNote journal via import_from_markdown()
       so the student can access their notes on opennote.com

Usage:
    saver = NoteSaver(featherless_service, opennote_client)
    note = saver.save(lesson, well_received_indices=[0, 2, 4, 7])
    print(note.preview())

Install:
    pip install opennote python-dotenv
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from opennote import OpennoteClient
from opennote.types.block_types import HeadingBlock, ParagraphBlock
from opennote.util.edit_operations import create_block

if TYPE_CHECKING:
    from featherless_service import FeatherlessService

from models import LessonContent

# ── Config ────────────────────────────────────────────────────────────────────

NOTES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "notes")


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class SavedNote:
    """
    A lesson note saved locally and optionally synced to OpenNote.
    """
    id: str                          # e.g. "photosynthesis_20260308_143022"
    title: str                       # e.g. "📚 Photosynthesis — Lesson Notes"
    topic: str                       # Original lesson topic
    content_markdown: str            # Full note as markdown
    chunks_saved: list[str]          # The well-received chunks
    created_at: str                  # ISO timestamp
    opennote_journal_id: Optional[str] = None   # Set after successful sync
    synced_to_opennote: bool = False

    def preview(self) -> str:
        lines = [
            f"Title:    {self.title}",
            f"Chunks:   {len(self.chunks_saved)} saved",
            f"Created:  {self.created_at}",
            f"OpenNote: {'✓ ' + self.opennote_journal_id if self.synced_to_opennote else '✗ not synced'}",
            f"\nPreview:\n{self.content_markdown[:400]}...",
        ]
        return "\n".join(lines)


# ── Note Saver ────────────────────────────────────────────────────────────────

class NoteSaver:
    """
    Saves well-received lesson chunks as structured notes — locally and to OpenNote.

    Usage:
        saver = NoteSaver(featherless, opennote_api_key="your_key")
        note = saver.save(lesson, well_received_indices=[0, 2, 4])
    """

    def __init__(self, featherless: "FeatherlessService", opennote_api_key: str = None):
        self.featherless = featherless
        self.opennote_api_key = opennote_api_key or os.getenv("OPENNOTE_API_KEY")
        self.client = OpennoteClient(api_key=self.opennote_api_key)
        os.makedirs(NOTES_DIR, exist_ok=True)

    # ── Primary Method ────────────────────────────────────────────────────────

    def save(self, lesson: LessonContent, well_received_indices: list[int]) -> SavedNote | None:
        """
        Saves well-received chunks from a lesson as a structured note.

        Steps:
          1. Pull the well-received chunks by index
          2. Format into clean markdown via Featherless
          3. Save locally to /data/notes/ as JSON + .md
          4. Push to OpenNote via import_from_markdown()

        Args:
            lesson:                 The completed LessonContent
            well_received_indices:  0-based indices of chunks the student understood well

        Returns:
            SavedNote if anything was saved, None if nothing qualified.
        """
        if not well_received_indices or not lesson.chunks:
            print("[NoteSaver] Nothing to save — no well-received chunks.")
            return None

        well_received = [
            lesson.chunks[i]
            for i in well_received_indices
            if i < len(lesson.chunks)
        ]

        if not well_received:
            print("[NoteSaver] No valid chunks to save.")
            return None

        print(f"[NoteSaver] Saving {len(well_received)}/{len(lesson.chunks)} chunks for '{lesson.topic}'")

        # Format into markdown
        content_markdown = self._format_note(lesson.topic, well_received)

        # Build note object
        now = datetime.now()
        note_id = f"{self._slugify(lesson.topic)}_{now.strftime('%Y%m%d_%H%M%S')}"
        title = f"📚 {lesson.topic.title()} — Lesson Notes"

        note = SavedNote(
            id=note_id,
            title=title,
            topic=lesson.topic,
            content_markdown=content_markdown,
            chunks_saved=well_received,
            created_at=now.isoformat(),
        )

        # Always save locally first (guaranteed backup)
        self._save_local(note)

        # Push to OpenNote
        self._sync_to_opennote(note)

        return note

    # ── OpenNote Sync ─────────────────────────────────────────────────────────

    def _sync_to_opennote(self, note: SavedNote) -> None:
        """
        Creates a real OpenNote journal from the note markdown.
        Uses import_from_markdown() — cleanest way to push formatted content.
        """
        try:
            print(f"[NoteSaver] Pushing to OpenNote: '{note.title}'...")

            result = self.client.journals.editor.import_from_markdown(
                markdown=note.content_markdown,
                title=note.title,
            )

            note.opennote_journal_id = result.journal_id
            note.synced_to_opennote = True

            # Update local JSON with the journal ID
            self._save_local(note)

            print(f"[NoteSaver] ✓ Synced to OpenNote — journal_id: {result.journal_id}")

        except Exception as e:
            print(f"[NoteSaver] OpenNote sync failed: {e} — note saved locally only.")

    def append_to_journal(self, journal_id: str, new_chunks: list[str], section_title: str = "More Notes") -> bool:
        """
        Appends new content to an existing OpenNote journal.
        Useful for adding notes from a follow-up lesson on the same topic.

        Args:
            journal_id:    The OpenNote journal ID to append to
            new_chunks:    New chunks to add
            section_title: Heading for the new section

        Returns:
            True if successful.
        """
        try:
            # Get current journal structure
            journal_info = self.client.journals.editor.model_info(journal_id=journal_id)
            last_block = journal_info.model.content[-1]

            from opennote.types.block_types import Position

            operations = [
                # Add a new section heading
                create_block(
                    position=Position.AFTER,
                    reference_id=last_block.attrs.id,
                    block=HeadingBlock(level=2, content=section_title),
                ),
            ]

            # Add each chunk as a paragraph after the heading
            # We re-fetch model_info after each to keep reference_id current
            self.client.journals.editor.edit(
                journal_id=journal_id,
                operations=operations,
                sync_realtime_state=True,
            )

            # Now append each chunk as its own paragraph
            for chunk in new_chunks:
                info = self.client.journals.editor.model_info(journal_id=journal_id)
                last = info.model.content[-1]
                self.client.journals.editor.edit(
                    journal_id=journal_id,
                    operations=[
                        create_block(
                            position=Position.AFTER,
                            reference_id=last.attrs.id,
                            block=ParagraphBlock(content=chunk),
                        )
                    ],
                    sync_realtime_state=True,
                )

            print(f"[NoteSaver] ✓ Appended {len(new_chunks)} chunks to journal {journal_id}")
            return True

        except Exception as e:
            print(f"[NoteSaver] Append failed: {e}")
            return False

    # ── Local Storage ─────────────────────────────────────────────────────────

    def _save_local(self, note: SavedNote) -> None:
        """Saves note as JSON and .md locally."""
        # JSON (machine-readable, includes OpenNote journal ID once synced)
        json_path = os.path.join(NOTES_DIR, f"{note.id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(note.__dict__, f, indent=2, ensure_ascii=False)

        # Markdown (human-readable)
        md_path = os.path.join(NOTES_DIR, f"{note.id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"---\n")
            f.write(f"title: {note.title}\n")
            f.write(f"topic: {note.topic}\n")
            f.write(f"created: {note.created_at}\n")
            f.write(f"opennote_journal_id: {note.opennote_journal_id or 'not synced'}\n")
            f.write(f"---\n\n")
            f.write(note.content_markdown)

        print(f"[NoteSaver] Saved locally: {note.id}")

    def load_all(self) -> list[SavedNote]:
        """Loads all saved notes from /data/notes/."""
        notes = []
        for filename in sorted(os.listdir(NOTES_DIR)):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(NOTES_DIR, filename), "r", encoding="utf-8") as f:
                        notes.append(SavedNote(**json.load(f)))
                except Exception as e:
                    print(f"[NoteSaver] Could not load {filename}: {e}")
        return notes

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_note(self, topic: str, chunks: list[str]) -> str:
        """Uses Featherless to format chunks into clean markdown. Falls back to simple format."""
        try:
            raw = "\n\n".join(f"- {c}" for c in chunks)
            response = self.featherless._chat(
                system=(
                    "You are a note-taking assistant. Format the given lesson excerpts "
                    "into clean, well-structured markdown study notes. Use headers and "
                    "bold key terms. Keep it concise and easy to review. "
                    "Do not add information not present in the original excerpts."
                ),
                user=(
                    f"Topic: {topic}\n\n"
                    f"Key concepts the student understood well:\n\n{raw}\n\n"
                    f"Format as clean markdown study notes."
                ),
                max_tokens=600,
            )
            return response.strip()
        except Exception as e:
            print(f"[NoteSaver] Formatting failed: {e} — using simple format")
            return self._simple_format(topic, chunks)

    def _simple_format(self, topic: str, chunks: list[str]) -> str:
        """Fallback: simple markdown without Featherless."""
        date = datetime.now().strftime("%B %d, %Y")
        lines = [f"# {topic.title()} — Lesson Notes", f"*{date}*\n", "## Key Concepts\n"]
        for chunk in chunks:
            lines.append(f"{chunk}\n")
        return "\n".join(lines)

    def _slugify(self, text: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in text.lower())[:50].strip("_")


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python note_saver.py
    Requires OPENNOTE_API_KEY and FEATHERLESS_API_KEY in .env
    """
    import sys
    sys.path.append(os.path.dirname(__file__))

    from featherless_service import FeatherlessService
    from models import LessonContent

    featherless = FeatherlessService()
    saver = NoteSaver(featherless)

    lesson = LessonContent(
        topic="How does photosynthesis work?",
        text="...",
        chunks=[
            "Photosynthesis is like a recipe plants use to make food using sunlight.",
            "Chloroplasts contain chlorophyll which captures sunlight energy.",
            "Chlorophyll gives plants their green color and drives photosynthesis.",
            "The plant converts CO2 and water into glucose using captured energy.",
            "Oxygen is released as a byproduct — which is what we breathe.",
            "Plants store excess glucose as starch for use at night or in winter.",
            "Photosynthesis is the foundation of most food chains on Earth.",
        ]
    )

    # Simulate: student understood chunks 0, 2, 4, 6
    note = saver.save(lesson, well_received_indices=[0, 2, 4, 6])

    if note:
        print(f"\n── Note saved ──")
        print(note.preview())

        print(f"\n── All saved notes ──")
        all_notes = saver.load_all()
        for n in all_notes:
            print(f"  - {n.title} (synced: {n.synced_to_opennote})")

    print("\n✓ NoteSaver test complete.")