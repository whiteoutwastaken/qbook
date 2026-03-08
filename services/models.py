"""
models.py
──────────
Shared data models for the Whisper app.
Imported by both opennote_service.py and featherless_service.py.
Keeping it here breaks the circular import between the two services.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LessonContent:
    """
    The full output of a lesson request. Passed through the entire pipeline.

    OpenNote populates:    topic, text, from_journal, video_url
    Featherless populates: image_queries, source_summaries, chunks
    ElevenLabs consumes:   chunks (one at a time)
    UI consumes:           image_queries, source_summaries, chunks
    """
    topic: str
    text: str
    image_queries: list[str] = field(default_factory=list)
    source_summaries: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    video_url: Optional[str] = None
    from_journal: bool = False

    def has_visuals(self) -> bool:
        return bool(self.image_queries or self.source_summaries)

    def is_ready(self) -> bool:
        return bool(self.text and self.chunks)

    def summary(self) -> str:
        parts = [f"Topic: '{self.topic}'", f"Text: {len(self.text)} chars"]
        if self.chunks:           parts.append(f"Chunks: {len(self.chunks)}")
        if self.image_queries:    parts.append(f"Images: {len(self.image_queries)}")
        if self.source_summaries: parts.append(f"Summaries: {len(self.source_summaries)}")
        if self.video_url:        parts.append("Video: yes")
        if self.from_journal:     parts.append("Source: journal")
        return " | ".join(parts)