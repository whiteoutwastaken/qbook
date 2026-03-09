"""
lesson_manager.py
──────────────────
The core state machine for the Whisper app.
Wires together OpenNote, Featherless, TTS, and NoteSaver
into a single live lesson session.

STATES:
  IDLE        → waiting for user to ask a question
  LOADING     → generating + chunking lesson
  SPEAKING    → TTS speaking current chunk
  WAITING     → chunk done, waiting for facial signal
  SIMPLIFYING → confusion detected, re-explaining chunk
  COMPLETE    → all chunks done, saving notes
  ERROR       → something went wrong

FACIAL SIGNALS (injected via signal()):
  "nod"        → move to next chunk
  "eureka"     → understood with realization, advance
  "confused"   → re-explain current chunk
  "focused"    → stay in WAITING
  "distracted" → nudge user
"""

import threading
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, TYPE_CHECKING

from models import LessonContent

if TYPE_CHECKING:
    from opennote_service    import OpenNoteService
    from featherless_service import FeatherlessService
    from note_saver          import NoteSaver


# ── States ────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE        = auto()
    LOADING     = auto()
    SPEAKING    = auto()
    WAITING     = auto()
    SIMPLIFYING = auto()
    COMPLETE    = auto()
    ERROR       = auto()


# ── Session Data ──────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    topic: str
    lesson: Optional[LessonContent] = None
    current_chunk_index: int = 0

    confusion_counts: dict       = field(default_factory=dict)
    well_received:    list[int]  = field(default_factory=list)
    confused_chunks:  list[int]  = field(default_factory=list)

    started_at: float          = field(default_factory=time.time)
    ended_at:   Optional[float] = None

    def mark_confused(self, index: int):
        self.confusion_counts[index] = self.confusion_counts.get(index, 0) + 1
        if index not in self.confused_chunks:
            self.confused_chunks.append(index)

    def mark_well_received(self, index: int):
        if index not in self.well_received and index not in self.confused_chunks:
            self.well_received.append(index)

    def confusion_count(self, index: int) -> int:
        return self.confusion_counts.get(index, 0)

    def duration_seconds(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at

    def summary(self) -> str:
        return (
            f"Topic: '{self.topic}' | "
            f"Chunks: {len(self.lesson.chunks) if self.lesson else 0} | "
            f"Well received: {len(self.well_received)} | "
            f"Confused on: {len(self.confused_chunks)} | "
            f"Duration: {self.duration_seconds():.0f}s"
        )


# ── Lesson Manager ────────────────────────────────────────────────────────────

class LessonManager:

    AGGRESSIVE_THRESHOLD = 2
    SIGNAL_TIMEOUT       = 15.0

    def __init__(
        self,
        opennote:    "OpenNoteService",
        featherless: "FeatherlessService",
        tts,
        note_saver:  "NoteSaver" = None,
    ):
        self.opennote    = opennote
        self.featherless = featherless
        self.tts         = tts
        self.note_saver  = note_saver

        self._state        = State.IDLE
        self._lock         = threading.Lock()
        self._signal_event = threading.Event()
        self._last_signal  = None
        self._session      : Optional[SessionData] = None
        self._thread       : Optional[threading.Thread] = None

        # Callbacks
        self.on_state_change : Callable[[State], None]         = lambda s: None
        self.on_chunk_start  : Callable[[int, str], None]      = lambda i, t: None
        self.on_chunk_done   : Callable[[int], None]           = lambda i: None
        self.on_visuals      : Callable[[list, list], None]    = lambda imgs, sums: None
        self.on_lesson_ready : Callable[[LessonContent], None] = lambda l: None
        self.on_complete     : Callable[[SessionData], None]   = lambda s: None
        self.on_error        : Callable[[str], None]           = lambda e: None
        self.on_waiting      : Callable[[int, int], None]      = lambda i, total: None

    # ── Public API ────────────────────────────────────────────────────────────

    def start_lesson(self, topic: str, journal_id: str = None) -> None:
        if self._state not in (State.IDLE, State.COMPLETE, State.ERROR):
            print(f"[LessonManager] Cannot start — current state: {self._state.name}")
            return
        self._session = SessionData(topic=topic)
        self._thread  = threading.Thread(
            target=self._run_lesson,
            args=(topic, journal_id),
            daemon=True,
        )
        self._thread.start()

    def signal(self, signal: str) -> None:
        with self._lock:
            self._last_signal = signal
        self._signal_event.set()
        print(f"[LessonManager] Signal received: '{signal}'")

    def stop(self) -> None:
        self._set_state(State.COMPLETE)
        self._signal_event.set()

    @property
    def state(self) -> State:
        return self._state

    @property
    def session(self) -> Optional[SessionData]:
        return self._session

    def is_active(self) -> bool:
        return self._state not in (State.IDLE, State.COMPLETE, State.ERROR)

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def _run_lesson(self, topic: str, journal_id: str = None) -> None:
        lesson = self.opennote.generate_lesson(topic, journal_id=journal_id)
        print("[DEBUG] generate_lesson done")
        lesson = self.featherless.enrich_lesson(lesson)
        print("[DEBUG] enrich_lesson done")
        lesson = self.featherless.chunk_lesson(lesson)
        print("[DEBUG] chunk_lesson done")
        try:
            # Phase 1: Load
            self._set_state(State.LOADING)
            print(f"[LessonManager] Loading lesson: '{topic}'")

            lesson = self.opennote.generate_lesson(topic, journal_id=journal_id)
            lesson = self.featherless.enrich_lesson(lesson)
            lesson = self.featherless.chunk_lesson(lesson)

            self._session.lesson = lesson
            self.on_lesson_ready(lesson)

            if lesson.has_visuals():
                self.on_visuals(lesson.image_queries, lesson.source_summaries)

            if not lesson.chunks:
                raise ValueError("Lesson has no chunks after processing.")

            print(f"[LessonManager] Lesson ready — {len(lesson.chunks)} chunks")

            # Phase 2: Deliver chunks
            total = len(lesson.chunks)
            i = 0

            while i < total:
                chunk = lesson.chunks[i]

                self._set_state(State.SPEAKING)
                self.on_chunk_start(i, chunk)
                print(f"[LessonManager] Chunk {i+1}/{total}")
                self.tts.speak(chunk, blocking=True)
                self.on_chunk_done(i)

                self._set_state(State.WAITING)
                self.on_waiting(i, total)
                sig = self._wait_for_signal()

                if sig in ("nod", "eureka"):
                    if sig == "eureka":
                        print(f"[LessonManager] Eureka!")
                        self.tts.speak("Great, looks like that clicked!", blocking=True)
                    else:
                        print(f"[LessonManager] Nod — advancing")
                    self._session.mark_well_received(i)
                    i += 1

                elif sig == "confused":
                    count      = self._session.confusion_count(i)
                    aggressive = count >= self.AGGRESSIVE_THRESHOLD
                    self._session.mark_confused(i)
                    print(f"[LessonManager] Confused (x{count+1}), aggressive={aggressive}")
                    self._set_state(State.SIMPLIFYING)
                    simpler = self.featherless.simplify_chunk(chunk, topic, aggressive=aggressive)
                    self.on_chunk_start(i, simpler)
                    self.tts.speak(simpler, blocking=True)
                    self.on_chunk_done(i)
                    if count >= self.AGGRESSIVE_THRESHOLD + 1:
                        print(f"[LessonManager] Confused 3x — force advancing")
                        i += 1

                elif sig == "distracted":
                    print(f"[LessonManager] Distracted — nudging")
                    self.tts.speak("Hey, let's refocus. I'll repeat that.", blocking=True)

                elif sig == "timeout":
                    print(f"[LessonManager] Timeout — auto advancing")
                    self._session.mark_well_received(i)
                    i += 1

                elif sig == "stop":
                    print(f"[LessonManager] Stop signal")
                    break

            # Phase 3: Complete
            self._session.ended_at = time.time()
            self._set_state(State.COMPLETE)
            print(f"[LessonManager] Complete — {self._session.summary()}")

            self.tts.speak(
                f"Great work! We've covered all {total} parts of the lesson.",
                blocking=True
            )

            if self.note_saver and self._session.well_received:
                print(f"[LessonManager] Saving {len(self._session.well_received)} chunks to OpenNote...")
                note = self.note_saver.save(lesson, self._session.well_received)
                if note and note.synced_to_opennote:
                    self.tts.speak("Your notes have been saved to OpenNote.", blocking=False)

            self.on_complete(self._session)

        except Exception as e:
            print(f"[LessonManager] ERROR: {e}")
            self._set_state(State.ERROR)
            self.on_error(str(e))

    # ── Signal Handling ───────────────────────────────────────────────────────

    def _wait_for_signal(self) -> str:
        self._signal_event.clear()
        self._last_signal = None
        triggered = self._signal_event.wait(timeout=self.SIGNAL_TIMEOUT)
        if not triggered:
            return "timeout"
        with self._lock:
            sig = self._last_signal
        if self._state == State.COMPLETE:
            return "stop"
        return sig or "timeout"

    # ── State ─────────────────────────────────────────────────────────────────

    def _set_state(self, state: State) -> None:
        with self._lock:
            if self._state != state:
                print(f"[LessonManager] State: {self._state.name} → {state.name}")
                self._state = state
        self.on_state_change(state)


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services"))

    from opennote_service    import OpenNoteService
    from featherless_service import FeatherlessService
    from note_saver          import NoteSaver

    class MockTTS:
        def speak(self, text, blocking=True):
            print(f"  🔊 \"{text[:80]}\"")
            time.sleep(0.2)

    featherless = FeatherlessService()
    opennote    = OpenNoteService(featherless=featherless)
    note_saver  = NoteSaver(featherless)
    manager     = LessonManager(opennote, featherless, MockTTS(), note_saver)

    manager.on_state_change = lambda s: print(f"\n── {s.name} ──")
    manager.on_chunk_start  = lambda i, t: print(f"  [{i+1}] {t[:60]}")
    manager.on_waiting      = lambda i, n: print(f"  Waiting ({i+1}/{n})...")
    manager.on_complete     = lambda s: print(f"\n✓ {s.summary()}")
    manager.on_error        = lambda e: print(f"\n✗ {e}")

    signal_pattern = ["nod", "nod", "confused", "nod", "eureka", "nod", "nod", "nod", "nod", "nod", "nod", "nod", "nod"]
    idx = [0]

    def sim():
        time.sleep(2)
        while manager.is_active():
            if manager.state == State.WAITING:
                time.sleep(1.2)
                sig = signal_pattern[idx[0]] if idx[0] < len(signal_pattern) else "nod"
                idx[0] += 1
                print(f"\n  👤 Signal: '{sig}'")
                manager.signal(sig)
                time.sleep(0.3)
            else:
                time.sleep(0.1)

    threading.Thread(target=sim, daemon=True).start()
    manager.start_lesson("How does photosynthesis work?")
    while manager.is_active():
        time.sleep(0.2)
    print("\n✓ Done.")