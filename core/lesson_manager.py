"""
lesson_manager.py
──────────────────
The core state machine for the Whisper app.
Wires together OpenNote, Featherless, ElevenLabs, and NoteSaver
into a single live lesson session.

STATES:
  IDLE        → waiting for user to ask a question
  LOADING     → OpenNote + Featherless generating + chunking lesson
  SPEAKING    → ElevenLabs speaking current chunk
  WAITING     → chunk done, waiting for facial signal (nod / confused / distracted)
  SIMPLIFYING → confusion detected, Featherless re-explaining chunk
  COMPLETE    → all chunks done, saving notes
  ERROR       → something went wrong

FACIAL SIGNALS (from facial_monitor.py, injected via signal()):
  "nod"        → move to next chunk
  "confused"   → re-explain current chunk (simplify)
  "focused"    → stay in WAITING, keep watching
  "distracted" → pause lesson, gently nudge user

USAGE:
  manager = LessonManager(opennote, featherless, elevenlabs, note_saver)
  manager.start_lesson("How does photosynthesis work?")
  # facial_monitor calls manager.signal("nod") or manager.signal("confused")
  # from a separate thread
"""

import threading
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, TYPE_CHECKING

from models import LessonContent

if TYPE_CHECKING:
    from opennote_service import OpenNoteService
    from featherless_service import FeatherlessService
    from elevenlabs_service import ElevenLabsService
    from note_saver import NoteSaver


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
    """
    Tracks everything that happens during a lesson session.
    Used by NoteSaver at the end to determine what to save.
    """
    topic: str
    lesson: Optional[LessonContent] = None
    current_chunk_index: int = 0

    # Per-chunk tracking
    confusion_counts: dict = field(default_factory=dict)   # {chunk_index: int}
    well_received: list[int] = field(default_factory=list) # chunk indices with no confusion + nod
    confused_chunks: list[int] = field(default_factory=list)

    # Timing
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

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
    """
    Drives the full lesson session state machine.

    Responsibilities:
      - Orchestrates OpenNote → Featherless → ElevenLabs pipeline
      - Reacts to facial signals from facial_monitor.py
      - Tracks per-chunk comprehension (nod = understood, confused = re-explain)
      - Saves well-received chunks to OpenNote via NoteSaver at session end

    Thread safety:
      - start_lesson() runs the loading + speaking loop on a background thread
      - signal() can be called safely from facial_monitor's thread at any time
      - State transitions are protected by a threading.Lock

    Usage:
        manager = LessonManager(opennote, featherless, elevenlabs, note_saver)

        # Hook up UI callbacks
        manager.on_state_change = lambda state: print(f"State: {state}")
        manager.on_chunk_start  = lambda i, text: print(f"Speaking: {text}")
        manager.on_visuals      = lambda imgs, summaries: update_ui(imgs, summaries)
        manager.on_complete     = lambda session: show_summary(session)

        # Start lesson (non-blocking)
        manager.start_lesson("How does photosynthesis work?")

        # From facial_monitor thread:
        manager.signal("nod")       # move to next chunk
        manager.signal("confused")  # re-explain
    """

    # How many times confused before using aggressive simplification
    AGGRESSIVE_THRESHOLD = 2

    # Max seconds to wait for a facial signal before auto-advancing
    SIGNAL_TIMEOUT = 15.0

    def __init__(
        self,
        opennote:   "OpenNoteService",
        featherless:"FeatherlessService",
        elevenlabs: "ElevenLabsService",
        note_saver: "NoteSaver" = None,
    ):
        self.opennote    = opennote
        self.featherless = featherless
        self.elevenlabs  = elevenlabs
        self.note_saver  = note_saver

        self._state      = State.IDLE
        self._lock       = threading.Lock()
        self._signal_event = threading.Event()
        self._last_signal  = None
        self._session    : Optional[SessionData] = None
        self._thread     : Optional[threading.Thread] = None

        # ── Callbacks (wire these up in your UI) ──────────────────────────────
        self.on_state_change : Callable[[State], None]             = lambda s: None
        self.on_chunk_start  : Callable[[int, str], None]          = lambda i, t: None
        self.on_chunk_done   : Callable[[int], None]               = lambda i: None
        self.on_visuals      : Callable[[list, list], None]        = lambda imgs, sums: None
        self.on_lesson_ready : Callable[[LessonContent], None]     = lambda l: None
        self.on_complete     : Callable[[SessionData], None]       = lambda s: None
        self.on_error        : Callable[[str], None]               = lambda e: None
        self.on_waiting      : Callable[[int, int], None]          = lambda i, total: None  # chunk i of total

    # ── Public API ────────────────────────────────────────────────────────────

    def start_lesson(self, topic: str, journal_id: str = None) -> None:
        """
        Starts a lesson session on a background thread.
        Non-blocking — returns immediately.

        Args:
            topic:      The topic the user asked about
            journal_id: Optional specific OpenNote journal to use
        """
        if self._state not in (State.IDLE, State.COMPLETE, State.ERROR):
            print(f"[LessonManager] Cannot start — current state: {self._state.name}")
            return

        self._session = SessionData(topic=topic)
        self._thread = threading.Thread(
            target=self._run_lesson,
            args=(topic, journal_id),
            daemon=True,
        )
        self._thread.start()

    def signal(self, signal: str) -> None:
        """
        Receives a facial signal from facial_monitor.py.
        Call this from any thread — it's thread-safe.

        Args:
            signal: One of "nod", "confused", "focused", "distracted"
        """
        with self._lock:
            self._last_signal = signal
        self._signal_event.set()
        print(f"[LessonManager] Signal received: '{signal}'")

    def stop(self) -> None:
        """Stops the current lesson gracefully."""
        print("[LessonManager] Stopping lesson...")
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
        """Main lesson loop — runs on background thread."""
        try:
            # ── Phase 1: Load lesson ──────────────────────────────────────────
            self._set_state(State.LOADING)
            print(f"[LessonManager] Loading lesson: '{topic}'")

            lesson = self.opennote.generate_lesson(topic, journal_id=journal_id)
            lesson = self.featherless.enrich_lesson(lesson)
            lesson = self.featherless.chunk_lesson(lesson)

            self._session.lesson = lesson
            self.on_lesson_ready(lesson)

            # Send visuals to UI immediately
            if lesson.has_visuals():
                self.on_visuals(lesson.image_queries, lesson.source_summaries)

            if not lesson.chunks:
                raise ValueError("Lesson has no chunks after processing.")

            print(f"[LessonManager] Lesson ready — {len(lesson.chunks)} chunks")

            # ── Phase 2: Speak chunks ─────────────────────────────────────────
            total = len(lesson.chunks)
            i = 0

            while i < total:
                chunk = lesson.chunks[i]

                # Speak the chunk
                self._set_state(State.SPEAKING)
                self.on_chunk_start(i, chunk)
                print(f"[LessonManager] Speaking chunk {i+1}/{total}")
                self.elevenlabs.speak(chunk, blocking=True)
                self.on_chunk_done(i)

                # Wait for facial signal
                self._set_state(State.WAITING)
                self.on_waiting(i, total)
                signal = self._wait_for_signal()

                elif signal == "nod" or signal == "eureka":
                    # Understood — mark well received, advance
                    if signal == "eureka":
                        print(f"[LessonManager] Eureka detected — student had a realization!")
                        self.elevenlabs.speak("Great, looks like that clicked!", blocking=True)
                    else:
                        print(f"[LessonManager] Nod detected — advancing")
                    self._session.mark_well_received(i)
                    i += 1

                elif signal == "confused":
                    # Confused — re-explain
                    count = self._session.confusion_count(i)
                    self._session.mark_confused(i)
                    aggressive = count >= self.AGGRESSIVE_THRESHOLD

                    print(f"[LessonManager] Confused (x{count+1}) — simplifying (aggressive={aggressive})")
                    self._set_state(State.SIMPLIFYING)

                    simpler = self.featherless.simplify_chunk(chunk, topic, aggressive=aggressive)
                    self.on_chunk_start(i, simpler)
                    self.elevenlabs.speak(simpler, blocking=True)
                    self.on_chunk_done(i)

                    # After 3 confusions, force advance so we don't get stuck
                    if count >= self.AGGRESSIVE_THRESHOLD + 1:
                        print(f"[LessonManager] Confused 3x — force advancing")
                        i += 1

                elif signal == "distracted":
                    # Distracted — pause and nudge, re-speak chunk
                    print(f"[LessonManager] Distracted — nudging user")
                    nudge = "Hey, let's refocus. I'll repeat that last part."
                    self.elevenlabs.speak(nudge, blocking=True)
                    # Don't advance — re-speak same chunk next iteration

                elif signal == "timeout":
                    # No signal received — auto advance
                    print(f"[LessonManager] Signal timeout — auto advancing")
                    self._session.mark_well_received(i)
                    i += 1

                elif signal == "stop":
                    print(f"[LessonManager] Stop signal — ending lesson")
                    break

            # ── Phase 3: Complete ─────────────────────────────────────────────
            self._session.ended_at = time.time()
            self._set_state(State.COMPLETE)

            print(f"[LessonManager] Lesson complete — {self._session.summary()}")

            # Speak closing line
            self.elevenlabs.speak(
                f"Great work! We've covered all {total} parts of the lesson.",
                blocking=True
            )

            # Save well-received chunks to OpenNote
            if self.note_saver and self._session.well_received:
                print(f"[LessonManager] Saving {len(self._session.well_received)} well-received chunks...")
                note = self.note_saver.save(lesson, self._session.well_received)
                if note and note.synced_to_opennote:
                    self.elevenlabs.speak(
                        "Your notes have been saved to OpenNote.",
                        blocking=False
                    )

            self.on_complete(self._session)

        except Exception as e:
            print(f"[LessonManager] ERROR: {e}")
            self._set_state(State.ERROR)
            self.on_error(str(e))

    # ── Signal Handling ───────────────────────────────────────────────────────

    def _wait_for_signal(self) -> str:
        """
        Blocks until a facial signal arrives or timeout is reached.
        Returns the signal string or "timeout".
        """
        self._signal_event.clear()
        self._last_signal = None

        triggered = self._signal_event.wait(timeout=self.SIGNAL_TIMEOUT)

        if not triggered:
            return "timeout"

        with self._lock:
            signal = self._last_signal

        # Stop signal can arrive at any time
        if self._state == State.COMPLETE:
            return "stop"

        return signal or "timeout"

    # ── State Management ──────────────────────────────────────────────────────

    def _set_state(self, state: State) -> None:
        """Thread-safe state transition."""
        with self._lock:
            if self._state != state:
                print(f"[LessonManager] State: {self._state.name} → {state.name}")
                self._state = state
        self.on_state_change(state)


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Runs a full simulated lesson session WITHOUT camera or audio.
    Facial signals are simulated automatically so you can test
    the full state machine without any hardware.

    Usage: python lesson_manager.py
    Requires OPENNOTE_API_KEY and FEATHERLESS_API_KEY in .env
    Set SIMULATE_ELEVENLABS=true in .env to skip actual audio.
    """
    import os, sys
    sys.path.append(os.path.dirname(__file__))

    from opennote_service import OpenNoteService
    from featherless_service import FeatherlessService
    from elevenlabs_service import ElevenLabsService
    from note_saver import NoteSaver

    # ── Mock ElevenLabs so we can test without audio ──────────────────────────
    class MockElevenLabs:
        def speak(self, text, blocking=True):
            print(f"  🔊 [AUDIO] \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
            time.sleep(0.3)  # Simulate speaking time

    # ── Init services ─────────────────────────────────────────────────────────
    print("Initializing services...")
    featherless = FeatherlessService()
    opennote    = OpenNoteService(featherless=featherless)
    elevenlabs  = MockElevenLabs()
    note_saver  = NoteSaver(featherless)

    # ── Init manager ──────────────────────────────────────────────────────────
    manager = LessonManager(opennote, featherless, elevenlabs, note_saver)

    # ── Wire up callbacks ─────────────────────────────────────────────────────
    manager.on_state_change = lambda s: print(f"\n── State: {s.name} ──")
    manager.on_chunk_start  = lambda i, t: print(f"  Chunk {i+1}: {t[:60]}...")
    manager.on_chunk_done   = lambda i: print(f"  Chunk {i+1} done")
    manager.on_visuals      = lambda imgs, sums: print(f"  Visuals: {imgs}")
    manager.on_waiting      = lambda i, total: print(f"  Waiting for signal ({i+1}/{total})...")
    manager.on_complete     = lambda s: print(f"\n✓ Session complete\n  {s.summary()}")
    manager.on_error        = lambda e: print(f"\n✗ Error: {e}")

    # ── Simulate facial signals after each chunk ──────────────────────────────
    # Pattern: nod, nod, confused, nod, confused, confused, nod...
    signal_pattern = ["nod", "nod", "confused", "nod", "confused", "confused", "nod", "nod", "nod", "nod", "nod", "nod", "nod"]
    signal_index = [0]

    def simulate_signals():
        """Sends simulated facial signals with a short delay after each chunk."""
        time.sleep(2)  # Wait for lesson to load
        while manager.is_active():
            if manager.state == State.WAITING:
                time.sleep(1.5)  # Simulate user thinking time
                if signal_index[0] < len(signal_pattern):
                    sig = signal_pattern[signal_index[0]]
                    signal_index[0] += 1
                else:
                    sig = "nod"
                print(f"\n  👤 [FACE] Sending signal: '{sig}'")
                manager.signal(sig)
                time.sleep(0.5)
            else:
                time.sleep(0.1)

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  LESSON MANAGER TEST (simulated signals)")
    print("=" * 55)

    signal_thread = threading.Thread(target=simulate_signals, daemon=True)
    signal_thread.start()

    manager.start_lesson("How does photosynthesis work?")

    # Wait for lesson to complete
    while manager.is_active():
        time.sleep(0.2)

    print("\n✓ Lesson manager test complete.")