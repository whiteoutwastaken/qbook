import os
import sys
import time
import argparse
import asyncio
import io

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "services"))
sys.path.append(os.path.join(ROOT, "core"))

from dotenv import load_dotenv
load_dotenv()

import pygame
import edge_tts

from models              import LessonContent
from opennote_service    import OpenNoteService
from featherless_service import FeatherlessService
from note_saver          import NoteSaver
from lesson_manager      import LessonManager, State
from facial_monitor      import FacialMonitor
from stt_service         import STTService


# ── Edge TTS — Free Neural Speech ────────────────────────────────────────────

class EdgeTTS:
    """
    Real TTS using Microsoft Edge's Neural voices.
    Uses 'edge-tts' library for free, high-quality audio.
    """
    def __init__(self, voice="en-GB-SoniaNeural"):
        self.voice = voice
        # Sonia is a great British 'tutor' voice, but 'en-US-AvaMultilingualNeural' is also excellent.
        pygame.mixer.init()

    def speak(self, text: str, blocking: bool = True):
        """
        Runs the async TTS process in a synchronous wrapper.
        """
        if not text.strip():
            return

        print(f"[EdgeTTS] Speaking: \"{text[:50]}...\"")
        
        try:
            # We run the async process here since the main app is sync
            asyncio.run(self._generate_and_play(text, blocking))
        except Exception as e:
            print(f"[EdgeTTS] Error: {e}")
            print(f"\n🔊 {text}\n")

    async def _generate_and_play(self, text, blocking):
        communicate = edge_tts.Communicate(text, self.voice)
        audio_data = io.BytesIO()
        
        # Stream audio chunks into memory
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        audio_data.seek(0)
        
        # Load and play using pygame
        pygame.mixer.music.load(audio_data, "mp3")
        pygame.mixer.music.play()

        if blocking:
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

    def is_speaking(self) -> bool:
        return pygame.mixer.music.get_busy()


# ── Main App ──────────────────────────────────────────────────────────────────

class WhisperApp:

    def __init__(self, typed_mode: bool = False, no_cam: bool = False):
        self.typed_mode = typed_mode
        self.no_cam     = no_cam
        self._running   = False

        print("\n" + "=" * 55)
        print("   WHISPER — AI Tutor  (Edge-TTS Enabled)")
        print("=" * 55)

        print("\n[Whisper] Initialising services...")

        self.featherless = FeatherlessService()
        self.opennote    = OpenNoteService(featherless=self.featherless)
        
        # REPLACED MockTTS with EdgeTTS
        self.tts         = EdgeTTS(voice="en-US-AvaMultilingualNeural")
        self.note_saver  = NoteSaver(self.featherless)
        self.stt         = STTService(model="base")

        self.manager = LessonManager(
            opennote    = self.opennote,
            featherless = self.featherless,
            tts         = self.tts,
            note_saver  = self.note_saver,
        )
        self._wire_callbacks()

        self.monitor = FacialMonitor(
            featherless    = self.featherless,
            lesson_manager = self.manager,
        )

        print("[Whisper] Ready\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _wire_callbacks(self):
        self.manager.on_state_change = self._on_state_change
        self.manager.on_chunk_start  = self._on_chunk_start
        self.manager.on_lesson_ready = self._on_lesson_ready
        self.manager.on_waiting      = self._on_waiting
        self.manager.on_complete     = self._on_complete
        self.manager.on_error        = self._on_error

    def _on_state_change(self, state: State):
        print(f"[Whisper] ── {state.name} ──")

    def _on_chunk_start(self, index: int, text: str):
        # LessonManager handles the .speak() call internally
        print(f"\n   [{index + 1}] {text}")

    def _on_lesson_ready(self, lesson: LessonContent):
        print(f"[Whisper] Lesson ready — {len(lesson.chunks)} chunks")

    def _on_waiting(self, index: int, total: int):
        print(f"[Whisper] Waiting for signal ({index + 1}/{total})...")

    def _on_complete(self, session):
        print(f"\n[Whisper] ✓ Session complete")

    def _on_error(self, error: str):
        print(f"[Whisper] ERROR: {error}")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        self._running = True

        if not self.no_cam:
            self.monitor.start()

        # Intro
        self.tts.speak("Hey, I'm Whisper — your personal tutor. What do you want to learn about?")

        try:
            while self._running:
                topic = self._get_topic()

                if not topic or topic.lower() in ("quit", "exit", "bye", "q"):
                    break

                print(f"\n[Whisper] Starting lesson: \"{topic}\"")
                self.manager.start_lesson(topic)

                while self.manager.state == State.IDLE:
                    time.sleep(0.1)

                while self.manager.is_active():
                    time.sleep(0.2)

                time.sleep(0.5)
                if not self._ask_another():
                    break

        except KeyboardInterrupt:
            print("\n[Whisper] Interrupted")
        finally:
            self._shutdown()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_topic(self) -> str:
        if self.typed_mode:
            return input("[Whisper] What do you want to learn? > ").strip()

        print("[Whisper] Listening for your question...")
        topic = self.stt.listen(prompt="educational question")

        if not topic:
            return input("[Whisper] Didn't catch that. Type your topic > ").strip()

        print(f"[Whisper] Heard: \"{topic}\"")
        return topic

    def _ask_another(self) -> bool:
        self.tts.speak("Would you like to learn about something else?")
        if self.typed_mode:
            return input("\n[Whisper] Another lesson? (y/n) > ").lower() in ("y", "yes", "")
        
        response = self.stt.listen()
        if not response:
            return input("[Whisper] Another lesson? (y/n) > ").lower() in ("y", "yes", "")

        return any(w in response.lower() for w in ("yes", "sure", "yeah", "yep", "next"))

    def _shutdown(self):
        print("\n[Whisper] Shutting down...")
        if not self.no_cam:
            self.monitor.stop()
        self.manager.stop()
        self._running = False
        print("[Whisper] Goodbye.")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper AI Tutor")
    parser.add_argument("--type",   action="store_true", help="Typed input mode")
    parser.add_argument("--no-cam", action="store_true", help="Disable camera")
    args = parser.parse_args()

    app = WhisperApp(typed_mode=args.type, no_cam=args.no_cam)
    app.run()