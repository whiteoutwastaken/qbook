"""
stt_service.py
───────────────
Speech-to-text using local OpenAI Whisper.
Runs entirely on your machine — no API cost, no internet required.

HOW IT WORKS:
  - Records audio from microphone until silence is detected
  - Transcribes using Whisper (tiny/base model — fast, accurate enough)
  - Returns plain text of what was said

MODELS (trade-off between speed and accuracy):
  "tiny"   — ~1s transcription, good enough for short questions
  "base"   — ~2s transcription, noticeably better accuracy
  "small"  — ~4s transcription, best for noisy environments

INSTALL:
  pip install openai-whisper sounddevice scipy

USAGE:
  stt = STTService()
  text = stt.listen()          # blocks until user finishes speaking
  print(text)                  # "how does photosynthesis work"
"""

import os
import time
import tempfile
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from typing import Optional, Callable

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL       = "base"     # tiny / base / small
SAMPLE_RATE         = 16000      # Whisper expects 16kHz
SILENCE_THRESHOLD   = 0.01       # RMS below this = silence
SILENCE_DURATION    = 1.5        # Seconds of silence before stopping
MAX_RECORD_SECONDS  = 15         # Hard cap — won't record forever
PRE_ROLL_SECONDS    = 0.3        # Buffer before speech starts


# ── STT Service ───────────────────────────────────────────────────────────────

class STTService:
    """
    Local speech-to-text using OpenAI Whisper.

    Usage:
        stt = STTService()
        text = stt.listen()   # records until silence, returns transcript
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self._model     = None
        self._lock      = threading.Lock()

        # Callback — called when transcription is ready
        self.on_transcript: Callable[[str], None] = lambda t: None

        print(f"[STT] Loading Whisper '{model}' model...")
        self._load_model()

    def _load_model(self):
        try:
            import whisper
            self._model = whisper.load_model(self.model_name)
            print(f"[STT] Whisper '{self.model_name}' ready")
        except ImportError:
            print("[STT] ERROR: openai-whisper not installed.")
            print("[STT] Run: pip install openai-whisper")

    # ── Public API ────────────────────────────────────────────────────────────

    def listen(self, prompt: str = None) -> Optional[str]:
        """
        Records from microphone until silence, then transcribes.
        Blocks until transcription is complete.

        Args:
            prompt: Optional text hint to Whisper (improves accuracy for
                    domain-specific words like "photosynthesis", "mitosis")

        Returns:
            Transcribed text string, or None if nothing was heard.
        """
        if not self._model:
            print("[STT] Model not loaded — cannot listen")
            return None

        print("[STT] Listening... (speak now)")
        audio = self._record_until_silence()

        if audio is None or len(audio) == 0:
            print("[STT] No audio captured")
            return None

        print("[STT] Transcribing...")
        text = self._transcribe(audio, prompt=prompt)

        if text:
            print(f"[STT] Heard: \"{text}\"")
            self.on_transcript(text)

        return text

    def listen_async(self, callback: Callable[[str], None], prompt: str = None) -> None:
        """
        Non-blocking version of listen().
        Calls callback(text) when transcription is ready.
        """
        def _run():
            text = self.listen(prompt=prompt)
            if text:
                callback(text)

        threading.Thread(target=_run, daemon=True).start()

    def is_ready(self) -> bool:
        return self._model is not None

    # ── Recording ─────────────────────────────────────────────────────────────

    def _record_until_silence(self) -> Optional[np.ndarray]:
        """
        Records microphone audio, stopping when silence is detected.
        Returns numpy float32 array at SAMPLE_RATE.
        """
        frames          = []
        silence_start   = None
        recording       = False
        started_at      = time.time()

        chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
            ) as stream:

                while True:
                    chunk, _ = stream.read(chunk_size)
                    rms = float(np.sqrt(np.mean(chunk ** 2)))

                    if rms > SILENCE_THRESHOLD:
                        # Sound detected
                        if not recording:
                            print("[STT] Speech detected — recording")
                            recording = True
                        silence_start = None
                        frames.append(chunk.copy())

                    elif recording:
                        # Was recording, now silence
                        frames.append(chunk.copy())
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            print("[STT] Silence detected — stopping")
                            break

                    # Hard cap
                    if time.time() - started_at > MAX_RECORD_SECONDS:
                        print("[STT] Max duration reached — stopping")
                        break

                    # Timeout if no speech at all after 8s
                    if not recording and time.time() - started_at > 8.0:
                        print("[STT] No speech detected — timing out")
                        return None

        except Exception as e:
            print(f"[STT] Recording error: {e}")
            return None

        if not frames:
            return None

        audio = np.concatenate(frames, axis=0).flatten()
        return audio

    # ── Transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray, prompt: str = None) -> Optional[str]:
        """
        Transcribes audio using Whisper.
        Saves to a temp WAV file — Whisper needs a file path.
        """
        try:
            # Write to temp WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            audio_int16 = (audio * 32767).astype(np.int16)
            write_wav(tmp_path, SAMPLE_RATE, audio_int16)

            # Transcribe
            options = {"language": "en", "task": "transcribe"}
            if prompt:
                options["initial_prompt"] = prompt

            with self._lock:
                result = self._model.transcribe(tmp_path, **options)

            text = result.get("text", "").strip()
            return text if text else None

        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python services/stt_service.py
    Install: pip install openai-whisper sounddevice scipy
    """
    stt = STTService(model="base")

    if not stt.is_ready():
        print("Whisper failed to load.")
        exit(1)

    print("\n" + "=" * 55)
    print("  STT TEST — ask a question out loud")
    print("  e.g. 'How does photosynthesis work?'")
    print("=" * 55 + "\n")

    text = stt.listen()

    if text:
        print(f"\n✓ Transcribed: \"{text}\"")
    else:
        print("\n✗ Nothing heard.")