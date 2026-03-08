"""
facial_monitor.py
──────────────────
Captures webcam frames, preprocesses with YOLO face detection,
then sends tight face crops to Featherless/Gemma for expression analysis.

PIPELINE:
  Raw webcam frame
        ↓
  YOLO face detection (local, ~5ms, no API cost)
        ↓
  Face found?  → crop + 25% padding + CLAHE contrast enhancement
  No face?     → skip API call entirely
        ↓
  224x224 face crop → Gemma vision
        ↓
  lesson_manager.signal("confused" / "nodding" / "distracted" / "eureka")

INSTALL:
  pip install opencv-python ultralytics

USAGE:
  monitor = FacialMonitor(featherless_service, lesson_manager)
  monitor.start()
  monitor.stop()
"""

import cv2
import base64
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from featherless_service import FeatherlessService
    from lesson_manager import LessonManager, State

# ── Config ────────────────────────────────────────────────────────────────────

ANALYSIS_INTERVAL    = 0.5
API_COOLDOWN         = 1.0
SMOOTHING_COUNT      = 2
JPEG_QUALITY         = 80
DEFAULT_CAMERA_INDEX = 0
IGNORED_STATES       = {"unknown", "focused"}
FACE_PADDING         = 0.25
CROP_SIZE            = 224


# ── YOLO Face Preprocessor ────────────────────────────────────────────────────

class FacePreprocessor:
    """
    Local YOLO face detection + CLAHE enhancement.
    No API calls — runs in ~5ms on CPU.
    Auto-downloads yolov8n-face.pt (~6MB) on first use.
    """

    def __init__(self):
        self._model = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            try:
                self._model = YOLO("yolov8n-face.pt")
                print("[FacePreprocessor] Loaded yolov8n-face model")
            except Exception:
                self._model = YOLO("yolov8n.pt")
                print("[FacePreprocessor] Loaded yolov8n fallback model")
        except ImportError:
            print("[FacePreprocessor] ultralytics not installed — YOLO disabled")
            print("[FacePreprocessor] Run: pip install ultralytics")
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def process(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], dict]:
        """
        Detects face, crops + pads, applies CLAHE, resizes to CROP_SIZE.

        Returns:
            (face_img, meta) — face_img is None if no face detected
        """
        if not self.available:
            return cv2.resize(frame, (CROP_SIZE, CROP_SIZE)), {"face_count": 0, "yolo_available": False}

        try:
            results = self._model(frame, verbose=False, conf=0.4)
            boxes   = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return None, {"face_count": 0, "yolo_available": True}

            # Highest-confidence detection
            best = max(boxes, key=lambda b: float(b.conf[0]))
            conf = float(best.conf[0])
            x1, y1, x2, y2 = map(int, best.xyxy[0])

            # Pad
            h, w   = frame.shape[:2]
            pad_x  = int((x2 - x1) * FACE_PADDING)
            pad_y  = int((y2 - y1) * FACE_PADDING)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return None, {"face_count": 0, "yolo_available": True}

            # CLAHE on L channel — sharpens micro-expressions
            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            enhanced = cv2.merge([self._clahe.apply(l), a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            final = cv2.resize(enhanced, (CROP_SIZE, CROP_SIZE))

            return final, {
                "face_count": len(boxes),
                "yolo_conf":  conf,
                "bbox":       (x1, y1, x2, y2),
                "yolo_available": True,
            }

        except Exception as e:
            print(f"[FacePreprocessor] Error: {e} — using full frame")
            return cv2.resize(frame, (CROP_SIZE, CROP_SIZE)), {"face_count": 0, "error": str(e)}


# ── Facial Monitor ────────────────────────────────────────────────────────────

class FacialMonitor:
    """
    Webcam → YOLO crop → Gemma expression detection → lesson_manager signal.

    Usage:
        monitor = FacialMonitor(featherless, lesson_manager)
        monitor.start()
        monitor.stop()
    """

    def __init__(
        self,
        featherless:      "FeatherlessService",
        lesson_manager:   "LessonManager" = None,
        camera_index:     int   = DEFAULT_CAMERA_INDEX,
        analysis_interval: float = ANALYSIS_INTERVAL,
    ):
        self.featherless       = featherless
        self.lesson_manager    = lesson_manager
        self.camera_index      = camera_index
        self.analysis_interval = analysis_interval

        self._running       = False
        self._thread        : Optional[threading.Thread] = None
        self._cap           : Optional[cv2.VideoCapture] = None
        self._lock          = threading.Lock()
        self._last_api_call = 0.0
        self._recent_states : list[str] = []
        self._last_meta     : dict = {}

        self.preprocessor   = FacePreprocessor()
        self.executor       = ThreadPoolExecutor(max_workers=2)

        # Callbacks
        self.on_frame          : Callable[[np.ndarray], None] = lambda f: None
        self.on_state_detected : Callable[[str, float], None] = lambda s, c: None
        self.on_no_face        : Callable[[], None]           = lambda: None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if self._running:
            print("[FacialMonitor] Already running.")
            return True

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[FacialMonitor] ERROR: Could not open camera {self.camera_index}.")
            print(f"[FacialMonitor] Try a different camera_index (0, 1, 2...)")
            return False

        self._cap           = cap
        self._running       = True
        self._last_api_call = 0.0
        self._thread        = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        print(f"[FacialMonitor] Started — camera {self.camera_index}, "
              f"YOLO={'on' if self.preprocessor.available else 'off'}, "
              f"interval={self.analysis_interval}s")
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()
        print("[FacialMonitor] Stopped.")

    def is_running(self) -> bool:
        return self._running

    def capture_frame(self) -> Optional[np.ndarray]:
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    # ── Monitor Loop ──────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        print("[FacialMonitor] Monitor loop started.")

        while self._running:
            loop_start = time.time()

            if self._should_analyze():
                frame = self._grab_frame()

                if frame is not None:
                    self.on_frame(frame)

                    # YOLO preprocess
                    face_img, meta = self.preprocessor.process(frame)
                    self._last_meta = meta

                    if face_img is None:
                        print("[FacialMonitor] No face detected — skipping API call")
                        self.on_no_face()
                    else:
                        if meta.get("yolo_available") and meta.get("face_count", 0) > 0:
                            print(f"[FacialMonitor] Face crop ready "
                                  f"(YOLO {meta.get('yolo_conf', 0):.0%}) — "
                                  f"sending {CROP_SIZE}px to Gemma")

                        b64 = self._encode_frame(face_img)
                        if b64:
                            now = time.time()
                            if now - self._last_api_call >= API_COOLDOWN:
                                self._last_api_call = now
                                self._analyze_and_signal(b64)
                else:
                    print("[FacialMonitor] Frame capture failed — skipping")

            elapsed = time.time() - loop_start
            time.sleep(max(0, self.analysis_interval - elapsed))

        print("[FacialMonitor] Monitor loop ended.")

    def _should_analyze(self) -> bool:
        if not self.lesson_manager:
            return True
        try:
            from lesson_manager import State
            return self.lesson_manager.state == State.WAITING
        except Exception:
            return True

    def _grab_frame(self) -> Optional[np.ndarray]:
        if not self._cap or not self._cap.isOpened():
            return None
        # Flush buffer for freshest frame
        for _ in range(3):
            self._cap.grab()
        ret, frame = self._cap.read()
        return frame if ret else None

    def _encode_frame(self, frame: np.ndarray) -> Optional[str]:
        try:
            success, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not success:
                return None
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print(f"[FacialMonitor] Encode error: {e}")
            return None

    def _analyze_and_signal(self, face_b64: str) -> None:
        try:
            result = self.featherless.analyze_face(frame_base64=face_b64)

            if result.confidence < 0.60:
                print(f"[FacialMonitor] Low confidence ({result.confidence:.0%}) — skipping")
                return

            print(f"[FacialMonitor] Detected: {result.state} "
                  f"({result.confidence:.0%}) — {result.note}")
            self.on_state_detected(result.state, result.confidence)

            with self._lock:
                self._recent_states.append(result.state)
                if len(self._recent_states) > SMOOTHING_COUNT:
                    self._recent_states.pop(0)

                if len(self._recent_states) == SMOOTHING_COUNT:
                    if len(set(self._recent_states)) == 1:
                        confirmed = self._recent_states[0]
                        self._recent_states.clear()
                        if confirmed not in IGNORED_STATES:
                            self._signal(confirmed)
                        else:
                            print(f"[FacialMonitor] '{confirmed}' — holding, no signal")

        except Exception as e:
            print(f"[FacialMonitor] Analysis error: {e}")

    def _signal(self, state: str) -> None:
        print(f"[FacialMonitor] ✓ Signalling lesson_manager: '{state}'")
        if self.lesson_manager:
            self.lesson_manager.signal(state)


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone test. Prints detections to terminal.
    No display window — avoids Windows MSMF camera conflict.

    Install: pip install opencv-python ultralytics
    Usage:   python core\\facial_monitor.py
    Requires FEATHERLESS_API_KEY in .env
    """
    import os, sys, time
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "services"))

    from featherless_service import FeatherlessService

    featherless = FeatherlessService()
    monitor     = FacialMonitor(featherless, lesson_manager=None)

    monitor.on_state_detected = lambda s, c: print(f"  → Confirmed: {s} ({c:.0%})")
    monitor.on_no_face        = lambda: print("  → No face in frame")

    print("=" * 55)
    print("  FACIAL MONITOR TEST  (YOLO + Gemma)")
    print("  Look at your camera. Press Ctrl+C to stop.")
    print("=" * 55)

    if not monitor.start():
        print("Could not open camera.")
        sys.exit(1)

    try:
        print("\nMonitor running — press Ctrl+C to stop.\n")
        while monitor.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        monitor.stop()