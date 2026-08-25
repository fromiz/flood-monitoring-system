from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig, settings
from .database import session_scope
from .demo_source import DemoFloodSource
from .inference import FloodDetector, InferenceResult, draw_result
from .models import FloodEvent


def open_source(source: str):
    if source.startswith("demo://"):
        return DemoFloodSource()
    if source.isdigit():
        return cv2.VideoCapture(int(source))

    if "://" in source:
        return cv2.VideoCapture(source)

    resolved = Path(source).expanduser()
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[1] / resolved
    return cv2.VideoCapture(str(resolved))


def normalized_roi_to_pixels(roi, width: int, height: int):
    if not roi:
        return None
    return np.array([(int(x * width), int(y * height)) for x, y in roi], dtype=np.int32)


def apply_roi_mask(frame: np.ndarray, roi_px: np.ndarray | None) -> np.ndarray:
    if roi_px is None:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_px], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


class CameraWorker:
    def __init__(self, config: CameraConfig, detector: FloodDetector) -> None:
        self.config = config
        self.detector = detector
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.current_level = 0
        self.current_confidence = 0.0
        self.fps = 0.0
        self.last_error: str | None = None
        self._level_window = deque(maxlen=max(1, settings.alert_consecutive_frames))
        self._last_event_at = 0.0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{self.config.camera_id}")
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def snapshot(self) -> dict:
        return {
            "camera_id": self.config.camera_id,
            "name": self.config.name,
            "site_name": self.config.site_name,
            "address": self.config.address,
            "source": self.config.source,
            "lat": self.config.lat,
            "lon": self.config.lon,
            "running": self.running,
            "current_level": self.current_level,
            "current_confidence": self.current_confidence,
            "fps": self.fps,
            "last_error": self.last_error,
        }

    def _run(self) -> None:
        capture = None
        frame_count = 0
        tick = time.monotonic()
        try:
            capture = open_source(self.config.source)
            if not capture.isOpened():
                raise RuntimeError(f"영상 소스를 열 수 없습니다: {self.config.source}")

            while self.running:
                ok, frame = capture.read()
                if not ok:
                    if self.config.source.startswith("demo://"):
                        continue
                    time.sleep(1)
                    capture.release()
                    capture = open_source(self.config.source)
                    continue

                frame_count += 1
                if frame_count % max(1, settings.process_every_n_frames) != 0:
                    continue

                h, w = frame.shape[:2]
                roi_px = normalized_roi_to_pixels(self.config.roi, w, h)
                inference_frame = apply_roi_mask(frame, roi_px)
                result = self.detector.predict(inference_frame)
                annotated = draw_result(frame, result, roi_px)

                now = time.monotonic()
                elapsed = now - tick
                if elapsed >= 1:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    tick = now

                self.current_level = result.max_level
                self.current_confidence = result.max_confidence
                self._process_alert(result, annotated)

                success, encoded = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
                )
                if success:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()

                time.sleep(0.01)
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            if capture is not None:
                capture.release()
            self.running = False

    def _process_alert(self, result: InferenceResult, frame: np.ndarray) -> None:
        self._level_window.append(result.max_level)
        if len(self._level_window) < self._level_window.maxlen:
            return
        if min(self._level_window) < settings.alert_min_level:
            return

        now = time.monotonic()
        if now - self._last_event_at < settings.event_cooldown_seconds:
            return
        self._last_event_at = now

        output_dir = Path("recordings") / self.config.camera_id
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f.jpg")
        image_path = output_dir / filename
        cv2.imwrite(str(image_path), frame)

        details = json.dumps(
            {
                "level_window": list(self._level_window),
                "level_name": f"level_{result.max_level}",
                "site_name": self.config.site_name,
                "display_name": self.config.site_name,
                "address": self.config.address,
                "region": self.config.site_name,
                "lat": self.config.lat,
                "lon": self.config.lon,
            },
            ensure_ascii=False,
        )
        with session_scope() as session:
            session.add(
                FloodEvent(
                    camera_id=self.config.camera_id,
                    camera_name=self.config.name,
                    level=result.max_level,
                    confidence=result.max_confidence,
                    image_path=str(image_path),
                    details=details,
                )
            )

    def mjpeg(self):
        while True:
            if not self.running and self.latest_jpeg is None:
                break
            with self.lock:
                jpeg = self.latest_jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)


class CameraManager:
    def __init__(self, detector: FloodDetector) -> None:
        self.workers = {
            cfg.camera_id: CameraWorker(cfg, detector)
            for cfg in settings.cameras()
        }

    def start_all(self) -> None:
        for worker in self.workers.values():
            worker.start()

    def stop_all(self) -> None:
        for worker in self.workers.values():
            worker.stop()

    def get(self, camera_id: str) -> CameraWorker | None:
        return self.workers.get(camera_id)

    def snapshots(self) -> list[dict]:
        return [worker.snapshot() for worker in self.workers.values()]
