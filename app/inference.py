from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LEVEL_NAMES = {
    0: "Level 0 · 정상/젖은 노면",
    1: "Level 1 · 0~12cm",
    2: "Level 2 · 12~35cm",
    3: "Level 3 · 35~60cm",
    4: "Level 4 · 60cm 이상",
}


@dataclass
class Detection:
    level: int
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass
class InferenceResult:
    detections: list[Detection]
    max_level: int
    max_confidence: float


class FloodDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        iou: float = 0.45,
        device: str = "",
        demo_mode: bool = False,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.device = device or None
        self.demo_mode = demo_mode or not model_path
        self.model: Any | None = None
        self._background = cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=36, detectShadows=True)

        if not self.demo_mode:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
            from ultralytics import YOLO
            self.model = YOLO(str(path))

    def predict(self, frame: np.ndarray) -> InferenceResult:
        if self.demo_mode:
            return self._demo_predict(frame)

        # Legacy detector calls share the same single inference owner so
        # enabling legacy workers cannot reintroduce concurrent CUDA predicts.
        from .vehicle_flood_pipeline import scheduled_predict

        result = scheduled_predict(
            self.model,
            frame,
            kind="legacy",
            live_priority=False,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id not in LEVEL_NAMES:
                    continue
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(class_id, confidence, (x1, y1, x2, y2)))

        return self._summarize(detections)

    def _demo_predict(self, frame: np.ndarray) -> InferenceResult:
        """영상 기반 데모 차량 후보 검출. 실제 운영에서는 학습된 YOLO 모델을 사용합니다."""
        h, w = frame.shape[:2]
        mask = self._background.apply(frame)
        mask = cv2.threshold(mask, 210, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < w*h*0.0025 or area > w*h*0.28 or bw < 35 or bh < 20:
                continue
            ratio = bw / max(1, bh)
            if ratio < 0.7 or ratio > 5.5:
                continue
            bottom = (y + bh) / h
            level = 1 if bottom < .58 else 2 if bottom < .72 else 3 if bottom < .87 else 4
            confidence = min(.96, .58 + area/(w*h)*2.8)
            detections.append(Detection(level, confidence, (x, y, x+bw, y+bh)))
        if not detections:
            elapsed = time.monotonic()
            level = min(4, max(1, int((math.sin(elapsed / 6.0) + 1) * 1.5 + 1)))
            detections=[Detection(level, .82, (int(w*.36),int(h*.42),int(w*.63),int(h*.82)))]
        return self._summarize(detections)

    @staticmethod
    def _summarize(detections: list[Detection]) -> InferenceResult:
        if not detections:
            return InferenceResult([], 0, 0.0)
        highest = max(detections, key=lambda item: (item.level, item.confidence))
        return InferenceResult(detections, highest.level, highest.confidence)


def draw_result(frame: np.ndarray, result: InferenceResult, roi_px: np.ndarray | None = None) -> np.ndarray:
    output = frame.copy()

    if roi_px is not None and len(roi_px) >= 3:
        overlay = output.copy()
        cv2.fillPoly(overlay, [roi_px], (255, 255, 255))
        output = cv2.addWeighted(overlay, 0.08, output, 0.92, 0)
        cv2.polylines(output, [roi_px], True, (255, 255, 255), 2)

    for det in result.detections:
        x1, y1, x2, y2 = det.xyxy
        palette = {0:(255,200,88),1:(164,214,66),2:(102,224,255),3:(77,154,255),4:(103,69,255)}
        color = palette.get(det.level, (255,255,255))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        label = f"VEHICLE  L{det.level}  {det.confidence:.2f}"
        cv2.rectangle(output, (x1, max(0, y1 - 30)), (min(output.shape[1], x1 + 245), y1), color, -1)
        cv2.putText(
            output, label, (x1 + 5, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 20, 30), 2, cv2.LINE_AA
        )

    status = LEVEL_NAMES[result.max_level]
    cv2.rectangle(output, (0, 0), (output.shape[1], 52), (15, 20, 30), -1)
    cv2.putText(
        output, status, (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )
    return output
