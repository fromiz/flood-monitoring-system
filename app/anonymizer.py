from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import settings
from .vehicle_flood_pipeline import scheduled_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_model: YOLO | None = None
_model_error: str | None = None
_model_path: Path | None = None
_model_lock = threading.Lock()
_haar_lock = threading.Lock()
_face_cascade = None
_plate_cascade = None


def _resolve_path(raw: str) -> Path:
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_yolo() -> YOLO | None:
    global _model, _model_error, _model_path
    if not settings.anonymizer_enabled:
        _model_error = 'ANONYMIZER_ENABLED=false'
        return None
    if _model is not None:
        return _model

    raw = settings.anonymizer_model_path.strip()
    if not raw:
        _model_error = 'ANONYMIZER_MODEL_PATH가 비어 있어 OpenCV 보조 탐지기를 사용합니다.'
        return None

    resolved = _resolve_path(raw)
    _model_path = resolved
    if not resolved.is_file():
        _model_error = f'비식별화 YOLO 모델이 없어 OpenCV 보조 탐지기를 사용합니다: {resolved}'
        return None

    with _model_lock:
        if _model is not None:
            return _model
        try:
            model = YOLO(str(resolved))
            if torch.cuda.is_available():
                model.to('cuda')
            _model = model
            _model_error = None
            return _model
        except Exception as exc:
            _model_error = f'비식별화 모델 로드 실패: {exc}'
            return None


def _load_haar() -> tuple[Any, Any]:
    global _face_cascade, _plate_cascade
    with _haar_lock:
        if _face_cascade is None:
            _face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        if _plate_cascade is None:
            _plate_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
    return _face_cascade, _plate_cascade


def _safe_device_name() -> tuple[str, str | None]:
    """Return a serializable device name without allowing CUDA checks to fail."""
    try:
        return (
            "cuda" if bool(torch.cuda.is_available()) else "cpu",
            None,
        )
    except Exception as exc:
        return "cpu", f"CUDA 상태 확인 실패: {type(exc).__name__}: {exc}"


def _safe_resolved_model_path() -> tuple[str, str | None]:
    """Resolve the configured path without letting a path error break the API."""
    try:
        raw = (
            settings.anonymizer_model_path
            or "weights/anonymizer_best.pt"
        )
        return str(_model_path or _resolve_path(raw)), None
    except Exception as exc:
        return (
            str(settings.anonymizer_model_path or ""),
            f"모델 경로 확인 실패: {type(exc).__name__}: {exc}",
        )


def _haar_status() -> tuple[bool, str | None]:
    """Check both OpenCV cascade files and instances safely."""
    try:
        face, plate = _load_haar()

        face_ready = bool(
            face is not None
            and hasattr(face, "empty")
            and not face.empty()
        )
        plate_ready = bool(
            plate is not None
            and hasattr(plate, "empty")
            and not plate.empty()
        )

        if face_ready and plate_ready:
            return True, None

        missing = []
        if not face_ready:
            missing.append("얼굴 Haar")
        if not plate_ready:
            missing.append("번호판 Haar")
        return False, "·".join(missing) + " 탐지기를 열지 못했습니다."
    except Exception as exc:
        return (
            False,
            f"OpenCV Haar 준비 실패: {type(exc).__name__}: {exc}",
        )


def status() -> dict[str, Any]:
    """
    Return privacy status as plain JSON-compatible values.

    This function must never raise. A status/diagnostic endpoint should not
    become HTTP 500 merely because an optional YOLO model, CUDA runtime or
    OpenCV cascade has a problem.
    """
    enabled = bool(settings.anonymizer_enabled)
    device, device_error = _safe_device_name()
    resolved_path, path_error = _safe_resolved_model_path()
    fallback_ready, haar_error = _haar_status()

    model = None
    model_error = None

    if enabled:
        try:
            model = _load_yolo()
        except Exception as exc:
            model_error = (
                f"비식별화 YOLO 상태 확인 실패: "
                f"{type(exc).__name__}: {exc}"
            )

    model_loaded = model is not None

    if not enabled:
        backend = "disabled"
    elif model_loaded:
        backend = "yolo"
    elif fallback_ready:
        backend = "opencv-haar"
    else:
        backend = "disabled"

    errors = [
        text
        for text in (
            model_error,
            _model_error,
            haar_error,
            device_error,
            path_error,
        )
        if text
    ]

    # Do not describe the missing optional YOLO model as a fatal error when
    # the OpenCV fallback is ready.
    operational = bool(
        enabled
        and (
            model_loaded
            or fallback_ready
        )
    )

    return {
        "ok": True,
        "operational": operational,
        "enabled": enabled,
        "backend": backend,
        "model_loaded": model_loaded,
        "configured_path": str(
            settings.anonymizer_model_path or ""
        ),
        "resolved_path": str(resolved_path),
        "fallback_ready": bool(fallback_ready),
        "face_cascade_ready": bool(fallback_ready),
        "plate_cascade_ready": bool(fallback_ready),
        "device": str(device),
        "warning": (
            "전용 YOLO 비식별화 모델이 없어 "
            "OpenCV Haar 보조 탐지기를 사용합니다."
            if (
                enabled
                and not model_loaded
                and fallback_ready
            )
            else None
        ),
        "error": " | ".join(errors) if errors else None,
    }


def safe_status() -> dict[str, Any]:
    """
    Last-resort wrapper used by the FastAPI route.

    Even an unexpected programming/runtime error is converted to a readable
    HTTP 200 diagnostic response so the dashboard can show the real state.
    """
    try:
        return status()
    except BaseException as exc:
        # BaseException is intentional here: this is a tiny diagnostic
        # endpoint and must stay available even for unusual runtime failures.
        device, _ = _safe_device_name()
        fallback_ready, haar_error = _haar_status()

        return {
            "ok": False,
            "operational": bool(
                settings.anonymizer_enabled
                and fallback_ready
            ),
            "enabled": bool(settings.anonymizer_enabled),
            "backend": (
                "opencv-haar"
                if fallback_ready
                else "disabled"
            ),
            "model_loaded": False,
            "configured_path": str(
                settings.anonymizer_model_path or ""
            ),
            "resolved_path": "",
            "fallback_ready": bool(fallback_ready),
            "face_cascade_ready": bool(fallback_ready),
            "plate_cascade_ready": bool(fallback_ready),
            "device": str(device),
            "warning": (
                "상태 조회 일부가 실패했지만 OpenCV Haar "
                "보조 탐지기는 준비되어 있습니다."
                if fallback_ready
                else None
            ),
            "error": (
                f"상태 조회 예외: {type(exc).__name__}: {exc}"
                + (
                    f" | {haar_error}"
                    if haar_error
                    else ""
                )
            ),
        }


def _expand_bbox(bbox: list[float], scale: float, shape) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width, height = (x2 - x1) * scale, (y2 - y1) * scale
    frame_h, frame_w = shape[:2]
    return [
        max(0.0, cx - width / 2.0),
        max(0.0, cy - height / 2.0),
        min(float(frame_w), cx + width / 2.0),
        min(float(frame_h), cy + height / 2.0),
    ]


def _normalize_privacy_label(label: str) -> str:
    """Normalize model class names so only privacy classes are accepted."""
    return (
        str(label)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _privacy_kind(label: str) -> str | None:
    """
    Map known face/license-plate class names to canonical privacy labels.

    Everything else (car, vehicle, person, road, etc.) is deliberately
    rejected so a generic detector can never mosaic an entire vehicle/road.
    """
    key = _normalize_privacy_label(label)

    if key in {
        "face",
        "faces",
        "humanface",
    }:
        return "face"

    if key in {
        "plate",
        "licenseplate",
        "licenceplate",
        "numberplate",
        "vehicleplate",
        "registrationplate",
    }:
        return "license_plate"

    return None


def _valid_privacy_box(
    bbox: list[float],
    kind: str,
    frame_shape,
) -> bool:
    """Reject impossible/oversized boxes before mosaic expansion."""
    if len(bbox) != 4:
        return False

    frame_h, frame_w = frame_shape[:2]
    frame_area = float(max(1, frame_w * frame_h))

    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    if width < 4.0 or height < 4.0:
        return False

    area_ratio = (width * height) / frame_area
    aspect_ratio = width / max(height, 1.0)

    if kind == "face":
        # Faces should be roughly square. This also blocks large road/building
        # false positives from the Haar fallback.
        if not 0.45 <= aspect_ratio <= 2.0:
            return False
        face_area_max = min(
            float(settings.anonymizer_face_max_area_ratio),
            float(settings.anonymizer_face_hard_max_area_ratio),
            0.010,
        )
        if area_ratio > face_area_max:
            return False
        if width / max(1.0, frame_w) > 0.18 or height / max(1.0, frame_h) > 0.25:
            return False
        return True

    if kind == "license_plate":
        # Plates are horizontally elongated and should occupy only a small
        # fraction of a CCTV frame. Vehicle-sized boxes are rejected here.
        min_aspect = max(
            float(settings.anonymizer_plate_min_aspect),
            float(settings.anonymizer_plate_hard_min_aspect),
        )
        max_aspect = min(
            float(settings.anonymizer_plate_max_aspect),
            float(settings.anonymizer_plate_hard_max_aspect),
        )
        if not min_aspect <= aspect_ratio <= max_aspect:
            return False
        plate_area_max = min(
            float(settings.anonymizer_plate_max_area_ratio),
            float(settings.anonymizer_plate_hard_max_area_ratio),
            0.0045,
        )
        if area_ratio > plate_area_max:
            return False
        if width / max(1.0, frame_w) > 0.22 or height / max(1.0, frame_h) > 0.09:
            return False
        return True

    return False


def detect(frame) -> dict[str, Any]:
    """
    Detect face/license-plate regions only.

    Privacy YOLO is preferred; OpenCV Haar is a fallback. All detections pass
    through a strict class whitelist and box-shape/size validation so cars,
    roads and other large regions cannot be mosaicked accidentally.
    """
    if not settings.anonymizer_enabled:
        return {
            "detections": [],
            "inference_ms": 0,
            "backend": "disabled",
        }

    started = time.perf_counter()
    model = _load_yolo()
    detections: list[dict[str, Any]] = []

    if model is not None:
        kwargs: dict[str, Any] = {
            "conf": float(settings.anonymizer_confidence),
            "imgsz": int(settings.anonymizer_imgsz),
            "device": 0 if torch.cuda.is_available() else "cpu",
            "verbose": False,
        }
        result = scheduled_predict(
            model,
            frame,
            kind="privacy",
            live_priority=True,
            **kwargs,
        )[0]

        boxes = result.boxes if result.boxes is not None else []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if isinstance(model.names, dict):
                raw_label = str(model.names.get(cls_id, cls_id))
            else:
                raw_label = str(model.names[cls_id])

            kind = _privacy_kind(raw_label)
            if kind is None:
                # Critical safety filter: never mosaic car/vehicle/person/road.
                continue

            bbox = [float(v) for v in box.xyxy[0].tolist()]
            if not _valid_privacy_box(bbox, kind, frame.shape):
                continue

            detections.append({
                "bbox": _expand_bbox(
                    bbox,
                    float(settings.anonymizer_box_scale),
                    frame.shape,
                ),
                "label": kind,
                "conf": round(conf, 3),
            })

        backend = "yolo"
    else:
        face_cascade, plate_cascade = _load_haar()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        if face_cascade is not None and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.13,
                minNeighbors=9,
                minSize=(20, 20),
            )
            for x, y, w, h in faces:
                bbox = [
                    float(x),
                    float(y),
                    float(x + w),
                    float(y + h),
                ]
                if not _valid_privacy_box(bbox, "face", frame.shape):
                    continue
                detections.append({
                    "bbox": _expand_bbox(
                        bbox,
                        float(settings.anonymizer_box_scale),
                        frame.shape,
                    ),
                    "label": "face",
                    "conf": 1.0,
                })

        if plate_cascade is not None and not plate_cascade.empty():
            plates = plate_cascade.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=10,
                minSize=(34, 11),
            )
            for x, y, w, h in plates:
                bbox = [
                    float(x),
                    float(y),
                    float(x + w),
                    float(y + h),
                ]
                if not _valid_privacy_box(
                    bbox,
                    "license_plate",
                    frame.shape,
                ):
                    continue
                detections.append({
                    "bbox": _expand_bbox(
                        bbox,
                        float(settings.anonymizer_box_scale),
                        frame.shape,
                    ),
                    "label": "license_plate",
                    "conf": 1.0,
                })

        backend = "opencv-haar"

    return {
        "detections": detections,
        "inference_ms": round(
            (time.perf_counter() - started) * 1000
        ),
        "backend": backend,
    }

def pixelate_region(frame, bbox: list[float], blocks: int = 12) -> None:
    """Pixelate one region. Coarse blocks make reconstruction more difficult."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    divisor = max(3, int(blocks))
    small_w = max(1, (x2 - x1) // divisor)
    small_h = max(1, (y2 - y1) // divisor)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(
        small,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_NEAREST,
    )


def apply(frame, detections: list[dict[str, Any]]):
    """Mosaic only validated face/license-plate detections."""
    for detection in detections:
        kind = _privacy_kind(str(detection.get("label") or ""))
        if kind is None:
            continue

        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            continue

        # Re-check the expanded box with a looser geometric guard here. The
        # strict check already ran before expansion; this second check prevents
        # externally supplied detections from mosaicking most of the frame.
        x1, y1, x2, y2 = [float(v) for v in bbox]
        h, w = frame.shape[:2]
        area_ratio = (
            max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ) / float(max(1, w * h))
        if kind == "face":
            if area_ratio > float(settings.anonymizer_face_hard_max_area_ratio) * 1.35:
                continue
        elif kind == "license_plate":
            if area_ratio > float(settings.anonymizer_plate_hard_max_area_ratio) * 1.35:
                continue
        else:
            continue

        pixelate_region(
            frame,
            [x1, y1, x2, y2],
            settings.anonymizer_blocks,
        )
    return frame
