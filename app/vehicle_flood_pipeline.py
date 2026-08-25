from __future__ import annotations

import logging
import threading
import time
import traceback
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import settings
from .stage_consensus import choose_stage_by_count_then_confidence
from .stage_policy import qualifies_stage_confidence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_MODEL_LOCK = threading.RLock()
_LIVE_INFERENCE_PRIORITY = threading.Event()
_MODELS: dict[str, YOLO | None] = {
    "vehicle": None,
    "tire_level": None,
    "car_flood_cls": None,
}
_MODEL_PATHS: dict[str, Path | None] = {key: None for key in _MODELS}
_MODEL_ERRORS: dict[str, str | None] = {key: None for key in _MODELS}
_CUDA_FALLBACK_WARNED = False
logger = logging.getLogger(__name__)


class _InferenceJob:
    """One synchronous request serviced by the single inference owner thread."""

    __slots__ = (
        "kind", "model", "source", "kwargs", "live_priority", "submitted_at",
        "done", "result", "error", "traceback_text", "cancelled",
    )

    def __init__(
        self,
        kind: str,
        model: YOLO,
        source: Any,
        kwargs: dict[str, Any],
        *,
        live_priority: bool,
    ) -> None:
        self.kind = str(kind)
        self.model = model
        self.source = source
        self.kwargs = dict(kwargs)
        self.live_priority = bool(live_priority)
        self.submitted_at = time.monotonic()
        self.done = threading.Event()
        self.result = None
        self.error: BaseException | None = None
        self.traceback_text: str | None = None
        self.cancelled = False


class _CentralInferenceScheduler:
    """Single owner for every Ultralytics predict() call.

    V8.5.x allowed the vehicle model and the tire/body stage models to execute
    from different camera threads under different locks.  On CUDA those locks
    did not serialize the GPU itself, so several model.predict() calls could
    overlap and latency exploded from ~100 ms to 40-80 seconds as more CCTV
    windows opened.

    This scheduler is the only thread allowed to call model.predict().  Live
    vehicle geometry has first priority, stage work gets a bounded turn after a
    short vehicle burst, and background work runs only when live queues are
    empty.  No busy-wait polling is used.
    """

    def __init__(self) -> None:
        self.max_vehicle_burst = max(
            1, int(getattr(settings, "ai_scheduler_vehicle_burst", 4))
        )
        self.micro_batch_wait_seconds = max(
            0.0,
            min(
                0.050,
                float(getattr(settings, "ai_scheduler_micro_batch_wait_ms", 10.0))
                / 1000.0,
            ),
        )
        self.max_vehicle_batch = max(
            1,
            min(8, int(getattr(settings, "ai_scheduler_max_vehicle_batch", 4))),
        )
        self._condition = threading.Condition()
        self._vehicle_live: deque[_InferenceJob] = deque()
        self._stage_live: deque[_InferenceJob] = deque()
        self._background: deque[_InferenceJob] = deque()
        self._thread: threading.Thread | None = None
        self._started = False
        self._vehicle_burst = 0

        self._running_kind: str | None = None
        self._running_since = 0.0
        self._last_kind: str | None = None
        self._last_duration_ms: float | None = None
        self._last_queue_ms: float | None = None
        self._last_batch_size = 0
        self._max_duration_ms = 0.0
        self._completed = 0
        self._failed = 0
        self._last_error: str | None = None

    def _ensure_started(self) -> None:
        with self._condition:
            if self._started and self._thread is not None and self._thread.is_alive():
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="central-gpu-inference",
                daemon=True,
            )
            self._thread.start()

    def notify_priority_change(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _lane(self, job: _InferenceJob) -> deque[_InferenceJob]:
        if not job.live_priority:
            return self._background
        if job.kind == "vehicle":
            return self._vehicle_live
        return self._stage_live

    def submit(
        self,
        kind: str,
        model: YOLO,
        source: Any,
        kwargs: dict[str, Any],
        *,
        live_priority: bool,
        timeout_seconds: float = 120.0,
    ):
        self._ensure_started()
        job = _InferenceJob(
            kind,
            model,
            source,
            kwargs,
            live_priority=live_priority,
        )
        with self._condition:
            self._lane(job).append(job)
            self._condition.notify_all()

        if not job.done.wait(max(5.0, float(timeout_seconds))):
            with self._condition:
                job.cancelled = True
                self._condition.notify_all()
            status = self.status()
            raise TimeoutError(
                "중앙 AI 스케줄러 응답이 지연되고 있습니다 "
                f"(kind={kind}, running={status['running_kind']}, "
                f"vehicle_q={status['vehicle_queue']}, "
                f"stage_q={status['stage_queue']}, "
                f"background_q={status['background_queue']})."
            )
        if job.error is not None:
            detail = job.traceback_text or repr(job.error)
            raise RuntimeError(
                f"{kind} 중앙 추론 실패: {job.error}\\n{detail}"
            )
        return job.result

    @staticmethod
    def _batch_key(job: _InferenceJob) -> tuple[Any, ...] | None:
        # Only full-frame ndarray vehicle requests are micro-batched. Tile rescue
        # and stage calls already pass a list and should remain one logical job.
        if job.kind != "vehicle" or not isinstance(job.source, np.ndarray):
            return None
        try:
            kwargs_key = tuple(sorted(job.kwargs.items()))
        except Exception:
            return None
        return (id(job.model), kwargs_key)

    def _take_vehicle_batch_locked(self) -> list[_InferenceJob]:
        first = self._vehicle_live.popleft()
        batch = [first]
        key = self._batch_key(first)
        if key is None:
            return batch

        deadline = time.monotonic() + self.micro_batch_wait_seconds
        while len(batch) < self.max_vehicle_batch:
            if self._vehicle_live:
                candidate = self._vehicle_live[0]
                if candidate.cancelled:
                    self._vehicle_live.popleft()
                    continue
                if self._batch_key(candidate) != key:
                    break
                batch.append(self._vehicle_live.popleft())
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Condition.wait releases the lock so other camera threads can enqueue.
            self._condition.wait(timeout=remaining)
        return batch

    def _drop_cancelled_locked(self) -> None:
        for lane in (self._vehicle_live, self._stage_live, self._background):
            while lane and lane[0].cancelled:
                lane.popleft()

    def _next_jobs(self) -> list[_InferenceJob]:
        with self._condition:
            while True:
                self._drop_cancelled_locked()
                # Vehicle geometry wins most turns, but after a short burst the
                # newest tire/body stage job gets one chance. This prevents both
                # vehicle starvation and stage livelock.
                if self._vehicle_live and (
                    self._vehicle_burst < self.max_vehicle_burst
                    or not self._stage_live
                ):
                    batch = self._take_vehicle_batch_locked()
                    self._vehicle_burst += len(batch)
                    return batch

                if self._stage_live:
                    self._vehicle_burst = 0
                    return [self._stage_live.popleft()]

                if self._vehicle_live:
                    batch = self._take_vehicle_batch_locked()
                    self._vehicle_burst += len(batch)
                    return batch

                # Background scans must never jump in front of an active live
                # session. They simply sleep on the condition; no polling loop.
                if self._background and not _LIVE_INFERENCE_PRIORITY.is_set():
                    self._vehicle_burst = 0
                    return [self._background.popleft()]

                self._condition.wait(timeout=0.25)

    def _execute(self, jobs: list[_InferenceJob]) -> None:
        first = jobs[0]
        started = time.perf_counter()
        queue_ms = max(
            0.0,
            (time.monotonic() - min(job.submitted_at for job in jobs)) * 1000.0,
        )
        with self._condition:
            self._running_kind = first.kind
            self._running_since = time.monotonic()

        def _source_shape(value: Any) -> str:
            try:
                if isinstance(value, np.ndarray):
                    return "x".join(str(int(v)) for v in value.shape[:2])
                if isinstance(value, (list, tuple)):
                    shapes = [_source_shape(item) for item in value[:6]]
                    return "[" + ",".join(shapes) + ("]" if len(value) <= 6 else ",...]")
            except Exception:
                pass
            return type(value).__name__

        source_shapes = [_source_shape(job.source) for job in jobs]
        imgsz_value = first.kwargs.get("imgsz")
        rect_value = first.kwargs.get("rect", "default")

        try:
            with torch.inference_mode():
                if len(jobs) > 1:
                    # Ultralytics accepts a list of ndarray frames and returns one
                    # result object per source. This is a tiny 10 ms micro-batch,
                    # not a latency-heavy offline batch.
                    outputs = first.model.predict(
                        source=[job.source for job in jobs],
                        **first.kwargs,
                    )
                    if len(outputs) != len(jobs):
                        raise RuntimeError(
                            f"micro-batch result count mismatch: "
                            f"{len(outputs)} != {len(jobs)}"
                        )
                    for job, output in zip(jobs, outputs):
                        job.result = [output]
                else:
                    first.result = first.model.predict(
                        source=first.source,
                        **first.kwargs,
                    )
        except BaseException as exc:
            tb = traceback.format_exc()
            logger.error(
                "Central inference failed kind=%s batch=%s: %s\\n%s",
                first.kind,
                len(jobs),
                exc,
                tb,
            )
            for job in jobs:
                job.error = exc
                job.traceback_text = tb
            with self._condition:
                self._failed += len(jobs)
                self._last_error = f"{first.kind}: {exc}"
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            with self._condition:
                self._last_kind = first.kind
                self._last_duration_ms = round(duration_ms, 1)
                self._last_queue_ms = round(queue_ms, 1)
                self._last_batch_size = len(jobs)
                self._max_duration_ms = max(self._max_duration_ms, duration_ms)
                if all(job.error is None for job in jobs):
                    self._completed += len(jobs)
                    self._last_error = None
                self._running_kind = None
                self._running_since = 0.0
            if duration_ms >= 2000.0 or queue_ms >= 2000.0:
                logger.warning(
                    "AI SCHED62 slow kind=%s batch=%s run_ms=%.0f queue_ms=%.0f "
                    "imgsz=%s rect=%s shapes=%s cudnn_benchmark=%s qv=%s qs=%s qb=%s",
                    first.kind,
                    len(jobs),
                    duration_ms,
                    queue_ms,
                    imgsz_value,
                    rect_value,
                    source_shapes,
                    bool(torch.backends.cudnn.benchmark),
                    len(self._vehicle_live),
                    len(self._stage_live),
                    len(self._background),
                )
            for job in jobs:
                job.done.set()

    def _run(self) -> None:
        logger.warning("Central GPU scheduler started: single predict owner")
        while True:
            jobs = self._next_jobs()
            self._execute(jobs)

    def status(self) -> dict[str, Any]:
        with self._condition:
            running_age = (
                max(0.0, time.monotonic() - self._running_since)
                if self._running_since
                else None
            )
            return {
                "architecture": "single-owner-priority-scheduler",
                "thread_alive": bool(self._thread and self._thread.is_alive()),
                "running_kind": self._running_kind,
                "running_age_seconds": (
                    round(running_age, 3) if running_age is not None else None
                ),
                "vehicle_queue": len(self._vehicle_live),
                "stage_queue": len(self._stage_live),
                "background_queue": len(self._background),
                "last_kind": self._last_kind,
                "last_duration_ms": self._last_duration_ms,
                "last_queue_ms": self._last_queue_ms,
                "last_batch_size": self._last_batch_size,
                "max_duration_ms": round(self._max_duration_ms, 1),
                "completed": self._completed,
                "failed": self._failed,
                "last_error": self._last_error,
                "live_priority_active": _LIVE_INFERENCE_PRIORITY.is_set(),
            }


_INFERENCE_SCHEDULER = _CentralInferenceScheduler()

# RTX/CUDA runtime tuning.  We intentionally do NOT pass Ultralytics' deprecated
# ``half=`` predict argument; model inference remains CUDA accelerated without
# flooding the console with deprecation warnings.
if torch.cuda.is_available():
    try:
        # V8.6.2: this service intentionally mixes several inference shapes:
        # full-frame vehicle detection, rescue tiles, and variable vehicle crops
        # for tire/body classification. cuDNN benchmark mode can benchmark every
        # newly-seen convolution shape before caching it. Disable the autotuner
        # so latency stays predictable; keep high matmul precision enabled.
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"
        logger.warning(
            "CUDA62 tuning cudnn_benchmark=%s cudnn=%s torch=%s cuda=%s gpu=%s",
            bool(torch.backends.cudnn.benchmark),
            torch.backends.cudnn.version(),
            getattr(torch, "__version__", "unknown"),
            getattr(torch.version, "cuda", None),
            gpu_name,
        )
    except Exception:
        logger.exception("CUDA62 runtime tuning failed")



def _selected_device() -> str:
    requested = str(getattr(settings, "ai_device", "auto") or "auto").strip().lower()
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            # Claude's device-fallback build correctly identified one important
            # failure mode: a CUDA preference must not silently disable all three
            # models when the CUDA runtime is unavailable. Keep the warning, but
            # fail open to CPU so best.pt boxes are still produced.
            global _CUDA_FALLBACK_WARNED
            if not _CUDA_FALLBACK_WARNED:
                _CUDA_FALLBACK_WARNED = True
                print(
                    "[AI_DEVICE] CUDA를 사용할 수 없어 CPU로 자동 전환합니다. "
                    "GPU를 원하면 CUDA 지원 PyTorch/드라이버를 확인하세요.",
                    flush=True,
                )
            return "cpu"
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ai_uses_cuda() -> bool:
    return _selected_device() == "cuda"



def _resolve_model_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _model_specs() -> dict[str, tuple[str, str]]:
    return {
        "vehicle": (settings.stage2_model_path, "STAGE2_MODEL_PATH"),
        "tire_level": (settings.tire_level_model_path, "TIRE_LEVEL_MODEL_PATH"),
        "car_flood_cls": (settings.car_flood_cls_model_path, "CAR_FLOOD_CLS_MODEL_PATH"),
    }


def _load_one(kind: str) -> YOLO | None:
    current = _MODELS.get(kind)
    if current is not None:
        return current

    raw_path, env_name = _model_specs()[kind]
    if not str(raw_path or "").strip():
        _MODEL_ERRORS[kind] = f"{env_name}가 설정되지 않았습니다."
        return None

    resolved = _resolve_model_path(raw_path)
    _MODEL_PATHS[kind] = resolved
    if not resolved.is_file():
        _MODEL_ERRORS[kind] = f"모델 파일을 찾을 수 없습니다: {resolved}"
        return None

    # Ultralytics .pt checkpoints use the PyTorch zip container. Detect a
    # truncated transfer before YOLO emits a very long internal miniz trace.
    try:
        if not zipfile.is_zipfile(resolved):
            _MODEL_ERRORS[kind] = (
                f"손상된 체크포인트: {resolved.name} "
                "(파일을 원본 모델로 다시 교체하세요.)"
            )
            return None
    except OSError as exc:
        _MODEL_ERRORS[kind] = f"모델 파일 검사 실패: {resolved.name} · {exc}"
        return None

    with _MODEL_LOCK:
        if _MODELS.get(kind) is not None:
            return _MODELS[kind]
        try:
            model = YOLO(str(resolved))
            if _selected_device() == "cuda":
                model.to("cuda")
            else:
                model.to("cpu")
            _MODELS[kind] = model
            _MODEL_ERRORS[kind] = None
            return model
        except Exception as exc:
            _MODEL_ERRORS[kind] = f"모델 로드 실패: {exc}"
            return None


def load_models() -> tuple[YOLO, YOLO | None, YOLO | None] | None:
    vehicle = _load_one("vehicle")
    tire = _load_one("tire_level")
    body = _load_one("car_flood_cls")
    # The vehicle detector is mandatory. Either stage model is sufficient:
    # a broken tire model falls back to body classification for every crop,
    # and a missing body model can still use valid tire detections.
    if vehicle is None or (tire is None and body is None):
        return None
    return vehicle, tire, body


def model_error() -> str:
    errors = [
        f"{kind}: {message}"
        for kind, message in _MODEL_ERRORS.items()
        if message
    ]
    return " | ".join(errors) or "3개 AI 모델을 모두 불러오지 못했습니다."


def _names(model: YOLO | None) -> dict[str, str]:
    if model is None:
        return {}
    try:
        return {str(k): str(v) for k, v in dict(model.names).items()}
    except Exception:
        return {}


def model_status() -> dict[str, Any]:
    bundle = load_models()
    specs = _model_specs()
    details: dict[str, dict[str, Any]] = {}
    for kind, (configured, _env_name) in specs.items():
        resolved = _MODEL_PATHS.get(kind) or _resolve_model_path(configured)
        details[kind] = {
            "loaded": _MODELS.get(kind) is not None,
            "configured_path": configured,
            "resolved_path": str(resolved),
            "exists": resolved.is_file(),
            "task": getattr(_MODELS.get(kind), "task", None),
            "classes": _names(_MODELS.get(kind)),
            "error": _MODEL_ERRORS.get(kind),
        }

    return {
        "loaded": bundle is not None,
        "degraded": bundle is not None and any(
            _MODELS.get(kind) is None for kind in ("tire_level", "car_flood_cls")
        ),
        # Backward-compatible top-level fields used by the dashboard.
        "configured_path": settings.stage2_model_path,
        "resolved_path": details["vehicle"]["resolved_path"],
        "exists": details["vehicle"]["exists"],
        "device": _selected_device(),
        "cuda_available": torch.cuda.is_available(),
        "device_requested": str(getattr(settings, "ai_device", "auto") or "auto"),
        "classes": details["vehicle"]["classes"],
        "pipeline": "vehicle -> tire_level -> car_flood_cls fallback",
        "models": details,
        "scheduler": _INFERENCE_SCHEDULER.status(),
        "warning": model_error() if bundle is not None and any(_MODEL_ERRORS.values()) else None,
        "error": None if bundle is not None else model_error(),
    }



def _vehicle_display_label(model: YOLO, cls_id: int) -> str:
    """Return a generic label when vehicle-type judgement is disabled."""
    if not bool(getattr(settings, "vehicle_type_labels_enabled", True)):
        return "vehicle"
    try:
        return str(model.names[int(cls_id)])
    except Exception:
        return "vehicle"


def _stage_from_class(cls_id: int, label: str) -> int:
    if 0 <= int(cls_id) <= 4:
        return int(cls_id)
    digits = [int(ch) for ch in str(label) if ch.isdigit()]
    return max(0, min(4, digits[0] if digits else 0))


def _device_kwargs() -> dict[str, Any]:
    # ``half=`` is deprecated by the installed Ultralytics build and prints a
    # warning on every prediction.  Device selection alone is sufficient here;
    # the three .pt models stay on CUDA when AI_DEVICE=cuda.
    return {
        "verbose": False,
        "device": 0 if _selected_device() == "cuda" else "cpu",
    }


def set_live_inference_priority(active: bool) -> None:
    """Tell the central scheduler whether interactive CCTV is active.

    V8.5.x used this flag inside busy-wait loops. V8.6.2 only uses it as a
    scheduling hint: live work is queued normally, while background work sleeps
    on a condition until the live session ends.
    """
    if active:
        _LIVE_INFERENCE_PRIORITY.set()
    else:
        _LIVE_INFERENCE_PRIORITY.clear()
    _INFERENCE_SCHEDULER.notify_priority_change()


def inference_scheduler_status() -> dict[str, Any]:
    return _INFERENCE_SCHEDULER.status()


def scheduled_predict(
    model: YOLO,
    source: Any,
    *,
    kind: str,
    live_priority: bool = False,
    **kwargs: Any,
):
    """Route an Ultralytics call through the single inference owner."""
    predict_kwargs = _device_kwargs()
    predict_kwargs.update(kwargs)
    return _INFERENCE_SCHEDULER.submit(
        str(kind),
        model,
        source,
        predict_kwargs,
        live_priority=live_priority,
    )


def _predict(
    model: YOLO,
    source: Any,
    *,
    live_priority: bool = False,
    **kwargs: Any,
):
    """Submit one flood-model call to the single inference owner.

    No camera/stage caller thread invokes Ultralytics directly. This is the
    central V8.6.2 concurrency invariant and removes the separate vehicle/stage
    lock race.
    """
    model_kind = next(
        (kind for kind, loaded in _MODELS.items() if loaded is model),
        "vehicle",
    )
    return scheduled_predict(
        model,
        source,
        kind=model_kind,
        live_priority=live_priority,
        **kwargs,
    )


def detect_vehicle_boxes(
    frame: np.ndarray,
    *,
    vehicle_imgsz: int | None = None,
    live_priority: bool = False,
    allow_rescue: bool = True,
) -> list[dict[str, Any]]:
    """Direct best.pt geometry path for live CCTV.

    V8.5.31 deliberately removes all temporal confirmation and downstream stage
    dependencies from *box visibility*.  If best.pt returns a box, it is drawn.
    To recover tiny/night-time vehicles, a zero/few-box full-frame pass is
    followed by overlapping 2x2 tiles and, only when the scene is dark or the
    normal pass found nothing, a CLAHE-enhanced detector-only pass.

    The tire/body models never decide whether a vehicle rectangle is visible.
    """
    vehicle_model = _load_one("vehicle")
    if vehicle_model is None:
        raise RuntimeError(_MODEL_ERRORS.get("vehicle") or "차량 모델 로드 실패")

    h, w = frame.shape[:2]
    if h < 16 or w < 16:
        return []

    # V8.5.36: do NOT hard-prune low-confidence YOLO outputs inside Ultralytics.
    # The municipal night CCTV footage can produce real vehicles below 3% raw
    # detector confidence. V8.5.33 raised the model-side floor to 3%, which made
    # the detector return zero boxes on feeds that V8.5.32 could see. Pull the
    # raw geometry again and reject noise later with temporal confirmation +
    # class-agnostic de-duplication instead of deleting it before tracking.
    raw_conf = (
        max(0.0005, min(0.01, float(settings.vehicle_detection_confidence)))
        if ai_uses_cuda()
        else max(0.002, min(0.03, float(settings.vehicle_detection_confidence)))
    )
    full_imgsz = max(640, min(1280, int(vehicle_imgsz or settings.vehicle_detection_imgsz)))
    collected: list[tuple[list[float], int, float, str]] = []
    detector_total_started = time.perf_counter()
    full_ms = 0.0
    rescue_ms = 0.0
    clahe_ms = 0.0

    def collect_result(result: Any, *, offset_x: int = 0, offset_y: int = 0, source: str) -> None:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return
        for box in boxes:
            try:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
            except Exception:
                continue
            x1 += offset_x; x2 += offset_x
            y1 += offset_y; y2 += offset_y
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < 6.0 or box_h < 6.0:
                continue
            aspect = box_w / max(1.0, box_h)
            area_ratio = (box_w * box_h) / max(1.0, float(w * h))
            # Extremely flat/tall geometry in the diagnostic build produced
            # long rectangles across road markings and building edges.
            if aspect < 0.28 or aspect > 7.0 or area_ratio > 0.45:
                continue
            collected.append(([x1, y1, x2, y2], cls_id, conf, source))

    full_started = time.perf_counter()
    full_results = _predict(
        vehicle_model,
        frame,
        conf=raw_conf,
        iou=0.55,
        imgsz=full_imgsz,
        max_det=300,
        live_priority=live_priority,
    )
    full_ms = (time.perf_counter() - full_started) * 1000.0
    if full_results:
        collect_result(full_results[0], source="direct_full")

    rescue_target = max(2, int(getattr(settings, "vehicle_detection_rescue_min_count", 3)))
    if allow_rescue and len(collected) < rescue_target:
        # 2x2 overlapping crops enlarge far vehicles in both axes.  The previous
        # vertical-only rescue left upper-road cars too small in 16:9 CCTV.
        overlap = 0.14
        tile_w = max(160, int(w * 0.58))
        tile_h = max(120, int(h * 0.58))
        x_starts = [0, max(0, w - tile_w)]
        y_starts = [0, max(0, h - tile_h)]
        specs: list[tuple[int,int,int,int]] = []
        for y1 in y_starts:
            for x1 in x_starts:
                x1 = max(0, min(w - tile_w, x1))
                y1 = max(0, min(h - tile_h, y1))
                x2 = min(w, x1 + tile_w)
                y2 = min(h, y1 + tile_h)
                spec=(x1,y1,x2,y2)
                if spec not in specs: specs.append(spec)
        tiles=[frame[y1:y2, x1:x2] for x1,y1,x2,y2 in specs]
        if tiles:
            rescue_started = time.perf_counter()
            tile_results = _predict(
                vehicle_model,
                tiles,
                conf=raw_conf,
                iou=0.55,
                imgsz=max(704, min(960, full_imgsz)),
                max_det=180,
                live_priority=live_priority,
            )
            rescue_ms = (time.perf_counter() - rescue_started) * 1000.0
            for spec, result in zip(specs, tile_results):
                x1,y1,_,_=spec
                collect_result(result, offset_x=x1, offset_y=y1, source="direct_tile")

    # Night CCTV domain rescue.  This never changes the displayed frame; it only
    # gives best.pt a contrast-normalised copy when the ordinary path has no/few
    # candidates.  Running it conditionally keeps GPU load bounded.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    if allow_rescue and len(collected) < 1 and mean_luma < 125.0:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2BGR)
        clahe_started = time.perf_counter()
        enhanced_results = _predict(
            vehicle_model,
            enhanced,
            conf=raw_conf,
            iou=0.55,
            imgsz=full_imgsz,
            max_det=300,
            live_priority=live_priority,
        )
        clahe_ms = (time.perf_counter() - clahe_started) * 1000.0
        if enhanced_results:
            collect_result(enhanced_results[0], source="direct_clahe")

    detector_total_ms = (time.perf_counter() - detector_total_started) * 1000.0
    if detector_total_ms >= 2000.0:
        logger.warning(
            "DET62 slow total_ms=%.0f full_ms=%.0f rescue_ms=%.0f clahe_ms=%.0f "
            "imgsz=%s frame=%sx%s raw_candidates=%s cudnn_benchmark=%s",
            detector_total_ms, full_ms, rescue_ms, clahe_ms, full_imgsz, w, h,
            len(collected), bool(torch.backends.cudnn.benchmark),
        )

    # Strong class-agnostic de-duplication across full/tile/CLAHE passes.
    # The same car can have shifted boxes whose IoU is only 0.3-0.5, so IoU
    # alone left two or three rectangles on one vehicle. Also suppress boxes
    # that substantially contain one another or have almost the same centre.
    collected.sort(key=lambda item: item[2], reverse=True)
    kept: list[tuple[list[float], int, float, str]] = []

    def overlap_metrics(a: list[float], b: list[float]) -> tuple[float, float, float]:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        bb = max(1.0, (bx2 - bx1) * (by2 - by1))
        iou_value = inter / max(1.0, aa + bb - inter)
        overlap_smaller = inter / max(1.0, min(aa, bb))
        acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
        bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        diag = max(8.0, (max(ax2-ax1, bx2-bx1) ** 2 + max(ay2-ay1, by2-by1) ** 2) ** 0.5)
        center_ratio = ((acx-bcx) ** 2 + (acy-bcy) ** 2) ** 0.5 / diag
        return iou_value, overlap_smaller, center_ratio

    for item in collected:
        bbox, cls_id, conf, source = item
        duplicate = False
        for prev in kept:
            iou_value, overlap_smaller, center_ratio = overlap_metrics(bbox, prev[0])
            if iou_value >= 0.38 or overlap_smaller >= 0.72:
                duplicate = True
                break
            if center_ratio <= 0.12 and iou_value >= 0.18:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(item)
        if len(kept) >= 40:
            break

    detections: list[dict[str, Any]] = []
    for bbox, cls_id, conf, source in kept:
        clipped = [
            max(0.0, min(float(w - 1), bbox[0])),
            max(0.0, min(float(h - 1), bbox[1])),
            max(1.0, min(float(w), bbox[2])),
            max(1.0, min(float(h), bbox[3])),
        ]
        detections.append({
            "label": "VEHICLE DETECTED",
            "source_label": _vehicle_display_label(vehicle_model, cls_id),
            "class_id": int(cls_id),
            "track_id": None,
            "raw_stage": None,
            "stage": None,
            "stage_valid": False,
            "stage_policy": None,
            "stage_source": "vehicle_only",
            "stage_model_label": "pending",
            "conf": round(float(conf), 4),
            "bbox": [round(float(v), 1) for v in clipped],
            "vehicle_class_id": int(cls_id),
            "vehicle_label": _vehicle_display_label(vehicle_model, cls_id),
            "vehicle_conf": round(float(conf), 4),
            "detector_source": source,
            "tire_detections": [],
            # Visibility below 12% is decided by the temporal confirmer.
            "_provisional": bool(float(conf) < 0.12),
            "_confirmed": bool(float(conf) >= 0.12),
        })
    return detections


def _crop_vehicle(frame: np.ndarray, bbox: list[float], pad_ratio: float | None = None) -> np.ndarray | None:
    height, width = frame.shape[:2]
    if pad_ratio is None:
        pad_ratio = max(0.0, min(0.25, float(settings.vehicle_crop_pad_ratio)))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio
    ix1 = max(0, int(x1 - pad_x))
    iy1 = max(0, int(y1 - pad_y))
    ix2 = min(width, int(x2 + pad_x))
    iy2 = min(height, int(y2 + pad_y))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    crop = frame[iy1:iy2, ix1:ix2]
    return crop.copy() if crop.size else None


def _best_tire_stage(result: Any, model: YOLO) -> tuple[int, float, str, int, list[dict[str, Any]]] | None:
    candidates: list[tuple[int, float, str, int, list[float]]] = []
    boxes = result.boxes if getattr(result, "boxes", None) is not None else []
    for box in boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = str(model.names[cls_id])
        stage = _stage_from_class(cls_id, label)
        bbox = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        candidates.append((stage, confidence, label, cls_id, bbox))

    if not candidates:
        return None

    # V8.5.0: multiple tire detections use strict frequency mode as well.
    # A single false high-stage tire must not promote the whole vehicle.
    counts = {level: 0 for level in range(5)}
    confidence_sums = {level: 0.0 for level in range(5)}
    for item in candidates:
        level = int(item[0])
        counts[level] += 1
        confidence_sums[level] += float(item[1])
    winning_stage, _winning_confidence, _averages = (
        choose_stage_by_count_then_confidence(counts, confidence_sums)
    )
    winning = [item for item in candidates if int(item[0]) == winning_stage]
    best = max(winning, key=lambda item: item[1])
    details = [
        {
            "stage": item[0],
            "conf": round(item[1], 3),
            "label": item[2],
            "class_id": item[3],
            "bbox_in_vehicle_crop": item[4],
        }
        for item in candidates
    ]
    return best[0], best[1], best[2], best[3], details


def _body_stage(result: Any, model: YOLO) -> tuple[int, float, str, int] | None:
    probs = getattr(result, "probs", None)
    if probs is None:
        return None
    try:
        cls_id = int(probs.top1)
        confidence = float(probs.top1conf)
    except Exception:
        return None
    label = str(model.names[cls_id])
    return _stage_from_class(cls_id, label), confidence, label, cls_id


def infer_vehicle_flood(
    frame: np.ndarray,
    *,
    vehicle_imgsz: int | None = None,
    stage_floor: tuple[int, float] | None = None,
    vehicle_candidates: list[dict[str, Any]] | None = None,
    live_priority: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Run the requested hierarchy on one CCTV frame.

    1. Detect vehicles with best.pt.
    2. Crop each detected vehicle and run tire_level.pt.
    3. If no tire is detected in that vehicle crop, run car_flood_cls.pt.

    Returns one flood-stage record per detected vehicle and the representative
    highest-risk vehicle record.
    """
    bundle = load_models()
    if bundle is None:
        raise RuntimeError(model_error())
    vehicle_model, tire_model, body_model = bundle

    # Use the same confidence policy as the live best.pt pass.  The previous
    # 0.03-0.10 clamp flooded background inference with false vehicle crops,
    # which could hold the GPU for tens of seconds and starve live CCTV boxes.
    base_conf = min(
        max(0.005, float(settings.vehicle_detection_confidence)),
        0.008 if ai_uses_cuda() else 0.03,
    )

    # Collect full-frame boxes first. Small/far vehicles are frequently missed
    # after a whole 16:9 CCTV frame is resized for YOLO, so V8.5.2 optionally
    # performs a second best.pt pass on two overlapping half-width tiles when
    # the first pass returns only a few cars. The extra pass is detector-only;
    # tire/body classification is still batched once after de-duplication.
    candidates: list[dict[str, Any]] = []

    def add_candidate(
        bbox: list[float],
        cls_id: int,
        conf: float,
        source: str,
        track_id: int | None = None,
    ) -> None:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 - x1 < 8 or y2 - y1 < 8:
            return
        # Suppress duplicate full/tile boxes using IoU. Prefer the stronger box.
        for existing in candidates:
            ex1, ey1, ex2, ey2 = [float(v) for v in existing["bbox"]]
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = (x2 - x1) * (y2 - y1) + (ex2 - ex1) * (ey2 - ey1) - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= 0.55:
                if conf > float(existing["vehicle_conf"]):
                    existing.update({
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "vehicle_class_id": int(cls_id),
                        "vehicle_label": _vehicle_display_label(vehicle_model, cls_id),
                        "vehicle_conf": round(float(conf), 3),
                        "detector_source": source,
                        "track_id": track_id if track_id is not None else existing.get("track_id"),
                    })
                return
        candidates.append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "vehicle_class_id": int(cls_id),
            "vehicle_label": _vehicle_display_label(vehicle_model, cls_id),
            "vehicle_conf": round(float(conf), 3),
            "detector_source": source,
            "track_id": track_id,
        })

    if vehicle_candidates is not None:
        for vehicle in vehicle_candidates:
            bbox = vehicle.get("bbox") or []
            if len(bbox) != 4:
                continue
            add_candidate(
                [float(value) for value in bbox],
                int(vehicle.get("vehicle_class_id", vehicle.get("class_id", 0)) or 0),
                float(vehicle.get("vehicle_conf", vehicle.get("conf", 0.0)) or 0.0),
                str(vehicle.get("detector_source") or "live_fast"),
                (
                    int(vehicle.get("track_id"))
                    if vehicle.get("track_id") is not None
                    else None
                ),
            )
    else:
        vehicle_results = _predict(
            vehicle_model,
            frame,
            conf=base_conf,
            iou=max(0.1, min(0.95, float(settings.vehicle_detection_iou))),
            imgsz=int(vehicle_imgsz or settings.vehicle_detection_imgsz),
            max_det=220,
            live_priority=live_priority,
        )
        vehicle_result = vehicle_results[0]
        full_boxes = vehicle_result.boxes if getattr(vehicle_result, "boxes", None) is not None else []
        for box in full_boxes:
            add_candidate(
                [float(v) for v in box.xyxy[0].tolist()],
                int(box.cls[0]),
                float(box.conf[0]),
                "full",
            )

    if (
        vehicle_candidates is None
        and bool(settings.vehicle_detection_rescue_enabled)
        and len(candidates) < max(1, int(settings.vehicle_detection_rescue_min_count))
        and frame.shape[1] >= 480
    ):
        height, width = frame.shape[:2]
        overlap = max(0.0, min(0.30, float(settings.vehicle_detection_rescue_overlap)))
        # Three overlapping vertical tiles retain substantially more pixels per
        # distant vehicle than the old two-half pass.  This is especially useful
        # for wide municipal CCTV views while keeping classification batched.
        tile_width = max(240, int(width * 0.46))
        overlap_px = int(tile_width * overlap)
        step = max(1, tile_width - overlap_px)
        starts = [0, min(step, width - tile_width), max(0, width - tile_width)]
        tile_specs = []
        for start in starts:
            spec = (max(0, int(start)), min(width, int(start) + tile_width))
            if spec not in tile_specs:
                tile_specs.append(spec)
        tiles = [frame[:, x1:x2] for x1, x2 in tile_specs]
        tile_results = _predict(
            vehicle_model,
            tiles,
            conf=max(0.006, min(base_conf, float(settings.vehicle_detection_rescue_confidence))),
            iou=max(0.1, min(0.95, float(settings.vehicle_detection_iou))),
            imgsz=max(416, int(settings.vehicle_detection_rescue_imgsz)),
            max_det=140,
            live_priority=live_priority,
        )
        for (offset_x, _tile_x2), tile_result in zip(tile_specs, tile_results):
            tile_boxes = tile_result.boxes if getattr(tile_result, "boxes", None) is not None else []
            for box in tile_boxes:
                tx1, ty1, tx2, ty2 = [float(v) for v in box.xyxy[0].tolist()]
                add_candidate(
                    [tx1 + offset_x, ty1, tx2 + offset_x, ty2],
                    int(box.cls[0]),
                    float(box.conf[0]),
                    "tile",
                )

    # Stronger boxes first gives the downstream crop/classifiers stable input.
    candidates.sort(key=lambda item: float(item["vehicle_conf"]), reverse=True)
    # A bad/low-threshold frame must never enqueue hundreds of tire/body crops.
    # Live geometry still draws all accepted boxes; stage classification is
    # bounded to the strongest vehicles so one camera cannot monopolise CUDA.
    # Stage inference is secondary to live geometry.  A single scene with many
    # cars must not monopolise the RTX GPU for a huge tire/body batch.  Boxes are
    # still drawn for every confirmed best.pt vehicle; only stage classification
    # is capped to the strongest crops per cycle.
    max_stage_vehicles = (
        max(1, int(settings.stage_max_vehicles_per_cycle))
        if ai_uses_cuda()
        else max(1, min(3, int(settings.stage_max_vehicles_per_cycle)))
    )
    if len(candidates) > max_stage_vehicles:
        candidates = candidates[:max_stage_vehicles]

    vehicle_records: list[dict[str, Any]] = []
    vehicle_crops: list[np.ndarray] = []
    for vehicle in candidates:
        crop = _crop_vehicle(frame, list(vehicle["bbox"]))
        if crop is None:
            continue
        vehicle_records.append(dict(vehicle))
        vehicle_crops.append(crop)

    if not vehicle_records:
        return [], None

    tire_results = (
        _predict(
            tire_model,
            vehicle_crops,
            conf=max(0.01, float(settings.tire_model_confidence)),
            iou=float(settings.model_iou),
            imgsz=int(settings.tire_inference_imgsz),
            live_priority=live_priority,
        )
        if tire_model is not None else [None] * len(vehicle_crops)
    )

    fallback_indices: list[int] = []
    tire_stages: dict[int, tuple[int, float, str, int, list[dict[str, Any]]]] = {}
    for index, result in enumerate(tire_results):
        tire_stage = (
            _best_tire_stage(result, tire_model)
            if result is not None and tire_model is not None else None
        )
        if tire_stage is None:
            fallback_indices.append(index)
        else:
            tire_stages[index] = tire_stage

    body_stages: dict[int, tuple[int, float, str, int]] = {}
    if fallback_indices and body_model is not None:
        body_results = _predict(
            body_model,
            [vehicle_crops[index] for index in fallback_indices],
            imgsz=int(settings.car_flood_cls_imgsz),
            live_priority=live_priority,
        )
        for vehicle_index, result in zip(fallback_indices, body_results):
            classified = _body_stage(result, body_model)
            if classified is not None:
                body_stages[vehicle_index] = classified

    detections: list[dict[str, Any]] = []
    for index, vehicle in enumerate(vehicle_records):
        if index in tire_stages:
            raw_stage, confidence, stage_label, stage_cls_id, tire_details = tire_stages[index]
            stage_source = "tire"
        else:
            classified = body_stages.get(index)
            if classified is None:
                # Keep the best.pt vehicle box for tracking/privacy even when
                # neither stage model produced a usable flood level. The box is
                # excluded from stage majority until a tire/body result exists.
                detections.append({
                    "label": "VEHICLE DETECTED",
                    "source_label": vehicle["vehicle_label"],
                    "class_id": int(vehicle["vehicle_class_id"]),
                    "track_id": vehicle.get("track_id"),
                    "raw_stage": None,
                    "stage": None,
                    "stage_valid": False,
                    "stage_policy": None,
                    "stage_source": "vehicle_only",
                    "stage_model_label": "unclassified",
                    "conf": round(float(vehicle["vehicle_conf"]), 3),
                    "bbox": list(vehicle["bbox"]),
                    "vehicle_class_id": vehicle["vehicle_class_id"],
                    "vehicle_label": vehicle["vehicle_label"],
                    "vehicle_conf": vehicle["vehicle_conf"],
                    "detector_source": vehicle.get("detector_source", "full"),
                    "tire_detections": [],
                })
                continue
            raw_stage, confidence, stage_label, stage_cls_id = classified
            tire_details = []
            stage_source = "car_body"

        stage = max(0, min(4, int(raw_stage)))
        model_confidence = max(0.0, min(1.0, float(confidence)))
        stage_valid = qualifies_stage_confidence(
            model_confidence, settings.stage_min_confidence
        )
        stage_policy = None
        if stage_valid and stage_floor is not None:
            floor_level = max(0, min(4, int(stage_floor[0])))
            floor_conf = max(0.0, min(1.0, float(stage_floor[1])))
            if stage < floor_level:
                stage = floor_level
                confidence = max(float(confidence), floor_conf)
                stage_policy = "test_minimum"

        detections.append({
            "label": (
                f"VEHICLE Lev{stage}"
                if stage_valid
                else "VEHICLE DETECTED · STAGE <70%"
            ),
            "source_label": stage_label,
            "class_id": int(stage_cls_id),
            "track_id": vehicle.get("track_id"),
            "raw_stage": int(raw_stage),
            "stage": int(stage) if stage_valid else None,
            "stage_valid": bool(stage_valid),
            "stage_rejected_low_confidence": not stage_valid,
            "stage_min_confidence": round(float(settings.stage_min_confidence), 3),
            "stage_policy": stage_policy,
            "stage_source": stage_source,
            "stage_model_label": stage_label,
            "conf": round(float(confidence), 3),
            "bbox": list(vehicle["bbox"]),
            "vehicle_class_id": vehicle["vehicle_class_id"],
            "vehicle_label": vehicle["vehicle_label"],
            "vehicle_conf": vehicle["vehicle_conf"],
            "detector_source": vehicle.get("detector_source", "full"),
            "tire_detections": tire_details,
        })

    if not detections:
        return [], None

    valid_stage_detections = [
        item
        for item in detections
        if bool(item.get("stage_valid", item.get("stage") is not None))
    ]
    best = (
        max(
            valid_stage_detections,
            key=lambda item: (int(item.get("stage") or 0), float(item.get("conf") or 0.0)),
        )
        if valid_stage_detections
        else None
    )
    return detections, dict(best) if best is not None else None
