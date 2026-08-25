from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import math
import re
import time
from urllib.parse import quote
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, or_, select

from .camera import CameraManager
from .config import settings
from .database import init_db, session_scope
from .inference import FloodDetector
from .flood_map import build_depth_surface, level_to_depth_cm
from .external_data import get_river_levels, get_sewer_levels
from .realtime_weather import weather_service
from .environment_history import (
    environment_history_for_location,
    environment_history_recorder,
)
from .models import FloodEvent
from .schemas import CameraOut, EventOut
from .stage_consensus import choose_stage_by_count_then_confidence
from .stage_policy import (
    is_authoritative_stage_record,
    qualifies_stage_confidence,
)
from .pohang_cctv import (
    fetch_pohang_cctv,
    analyze_stream,
    annotated_mjpeg,
    raw_mjpeg,
    raw_snapshot,
    annotated_snapshot,
    live_transport_packet,
    camera_worker_status,
    has_live_cctv_clients,
    model_status,
    probe_stream,
)
from .anonymizer import safe_status as anonymizer_status
from .vworld import fetch_tile, map_config
from .dem_terrain import DemUnavailable, dem_store, terrain_flood_model


detector = FloodDetector(
    model_path=settings.model_path,
    confidence=settings.model_confidence,
    iou=settings.model_iou,
    device=settings.device,
    demo_mode=settings.demo_mode,
)
manager = CameraManager(detector)

_stage_save_lock = Lock()
_stage_db_lock = Lock()
_stage_last_saved: dict[str, tuple[datetime, int]] = {}

_vworld_overlay_lock = Lock()
_vworld_overlay_cache: dict = {
    "cameras": [],
    "flood": {"type": "FeatureCollection", "features": []},
}
_terrain_surface_lock = Lock()
_terrain_surface_cache: dict[str, dict] = {}


def _normalise_coordinate(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _is_test_camera_identity(
    camera_id: str | None,
    camera_name: str | None = None,
) -> bool:
    """Match the bundled test CCTV across its current ID, legacy IDs and name."""
    return bool(
        settings.is_test_cctv_alias(camera_id)
        or _normalised_camera_key(camera_name)
        == _normalised_camera_key(settings.test_cctv_name)
    )


def _trusted_stage(
    camera_id: str | None,
    camera_name: str | None,
    stage: int | float | None,
    confidence: int | float | None = 0.0,
) -> tuple[int, float, bool]:
    """Apply the confirmed test-video floor at the final authority boundary.

    Applying this once at every API/DB boundary prevents a live Lev0 packet,
    an old database row or a periodic list refresh from undoing the same test
    video's trusted minimum stage.
    """
    resolved_stage = max(0, min(4, int(stage or 0)))
    resolved_confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    applied = bool(
        settings.test_cctv_trusted_baseline
        and _is_test_camera_identity(camera_id, camera_name)
        and resolved_stage < max(1, min(4, int(settings.test_cctv_min_level)))
    )
    if applied:
        resolved_stage = max(1, min(4, int(settings.test_cctv_min_level)))
        resolved_confidence = max(
            resolved_confidence,
            max(0.0, min(1.0, float(settings.test_cctv_min_confidence))),
        )
    return resolved_stage, resolved_confidence, applied


def _event_region(address: str, camera_name: str) -> str:
    text = (address or camera_name or "포항 CCTV").strip()
    return text or "포항 CCTV"


def _normalised_camera_key(value: str | None) -> str:
    return re.sub(
        r"[^a-z0-9가-힣]+",
        "",
        str(value or "").strip().lower(),
    )


def _parse_history_datetime(value: str, fallback: datetime) -> datetime:
    """Convert browser ISO timestamps to the naive UTC used by SQLite."""
    text = str(value or "").strip()
    if not text:
        return fallback
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _history_camera_aliases(
    camera_id: str,
    region: str,
) -> list[str]:
    """Return current and historical IDs/names for one CCTV."""
    catalog = _authoritative_cctv_catalog()
    metadata = _configured_event_metadata(
        camera_id,
        region,
        catalog,
    )
    aliases = {
        str(camera_id or "").strip(),
        str(region or "").strip(),
        str(metadata.get("camera_id") or "").strip(),
        str(metadata.get("camera_name") or "").strip(),
        str(metadata.get("display_name") or "").strip(),
        str(metadata.get("site_name") or "").strip(),
    }
    canonical_id = str(metadata.get("camera_id") or camera_id or "")
    if settings.is_test_cctv_alias(canonical_id):
        aliases.update(settings.test_cctv_aliases())
    return sorted(value for value in aliases if value)


def _authoritative_cctv_catalog() -> dict[str, dict]:
    """
    Build the event-name catalog from the exact CCTV collection returned by
    /api/cctv/pohang. The right-side list and bottom event cards therefore
    use one authoritative name/address/coordinate source.
    """
    catalog: dict[str, dict] = {}

    try:
        cameras = _load_pohang_cameras(force=False)
    except Exception:
        # The test camera must remain available even during a public CCTV
        # API outage.
        cameras = []
        if settings.test_cctv_enabled:
            cameras.append(
                {
                    "id": settings.test_cctv_id,
                    "name": settings.test_cctv_name,
                    "address": settings.test_cctv_address,
                    "lat": settings.test_cctv_lat,
                    "lon": settings.test_cctv_lon,
                    "local_test": True,
                }
            )

    canonical_by_id: dict[str, dict] = {}

    for index, camera in enumerate(cameras):
        camera_id = str(
            camera.get("id")
            or f"CAM-{index + 1}"
        ).strip()
        camera_name = str(
            camera.get("name")
            or camera_id
        ).strip()
        address = str(
            camera.get("address")
            or camera_name
        ).strip()
        metadata = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "display_name": camera_name,
            "site_name": camera_name,
            "address": address,
            "lat": _normalise_coordinate(camera.get("lat")),
            "lon": _normalise_coordinate(camera.get("lon")),
            "local_test": bool(camera.get("local_test")),
            "matched_cctv": True,
        }

        canonical_by_id[_normalised_camera_key(camera_id)] = metadata

        for value in (
            camera_id,
            camera_name,
            address,
        ):
            key = _normalised_camera_key(value)
            if key:
                catalog[key] = metadata

    # Historical E06-041 records belong to the local flood-test video.
    # Alias insertion uses setdefault, so a real public CCTV with that exact
    # ID would take priority and would never be overwritten.
    test_metadata = canonical_by_id.get(
        _normalised_camera_key(settings.test_cctv_id)
    )
    if test_metadata is not None:
        for alias in settings.test_cctv_aliases():
            key = _normalised_camera_key(alias)
            if key:
                catalog.setdefault(key, test_metadata)

    return catalog


def _configured_event_metadata(
    camera_id: str | None,
    camera_name: str | None,
    catalog: dict[str, dict] | None = None,
) -> dict:
    source = (
        catalog
        if catalog is not None
        else _authoritative_cctv_catalog()
    )

    # ID has priority. Name matching is only a fallback for old DB rows.
    for value in (camera_id, camera_name):
        key = _normalised_camera_key(value)
        if key and key in source:
            return source[key]
    return {}


def _human_camera_name(camera_name: str | None) -> str:
    text = str(camera_name or "").strip()
    text = re.sub(
        r"^[A-Za-z]\d{2}[-_]\d{3}\s*",
        "",
        text,
    ).strip()
    return text or str(camera_name or "포항 CCTV")


def _event_details(row: FloodEvent) -> dict:
    raw = row.details or ""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"label": str(parsed)}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"label": str(raw)}


def _event_payload(
    row: FloodEvent,
    catalog: dict[str, dict] | None = None,
) -> dict:
    details = _event_details(row)
    configured = _configured_event_metadata(
        row.camera_id,
        row.camera_name,
        catalog,
    )
    matched = bool(configured)

    if matched:
        # Authoritative right-side CCTV list always wins over stale values
        # stored in old event details.
        canonical_camera_id = str(
            configured.get("camera_id")
            or row.camera_id
            or ""
        )
        canonical_camera_name = str(
            configured.get("camera_name")
            or row.camera_name
            or canonical_camera_id
        )
        display_name = canonical_camera_name
        site_name = canonical_camera_name
        address = str(
            configured.get("address")
            or canonical_camera_name
        ).strip()
        lat = _normalise_coordinate(configured.get("lat"))
        lon = _normalise_coordinate(configured.get("lon"))
    else:
        canonical_camera_id = str(row.camera_id or "")
        canonical_camera_name = str(
            row.camera_name
            or row.camera_id
            or "포항 CCTV"
        )
        display_name = str(
            details.get("display_name")
            or details.get("site_name")
            or _human_camera_name(canonical_camera_name)
            or canonical_camera_name
        ).strip()
        site_name = str(
            details.get("site_name")
            or display_name
        ).strip()
        address = str(
            details.get("address")
            or details.get("region")
            or site_name
        ).strip()
        lat = _normalise_coordinate(details.get("lat"))
        lon = _normalise_coordinate(details.get("lon"))

    level, _event_confidence, _floor_applied = _trusted_stage(
        canonical_camera_id,
        canonical_camera_name,
        row.level,
        row.confidence,
    )

    return {
        "id": row.id,
        # camera_id/name are canonical when a current CCTV-list match exists.
        "camera_id": canonical_camera_id,
        "camera_name": canonical_camera_name,
        # Preserve original DB identity for diagnostics.
        "source_camera_id": row.camera_id,
        "source_camera_name": row.camera_name,
        "matched_cctv": matched,
        "level": level,
        "confidence": _event_confidence,
        "detected_at": row.detected_at,
        "image_path": row.image_path,
        "details": row.details,
        "display_name": display_name,
        "site_name": site_name,
        "address": address or None,
        "region": address or site_name or display_name,
        "lat": lat,
        "lon": lon,
        "level_label": f"Lev{level}",
        "depth_cm": int(level_to_depth_cm(level)),
        "is_flooded": level >= 1,
    }


def _save_stage_event(
    camera_id: str,
    camera_name: str,
    result: dict,
    *,
    address: str = "",
    lat: float | None = None,
    lon: float | None = None,
    record_source: str = "foreground",
) -> None:
    stage = result.get("stage")
    if stage is None:
        return

    stage, effective_confidence, test_floor_applied = _trusted_stage(
        camera_id,
        camera_name,
        stage,
        result.get("conf"),
    )
    if not qualifies_stage_confidence(
        effective_confidence, settings.stage_min_confidence
    ):
        return
    # The bundled flood-test source is explicitly configured as a trusted
    # validation baseline. It must be allowed to create the bottom alarm card
    # even before the multi-frame public-CCTV confirmation gate completes.
    if stage > 0 and not bool(result.get("positive_confirmed")) and not test_floor_applied:
        return
    now = datetime.utcnow()
    key = camera_id or camera_name
    with _stage_save_lock:
        previous = _stage_last_saved.get(key)
        if previous and previous[1] == stage and (now - previous[0]).total_seconds() < 10:
            return
        _stage_last_saved[key] = (now, stage)

    details = json.dumps(
        {
            "label": result.get("label"),
            "site_name": _event_region(address, camera_name),
            "display_name": _event_region(address, camera_name),
            "address": (address or "").strip(),
            "region": _event_region(address, camera_name),
            "lat": _normalise_coordinate(lat),
            "lon": _normalise_coordinate(lon),
            "level_label": f"Lev{stage}",
            "depth_cm": int(level_to_depth_cm(stage)),
            "detections": len(result.get("detections") or []),
            "stage_votes": result.get("stage_votes") or {},
            "stage_confidence_averages": (
                result.get("stage_confidence_averages") or {}
            ),
            "stage_spatial": result.get("stage_spatial"),
            "stage_ema": result.get("stage_ema"),
            "stage_ema_alpha": result.get("stage_ema_alpha"),
            "stage_policy": (
                "trusted_test_baseline"
                if test_floor_applied
                else result.get("stage_policy")
            ),
            "positive_confirmed": bool(result.get("positive_confirmed")),
            "positive_confirmation": result.get("positive_confirmation") or {},
            "record_source": record_source,
            "background": record_source == "background",
            "inference_ok": True,
            "recorded_at_utc": now.isoformat(timespec="seconds") + "Z",
        },
        ensure_ascii=False,
    )

    # SQLite는 동시에 여러 쓰기 트랜잭션을 처리하지 못하므로
    # 백그라운드 작업자들의 기록을 짧게 직렬화합니다.
    with _stage_db_lock:
        with session_scope() as session:
            session.add(
                FloodEvent(
                    camera_id=camera_id or camera_name,
                    camera_name=camera_name or camera_id,
                    level=stage,
                    confidence=effective_confidence,
                    detected_at=now,
                    details=details,
                )
            )


class BackgroundCctvMonitor:
    """Round-robin CCTV inference that runs without any browser window."""

    def __init__(self) -> None:
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._status: dict = {
            "enabled": bool(settings.background_cctv_enabled),
            "running": False,
            "scanning": False,
            "cycle": 0,
            "cycle_seconds": max(30, int(settings.background_cctv_cycle_seconds)),
            "worker_count": max(1, min(12, int(settings.background_cctv_workers))),
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "stored": 0,
            "current_camera": None,
            "started_at": None,
            "last_cycle_started_at": None,
            "last_cycle_finished_at": None,
            "last_success_at": None,
            "last_recorded_at": None,
            "recent_errors": [],
            "camera_status": {},
        }

    @staticmethod
    def _iso_now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _update(self, **values) -> None:
        with self._lock:
            self._status.update(values)

    def _add_error(self, camera_name: str, error: str) -> None:
        with self._lock:
            errors = list(self._status.get("recent_errors") or [])
            errors.append({
                "camera": camera_name,
                "error": str(error)[:300],
                "at": self._iso_now(),
            })
            self._status["recent_errors"] = errors[-20:]

    def _update_camera_status(
        self,
        camera: dict,
        *,
        ok: bool,
        stored: bool = False,
        stage: int | None = None,
        confidence: float = 0.0,
        error: str | None = None,
    ) -> None:
        camera_id = str(camera.get("id") or camera.get("name") or "CCTV")
        camera_name = str(camera.get("name") or camera_id)
        now_text = self._iso_now()
        with self._lock:
            statuses = dict(self._status.get("camera_status") or {})
            previous = dict(statuses.get(camera_id) or {})
            failures = 0 if ok else int(previous.get("consecutive_failures") or 0) + 1
            statuses[camera_id] = {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "checked_at": now_text,
                "stream_ok": bool(ok),
                "stored": bool(stored),
                "stage": stage,
                "confidence": round(float(confidence or 0.0), 3),
                "last_recorded_at": (
                    now_text if stored else previous.get("last_recorded_at")
                ),
                "last_error": str(error)[:300] if error else None,
                "consecutive_failures": failures,
            }
            # Avoid an unbounded status payload when the public list changes.
            if len(statuses) > 200:
                statuses = dict(list(statuses.items())[-200:])
            self._status["camera_status"] = statuses

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._status, ensure_ascii=False))

    def start(self) -> None:
        if not settings.background_cctv_enabled:
            self._update(enabled=False, running=False)
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = Thread(
            target=self._run,
            name="background-cctv-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._update(running=False, scanning=False, current_camera=None)

    def _run(self) -> None:
        self._update(
            enabled=True,
            running=True,
            started_at=self._iso_now(),
        )

        start_delay = max(0, int(settings.background_cctv_start_delay_seconds))
        if self._stop_event.wait(start_delay):
            return

        while not self._stop_event.is_set():
            cycle_started = time.monotonic()
            try:
                self._scan_once()
            except Exception as exc:
                self._add_error("전체 순환", str(exc))
                self._update(scanning=False, current_camera=None)

            elapsed = time.monotonic() - cycle_started
            wait_seconds = max(
                5.0,
                float(settings.background_cctv_cycle_seconds) - elapsed,
            )
            if self._stop_event.wait(wait_seconds):
                break

        self._update(running=False, scanning=False, current_camera=None)

    def _scan_once(self) -> None:
        # Foreground CCTV must receive the model immediately. The previous
        # four-worker scan of 106 cameras monopolised the global model lock and
        # delayed the first live vehicle boxes by tens of seconds.
        if has_live_cctv_clients():
            self._update(
                scanning=False,
                current_camera=None,
                paused_for_live=True,
            )
            return
        self._update(paused_for_live=False)
        cameras = list(_load_pohang_cameras(force=False))
        cameras = [
            camera
            for camera in cameras
            if str(camera.get("stream_url") or "").strip()
        ]

        max_cameras = max(0, int(settings.background_cctv_max_cameras))
        if max_cameras:
            cameras = cameras[:max_cameras]

        with self._lock:
            next_cycle = int(self._status.get("cycle") or 0) + 1

        self._update(
            scanning=True,
            cycle=next_cycle,
            total=len(cameras),
            processed=0,
            success=0,
            failed=0,
            stored=0,
            current_camera=None,
            last_cycle_started_at=self._iso_now(),
            recent_errors=[],
        )

        if not cameras:
            self._update(
                scanning=False,
                last_cycle_finished_at=self._iso_now(),
            )
            return

        worker_count = max(
            1,
            min(
                len(cameras),
                12,
                int(settings.background_cctv_workers),
            ),
        )

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="cctv-scan",
        ) as executor:
            futures = {
                executor.submit(self._process_camera, camera): camera
                for camera in cameras
            }

            for future in as_completed(futures):
                if self._stop_event.is_set():
                    break
                if has_live_cctv_clients():
                    for pending in futures:
                        pending.cancel()
                    self._update(
                        scanning=False,
                        current_camera=None,
                        paused_for_live=True,
                    )
                    break

                camera = futures[future]
                name = str(camera.get("name") or camera.get("id") or "CCTV")
                ok = False
                stored = False
                result: dict = {}
                try:
                    result = future.result()
                    ok = bool(result.get("ok"))
                    stored = bool(result.get("stored"))
                    if not ok:
                        self._add_error(
                            name,
                            result.get("error") or "분석 실패",
                        )
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                    self._add_error(name, str(exc))

                self._update_camera_status(
                    camera,
                    ok=ok,
                    stored=stored,
                    stage=result.get("stage"),
                    confidence=float(result.get("confidence") or 0.0),
                    error=result.get("error"),
                )

                with self._lock:
                    self._status["processed"] = int(self._status.get("processed") or 0) + 1
                    field = "success" if ok else "failed"
                    self._status[field] = int(self._status.get(field) or 0) + 1
                    if stored:
                        self._status["stored"] = int(self._status.get("stored") or 0) + 1
                        self._status["last_recorded_at"] = self._iso_now()
                    self._status["current_camera"] = name
                    if ok:
                        self._status["last_success_at"] = self._iso_now()

        self._update(
            scanning=False,
            current_camera=None,
            last_cycle_finished_at=self._iso_now(),
        )

    def _process_camera(self, camera: dict) -> dict:
        # Live CCTV has absolute priority. A round-robin background scan used to
        # start a large tire/body CUDA batch just before a window opened, making
        # the visible stage worker wait tens of seconds. Skip background work
        # while any browser CCTV is active; the live worker records its result.
        if has_live_cctv_clients():
            return {
                "ok": True,
                "camera_id": str(camera.get("id") or camera.get("name") or "CCTV"),
                "stage": None,
                "confidence": 0.0,
                "stored": False,
                "skipped_live_priority": True,
            }

        stream_url = str(camera.get("stream_url") or "").strip()
        camera_id = str(camera.get("id") or camera.get("name") or "CCTV")
        camera_name = str(camera.get("name") or camera_id)
        if not stream_url:
            return {
                "ok": False,
                "camera_id": camera_id,
                "error": "스트림 URL 없음",
            }

        # Browser/MJPEG window state is deliberately ignored. Every cycle reads
        # a fresh frame and runs the same Stage2 YOLO used by the CCTV window.
        result = dict(analyze_stream(stream_url, force=True))
        stage = result.get("stage")

        # A public positive is intentionally not authoritative from one frame.
        # When no live worker owns this stream, collect the remaining fresh
        # samples now so a real flood can still be confirmed in one background
        # cycle instead of waiting for three five-minute scans.
        confirmation = result.get("positive_confirmation") or {}
        if (
            stage is None
            and confirmation.get("pending")
            and result.get("source") != "live_worker"
        ):
            for _ in range(4):
                time.sleep(0.75)
                result = dict(analyze_stream(stream_url, force=True))
                stage = result.get("stage")
                if stage is not None:
                    break
                confirmation = result.get("positive_confirmation") or {}
                if not confirmation.get("pending"):
                    break

        # A successfully decoded frame with no detected vehicle is a valid Lev0.
        if (
            stage is None
            and not result.get("error")
            and result.get("label") == "탐지 없음"
        ):
            stage = 0
            result["stage"] = 0
            result["label"] = "Lev0 · 탐지 없음"

        # A dead/expired source cannot be inferred. Do not write a false Lev0;
        # retain the failure only in background-status diagnostics.
        if stage is None:
            return {
                "ok": False,
                "camera_id": camera_id,
                "error": (
                    result.get("error")
                    or result.get("label")
                    or "판정값 없음"
                ),
            }

        stage = max(0, min(4, int(stage)))
        if stage == 0 and not settings.background_cctv_store_level0:
            return {
                "ok": True,
                "camera_id": camera_id,
                "stage": stage,
                "confidence": float(result.get("conf") or 0.0),
                "stored": False,
            }

        result["stage"] = stage
        _save_stage_event(
            camera_id,
            camera_name,
            result,
            address=str(camera.get("address") or "포항 CCTV"),
            lat=_normalise_coordinate(camera.get("lat")),
            lon=_normalise_coordinate(camera.get("lon")),
            record_source="background",
        )
        return {
            "ok": True,
            "camera_id": camera_id,
            "stage": stage,
            "confidence": float(result.get("conf") or 0.0),
            "stored": True,
        }


background_cctv_monitor = BackgroundCctvMonitor()


class ContinuousLocalCctvRecorder:
    """Persist local/test CCTV inference even when no CCTV window is open."""

    def __init__(self) -> None:
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._status: dict = {
            "enabled": bool(settings.background_local_enabled),
            "running": False,
            "interval_seconds": max(
                10,
                int(settings.background_local_interval_seconds),
            ),
            "camera_count": 0,
            "checks": 0,
            "stored": 0,
            "failed": 0,
            "last_started_at": None,
            "last_recorded_at": None,
            "last_stage": None,
            "last_confidence": 0.0,
            "last_error": None,
        }

    @staticmethod
    def _iso_now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _update(self, **values) -> None:
        with self._lock:
            self._status.update(values)

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._status, ensure_ascii=False))

    def start(self) -> None:
        if not (
            settings.background_cctv_enabled
            and settings.background_local_enabled
        ):
            self._update(enabled=False, running=False)
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._run,
            name="continuous-local-cctv-recorder",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._update(running=False)

    def _local_cameras(self) -> list[dict]:
        # 로컬 테스트 기록은 포항 공공 CCTV API 응답을 기다리지 않습니다.
        # 기관 API가 느리거나 장애여도 번들 영상의 추론·저장은 즉시 계속됩니다.
        cameras: list[dict] = []
        if settings.test_cctv_enabled and str(
            settings.test_cctv_video_path or ""
        ).strip():
            cameras.append({
                "id": settings.test_cctv_id,
                "name": settings.test_cctv_name,
                "address": settings.test_cctv_address,
                "lat": settings.test_cctv_lat,
                "lon": settings.test_cctv_lon,
                "stream_url": settings.test_cctv_video_path,
                "local_test": True,
                "minimum_stage": max(
                    1,
                    min(4, int(settings.test_cctv_min_level)),
                ),
                "trusted_baseline": bool(
                    settings.test_cctv_trusted_baseline
                ),
            })
        return cameras

    @staticmethod
    def _baseline_result() -> dict:
        stage = max(1, min(4, int(settings.test_cctv_min_level)))
        confidence = max(
            0.0,
            min(1.0, float(settings.test_cctv_min_confidence)),
        )
        return {
            "stage": stage,
            "label": f"TRUSTED TEST Lev{stage}",
            "conf": confidence,
            "detections": [],
            "stage_votes": {
                f"Lev{level}": 1 if level == stage else 0
                for level in range(5)
            },
            "stage_confidence_averages": {
                f"Lev{level}": confidence if level == stage else 0.0
                for level in range(5)
            },
            "stage_policy": "trusted_test_baseline",
            "positive_confirmed": True,
            "positive_confirmation": {
                "accepted": True,
                "pending": False,
                "reason": "trusted_test_baseline",
            },
        }

    def _store_baseline(self, camera: dict) -> dict:
        result = self._baseline_result()
        _save_stage_event(
            str(camera.get("id") or settings.test_cctv_id),
            str(camera.get("name") or settings.test_cctv_name),
            result,
            address=str(camera.get("address") or settings.test_cctv_address),
            lat=_normalise_coordinate(camera.get("lat")),
            lon=_normalise_coordinate(camera.get("lon")),
            record_source="background",
        )
        return {
            "ok": True,
            "stored": True,
            "stage": result["stage"],
            "confidence": result["conf"],
        }

    def _run(self) -> None:
        interval = max(
            10.0,
            float(settings.background_local_interval_seconds),
        )
        self._update(
            enabled=True,
            running=True,
            interval_seconds=int(interval),
            last_started_at=self._iso_now(),
        )

        # 전체 CCTV 순환보다 먼저 첫 기록을 만들되 서버 초기화는 잠시 기다립니다.
        initial_delay = min(
            3.0,
            max(0.0, float(settings.background_cctv_start_delay_seconds)),
        )
        if self._stop_event.wait(initial_delay):
            return

        while not self._stop_event.is_set():
            cycle_started = time.monotonic()
            try:
                cameras = self._local_cameras()
                self._update(camera_count=len(cameras))
                for camera in cameras:
                    if self._stop_event.is_set():
                        break
                    # A trusted bundled validation clip already has an explicit
                    # baseline policy. Do not spend best/tire/body GPU time on a
                    # background copy before the browser opens; V8.5.36 logs showed
                    # live workers subscribed while their detector result stayed
                    # uninitialised. This removes that startup lock/contention path.
                    if bool(camera.get("trusted_baseline")):
                        result = self._store_baseline(camera)
                    else:
                        # Non-trusted local sources still use the real AI pipeline.
                        result = background_cctv_monitor._process_camera(camera)
                    ok = bool(result.get("ok"))
                    stored = bool(result.get("stored"))
                    with self._lock:
                        self._status["checks"] = int(
                            self._status.get("checks") or 0
                        ) + 1
                        if stored:
                            self._status["stored"] = int(
                                self._status.get("stored") or 0
                            ) + 1
                            self._status["last_recorded_at"] = self._iso_now()
                        if not ok:
                            self._status["failed"] = int(
                                self._status.get("failed") or 0
                            ) + 1
                        self._status["last_stage"] = result.get("stage")
                        self._status["last_confidence"] = round(
                            float(result.get("confidence") or 0.0),
                            3,
                        )
                        self._status["last_error"] = (
                            None if ok else str(result.get("error") or "분석 실패")[:300]
                        )
            except Exception as exc:
                self._update(last_error=str(exc)[:300])

            elapsed = time.monotonic() - cycle_started
            if self._stop_event.wait(max(1.0, interval - elapsed)):
                break

        self._update(running=False)


continuous_local_recorder = ContinuousLocalCctvRecorder()


def _combined_background_status() -> dict:
    status = background_cctv_monitor.snapshot()
    local = continuous_local_recorder.snapshot()
    status["continuous_local"] = local
    status["local_stored"] = int(local.get("stored") or 0)
    status["local_last_recorded_at"] = local.get("last_recorded_at")
    status["local_last_stage"] = local.get("last_stage")
    return status


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Persist the click-independent test history synchronously. Requests can
    # arrive immediately after startup, before the recorder thread gets CPU.
    if settings.test_cctv_enabled and settings.test_cctv_trusted_baseline:
        for camera in continuous_local_recorder._local_cameras():
            continuous_local_recorder._store_baseline(camera)
    if settings.legacy_camera_workers_enabled:
        manager.start_all()
    weather_service.start()
    environment_history_recorder.start()
    background_cctv_monitor.start()
    continuous_local_recorder.start()
    yield
    continuous_local_recorder.stop()
    background_cctv_monitor.stop()
    environment_history_recorder.stop()
    weather_service.stop()
    if settings.legacy_camera_workers_enabled:
        manager.stop_all()


app = FastAPI(
    title="실시간 도시침수 모니터링 API",
    version="8.6.4",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse("app/static/index.html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/map/config")
def map_provider_config():
    return map_config()


@app.get("/vworld3d", response_class=HTMLResponse)
def vworld_webgl_3d():
    template_path = Path("app/static/vworld3d.html")
    template = template_path.read_text(encoding="utf-8")

    api_key = settings.vworld_api_key.strip()
    script_tag = ""
    if api_key:
        encoded_key = quote(api_key, safe="")
        script_tag = (
            '<script type="text/javascript" '
            'src="https://map.vworld.kr/js/webglMapInit.js.do'
            f'?version=3.0&apiKey={encoded_key}"></script>'
        )

    test_camera = {
        "id": settings.test_cctv_id,
        "name": settings.test_cctv_name,
        "address": settings.test_cctv_address,
        "lat": settings.test_cctv_lat,
        "lon": settings.test_cctv_lon,
    }

    html = (
        template
        .replace("__VWORLD_SCRIPT__", script_tag)
        .replace(
            "__TEST_CAMERA_JSON__",
            json.dumps(test_camera, ensure_ascii=False),
        )
        .replace(
            "__CCTV_GROUND_OFFSET_M__",
            json.dumps(settings.vworld_cctv_ground_offset_m),
        )
        .replace(
            "__CCTV_OCCLUSION_TOLERANCE_M__",
            json.dumps(settings.vworld_cctv_occlusion_tolerance_m),
        )
        .replace(
            "__CCTV_OCCLUSION_CHECK_FRAMES__",
            json.dumps(settings.vworld_cctv_occlusion_check_frames),
        )
        .replace(
            "__CCTV_FAR_SCALE_DISTANCE_M__",
            json.dumps(settings.vworld_cctv_far_scale_distance_m),
        )
    )
    return HTMLResponse(
        html,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/map/vworld/{layer}/{z}/{x}/{y}.{extension}")
def vworld_tile(
    request: Request,
    layer: str,
    z: int,
    x: int,
    y: int,
    extension: str,
):
    expected_extension = "jpeg" if layer == "Satellite" else "png"
    if extension.lower() not in {expected_extension, "jpg" if layer == "Satellite" else expected_extension}:
        raise HTTPException(status_code=404, detail="타일 확장자가 올바르지 않습니다.")
    try:
        tile = fetch_tile(
            layer,
            z,
            x,
            y,
            referer=request.headers.get("referer", ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(
        content=tile.content,
        media_type=tile.media_type,
        headers={
            "Cache-Control": "public, max-age=3600",
            "X-VWorld-Cache": tile.cache_status,
        },
    )


@app.get("/api/dem/status")
def dem_status():
    return dem_store.status()


@app.post("/api/dem/prefetch")
def dem_prefetch():
    try:
        dem_store.clear_failures()
        return dem_store.prefetch()
    except DemUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/dem/retry")
def dem_retry():
    dem_store.clear_failures()
    return {
        "ok": True,
        "message": "DEM 재시도 잠금을 해제했습니다.",
        "status": dem_store.status(),
    }


@app.get("/api/dem/elevation")
def dem_elevation(
    lat: float = Query(..., ge=35.0, le=37.0),
    lon: float = Query(..., ge=128.0, le=131.0),
):
    try:
        elevation_m = float(dem_store.elevation(lon, lat))
    except DemUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"DEM 고도 조회 실패: {exc}",
        ) from exc
    return {
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "elevation_m": round(elevation_m, 2),
        "source": "DEM",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "demo_mode": settings.demo_mode,
        "camera_count": len(manager.workers),
        "background_cctv": _combined_background_status(),
        "environment_history": environment_history_recorder.snapshot(),
    }


@app.get("/api/environment-history/status")
async def environment_history_status():
    return environment_history_recorder.snapshot()


@app.get("/api/cameras", response_model=list[CameraOut])
def cameras():
    return manager.snapshots()


@app.get("/api/events", response_model=list[EventOut])
def events(
    limit: int = Query(default=100, ge=1, le=1000),
    min_level: int = Query(default=0, ge=0, le=4),
    changes_only: bool = Query(default=False),
):
    catalog = _authoritative_cctv_catalog()
    with session_scope() as session:
        if not changes_only:
            rows = session.scalars(
                select(FloodEvent)
                .where(FloodEvent.level >= min_level)
                .order_by(desc(FloodEvent.detected_at))
                .limit(limit)
            ).all()
            return [_event_payload(row, catalog) for row in rows]

        # Bottom alarm board mode: keep the database/history cadence intact, but
        # surface a card only when that camera's accepted stage actually changes.
        # Scan Lev0 as well so Lev1 -> Lev0 -> Lev1 is recognised as a new event.
        scan_limit = min(5000, max(240, int(limit) * 40))
        rows = session.scalars(
            select(FloodEvent)
            .order_by(desc(FloodEvent.detected_at))
            .limit(scan_limit)
        ).all()

        by_camera: dict[str, list[FloodEvent]] = {}
        for row in rows:
            key = str(row.camera_id or row.camera_name or "").strip()
            if not key:
                continue
            by_camera.setdefault(key, []).append(row)

        changed_rows: list[FloodEvent] = []
        for camera_rows in by_camera.values():
            # camera_rows are newest -> oldest. A row is a transition only when
            # its immediately older stage differs. The oldest known row is the
            # initial state and is shown once when it already represents flooding.
            for index, row in enumerate(camera_rows):
                older = camera_rows[index + 1] if index + 1 < len(camera_rows) else None
                if older is not None and int(row.level) == int(older.level):
                    continue
                if int(row.level) < int(min_level):
                    continue
                changed_rows.append(row)

        changed_rows.sort(key=lambda row: row.detected_at, reverse=True)
        return [
            _event_payload(row, catalog)
            for row in changed_rows[:limit]
        ]



@app.get("/api/history")
def history(
    region: str = Query(default=""),
    camera_id: str = Query(default=""),
    camera_lat: float | None = Query(default=None, ge=35.0, le=37.0),
    camera_lon: float | None = Query(default=None, ge=128.0, le=131.0),
    bucket_minutes: int = Query(default=5),
    hours: int = Query(default=24, ge=1, le=24 * 365),
    start: str = Query(default=""),
    end: str = Query(default=""),
    include_environment: bool = Query(default=True),
):
    """Return CCTV flood stage together with historical rain/sewer/river data.

    Flood buckets use stored confirmed stages; equal counts use average
    confidence and then the higher stage, matching the live policy. Environmental series are read from SQLite, not from
    the current live API response, so a past period remains reproducible.
    """
    if bucket_minutes not in (1, 5, 30, 60):
        raise HTTPException(
            status_code=400,
            detail="bucket_minutes는 1, 5, 30, 60만 지원합니다.",
        )

    now = datetime.utcnow()
    try:
        range_end = _parse_history_datetime(end, now)
        range_start = _parse_history_datetime(
            start,
            range_end - timedelta(hours=hours),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="start/end 날짜 형식이 올바르지 않습니다.",
        ) from exc

    if range_end <= range_start:
        raise HTTPException(
            status_code=400,
            detail="종료 시간은 시작 시간보다 뒤여야 합니다.",
        )
    if range_end - range_start > timedelta(days=365):
        raise HTTPException(
            status_code=400,
            detail="최대 조회 범위는 1년입니다.",
        )

    direct_lat = _normalise_coordinate(camera_lat)
    direct_lon = _normalise_coordinate(camera_lon)
    if camera_id and direct_lat is not None and direct_lon is not None:
        aliases = {str(camera_id).strip(), str(region).strip()}
        if settings.is_test_cctv_alias(camera_id):
            aliases.update(settings.test_cctv_aliases())
        aliases = sorted(value for value in aliases if value)
        metadata = {
            "camera_id": camera_id,
            "camera_name": region,
            "lat": direct_lat,
            "lon": direct_lon,
        }
    else:
        aliases = _history_camera_aliases(camera_id, region)
        metadata = _configured_event_metadata(camera_id, region)

    is_test_history = bool(
        _is_test_camera_identity(camera_id, region)
        or any(settings.is_test_cctv_alias(alias) for alias in aliases)
    )

    def apply_alias_filter(stmt):
        if not aliases:
            return stmt
        return stmt.where(
            or_(
                FloodEvent.camera_id.in_(aliases),
                FloodEvent.camera_name.in_(aliases),
            )
        )

    with session_scope() as session:
        stmt = apply_alias_filter(
            select(FloodEvent)
            .where(FloodEvent.detected_at >= range_start)
            .where(FloodEvent.detected_at <= range_end)
            .order_by(FloodEvent.detected_at.asc())
        )
        rows = list(session.scalars(stmt).all())
        previous_flood = session.scalar(
            apply_alias_filter(
                select(FloodEvent)
                .where(FloodEvent.detected_at < range_start)
                .order_by(desc(FloodEvent.detected_at))
                .limit(1)
            )
        )

    # Resolve camera coordinates for nearest environmental stations.
    camera_lat = _normalise_coordinate(metadata.get("lat")) if metadata else None
    camera_lon = _normalise_coordinate(metadata.get("lon")) if metadata else None
    if (camera_lat is None or camera_lon is None) and rows:
        for candidate in reversed(rows):
            details = _event_details(candidate)
            lat = _normalise_coordinate(details.get("lat"))
            lon = _normalise_coordinate(details.get("lon"))
            if lat is not None and lon is not None:
                camera_lat, camera_lon = lat, lon
                break

    source_counts = {
        "background": 0,
        "foreground": 0,
        "legacy": 0,
    }
    bucket_rows: dict[datetime, list[dict]] = {}
    for row in rows:
        details = _event_details(row)
        record_source = str(
            details.get("record_source") or "legacy"
        ).strip().lower()
        if record_source not in source_counts:
            record_source = "legacy"
        source_counts[record_source] += 1

        dt = row.detected_at.replace(second=0, microsecond=0)
        minute = (dt.minute // bucket_minutes) * bucket_minutes
        bucket = dt.replace(minute=minute)
        stored_stage, stored_confidence, _floor_applied = _trusted_stage(
            row.camera_id if is_test_history else "",
            row.camera_name if is_test_history else "",
            row.level,
            row.confidence,
        )
        stage_votes = {level: 1 if level == stored_stage else 0 for level in range(5)}
        confidence_sums = {
            level: (
                stored_confidence
                if level == stored_stage else 0.0
            )
            for level in range(5)
        }
        bucket_rows.setdefault(bucket, []).append({
            "row": row,
            "record_source": record_source,
            "votes": stage_votes,
            "confidence_sums": confidence_sums,
        })

    flood_points: list[dict] = []
    for bucket in sorted(bucket_rows):
        items = bucket_rows[bucket]
        counts = {level: 0 for level in range(5)}
        confidence_sums = {level: 0.0 for level in range(5)}
        for item in items:
            for level in range(5):
                counts[level] += int(item["votes"].get(level, 0))
                confidence_sums[level] += float(
                    item["confidence_sums"].get(level, 0.0)
                )
        winning_stage, confidence, confidence_averages = (
            choose_stage_by_count_then_confidence(counts, confidence_sums)
        )
        latest_item = max(items, key=lambda item: item["row"].detected_at)
        representative = latest_item["row"]
        record_sources = {item["record_source"] for item in items}
        record_source = (
            next(iter(record_sources))
            if len(record_sources) == 1
            else "mixed"
        )
        flood_points.append({
            "time": bucket.isoformat() + "Z",
            "level": int(winning_stage),
            "confidence": round(float(confidence), 3),
            "camera_name": representative.camera_name,
            "camera_id": representative.camera_id,
            "record_source": record_source,
            "background": record_sources == {"background"},
            "stage_votes": {f"Lev{level}": int(counts[level]) for level in range(5)},
            "stage_confidence_averages": {
                f"Lev{level}": round(float(confidence_averages[level]), 4)
                for level in range(5)
            },
            "consensus_method": "vehicle_vote_mode_confidence_tie",
        })

    stale_seconds = max(60, int(settings.stage_stale_seconds))
    prior_stage: int | None = None
    prior_confidence = 0.0
    if previous_flood is not None:
        prior_stage, prior_confidence, _prior_floor = _trusted_stage(
            previous_flood.camera_id if is_test_history else "",
            previous_flood.camera_name if is_test_history else "",
            previous_flood.level,
            previous_flood.confidence,
        )
        age_at_start = (range_start - previous_flood.detected_at).total_seconds()
        if 0 <= age_at_start <= stale_seconds:
            flood_points.insert(0, {
                "time": range_start.isoformat() + "Z",
                "level": int(prior_stage),
                "confidence": round(float(prior_confidence), 3),
                "camera_name": previous_flood.camera_name,
                "camera_id": previous_flood.camera_id,
                "record_source": "carried_forward",
                "background": True,
                "carried_forward": True,
                "consensus_method": "latest_saved_stage_at_range_start",
            })

    # Extend the most recent confirmed stage to the requested end. This turns a
    # single fresh saved point into an immediately visible horizontal segment,
    # instead of an almost invisible lone dot.
    latest_row_time = rows[-1].detected_at if rows else (
        previous_flood.detected_at if previous_flood is not None else None
    )
    if flood_points and latest_row_time is not None:
        age_at_end = (range_end - latest_row_time).total_seconds()
        last_time = _parse_history_datetime(flood_points[-1]["time"], range_end)
        if (
            0 <= age_at_end <= stale_seconds
            and range_end - last_time > timedelta(seconds=1)
        ):
            last_point = flood_points[-1]
            flood_points.append({
                **last_point,
                "time": range_end.isoformat() + "Z",
                "record_source": "carried_forward",
                "background": True,
                "carried_forward": True,
            })

    if include_environment:
        environment = environment_history_for_location(
            range_start=range_start,
            range_end=range_end,
            bucket_minutes=bucket_minutes,
            lat=camera_lat,
            lon=camera_lon,
        )
    else:
        environment = {
            sensor_type: {"points": [], "previous": None}
            for sensor_type in ("rain", "sewer", "river")
        }

    # Align the four series on one timeline for the dashboard. Values are held
    # until the next observation so the table/chart can show all four together;
    # raw source series are also returned separately below.
    flood_by_time = {point["time"]: point for point in flood_points}
    env_by_type: dict[str, dict[str, dict]] = {}
    timeline = set(flood_by_time)
    for sensor_type in ("rain", "sewer", "river"):
        series_points = environment.get(sensor_type, {}).get("points") or []
        env_by_type[sensor_type] = {point["time"]: point for point in series_points}
        timeline.update(env_by_type[sensor_type])

    current_flood: dict | None = None
    if previous_flood is not None:
        age_at_start = (range_start - previous_flood.detected_at).total_seconds()
        if 0 <= age_at_start <= stale_seconds:
            current_flood = {
                "level": int(prior_stage if prior_stage is not None else previous_flood.level),
                "confidence": round(float(prior_confidence), 3),
            }

    current_env: dict[str, dict | None] = {}
    for sensor_type in ("rain", "sewer", "river"):
        previous = environment.get(sensor_type, {}).get("previous")
        current_env[sensor_type] = previous

    combined_points: list[dict] = []
    for time_key in sorted(timeline):
        if time_key in flood_by_time:
            current_flood = flood_by_time[time_key]
        for sensor_type in ("rain", "sewer", "river"):
            if time_key in env_by_type[sensor_type]:
                current_env[sensor_type] = env_by_type[sensor_type][time_key]
        combined_points.append({
            "time": time_key,
            "level": (int(current_flood["level"]) if current_flood is not None else None),
            "confidence": (float(current_flood.get("confidence") or 0.0) if current_flood is not None else None),
            "rain_mm": (
                round(float(current_env["rain"]["value"]), 3)
                if current_env.get("rain") is not None else None
            ),
            "sewer_level_m": (
                round(float(current_env["sewer"]["value"]), 3)
                if current_env.get("sewer") is not None else None
            ),
            "river_level_m": (
                round(float(current_env["river"]["value"]), 3)
                if current_env.get("river") is not None else None
            ),
        })

    return {
        "region": region,
        "camera_id": camera_id,
        "camera_aliases": aliases,
        "camera_location": {"lat": camera_lat, "lon": camera_lon},
        "bucket_minutes": bucket_minutes,
        "start": range_start.isoformat() + "Z",
        "end": range_end.isoformat() + "Z",
        "total_rows": len(rows),
        "source_counts": source_counts,
        "points": flood_points,
        "environment": environment,
        "combined_points": combined_points,
        "environment_included": bool(include_environment),
        "environment_history_status": environment_history_recorder.snapshot(),
    }


@app.get("/api/events/latest", response_model=EventOut | None)
def latest_event():
    catalog = _authoritative_cctv_catalog()
    with session_scope() as session:
        row = session.scalar(
            select(FloodEvent)
            .order_by(desc(FloodEvent.detected_at))
            .limit(1)
        )
        return (
            _event_payload(row, catalog)
            if row is not None
            else None
        )


@app.get("/api/flood-map")
def flood_map():
    snapshots = manager.snapshots()
    return {"surface": build_depth_surface(snapshots), "cameras": [{**c, "depth_cm": level_to_depth_cm(c["current_level"])} for c in snapshots]}

@app.get("/api/weather")
def weather():
    return weather_service.snapshot()


@app.get("/api/weather/live")
def weather_live():
    return weather_service.snapshot()


@app.get("/api/weather/rain-grid")
def pohang_rain_grid():
    return weather_service.grid_snapshot()


@app.get("/api/weather/stream")
def weather_stream():
    return StreamingResponse(
        weather_service.sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sewer-levels")
def sewer_levels():
    return get_sewer_levels()


@app.get("/api/river-levels")
def river_levels():
    return get_river_levels()


def _load_pohang_cameras(force: bool = False) -> list[dict]:
    cameras: list[dict] = []
    api_error = None

    if settings.pohang_cctv_enabled:
        try:
            cameras = list(fetch_pohang_cctv(force=force))
        except Exception as exc:
            api_error = str(exc)

    if settings.test_cctv_enabled:
        test_camera = {
            "id": settings.test_cctv_id,
            "name": settings.test_cctv_name,
            "address": settings.test_cctv_address,
            "lat": settings.test_cctv_lat,
            "lon": settings.test_cctv_lon,
            "stream_url": settings.test_cctv_video_path,
            "local_test": True,
            "minimum_stage": max(
                1,
                min(4, int(settings.test_cctv_min_level)),
            ),
            "trusted_baseline": bool(settings.test_cctv_trusted_baseline),
            "api_warning": api_error,
        }
        cameras = [
            camera
            for camera in cameras
            if str(camera.get("id")) != settings.test_cctv_id
        ]
        cameras.insert(0, test_camera)

    if cameras:
        return cameras

    if api_error:
        raise RuntimeError(f"포항 CCTV API 연결 실패: {api_error}")
    return []


@app.get("/api/cctv/pohang")
def pohang_cctv(force: bool = False):
    try:
        return _load_pohang_cameras(force=force)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/cctv/name-sync")
def cctv_name_sync():
    """Diagnose right-list and event-name synchronization."""
    cameras = _load_pohang_cameras(force=False)
    catalog = _authoritative_cctv_catalog()
    configured = settings.cameras()

    return {
        "ok": True,
        "right_list_count": len(cameras),
        "test_camera_in_right_list": any(
            str(camera.get("id")) == str(settings.test_cctv_id)
            for camera in cameras
        ),
        "test_camera": next(
            (
                camera
                for camera in cameras
                if str(camera.get("id"))
                == str(settings.test_cctv_id)
            ),
            None,
        ),
        "local_ai_cameras": [
            {
                "camera_id": camera.camera_id,
                "name": camera.name,
                "address": camera.address,
                "lat": camera.lat,
                "lon": camera.lon,
            }
            for camera in configured
        ],
        "legacy_aliases": list(settings.test_cctv_aliases()),
        "catalog_key_count": len(catalog),
    }


@app.get("/api/stage")
def stage(
    url: str = Query(..., min_length=8),
    camera_id: str = Query(default=""),
    camera_name: str = Query(default=""),
    camera_address: str = Query(default=""),
    camera_lat: float | None = Query(default=None),
    camera_lon: float | None = Query(default=None),
):
    result = dict(analyze_stream(url))
    if result.get("stage") is not None:
        resolved_stage, resolved_confidence, floor_applied = _trusted_stage(
            camera_id,
            camera_name,
            result.get("stage"),
            result.get("conf"),
        )
        result["stage"] = resolved_stage
        result["conf"] = resolved_confidence
        if floor_applied:
            result["label"] = f"MODE Lev{resolved_stage}"
            result["stage_policy"] = "trusted_test_baseline"
    if (camera_name or camera_id) and not bool(result.get("pending")):
        _save_stage_event(
            camera_id or camera_name,
            camera_name or camera_id,
            result,
            address=camera_address,
            lat=camera_lat,
            lon=camera_lon,
            record_source="foreground",
        )
    return result


@app.get("/api/cctv/frame-raw")
def cctv_raw_frame(url: str = Query(..., min_length=8)):
    jpeg, ready, sequence = raw_snapshot(url, timeout=0.45)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-CCTV-Frame-Ready": "1" if ready else "0",
            "X-CCTV-Frame-Seq": str(sequence),
        },
    )


@app.get("/api/cctv/frame-annotated")
def cctv_annotated_frame(
    url: str = Query(..., min_length=8),
    focus: bool = Query(False),
):
    jpeg, ready, sequence = annotated_snapshot(
        url, timeout=0.20, focused=bool(focus)
    )
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-CCTV-Frame-Ready": "1" if ready else "0",
            "X-CCTV-Frame-Seq": str(sequence),
        },
    )


@app.websocket("/ws/cctv")
async def cctv_websocket(websocket: WebSocket):
    """Multiplex every open CCTV over one browser connection.

    V8.5.23 polled raw + annotated JPEG endpoints independently for every
    window.  Five windows could therefore keep Chrome's HTTP/1.1 connection
    slots permanently busy, causing frame requests and even worker-status
    requests to time out together.  One WebSocket carries newest-only JPEGs for
    all visible cameras and keeps the last good canvas frame on reconnect.
    """
    await websocket.accept()
    updates: asyncio.Queue[dict] = asyncio.Queue(maxsize=8)
    subscriptions: dict[str, str] = {}
    focused_key = ""
    last_sent: dict[str, tuple[str, int, float]] = {}
    next_due: dict[str, float] = {}
    last_heartbeat_at = 0.0

    async def receive_updates() -> None:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                continue
            # Keep only a few newest UI subscription updates. Rapid focus/window
            # changes must never queue behind stale messages.
            if updates.full():
                try:
                    updates.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await updates.put(message)

    receiver = asyncio.create_task(receive_updates())
    try:
        while True:
            if receiver.done():
                exc = receiver.exception()
                if exc is not None:
                    raise exc
                break

            while True:
                try:
                    message = updates.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if str(message.get("type") or "") != "subscribe":
                    continue
                next_subscriptions: dict[str, str] = {}
                for item in list(message.get("cameras") or [])[:12]:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("key") or "").strip()
                    url = str(item.get("url") or "").strip()
                    if not key or len(url) < 3 or len(url) > 4096:
                        continue
                    next_subscriptions[key] = url
                subscriptions = next_subscriptions
                focused_key = str(message.get("focused_key") or "")
                for stale_key in list(last_sent):
                    if stale_key not in subscriptions:
                        last_sent.pop(stale_key, None)
                        next_due.pop(stale_key, None)

            now = time.monotonic()
            sent_any = False
            if now - last_heartbeat_at >= 2.0:
                await websocket.send_text('{"type":"heartbeat"}')
                last_heartbeat_at = now
                sent_any = True
            ordered_subscriptions = list(subscriptions.items())
            if focused_key:
                ordered_subscriptions.sort(key=lambda item: 0 if item[0] == focused_key else 1)
            for key, url in ordered_subscriptions:
                due = float(next_due.get(key, 0.0))
                if now < due:
                    continue
                focused = key == focused_key
                packet = live_transport_packet(url, focused=focused)
                mode = str(packet.get("mode") or "status")
                seq = int(packet.get("seq") or 0)
                previous = last_sent.get(key)
                # Duplicate real frames are not useful. A status packet is sent
                # at 1 Hz so a newly opened window still receives feedback.
                changed = previous is None or previous[0] != mode or previous[1] != seq
                keepalive_due = previous is None or now - previous[2] >= 1.0
                if changed or (mode == "status" and keepalive_due):
                    jpeg = bytes(packet.get("jpeg") or b"")
                    if jpeg:
                        header = json.dumps(
                            {
                                "key": key,
                                "mode": mode,
                                "ready": bool(packet.get("ready")),
                                "seq": seq,
                                "age_seconds": packet.get("age_seconds"),
                                "detections": packet.get("detections") or [],
                                "detection_frame_width": packet.get("detection_frame_width") or 0,
                                "detection_frame_height": packet.get("detection_frame_height") or 0,
                                "detector_ms": packet.get("detector_ms"),
                                "stage": packet.get("stage"),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        payload = struct.pack(">I", len(header)) + header + jpeg
                        await websocket.send_bytes(payload)
                        last_sent[key] = (mode, seq, now)
                        sent_any = True
                # The browser always receives the newest frame only. 12 fps for
                # the focused window and ~8 fps for tiled windows is smoother in
                # practice than asking one event loop to push 15 fps x 4-6 JPEGs.
                # Lower aggregate decode/encode pressure removes micro-stalls
                # without creating a backlog.
                next_due[key] = now + (0.095 if focused else 0.155)

            await asyncio.sleep(0.008 if sent_any else 0.025)
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        pass
    finally:
        receiver.cancel()
        try:
            await receiver
        except BaseException:
            pass


@app.get("/api/stream-annotated")
def stream_annotated(url: str = Query(..., min_length=8)):
    return StreamingResponse(
        annotated_mjpeg(url),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/stream-raw")
def stream_raw(url: str = Query(..., min_length=8)):
    return StreamingResponse(
        raw_mjpeg(url),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/privacy-model")
def privacy_model():
    """
    Always return a JSON diagnostic response.

    Missing optional models or runtime diagnostics must not turn the
    dashboard status indicator into an opaque HTTP 500 error.
    """
    return anonymizer_status()




def _latest_stage_lookup(
    limit: int = 20000,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return the current stage per CCTV.

    Recent DB rows are resolved by strict frequency mode. Equal counts use
    average confidence, matching the live CCTV decision.
    When saved per-vehicle stage_votes exist, those votes are aggregated
    directly, so a rare higher-stage vehicle cannot dominate the map.
    """
    by_id_rows: dict[str, list[dict]] = {}
    by_name_rows: dict[str, list[dict]] = {}

    cutoff = datetime.utcnow() - timedelta(
        seconds=max(60, int(settings.stage_stale_seconds))
    )
    max_records = max(1, min(20, int(settings.stage_map_consensus_max_records)))
    min_records = max(1, min(max_records, int(settings.stage_map_consensus_min_records)))
    window_seconds = max(10, int(settings.stage_map_consensus_window_seconds))

    with session_scope() as session:
        rows = session.scalars(
            select(FloodEvent)
            .where(FloodEvent.detected_at >= cutoff)
            .order_by(desc(FloodEvent.detected_at))
            .limit(limit)
        ).all()

        for row in rows:
            try:
                details = json.loads(row.details or "{}")
                raw_votes = details.get("stage_votes") or {}
                raw_confidences = details.get("stage_confidence_averages") or {}
            except Exception:
                details = {}
                raw_votes = {}
                raw_confidences = {}
            # Positive rows created before the multi-frame confirmation policy
            # are not authoritative. This prevents one old false classifier
            # result from keeping the whole city at Lev4 for 20 minutes.
            if not is_authoritative_stage_record(row.level, details):
                continue
            stage_votes = {
                f"Lev{level}": max(0, int(raw_votes.get(f"Lev{level}") or 0))
                for level in range(5)
            }
            record = {
                "camera_id": str(row.camera_id or ""),
                "camera_name": str(row.camera_name or ""),
                "stage": max(0, min(4, int(row.level))),
                "confidence": round(float(row.confidence), 4),
                "positive_confirmed": bool(details.get("positive_confirmed")),
                "stage_votes": stage_votes,
                "stage_confidence_averages": {
                    f"Lev{level}": max(
                        0.0,
                        min(1.0, float(raw_confidences.get(f"Lev{level}") or 0.0)),
                    )
                    for level in range(5)
                },
                "detected_at": row.detected_at.isoformat(),
                "_detected_at": row.detected_at,
            }
            camera_id = record["camera_id"]
            camera_name = record["camera_name"]
            if camera_id:
                bucket = by_id_rows.setdefault(camera_id, [])
                if len(bucket) < max_records:
                    bucket.append(record)
            if camera_name:
                bucket = by_name_rows.setdefault(camera_name, [])
                if len(bucket) < max_records:
                    bucket.append(record)

    def resolve(records: list[dict]) -> dict:
        if not records:
            return {}
        newest = records[0]["_detected_at"]
        recent = [
            record
            for record in records
            if (newest - record["_detected_at"]).total_seconds() <= window_seconds
        ]
        if not recent:
            recent = [records[0]]

        # V8.5.2 policy:
        # 1) The newest event's per-vehicle vote is authoritative whenever it
        #    contains at least two classified vehicles. This means a current
        #    Lev0 x5 + Lev1 x1 frame is Lev0 on the map immediately.
        # 2) If the newest frame has too few classified vehicles, stabilise with
        #    the MODE of recent frame-level stages. Each frame gets ONE vote;
        #    a frame with many detections cannot outweigh several newer frames.
        latest = recent[0]
        latest_votes = latest.get("stage_votes") or {}
        latest_counts = {
            level: max(0, int(latest_votes.get(f"Lev{level}") or 0))
            for level in range(5)
        }
        latest_total = sum(latest_counts.values())

        chosen = dict(latest)
        if latest_total >= 2:
            latest_averages = latest.get("stage_confidence_averages") or {}
            latest_confidence_sums = {
                level: (
                    float(latest_averages.get(f"Lev{level}") or 0.0)
                    * latest_counts[level]
                )
                for level in range(5)
            }
            winning_stage, winning_confidence, confidence_averages = (
                choose_stage_by_count_then_confidence(
                    latest_counts,
                    latest_confidence_sums,
                )
            )
            chosen["stage"] = int(winning_stage)
            chosen["confidence"] = round(float(winning_confidence), 4)
            chosen["consensus_records"] = 1
            chosen["consensus_votes"] = {
                f"Lev{level}": int(latest_counts[level]) for level in range(5)
            }
            chosen["consensus_confidence_averages"] = {
                f"Lev{level}": round(float(confidence_averages[level]), 4)
                for level in range(5)
            }
            chosen["consensus_method"] = "latest_vehicle_mode_confidence_tie"
        else:
            frame_counts = {level: 0 for level in range(5)}
            frame_confidence_sums = {level: 0.0 for level in range(5)}
            selected_recent = recent[:max_records]
            for record in selected_recent:
                row_votes = record.get("stage_votes") or {}
                row_counts = {
                    level: max(0, int(row_votes.get(f"Lev{level}") or 0))
                    for level in range(5)
                }
                row_total = sum(row_counts.values())
                if row_total > 0:
                    row_averages = record.get("stage_confidence_averages") or {}
                    row_confidence_sums = {
                        level: (
                            float(row_averages.get(f"Lev{level}") or 0.0)
                            * row_counts[level]
                        )
                        for level in range(5)
                    }
                    row_stage, row_confidence, _row_stage_averages = (
                        choose_stage_by_count_then_confidence(
                            row_counts,
                            row_confidence_sums,
                        )
                    )
                else:
                    row_stage = max(0, min(4, int(record["stage"])))
                    row_confidence = max(
                        0.0,
                        min(1.0, float(record.get("confidence") or 0.0)),
                    )
                frame_counts[row_stage] += 1
                frame_confidence_sums[row_stage] += float(row_confidence)

            winning_stage, winning_confidence, confidence_averages = (
                choose_stage_by_count_then_confidence(
                    frame_counts,
                    frame_confidence_sums,
                )
            )
            chosen["stage"] = int(winning_stage)
            chosen["confidence"] = round(float(winning_confidence), 4)
            chosen["consensus_records"] = len(selected_recent)
            chosen["consensus_votes"] = {
                f"Lev{level}": int(frame_counts[level]) for level in range(5)
            }
            chosen["consensus_confidence_averages"] = {
                f"Lev{level}": round(float(confidence_averages[level]), 4)
                for level in range(5)
            }
            chosen["consensus_method"] = "recent_frame_mode_confidence_tie"

        chosen.pop("_detected_at", None)
        chosen.pop("stage_votes", None)
        chosen.pop("stage_confidence_averages", None)
        return chosen

    by_id = {
        camera_id: resolve(records)
        for camera_id, records in by_id_rows.items()
    }
    by_name = {
        camera_name: resolve(records)
        for camera_name, records in by_name_rows.items()
    }
    return by_id, by_name


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110_540.0
    return math.hypot(dx, dy)


def _rain_level(rain_mm: float) -> int:
    if rain_mm < 1.0:
        return 0
    if rain_mm < 5.0:
        return 1
    if rain_mm < 15.0:
        return 2
    if rain_mm < 30.0:
        return 3
    return 4


def _rain_depth_cm(rain_mm: float) -> float:
    if rain_mm < 1.0:
        return 0.0
    return min(90.0, max(6.0, rain_mm * 1.8))


def _weather_rain_points(snapshot: dict) -> list[dict]:
    points: list[dict] = []
    grid = snapshot.get("grid") or {}

    for index, point in enumerate(grid.get("points") or []):
        lat = _normalise_coordinate(point.get("lat"))
        lon = _normalise_coordinate(point.get("lon"))
        rain = _normalise_coordinate(point.get("rain_1h_mm"))
        if lat is None or lon is None or rain is None:
            continue
        points.append(
            {
                "id": str(point.get("id") or f"KMA-{index + 1}"),
                "name": str(point.get("name") or "기상청 강수격자"),
                "lat": lat,
                "lon": lon,
                "rain_mm": max(0.0, rain),
                "observed_at": point.get("observed_at"),
            }
        )

    station = snapshot.get("station") or {}
    station_rain = _normalise_coordinate(
        snapshot.get("rain_60m_mm")
        if snapshot.get("rain_60m_mm") is not None
        else snapshot.get("rain_1h_mm")
    )
    station_lat = _normalise_coordinate(
        station.get("lat") or settings.kma_aws_station_lat
    )
    station_lon = _normalise_coordinate(
        station.get("lon") or settings.kma_aws_station_lon
    )

    if (
        station_rain is not None
        and station_lat is not None
        and station_lon is not None
        and not points
    ):
        points.append(
            {
                "id": "KMA-AWS-POHANG",
                "name": str(station.get("name") or "포항 AWS"),
                "lat": station_lat,
                "lon": station_lon,
                "rain_mm": max(0.0, station_rain),
                "observed_at": snapshot.get("observed_at"),
            }
        )

    return points


def _rain_at_camera(lat: float, lon: float, rain_points: list[dict]) -> float:
    numerator = 0.0
    denominator = 0.0

    for point in rain_points:
        distance = _distance_m(
            lat,
            lon,
            float(point["lat"]),
            float(point["lon"]),
        )
        if distance > 35_000:
            continue
        weight = 1.0 / max(400.0, distance) ** 2
        numerator += float(point["rain_mm"]) * weight
        denominator += weight

    return numerator / denominator if denominator else 0.0


def _circle_feature(
    *,
    lon: float,
    lat: float,
    radius_m: float,
    properties: dict,
    segments: int = 40,
) -> dict:
    meters_per_degree_lon = max(
        1_000.0,
        111_320.0 * math.cos(math.radians(lat)),
    )
    meters_per_degree_lat = 110_540.0
    coordinates: list[list[float]] = []

    for index in range(segments + 1):
        angle = 2.0 * math.pi * index / segments
        coordinates.append(
            [
                lon
                + math.cos(angle) * radius_m / meters_per_degree_lon,
                lat
                + math.sin(angle) * radius_m / meters_per_degree_lat,
            ]
        )

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        },
    }


def _historical_flood_points(marker_rows: list[dict]) -> list[dict]:
    now = datetime.utcnow()
    cutoff = now - timedelta(days=max(1, int(settings.dem_history_lookback_days)))
    history_end = now - timedelta(minutes=10)
    by_id = {str(item.get("id") or ""): item for item in marker_rows}
    by_name = {str(item.get("name") or ""): item for item in marker_rows}
    with session_scope() as session:
        rows = list(session.scalars(
            select(FloodEvent)
            .where(FloodEvent.level >= 1)
            .where(FloodEvent.detected_at >= cutoff)
            .where(FloodEvent.detected_at <= history_end)
            .order_by(desc(FloodEvent.detected_at))
            .limit(5000)
        ).all())
    aggregated: dict[tuple[str, float, float], dict] = {}
    lookback = max(30.0, float(settings.dem_history_lookback_days) * 0.55)
    for row in rows:
        details = _event_details(row)
        if details.get("stage_confirmed") is False:
            continue
        lat = _normalise_coordinate(details.get("lat"))
        lon = _normalise_coordinate(details.get("lon"))
        marker = by_id.get(str(row.camera_id or "")) or by_name.get(str(row.camera_name or ""))
        if marker:
            lat = lat if lat is not None else _normalise_coordinate(marker.get("lat"))
            lon = lon if lon is not None else _normalise_coordinate(marker.get("lon"))
        if lat is None or lon is None:
            continue
        key = (str(row.camera_id or row.camera_name or ""), round(lat, 5), round(lon, 5))
        age_days = max(0.0, (now - row.detected_at).total_seconds() / 86400.0)
        candidate = {
            "camera_id": str(row.camera_id or ""), "lat": lat, "lon": lon,
            "stage": max(1, min(4, int(row.level))),
            "recency": round(math.exp(-age_days / lookback), 4),
        }
        prior = aggregated.get(key)
        if prior:
            candidate["stage"] = max(prior["stage"], candidate["stage"])
            candidate["recency"] = max(prior["recency"], candidate["recency"])
        aggregated[key] = candidate
    return sorted(aggregated.values(), key=lambda item: (item["camera_id"], item["lat"], item["lon"]))


def _surface_signature(marker_rows: list[dict], history_points: list[dict]) -> str:
    payload = {
        "schema": "terrain-depth-watermask-v8.6.3",
        "sources": sorted([{
            "id": str(item.get("id") or ""),
            "lat": round(float(item.get("lat") or 0), 7),
            "lon": round(float(item.get("lon") or 0), 7),
            "stage": int(item.get("ai_stage") or item.get("stage") or 0),
            "rain": round(float(item.get("rain_mm") or 0), 2),
        } for item in marker_rows if int(item.get("ai_stage") or item.get("stage") or 0) >= 1], key=lambda item: item["id"]),
        "history": history_points,
        "settings": {
            "cell": settings.dem_flood_cell_m, "radius": settings.dem_flood_max_radius_m,
            "road": settings.dem_road_data_layer, "hydro": settings.dem_hydro_data_layers,
            "threshold": settings.dem_flood_prone_min_score,
            "water_exclusion": settings.dem_water_exclusion_enabled,
            "river_buffer_m": settings.dem_river_exclusion_buffer_m,
            "sea_max_m": settings.dem_sea_exclusion_max_elevation_m,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@app.get("/api/map/vworld-overlays")
def vworld_overlays(test: bool = False):
    errors: list[str] = []
    try:
        cameras = _load_pohang_cameras(force=False)
    except Exception as exc:
        cameras = []
        errors.append(f"CCTV: {exc}")

    if not cameras:
        cameras = [{
            "id": settings.test_cctv_id,
            "name": settings.test_cctv_name,
            "address": settings.test_cctv_address,
            "lat": settings.test_cctv_lat,
            "lon": settings.test_cctv_lon,
            "stream_url": settings.test_cctv_video_path,
            "local_test": True,
        }]

    try:
        by_id, by_name = _latest_stage_lookup()
    except Exception as exc:
        by_id, by_name = {}, {}
        errors.append(f"침수 기록 DB: {exc}")

    try:
        weather = weather_service.snapshot()
        if not isinstance(weather, dict):
            weather = {}
    except Exception as exc:
        weather = {}
        errors.append(f"기상청: {exc}")

    try:
        rain_points = _weather_rain_points(weather)
    except Exception as exc:
        rain_points = []
        errors.append(f"강수 격자: {exc}")

    marker_rows: list[dict] = []
    test_forced_applied = False
    for index, camera in enumerate(cameras):
        try:
            lat = _normalise_coordinate(camera.get("lat"))
            lon = _normalise_coordinate(camera.get("lon"))
            if lat is None or lon is None:
                continue
            camera_id = str(camera.get("id") or f"POH-{index + 1:03d}")
            camera_name = str(camera.get("name") or camera_id)
            local_test = bool(camera.get("local_test"))

            # Public CCTV records must match by camera ID. Several Pohang feeds
            # share the same display name, so a name fallback can incorrectly
            # copy one camera's level to neighbouring cameras. Name fallback is
            # reserved for the bundled test CCTV / legacy aliases only.
            record = by_id.get(camera_id)
            if record is None and (
                local_test
                or settings.is_test_cctv_alias(camera_id)
            ):
                record = by_name.get(camera_name)
            record = record or {}

            # The bundled validation source is known to contain flooding. Its
            # baseline exists before the CCTV popup is opened, so list/map/
            # bottom board use the same initial stage.
            if local_test and settings.test_cctv_trusted_baseline:
                baseline = max(1, min(4, int(settings.test_cctv_min_level)))
                if int(record.get("stage") or 0) < baseline:
                    record = {
                        **record,
                        "stage": baseline,
                        "confidence": max(
                            float(record.get("confidence") or 0.0),
                            float(settings.test_cctv_min_confidence),
                        ),
                        "source": "trusted_test_baseline",
                    }

            # A currently open CCTV worker is newer and more authoritative than
            # a DB event from minutes ago. This prevents a stale Lev4 polygon
            # remaining on the map while the live window already reports Lev0.
            live_status = camera_worker_status(str(camera.get("stream_url") or ""))
            live_age = live_status.get("latest_result_age_seconds")
            live_stage = live_status.get("latest_stage")
            if (
                live_stage is not None
                and live_age is not None
                and float(live_age) <= 15.0
            ):
                record = {
                    **record,
                    "stage": max(0, min(4, int(live_stage))),
                    "confidence": float(live_status.get("latest_confidence") or 0.0),
                    "source": "live_worker",
                }

            # Apply the test-video floor *after* every possible data source.
            # V8.5.13 applied it before the live-worker override, so a fresh
            # Lev0 packet turned the list/marker green again.
            resolved_stage, resolved_confidence, floor_applied = _trusted_stage(
                camera_id,
                camera_name,
                record.get("stage"),
                record.get("confidence"),
            )
            record = {
                **record,
                "stage": resolved_stage,
                "confidence": resolved_confidence,
                "source": (
                    "trusted_test_baseline"
                    if floor_applied
                    else record.get("source")
                ),
            }

            ai_level = max(0, min(4, int(record.get("stage") or 0)))
            confidence = float(record.get("confidence") or 0.0)
            rain_mm = _rain_at_camera(lat, lon, rain_points)
            rain_level = _rain_level(rain_mm)

            # CCTV AI is authoritative for the map level. Rainfall is retained
            # only as a terrain/spread support variable and must never promote
            # a Lev0 CCTV to Lev1 (or raise Lev1 to Lev2, etc.).
            combined_level = ai_level
            # 실시간 AI 기록이 있으면 그 침수도를 최우선으로 사용합니다.
            # 수동 강제 테스트는 아직 관측 기록이 없는 테스트 CCTV에만 적용합니다.
            # The test control may focus/open the bundled CCTV, but it must not
            # fabricate a flood polygon. Only an actual AI stage >= 1 seeds DEM.
            # Marker depth follows the AI level exactly. Rainfall can change
            # the DEM-connected spread later, but not the displayed level/depth.
            depth_cm = float(level_to_depth_cm(combined_level))
            elevation_m = None
            try:
                elevation_m = round(dem_store.elevation(lon, lat), 2)
            except Exception as exc:
                if len(errors) < 8:
                    errors.append(f"DEM {camera_name}: {exc}")
            marker_rows.append({
                "map_id": f"{camera_id}:{index}:{lat:.7f}:{lon:.7f}",
                "camera_index": index,
                "id": camera_id,
                "name": camera_name,
                "address": str(camera.get("address") or "포항 CCTV"),
                "lat": lat,
                "lon": lon,
                "elevation_m": elevation_m,
                "stage": combined_level,
                "ai_stage": ai_level,
                "rain_level": rain_level,
                "rain_mm": round(rain_mm, 2),
                "depth_cm": round(depth_cm, 1),
                "confidence": confidence,
                "local_test": local_test,
                "level_source": "cctv_ai",
            })
        except Exception as exc:
            errors.append(f"CCTV {index + 1}: {exc}")


    try:
        history_points = _historical_flood_points(marker_rows)
    except Exception as exc:
        history_points = []
        errors.append(f"기존 침수 이력: {exc}")
    surface_signature = _surface_signature(marker_rows, history_points)
    with _terrain_surface_lock:
        terrain_result = _terrain_surface_cache.get(surface_signature)
    surface_cache_hit = terrain_result is not None
    if terrain_result is None:
        terrain_result = terrain_flood_model.build_surface(
            marker_rows, rain_points, history_points=history_points
        )
        if not terrain_result.get("errors"):
            with _terrain_surface_lock:
                _terrain_surface_cache.clear()
                _terrain_surface_cache[surface_signature] = terrain_result
    errors.extend(terrain_result.get("errors") or [])
    flood_geojson = terrain_result["geojson"]
    maximum_depth = float(terrain_result.get("maximum_depth_cm") or 0.0)

    payload = {
        "source": "DEM terrain-depth bands + VWorld land-only hydro mask V8.6.3",
        "surface_policy": "terrain_depth_bands_land_only_water_excluded_geojson",
        "level_policy": "cctv_ai_authoritative_rain_support_only",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "test": test,
        "test_forced": test_forced_applied,
        "camera_count": len(marker_rows),
        "active_camera_count": sum(1 for marker in marker_rows if int(marker.get("stage") or 0) >= 1),
        "flood_feature_count": len(flood_geojson.get("features") or []),
        "maximum_depth_cm": round(maximum_depth, 1),
        "weather_source": weather.get("source"),
        "weather_observed_at": weather.get("observed_at"),
        "rain_point_count": len(rain_points),
        "terrain_method": terrain_result.get("method"),
        "terrain_sources": terrain_result.get("sources"),
        "dem_status": terrain_result.get("dem_status"),
        "context_status": terrain_result.get("context_status"),
        "historical_flood_point_count": len(history_points),
        "model_schema": terrain_result.get("model_schema"),
        "surface_signature": surface_signature,
        "surface_cache_hit": surface_cache_hit,
        "surface_pending": bool(terrain_result.get("surface_pending")),
        "errors": errors[-20:],
        "cameras": marker_rows,
        "flood": flood_geojson,
    }

    if marker_rows and flood_geojson.get("features"):
        with _vworld_overlay_lock:
            _vworld_overlay_cache["cameras"] = marker_rows
            _vworld_overlay_cache["flood"] = flood_geojson

    return payload


@app.get("/api/cctv/stages")
def cctv_stages(
    limit: int = Query(default=5000, ge=1, le=20000),
):
    """Consensus stage per CCTV, using the same strict mode as the map."""
    by_id, by_name = _latest_stage_lookup(limit=limit)

    items: list[dict] = []
    seen_names: set[str] = set()
    for camera_id, record in by_id.items():
        item = dict(record)
        item["camera_id"] = item.get("camera_id") or camera_id
        items.append(item)
        if item.get("camera_name"):
            seen_names.add(str(item["camera_name"]))

    # Keep legacy rows that genuinely have no camera_id.
    for camera_name, record in by_name.items():
        if camera_name in seen_names:
            continue
        item = dict(record)
        if item.get("camera_id"):
            continue
        item["camera_name"] = item.get("camera_name") or camera_name
        items.append(item)

    # The right-side list is periodically replaced from this endpoint. Always
    # return the configured test CCTV with its trusted minimum, even before the
    # first model cycle and even if old recent DB rows still contain Lev0.
    test_index = next(
        (
            index
            for index, item in enumerate(items)
            if _is_test_camera_identity(
                item.get("camera_id"),
                item.get("camera_name"),
            )
        ),
        None,
    )
    existing = dict(items[test_index]) if test_index is not None else {}
    resolved_stage, resolved_confidence, floor_applied = _trusted_stage(
        settings.test_cctv_id,
        settings.test_cctv_name,
        existing.get("stage"),
        existing.get("confidence"),
    )
    test_item = {
        **existing,
        "camera_id": settings.test_cctv_id,
        "camera_name": settings.test_cctv_name,
        "stage": resolved_stage,
        "confidence": resolved_confidence,
        "detected_at": existing.get("detected_at") or datetime.utcnow().isoformat(),
        "consensus_method": existing.get("consensus_method") or "trusted_test_baseline",
        "stage_policy": (
            "trusted_test_baseline"
            if floor_applied or not existing
            else existing.get("stage_policy")
        ),
    }
    if test_index is None:
        items.insert(0, test_item)
    else:
        items[test_index] = test_item

    return {
        "items": items,
        "stale_after_seconds": max(60, int(settings.stage_stale_seconds)),
        "consensus": "recent_vehicle_vote_mode_confidence_tie",
        "background": _combined_background_status(),
    }


@app.get("/api/cctv/background-status")
async def cctv_background_status():
    return _combined_background_status()


@app.get("/api/cctv/worker-status")
async def cctv_stream_worker_status(
    url: str = Query(..., min_length=3),
):
    return camera_worker_status(url)


@app.get("/api/stage-model")
def stage_model():
    return model_status()


@app.get("/api/cctv/probe")
def cctv_probe(url: str = Query(..., min_length=8)):
    return probe_stream(url)


@app.get("/stream/{camera_id}")
def stream(camera_id: str):
    worker = manager.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다.")
    return StreamingResponse(
        worker.mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/recordings/{camera_id}/{filename}")
def recording(camera_id: str, filename: str):
    base = (Path("recordings") / camera_id).resolve()
    path = (base / filename).resolve()
    if base not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(path)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)
