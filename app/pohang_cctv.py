from __future__ import annotations

import base64
import logging
import os
import queue
from collections import deque
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
import torch
import urllib3

from .config import settings
from .anonymizer import detect as detect_privacy, apply as apply_privacy
from .stage_consensus import choose_stage_by_count_then_confidence
from .stage_policy import PositiveFloodConfirmation
from .vehicle_flood_pipeline import (
    detect_vehicle_boxes,
    infer_vehicle_flood,
    model_status as vehicle_flood_model_status,
    inference_scheduler_status,
    set_live_inference_priority,
    ai_uses_cuda,
)

# Keep CPU inference from monopolising every core. The capture/render threads
# need CPU time too; otherwise MJPEG appears to buffer even though the source
# stream itself is healthy.
if not ai_uses_cuda():
    try:
        torch.set_num_threads(max(1, min(8, int(settings.ai_cpu_threads))))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150 Safari/537.36"
)

_session = requests.Session()
_session.verify = False
# Pohang UTIS began rejecting some HLS playlist/segment requests that do not
# look as if they came from the CCTV map page. Keep the browser-like headers
# on the shared session so both the playlist fallback and background sampler
# can receive the same public stream that the CCTV list advertises.
_session.headers.update({
    "User-Agent": UA,
    "Accept": "*/*",
    "Referer": "https://utis.pohang.go.kr/trafficMap/cctvMap",
})

# requests.Session is not designed to be mutated concurrently by many CCTV
# workers. Give each capture thread its own connection pool while copying the
# UTIS cookies learned by the catalogue request. This also prevents a slow
# public camera from blocking another camera's playlist requests.
_stream_http_local = threading.local()

def _stream_http_session() -> requests.Session:
    session = getattr(_stream_http_local, "session", None)
    if session is None:
        session = requests.Session()
        session.verify = False
        session.headers.update({
            "User-Agent": UA,
            "Accept": "*/*",
            "Referer": "https://utis.pohang.go.kr/trafficMap/cctvMap",
        })
        _stream_http_local.session = session
    try:
        session.cookies.update(_session.cookies)
    except Exception:
        pass
    return session

def _stream_get(url: str, *, timeout, stream: bool = False):
    """GET a public CCTV resource with a permissive header fallback.

    Some stream origins accept the UTIS Referer while others reject cross-site
    Origin/Referer headers. Try the normal browser-like request first, then a
    plain media request only for explicit access-denied responses.
    """
    session = _stream_http_session()
    response = session.get(url, timeout=timeout, stream=stream)
    if response.status_code not in {401, 403, 406}:
        return response
    response.close()
    return requests.get(
        url,
        headers={"User-Agent": UA, "Accept": "*/*"},
        verify=False,
        timeout=timeout,
        stream=stream,
    )


# V8.6.2: camera workers still own capture/render state, but they no longer
# compete for CUDA directly. Every best/tire/body predict() call is submitted to
# vehicle_flood_pipeline's single-owner central inference scheduler.  This
# helper only adjusts how frequently each camera *requests* geometry when many
# views are open; GPU ordering/fairness is handled centrally.
def _active_ai_camera_load() -> int:
    """How many CameraWorkers currently have an annotated (AI) viewer.

    Reads the same `_camera_workers` registry / `has_annotated_clients()`
    lease that gates rendering, so it stays correct for both the MJPEG
    stream path (add_client/remove_client) and the polling snapshot path
    (renew_annotated_interest) without needing separate bookkeeping calls
    threaded through every call site.
    """
    workers_lock = globals().get("_camera_workers_lock")
    workers = globals().get("_camera_workers")
    if workers_lock is None or not isinstance(workers, dict):
        return 1
    with workers_lock:
        snapshot = list(workers.values())
    return sum(
        1 for worker in snapshot
        if worker.is_alive() and worker.has_annotated_clients()
    )


_SUSTAINABLE_CONCURRENT_AI_CAMERAS = max(
    1, int(getattr(settings, "sustainable_concurrent_ai_cameras", 2))
)


def _load_scaled_interval(base_interval: float) -> float:
    load = _active_ai_camera_load()
    # RTX-class GPU can service several live best.pt streams without the CPU
    # round-robin penalty.  Scaling from only two cameras unnecessarily slowed
    # geometry updates and made boxes appear to freeze.  Keep full cadence up
    # to six live views; above that, degrade gracefully. CPU keeps the original
    # conservative scaling.
    sustainable = (
        max(6, _SUSTAINABLE_CONCURRENT_AI_CAMERAS)
        if ai_uses_cuda()
        else _SUSTAINABLE_CONCURRENT_AI_CAMERAS
    )
    if load <= sustainable:
        return base_interval
    return base_interval * (load / sustainable)


def _adaptive_cuda_ai_interval(
    base_interval: float,
    *,
    focused: bool,
) -> tuple[float, str, dict[str, Any]]:
    """Choose the next best.pt request interval from real scheduler pressure.

    V8.6.1 already removed the catastrophic CUDA contention: in the supplied
    runtime log the steady-state detector was normally 40-110 ms, result age
    was ~0.02-0.34 s, and the vehicle queue was almost always empty. The small
    remaining visual lag therefore comes mostly from waiting for the next
    detector sample, not from a slow GPU.

    This function reduces that sampling delay when the central scheduler has
    headroom. It never predicts/interpolates a box and it never changes the
    tracking source of truth: every geometry update still comes from best.pt.
    When queue pressure or vehicle runtime rises, it automatically returns to
    the normal/busy cadence so tire/body stage work cannot be starved.
    """
    scheduler = inference_scheduler_status()
    normal = max(0.12, float(base_interval))
    if not ai_uses_cuda() or not bool(settings.ai_adaptive_cadence_enabled):
        multiplier = 1.0 if focused else (1.4 if not ai_uses_cuda() else 1.15)
        return _load_scaled_interval(normal * multiplier), "fixed", scheduler

    fast = max(0.12, min(normal, float(settings.ai_geometry_fast_interval_seconds)))
    busy = max(normal, float(settings.ai_geometry_busy_interval_seconds))
    qv = max(0, int(scheduler.get("vehicle_queue") or 0))
    qs = max(0, int(scheduler.get("stage_queue") or 0))
    running_kind = str(scheduler.get("running_kind") or "")
    running_age = float(scheduler.get("running_age_seconds") or 0.0)

    last_vehicle_ms = None
    last_queue_ms = None
    if scheduler.get("last_kind") == "vehicle":
        try:
            last_vehicle_ms = float(scheduler.get("last_duration_ms"))
        except (TypeError, ValueError):
            pass
        try:
            last_queue_ms = float(scheduler.get("last_queue_ms"))
        except (TypeError, ValueError):
            pass

    busy_now = (
        qv >= int(settings.ai_geometry_busy_vehicle_queue)
        or qs >= int(settings.ai_geometry_busy_stage_queue)
        or (running_kind == "vehicle" and running_age >= 0.22)
        or (
            last_vehicle_ms is not None
            and last_vehicle_ms >= float(settings.ai_geometry_busy_vehicle_ms)
        )
        or (
            last_queue_ms is not None
            and last_queue_ms >= float(settings.ai_geometry_busy_queue_ms)
        )
    )
    fast_ok = (
        not busy_now
        and qv <= int(settings.ai_geometry_fast_max_vehicle_queue)
        and qs <= int(settings.ai_geometry_fast_max_stage_queue)
        and not (running_kind == "vehicle" and running_age >= 0.14)
        and (
            last_vehicle_ms is None
            or last_vehicle_ms <= float(settings.ai_geometry_fast_max_vehicle_ms)
        )
        and (
            last_queue_ms is None
            or last_queue_ms <= float(settings.ai_geometry_fast_max_queue_ms)
        )
    )

    if busy_now:
        interval = busy
        mode = "busy"
    elif fast_ok:
        interval = fast
        mode = "fast"
    else:
        interval = normal
        mode = "normal"

    if not focused:
        interval *= max(1.0, float(settings.ai_geometry_nonfocused_multiplier))
    return _load_scaled_interval(interval), mode, scheduler


_cache: dict[str, Any] = {"at": 0.0, "items": []}
_analysis_cache: dict[str, dict[str, Any]] = {}
_analysis_cache_lock = threading.Lock()
_positive_flood_confirmation = PositiveFloodConfirmation(
    required_hits=int(settings.positive_flood_confirmation_hits),
    minimum_duration_seconds=float(settings.positive_flood_confirmation_seconds),
    minimum_positive_vehicles=int(settings.positive_flood_min_vehicles),
    minimum_positive_ratio=float(settings.positive_flood_min_ratio),
    minimum_confidence=float(settings.stage_min_confidence),
)


class Video_EMA_Smoother:
    """Frame-to-frame probability smoother used for V8.5.0 vehicle-stage consensus."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = max(0.01, min(1.0, float(alpha)))
        self.prev_smoothed_probs: np.ndarray | None = None

    def update(self, current_probs):
        probs = np.asarray(current_probs, dtype=np.float32)
        total = float(probs.sum())
        if total <= 0:
            probs = np.zeros(5, dtype=np.float32)
            probs[0] = 1.0
        else:
            probs = probs / total

        if self.prev_smoothed_probs is None:
            self.prev_smoothed_probs = probs.copy()
        else:
            self.prev_smoothed_probs = (
                self.alpha * probs
                + (1.0 - self.alpha) * self.prev_smoothed_probs
            )

        final_level_id = int(np.argmax(self.prev_smoothed_probs))
        final_conf = float(self.prev_smoothed_probs[final_level_id])
        return final_level_id, final_conf

    def reset(self):
        self.prev_smoothed_probs = None


_stage_smoothers: dict[str, Video_EMA_Smoother] = {}
_stage_smoother_last_at: dict[str, float] = {}
_stage_smoother_lock = threading.Lock()


def _normalise_remote_url(value: str) -> str:
    """Normalise protocol-relative CCTV URLs returned by some UTIS payloads."""
    text = str(value or "").strip()
    if text.startswith("//"):
        return "https:" + text
    return text


def _resolved_stream_source(stream_url: str) -> str:
    """Resolve bundled/local video paths independently of the launch cwd."""
    text = _normalise_remote_url(stream_url)
    if not text or "://" in text:
        return text

    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _stream_label(stream_url: str) -> str:
    parsed = urlparse(str(stream_url or ""))
    if parsed.scheme and parsed.netloc:
        return parsed.netloc
    return Path(str(stream_url or "")).name or "CCTV"


def _normalised_local_path(value: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(_resolved_stream_source(value)))
    except Exception:
        return os.path.normcase(str(value or "").strip())


def _is_test_stream(stream_url: str) -> bool:
    """Return True only for the configured bundled flood-test video."""
    source = str(stream_url or "").strip()
    configured = str(settings.test_cctv_video_path or "").strip()
    if not source or not configured:
        return False
    return _normalised_local_path(source) == _normalised_local_path(configured)


def _test_stream_floor() -> tuple[int, float]:
    # The bundled clip is a known flooded validation video. Keep at least
    # Lev1 even when an old .env still contains TEST_CCTV_MIN_LEVEL=0.
    level = max(1, min(4, int(settings.test_cctv_min_level)))
    confidence = max(0.0, min(1.0, float(settings.test_cctv_min_confidence)))
    return level, confidence


def _open_video_capture(source: str):
    """Open a stream with bounded FFmpeg open/read timeouts when supported."""
    params: list[int] = []
    open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if open_timeout is not None:
        params.extend([int(open_timeout), 1200])
    if read_timeout is not None:
        # The previous five-second timeout is exactly the long frozen interval
        # visible in the supplied recording. Reconnect to the HLS live edge
        # promptly instead of leaving the MJPEG response without new frames.
        params.extend([int(read_timeout), 600])

    if params:
        try:
            capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                params,
            )
            if capture.isOpened():
                return capture
            capture.release()
        except Exception:
            pass

    parsed = urlparse(str(source or ""))
    if parsed.scheme.lower() in {"http", "https", "rtsp", "rtmp"}:
        # Do not retry a failed network open without a timeout. OpenCV can
        # otherwise block the first MJPEG frame for tens of seconds.
        return cv2.VideoCapture()
    return cv2.VideoCapture(source)


def _status_jpeg(message: str, detail: str = "") -> bytes | None:
    """Create an MJPEG frame so a failed feed never leaves a black window."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (7, 17, 27)
    cv2.rectangle(frame, (18, 18), (622, 342), (45, 91, 116), 2)
    cv2.putText(
        frame,
        message[:48],
        (42, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (105, 215, 255),
        2,
        cv2.LINE_AA,
    )
    if detail:
        cv2.putText(
            frame,
            detail[:72],
            (42, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 202, 216),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "Automatic reconnect and HLS segment fallback are active",
        (42, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (110, 232, 174),
        1,
        cv2.LINE_AA,
    )
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 76],
    )
    return bytes(buffer) if ok else None



def model_status() -> dict[str, Any]:
    """Return status for the full best -> tire -> body flood pipeline."""
    return vehicle_flood_model_status()


def fetch_pohang_cctv(force: bool = False) -> list[dict[str, Any]]:
    now = time.time()
    if not force and _cache["items"] and now - _cache["at"] < settings.pohang_cctv_refresh_seconds:
        return _cache["items"]

    _session.get("https://utis.pohang.go.kr/trafficMap/cctvMap", timeout=12)
    resp = _session.post(
        "https://utis.pohang.go.kr/api/itcs/list",
        headers={
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://utis.pohang.go.kr/trafficMap/cctvMap",
            "Origin": "https://utis.pohang.go.kr",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("포항 CCTV API 응답 형식이 예상과 다릅니다.")

    cameras: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data:
        candidates = [item, *(item.get("detail") or [])]
        parent_name = item.get("ixr_nm")
        for row in candidates:
            stream = _normalise_remote_url(row.get("hmpg_cmra_url"))
            if not stream or stream in seen:
                continue
            seen.add(stream)
            try:
                lat = float(row.get("cmra_y_crdn"))
                lon = float(row.get("cmra_x_crdn"))
                if not (33 <= lat <= 39 and 124 <= lon <= 132):
                    lat, lon = None, None
            except (TypeError, ValueError):
                lat, lon = None, None
            cameras.append({
                "id": f"POH-{len(cameras)+1:03d}",
                "name": parent_name or row.get("istl_lctn") or row.get("drct_lctn") or "이름없음",
                "address": row.get("istl_lctn") or row.get("drct_lctn") or "",
                "lat": lat,
                "lon": lon,
                "stream_url": stream,
            })

    if not cameras:
        raise RuntimeError("사용 가능한 CCTV 스트림 URL을 찾지 못했습니다.")
    _cache.update({"at": now, "items": cameras})
    return cameras


def _playlist_uri_lines(text: str) -> list[str]:
    """Return URI lines from an HLS manifest without guessing by extension.

    Some public CCTV servers expose MPEG-TS segments through extensionless
    query URLs. The old extension filter treated those valid media playlists as
    empty, which left every CCTV window at "connecting".
    """
    return [
        line.strip()
        for line in str(text or "").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _parse_hls(stream_url: str, timeout: float = 12) -> tuple[str, list[str]]:
    stream_url = _normalise_remote_url(stream_url)
    master_resp = _stream_get(stream_url, timeout=timeout)
    master_resp.raise_for_status()
    master_text = master_resp.text
    uri_lines = _playlist_uri_lines(master_text)
    if not uri_lines:
        raise RuntimeError("HLS 재생목록이 비어 있습니다.")

    playlist_url = stream_url
    # EXT-X-STREAM-INF is the standards-based signal for a master playlist.
    # Do not infer the playlist type from file extensions: UTIS occasionally
    # returns extensionless segment URLs.
    is_master = "#EXT-X-STREAM-INF" in master_text.upper()
    if is_master:
        # Prefer the last advertised variant. Public CCTV masters commonly list
        # variants from low to high bandwidth, and a usable picture is more
        # valuable than repeatedly opening the first stale/placeholder variant.
        child = uri_lines[-1]
        playlist_url = urljoin(stream_url, child)
        media_resp = _stream_get(playlist_url, timeout=timeout)
        media_resp.raise_for_status()
        media_text = media_resp.text
        uri_lines = _playlist_uri_lines(media_text)
        if not uri_lines:
            raise RuntimeError("HLS 하위 재생목록이 비어 있습니다.")
        if "#EXT-X-STREAM-INF" in media_text.upper():
            raise RuntimeError("중첩 HLS 마스터 재생목록은 지원하지 않습니다.")

    segments = [urljoin(playlist_url, item) for item in uri_lines]
    if not segments:
        raise RuntimeError("HLS 영상 세그먼트를 찾지 못했습니다.")
    return playlist_url, segments


def _media_playlist_segments(playlist_url: str, *, quick: bool = True) -> list[str]:
    """Refresh a previously resolved media playlist without re-fetching master."""
    response = _stream_get(
        playlist_url,
        timeout=(0.75, 1.50) if quick else (3.0, 12.0),
    )
    response.raise_for_status()
    text = response.text
    if "#EXT-X-STREAM-INF" in text.upper():
        raise RuntimeError("HLS 미디어 재생목록이 마스터로 변경되었습니다.")
    uri_lines = _playlist_uri_lines(text)
    if not uri_lines:
        raise RuntimeError("HLS 미디어 재생목록이 비어 있습니다.")
    return [urljoin(playlist_url, item) for item in uri_lines]


def _choose_unseen_hls_segment(
    segments: list[str],
    previous_segment: str | None,
) -> str | None:
    if not segments:
        return None
    if previous_segment and previous_segment in segments:
        previous_index = len(segments) - 1 - segments[::-1].index(previous_segment)
        newer = segments[previous_index + 1:]
        # Monitoring prioritises freshness over replaying every stale segment.
        # If decoding fell behind, jump directly to the newest advertised live
        # segment instead of spending several seconds catching up.
        return newer[-1] if newer else None
    # Start at the live edge.  The downloader already retries the previous
    # segment when the newest URI has been advertised before its bytes are fully
    # published.  Starting one full segment behind added visible latency before
    # the first moving frame and made reconnects look like buffering.
    return segments[-1]


def _download_latest_hls_segment(
    stream_url: str,
    *,
    quick: bool = True,
    previous_segment: str | None = None,
    media_playlist_url: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Download the newest unseen HLS segment and reuse a resolved media URL."""
    if media_playlist_url:
        try:
            playlist_url = media_playlist_url
            segments = _media_playlist_segments(playlist_url, quick=quick)
        except Exception:
            playlist_url, segments = _parse_hls(
                stream_url,
                timeout=(0.75, 1.80) if quick else 12,
            )
    else:
        playlist_url, segments = _parse_hls(
            stream_url,
            timeout=(0.75, 1.80) if quick else 12,
        )

    segment_url = _choose_unseen_hls_segment(segments, previous_segment)
    if not segment_url:
        return playlist_url, None, None

    # If a playlist briefly advertises a segment before its bytes are ready,
    # try one older unseen candidate before declaring the feed unavailable.
    candidates = [segment_url]
    if previous_segment not in segments:
        try:
            idx = segments.index(segment_url)
            if idx > 0:
                candidates.append(segments[idx - 1])
        except ValueError:
            pass

    last_error: Exception | None = None
    for candidate in candidates:
        data_resp = None
        try:
            data_resp = _stream_get(
                candidate,
                timeout=(0.75, 1.90) if quick else (3.0, 15.0),
                stream=True,
            )
            data_resp.raise_for_status()
            suffix = Path(candidate.split("?", 1)[0]).suffix or ".ts"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as file:
                for chunk in data_resp.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        file.write(chunk)
                temp_name = file.name
            return playlist_url, candidate, temp_name
        except Exception as exc:
            last_error = exc
        finally:
            if data_resp is not None:
                try:
                    data_resp.close()
                except Exception:
                    pass
    if last_error is not None:
        raise last_error
    return playlist_url, None, None

class _HlsSegmentPrefetcher:
    """Keep one or two HLS segments downloaded ahead of the decoder.

    V8.5.22 downloaded the next segment only after OpenCV reached EOF on the
    current local file. That inserted a network pause at every segment boundary
    and became very obvious when several CCTV windows were open. This helper
    refreshes the playlist and downloads the next segment in a dedicated thread
    while the grab thread is still decoding the current segment.
    """

    def __init__(self, stream_url: str, parent_stop: threading.Event):
        self.stream_url = stream_url
        self.parent_stop = parent_stop
        self.stop_event = threading.Event()
        # Keep only the next newest segment.  A two-segment FIFO can become
        # several seconds behind the live edge if decoding or the browser pauses.
        # Latest-only replacement intentionally drops stale prefetched segments.
        self.items: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
        self.media_playlist_url: str | None = None
        self.previous_segment: str | None = None
        self.last_progress_at = 0.0
        self.last_error: str | None = None
        self.thread = threading.Thread(
            target=self._run, name="cctv-hls-prefetch", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=0.35)
        while True:
            try:
                _segment, path = self.items.get_nowait()
            except queue.Empty:
                break
            try:
                os.remove(path)
            except OSError:
                pass

    def get(self, timeout: float = 0.20) -> tuple[str, str] | None:
        try:
            return self.items.get(timeout=max(0.01, float(timeout)))
        except queue.Empty:
            return None

    def _run(self) -> None:
        while not self.stop_event.is_set() and not self.parent_stop.is_set():
            try:
                media_url, segment_url, temp_path = _download_latest_hls_segment(
                    self.stream_url,
                    quick=True,
                    previous_segment=self.previous_segment,
                    media_playlist_url=self.media_playlist_url,
                )
                if media_url:
                    self.media_playlist_url = media_url
                if not segment_url or not temp_path:
                    self.stop_event.wait(0.05)
                    continue
                if self.stop_event.is_set() or self.parent_stop.is_set():
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                    break
                self.previous_segment = segment_url
                self.last_progress_at = time.monotonic()
                self.last_error = None
                # Never let a queued-but-not-yet-decoded HLS segment become a
                # backlog. Replace it with the newer live-edge segment and delete
                # the stale temporary file. This trades skipped frames for low
                # latency, which is the correct behaviour for monitoring CCTV.
                if self.items.full():
                    try:
                        _old_segment, old_path = self.items.get_nowait()
                    except queue.Empty:
                        old_path = None
                    if old_path:
                        try:
                            os.remove(old_path)
                        except OSError:
                            pass
                try:
                    self.items.put_nowait((segment_url, temp_path))
                except queue.Full:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            except Exception as exc:
                self.last_error = str(exc)
                # Re-resolve a stale/rotated master token on the next attempt.
                self.media_playlist_url = None
                self.stop_event.wait(0.12)


def probe_stream(stream_url: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        playlist, segments = _parse_hls(stream_url)
        frame = _read_frame(stream_url)
        h, w = frame.shape[:2]
        return {
            "ok": True,
            "protocol": "HLS",
            "playlist_url": playlist,
            "segment_count": len(segments),
            "frame_width": w,
            "frame_height": h,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "protocol": "HLS",
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "error": str(exc),
        }


def _read_frame(
    stream_url: str,
    *,
    quick: bool = False,
    skip_direct: bool = False,
):
    source = _resolved_stream_source(stream_url)
    is_local = "://" not in str(stream_url or "")

    # OpenCV가 HLS 또는 로컬 영상을 직접 읽을 수 있으면 가장 먼저 사용합니다.
    cap = _open_video_capture(source) if not skip_direct else cv2.VideoCapture()
    if cap.isOpened():
        frame = None

        # 로컬 침수 테스트 영상은 시작부의 검은 프레임/인트로가 아니라
        # 실제 침수 장면을 읽습니다. 백그라운드 분석이 매번 파일 첫 장면만
        # 판정해 기록을 놓치던 문제를 방지합니다.
        if is_local and _is_test_stream(stream_url):
            sample_seconds = max(0.0, float(settings.test_cctv_sample_seconds))
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, sample_seconds * 1000.0)
            except Exception:
                pass
            reads = 4
        elif is_local:
            reads = 8
        else:
            # A network capture is opened at the current HLS live edge. One
            # decoded frame is enough; reading eight future frames here could
            # block for several read-timeout periods inside the live worker's
            # recovery path.
            reads = 1

        for _ in range(reads):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
        cap.release()
        if frame is not None:
            return frame
    else:
        cap.release()

    # 로컬 파일은 HLS 파서로 넘기지 않습니다.
    if "://" not in str(stream_url or ""):
        raise RuntimeError(
            f"로컬 CCTV 영상을 읽지 못했습니다: {source}"
        )

    # 직접 읽기가 실패하면 마지막 HLS 세그먼트를 내려받아 디코딩합니다.
    _, _, temp_path = _download_latest_hls_segment(
        stream_url,
        quick=quick,
    )
    if not temp_path:
        raise RuntimeError("새 HLS 영상 세그먼트를 기다리는 중입니다.")
    try:
        cap = _open_video_capture(temp_path)
        frame = None
        for _ in range(20):
            ok, candidate = cap.read()
            if not ok:
                break
            frame = candidate
        cap.release()
        if frame is None:
            raise RuntimeError("CCTV 영상 프레임을 디코딩하지 못했습니다.")
        return frame
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _stage_from_class(cls_id: int, label: str) -> int:
    # 모델 클래스가 0~4로 학습된 경우 class id를 그대로 사용합니다.
    if 0 <= cls_id <= 4:
        return cls_id
    # 클래스 이름에 단계 숫자가 포함된 모델도 처리합니다.
    digits = [int(ch) for ch in label if ch.isdigit()]
    return max(0, min(4, digits[0] if digits else 0))


def _apply_test_stage_floor(
    stream_url: str,
    raw_stage: int,
    confidence: float,
) -> tuple[int, float, bool]:
    """
    The bundled test clip is a known flooded validation source.

    The detector may classify one vehicle as level_0 on an individual frame.
    Keep the raw value for diagnostics, but do not let the known flooded test
    source fall below TEST_CCTV_MIN_LEVEL. Real public CCTV streams are never
    changed by this policy.
    """
    stage = max(0, min(4, int(raw_stage)))
    conf = max(0.0, min(1.0, float(confidence or 0.0)))
    if not _is_test_stream(stream_url) or not bool(settings.test_cctv_enforce_min_level):
        return stage, conf, False
    floor, floor_conf = _test_stream_floor()
    if stage >= floor:
        return stage, conf, False
    return floor, max(conf, floor_conf), True


def _frame_stage_probabilities(
    detections: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[int, int], dict[int, float], int, float, int]:
    """Convert vehicle stages into a strict frequency vote.

    The class with the largest VEHICLE COUNT is authoritative. Confidence is
    deliberately not allowed to overturn the count. When counts are tied,
    choose the stage with the higher average vehicle confidence.
    """
    votes = {level: 0 for level in range(5)}
    confidence_sums = {level: 0.0 for level in range(5)}

    valid_stage_count = 0
    for detection in detections:
        if detection.get("stage") is None or detection.get("stage_valid") is False:
            continue
        stage = max(0, min(4, int(detection.get("stage"))))
        confidence = max(0.0, min(1.0, float(detection.get("conf") or 0.0)))
        votes[stage] += 1
        confidence_sums[stage] += confidence
        valid_stage_count += 1

    total_votes = max(1, sum(votes.values()))
    probs = np.array(
        [votes[level] / float(total_votes) for level in range(5)],
        dtype=np.float32,
    )
    spatial_stage, stage_confidence, confidence_averages = (
        choose_stage_by_count_then_confidence(votes, confidence_sums)
    )
    return (
        probs, votes, confidence_averages, spatial_stage,
        stage_confidence, valid_stage_count,
    )


def _consensus_stage(
    stream_url: str,
    detections: list[dict[str, Any]],
) -> tuple[int | None, float, dict[str, Any]]:
    """Strict vehicle-frequency majority with EMA retained only as a diagnostic.

    Example: Lev0 x5 + Lev1 x1 MUST return Lev0 immediately, regardless of
    previous EMA state. This prevents a rare high-stage outlier from driving
    the CCTV badge or map.
    """
    (
        current_probs,
        votes,
        confidence_averages,
        spatial_stage,
        stage_confidence,
        valid_stage_count,
    ) = _frame_stage_probabilities(detections)

    if valid_stage_count == 0:
        diagnostics = {
            "votes": {f"Lev{level}": 0 for level in range(5)},
            "confidence_averages": {f"Lev{level}": 0.0 for level in range(5)},
            "spatial_stage": None,
            "ema_stage": None,
            "ema_conf": 0.0,
            "ema_alpha": float(settings.stage_ema_alpha),
            "decision_method": "no_vote_below_minimum_confidence",
            "test_floor_applied": False,
            "minimum_confidence": float(settings.stage_min_confidence),
            "valid_stage_count": 0,
            "positive_confirmed": False,
            "positive_confirmation": {},
        }
        return None, 0.0, diagnostics
    now = time.monotonic()
    key = (
        _normalised_local_path(stream_url)
        if "://" not in str(stream_url or "")
        else str(stream_url)
    )

    with _stage_smoother_lock:
        smoother = _stage_smoothers.get(key)
        last_at = float(_stage_smoother_last_at.get(key) or 0.0)
        if smoother is None:
            smoother = Video_EMA_Smoother(float(settings.stage_ema_alpha))
            _stage_smoothers[key] = smoother
        elif last_at and now - last_at > max(1.0, float(settings.stage_ema_reset_seconds)):
            smoother.reset()
        ema_stage, ema_conf = smoother.update(current_probs)
        _stage_smoother_last_at[key] = now

    # The actual output is the strict mode. EMA cannot override a vehicle-count
    # majority. Confidence is used only to resolve equal vehicle counts.
    if bool(settings.stage_strict_majority):
        final_stage = int(spatial_stage)
        final_conf = float(stage_confidence)
    else:
        final_stage = int(ema_stage)
        final_conf = float(ema_conf)

    # The bundled clip is a confirmed flooded validation source. Public CCTV
    # streams are never floored, and their confidence-tie decision is kept.
    policy_applied = bool(
        _is_test_stream(stream_url)
        and settings.test_cctv_trusted_baseline
        and final_stage < _test_stream_floor()[0]
    )
    if policy_applied:
        floor_stage, floor_conf = _test_stream_floor()
        final_stage = floor_stage
        final_conf = max(final_conf, floor_conf)

    diagnostics = {
        "votes": {f"Lev{level}": int(votes[level]) for level in range(5)},
        "confidence_averages": {
            f"Lev{level}": round(float(confidence_averages[level]), 4)
            for level in range(5)
        },
        "spatial_stage": int(spatial_stage),
        "ema_stage": int(ema_stage),
        "ema_conf": round(float(ema_conf), 3),
        "ema_alpha": float(settings.stage_ema_alpha),
        "decision_method": "vehicle_mode_confidence_tie_higher_stage_final_tie",
        "test_floor_applied": policy_applied,
        "minimum_confidence": float(settings.stage_min_confidence),
        "valid_stage_count": int(valid_stage_count),
    }

    positive_votes = sum(int(votes[level]) for level in range(1, 5))
    confirmation = _positive_flood_confirmation.evaluate(
        key,
        final_stage,
        final_conf,
        positive_votes=positive_votes,
        total_votes=valid_stage_count,
        trusted_test=bool(
            _is_test_stream(stream_url)
            and settings.test_cctv_trusted_baseline
        ),
    )
    diagnostics["positive_confirmation"] = confirmation
    diagnostics["positive_confirmed"] = bool(
        final_stage == 0 or confirmation.get("accepted")
    )
    if final_stage > 0 and not confirmation.get("accepted"):
        diagnostics["decision_method"] = str(confirmation.get("reason") or "positive_pending")
        return None, 0.0, diagnostics
    return int(final_stage), round(float(final_conf), 3), diagnostics


def _draw_tracked_result(frame, result, model) -> tuple[Any, list[dict[str, Any]], tuple[int, float, str, int] | None]:
    """Draw every YOLO detection on the exact frame that produced it."""
    annotated = frame.copy()
    detections: list[dict[str, Any]] = []
    best: tuple[int, float, str, int] | None = None
    stage_colors = {
        0: (120, 200, 90),
        1: (255, 190, 70),
        2: (70, 210, 255),
        3: (40, 130, 255),
        4: (70, 50, 255),
    }

    boxes = result.boxes if result.boxes is not None else []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = str(model.names[cls_id])
        stage = _stage_from_class(cls_id, label)
        xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        track_id = None
        if getattr(box, "id", None) is not None:
            try:
                track_id = int(box.id[0])
            except Exception:
                track_id = None

        color = stage_colors.get(stage, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        id_text = f" #{track_id}" if track_id is not None else ""
        vehicle_label = f"VEHICLE{id_text} Lev{stage}  {conf * 100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(
            vehicle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
        )
        label_top = max(0, y1 - th - baseline - 10)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(annotated.shape[1] - 1, x1 + tw + 12), y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            vehicle_label,
            (x1 + 6, max(th + 2, y1 - baseline - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (10, 16, 22),
            2,
            cv2.LINE_AA,
        )

        detections.append({
            "label": f"VEHICLE Lev{stage}",
            "source_label": label,
            "class_id": cls_id,
            "track_id": track_id,
            "stage": stage,
            "conf": round(conf, 3),
            "bbox": xyxy,
        })
        if best is None or (stage, conf) > (best[3], best[1]):
            best = (cls_id, conf, label, stage)

    overall_stage = best[3] if best is not None else 0
    banner_color = stage_colors.get(overall_stage, (255, 255, 255))
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 38), (10, 18, 28), -1)
    cv2.putText(
        annotated,
        f"AI VEHICLE FLOOD TRACKING  Lev{overall_stage}  vehicles:{len(detections)}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        banner_color,
        2,
        cv2.LINE_AA,
    )
    return annotated, detections, best


def _analyze_stream_uncached(stream_url: str) -> dict[str, Any]:
    status = vehicle_flood_model_status()
    if not bool(status.get("loaded")):
        return {
            "stage": None,
            "label": status.get("error") or "차량/타이어/차체 모델을 불러오지 못했습니다.",
            "conf": 0,
            "snapshot": None,
            "detections": [],
            "pipeline": "vehicle -> tire_level -> car_flood_cls fallback",
        }

    started = time.perf_counter()
    try:
        frame = _read_frame(stream_url)
        h, w = frame.shape[:2]
        analysis_width = max(
            int(settings.stage_max_width),
            int(settings.stage_tracking_frame_width),
        )
        if w > analysis_width:
            scale = analysis_width / float(w)
            frame = cv2.resize(
                frame,
                (analysis_width, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        # V8.5.0 pipeline:
        # best.pt detects the vehicle, tire_level.pt judges the cropped tire,
        # and car_flood_cls.pt is used only when no tire is detected.
        detections, _representative = infer_vehicle_flood(
            frame,
            vehicle_imgsz=(
                int(settings.vehicle_detection_imgsz)
                if ai_uses_cuda()
                else min(768, int(settings.vehicle_detection_imgsz))
            ),
            stage_floor=None,
        )

        final_stage, final_conf, consensus = _consensus_stage(
            stream_url,
            detections,
        )
        elapsed = round((time.perf_counter() - started) * 1000)

        annotated = _draw_live_detections(
            frame,
            detections,
            elapsed,
            final_stage,
            consensus["votes"],
        )
        ok, buf = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality],
        )
        snapshot = base64.b64encode(buf).decode("utf-8") if ok else None

        source_counts = {"tire": 0, "car_body": 0}
        for detection in detections:
            source = str(detection.get("stage_source") or "")
            if source in source_counts:
                source_counts[source] += 1

        return {
            "stage": final_stage,
            "label": (
                f"MODE Lev{final_stage}"
                if final_stage is not None
                else (
                    "FLOOD CANDIDATE · VERIFYING"
                    if (consensus.get("positive_confirmation") or {}).get("pending")
                    else "STAGE HOLD · CONFIDENCE <70%"
                )
            ),
            "conf": final_conf,
            "snapshot": snapshot,
            "detections": detections,
            "stage_votes": consensus["votes"],
            "stage_confidence_averages": consensus["confidence_averages"],
            "stage_spatial": consensus["spatial_stage"],
            "stage_ema": consensus["ema_stage"],
            "stage_ema_alpha": consensus["ema_alpha"],
            "positive_confirmed": bool(consensus.get("positive_confirmed")),
            "positive_confirmation": consensus.get("positive_confirmation") or {},
            "stage_policy": (
                "test_minimum"
                if consensus["test_floor_applied"]
                else "strict_vehicle_mode"
            ),
            "stage_source_counts": source_counts,
            "pipeline": "vehicle -> tire_level -> car_flood_cls fallback -> strict mode (EMA diagnostic)",
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "inference_ms": elapsed,
        }
    except Exception as exc:
        return {
            "stage": None,
            "label": "판정 실패",
            "error": str(exc),
            "conf": 0,
            "snapshot": None,
            "detections": [],
            "pipeline": "vehicle -> tire_level -> car_flood_cls fallback",
            "inference_ms": round((time.perf_counter() - started) * 1000),
        }


def analyze_stream(stream_url: str, force: bool = False) -> dict[str, Any]:
    # An open browser feed already owns a live worker. Reuse its result even
    # while classification is still pending; starting a second synchronous
    # /api/stage inference here was a hidden source of model-lock contention
    # and multi-second CCTV stalls.
    workers_lock = globals().get("_camera_workers_lock")
    workers = globals().get("_camera_workers")
    if workers_lock is not None and isinstance(workers, dict):
        with workers_lock:
            live_worker = workers.get(stream_url)
        if live_worker is not None and live_worker.is_alive():
            with live_worker.result_lock:
                return {
                    **live_worker.latest_result,
                    "source": "live_worker",
                    "pending": live_worker.latest_result.get("stage") is None,
                }

    now = time.monotonic()

    with _analysis_cache_lock:
        cached = _analysis_cache.get(stream_url)
        if (
            not force
            and cached
            and now - float(cached["at"]) < settings.stage_result_cache_seconds
        ):
            return cached["result"]

    result = _analyze_stream_uncached(stream_url)

    with _analysis_cache_lock:
        _analysis_cache[stream_url] = {
            "at": time.monotonic(),
            "result": result,
        }

        if len(_analysis_cache) > 150:
            oldest = min(
                _analysis_cache,
                key=lambda key: _analysis_cache[key]["at"],
            )
            _analysis_cache.pop(oldest, None)

    return result


# ---------------------------------------------------------------------------
# 실시간 CCTV 처리 워커
#
# 1) _grab_loop   : CCTV에서 최신 프레임만 계속 수집
# 2) _ai_loop     : 낮은 주기로 YOLO 추론
# 3) _render_loop : 최신 프레임에 최신 박스를 그려 MJPEG 생성
#
# 영상 출력이 YOLO 추론을 기다리지 않으므로 추론 중에도 CCTV가 멈추지 않습니다.
# ---------------------------------------------------------------------------

_STAGE_COLORS = {
    0: (120, 200, 90),
    1: (255, 190, 70),
    2: (70, 210, 255),
    3: (40, 130, 255),
    4: (70, 50, 255),
}

_camera_workers: dict[str, "CameraWorker"] = {}
_camera_workers_lock = threading.Lock()


def _copy_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for detection in detections:
        item = dict(detection)
        item["bbox"] = list(detection.get("bbox") or [])
        copied.append(item)
    return copied


def _clip_bbox(
    bbox: list[float],
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(x1 + 1.0, min(float(width - 1), float(x2)))
    y2 = max(y1 + 1.0, min(float(height - 1), float(y2)))
    return [x1, y1, x2, y2]


def _project_detections_by_velocity(
    detections: list[dict[str, Any]],
    *,
    delta_seconds: float,
    detector_interval_seconds: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Conservative display-only CUDA bbox prediction.

    Velocity comes only from successive real best.pt matches. Projected geometry
    is never written back to canonical tracking or stage inference, and every
    fresh detector packet replaces it.
    """
    dt = max(0.0, float(delta_seconds))
    if dt <= 0.0 or not detections:
        return _copy_detections(detections)

    reference = max(0.08, float(detector_interval_seconds))
    step_scale = min(1.0, dt / reference)
    projected: list[dict[str, Any]] = []
    for detection in detections:
        item = dict(detection)
        bbox = [float(v) for v in (item.get("bbox") or [])]
        velocity = item.get("_velocity") or [0.0, 0.0]
        if len(bbox) != 4 or len(velocity) < 2:
            projected.append(item)
            continue
        x1, y1, x2, y2 = bbox
        box_w = max(2.0, x2 - x1)
        box_h = max(2.0, y2 - y1)
        dx = float(velocity[0]) * step_scale
        dy = float(velocity[1]) * step_scale
        dx = max(-box_w * 0.40, min(box_w * 0.40, dx))
        dy = max(-box_h * 0.32, min(box_h * 0.32, dy))
        item["bbox"] = _clip_bbox(
            [x1 + dx, y1 + dy, x2 + dx, y2 + dy], width, height
        )
        projected.append(item)
    return projected


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center_ratio(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    diag = max(8.0, np.hypot(ax2 - ax1, ay2 - ay1), np.hypot(bx2 - bx1, by2 - by1))
    return float(np.hypot(acx - bcx, acy - bcy) / diag)


def _confirm_vehicle_candidates(
    current: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Confirm weak best.pt candidates without losing real night-CCTV cars.

    V8.5.33 used a 4% post-filter on top of a 3% model-side confidence floor.
    That was too aggressive for the Pohang night feeds and could legitimately
    reduce a busy road to ``vehicles 0``. V8.5.37 accepts the raw YOLO geometry
    again, but weak candidates must repeat at nearly the same moving position
    before they become visible. Static near-zero noise therefore never goes
    straight to the renderer or the flood-stage models.
    """
    confirmed: list[dict[str, Any]] = []
    next_pending: list[dict[str, Any]] = []
    used_pending: set[int] = set()

    for detection in current:
        item = dict(detection)
        conf = float(item.get("vehicle_conf") or item.get("conf") or 0.0)
        source = str(item.get("detector_source") or "")

        # No hard 3-4% cut: real municipal-night vehicles can be weaker than
        # that. Strong detections appear immediately; lower-confidence boxes
        # need progressively more repeated spatial agreement.
        if conf < float(settings.vehicle_confirm_raw_min_confidence):
            continue
        if conf >= float(settings.vehicle_confirm_immediate_confidence):
            required_hits = 1
        elif conf >= float(settings.vehicle_confirm_two_hit_confidence):
            required_hits = 2
        elif conf >= float(settings.vehicle_confirm_three_hit_confidence):
            required_hits = 3
        elif conf >= float(settings.vehicle_confirm_four_hit_confidence):
            required_hits = 4
        else:
            # Ultra-weak full-frame candidates get one last chance only when
            # they repeat for five detector passes. Tile/CLAHE ultra-weak noise
            # is much more common, so drop it here.
            if "full" not in source:
                continue
            required_hits = 5

        box = item.get("bbox") or []
        best_idx = None
        best_score = -1.0
        if len(box) == 4:
            for idx, previous in enumerate(pending):
                if idx in used_pending:
                    continue
                pbox = previous.get("bbox") or []
                if len(pbox) != 4:
                    continue
                iou = _bbox_iou(box, pbox)
                center = _bbox_center_ratio(box, pbox)
                # The RTX detector runs several times a second, but a moving
                # vehicle can still shift far enough that IoU alone is small.
                # Centre proximity reconnects that vehicle without letting an
                # unrelated road marking several box-widths away accumulate hits.
                if (
                    iou < float(settings.vehicle_confirm_match_min_iou)
                    and center > float(settings.vehicle_confirm_match_max_center_ratio)
                ):
                    continue
                score = iou * 3.5 + max(0.0, 1.0 - center)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        hits = 1
        was_confirmed = False
        if best_idx is not None:
            used_pending.add(best_idx)
            previous = pending[best_idx]
            hits = int(previous.get("_confirm_hits") or 1) + 1
            was_confirmed = bool(previous.get("_confirmed"))
        if was_confirmed:
            hits = max(hits, required_hits)

        item["_confirm_hits"] = hits
        item["_confirmed"] = bool(hits >= required_hits)
        item["_provisional"] = not item["_confirmed"]
        if item["_confirmed"]:
            confirmed.append(item)
        next_pending.append(item)

    # Keep only the strongest recent hypotheses. This prevents persistent road
    # texture from growing an unbounded confirmation list.
    next_pending.sort(
        key=lambda value: (
            bool(value.get("_confirmed")),
            float(value.get("vehicle_conf") or value.get("conf") or 0.0),
        ),
        reverse=True,
    )
    return confirmed, next_pending[:60]


_STAGE_STATE_KEYS = (
    "raw_stage",
    "stage",
    "stage_valid",
    "stage_rejected_low_confidence",
    "stage_min_confidence",
    "stage_policy",
    "stage_source",
    "stage_model_label",
    "tire_detections",
    "stage_conf",
)


def _merge_stage_state(
    base: dict[str, Any],
    classified: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge delayed tire/body state while preserving fresh best.pt identity.

    A rejected/unfinished stage result cannot erase a valid LevX already attached
    to the same track. The next valid stage result may replace it normally.
    """
    item = dict(base)
    if classified is None:
        return item
    old_valid = item.get("stage") is not None and item.get("stage_valid") is not False
    new_valid = (
        classified.get("stage") is not None
        and classified.get("stage_valid") is not False
    )
    if old_valid and not new_valid:
        item["stage_stale"] = True
        return item
    for key in _STAGE_STATE_KEYS:
        if key in classified:
            item[key] = classified[key]
    if classified.get("stage_conf") is not None or classified.get("conf") is not None:
        stage_conf = float(
            classified.get("stage_conf")
            if classified.get("stage_conf") is not None
            else (classified.get("conf") or 0.0)
        )
        item["stage_conf"] = round(stage_conf, 4)
    item["stage_stale"] = False
    return item


def _associate_detections(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    next_track_id: int,
) -> tuple[list[dict[str, Any]], int]:
    """Globally reconnect fresh detector boxes to previous vehicle tracks.

    The old greedy loop could attach an early detection to the wrong car and
    leave the real match unmatched. We score all possible pairs first and then
    take the best non-conflicting matches.
    """
    if not current:
        max_missed = 0 if ai_uses_cuda() else max(0, min(1, int(settings.stage_tracking_max_missed_ai)))
        carried: list[dict[str, Any]] = []
        for prev in previous:
            bbox = prev.get("bbox") or []
            missed = int(prev.get("_missed_ai") or 0) + 1
            if len(bbox) != 4 or missed > max_missed:
                continue
            item = dict(prev)
            item["_missed_ai"] = missed
            item["conf"] = round(max(0.0, float(item.get("conf") or 0.0) * 0.90), 3)
            carried.append(item)
        return carried, next_track_id

    bbox_alpha = max(0.50, min(1.0, float(settings.stage_tracking_bbox_alpha)))
    min_iou = max(0.0, min(1.0, float(settings.stage_tracking_min_iou)))
    max_center = max(0.1, float(settings.stage_tracking_max_center_ratio))

    candidates: list[tuple[float, int, int]] = []
    for prev_idx, prev in enumerate(previous):
        prev_box = prev.get("bbox") or []
        if len(prev_box) != 4:
            continue
        for cur_idx, cur in enumerate(current):
            cur_box = cur.get("bbox") or []
            if len(cur_box) != 4:
                continue
            iou = _bbox_iou(prev_box, cur_box)
            center_ratio = _bbox_center_ratio(prev_box, cur_box)
            if iou < min_iou and center_ratio > max_center:
                continue
            # IoU dominates. Centre proximity lets fast vehicles reconnect.
            score = iou * 4.0 + max(0.0, 1.0 - center_ratio)
            candidates.append((score, prev_idx, cur_idx))

    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_prev: set[int] = set()
    matched_cur: dict[int, int] = {}
    for _score, prev_idx, cur_idx in candidates:
        if prev_idx in matched_prev or cur_idx in matched_cur:
            continue
        matched_prev.add(prev_idx)
        matched_cur[cur_idx] = prev_idx

    associated: list[dict[str, Any]] = []
    for cur_idx, detection in enumerate(current):
        item = dict(detection)
        bbox = [float(v) for v in (item.get("bbox") or [])]
        prev_idx = matched_cur.get(cur_idx)
        if prev_idx is not None:
            prev = previous[prev_idx]
            prev_box = [float(v) for v in (prev.get("bbox") or bbox)]
            if len(prev_box) == 4 and len(bbox) == 4:
                # Fresh detector geometry should win quickly; excessive old-box
                # blending was a major source of visible lag.
                item["bbox"] = [
                    round(bbox_alpha * new + (1.0 - bbox_alpha) * old, 1)
                    for old, new in zip(prev_box, bbox)
                ]
            item["track_id"] = prev.get("track_id")
            item["_missed_ai"] = 0
            item["_flow_failures"] = 0
            # The fast best.pt pass intentionally has no flood stage yet.
            # Keep the last classified stage on the matched vehicle until the
            # slower tire/body result for this same frame arrives.
            if (
                (item.get("stage") is None or item.get("stage_valid") is False)
                and prev.get("stage") is not None
                and prev.get("stage_valid") is not False
            ):
                item = _merge_stage_state(item, prev)
                item["stage_stale"] = True
            if len(prev_box) == 4 and len(bbox) == 4:
                old_cx = (prev_box[0] + prev_box[2]) / 2.0
                old_cy = (prev_box[1] + prev_box[3]) / 2.0
                new_cx = (bbox[0] + bbox[2]) / 2.0
                new_cy = (bbox[1] + bbox[3]) / 2.0
                previous_velocity = prev.get("_velocity") or [0.0, 0.0]
                vx = 0.65 * (new_cx - old_cx) + 0.35 * float(previous_velocity[0])
                vy = 0.65 * (new_cy - old_cy) + 0.35 * float(previous_velocity[1])
                item["_velocity"] = [round(vx, 2), round(vy, 2)]
        else:
            item["track_id"] = next_track_id
            item["_missed_ai"] = 0
            item["_flow_failures"] = 0
            item["_velocity"] = [0.0, 0.0]
            next_track_id += 1
        associated.append(item)

    # Keep a recently missed track for a very short time. This prevents boxes
    # from blinking off when best.pt misses one frame, but expires stale tracks
    # quickly after the car leaves the scene. Previous boxes must already be
    # projected to the current AI frame before this function is called.
    # One missed detector cycle is enough to prevent blinking. Carrying five
    # cycles kept a box on the road for several seconds after its vehicle had
    # moved away and looked like tracking lag.
    max_missed = 0 if ai_uses_cuda() else max(0, min(1, int(settings.stage_tracking_max_missed_ai)))
    for prev_idx, prev in enumerate(previous):
        if prev_idx in matched_prev:
            continue
        missed = int(prev.get("_missed_ai") or 0) + 1
        bbox = prev.get("bbox") or []
        if missed > max_missed or len(bbox) != 4:
            continue
        carry = dict(prev)
        carry["_missed_ai"] = missed
        carry["conf"] = round(max(0.0, float(carry.get("conf") or 0.0) * 0.90), 3)
        associated.append(carry)

    # A carried one-cycle track can overlap a fresh detector box if matching
    # was ambiguous. Remove only near-identical geometry; nearby real vehicles
    # remain separate. Fresh detections (_missed_ai == 0) always win.
    deduped: list[dict[str, Any]] = []
    for item in sorted(
        associated,
        key=lambda d: (int(d.get("_missed_ai") or 0), -float(d.get("conf") or 0.0)),
    ):
        box = item.get("bbox") or []
        if len(box) != 4:
            continue
        duplicate = False
        for kept in deduped:
            kept_box = kept.get("bbox") or []
            if len(kept_box) != 4:
                continue
            iou = _bbox_iou(box, kept_box)
            x1, y1, x2, y2 = [float(v) for v in box]
            kx1, ky1, kx2, ky2 = [float(v) for v in kept_box]
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_a = max(1.0, (x2 - x1) * (y2 - y1))
            area_b = max(1.0, (kx2 - kx1) * (ky2 - ky1))
            overlap_smaller = inter / min(area_a, area_b)
            if iou >= 0.45 or overlap_smaller >= 0.72:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)

    return deduped, next_track_id


def _template_track_bbox(
    previous_gray,
    current_gray,
    bbox: list[float],
    *,
    min_score: float = 0.52,
) -> tuple[list[float] | None, float]:
    """Template-match fallback for vehicles with too few LK feature points.

    Small/distant cars often contain too few stable corners for Lucas-Kanade.
    Searching a limited neighbourhood for the previous vehicle crop is much
    safer than freezing the box on the road until the next YOLO result.
    """
    height, width = current_gray.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(6.0, x2 - x1), max(6.0, y2 - y1)
    tx1 = max(0, min(width - 2, int(round(x1))))
    ty1 = max(0, min(height - 2, int(round(y1))))
    tx2 = max(tx1 + 2, min(width, int(round(x2))))
    ty2 = max(ty1 + 2, min(height, int(round(y2))))
    template = previous_gray[ty1:ty2, tx1:tx2]
    if template.size == 0 or template.shape[0] < 8 or template.shape[1] < 8:
        return None, 0.0

    pad_x = int(max(12.0, bw * 0.70))
    pad_y = int(max(10.0, bh * 0.70))
    sx1 = max(0, tx1 - pad_x)
    sy1 = max(0, ty1 - pad_y)
    sx2 = min(width, tx2 + pad_x)
    sy2 = min(height, ty2 + pad_y)
    search = current_gray[sy1:sy2, sx1:sx2]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None, 0.0

    # Remove part of the lighting sensitivity while retaining vehicle texture.
    template_eq = cv2.equalizeHist(template)
    search_eq = cv2.equalizeHist(search)
    result = cv2.matchTemplate(search_eq, template_eq, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    score = float(max_val)
    if not np.isfinite(score) or score < min_score:
        return None, score if np.isfinite(score) else 0.0

    nx1 = float(sx1 + max_loc[0])
    ny1 = float(sy1 + max_loc[1])
    candidate = [nx1, ny1, nx1 + (tx2 - tx1), ny1 + (ty2 - ty1)]

    # Reject an implausibly large one-frame jump even if a repetitive road
    # texture happens to correlate with the vehicle crop.
    old_cx, old_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    new_cx, new_cy = (candidate[0] + candidate[2]) / 2.0, (candidate[1] + candidate[3]) / 2.0
    max_motion = max(0.20, min(0.90, float(settings.stage_tracking_max_motion_ratio)))
    if abs(new_cx - old_cx) > max(10.0, bw * max_motion):
        return None, score
    if abs(new_cy - old_cy) > max(10.0, bh * max_motion):
        return None, score
    return _clip_bbox(candidate, width, height), score


def _propagate_detections_robust(
    previous_gray,
    current_gray,
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Follow vehicle boxes between detector frames with LK + template fallback.

    Fresh YOLO geometry remains authoritative. Between YOLO results, robust
    forward/backward LK motion is preferred; small vehicles with too few
    corners fall back to local template matching instead of leaving the box
    frozen on the road.
    """
    if (
        not bool(settings.stage_tracking_enabled)
        or previous_gray is None
        or current_gray is None
        or previous_gray.shape != current_gray.shape
        or not detections
    ):
        return _copy_detections(detections)

    height, width = current_gray.shape[:2]
    updated: list[dict[str, Any]] = []
    max_scale_change = max(0.02, min(0.25, float(settings.stage_tracking_max_scale_change)))
    max_motion_ratio = max(0.15, min(0.90, float(settings.stage_tracking_max_motion_ratio)))
    min_inlier_ratio = max(0.25, min(0.90, float(settings.stage_tracking_min_inlier_ratio)))

    for detection in detections:
        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_width = max(6.0, x2 - x1)
        box_height = max(6.0, y2 - y1)
        ix1 = max(0, min(width - 2, int(np.floor(x1))))
        iy1 = max(0, min(height - 2, int(np.floor(y1))))
        ix2 = max(ix1 + 2, min(width, int(np.ceil(x2))))
        iy2 = max(iy1 + 2, min(height, int(np.ceil(y2))))

        item = dict(detection)
        old_box = _clip_bbox([x1, y1, x2, y2], width, height)

        def use_template_or_old() -> None:
            template_box, score = _template_track_bbox(
                previous_gray,
                current_gray,
                old_box,
                min_score=max(0.25, min(0.85, float(settings.stage_tracking_template_min_score))),
            )
            previous_velocity = item.get("_velocity") or [0.0, 0.0]
            if template_box is not None:
                old_cx = (old_box[0] + old_box[2]) / 2.0
                old_cy = (old_box[1] + old_box[3]) / 2.0
                new_cx = (template_box[0] + template_box[2]) / 2.0
                new_cy = (template_box[1] + template_box[3]) / 2.0
                vx = 0.70 * (new_cx - old_cx) + 0.30 * float(previous_velocity[0])
                vy = 0.70 * (new_cy - old_cy) + 0.30 * float(previous_velocity[1])
                item["bbox"] = template_box
                item["_velocity"] = [round(vx, 2), round(vy, 2)]
                item["tracking_method"] = "template"
            else:
                # Do not freeze a missed vehicle on the road. Extrapolate a
                # short distance from the last confirmed motion and decay it.
                vx = float(previous_velocity[0]) * 0.82
                vy = float(previous_velocity[1]) * 0.82
                predicted = [
                    old_box[0] + vx,
                    old_box[1] + vy,
                    old_box[2] + vx,
                    old_box[3] + vy,
                ]
                item["bbox"] = _clip_bbox(predicted, width, height)
                item["_velocity"] = [round(vx, 2), round(vy, 2)]
                item["tracking_method"] = "velocity"
            item["template_score"] = round(float(score), 3)

        # Keep most of the vehicle interior. The old 18% inset frequently left
        # too little texture on distant vehicles, so LK could not move the box.
        mask = np.zeros_like(previous_gray, dtype=np.uint8)
        inset_x = max(2, int((ix2 - ix1) * 0.10))
        inset_y = max(2, int((iy2 - iy1) * 0.08))
        bottom_exclude = max(
            0,
            int((iy2 - iy1) * max(0.0, min(0.35, float(settings.stage_tracking_lk_bottom_exclude_ratio)))),
        )
        mx1, my1 = ix1 + inset_x, iy1 + inset_y
        mx2, my2 = ix2 - inset_x, iy2 - inset_y - bottom_exclude
        if mx2 <= mx1 + 2 or my2 <= my1 + 2:
            use_template_or_old()
            updated.append(item)
            continue
        cv2.rectangle(mask, (mx1, my1), (mx2, my2), 255, -1)

        old_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=60,
            qualityLevel=0.008,
            minDistance=4,
            mask=mask,
            blockSize=5,
        )
        if old_points is None or len(old_points) < 4:
            use_template_or_old()
            updated.append(item)
            continue

        new_points, status_f, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, old_points, None,
            winSize=(27, 27), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.012),
        )
        if new_points is None or status_f is None:
            use_template_or_old()
            updated.append(item)
            continue

        back_points, status_b, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, new_points, None,
            winSize=(27, 27), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.012),
        )
        valid = status_f.reshape(-1) == 1
        if back_points is not None and status_b is not None:
            fb_error = np.linalg.norm(
                back_points.reshape(-1, 2) - old_points.reshape(-1, 2), axis=1
            )
            valid &= status_b.reshape(-1) == 1
            valid &= fb_error <= 1.8

        old_valid = old_points.reshape(-1, 2)[valid]
        new_valid = new_points.reshape(-1, 2)[valid]
        if len(old_valid) < 4:
            use_template_or_old()
            updated.append(item)
            continue

        movement = new_valid - old_valid
        median_motion = np.median(movement, axis=0)
        residual = np.linalg.norm(movement - median_motion, axis=1)
        mad = float(np.median(residual))
        coherent = residual <= max(1.8, 3.0 * max(0.5, mad))
        old_valid = old_valid[coherent]
        new_valid = new_valid[coherent]
        if len(old_valid) < 4:
            use_template_or_old()
            updated.append(item)
            continue

        matrix, inliers = cv2.estimateAffinePartial2D(
            old_valid, new_valid, method=cv2.RANSAC,
            ransacReprojThreshold=2.4, maxIters=160, confidence=0.985,
        )
        inlier_ratio = 0.0
        if inliers is not None and len(inliers):
            inlier_ratio = float(np.mean(inliers.reshape(-1) > 0))

        dx, dy = np.median(new_valid - old_valid, axis=0).tolist()
        scale = 1.0
        if matrix is not None and inlier_ratio >= min_inlier_ratio:
            a, b = float(matrix[0, 0]), float(matrix[0, 1])
            scale = float(np.sqrt(a * a + b * b))
            aff_dx, aff_dy = float(matrix[0, 2]), float(matrix[1, 2])
            if abs(aff_dx - dx) <= box_width * 0.28:
                dx = 0.65 * dx + 0.35 * aff_dx
            if abs(aff_dy - dy) <= box_height * 0.28:
                dy = 0.65 * dy + 0.35 * aff_dy
        elif mad > max(3.2, min(box_width, box_height) * 0.12):
            use_template_or_old()
            updated.append(item)
            continue

        max_dx = max(8.0, box_width * max_motion_ratio)
        max_dy = max(8.0, box_height * max_motion_ratio)
        dx = max(-max_dx, min(max_dx, float(dx)))
        dy = max(-max_dy, min(max_dy, float(dy)))
        scale = max(1.0 - max_scale_change, min(1.0 + max_scale_change, scale))

        old_cx, old_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        new_w, new_h = box_width * scale, box_height * scale
        candidate = [
            old_cx + dx - new_w / 2.0,
            old_cy + dy - new_h / 2.0,
            old_cx + dx + new_w / 2.0,
            old_cy + dy + new_h / 2.0,
        ]
        candidate = _clip_bbox(candidate, width, height)

        # Cross-check LK with local template matching. If LK has weak inliers
        # and the template has a strong match, prefer the template; otherwise
        # blend only when both trackers agree spatially.
        template_box, template_score = _template_track_bbox(
            previous_gray,
            current_gray,
            old_box,
            min_score=max(0.25, min(0.85, float(settings.stage_tracking_template_min_score))),
        )
        chosen = candidate
        tracking_method = "lk"
        if template_box is not None:
            lk_center = np.array([(candidate[0] + candidate[2]) / 2.0, (candidate[1] + candidate[3]) / 2.0])
            tm_center = np.array([(template_box[0] + template_box[2]) / 2.0, (template_box[1] + template_box[3]) / 2.0])
            agreement = float(np.linalg.norm(lk_center - tm_center))
            if inlier_ratio < 0.50 and float(template_score) >= 0.55:
                chosen = template_box
                tracking_method = "template-preferred"
            elif agreement <= max(5.0, min(box_width, box_height) * 0.22):
                chosen = [
                    round(0.70 * float(a) + 0.30 * float(b), 2)
                    for a, b in zip(candidate, template_box)
                ]
                tracking_method = "lk+template"

        chosen = _clip_bbox(chosen, width, height)
        new_cx = (chosen[0] + chosen[2]) / 2.0
        new_cy = (chosen[1] + chosen[3]) / 2.0
        previous_velocity = item.get("_velocity") or [0.0, 0.0]
        vx = 0.72 * (new_cx - old_cx) + 0.28 * float(previous_velocity[0])
        vy = 0.72 * (new_cy - old_cy) + 0.28 * float(previous_velocity[1])
        item["bbox"] = chosen
        item["_velocity"] = [round(vx, 2), round(vy, 2)]
        item["flow_inlier_ratio"] = round(inlier_ratio, 3)
        item["template_score"] = round(float(template_score), 3)
        item["tracking_method"] = tracking_method
        updated.append(item)

    return updated


def _propagate_detections(
    previous_gray,
    current_gray,
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate boxes between detector passes without letting them deform.

    V8.5.21 estimated both motion and scale from a small set of LK points.
    Road markings or shadows could make those points spread apart, causing a
    vehicle box to grow/shrink or jump. V8.5.22 keeps the detector's width and
    height fixed between AI passes and accepts only coherent median motion.
    Fresh YOLO geometry still replaces the tracked box every detector cycle.
    """
    if (
        not bool(settings.stage_tracking_enabled)
        or previous_gray is None
        or current_gray is None
        or previous_gray.shape != current_gray.shape
        or not detections
    ):
        return _copy_detections(detections)

    height, width = current_gray.shape[:2]
    updated: list[dict[str, Any]] = []

    for detection in detections:
        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_width = max(2.0, x2 - x1)
        box_height = max(2.0, y2 - y1)
        item = dict(detection)
        item["bbox"] = _clip_bbox([x1, y1, x2, y2], width, height)
        previous_velocity = item.get("_velocity") or [0.0, 0.0]
        tracked = False

        # Use vehicle interior features only. The lower edge is where road
        # paint and reflections most often pull LK away from the vehicle.
        inset_x = max(2, int(box_width * 0.12))
        inset_y = max(2, int(box_height * 0.10))
        bottom_exclude = max(
            2,
            int(
                box_height
                * max(
                    0.08,
                    min(0.32, float(settings.stage_tracking_lk_bottom_exclude_ratio)),
                )
            ),
        )
        rx1 = int(max(0, x1 + inset_x))
        ry1 = int(max(0, y1 + inset_y))
        rx2 = int(min(width, x2 - inset_x))
        ry2 = int(min(height, y2 - bottom_exclude))

        old_points = None
        if rx2 - rx1 >= 8 and ry2 - ry1 >= 8:
            roi = previous_gray[ry1:ry2, rx1:rx2]
            corners = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=28,
                qualityLevel=0.020,
                minDistance=3,
                blockSize=5,
            )
            if corners is not None and len(corners) >= 4:
                old_points = corners.reshape(-1, 2) + [rx1, ry1]

        if old_points is not None:
            old_points = old_points.astype(np.float32).reshape(-1, 1, 2)
            new_points, status, flow_error = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                old_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    14,
                    0.025,
                ),
            )

            if new_points is not None and status is not None:
                valid = status.reshape(-1) == 1
                if flow_error is not None and int(valid.sum()) >= 4:
                    errs = flow_error.reshape(-1)
                    valid_errs = errs[valid]
                    if valid_errs.size:
                        med_err = float(np.median(valid_errs))
                        valid &= errs <= max(10.0, med_err * 2.8 + 1.0)

                if int(valid.sum()) >= 4:
                    old_valid = old_points.reshape(-1, 2)[valid]
                    new_valid = new_points.reshape(-1, 2)[valid]
                    movement = new_valid - old_valid
                    median_motion = np.median(movement, axis=0)
                    residual = np.linalg.norm(movement - median_motion, axis=1)
                    mad = float(np.median(residual)) if residual.size else 0.0
                    coherent = residual <= max(1.4, 2.8 * max(0.45, mad))
                    coherent_count = int(coherent.sum())
                    if coherent_count >= 3:
                        dx, dy = np.median(movement[coherent], axis=0).tolist()
                        inlier_ratio = coherent_count / max(1, len(movement))
                        # One rendered frame should only move a modest fraction
                        # of the box. Larger jumps are usually background LK.
                        max_dx = max(4.0, box_width * 0.24)
                        max_dy = max(4.0, box_height * 0.24)
                        dx = max(-max_dx, min(max_dx, float(dx)))
                        dy = max(-max_dy, min(max_dy, float(dy)))
                        dx = 0.84 * dx + 0.16 * float(previous_velocity[0])
                        dy = 0.84 * dy + 0.16 * float(previous_velocity[1])

                        # Weak coherence with a large jump is safer to ignore.
                        motion_ratio = max(
                            abs(dx) / max(1.0, box_width),
                            abs(dy) / max(1.0, box_height),
                        )
                        vehicle_conf = float(
                            item.get("vehicle_conf") or item.get("conf") or 0.0
                        )
                        required_inlier = 0.60 if vehicle_conf < 0.45 else 0.45
                        safe_tiny_motion = motion_ratio <= 0.06 and vehicle_conf >= 0.45
                        if inlier_ratio >= required_inlier or safe_tiny_motion:
                            item["bbox"] = _clip_bbox(
                                [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
                                width,
                                height,
                            )
                            item["_velocity"] = [round(float(dx), 2), round(float(dy), 2)]
                            item["_flow_failures"] = 0
                            item["flow_inlier_ratio"] = round(float(inlier_ratio), 3)
                            item["tracking_method"] = "lk-translate"
                            tracked = True

        if not tracked:
            # Never keep extrapolating indefinitely. At most two render frames
            # use a quickly decaying velocity, then the box waits for fresh YOLO
            # geometry instead of drifting across the road.
            failures = int(item.get("_flow_failures") or 0) + 1
            decay = 0.45 if failures == 1 else (0.18 if failures == 2 else 0.0)
            vx = float(previous_velocity[0]) * decay
            vy = float(previous_velocity[1]) * decay
            max_dx = max(3.0, box_width * 0.12)
            max_dy = max(3.0, box_height * 0.12)
            vx = max(-max_dx, min(max_dx, vx))
            vy = max(-max_dy, min(max_dy, vy))
            item["bbox"] = _clip_bbox(
                [x1 + vx, y1 + vy, x2 + vx, y2 + vy],
                width,
                height,
            )
            item["_velocity"] = [round(vx, 2), round(vy, 2)]
            item["_flow_failures"] = failures
            item["tracking_method"] = "hold" if decay == 0.0 else "velocity-decay"

        updated.append(item)

    return updated


def _draw_live_detections(
    frame,
    detections: list[dict[str, Any]],
    inference_ms: int | None,
    stage_override: int | None = None,
    stage_votes: dict[str, int] | None = None,
    *,
    draw_boxes: bool = True,
):
    """Draw CCTV status; per-vehicle boxes may be left to browser vector overlay."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    best_stage = 0
    occupied_labels: list[tuple[int, int, int, int]] = []

    def overlaps(rect: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        for ax1, ay1, ax2, ay2 in occupied_labels:
            if min(x2, ax2) > max(x1, ax1) and min(y2, ay2) > max(y1, ay1):
                return True
        return False

    # Small/distant vehicles are drawn first so their labels get a fair chance
    # before a large foreground box consumes the same label area.
    ordered = sorted(
        detections,
        key=lambda item: (
            ((item.get("bbox") or [0, 0, 0, 0])[2] - (item.get("bbox") or [0, 0, 0, 0])[0])
            * ((item.get("bbox") or [0, 0, 0, 0])[3] - (item.get("bbox") or [0, 0, 0, 0])[1])
            if len(item.get("bbox") or []) == 4 else 0
        ),
    )

    for detection in (ordered if draw_boxes else []):
        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [
            int(round(value))
            for value in _clip_bbox(
                [float(value) for value in bbox],
                width,
                height,
            )
        ]
        stage_value = detection.get("stage")
        stage_valid = stage_value is not None and detection.get("stage_valid") is not False
        confidence = float(
            detection.get("stage_conf")
            if (stage_valid or detection.get("stage_rejected_low_confidence"))
            and detection.get("stage_conf") is not None
            else (detection.get("vehicle_conf") or detection.get("conf") or 0.0)
        )
        if stage_valid:
            stage = max(0, min(4, int(stage_value)))
            best_stage = max(best_stage, stage)
            color = _STAGE_COLORS.get(stage, (255, 255, 255))
        else:
            stage = None
            color = (190, 190, 190)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        track_id = detection.get("track_id")
        track_text = f" #{track_id}" if track_id is not None else ""
        source = str(detection.get("stage_source") or "")
        source_text = " TIRE" if source == "tire" else (" BODY" if source == "car_body" else "")
        vehicle_name = str(
            detection.get("vehicle_label")
            or detection.get("source_label")
            or "vehicle"
        ).replace("_", " ").strip().upper()
        if stage_valid:
            label = f"{vehicle_name}{track_text} Lev{stage} {confidence * 100:.0f}%{source_text}"
        elif detection.get("stage_rejected_low_confidence"):
            label = f"{vehicle_name}{track_text} HOLD {confidence * 100:.0f}%"
        elif detection.get("_provisional"):
            label = f"{vehicle_name}{track_text} CHECK {confidence * 100:.0f}%"
        else:
            label = f"{vehicle_name}{track_text} DET {confidence * 100:.0f}%"

        font_scale = 0.46 if width <= 720 else 0.50
        thickness = 1 if width <= 720 else 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_h = text_height + baseline + 7
        label_w = min(width, text_width + 10)
        label_x = max(0, min(width - label_w, x1))

        # Try above, then below, then progressively higher slots. This avoids
        # the stacked green text blocks seen when several cars are close.
        candidate_tops = [
            max(0, y1 - label_h),
            min(max(0, height - label_h), y2 + 2),
        ]
        for step in range(1, 5):
            candidate_tops.append(max(0, y1 - label_h - step * (label_h + 2)))
        label_top = candidate_tops[0]
        for top in candidate_tops:
            rect = (label_x, int(top), label_x + label_w, int(top) + label_h)
            if not overlaps(rect):
                label_top = int(top)
                break
        rect = (label_x, label_top, label_x + label_w, label_top + label_h)
        occupied_labels.append(rect)

        cv2.rectangle(
            annotated,
            (rect[0], rect[1]),
            (min(width - 1, rect[2]), min(height - 1, rect[3])),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (label_x + 5, min(height - 2, label_top + text_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (8, 14, 20),
            thickness,
            cv2.LINE_AA,
        )

    if not draw_boxes:
        for detection in detections:
            stage_value = detection.get("stage")
            if stage_value is not None and detection.get("stage_valid") is not False:
                best_stage = max(best_stage, max(0, min(4, int(stage_value))))

    if stage_override is not None:
        best_stage = max(0, min(4, int(stage_override)))

    latency_text = f" | AI {inference_ms} ms" if inference_ms is not None else ""
    vote_text = ""
    if stage_votes:
        vote_text = " | vote " + "/".join(
            str(int(stage_votes.get(f"Lev{level}", 0)))
            for level in range(5)
        )
    banner = (
        f"AI VEHICLE FLOOD Lev{best_stage} | "
        f"vehicles {len(detections)}{vote_text}{latency_text}"
    )
    banner_color = _STAGE_COLORS.get(best_stage, (255, 255, 255))
    cv2.rectangle(annotated, (0, 0), (width, 36), (8, 16, 26), -1)
    cv2.putText(
        annotated,
        banner,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        banner_color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def _update_haar_privacy_confirmations(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    previous_gray=None,
    current_gray=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Require repeated Haar detections before mosaic is displayed.

    A wall/sign false positive is commonly present for only one Haar pass.
    Real faces/plates tend to reappear close to the same location. This small
    temporal gate substantially reduces accidental wall mosaics without
    changing dedicated privacy-YOLO behaviour.
    """
    threshold = max(0.05, min(0.80, float(settings.anonymizer_haar_confirm_iou)))
    required = max(3, min(5, int(settings.anonymizer_haar_confirm_hits)))
    require_motion = bool(settings.anonymizer_haar_require_motion)
    motion_threshold = max(0.0, float(settings.anonymizer_haar_motion_mean_threshold))
    used_prev: set[int] = set()
    state: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []

    for detection in current:
        item = dict(detection)
        bbox = item.get("bbox") or []
        label = str(item.get("label") or "")
        motion_score = None
        if (
            require_motion
            and previous_gray is not None
            and current_gray is not None
            and getattr(previous_gray, "shape", None) == getattr(current_gray, "shape", None)
            and len(bbox) == 4
        ):
            h, w = current_gray.shape[:2]
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                prev_roi = previous_gray[y1:y2, x1:x2]
                curr_roi = current_gray[y1:y2, x1:x2]
                if prev_roi.size and curr_roi.size:
                    motion_score = float(cv2.absdiff(prev_roi, curr_roi).mean())
        item["_haar_motion"] = round(float(motion_score or 0.0), 2)
        best_idx = None
        best_iou = 0.0
        if len(bbox) == 4:
            for idx, prev in enumerate(previous):
                if idx in used_prev or str(prev.get("label") or "") != label:
                    continue
                prev_bbox = prev.get("bbox") or []
                if len(prev_bbox) != 4:
                    continue
                iou = _bbox_iou(prev_bbox, bbox)
                if iou >= threshold and iou > best_iou:
                    best_idx, best_iou = idx, iou
        hits = 1
        if best_idx is not None:
            used_prev.add(best_idx)
            hits = int(previous[best_idx].get("_haar_hits") or 1) + 1
        item["_haar_hits"] = min(required + 1, hits)
        state.append(item)
        motion_ok = (
            not require_motion
            or previous_gray is None
            or current_gray is None
            or float(motion_score or 0.0) >= motion_threshold
        )
        if hits >= required and motion_ok:
            confirmed.append(dict(item))
    return state, confirmed


def _filter_privacy_detections_for_scene(
    privacy_detections: list[dict[str, Any]],
    vehicle_detections: list[dict[str, Any]],
    frame_shape,
    backend: str,
) -> list[dict[str, Any]]:
    """Reject privacy boxes that are implausible in the current scene.

    Haar license-plate false positives on walls/signs were the main source of
    large accidental mosaics. A plate centre must sit inside a currently
    tracked vehicle box (slightly expanded), and its size must be plausible
    relative to that vehicle. Faces remain independent because pedestrians are
    valid privacy targets.
    """
    if not privacy_detections:
        return []

    frame_h, frame_w = frame_shape[:2]
    vehicles = []
    for vehicle in vehicle_detections:
        bbox = vehicle.get("bbox") or []
        if len(bbox) != 4:
            continue
        vx1, vy1, vx2, vy2 = [float(v) for v in bbox]
        vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
        pad_x, pad_y = vw * 0.18, vh * 0.18
        vehicles.append((
            max(0.0, vx1 - pad_x),
            max(0.0, vy1 - pad_y),
            min(float(frame_w), vx2 + pad_x),
            min(float(frame_h), vy2 + pad_y),
            vw, vh,
        ))

    filtered: list[dict[str, Any]] = []
    for detection in privacy_detections:
        kind = str(detection.get("label") or "").strip().lower()
        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if kind == "license_plate" and bool(settings.anonymizer_plate_require_vehicle):
            matched = False
            for vx1, vy1, vx2, vy2, vw, vh in vehicles:
                if not (vx1 <= cx <= vx2 and vy1 <= cy <= vy2):
                    continue
                # A plate should be much smaller than the vehicle body.
                if bw > vw * 0.55 or bh > vh * 0.38:
                    continue
                if (bw * bh) > (vw * vh) * 0.18:
                    continue
                matched = True
                break
            if not matched:
                continue

        if kind == "face":
            # Additional CCTV-scale guard after box expansion.
            if bw > frame_w * 0.20 or bh > frame_h * 0.27:
                continue

        filtered.append(dict(detection))

    return filtered


class CameraWorker:
    """
    하나의 CCTV 주소당 하나만 생성되는 공유 워커입니다.

    - grab thread: 최신 CCTV 프레임 수집
    - AI thread: 설정된 간격으로 YOLO 추론
    - render thread: 프레임을 계속 송출하고 박스는 optical flow로 이동
    """

    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        stream_scheme = urlparse(
            str(_resolved_stream_source(stream_url) or "")
        ).scheme.lower()
        self.network_stream = stream_scheme in {"http", "https", "rtsp", "rtmp"}
        self.http_stream = stream_scheme in {"http", "https"}

        self.stop_event = threading.Event()
        self.frame_condition = threading.Condition()
        self.jpeg_condition = threading.Condition()
        self.raw_jpeg_condition = threading.Condition()
        self.result_lock = threading.Lock()
        # Canonical geometry + stage state only; never held during model.predict().
        self.track_state_lock = threading.Lock()
        self.client_lock = threading.Lock()

        self.latest_frame = None
        self.latest_frame_seq = 0
        self.latest_frame_at = 0.0
        self.gray_history = deque(maxlen=max(12, int(settings.stage_tracking_history_frames)))
        self.last_stream_error: str | None = None
        self.last_stream_ok_at = 0.0
        self.reconnect_count = 0
        self.fallback_frame_count = 0
        self.hls_stall_resets = 0

        self.latest_ai_packet: dict[str, Any] | None = None
        self.latest_ai_version = 0
        self.latest_stage_packet: dict[str, Any] | None = None
        self.latest_stage_packet_version = 0
        # Vehicle geometry and flood-stage classification are deliberately
        # decoupled.  The detector continuously replaces this single-slot job;
        # the slower classifier consumes only the newest available frame.
        self.stage_condition = threading.Condition()
        self.latest_stage_job: dict[str, Any] | None = None
        self.latest_stage_job_version = 0
        self.latest_result: dict[str, Any] = {
            "stage": None,
            "label": "분석 대기",
            "conf": 0,
            "snapshot": None,
            "detections": [],
            "inference_ms": None,
        }
        self.latest_result_at = 0.0
        self.next_track_id = 1
        self.last_associated_detections: list[dict[str, Any]] = []
        self.last_ai_frame_seq = -1
        self.last_ai_gray = None
        self.last_rescue_at = 0.0
        self.pending_vehicle_detections: list[dict[str, Any]] = []
        self.last_ai_error_log_at = 0.0
        self.last_zero_box_log_at = 0.0
        self.last_box_diag_log_at = 0.0
        self.last_transport_diag_log_at = 0.0
        self.last_stage_error_log_at = 0.0
        # V8.5.37: once a browser subscribes to annotated CCTV, keep the detector
        # producer enabled for the lifetime of this worker.  V8.5.37 relied only
        # on a short renewable lease; the user log proved the WebSocket lease was
        # alive while the AI producer never published a first result.
        self.ai_activation = threading.Event()
        self.ai_thread: threading.Thread | None = None
        self.last_ai_loop_heartbeat_at = 0.0
        self.last_ai_attempt_at = 0.0
        self.last_ai_success_at = 0.0
        self.last_ai_state = "created"

        self.latest_privacy_packet: dict[str, Any] | None = None
        self.latest_privacy_version = 0
        self.haar_privacy_state: list[dict[str, Any]] = []
        self.haar_privacy_gray = None
        # Haar fallback scans are CPU-heavy. Running them every 0.55 seconds
        # starved JPEG rendering on CPU-only systems; privacy remains active,
        # but at a rate that leaves capture/tracking responsive.
        self.privacy_interval = max(
            2.0,
            min(3.0, float(settings.anonymizer_refresh_seconds) * 1.5),
        )

        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.latest_raw_jpeg: bytes | None = None
        self.latest_raw_jpeg_seq = 0
        self.latest_raw_frame_at = 0.0
        self.latest_annotated_frame_at = 0.0

        self.clients = 0
        self.annotated_clients = 0
        self.last_annotated_disconnect_at = 0.0
        # Short JPEG snapshot requests use a renewable lease instead of an
        # endless multipart connection. This keeps AI/rendering alive for every
        # visible CCTV without hitting Chromium's long-connection limits.
        self.annotated_interest_until = 0.0
        # The focused CCTV gets a faster detector cadence. Other open windows
        # still receive boxes, but at a slower geometry cadence so five or six
        # simultaneous YOLO jobs do not starve capture/JPEG rendering.
        self.focused_interest_until = 0.0
        self.raw_clients = 0
        # Snapshot requests are intentionally short-lived. Keep a small raw
        # interest lease so the renderer continues producing fresh raw JPEGs
        # between requests instead of stopping at exactly the wrong moment.
        self.raw_interest_until = 0.0
        self.last_client_at = time.monotonic()
        self.started = False
        self.threads: list[threading.Thread] = []

        # Capture/render and AI are separate threads.  On the RTX GPU there is no
        # reason to artificially throttle visible video to 9 FPS or best.pt to
        # one pass every ~0.55 s: those caps were visible in the user's recording
        # as "buffering" even when the source stream itself was healthy.
        if ai_uses_cuda():
            self.ai_interval = max(
                0.18,
                min(0.45, float(settings.stage_box_detection_interval_seconds)),
            )
            self.stage_interval = max(
                0.80,
                min(1.40, float(settings.stage_stream_interval_seconds)),
            )
            # 960px AI frames stay available to best.pt, but browser JPEG output
            # is encoded at a lower cadence/size below. This removes the last
            # visible stutter caused by 4-6 simultaneous 960px JPEG encoders.
            self.render_fps = 12.0
            self.raw_render_interval = 1.0 / 10.0
        else:
            self.ai_interval = max(
                0.55,
                min(1.20, float(settings.stage_box_detection_interval_seconds)),
            )
            self.stage_interval = max(
                1.60,
                min(3.0, float(settings.stage_stream_interval_seconds)),
            )
            self.render_fps = 8.0
            self.raw_render_interval = 0.18
        self.render_interval = 1.0 / self.render_fps
        self.last_scheduled_ai_interval = self.ai_interval
        self.last_ai_cadence_mode = "initial"
        # Stop a closed CCTV quickly, but not so quickly that a single missed
        # poll kills it. `/api/cctv/frame-annotated` is polled by the browser;
        # a backgrounded/minimised tab is throttled by Chrome/Firefox to as
        # infrequently as once a minute, and a brief server-side hiccup (e.g.
        # several YOLO passes queued at once) can also delay one poll past a
        # very short window. Either case used to trip an 8s idle timeout,
        # tear down all four worker threads, and make a long-open CCTV window
        # go blank and restart from "CCTV CONNECTING" (문제4). 25s still frees
        # a genuinely closed window quickly while tolerating both.
        self.idle_timeout = 25.0

        # HLS는 하나의 세그먼트에 들어 있는 여러 프레임을 OpenCV가 한꺼번에
        # 빠르게 디코딩한 뒤 다음 세그먼트를 기다리는 경우가 있습니다.
        # 소스 FPS에 맞춰 프레임 수집 속도를 제한해 정지 후 빨라지는 현상을 막습니다.
        self.source_fps = 25.0
        self.capture_interval = 1.0 / self.source_fps

        # 출력 해상도를 제한해 JPEG와 optical flow 부하를 줄입니다.
        configured_width = max(
            480,
            int(settings.stage_max_width),
            int(settings.stage_tracking_frame_width),
        )
        # Only the focused feed runs tracking. Keep enough source pixels for
        # small vehicles there; raw windows are downscaled before JPEG output.
        self.output_width = max(480, min(960, configured_width))
        requested_imgsz = int(settings.stage_inference_imgsz)
        self.inference_imgsz = max(
            320,
            min(512, requested_imgsz),
        )
        # Vehicle geometry and flood-stage classification need different input
        # sizes. V8.5.20 accidentally fed the small 416px stage-classifier size
        # into best.pt as well, even though VEHICLE_DETECTION_IMGSZ=960 was
        # configured. That caused distant cars to disappear or receive only a
        # couple of boxes in wide municipal CCTV views. Keep the fast stage
        # classifier at its smaller size, but let the dedicated vehicle detector
        # use the resolution explicitly configured for box geometry.
        self.vehicle_detector_imgsz = max(
            512,
            min(960, int(settings.vehicle_detection_imgsz)),
        )
        self.jpeg_quality = max(
            60,
            min(64, int(settings.jpeg_quality)),
        )

    def start(self) -> None:
        if self.started:
            return
        self.started = True

        startup_jpeg = _status_jpeg(
            "CCTV CONNECTING",
            _stream_label(self.stream_url),
        )
        if startup_jpeg is not None:
            with self.jpeg_condition:
                self.latest_jpeg = startup_jpeg
                self.latest_jpeg_seq += 1
                self.jpeg_condition.notify_all()
            with self.raw_jpeg_condition:
                self.latest_raw_jpeg = startup_jpeg
                self.latest_raw_jpeg_seq += 1
                self.raw_jpeg_condition.notify_all()

        self.threads = [
            threading.Thread(
                target=self._grab_loop,
                name="cctv-grab",
                daemon=True,
            ),
            threading.Thread(
                target=self._ai_loop,
                name="cctv-box-detector",
                daemon=True,
            ),
            threading.Thread(
                target=self._stage_loop,
                name="cctv-stage-classifier",
                daemon=True,
            ),
            threading.Thread(
                target=self._privacy_loop,
                name="cctv-privacy",
                daemon=True,
            ),
            threading.Thread(
                target=self._render_loop,
                name="cctv-render",
                daemon=True,
            ),
        ]
        for thread in self.threads:
            if thread.name == "cctv-box-detector":
                self.ai_thread = thread
            thread.start()

    def is_alive(self) -> bool:
        if not self.started or self.stop_event.is_set():
            return False
        # A Python thread can terminate on an unexpected decoder/OpenCV error
        # without setting stop_event. Treat loss of the critical capture or
        # render thread as a dead worker so the next HTTP request replaces it
        # instead of reusing a permanently frozen object.
        if self.threads:
            critical = [
                thread for thread in self.threads
                if thread.name in {"cctv-grab", "cctv-render", "cctv-box-detector"}
            ]
            if critical and not all(thread.is_alive() for thread in critical):
                return False
        return True

    def stop(self) -> None:
        """Cancel capture/inference promptly after the final client closes."""
        self.stop_event.set()
        with self.frame_condition:
            self.frame_condition.notify_all()
        with self.stage_condition:
            self.stage_condition.notify_all()
        with self.jpeg_condition:
            self.jpeg_condition.notify_all()
        with self.raw_jpeg_condition:
            self.raw_jpeg_condition.notify_all()

    def add_client(self, *, annotated: bool = True) -> None:
        reset_tracking = False
        now = time.monotonic()
        with self.client_lock:
            if (
                annotated
                and self.annotated_clients == 0
                and self.last_annotated_disconnect_at
                and now - self.last_annotated_disconnect_at > 1.5
            ):
                reset_tracking = True
            self.clients += 1
            if annotated:
                self.annotated_clients += 1
            else:
                self.raw_clients += 1
                self.raw_interest_until = max(
                    self.raw_interest_until,
                    now + max(0.5, float(settings.cctv_raw_interest_seconds)),
                )
            self.last_client_at = now
        if reset_tracking:
            with self.frame_condition:
                self.gray_history.clear()
            with self.track_state_lock:
                self.last_associated_detections = []
            self.last_ai_gray = None
            self.last_ai_frame_seq = -1
        if annotated:
            self.ai_activation.set()
            set_live_inference_priority(True)

    def remove_client(self, *, annotated: bool = True) -> None:
        now = time.monotonic()
        with self.client_lock:
            self.clients = max(0, self.clients - 1)
            if annotated:
                self.annotated_clients = max(0, self.annotated_clients - 1)
                if self.annotated_clients == 0:
                    # Keep track geometry across a brief browser MJPEG recycle.
                    # It is reset on the next add only if the gap exceeds 1.5 s.
                    self.last_annotated_disconnect_at = now
            else:
                self.raw_clients = max(0, self.raw_clients - 1)
            self.last_client_at = now


    def renew_annotated_interest(
        self, seconds: float = 1.6, *, focused: bool = False
    ) -> None:
        """Keep annotated AI output alive for short snapshot clients."""
        now = time.monotonic()
        with self.client_lock:
            lease_until = now + max(0.8, min(3.0, float(seconds)))
            self.annotated_interest_until = max(
                self.annotated_interest_until, lease_until
            )
            if focused:
                self.focused_interest_until = max(
                    self.focused_interest_until, lease_until
                )
            self.last_client_at = now
        # Sticky activation: a tiny timing gap in WebSocket lease renewal must
        # never turn the detector producer back off while this worker is alive.
        self.ai_activation.set()
        set_live_inference_priority(True)

    def renew_raw_interest(self, seconds: float = 1.8) -> None:
        """Keep lightweight raw JPEG production alive for WebSocket fallback."""
        now = time.monotonic()
        with self.client_lock:
            self.raw_interest_until = max(
                self.raw_interest_until,
                now + max(0.8, min(3.0, float(seconds))),
            )
            self.last_client_at = now

    def has_focus_interest(self) -> bool:
        with self.client_lock:
            return time.monotonic() < self.focused_interest_until

    def has_annotated_clients(self) -> bool:
        with self.client_lock:
            return (
                self.annotated_clients > 0
                or time.monotonic() < self.annotated_interest_until
            )

    def has_raw_clients(self) -> bool:
        with self.client_lock:
            return self.raw_clients > 0 or time.monotonic() < self.raw_interest_until

    def _is_idle(self) -> bool:
        with self.client_lock:
            now = time.monotonic()
            return (
                self.clients == 0
                and now >= self.annotated_interest_until
                and now >= self.raw_interest_until
                and now - self.last_client_at > self.idle_timeout
            )

    def _resize_for_output(self, frame):
        height, width = frame.shape[:2]
        if width <= self.output_width:
            return frame
        scale = self.output_width / float(width)
        return cv2.resize(
            frame,
            (
                self.output_width,
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    def _open_capture(self):
        source = _resolved_stream_source(self.stream_url)
        capture = _open_video_capture(source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Start the bundled flood-validation clip at a known flooded scene,
        # matching the background sampler. Public CCTV streams are untouched.
        if capture.isOpened() and _is_test_stream(self.stream_url):
            try:
                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    max(0.0, float(settings.test_cctv_sample_seconds)) * 1000.0,
                )
            except Exception:
                pass

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(source_fps) or not 5.0 <= source_fps <= 60.0:
            source_fps = 25.0

        self.source_fps = source_fps
        self.capture_interval = 1.0 / source_fps
        return capture

    def _publish_frame(self, frame) -> None:
        frame = self._resize_for_output(frame)
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.has_annotated_clients()
            else None
        )
        with self.frame_condition:
            self.latest_frame = frame
            self.latest_frame_seq += 1
            self.latest_frame_at = time.monotonic()
            if gray is not None:
                self.gray_history.append((self.latest_frame_seq, gray))
            self.last_stream_ok_at = self.latest_frame_at
            self.last_stream_error = None
            self.frame_condition.notify_all()

    def _gray_frames_between(self, start_seq: int, end_seq: int) -> list[tuple[int, Any]]:
        """Return stored gray frames after start_seq through end_seq."""
        with self.frame_condition:
            return [
                (int(seq), gray.copy())
                for seq, gray in self.gray_history
                if int(start_seq) < int(seq) <= int(end_seq)
            ]

    def _publish_stream_error(self, error: Exception | str) -> None:
        message = str(error or "영상 수신 실패")
        self.last_stream_error = message

        # Once a real frame has been shown, never overwrite it with a reconnect
        # card. Stateless browser snapshots keep displaying the last good JPEG
        # while the HLS prefetcher repairs the source in the background. This is
        # what prevents long-running CCTV windows from suddenly disappearing.
        if self.last_stream_ok_at:
            return

        jpeg = _status_jpeg(
            "CCTV STREAM RETRY",
            f"{_stream_label(self.stream_url)} | {message}",
        )
        if jpeg is None:
            return
        with self.jpeg_condition:
            self.latest_jpeg = jpeg
            self.latest_jpeg_seq += 1
            self.jpeg_condition.notify_all()
        # Non-focused windows use /api/stream-raw. Mirror the diagnostic frame
        # there as well so a failed public source never remains an opaque black
        # rectangle while the annotated window shows a reconnect message.
        with self.raw_jpeg_condition:
            self.latest_raw_jpeg = jpeg
            self.latest_raw_jpeg_seq += 1
            self.raw_jpeg_condition.notify_all()

    def _grab_loop(self) -> None:
        capture = None
        capture_is_hls_segment = False
        fallback_temp_path: str | None = None
        last_fallback_segment: str | None = None
        fallback_media_playlist_url: str | None = None
        hls_has_succeeded = False
        # HTTP CCTV feeds advertised by Pohang are HLS in normal operation.
        # Start with the header-aware requests/HLS path instead of first paying
        # an OpenCV/FFmpeg open timeout that was responsible for the 3-5 second
        # black/connecting interval in the supplied recording. If the quick HLS
        # probe fails once, fall back to a bounded direct VideoCapture attempt.
        hls_first_attempt = bool(self.http_stream)
        hls_fallback_until = float("inf") if hls_first_attempt else 0.0
        next_fallback_at = 0.0
        last_hls_progress_at = 0.0
        last_hls_reset_at = 0.0
        hls_watch_started_at = time.monotonic()
        hls_prefetcher: _HlsSegmentPrefetcher | None = None

        def close_capture() -> None:
            nonlocal capture, capture_is_hls_segment, fallback_temp_path
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            capture = None
            capture_is_hls_segment = False
            if fallback_temp_path:
                try:
                    os.remove(fallback_temp_path)
                except OSError:
                    pass
                fallback_temp_path = None

        try:
            while not self.stop_event.is_set():
                if self._is_idle():
                    self.stop_event.set()
                    break

                now = time.monotonic()
                if capture is None or not capture.isOpened():
                    close_capture()

                    if self.network_stream and now < hls_fallback_until:
                        # Keep network I/O one segment ahead of decoding. The old
                        # download-after-EOF loop paused at every HLS boundary.
                        if hls_prefetcher is None:
                            hls_prefetcher = _HlsSegmentPrefetcher(
                                self.stream_url, self.stop_event
                            )
                            hls_prefetcher.start()
                            hls_watch_started_at = time.monotonic()

                        item = hls_prefetcher.get(timeout=0.18)
                        if item is None:
                            now_wait = time.monotonic()
                            progress_base = (
                                hls_prefetcher.last_progress_at
                                or last_hls_progress_at
                                or hls_watch_started_at
                            )
                            stalled_for = max(0.0, now_wait - progress_base)
                            reset_after = max(2.5, float(settings.cctv_hls_stall_reset_seconds))
                            hard_after = max(
                                reset_after + 2.0,
                                float(settings.cctv_hls_hard_reconnect_seconds),
                            )
                            if stalled_for >= reset_after and now_wait - last_hls_reset_at >= 1.5:
                                old_prefetcher = hls_prefetcher
                                hls_prefetcher = _HlsSegmentPrefetcher(
                                    self.stream_url, self.stop_event
                                )
                                hls_prefetcher.start()
                                old_prefetcher.close()
                                last_hls_reset_at = now_wait
                                self.hls_stall_resets += 1
                                hls_watch_started_at = now_wait
                            # An HTTP URL is usually HLS, but a malformed or
                            # token-expired playlist should not leave the CCTV
                            # window waiting 8+ seconds before trying FFmpeg.
                            # Once the quick HLS path has produced an actual
                            # error for ~1.6 s, immediately give direct capture
                            # one bounded chance.
                            if (
                                not hls_has_succeeded
                                and hls_prefetcher.last_error
                                and stalled_for >= 1.6
                            ):
                                self._publish_stream_error(hls_prefetcher.last_error)
                                old_prefetcher = hls_prefetcher
                                hls_prefetcher = None
                                old_prefetcher.close()
                                hls_first_attempt = False
                                hls_fallback_until = 0.0
                                continue
                            if stalled_for >= hard_after and not hls_has_succeeded:
                                self._publish_stream_error(
                                    hls_prefetcher.last_error
                                    or f"HLS live edge stalled {stalled_for:.1f}s"
                                )
                                old_prefetcher = hls_prefetcher
                                hls_prefetcher = None
                                old_prefetcher.close()
                                hls_first_attempt = False
                                hls_fallback_until = 0.0
                            continue

                        segment_url, temp_path = item
                        capture = _open_video_capture(temp_path)
                        if not capture.isOpened():
                            try:
                                os.remove(temp_path)
                            except OSError:
                                pass
                            self._publish_stream_error(
                                "HLS 세그먼트를 디코딩하지 못했습니다."
                            )
                            continue
                        fallback_temp_path = temp_path
                        capture_is_hls_segment = True
                        last_fallback_segment = segment_url
                        last_hls_progress_at = time.monotonic()
                        hls_watch_started_at = last_hls_progress_at
                        self.fallback_frame_count += 1
                        hls_has_succeeded = True
                        hls_fallback_until = float("inf")
                        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                        if np.isfinite(source_fps) and 5.0 <= source_fps <= 60.0:
                            self.source_fps = source_fps
                            self.capture_interval = 1.0 / source_fps
                    else:
                        hls_watch_started_at = time.monotonic()
                        capture = self._open_capture()
                        self.reconnect_count += 1
                        capture_is_hls_segment = False
                        if not capture.isOpened() and self.network_stream:
                            # Direct FFmpeg access often lacks the Referer that
                            # the public CCTV origin now expects. The requests
                            # HLS path below carries browser-like headers.
                            hls_fallback_until = time.monotonic() + 30.0
                            next_fallback_at = 0.0
                            close_capture()
                            continue

                cycle_started = time.monotonic()
                ok, frame = capture.read() if capture is not None and capture.isOpened() else (False, None)

                if ok and frame is not None:
                    self._publish_frame(frame)
                    elapsed = time.monotonic() - cycle_started
                    wait_seconds = max(0.0, self.capture_interval - elapsed)
                    # A blocking direct network read is already source-paced. A
                    # downloaded HLS segment is a local file and must always be
                    # paced, otherwise it plays as a burst then freezes.
                    if (
                        self.network_stream
                        and not capture_is_hls_segment
                        and elapsed >= self.capture_interval * 0.85
                    ):
                        wait_seconds = 0.0
                    if wait_seconds > 0:
                        self.stop_event.wait(wait_seconds)
                    continue

                was_segment = capture_is_hls_segment
                close_capture()
                if self.network_stream:
                    if was_segment:
                        # Normal end-of-segment: fetch the next manifest entry
                        # immediately without classifying it as a disconnect.
                        next_fallback_at = time.monotonic() + 0.03
                    else:
                        hls_fallback_until = (
                            float("inf") if hls_has_succeeded
                            else time.monotonic() + 30.0
                        )
                        next_fallback_at = 0.0
                    continue

                # Local files loop from their configured start point.
                self.stop_event.wait(0.05)
        except Exception as exc:
            self._publish_stream_error(exc)
        finally:
            if hls_prefetcher is not None:
                hls_prefetcher.close()
            close_capture()
            self.stop_event.set()
            with self.frame_condition:
                self.frame_condition.notify_all()
            with self.jpeg_condition:
                self.jpeg_condition.notify_all()
            with self.raw_jpeg_condition:
                self.raw_jpeg_condition.notify_all()

    def _get_latest_frame(
        self,
        last_seq: int = -1,
        timeout: float = 0.5,
    ) -> tuple[int, Any | None]:
        with self.frame_condition:
            if (
                self.latest_frame is None
                or self.latest_frame_seq == last_seq
            ):
                self.frame_condition.wait(timeout=timeout)

            if self.latest_frame is None:
                return self.latest_frame_seq, None

            return self.latest_frame_seq, self.latest_frame.copy()

    def _ai_loop(self) -> None:
        # Stagger cameras slightly so five newly opened windows do not all enqueue
        # best.pt on the same millisecond.
        jitter = (sum(self.stream_url.encode("utf-8")) % 120) / 1000.0
        next_inference_at = time.monotonic() + jitter
        last_inferred_seq = -1

        self.last_ai_state = "thread_started"
        self.last_ai_loop_heartbeat_at = time.monotonic()
        logger.warning("CCTV AI62 thread-start [%s]", _stream_label(self.stream_url))
        while not self.stop_event.is_set():
            self.last_ai_loop_heartbeat_at = time.monotonic()
            if self._is_idle():
                self.last_ai_state = "idle_stop"
                self.stop_event.set()
                break

            # V8.5.37 uses a sticky activation event rather than the short lease
            # itself as the AI gate. The worker still dies after idle_timeout, so
            # raw-only/background workers do not become permanent GPU consumers.
            if not self.ai_activation.is_set():
                self.last_ai_state = "waiting_subscription"
                self.ai_activation.wait(0.08)
                continue

            now = time.monotonic()
            if now < next_inference_at:
                time.sleep(min(0.02, next_inference_at - now))
                continue

            self.last_ai_state = "waiting_frame"
            frame_seq, frame = self._get_latest_frame(
                last_seq=last_inferred_seq,
                timeout=0.25,
            )
            self.last_ai_loop_heartbeat_at = time.monotonic()
            if frame is None or frame_seq == last_inferred_seq:
                continue

            inference_cycle_at = time.monotonic()
            self.last_ai_attempt_at = inference_cycle_at
            self.last_ai_state = "scheduler_wait"
            started = time.perf_counter()
            fast_detections: list[dict[str, Any]] = []
            try:
                current_ai_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                with self.track_state_lock:
                    projected_previous = _copy_detections(self.last_associated_detections)
                    tracked_count = len(self.last_associated_detections)
                # CPU needs short optical-flow interpolation between slower
                # best.pt passes.  On GPU the detector runs several times per
                # second; carrying LK flow across road markings was the main
                # cause of stretched/shifted boxes in the supplied recording.
                if (
                    not ai_uses_cuda()
                    and projected_previous
                    and self.last_ai_gray is not None
                    and self.last_ai_frame_seq >= 0
                ):
                    prev_gray = self.last_ai_gray
                    history = self._gray_frames_between(self.last_ai_frame_seq, frame_seq)
                    for _seq, history_gray in history:
                        if not projected_previous:
                            break
                        projected_previous = _propagate_detections(
                            prev_gray, history_gray, projected_previous
                        )
                        prev_gray = history_gray

                # Phase 1: publish one lightweight best.pt pass immediately.
                # This is the supplied Claude build's useful behaviour, but the
                # vehicle class is deliberately NOT misused as a flood level.
                detector_started = time.perf_counter()
                rescue_target = max(
                    1, int(settings.vehicle_detection_rescue_min_count)
                )
                # Do not disable the tiled rescue merely because one or two
                # vehicles are already being tracked. That V8.5.20 condition
                # made small/distant vehicles effectively impossible to add
                # after the first detector pass. Throttle the expensive rescue
                # instead, and keep using it while the tracked count is below
                # the configured target.
                allow_rescue = bool(
                    self.last_ai_success_at > 0.0
                    and tracked_count < rescue_target
                    and time.monotonic() - self.last_rescue_at
                    >= max(1.0, float(settings.vehicle_detection_rescue_interval_seconds))
                )
                if allow_rescue:
                    self.last_rescue_at = time.monotonic()
                # live_priority=True is intentionally unconditional here: this
                # flag distinguishes an interactive CameraWorker (any open
                # browser window) from a true background scan job, not "this
                # particular window is focused". Making it track focus made
                # every non-focused-but-open window spin-wait forever behind
                # _LIVE_INFERENCE_PRIORITY, since that flag stays set as long
                # as any window is open. See vehicle_flood_pipeline._predict.
                raw_detected_boxes = detect_vehicle_boxes(
                    frame,
                    vehicle_imgsz=self.vehicle_detector_imgsz,
                    live_priority=True,
                    allow_rescue=allow_rescue,
                )
                # V8.5.32 diagnostic/production rule: best.pt geometry is
                # published immediately and can never be hidden by tire/body
                # classification or by a temporal-confirmation gate. Very weak
                # boxes are marked provisional and are not sent to stage models,
                # but they are still visible so a real detector output can be
                # distinguished from a downstream pipeline problem.
                stage_confirmed_boxes, next_pending = _confirm_vehicle_candidates(
                    raw_detected_boxes,
                    self.pending_vehicle_detections,
                )
                self.pending_vehicle_detections = next_pending

                # V8.5.37 separates *box visibility* from *stage eligibility*.
                # V8.5.34 used the temporal confirmer as a hard
                # visibility gate. On moving night-CCTV cars the raw best.pt box
                # can legitimately shift enough between 0.24 s passes that a
                # low-confidence candidate never accumulates 2-5 hits, leaving
                # the UI at vehicles:0 even though best.pt is returning boxes.
                #
                # On CUDA, publish current-frame best.pt geometry immediately
                # after the detector's strong class-agnostic de-duplication. A
                # dynamic relative floor removes the near-zero tail without
                # bringing back V8.5.32's CHECK-box storm. Temporal confirmation
                # is retained only to decide whether tire/body classification is
                # allowed for a weak candidate.
                if ai_uses_cuda() and raw_detected_boxes:
                    top_conf = max(
                        float(item.get("vehicle_conf") or item.get("conf") or 0.0)
                        for item in raw_detected_boxes
                    )
                    visible_floor = max(
                        float(settings.vehicle_visible_min_confidence),
                        min(
                            float(settings.vehicle_visible_max_floor),
                            top_conf * float(settings.vehicle_visible_relative_floor),
                        ),
                    )
                    confirmed_geometry = [
                        item for item in stage_confirmed_boxes
                        if len(item.get("bbox") or []) == 4
                    ]
                    detected_boxes = []
                    for raw_item in raw_detected_boxes:
                        conf = float(raw_item.get("vehicle_conf") or raw_item.get("conf") or 0.0)
                        if conf < visible_floor:
                            continue
                        item = dict(raw_item)
                        box = item.get("bbox") or []
                        stage_eligible = (
                            conf >= float(settings.vehicle_stage_immediate_confidence)
                        )
                        if len(box) == 4 and not stage_eligible:
                            for confirmed_item in confirmed_geometry:
                                cbox = confirmed_item.get("bbox") or []
                                if len(cbox) != 4:
                                    continue
                                if (
                                    _bbox_iou(box, cbox)
                                    >= float(settings.vehicle_stage_confirm_iou)
                                    or _bbox_center_ratio(box, cbox)
                                    <= float(settings.vehicle_stage_confirm_center_ratio)
                                ):
                                    stage_eligible = True
                                    break
                        item["_stage_eligible"] = bool(stage_eligible)
                        item["_confirmed"] = bool(stage_eligible)
                        item["_provisional"] = not bool(stage_eligible)
                        detected_boxes.append(item)
                        if len(detected_boxes) >= 12:
                            break
                    # If every candidate fell just below the relative floor, show
                    # the single strongest geometrically valid raw box instead of
                    # reverting to vehicles:0. It remains CHECK-only until the
                    # temporal confirmer accepts it and is never sent to stage AI.
                    if not detected_boxes and top_conf >= 0.001:
                        item = dict(raw_detected_boxes[0])
                        item["_stage_eligible"] = False
                        item["_confirmed"] = False
                        item["_provisional"] = True
                        detected_boxes = [item]
                else:
                    detected_boxes = _copy_detections(stage_confirmed_boxes)
                    visible_floor = 0.0
                    for item in detected_boxes:
                        item["_stage_eligible"] = True
                        item["_provisional"] = False
                        item["_confirmed"] = True

                if not raw_detected_boxes:
                    zero_now = time.monotonic()
                    if zero_now - self.last_zero_box_log_at >= 5.0:
                        logger.warning(
                            "CCTV best.pt DIRECT raw boxes=0 [%s] · frame=%sx%s · mean=%.1f · imgsz=%s · device=%s",
                            _stream_label(self.stream_url),
                            frame.shape[1], frame.shape[0], float(frame.mean()),
                            self.vehicle_detector_imgsz,
                            "cuda" if ai_uses_cuda() else "cpu",
                        )
                        self.last_zero_box_log_at = zero_now
                elif not stage_confirmed_boxes:
                    diag_now = time.monotonic()
                    if diag_now - self.last_box_diag_log_at >= 5.0:
                        top_conf = max(
                            float(item.get("vehicle_conf") or item.get("conf") or 0.0)
                            for item in raw_detected_boxes
                        )
                        # Use WARNING deliberately. An earlier diagnostic used logger.info(),
                        # but this application never lowers the root logger from
                        # Python's default WARNING level, so the diagnostic the
                        # user was asked to paste could not actually appear.
                        logger.warning(
                            "CCTV best.pt raw=%s visible=%s confirmed_for_stage=0 pending=%s top_conf=%.4f floor=%.4f [%s]",
                            len(raw_detected_boxes),
                            len(detected_boxes),
                            len(self.pending_vehicle_detections),
                            top_conf,
                            float(visible_floor),
                            _stream_label(self.stream_url),
                        )
                        self.last_box_diag_log_at = diag_now
                detector_ms = round(
                    (time.perf_counter() - detector_started) * 1000
                )

                # V8.5.37 always emits one compact detector telemetry line every
                # five seconds while AI is active. Earlier diagnostics fired only
                # for raw=0 or confirmed=0, so a healthy detector + broken renderer
                # produced no clue at all. This line makes the next failure
                # unambiguous without flooding PowerShell.
                telemetry_now = time.monotonic()
                if telemetry_now - self.last_box_diag_log_at >= 5.0:
                    telemetry_top = max(
                        [float(item.get("vehicle_conf") or item.get("conf") or 0.0)
                         for item in raw_detected_boxes] or [0.0]
                    )
                    scheduler = inference_scheduler_status()
                    logger.warning(
                        "CCTV BOX62 raw=%s visible=%s stage=%s top=%.4f floor=%.4f "
                        "ms=%s cadence=%s/%.3f sched_last=%s/%s qv=%s qs=%s batch=%s [%s]",
                        len(raw_detected_boxes),
                        len(detected_boxes),
                        len(stage_confirmed_boxes),
                        telemetry_top,
                        float(visible_floor),
                        detector_ms,
                        self.last_ai_cadence_mode,
                        float(self.last_scheduled_ai_interval),
                        scheduler.get("last_kind"),
                        scheduler.get("last_duration_ms"),
                        scheduler.get("vehicle_queue"),
                        scheduler.get("stage_queue"),
                        scheduler.get("last_batch_size"),
                        _stream_label(self.stream_url),
                    )
                    self.last_box_diag_log_at = telemetry_now

                # Never throw away a successful best.pt result merely because
                # capture advanced while CUDA was working. V8.5.27 did exactly
                # that, and with several cameras the detector could remain
                # permanently "stale", leaving every banner at vehicles 0.
                # Instead, translate the fresh detector boxes through a bounded
                # set of captured gray frames and publish them on the newest
                # picture. Box width/height remain fixed; only coherent motion is
                # applied, so this avoids the old stretched-box failure mode.
                source_frame_seq = int(frame_seq)
                source_gray = current_ai_gray
                with self.frame_condition:
                    newest_seq = int(self.latest_frame_seq)
                    newest_frame = (
                        self.latest_frame.copy()
                        if self.latest_frame is not None else None
                    )
                frames_behind = max(0, newest_seq - source_frame_seq)
                detector_lag_seconds = frames_behind / max(1.0, float(self.source_fps))

                if (
                    not ai_uses_cuda()
                    and frames_behind > 0
                    and newest_frame is not None
                    and detector_lag_seconds <= 0.90
                ):
                    history = self._gray_frames_between(source_frame_seq, newest_seq)
                    # At most six LK steps keeps CPU cost bounded even when the
                    # detector waited behind another camera. Always include the
                    # newest gray frame so geometry lands on the displayed frame.
                    if len(history) > 6:
                        indexes = np.linspace(0, len(history) - 1, 6, dtype=int)
                        history = [history[int(index)] for index in indexes]
                    projected_current = _copy_detections(detected_boxes)
                    projected_pending = _copy_detections(self.pending_vehicle_detections)
                    prev_gray = source_gray
                    for _seq, history_gray in history:
                        if projected_current:
                            projected_current = _propagate_detections(
                                prev_gray, history_gray, projected_current
                            )
                        if projected_pending:
                            projected_pending = _propagate_detections(
                                prev_gray, history_gray, projected_pending
                            )
                        prev_gray = history_gray
                    if history:
                        detected_boxes = projected_current
                        self.pending_vehicle_detections = projected_pending
                        frame_seq = newest_seq
                        frame = newest_frame
                        current_ai_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                fast_detections, self.next_track_id = _associate_detections(
                    projected_previous,
                    detected_boxes,
                    self.next_track_id,
                )
                with self.track_state_lock:
                    # Stage may finish while best.pt is running. Merge the newest
                    # canonical Lev state back onto fresh geometry before publish,
                    # so a late AI write cannot erase a just-finished stage update.
                    canonical_by_track = {
                        int(item["track_id"]): item
                        for item in self.last_associated_detections
                        if item.get("track_id") is not None
                    }
                    merged_fast: list[dict[str, Any]] = []
                    for item in fast_detections:
                        track_id = item.get("track_id")
                        canonical = (
                            canonical_by_track.get(int(track_id))
                            if track_id is not None
                            else None
                        )
                        merged_fast.append(_merge_stage_state(item, canonical))
                    self.last_associated_detections = _copy_detections(merged_fast)
                    fast_detections = _copy_detections(merged_fast)
                self.last_ai_frame_seq = int(frame_seq)
                self.last_ai_gray = current_ai_gray

                with self.result_lock:
                    previous_stage = self.latest_result.get("stage")
                    previous_votes = dict(self.latest_result.get("stage_votes") or {})
                    self.latest_ai_version += 1
                    self.latest_ai_packet = {
                        "version": self.latest_ai_version,
                        "frame_seq": frame_seq,
                        "frame": frame,
                        "at": time.monotonic(),
                        "detections": _copy_detections(fast_detections),
                        "geometry_update": True,
                        "stage": previous_stage,
                        "stage_votes": previous_votes,
                        "inference_ms": detector_ms,
                    }
                    self.latest_result = {
                        **self.latest_result,
                        "label": (
                            self.latest_result.get("label")
                            if previous_stage is not None
                            else "차량 박스 표시 · 침수단계 분석 중"
                        ),
                        "detections": _copy_detections(fast_detections),
                        "detector_ms": detector_ms,
                        "raw_detection_count": len(raw_detected_boxes),
                        "visible_detection_count": len(fast_detections),
                        "stage_confirmed_count": len(stage_confirmed_boxes),
                        "visible_confidence_floor": round(float(visible_floor), 4),
                        "pending": True,
                    }
                    # Geometry is already a valid live AI result even while
                    # tire/body classification is pending. Keep diagnostics current.
                    self.latest_result_at = time.monotonic()
                    self.last_ai_success_at = self.latest_result_at
                    self.last_ai_state = "published"

                # Classification receives a latest-only job using the same
                # current-frame boxes already displayed to the user.  This keeps
                # best -> vehicle crop -> tire_level -> body fallback aligned to
                # the visible frame and preserves track IDs into stage results.
                stage_candidates = [
                    item for item in fast_detections
                    if int(item.get("_missed_ai") or 0) == 0
                    and bool(item.get("_stage_eligible", not item.get("_provisional")))
                ]
                stage_candidates.sort(
                    key=lambda item: float(item.get("vehicle_conf") or item.get("conf") or 0.0),
                    reverse=True,
                )
                stage_cap = max(
                    1,
                    int(settings.stage_max_vehicles_per_cycle)
                    if ai_uses_cuda()
                    else min(3, int(settings.stage_max_vehicles_per_cycle)),
                )
                stage_candidates = stage_candidates[:stage_cap]
                if stage_candidates:
                    with self.stage_condition:
                        self.latest_stage_job_version += 1
                        self.latest_stage_job = {
                            "version": self.latest_stage_job_version,
                            "frame_seq": int(frame_seq),
                            "frame": frame,
                            "vehicle_candidates": _copy_detections(stage_candidates),
                            "fast_detections": _copy_detections(stage_candidates),
                            "detector_ms": detector_ms,
                            "queued_at": time.monotonic(),
                            "detector_lag_seconds": round(detector_lag_seconds, 3),
                        }
                        self.stage_condition.notify_all()

            except Exception as exc:
                error_now = time.monotonic()
                if error_now - self.last_ai_error_log_at >= 5.0:
                    logger.exception(
                        "CCTV vehicle detector failed [%s]",
                        _stream_label(self.stream_url),
                    )
                    self.last_ai_error_log_at = error_now
                self.last_ai_state = f"error:{type(exc).__name__}"
                with self.result_lock:
                    previous_stage = self.latest_result.get("stage")
                    previous_conf = self.latest_result.get("conf") or 0
                    self.latest_result = {
                        "stage": previous_stage,
                        "label": (
                            "차량 박스 정상 · 단계 판정 실패"
                            if fast_detections
                            else "차량 탐지 실패"
                        ),
                        "error": str(exc),
                        "conf": previous_conf,
                        "snapshot": None,
                        "detections": _copy_detections(fast_detections),
                        "pipeline": "fast vehicle boxes -> tire/body stage fallback",
                        "inference_ms": round((time.perf_counter() - started) * 1000),
                    }
                    self.latest_result_at = time.monotonic()

            last_inferred_seq = frame_seq
            # Start-to-start cadence remains V8.6.2 adaptive scheduling.
            # V8.6.4 adds only a short display-copy velocity correction; detector
            # cadence, central single-owner GPU scheduling, and tire/body fairness
            # remain authoritative and unchanged.
            current_ai_interval, cadence_mode, _cadence_scheduler = (
                _adaptive_cuda_ai_interval(
                    self.ai_interval,
                    focused=self.has_focus_interest(),
                )
            )
            self.last_scheduled_ai_interval = current_ai_interval
            self.last_ai_cadence_mode = cadence_mode
            next_inference_at = max(
                time.monotonic(), inference_cycle_at + current_ai_interval
            )

    def _stage_loop(self) -> None:
        """Classify flood stage without ever blocking current box geometry."""
        consumed_version = 0
        next_stage_at = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < next_stage_at:
                self.stop_event.wait(min(0.05, next_stage_at - now))
                continue
            with self.stage_condition:
                if (
                    self.latest_stage_job is None
                    or self.latest_stage_job_version == consumed_version
                ):
                    self.stage_condition.wait(timeout=0.25)
                job = self.latest_stage_job

            if job is None or int(job["version"]) == consumed_version:
                continue
            consumed_version = int(job["version"])
            # Start-to-start throttle. If classification itself takes longer
            # than the interval there is no extra delay; otherwise leave CPU/GPU
            # headroom for capture, rendering and the next geometry pass.
            focus_multiplier = 1.0 if self.has_focus_interest() else 1.25
            current_stage_interval = _load_scaled_interval(
                max(self.stage_interval, self.stage_interval * focus_multiplier)
            )
            next_stage_at = time.monotonic() + current_stage_interval
            started = time.perf_counter()

            try:
                # live_priority=True unconditionally — see the matching note in
                # _ai_loop. This call already reuses job["vehicle_candidates"]
                # from _ai_loop instead of re-running vehicle detection, so it
                # is one predict() call (tire) plus at most one more (body
                # fallback), not a second full geometry pass.
                detections, _representative = infer_vehicle_flood(
                    job["frame"],
                    vehicle_imgsz=self.inference_imgsz,
                    stage_floor=None,
                    vehicle_candidates=job["vehicle_candidates"],
                    live_priority=True,
                )
                # infer_vehicle_flood() receives the already-tracked best.pt
                # candidates and preserves each track_id through the vehicle crop
                # -> tire_level -> car_flood_cls fallback path.  Re-associating the
                # classification result here could attach a delayed stage to the
                # neighbouring car or invent a new ID.  Keep the original track IDs
                # so the render loop merges labels onto the exact vehicle geometry.
                if any(item.get("track_id") is None for item in detections):
                    detections, _unused_next_id = _associate_detections(
                        job["fast_detections"], detections, self.next_track_id
                    )
                final_stage, final_conf, consensus = _consensus_stage(
                    self.stream_url, detections
                )

                classified_by_track = {
                    int(item["track_id"]): item
                    for item in detections
                    if item.get("track_id") is not None
                }
                with self.track_state_lock:
                    merged_tracks: list[dict[str, Any]] = []
                    for tracked in self.last_associated_detections:
                        track_id = tracked.get("track_id")
                        classified = (
                            classified_by_track.get(int(track_id))
                            if track_id is not None
                            else None
                        )
                        merged_tracks.append(_merge_stage_state(tracked, classified))
                    self.last_associated_detections = merged_tracks
                    stable_detections = _copy_detections(self.last_associated_detections)

                elapsed = round((time.perf_counter() - started) * 1000)
                source_counts = {"tire": 0, "car_body": 0}
                for detection in detections:
                    source = str(detection.get("stage_source") or "")
                    if source in source_counts:
                        source_counts[source] += 1

                stage_result = {
                    "stage": final_stage,
                    "label": (
                        f"MODE Lev{final_stage}"
                        if final_stage is not None
                        else (
                            "FLOOD CANDIDATE · VERIFYING"
                            if (consensus.get("positive_confirmation") or {}).get("pending")
                            else "STAGE HOLD · CONFIDENCE <70%"
                        )
                    ),
                    "conf": final_conf,
                    "snapshot": None,
                    "detections": _copy_detections(stable_detections),
                    "stage_votes": consensus["votes"],
                    "stage_confidence_averages": consensus["confidence_averages"],
                    "stage_spatial": consensus["spatial_stage"],
                    "stage_ema": consensus["ema_stage"],
                    "stage_ema_alpha": consensus["ema_alpha"],
                    "positive_confirmed": bool(consensus.get("positive_confirmed")),
                    "positive_confirmation": consensus.get("positive_confirmation") or {},
                    "stage_policy": (
                        "test_minimum" if consensus["test_floor_applied"]
                        else "strict_vehicle_mode"
                    ),
                    "stage_source_counts": source_counts,
                    "pipeline": "independent boxes + latest-only tire/body stage",
                    "detector_ms": job["detector_ms"],
                    "pending": False,
                    "frame_width": int(job["frame"].shape[1]),
                    "frame_height": int(job["frame"].shape[0]),
                    "inference_ms": elapsed,
                }

                # No qualified vote means no stage change and no DB/cache
                # authority. Keep the last accepted stage while still showing
                # the current vehicle boxes and the rejected-confidence state.
                if final_stage is None:
                    confirmation_pending = bool(
                        (consensus.get("positive_confirmation") or {}).get("pending")
                    )
                    with self.result_lock:
                        self.latest_result = {
                            **self.latest_result,
                            "label": (
                                "침수 후보 재확인 중 · 단계 미반영"
                                if confirmation_pending
                                else "단계 유지 · 신뢰도 70% 미만"
                            ),
                            "detections": _copy_detections(stable_detections),
                            "pending": False,
                            "stage_rejected_low_confidence": not confirmation_pending,
                            "positive_confirmation": consensus.get("positive_confirmation") or {},
                            "stage_min_confidence": float(settings.stage_min_confidence),
                        }
                    continue

                # A stage packet carries labels only. Its old geometry can
                # never replace boxes already tracked on newer frames.
                with self.result_lock:
                    self.latest_stage_packet_version += 1
                    self.latest_stage_packet = {
                        "version": self.latest_stage_packet_version,
                        "frame_seq": job["frame_seq"],
                        "frame": job["frame"],
                        "at": time.monotonic(),
                        "detections": _copy_detections(stable_detections),
                        "geometry_update": False,
                        "stage": final_stage,
                        "stage_votes": dict(consensus["votes"]),
                        "inference_ms": elapsed,
                    }
                    self.latest_result = stage_result
                    self.latest_result_at = time.monotonic()

                with _analysis_cache_lock:
                    _analysis_cache[self.stream_url] = {
                        "at": time.monotonic(), "result": stage_result
                    }
            except Exception as exc:
                error_now = time.monotonic()
                if error_now - self.last_stage_error_log_at >= 5.0:
                    logger.exception(
                        "CCTV tire/body stage pipeline failed [%s]",
                        _stream_label(self.stream_url),
                    )
                    self.last_stage_error_log_at = error_now
                with self.result_lock:
                    self.latest_result = {
                        **self.latest_result,
                        "pending": False,
                        "error": str(exc),
                        "label": "차량 박스 정상 · 단계 판정 실패",
                    }

    def _privacy_loop(self) -> None:
        """Run face/license-plate detection independently from flood inference."""
        if not settings.anonymizer_enabled:
            return

        next_inference_at = 0.0
        last_inferred_seq = -1
        while not self.stop_event.is_set():
            if self._is_idle():
                self.stop_event.set()
                break

            if not self.has_annotated_clients():
                self.stop_event.wait(0.12)
                continue

            now = time.monotonic()
            if now < next_inference_at:
                self.stop_event.wait(min(0.03, next_inference_at - now))
                continue

            frame_seq, frame = self._get_latest_frame(
                last_seq=last_inferred_seq,
                timeout=0.25,
            )
            if frame is None or frame_seq == last_inferred_seq:
                continue

            try:
                result = detect_privacy(frame)
                backend = str(result.get("backend") or "unknown")
                raw_detections = _copy_detections(result.get("detections") or [])
                if backend == "opencv-haar":
                    current_haar_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.haar_privacy_state, detections = _update_haar_privacy_confirmations(
                        self.haar_privacy_state,
                        raw_detections,
                        self.haar_privacy_gray,
                        current_haar_gray,
                    )
                    self.haar_privacy_gray = current_haar_gray
                else:
                    self.haar_privacy_state = []
                    self.haar_privacy_gray = None
                    detections = raw_detections
                with self.result_lock:
                    self.latest_privacy_version += 1
                    self.latest_privacy_packet = {
                        "version": self.latest_privacy_version,
                        "frame_seq": frame_seq,
                        "frame": frame,
                        "at": time.monotonic(),
                        "detections": detections,
                        "inference_ms": int(result.get("inference_ms") or 0),
                        "backend": backend,
                    }
            except Exception:
                # Privacy failure must not stop the CCTV stream.
                pass

            last_inferred_seq = frame_seq
            next_inference_at = time.monotonic() + self.privacy_interval

    def _render_loop(self) -> None:
        last_frame_seq = -1
        applied_ai_version = -1
        applied_stage_version = -1
        previous_gray = None
        tracked_detections: list[dict[str, Any]] = []
        last_ai_at = 0.0
        last_geometry_at = 0.0
        last_velocity_projection_at = 0.0
        velocity_projection_seconds = 0.0
        inference_ms: int | None = None
        stage_override: int | None = None
        stage_votes: dict[str, int] = {}

        applied_privacy_version = -1
        privacy_previous_gray = None
        privacy_detections: list[dict[str, Any]] = []
        last_privacy_at = 0.0
        privacy_backend = "off"

        next_render_at = 0.0
        next_raw_render_at = 0.0

        while not self.stop_event.is_set():
            if self._is_idle():
                self.stop_event.set()
                break

            now = time.monotonic()
            if now < next_render_at:
                time.sleep(
                    min(0.01, next_render_at - now)
                )
                continue

            frame_seq, frame = self._get_latest_frame(
                last_seq=last_frame_seq,
                timeout=0.25,
            )
            if frame is None or frame_seq == last_frame_seq:
                continue

            # Raw subscribers receive a lightweight stream from this same
            # capture worker. No second HLS/OpenCV connection and no AI work is
            # created for background CCTV windows.
            if self.has_raw_clients() and now >= next_raw_render_at:
                raw_frame = frame
                if raw_frame.shape[1] > 640:
                    raw_scale = 640.0 / float(raw_frame.shape[1])
                    raw_frame = cv2.resize(
                        raw_frame,
                        (640, max(1, int(raw_frame.shape[0] * raw_scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                raw_ok, raw_buffer = cv2.imencode(
                    ".jpg", raw_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, max(55, self.jpeg_quality - 4)],
                )
                if raw_ok:
                    with self.raw_jpeg_condition:
                        self.latest_raw_jpeg = bytes(raw_buffer)
                        self.latest_raw_jpeg_seq += 1
                        self.latest_raw_frame_at = time.monotonic()
                        self.raw_jpeg_condition.notify_all()
                next_raw_render_at = time.monotonic() + self.raw_render_interval

            if not self.has_annotated_clients():
                last_frame_seq = frame_seq
                next_render_at = time.monotonic() + self.render_interval
                continue

            current_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            with self.result_lock:
                packet = self.latest_ai_packet
                stage_packet = self.latest_stage_packet
                privacy_packet = self.latest_privacy_packet

            # Fresh AI detections were produced on an older captured frame.
            # Project them forward through the stored frame history one step at
            # a time instead of performing one large optical-flow jump.
            applied_new_ai = False
            if (
                packet is not None
                and packet["version"] != applied_ai_version
            ):
                if bool(packet.get("geometry_update", True)):
                    tracked_detections = _copy_detections(packet["detections"])
                    if ai_uses_cuda():
                        # The canonical bbox is still the real best.pt output.
                        # Only the display copy is projected for a very short
                        # horizon to compensate capture/inference age.
                        packet_seq = int(packet.get("frame_seq") or frame_seq)
                        capture_lag = max(
                            0.0,
                            (int(frame_seq) - packet_seq) / max(1.0, float(self.source_fps)),
                        )
                        max_projection = (
                            max(0.0, min(0.20, float(settings.vehicle_display_projection_max_seconds)))
                            if settings.vehicle_display_projection_enabled
                            else 0.0
                        )
                        initial_projection = min(max_projection, capture_lag)
                        if initial_projection > 0.0 and tracked_detections:
                            tracked_detections = _project_detections_by_velocity(
                                tracked_detections,
                                delta_seconds=initial_projection,
                                detector_interval_seconds=max(
                                    0.08, float(self.last_scheduled_ai_interval)
                                ),
                                width=frame.shape[1],
                                height=frame.shape[0],
                            )
                        velocity_projection_seconds = initial_projection
                        last_velocity_projection_at = time.monotonic()
                        previous_gray = None
                    else:
                        packet_gray = cv2.cvtColor(packet["frame"], cv2.COLOR_BGR2GRAY)
                        history = self._gray_frames_between(
                            int(packet.get("frame_seq") or 0),
                            int(frame_seq),
                        )
                        prev_gray = packet_gray
                        for _seq, history_gray in history:
                            if not tracked_detections:
                                break
                            tracked_detections = _propagate_detections(
                                prev_gray, history_gray, tracked_detections
                            )
                            prev_gray = history_gray
                        previous_gray = current_gray
                        velocity_projection_seconds = 0.0
                        last_velocity_projection_at = time.monotonic()
                    applied_new_ai = True
                    last_ai_at = float(packet["at"])
                    last_geometry_at = time.monotonic()
                else:
                    # Merge the delayed tire/body classification into the
                    # boxes already projected to the current frame. Geometry,
                    # velocity and tracking diagnostics remain untouched.
                    classified_by_track = {
                        int(item["track_id"]): item
                        for item in packet.get("detections") or []
                        if item.get("track_id") is not None
                    }
                    tracked_detections = [
                        _merge_stage_state(
                            tracked,
                            classified_by_track.get(int(tracked["track_id"]))
                            if tracked.get("track_id") is not None
                            else None,
                        )
                        for tracked in tracked_detections
                    ]

                applied_ai_version = int(packet["version"])
                inference_ms = int(packet["inference_ms"])
                stage_override = packet.get("stage")
                stage_votes = dict(packet.get("stage_votes") or {})

            # Stage results have an independent mailbox. A slow label update
            # can therefore never overwrite or hide a newer geometry packet.
            if (
                stage_packet is not None
                and stage_packet["version"] != applied_stage_version
            ):
                classified_by_track = {
                    int(item["track_id"]): item
                    for item in stage_packet.get("detections") or []
                    if item.get("track_id") is not None
                }
                tracked_detections = [
                    _merge_stage_state(
                        tracked,
                        classified_by_track.get(int(tracked["track_id"]))
                        if tracked.get("track_id") is not None
                        else None,
                    )
                    for tracked in tracked_detections
                ]
                applied_stage_version = int(stage_packet["version"])
                stage_override = stage_packet.get("stage")
                stage_votes = dict(stage_packet.get("stage_votes") or {})

            # Between AI results, follow the immediately preceding displayed
            # frame. This is a small, well-conditioned LK step. V8.5.20 used a
            # fixed 0.65 s expiry even when a CPU best.pt pass itself took
            # longer, so correct boxes disappeared between detector results.
            # Hold geometry long enough for the measured detector latency while
            # still enforcing a bounded maximum to avoid stale road-texture
            # boxes.
            detector_seconds = max(0.0, float(inference_ms or 0) / 1000.0)
            geometry_flow_seconds = max(
                1.25,
                float(settings.stage_tracking_max_flow_seconds),
                min(2.40, detector_seconds * 1.45 + 0.45),
            )
            if (
                not ai_uses_cuda()
                and not applied_new_ai
                and previous_gray is not None
                and tracked_detections
                and last_geometry_at
                and time.monotonic() - last_geometry_at
                <= geometry_flow_seconds
            ):
                tracked_detections = _propagate_detections(
                    previous_gray, current_gray, tracked_detections
                )

            if (
                ai_uses_cuda()
                and settings.vehicle_display_projection_enabled
                and not applied_new_ai
                and tracked_detections
                and last_geometry_at
            ):
                projection_now = time.monotonic()
                if last_velocity_projection_at <= 0.0:
                    last_velocity_projection_at = projection_now
                step_seconds = max(0.0, projection_now - last_velocity_projection_at)
                max_projection = max(
                    0.0, min(0.20, float(settings.vehicle_display_projection_max_seconds))
                )
                remaining_seconds = max(0.0, max_projection - velocity_projection_seconds)
                step_seconds = min(step_seconds, remaining_seconds)
                if step_seconds > 0.0:
                    tracked_detections = _project_detections_by_velocity(
                        tracked_detections,
                        delta_seconds=step_seconds,
                        detector_interval_seconds=max(
                            0.08, float(self.last_scheduled_ai_interval)
                        ),
                        width=frame.shape[1],
                        height=frame.shape[0],
                    )
                    velocity_projection_seconds += step_seconds
                last_velocity_projection_at = projection_now

            geometry_hold_seconds = (
                # CUDA inference can still pause briefly when tire/body batches
                # are executing. Do not erase the last verified best.pt box
                # after only 1.2 s; keep it visible while the next geometry
                # packet catches up. Geometry is never replaced by old stage
                # packets, so this does not reintroduce the old box-jump bug.
                max(8.0, min(20.0, detector_seconds * 5.0 + 8.0))
                if ai_uses_cuda()
                else max(8.0, min(20.0, detector_seconds * 4.0 + 8.0))
            )
            if (
                tracked_detections
                and last_geometry_at
                and time.monotonic() - last_geometry_at
                > geometry_hold_seconds
            ):
                # After the optical-flow safety window expires, keep the last
                # verified geometry stationary until the next detector result.
                # Only clear it after a longer bounded hold. This prevents boxes
                # blinking off on non-focused cameras when CPU inference is
                # round-robin across several open windows, while avoiding long
                # drift from stale road texture.
                tracked_detections = []

            # Privacy boxes are refreshed frequently. Haar boxes are NOT
            # optical-flow propagated because a false plate on a wall/sign can
            # otherwise drift into a huge mosaic. A dedicated privacy YOLO may
            # still use short optical-flow interpolation.
            if (
                privacy_packet is not None
                and privacy_packet["version"] != applied_privacy_version
            ):
                privacy_detections = _copy_detections(privacy_packet["detections"])
                privacy_previous_gray = cv2.cvtColor(
                    privacy_packet["frame"], cv2.COLOR_BGR2GRAY
                )
                applied_privacy_version = int(privacy_packet["version"])
                last_privacy_at = float(privacy_packet["at"])
                privacy_backend = str(privacy_packet.get("backend") or "unknown")

            allow_privacy_flow = not (
                bool(settings.anonymizer_disable_haar_flow)
                and privacy_backend == "opencv-haar"
            )
            if (
                allow_privacy_flow
                and privacy_previous_gray is not None
                and privacy_detections
            ):
                privacy_detections = _propagate_detections(
                    privacy_previous_gray, current_gray, privacy_detections
                )

            if (
                last_privacy_at
                and time.monotonic() - last_privacy_at
                > max(1.0, self.privacy_interval * 3.5)
            ):
                privacy_detections = []

            privacy_previous_gray = current_gray

            # 오래된 박스가 남는 것을 방지합니다.
            if (
                last_ai_at
                and time.monotonic() - last_ai_at
                > max(12.0, self.ai_interval * 8.0)
            ):
                tracked_detections = []
                stage_override = None
                stage_votes = {}

            previous_gray = current_gray
            privacy_frame = frame.copy()
            scene_privacy_detections = _filter_privacy_detections_for_scene(
                privacy_detections,
                tracked_detections,
                frame.shape,
                privacy_backend,
            )
            if scene_privacy_detections:
                apply_privacy(privacy_frame, scene_privacy_detections)
            # V8.5.37 uses the server-rendered JPEG as the single authoritative
            # vehicle-box renderer. V8.5.35 deliberately disabled server boxes
            # and relied entirely on the browser vector overlay; the user's live
            # run showed smooth video but no visible rectangles. The older FULL
            # build had already proven that server-side OpenCV rectangles are
            # visible in this exact UI. Keep WebSocket detection metadata only
            # for diagnostics/API consumers, and draw each vehicle exactly once
            # here before JPEG encoding.
            annotated = _draw_live_detections(
                privacy_frame,
                tracked_detections,
                inference_ms,
                stage_override,
                stage_votes,
                draw_boxes=True,
            )
            if settings.anonymizer_enabled:
                cv2.putText(
                    annotated,
                    f"PRIVACY {privacy_backend} {len(scene_privacy_detections)}",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (225, 235, 245),
                    1,
                    cv2.LINE_AA,
                )

            jpeg_frame = annotated
            if jpeg_frame.shape[1] > 640:
                jpeg_scale = 640.0 / float(jpeg_frame.shape[1])
                jpeg_frame = cv2.resize(
                    jpeg_frame,
                    (640, max(1, int(jpeg_frame.shape[0] * jpeg_scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buffer = cv2.imencode(
                ".jpg",
                jpeg_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    max(56, min(62, self.jpeg_quality)),
                ],
            )
            if ok:
                with self.jpeg_condition:
                    self.latest_jpeg = bytes(buffer)
                    self.latest_jpeg_seq += 1
                    self.latest_annotated_frame_at = time.monotonic()
                    self.jpeg_condition.notify_all()

            last_frame_seq = frame_seq
            next_render_at = (
                time.monotonic() + self.render_interval
            )

    def wait_for_jpeg(
        self,
        last_seq: int,
        timeout: float = 2.0,
    ) -> tuple[int, bytes | None]:
        with self.jpeg_condition:
            if (
                self.latest_jpeg is None
                or self.latest_jpeg_seq == last_seq
            ):
                self.jpeg_condition.wait(timeout=timeout)

            return self.latest_jpeg_seq, self.latest_jpeg

    def wait_for_raw_jpeg(
        self,
        last_seq: int,
        timeout: float = 2.0,
    ) -> tuple[int, bytes | None]:
        with self.raw_jpeg_condition:
            if (
                self.latest_raw_jpeg is None
                or self.latest_raw_jpeg_seq == last_seq
            ):
                self.raw_jpeg_condition.wait(timeout=timeout)
            return self.latest_raw_jpeg_seq, self.latest_raw_jpeg


def _get_camera_worker(stream_url: str) -> CameraWorker:
    with _camera_workers_lock:
        worker = _camera_workers.get(stream_url)
        if worker is None or not worker.is_alive():
            if worker is not None:
                try:
                    worker.stop()
                except Exception:
                    pass
            worker = CameraWorker(stream_url)
            _camera_workers[stream_url] = worker
            worker.start()
        return worker


def camera_worker_status(stream_url: str) -> dict[str, Any]:
    with _camera_workers_lock:
        worker = _camera_workers.get(stream_url)

    if worker is None:
        return {
            "active": False,
            "clients": 0,
            "annotated_clients": 0,
            "raw_clients": 0,
            "last_error": None,
            "last_frame_age_seconds": None,
            "last_render_age_seconds": None,
            "latest_result_age_seconds": None,
            "reconnect_count": 0,
            "fallback_frame_count": 0,
            "hls_stall_resets": 0,
            "latest_detection_count": 0,
            "annotated_interest_active": False,
            "focused_interest_active": False,
            "ai_cadence_mode": None,
            "ai_interval_seconds": None,
            "inference_scheduler": inference_scheduler_status(),
        }

    age = (
        max(0.0, time.monotonic() - worker.last_stream_ok_at)
        if worker.last_stream_ok_at
        else None
    )
    render_age = (
        max(0.0, time.monotonic() - worker.latest_annotated_frame_at)
        if worker.latest_annotated_frame_at
        else None
    )
    result_age = (
        max(0.0, time.monotonic() - worker.latest_result_at)
        if worker.latest_result_at
        else None
    )
    return {
        "active": worker.is_alive(),
        "clients": worker.clients,
        "annotated_clients": worker.annotated_clients,
        "raw_clients": worker.raw_clients,
        "last_error": worker.last_stream_error,
        "last_frame_age_seconds": (round(age, 2) if age is not None else None),
        "last_render_age_seconds": (
            round(render_age, 2) if render_age is not None else None
        ),
        "latest_result_age_seconds": (
            round(result_age, 2) if result_age is not None else None
        ),
        "reconnect_count": worker.reconnect_count,
        "fallback_frame_count": worker.fallback_frame_count,
        "hls_stall_resets": worker.hls_stall_resets,
        "latest_stage": worker.latest_result.get("stage"),
        "latest_confidence": worker.latest_result.get("conf"),
        "ai_thread_alive": bool(worker.ai_thread and worker.ai_thread.is_alive()),
        "ai_state": worker.last_ai_state,
        "last_ai_heartbeat_age_seconds": (
            round(max(0.0, time.monotonic() - worker.last_ai_loop_heartbeat_at), 2)
            if worker.last_ai_loop_heartbeat_at else None
        ),
        "last_ai_attempt_age_seconds": (
            round(max(0.0, time.monotonic() - worker.last_ai_attempt_at), 2)
            if worker.last_ai_attempt_at else None
        ),
        "last_ai_success_age_seconds": (
            round(max(0.0, time.monotonic() - worker.last_ai_success_at), 2)
            if worker.last_ai_success_at else None
        ),
        "raw_detection_count": int(worker.latest_result.get("raw_detection_count") or 0),
        "visible_detection_count": int(worker.latest_result.get("visible_detection_count") or len(worker.latest_result.get("detections") or [])),
        "stage_confirmed_count": int(worker.latest_result.get("stage_confirmed_count") or 0),
        "visible_confidence_floor": worker.latest_result.get("visible_confidence_floor"),
        "latest_detection_count": len(worker.latest_result.get("detections") or []),
        "annotated_interest_active": worker.has_annotated_clients(),
        "focused_interest_active": worker.has_focus_interest(),
        "ai_cadence_mode": worker.last_ai_cadence_mode,
        "ai_interval_seconds": round(float(worker.last_scheduled_ai_interval), 3),
        "inference_scheduler": inference_scheduler_status(),
    }


def has_live_cctv_clients() -> bool:
    """True while a browser has a current annotated-stream interest lease."""
    with _camera_workers_lock:
        workers = list(_camera_workers.values())
    active = any(
        worker.is_alive() and worker.has_annotated_clients()
        for worker in workers
    )
    if not active:
        # Snapshot leases expire without a disconnect callback. Clear the global
        # live-priority gate when the last lease naturally expires so background
        # CCTV analysis can resume.
        set_live_inference_priority(False)
    return active


def _transport_detection_metadata(worker: "CameraWorker") -> dict[str, Any]:
    """Small JSON-safe detector payload for browser-side vector boxes.

    The detector result and the rendered JPEG used to travel through different
    timing paths. The API could report valid best.pt detections while a WebSocket
    frame was raw/fallback, so the user saw no rectangles. Sending the latest
    geometry with every transport frame makes video delivery independent from AI
    box rendering and keeps boxes visible even during a raw-frame fallback.
    """
    with worker.result_lock:
        result = dict(worker.latest_result or {})
        raw = list(result.get("detections") or [])[:32]
        detector_ms = result.get("detector_ms")
        stage = result.get("stage")
    with worker.frame_condition:
        frame = worker.latest_frame
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
        else:
            frame_width = frame_height = 0

    detections: list[dict[str, Any]] = []
    for item in raw:
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        try:
            xyxy = [round(float(v), 1) for v in bbox]
        except (TypeError, ValueError):
            continue
        stage_valid = bool(
            item.get("stage") is not None
            and item.get("stage_valid") is not False
        )
        display_conf = (
            item.get("stage_conf")
            if (stage_valid or item.get("stage_rejected_low_confidence"))
            and item.get("stage_conf") is not None
            else item.get("vehicle_conf", item.get("conf"))
        )
        detections.append({
            "bbox": xyxy,
            "track_id": item.get("track_id"),
            "conf": round(float(display_conf or 0.0), 4),
            "stage": item.get("stage") if stage_valid else None,
            "stage_valid": stage_valid,
            "stage_source": str(item.get("stage_source") or ""),
            "provisional": bool(item.get("_provisional")),
            "confirmed": bool(item.get("_confirmed", not item.get("_provisional"))),
            "vehicle_label": str(item.get("vehicle_label") or item.get("source_label") or "vehicle"),
        })
    return {
        "detections": detections,
        "detection_frame_width": int(frame_width),
        "detection_frame_height": int(frame_height),
        "detector_ms": detector_ms,
        "stage": stage,
    }


def live_transport_packet(
    stream_url: str,
    *,
    focused: bool = False,
) -> dict[str, Any]:
    """Return the newest ready JPEG for the single multiplexed WebSocket."""
    worker = _get_camera_worker(stream_url)
    worker.renew_annotated_interest(2.4, focused=focused)
    metadata = _transport_detection_metadata(worker)
    now = time.monotonic()
    # Always expose whether the browser transport is seeing the detector state.
    # This fires independently from _ai_loop, so even a lease/gating failure is
    # visible in PowerShell instead of looking like a silent renderer problem.
    if now - worker.last_transport_diag_log_at >= 5.0:
        result_age = (
            max(0.0, now - worker.latest_result_at)
            if worker.latest_result_at else None
        )
        scheduler = inference_scheduler_status()
        logger.warning(
            "CCTV WS62 annotated=%s focus=%s ai_alive=%s ai_state=%s "
            "detmeta=%s result_age=%s cadence=%s/%.3f sched=%s/%s qv=%s qs=%s batch=%s [%s]",
            worker.has_annotated_clients(),
            worker.has_focus_interest(),
            bool(worker.ai_thread and worker.ai_thread.is_alive()),
            worker.last_ai_state,
            len(metadata.get("detections") or []),
            (round(result_age, 2) if result_age is not None else None),
            worker.last_ai_cadence_mode,
            float(worker.last_scheduled_ai_interval),
            scheduler.get("running_kind"),
            scheduler.get("running_age_seconds"),
            scheduler.get("vehicle_queue"),
            scheduler.get("stage_queue"),
            scheduler.get("last_batch_size"),
            _stream_label(stream_url),
        )
        worker.last_transport_diag_log_at = now
    annotated_age = (
        max(0.0, now - worker.latest_annotated_frame_at)
        if worker.latest_annotated_frame_at else None
    )
    if worker.latest_jpeg is not None and annotated_age is not None and annotated_age <= 1.10:
        return {
            "mode": "annotated", "ready": True,
            "seq": int(worker.latest_jpeg_seq), "jpeg": worker.latest_jpeg,
            "age_seconds": round(float(annotated_age), 3),
            **metadata,
        }

    worker.renew_raw_interest(2.4)
    raw_age = (
        max(0.0, now - worker.latest_raw_frame_at)
        if worker.latest_raw_frame_at else None
    )
    if worker.latest_raw_jpeg is not None and raw_age is not None and raw_age <= 1.0:
        return {
            "mode": "raw", "ready": True,
            "seq": int(worker.latest_raw_jpeg_seq), "jpeg": worker.latest_raw_jpeg,
            "age_seconds": round(float(raw_age), 3),
            **metadata,
        }

    jpeg = worker.latest_raw_jpeg or worker.latest_jpeg
    if jpeg is None:
        jpeg = _status_jpeg("CCTV CONNECTING", _stream_label(stream_url)) or b""
    return {
        "mode": "status", "ready": False,
        "seq": max(int(worker.latest_raw_jpeg_seq), int(worker.latest_jpeg_seq)),
        "jpeg": jpeg, "age_seconds": None,
        **metadata,
    }


def raw_snapshot(
    stream_url: str,
    *,
    timeout: float = 0.45,
) -> tuple[bytes, bool, int]:
    """Return a recent raw JPEG without holding an HTTP connection open.

    V8.5.22 keeps a short raw-interest lease inside the shared camera worker.
    That means tiled-window snapshot requests usually return an already-fresh
    JPEG immediately instead of each request waiting for the renderer to notice
    a transient raw client. This removes request pile-ups and visible stutter.
    """
    worker = _get_camera_worker(stream_url)
    worker.add_client(annotated=False)
    deadline = time.monotonic() + max(0.05, min(0.70, float(timeout)))
    seq = int(worker.latest_raw_jpeg_seq)
    jpeg = worker.latest_raw_jpeg
    try:
        age = (
            time.monotonic() - worker.latest_raw_frame_at
            if worker.latest_raw_frame_at else None
        )
        # Raw JPEGs are intentionally encoded at about 5 fps to keep multiple
        # background windows cheap. Reusing one that is at most 220 ms old is
        # effectively live and avoids waiting a worker thread for every request.
        if jpeg is not None and age is not None and age <= 0.22:
            return jpeg, True, seq

        initial_seq = seq
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                break
            next_seq, next_jpeg = worker.wait_for_raw_jpeg(
                seq,
                timeout=min(0.14, remaining),
            )
            if next_jpeg is not None:
                jpeg = next_jpeg
            seq = int(next_seq)

            age = (
                time.monotonic() - worker.latest_raw_frame_at
                if worker.latest_raw_frame_at else None
            )
            ready = age is not None and age <= 2.0
            if ready and seq > initial_seq and jpeg is not None:
                return jpeg, True, seq

        if jpeg is None:
            jpeg = _status_jpeg(
                "CCTV CONNECTING",
                _stream_label(stream_url),
            ) or b""
        age = (
            time.monotonic() - worker.latest_raw_frame_at
            if worker.latest_raw_frame_at else None
        )
        return jpeg, bool(age is not None and age <= 2.0), int(seq)
    finally:
        worker.remove_client(annotated=False)

def annotated_snapshot(
    stream_url: str,
    *,
    timeout: float = 0.20,
    focused: bool = False,
) -> tuple[bytes, bool, int]:
    """Return the newest AI JPEG immediately using a renewable interest lease.

    V8.5.23 deliberately does not wait inside the HTTP request for the next AI
    frame. With several CCTV windows open, blocking snapshot handlers caused a
    thread-pool queue and made every window stutter together. The browser polls
    the sequence header and keeps the previous decoded JPEG until a newer one is
    available, so an immediate cached response is both smoother and more robust.
    """
    del timeout  # kept for API compatibility with older callers
    worker = _get_camera_worker(stream_url)
    worker.renew_annotated_interest(1.8, focused=focused)
    seq = int(worker.latest_jpeg_seq)
    jpeg = worker.latest_jpeg
    if jpeg is None:
        jpeg = _status_jpeg(
            "CCTV CONNECTING", _stream_label(stream_url)
        ) or b""
    age = (
        time.monotonic() - worker.latest_annotated_frame_at
        if worker.latest_annotated_frame_at else None
    )
    return jpeg, bool(age is not None and age <= 2.5), seq


def annotated_mjpeg(stream_url: str) -> Iterator[bytes]:
    """
    브라우저 연결마다 YOLO를 새로 실행하지 않고 CCTV 주소별 공유 워커의
    최신 JPEG만 전송합니다.
    """
    # Set priority before the worker thread starts so a background inference
    # cannot slip into the model queue during the small creation race.
    set_live_inference_priority(True)
    worker = _get_camera_worker(stream_url)
    worker.add_client(annotated=True)
    last_seq = -1
    initial_jpeg_seq = int(worker.latest_jpeg_seq)
    first_real_deadline = time.monotonic() + 1.20

    header = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Cache-Control: no-store, no-cache, must-revalidate\r\n"
        b"Pragma: no-cache\r\n\r\n"
    )

    try:
        while worker.is_alive():
            jpeg_seq, jpeg = worker.wait_for_jpeg(
                last_seq,
                timeout=1.0,
            )
            if jpeg is None:
                jpeg = _status_jpeg(
                    "CCTV CONNECTING",
                    _stream_label(stream_url),
                )
                if jpeg is None:
                    continue
            # Keep the startup/status JPEG behind the short raw-snapshot layer
            # for a moment. This lets a real camera frame become the first AI
            # image the browser promotes instead of replacing an already-live
            # raw picture with another "CONNECTING" card.
            if (
                time.monotonic() < first_real_deadline
                and (
                    not worker.last_stream_ok_at
                    or int(jpeg_seq) <= initial_jpeg_seq
                )
            ):
                last_seq = jpeg_seq
                continue
            # Yield even when the sequence did not advance. A sync
            # StreamingResponse iterator runs in Starlette's worker pool; an
            # endless internal wait here permanently occupied a pool thread for
            # each stalled CCTV and eventually made lightweight dashboard APIs
            # fail with browser-side "Failed to fetch". A one-second repeated
            # frame is also a useful MJPEG keepalive and lets disconnects unwind.
            last_seq = jpeg_seq
            yield header + jpeg + b"\r\n"
    finally:
        worker.remove_client(annotated=True)
        if not has_live_cctv_clients():
            set_live_inference_priority(False)


def raw_mjpeg(stream_url: str) -> Iterator[bytes]:
    """Lightweight live video for non-focused CCTV windows."""
    worker = _get_camera_worker(stream_url)
    worker.add_client(annotated=False)
    last_seq = -1
    header = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Cache-Control: no-store, no-cache, must-revalidate\r\n"
        b"Pragma: no-cache\r\n\r\n"
    )
    try:
        while worker.is_alive():
            jpeg_seq, jpeg = worker.wait_for_raw_jpeg(last_seq, timeout=1.0)
            if jpeg is None:
                jpeg = _status_jpeg(
                    "CCTV CONNECTING",
                    _stream_label(stream_url),
                )
                if jpeg is None:
                    continue
            last_seq = jpeg_seq
            yield header + jpeg + b"\r\n"
    finally:
        worker.remove_client(annotated=False)
