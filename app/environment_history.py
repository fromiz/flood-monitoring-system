from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError

from .config import settings
from .database import session_scope
from .external_data import get_river_levels, get_sewer_levels
from .models import EnvironmentalObservation
from .realtime_weather import weather_service

KST = timezone(timedelta(hours=9))


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _naive_utc(value: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return fallback or datetime.utcnow()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            # Common compact public-API timestamps.
            parsed = None
            for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                return fallback or datetime.utcnow()

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _safe_details(payload: dict[str, Any]) -> str:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
    return json.dumps(safe, ensure_ascii=False)


def _sensor_level(sensor: dict[str, Any]) -> float | None:
    value = _float(sensor.get("level_m"))
    if value is not None:
        return value
    value = _float(sensor.get("water_level_m"))
    if value is not None:
        return value
    value = _float(sensor.get("wal"))
    if value is not None:
        return value
    cm = _float(sensor.get("level_cm"))
    if cm is None:
        cm = _float(sensor.get("water_level_cm"))
    if cm is None:
        cm = _float(sensor.get("value"))
    return None if cm is None else cm / 100.0


def _rain_observations(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    grid = snapshot.get("grid") or {}
    for point in grid.get("points") or []:
        rain = _float(point.get("rain_1h_mm"))
        if rain is None:
            continue
        sensor_id = str(point.get("id") or point.get("name") or "RAIN-GRID").strip()
        observations.append({
            "sensor_type": "rain",
            "sensor_id": sensor_id,
            "sensor_name": str(point.get("name") or sensor_id),
            "value": max(0.0, rain),
            "unit": "mm/60m",
            "lat": _float(point.get("lat")),
            "lon": _float(point.get("lon")),
            "observed_at": point.get("observed_at") or grid.get("updated_at") or snapshot.get("observed_at"),
            "source": str(grid.get("source") or snapshot.get("source") or "KMA"),
            "details": {
                "rain_1h_mm": max(0.0, rain),
                "grid": True,
            },
        })

    # Preserve the official station history as well. It is useful when the
    # 500m grid API is unavailable and is intentionally a separate sensor ID.
    station = snapshot.get("station") or {}
    station_rain = _float(snapshot.get("rain_60m_mm"))
    if station_rain is None:
        station_rain = _float(snapshot.get("rain_1h_mm"))
    if station_rain is not None:
        station_id = str(station.get("id") or "KMA-POHANG").strip()
        observations.append({
            "sensor_type": "rain",
            "sensor_id": f"KMA-{station_id}",
            "sensor_name": str(station.get("name") or "포항 기상관측"),
            "value": max(0.0, station_rain),
            "unit": "mm/60m",
            "lat": _float(station.get("lat")),
            "lon": _float(station.get("lon")),
            "observed_at": snapshot.get("observed_at"),
            "source": str(snapshot.get("source") or "KMA"),
            "details": {
                "rain_1m_mm": _float(snapshot.get("rain_1m_mm")),
                "rain_60m_mm": max(0.0, station_rain),
                "rain_day_mm": _float(snapshot.get("rain_day_mm")),
                "temperature_c": _float(snapshot.get("temperature_c")),
                "humidity_pct": _float(snapshot.get("humidity_pct")),
                "wind_ms": _float(snapshot.get("wind_ms")),
                "grid": False,
            },
        })
    return observations


def _water_observations(kind: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    sensors = payload.get("sensors") or payload.get("items") or []
    default_observed = payload.get("updated_at")
    source = str(payload.get("source") or kind)
    for index, sensor in enumerate(sensors):
        if not isinstance(sensor, dict):
            continue
        level = _sensor_level(sensor)
        if level is None:
            continue
        sensor_id = str(sensor.get("id") or sensor.get("code") or f"{kind.upper()}-{index + 1}").strip()
        observations.append({
            "sensor_type": kind,
            "sensor_id": sensor_id,
            "sensor_name": str(sensor.get("name") or sensor.get("station_name") or sensor_id),
            "value": level,
            "unit": "m",
            "lat": _float(sensor.get("lat")),
            "lon": _float(sensor.get("lon")),
            "observed_at": sensor.get("observed_at") or sensor.get("time") or default_observed,
            "source": source,
            "details": {
                "level_m": level,
                "flow_cms": _float(sensor.get("flow_cms") if sensor.get("flow_cms") is not None else sensor.get("flux")),
                "warning_m": _float(sensor.get("warning_m")),
                "danger_m": _float(sensor.get("danger_m")),
            },
        })
    return observations


def collect_environment_snapshot() -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    observations.extend(_rain_observations(weather_service.snapshot()))
    observations.extend(_water_observations("sewer", get_sewer_levels()))
    observations.extend(_water_observations("river", get_river_levels()))
    return observations


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


class EnvironmentalHistoryRecorder:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "running": False,
            "stored": 0,
            "duplicates": 0,
            "last_run_at": None,
            "last_stored_at": None,
            "last_error": None,
            "last_counts": {"rain": 0, "sewer": 0, "river": 0},
        }

    def start(self) -> None:
        if not bool(settings.environment_history_enabled):
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="environment-history", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._status)
            result["last_counts"] = dict(self._status.get("last_counts") or {})
        result["enabled"] = bool(settings.environment_history_enabled)
        result["interval_seconds"] = int(settings.environment_history_interval_seconds)
        result["retention_days"] = int(settings.environment_history_retention_days)
        return result

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)

    def collect_once(self) -> dict[str, Any]:
        now = datetime.utcnow()
        raw = collect_environment_snapshot()
        rows: list[EnvironmentalObservation] = []
        counts = {"rain": 0, "sewer": 0, "river": 0}
        duplicate_count = 0

        # De-duplicate in-memory first because a grid/station payload can contain
        # repeated aliases in the same collection cycle.
        seen: set[tuple[str, str, datetime]] = set()
        for item in raw:
            sensor_type = str(item.get("sensor_type") or "").strip().lower()
            sensor_id = str(item.get("sensor_id") or "").strip()
            value = _float(item.get("value"))
            if sensor_type not in counts or not sensor_id or value is None:
                continue
            observed_at = _naive_utc(item.get("observed_at"), now)
            # Public data often has second/millisecond differences for the same
            # observation. Store at second precision to keep deterministic keys.
            observed_at = observed_at.replace(microsecond=0)
            key = (sensor_type, sensor_id, observed_at)
            if key in seen:
                continue
            seen.add(key)
            rows.append(EnvironmentalObservation(
                sensor_type=sensor_type,
                sensor_id=sensor_id,
                sensor_name=str(item.get("sensor_name") or sensor_id),
                value=float(value),
                unit=str(item.get("unit") or ""),
                lat=_float(item.get("lat")),
                lon=_float(item.get("lon")),
                observed_at=observed_at,
                recorded_at=now,
                source=str(item.get("source") or "") or None,
                details=_safe_details(item.get("details") or {}),
            ))

        stored = 0
        if rows:
            # Query existing unique keys before insert. This avoids filling the
            # SQLite log with expected UNIQUE failures every polling cycle.
            with session_scope() as session:
                for row in rows:
                    exists = session.scalar(
                        select(EnvironmentalObservation.id)
                        .where(EnvironmentalObservation.sensor_type == row.sensor_type)
                        .where(EnvironmentalObservation.sensor_id == row.sensor_id)
                        .where(EnvironmentalObservation.observed_at == row.observed_at)
                        .limit(1)
                    )
                    if exists is not None:
                        duplicate_count += 1
                        continue
                    session.add(row)
                    counts[row.sensor_type] += 1
                    stored += 1

        # Lightweight retention cleanup, at most once per run.
        retention_days = max(1, int(settings.environment_history_retention_days))
        cutoff = now - timedelta(days=retention_days)
        with session_scope() as session:
            old_ids = list(session.scalars(
                select(EnvironmentalObservation.id)
                .where(EnvironmentalObservation.observed_at < cutoff)
                .order_by(EnvironmentalObservation.id.asc())
                .limit(5000)
            ).all())
            if old_ids:
                for row_id in old_ids:
                    row = session.get(EnvironmentalObservation, row_id)
                    if row is not None:
                        session.delete(row)

        with self._lock:
            self._status["stored"] = int(self._status.get("stored") or 0) + stored
            self._status["duplicates"] = int(self._status.get("duplicates") or 0) + duplicate_count
            self._status["last_run_at"] = now.isoformat(timespec="seconds") + "Z"
            self._status["last_counts"] = counts
            self._status["last_error"] = None
            if stored:
                self._status["last_stored_at"] = now.isoformat(timespec="seconds") + "Z"
        return {"stored": stored, "duplicates": duplicate_count, "counts": counts}

    def _run(self) -> None:
        self._update(running=True)
        delay = max(0, int(settings.environment_history_start_delay_seconds))
        if delay and self._stop.wait(delay):
            self._update(running=False)
            return
        interval = max(15, int(settings.environment_history_interval_seconds))
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.collect_once()
            except (IntegrityError, Exception) as exc:
                self._update(
                    last_run_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    last_error=str(exc)[:500],
                )
            elapsed = time.monotonic() - started
            if self._stop.wait(max(1.0, interval - elapsed)):
                break
        self._update(running=False)


environment_history_recorder = EnvironmentalHistoryRecorder()


def _candidate_sensor(session, sensor_type: str, range_end: datetime) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            EnvironmentalObservation.sensor_id,
            func.max(EnvironmentalObservation.sensor_name),
            func.max(EnvironmentalObservation.lat),
            func.max(EnvironmentalObservation.lon),
            func.max(EnvironmentalObservation.observed_at),
        )
        .where(EnvironmentalObservation.sensor_type == sensor_type)
        .where(EnvironmentalObservation.observed_at <= range_end)
        .group_by(EnvironmentalObservation.sensor_id)
    ).all()
    return [
        {
            "sensor_id": row[0],
            "sensor_name": row[1] or row[0],
            "lat": _float(row[2]),
            "lon": _float(row[3]),
            "latest_at": row[4],
        }
        for row in rows
    ]


def _choose_sensor(candidates: list[dict[str, Any]], lat: float | None, lon: float | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    if lat is not None and lon is not None:
        located = [item for item in candidates if item.get("lat") is not None and item.get("lon") is not None]
        if located:
            chosen = min(
                located,
                key=lambda item: _haversine_m(lat, lon, float(item["lat"]), float(item["lon"])),
            )
            chosen = dict(chosen)
            chosen["distance_m"] = round(
                _haversine_m(lat, lon, float(chosen["lat"]), float(chosen["lon"])), 1
            )
            return chosen
    return dict(max(candidates, key=lambda item: item.get("latest_at") or datetime.min))


def _bucket_time(dt: datetime, bucket_minutes: int) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    minute = (dt.minute // bucket_minutes) * bucket_minutes
    return dt.replace(minute=minute)


def environment_history_for_location(
    *,
    range_start: datetime,
    range_end: datetime,
    bucket_minutes: int,
    lat: float | None,
    lon: float | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    with session_scope() as session:
        for sensor_type in ("rain", "sewer", "river"):
            candidates = _candidate_sensor(session, sensor_type, range_end)
            active_candidates = [
                item for item in candidates
                if item.get("latest_at") is not None
                and item["latest_at"] >= range_start
            ]
            chosen = _choose_sensor(
                active_candidates or candidates,
                lat,
                lon,
            )
            if chosen is None:
                output[sensor_type] = {
                    "sensor_id": None,
                    "sensor_name": None,
                    "unit": "mm/60m" if sensor_type == "rain" else "m",
                    "distance_m": None,
                    "points": [],
                    "previous": None,
                }
                continue

            sensor_id = str(chosen["sensor_id"])
            rows = list(session.scalars(
                select(EnvironmentalObservation)
                .where(EnvironmentalObservation.sensor_type == sensor_type)
                .where(EnvironmentalObservation.sensor_id == sensor_id)
                .where(EnvironmentalObservation.observed_at >= range_start)
                .where(EnvironmentalObservation.observed_at <= range_end)
                .order_by(EnvironmentalObservation.observed_at.asc())
            ).all())
            previous = session.scalar(
                select(EnvironmentalObservation)
                .where(EnvironmentalObservation.sensor_type == sensor_type)
                .where(EnvironmentalObservation.sensor_id == sensor_id)
                .where(EnvironmentalObservation.observed_at < range_start)
                .order_by(desc(EnvironmentalObservation.observed_at))
                .limit(1)
            )

            buckets: dict[datetime, EnvironmentalObservation] = {}
            for row in rows:
                bucket = _bucket_time(row.observed_at, bucket_minutes)
                current = buckets.get(bucket)
                if current is None or row.observed_at >= current.observed_at:
                    buckets[bucket] = row

            def payload(row: EnvironmentalObservation, bucket: datetime | None = None) -> dict[str, Any]:
                try:
                    details = json.loads(row.details or "{}")
                except Exception:
                    details = {}
                return {
                    "time": (bucket or row.observed_at).isoformat() + "Z",
                    "observed_at": row.observed_at.isoformat() + "Z",
                    "value": round(float(row.value), 3),
                    "unit": row.unit,
                    "source": row.source,
                    "details": details,
                }

            points = [payload(buckets[key], key) for key in sorted(buckets)]
            output[sensor_type] = {
                "sensor_id": sensor_id,
                "sensor_name": chosen.get("sensor_name"),
                "lat": chosen.get("lat"),
                "lon": chosen.get("lon"),
                "distance_m": chosen.get("distance_m"),
                "unit": (points[-1]["unit"] if points else (previous.unit if previous else ("mm/60m" if sensor_type == "rain" else "m"))),
                "points": points,
                "previous": payload(previous) if previous is not None else None,
            }
    return output
