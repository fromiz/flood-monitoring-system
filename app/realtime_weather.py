from __future__ import annotations

import copy
import csv
import io
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from urllib.parse import urlencode, unquote
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import settings
from .external_data import (
    POHANG_RAIN_SAMPLES,
    get_pohang_rainfall_grid,
    get_weather,
)


try:
    KST = ZoneInfo("Asia/Seoul")
except ZoneInfoNotFoundError:
    # tzdata 설치 전에도 Windows에서 한국 표준시로 동작
    KST = timezone(timedelta(hours=9), name="KST")
APIHUB_BASE = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url"
APIHUB_URL_BASE = "https://apihub.kma.go.kr/api/typ01/url"


def _now_kst() -> datetime:
    return datetime.now(KST)


def _iso_kst(value: datetime | None = None) -> str:
    return (value or _now_kst()).isoformat(timespec="seconds")


def _valid_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        number = float(match.group())

    # 기상청 텍스트 API의 결측값은 보통 큰 음수로 표시됩니다.
    if number <= -9:
        return None
    return number


def _parse_time(value: Any) -> datetime | None:
    digits = re.sub(r"\D", "", str(value or ""))
    for length, fmt in (
        (12, "%Y%m%d%H%M"),
        (10, "%Y%m%d%H"),
        (8, "%Y%m%d"),
    ):
        if len(digits) >= length:
            try:
                return datetime.strptime(
                    digits[:length],
                    fmt,
                ).replace(tzinfo=KST)
            except ValueError:
                continue
    return None


def parse_apihub_table(
    text: str,
) -> list[dict[str, str]]:
    """
    기상청 API허브 AWS 매분자료를 파싱합니다.

    헤더는 공백으로 구분되고,
    관측 데이터는 쉼표로 구분됩니다.
    """
    lines = [
        line.strip().lstrip("\ufeff")
        for line in text.replace("\r", "").split("\n")
        if line.strip()
    ]

    header: list[str] | None = None
    data_lines: list[str] = []

    for line in lines:
        if line.startswith("#"):
            candidate = line.lstrip("#").strip()

            if candidate.startswith("YYMMDDHHMI"):
                fields = (
                    [value.strip() for value in next(csv.reader([candidate]))]
                    if "," in candidate
                    else re.split(r"\s+", candidate)
                )
            elif candidate.upper().startswith("TM,"):
                # 구형/진단용 APIHub 응답은 헤더도 CSV 형식입니다.
                fields = [
                    value.strip()
                    for value in next(csv.reader([candidate]))
                ]
            elif re.match(r"^TM\s+STN(?:\s|$)", candidate.upper()):
                # APIHub awsh.php and some diagnostic responses use a
                # whitespace-separated header and data rows.
                fields = re.split(r"\s+", candidate)
            else:
                fields = []

            if fields:
                header = [
                    field.upper().replace("-", "_")
                    for field in fields
                ]

                # 기존 처리 코드가 사용하는 이름으로 통일
                header[0] = "TM"

            # 설명·헤더 줄은 데이터로 처리하지 않음
            continue

        # AWS minute data is normally 12 digits; hourly fallback data can be
        # 10 or 12 digits. Both CSV (disp=1) and fixed/whitespace (disp=0)
        # forms are accepted because APIHub deployments return both.
        if re.match(r"^\d{10,12}(?:,|\s)", line):
            data_lines.append(line)

    if not header or not data_lines:
        return []

    rows: list[dict[str, str]] = []

    for line in data_lines:
        values = (
            [value.strip() for value in next(csv.reader([line]))]
            if "," in line
            else [value for value in re.split(r"\s+", line) if value]
        )

        # 응답 끝의 불필요한 '=‘ 열 제거
        if values and values[-1] == "=":
            values.pop()

        if len(values) < len(header):
            values.extend(
                [""] * (len(header) - len(values))
            )

        if len(values) > len(header):
            values = values[: len(header)]

        row = dict(zip(header, values))

        if _parse_time(row.get("TM")) is not None:
            rows.append(row)

    return rows


def parse_grid_point_text(text: str) -> tuple[datetime | None, float | None]:
    """
    500m 특정지점 ASCII 응답에서 마지막 유효시각과 강수값을 추출합니다.
    응답 열 수가 달라도 첫 번째 12자리 시각과 마지막 유효 실수를 사용합니다.
    """
    parsed: list[tuple[datetime, float]] = []

    for raw_line in text.replace("\r", "").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = [
            token
            for token in re.split(r"[\s,]+", line)
            if token
        ]
        observed_at = None
        for token in tokens:
            observed_at = _parse_time(token)
            if observed_at is not None:
                break
        if observed_at is None:
            continue

        numeric_values = [
            _valid_number(token)
            for token in tokens
        ]
        numeric_values = [
            value
            for value in numeric_values
            if value is not None
        ]
        if not numeric_values:
            continue

        # 특정지점 단일요소 응답의 마지막 숫자가 조회 요소 값입니다.
        value = numeric_values[-1]
        parsed.append((observed_at, max(0.0, value)))

    return parsed[-1] if parsed else (None, None)


def _pick(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = _valid_number(row.get(name))
        if value is not None:
            return value
    return None


class RealtimeWeatherService:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._version = 0

        self._minute_history: list[dict[str, Any]] = []
        self._last_good_aws: dict[str, Any] | None = None
        self._last_good_grid: dict[str, Any] | None = None
        self._last_good_forecast: list[dict[str, Any]] = []

        self._snapshot: dict[str, Any] = self._empty_snapshot()

    def _empty_snapshot(self) -> dict[str, Any]:
        api_key = bool(settings.kma_apihub_auth_key.strip())
        portal_key = bool(settings.kma_service_key.strip())
        mode = (
            "apihub_realtime"
            if api_key
            else "data_go_kr"
            if portal_key
            else "unconfigured"
        )
        return {
            "version": 0,
            "configured": api_key or portal_key,
            "mode": mode,
            "source": (
                "KMA APIHub AWS"
                if api_key
                else "KMA 단기예보"
                if portal_key
                else "KMA"
            ),
            "station": {
                "id": settings.kma_aws_station_id,
                "name": settings.kma_aws_station_name,
                "lat": settings.kma_aws_station_lat,
                "lon": settings.kma_aws_station_lon,
            },
            "rain_1m_mm": None,
            "rain_60m_mm": None,
            "rain_day_mm": None,
            "rain_1h_mm": None,
            "temperature_c": None,
            "humidity_pct": None,
            "wind_ms": None,
            "observed_at": None,
            "received_at": None,
            "age_seconds": None,
            "stale": True,
            "minute_history": [],
            "forecast": [],
            "grid": {
                "configured": api_key or portal_key,
                "source": "KMA",
                "updated_at": None,
                "max_rain_1h_mm": 0.0,
                "points": [],
            },
            "polling": {
                "aws_seconds": settings.kma_aws_poll_seconds,
                "grid_seconds": settings.kma_grid_poll_seconds,
                "forecast_seconds": settings.kma_forecast_poll_seconds,
            },
            "errors": {},
            "message": (
                "KMA_APIHUB_AUTH_KEY 또는 KMA_SERVICE_KEY를 설정하세요."
                if not (api_key or portal_key)
                else "기상청 수집기 시작 대기"
            ),
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="kma-realtime-weather",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            result = copy.deepcopy(self._snapshot)

        observed = result.get("observed_at")
        observed_at = (
            datetime.fromisoformat(observed)
            if observed
            else None
        )
        if observed_at is not None:
            age = max(
                0,
                int((_now_kst() - observed_at).total_seconds()),
            )
            result["age_seconds"] = age
            result["stale"] = (
                age > settings.kma_weather_stale_seconds
            )
        return result

    def grid_snapshot(self) -> dict[str, Any]:
        return self.snapshot().get("grid", {})

    def wait_for_update(
        self,
        last_version: int,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._version <= last_version
                and not self._stop_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
        return self.snapshot()

    def sse(self) -> Iterator[bytes]:
        last_version = -1
        try:
            while not self._stop_event.is_set():
                snapshot = self.wait_for_update(
                    last_version,
                    timeout=15.0,
                )
                version = int(snapshot.get("version", 0))
                if version != last_version:
                    payload = json.dumps(
                        snapshot,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    yield (
                        f"id: {version}\n"
                        "event: weather\n"
                        f"data: {payload}\n\n"
                    ).encode("utf-8")
                    last_version = version
                else:
                    yield b": keep-alive\n\n"
        except GeneratorExit:
            return

    def _publish(
        self,
        *,
        aws: dict[str, Any] | None = None,
        grid: dict[str, Any] | None = None,
        forecast: list[dict[str, Any]] | None = None,
        error_key: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._condition:
            next_snapshot = copy.deepcopy(self._snapshot)

            if aws is not None:
                next_snapshot.update(aws)
                next_snapshot["minute_history"] = copy.deepcopy(
                    self._minute_history
                )
            if grid is not None:
                next_snapshot["grid"] = copy.deepcopy(grid)
            if forecast is not None:
                next_snapshot["forecast"] = copy.deepcopy(forecast)

            errors = dict(next_snapshot.get("errors") or {})
            if error_key:
                if error:
                    errors[error_key] = error
                else:
                    errors.pop(error_key, None)
            next_snapshot["errors"] = errors

            next_snapshot["received_at"] = _iso_kst()
            next_snapshot["configured"] = bool(
                settings.kma_apihub_auth_key.strip()
                or settings.kma_service_key.strip()
            )
            next_snapshot["mode"] = (
                "apihub_realtime"
                if settings.kma_apihub_auth_key.strip()
                else "data_go_kr"
                if settings.kma_service_key.strip()
                else "unconfigured"
            )
            next_snapshot["message"] = (
                "기상청 API허브 AWS 매분자료 실시간 수집"
                if next_snapshot["mode"] == "apihub_realtime"
                else "공공데이터포털 초단기실황 주기 수집"
                if next_snapshot["mode"] == "data_go_kr"
                else "기상청 인증키가 설정되지 않았습니다."
            )

            self._version += 1
            next_snapshot["version"] = self._version
            self._snapshot = next_snapshot
            self._condition.notify_all()

    def _run(self) -> None:
        next_aws = 0.0
        next_grid = 0.0
        next_forecast = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            api_key = settings.kma_apihub_auth_key.strip()
            portal_key = settings.kma_service_key.strip()

            if not api_key and not portal_key:
                self._publish()
                self._stop_event.wait(30.0)
                continue

            if now >= next_aws:
                if api_key:
                    self._poll_aws()
                else:
                    self._poll_portal_current()
                next_aws = (
                    time.monotonic()
                    + max(30, settings.kma_aws_poll_seconds)
                )

            if now >= next_grid:
                if api_key:
                    self._poll_high_resolution_grid()
                else:
                    self._poll_portal_grid()
                next_grid = (
                    time.monotonic()
                    + max(120, settings.kma_grid_poll_seconds)
                )

            if now >= next_forecast:
                self._poll_forecast()
                next_forecast = (
                    time.monotonic()
                    + max(300, settings.kma_forecast_poll_seconds)
                )

            due = min(next_aws, next_grid, next_forecast)
            self._stop_event.wait(
                max(0.25, min(2.0, due - time.monotonic()))
            )

    def _request_text(
        self,
        endpoint: str,
        params: dict[str, Any],
        timeout: int = 12,
        *,
        base_url: str = APIHUB_BASE,
    ) -> str:
        auth_key = settings.kma_apihub_auth_key.strip()
        if not auth_key:
            raise RuntimeError(
                "KMA_APIHUB_AUTH_KEY가 설정되지 않았습니다."
            )

        # Users often paste the URL-encoded key shown by a portal. Passing that
        # string through urlencode() again turns %2B/%2F/%3D into %252B/... and
        # KMA APIHub responds with HTTP 401. Decode once, then let urlencode()
        # perform exactly one encoding pass.
        normalised_key = unquote(auth_key)
        url = (
            f"{base_url}/{endpoint}?"
            + urlencode({**params, "authKey": normalised_key})
        )
        request = Request(
            url,
            headers={
                "User-Agent": "PohangFloodControl/7.3",
                "Accept": "text/plain,*/*",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError(
                    "기상청 APIHub 인증 실패(HTTP %s). "
                    "KMA_APIHUB_AUTH_KEY를 확인하거나 KMA_SERVICE_KEY 폴백을 설정하세요."
                    % exc.code
                ) from exc
            raise

        for encoding in ("utf-8", "euc-kr", "cp949"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def _fetch_aws_rows(self) -> list[dict[str, str]]:
        now = _now_kst()
        history_minutes = 75 if not self._minute_history else 8
        start = now - timedelta(minutes=history_minutes)

        text = self._request_text(
            "nph-aws2_min",
            {
                "tm1": start.strftime("%Y%m%d%H%M"),
                "tm2": now.strftime("%Y%m%d%H%M"),
                "stn": settings.kma_aws_station_id,
                "disp": 1,
                "help": 1,
            },
        )
        rows = parse_apihub_table(text)
        if not rows:
            lowered = text.lower()
            if "error" in lowered or "인증" in text:
                raise RuntimeError(text[:300].strip())
            raise RuntimeError(
                "AWS 매분자료 응답에서 데이터 행을 찾지 못했습니다."
            )
        return rows

    def _fetch_hourly_weather_row(self) -> dict[str, str] | None:
        """Fetch a same-key APIHub fallback for TA/HM/WS.

        A few AWS minute stations temporarily publish rain while leaving one
        or more meteorological fields missing. APIHub's official hourly AWS
        endpoint exposes TA, HM and WS with the same auth key, so it is used
        only to fill fields that remain unavailable after the minute history
        has been searched.
        """
        text = self._request_text(
            "awsh.php",
            {
                "stn": settings.kma_aws_station_id,
                "disp": 1,
                "help": 1,
            },
            timeout=10,
            base_url=APIHUB_URL_BASE,
        )
        rows = parse_apihub_table(text)
        return rows[-1] if rows else None

    @staticmethod
    def _latest_valid_metric(
        history: list[dict[str, Any]],
        field: str,
        latest_at: datetime,
        *,
        max_age_minutes: int = 90,
    ) -> float | None:
        cutoff = latest_at - timedelta(minutes=max_age_minutes)
        for item in reversed(history):
            observed_raw = item.get("observed_at")
            try:
                observed_at = (
                    observed_raw
                    if isinstance(observed_raw, datetime)
                    else datetime.fromisoformat(str(observed_raw))
                )
            except (TypeError, ValueError):
                continue
            if observed_at < cutoff:
                break
            value = _valid_number(item.get(field))
            if value is not None:
                return value
        return None

    def _rows_to_aws(
        self,
        rows: list[dict[str, str]],
    ) -> dict[str, Any]:
        parsed_rows: list[dict[str, Any]] = []

        for row in rows:
            observed_at = _parse_time(row.get("TM"))
            if observed_at is None:
                continue

            parsed_rows.append(
                {
                    "observed_at": observed_at,
                    "rain_day_mm": _pick(
                        row,
                        "RN_DAY",
                        "RN_DAY_QC",
                        "RN_D",
                    ),
                    "rain_60m_api_mm": _pick(
                        row,
                        "RN_HR1",
                        "RN_60M",
                        "RN_1H",
                    ),
                    "temperature_c": _pick(
                        row,
                        "TA",
                        "TA_AVG",
                        "T1H",
                    ),
                    "humidity_pct": _pick(
                        row,
                        "HM",
                        "HM_AVG",
                        "REH",
                    ),
                    "wind_ms": _pick(
                        row,
                        "WS",
                        "WS1",
                        "WS1_AVG",
                        "WSD",
                        "WS_10M",
                    ),
                }
            )

        parsed_rows.sort(
            key=lambda item: item["observed_at"]
        )

        if not parsed_rows:
            raise RuntimeError(
                "유효한 AWS 관측행이 없습니다."
            )

        combined: dict[str, dict[str, Any]] = {}

        # 이전 수집 이력은 observed_at이 문자열일 수도 있고
        # datetime 객체일 수도 있으므로 둘 다 처리합니다.
        for history_item in self._minute_history:
            observed_value = history_item.get("observed_at")

            if not observed_value:
                continue

            observed_key = (
                observed_value.isoformat()
                if isinstance(observed_value, datetime)
                else str(observed_value)
            )

            combined[observed_key] = {
                **history_item,
                "observed_at": observed_key,
                "rain_1m_mm": float(
                    history_item.get("rain_1m_mm") or 0.0
                ),
            }

        previous_day: float | None = None
        previous_date = None

        # 이번 API 응답 구간 전체를 이용해 분 강수량을 계산합니다.
        for item in parsed_rows:
            current_day = item.get("rain_day_mm")
            current_date = item["observed_at"].date()
            minute_rain = 0.0

            if (
                current_day is not None
                and previous_day is not None
            ):
                if (
                    current_date != previous_date
                    or current_day < previous_day
                ):
                    minute_rain = max(0.0, current_day)
                else:
                    minute_rain = max(
                        0.0,
                        current_day - previous_day,
                    )

            observed_key = item["observed_at"].isoformat()

            combined[observed_key] = {
                **item,
                "observed_at": observed_key,
                "rain_1m_mm": round(minute_rain, 3),
            }

            if current_day is not None:
                previous_day = current_day
                previous_date = current_date

        history = sorted(
            combined.values(),
            key=lambda item: item["observed_at"],
        )

        cutoff = _now_kst() - timedelta(hours=3)
        history = [
            item
            for item in history
            if datetime.fromisoformat(
                item["observed_at"]
            ) >= cutoff
        ]

        self._minute_history = history[-180:]

        if not self._minute_history:
            raise RuntimeError(
                "최근 3시간 이내 AWS 관측자료가 없습니다."
            )

        latest = self._minute_history[-1]
        latest_at = datetime.fromisoformat(
            latest["observed_at"]
        )
        hour_cutoff = latest_at - timedelta(minutes=60)

        rain_60m = sum(
            float(item.get("rain_1m_mm") or 0.0)
            for item in self._minute_history
            if datetime.fromisoformat(
                item["observed_at"]
            ) > hour_cutoff
        )

        if (
            latest.get("rain_60m_api_mm") is not None
            and latest["rain_60m_api_mm"] >= 0
        ):
            rain_60m = latest["rain_60m_api_mm"]

        # The newest minute can contain -9 / -99 for selected sensors even
        # though the immediately preceding observations are valid. Carry each
        # weather element forward independently for at most 90 minutes.
        temperature = self._latest_valid_metric(
            self._minute_history,
            "temperature_c",
            latest_at,
        )
        humidity = self._latest_valid_metric(
            self._minute_history,
            "humidity_pct",
            latest_at,
        )
        wind = self._latest_valid_metric(
            self._minute_history,
            "wind_ms",
            latest_at,
        )

        metric_source = "minute"
        if temperature is None or humidity is None or wind is None:
            try:
                hourly = self._fetch_hourly_weather_row()
            except Exception:
                hourly = None
            if hourly:
                if temperature is None:
                    temperature = _pick(hourly, "TA", "TA_AVG", "T1H")
                if humidity is None:
                    humidity = _pick(hourly, "HM", "HM_AVG", "REH")
                if wind is None:
                    wind = _pick(hourly, "WS", "WS1_AVG", "WS1", "WSD")
                metric_source = "minute+hourly"

        return {
            "source": (
                "KMA APIHub AWS"
                if metric_source == "minute"
                else "KMA APIHub AWS · hourly fill"
            ),
            "station": {
                "id": settings.kma_aws_station_id,
                "name": settings.kma_aws_station_name,
                "lat": settings.kma_aws_station_lat,
                "lon": settings.kma_aws_station_lon,
            },
            "rain_1m_mm": round(
                float(latest.get("rain_1m_mm") or 0.0),
                3,
            ),
            "rain_60m_mm": round(float(rain_60m), 2),
            "rain_1h_mm": round(float(rain_60m), 2),
            "rain_day_mm": latest.get("rain_day_mm"),
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "wind_ms": wind,
            "observed_at": latest["observed_at"],
        }

    def _poll_aws(self) -> None:
        try:
            aws = self._rows_to_aws(self._fetch_aws_rows())
            self._last_good_aws = aws
            self._publish(
                aws=aws,
                error_key="aws",
                error=None,
            )
        except Exception as exc:
            # If both keys are configured, an expired/incorrect APIHub key must
            # not take weather offline. Use the public-data portal current
            # observation as a live fallback and retain the APIHub error only
            # when that fallback also fails.
            if settings.kma_service_key.strip():
                try:
                    self._poll_portal_current()
                    return
                except Exception:
                    pass
            self._publish(
                aws=self._last_good_aws,
                error_key="aws",
                error=str(exc),
            )

    def _poll_portal_current(self) -> None:
        try:
            current = get_weather()
            observed_at = current.get("observed_at")
            aws = {
                "source": "KMA 초단기실황",
                "rain_1m_mm": None,
                "rain_60m_mm": current.get("rain_1h_mm"),
                "rain_1h_mm": current.get("rain_1h_mm"),
                "rain_day_mm": None,
                "temperature_c": current.get("temperature_c"),
                "humidity_pct": current.get("humidity_pct"),
                "wind_ms": current.get("wind_ms"),
                "observed_at": observed_at,
            }
            self._last_good_aws = aws
            self._publish(
                aws=aws,
                forecast=current.get("forecast") or None,
                error_key="aws",
                error=None,
            )
        except Exception as exc:
            self._publish(
                aws=self._last_good_aws,
                error_key="aws",
                error=str(exc),
            )

    def _fetch_grid_point(
        self,
        point: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now_kst()
        end = now - timedelta(minutes=8)
        end = end.replace(
            minute=(end.minute // 5) * 5,
            second=0,
            microsecond=0,
        )
        start = end - timedelta(minutes=20)

        text = self._request_text(
            "nph-sfc_obs_nc_pt_api",
            {
                "obs": "rn_60m",
                "tm1": start.strftime("%Y%m%d%H%M"),
                "tm2": end.strftime("%Y%m%d%H%M"),
                "itv": 5,
                "lon": point["lon"],
                "lat": point["lat"],
            },
            timeout=15,
        )
        observed_at, rain = parse_grid_point_text(text)
        if observed_at is None or rain is None:
            raise RuntimeError(
                f"{point['name']} 500m 격자 응답 파싱 실패"
            )

        return {
            **point,
            "rain_1h_mm": round(rain, 2),
            "observed_at": observed_at.isoformat(),
        }

    def _poll_high_resolution_grid(self) -> None:
        points: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        self._fetch_grid_point,
                        point,
                    ): point
                    for point in POHANG_RAIN_SAMPLES
                }
                for future in as_completed(futures):
                    point = futures[future]
                    try:
                        points.append(future.result())
                    except Exception as exc:
                        errors.append(
                            f"{point['name']}: {exc}"
                        )

            points.sort(key=lambda item: item["id"])
            if not points:
                raise RuntimeError(
                    "; ".join(errors)
                    or "500m 강수 격자 데이터가 없습니다."
                )

            observed_times = [
                point["observed_at"]
                for point in points
                if point.get("observed_at")
            ]
            grid = {
                "configured": True,
                "source": "KMA APIHub 500m rn_60m",
                "updated_at": (
                    max(observed_times)
                    if observed_times
                    else _iso_kst()
                ),
                "max_rain_1h_mm": max(
                    float(point["rain_1h_mm"])
                    for point in points
                ),
                "points": points,
                "partial_errors": errors,
            }
            self._last_good_grid = grid
            self._publish(
                grid=grid,
                error_key="grid",
                error="; ".join(errors) if errors else None,
            )
        except Exception as exc:
            self._publish(
                grid=self._last_good_grid,
                error_key="grid",
                error=str(exc),
            )

    def _poll_portal_grid(self) -> None:
        try:
            grid = get_pohang_rainfall_grid()
            self._last_good_grid = grid
            self._publish(
                grid=grid,
                error_key="grid",
                error=None,
            )
        except Exception as exc:
            self._publish(
                grid=self._last_good_grid,
                error_key="grid",
                error=str(exc),
            )

    def _poll_forecast(self) -> None:
        if not settings.kma_service_key.strip():
            return
        try:
            weather = get_weather()
            forecast = weather.get("forecast") or []
            self._last_good_forecast = forecast
            self._publish(
                forecast=forecast,
                error_key="forecast",
                error=None,
            )
        except Exception as exc:
            self._publish(
                forecast=self._last_good_forecast,
                error_key="forecast",
                error=str(exc),
            )


weather_service = RealtimeWeatherService()
