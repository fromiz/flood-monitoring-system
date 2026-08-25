from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, unquote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import settings


try:
    KST = ZoneInfo("Asia/Seoul")
except ZoneInfoNotFoundError:
    # Windows에 IANA timezone DB가 없을 때도 서버가 시작되도록 고정 UTC+9 사용
    KST = timezone(timedelta(hours=9), name="KST")
_cache: dict[str, tuple[float, Any]] = {}

# 포항 운영 범위의 기상청 동네예보 격자 표본점입니다.
# 각 좌표는 호출 전에 기상청 DFS 격자(nx, ny)로 변환됩니다.
POHANG_RAIN_SAMPLES = [
    {"id": "PH-A", "name": "포항 남서", "lat": 35.84, "lon": 129.18},
    {"id": "PH-B", "name": "포항 남동", "lat": 35.84, "lon": 129.43},
    {"id": "PH-C", "name": "포항 도심 서부", "lat": 36.00, "lon": 129.20},
    {"id": "PH-D", "name": "포항 도심", "lat": 36.02, "lon": 129.35},
    {"id": "PH-E", "name": "포항 도심 동부", "lat": 36.02, "lon": 129.50},
    {"id": "PH-F", "name": "포항 북서", "lat": 36.16, "lon": 129.20},
    {"id": "PH-G", "name": "포항 북부", "lat": 36.18, "lon": 129.38},
    {"id": "PH-H", "name": "포항 북동", "lat": 36.27, "lon": 129.48},
]


def _cached(key: str, ttl: int, loader):
    now = time.time()
    cached = _cache.get(key)
    if cached and now - cached[0] < ttl:
        return cached[1]
    value = loader()
    _cache[key] = (now, value)
    return value


def _service_key() -> str:
    # 공공데이터포털에서 받은 일반 인증키와 URL 인코딩 인증키 모두 허용합니다.
    return unquote((settings.kma_service_key or "").strip())


def _request_json(endpoint: str, params: dict[str, Any]) -> dict:
    key = _service_key()
    if not key:
        raise RuntimeError("KMA_SERVICE_KEY가 설정되지 않았습니다.")

    query = {
        "serviceKey": key,
        "pageNo": 1,
        "numOfRows": 1000,
        "dataType": "JSON",
        **params,
    }
    url = (
        "https://apis.data.go.kr/1360000/"
        f"VilageFcstInfoService_2.0/{endpoint}?"
        + urlencode(query)
    )
    request = Request(
        url,
        headers={
            "User-Agent": "PohangFloodControl/7.2",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=8) as response:
        data = json.load(response)

    header = data.get("response", {}).get("header", {})
    result_code = str(header.get("resultCode", ""))
    if result_code not in {"00", "0"}:
        raise RuntimeError(
            f"기상청 응답 오류 {result_code}: "
            f"{header.get('resultMsg', '알 수 없는 오류')}"
        )
    return data


def _items(data: dict) -> list[dict]:
    body = data.get("response", {}).get("body", {})
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        return [items]
    return list(items or [])


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or "없음" in text:
        return 0.0

    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return default

    number = float(match.group())
    if "미만" in text:
        return number / 2.0
    return number


def _precipitation_name(value: Any) -> str:
    code = int(_number(value, 0))
    return {
        0: "없음",
        1: "비",
        2: "비/눈",
        3: "눈",
        5: "빗방울",
        6: "빗방울/눈날림",
        7: "눈날림",
    }.get(code, f"코드 {code}")


def latlon_to_kma_grid(lat: float, lon: float) -> tuple[int, int]:
    """기상청 동네예보 Lambert Conformal Conic 격자로 변환합니다."""
    re_km = 6371.00877
    grid_km = 5.0
    slat1 = 30.0
    slat2 = 60.0
    olon = 126.0
    olat = 38.0
    xo = 43.0
    yo = 136.0
    degrad = math.pi / 180.0

    re_grid = re_km / grid_km
    slat1_rad = slat1 * degrad
    slat2_rad = slat2 * degrad
    olon_rad = olon * degrad
    olat_rad = olat * degrad

    sn = (
        math.log(math.cos(slat1_rad) / math.cos(slat2_rad))
        / math.log(
            math.tan(math.pi * 0.25 + slat2_rad * 0.5)
            / math.tan(math.pi * 0.25 + slat1_rad * 0.5)
        )
    )
    sf = (
        math.tan(math.pi * 0.25 + slat1_rad * 0.5) ** sn
        * math.cos(slat1_rad)
        / sn
    )
    ro = (
        re_grid
        * sf
        / math.tan(math.pi * 0.25 + olat_rad * 0.5) ** sn
    )
    ra = (
        re_grid
        * sf
        / math.tan(
            math.pi * 0.25 + lat * degrad * 0.5
        ) ** sn
    )
    theta = lon * degrad - olon_rad
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    x = int(ra * math.sin(theta) + xo + 0.5)
    y = int(ro - ra * math.cos(theta) + yo + 0.5)
    return x, y


def _nowcast_candidates() -> list[tuple[str, str]]:
    now = datetime.now(KST)
    candidates: list[tuple[str, str]] = []
    # 초단기실황 자료가 API에 반영되는 지연을 고려해 최근 발표시각을 순서대로 시도합니다.
    for minutes in (40, 70, 100, 130):
        target = now - timedelta(minutes=minutes)
        candidates.append(
            (target.strftime("%Y%m%d"), target.strftime("%H00"))
        )
    return list(dict.fromkeys(candidates))


def _forecast_candidates() -> list[tuple[str, str]]:
    now = datetime.now(KST)
    candidates: list[tuple[str, str]] = []
    for minutes in (45, 75, 105, 135):
        target = now - timedelta(minutes=minutes)
        base = target.replace(minute=30, second=0, microsecond=0)
        candidates.append(
            (base.strftime("%Y%m%d"), base.strftime("%H%M"))
        )
    return list(dict.fromkeys(candidates))


def _fetch_nowcast(nx: int, ny: int) -> dict:
    last_error: Exception | None = None
    for base_date, base_time in _nowcast_candidates():
        try:
            data = _request_json(
                "getUltraSrtNcst",
                {
                    "base_date": base_date,
                    "base_time": base_time,
                    "nx": nx,
                    "ny": ny,
                },
            )
            items = _items(data)
            if not items:
                raise RuntimeError("기상청 실황 자료가 비어 있습니다.")

            values = {
                str(item.get("category")): item.get("obsrValue")
                for item in items
            }
            return {
                "base_date": base_date,
                "base_time": base_time,
                "rain_1h_mm": _number(values.get("RN1")),
                "temperature_c": _number(values.get("T1H")),
                "humidity_pct": _number(values.get("REH")),
                "wind_ms": _number(values.get("WSD")),
                "precipitation_type": _precipitation_name(
                    values.get("PTY")
                ),
                "observed_at": (
                    f"{base_date[:4]}-{base_date[4:6]}-"
                    f"{base_date[6:8]}T{base_time[:2]}:"
                    f"{base_time[2:]}:00+09:00"
                ),
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"초단기실황 조회 실패: {last_error}")


def _fetch_forecast(nx: int, ny: int) -> list[dict]:
    last_error: Exception | None = None
    for base_date, base_time in _forecast_candidates():
        try:
            data = _request_json(
                "getUltraSrtFcst",
                {
                    "base_date": base_date,
                    "base_time": base_time,
                    "nx": nx,
                    "ny": ny,
                },
            )
            rows = _items(data)
            if not rows:
                raise RuntimeError("기상청 초단기예보 자료가 비어 있습니다.")

            grouped: dict[str, dict] = {}
            for row in rows:
                date = str(row.get("fcstDate", ""))
                time_text = str(row.get("fcstTime", "")).zfill(4)
                if not date or not time_text:
                    continue
                key = f"{date}{time_text}"
                point = grouped.setdefault(
                    key,
                    {
                        "forecast_at": (
                            f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                            f"T{time_text[:2]}:{time_text[2:]}:"
                            "00+09:00"
                        )
                    },
                )
                category = str(row.get("category", ""))
                value = row.get("fcstValue")
                if category == "RN1":
                    point["rain_1h_mm"] = _number(value)
                elif category == "T1H":
                    point["temperature_c"] = _number(value)
                elif category == "REH":
                    point["humidity_pct"] = _number(value)
                elif category == "WSD":
                    point["wind_ms"] = _number(value)
                elif category == "PTY":
                    point["precipitation_type"] = (
                        _precipitation_name(value)
                    )

            now = datetime.now(KST)
            result = []
            for key in sorted(grouped):
                point = grouped[key]
                try:
                    dt = datetime.fromisoformat(
                        point["forecast_at"]
                    )
                except ValueError:
                    continue
                if dt >= now - timedelta(minutes=15):
                    point.setdefault("rain_1h_mm", 0.0)
                    point.setdefault("precipitation_type", "없음")
                    result.append(point)
            return result[:6]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"초단기예보 조회 실패: {last_error}")


def get_weather() -> dict:
    def load():
        if not _service_key():
            return {
                "configured": False,
                "source": "KMA",
                "station": settings.kma_station_name,
                "grid": {
                    "nx": settings.kma_nx,
                    "ny": settings.kma_ny,
                },
                "rain_1h_mm": None,
                "temperature_c": None,
                "humidity_pct": None,
                "wind_ms": None,
                "precipitation_type": None,
                "observed_at": None,
                "forecast": [],
                "message": (
                    "실제 기상청 연동을 위해 .env의 "
                    "KMA_SERVICE_KEY를 입력하세요."
                ),
            }

        current = _fetch_nowcast(
            settings.kma_nx,
            settings.kma_ny,
        )
        try:
            forecast = _fetch_forecast(
                settings.kma_nx,
                settings.kma_ny,
            )
        except Exception as exc:
            forecast = []
            current["forecast_error"] = str(exc)

        return {
            "configured": True,
            "source": "KMA",
            "station": settings.kma_station_name,
            "grid": {
                "nx": settings.kma_nx,
                "ny": settings.kma_ny,
            },
            **current,
            "forecast": forecast,
            "message": "기상청 초단기실황·초단기예보",
        }

    try:
        return _cached(
            "weather:v72",
            settings.weather_refresh_seconds,
            load,
        )
    except Exception as exc:
        return {
            "configured": bool(_service_key()),
            "source": "error",
            "station": settings.kma_station_name,
            "grid": {
                "nx": settings.kma_nx,
                "ny": settings.kma_ny,
            },
            "rain_1h_mm": None,
            "temperature_c": None,
            "humidity_pct": None,
            "wind_ms": None,
            "precipitation_type": None,
            "observed_at": datetime.now(KST).isoformat(
                timespec="minutes"
            ),
            "forecast": [],
            "message": f"기상청 연동 오류: {exc}",
        }


def get_pohang_rainfall_grid() -> dict:
    def load():
        if not _service_key():
            return {
                "configured": False,
                "source": "KMA",
                "updated_at": None,
                "points": [],
                "message": (
                    "KMA_SERVICE_KEY가 없어 실강수 격자를 "
                    "불러오지 않았습니다."
                ),
            }

        grid_groups: dict[tuple[int, int], list[dict]] = {}
        for sample in POHANG_RAIN_SAMPLES:
            nx, ny = latlon_to_kma_grid(
                sample["lat"],
                sample["lon"],
            )
            item = {**sample, "nx": nx, "ny": ny}
            grid_groups.setdefault((nx, ny), []).append(item)

        results: dict[tuple[int, int], dict] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_fetch_nowcast, nx, ny): (nx, ny)
                for nx, ny in grid_groups
            }
            for future in as_completed(futures):
                grid = futures[future]
                try:
                    results[grid] = future.result()
                except Exception as exc:
                    results[grid] = {"error": str(exc)}

        points = []
        observed_times = []
        for grid, samples in grid_groups.items():
            weather = results.get(grid, {})
            for sample in samples:
                point = {**sample}
                if "error" in weather:
                    point.update(
                        {
                            "rain_1h_mm": None,
                            "temperature_c": None,
                            "humidity_pct": None,
                            "wind_ms": None,
                            "precipitation_type": None,
                            "observed_at": None,
                            "error": weather["error"],
                        }
                    )
                else:
                    point.update(weather)
                    if weather.get("observed_at"):
                        observed_times.append(
                            weather["observed_at"]
                        )
                points.append(point)

        valid = [
            point
            for point in points
            if point.get("rain_1h_mm") is not None
        ]
        return {
            "configured": True,
            "source": "KMA",
            "updated_at": (
                max(observed_times) if observed_times else None
            ),
            "max_rain_1h_mm": max(
                (
                    float(point["rain_1h_mm"])
                    for point in valid
                ),
                default=0.0,
            ),
            "points": points,
            "message": (
                f"기상청 포항 격자 {len(valid)}/"
                f"{len(points)}개 조회"
            ),
        }

    try:
        return _cached(
            "pohang-rain-grid:v72",
            settings.weather_refresh_seconds,
            load,
        )
    except Exception as exc:
        return {
            "configured": bool(_service_key()),
            "source": "error",
            "updated_at": datetime.now(KST).isoformat(
                timespec="minutes"
            ),
            "max_rain_1h_mm": 0.0,
            "points": [],
            "message": f"포항 강수 격자 오류: {exc}",
        }



def _water_value(item: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = item.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            if match:
                return float(match.group())
    return None


def _first_text(item: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = item.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _find_item_list(value: Any) -> list[dict[str, Any]]:
    """Find the most likely sensor item list in common public-API envelopes."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []

    sensor_keys = {
        "level_m", "water_level_m", "level", "wal", "wl",
        "wlobscd", "wlobsCd", "wlobsnm", "station_id",
        "stationId", "station_name", "stationName", "flux",
    }
    if sensor_keys.intersection(value.keys()):
        return [value]

    for key in (
        "sensors", "items", "item", "records", "data", "list",
        "result", "results",
    ):
        child = value.get(key)
        found = _find_item_list(child)
        if found:
            return found

    for key in ("response", "body", "header"):
        child = value.get(key)
        found = _find_item_list(child)
        if found:
            return found

    for child in value.values():
        found = _find_item_list(child)
        if found:
            return found
    return []


def _xml_to_dict(element: ET.Element) -> dict[str, Any]:
    children = list(element)
    if not children:
        return {element.tag.split("}")[-1]: (element.text or "").strip()}
    result: dict[str, Any] = {}
    for child in children:
        tag = child.tag.split("}")[-1]
        grandchildren = list(child)
        if grandchildren:
            child_value = _xml_to_dict(child)
            child_value = child_value.get(tag, child_value)
        else:
            child_value = (child.text or "").strip()
        if tag in result:
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(child_value)
        else:
            result[tag] = child_value
    return {element.tag.split("}")[-1]: result}


def _decode_api_payload(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return _xml_to_dict(ET.fromstring(text))
        except ET.ParseError as exc:
            raise RuntimeError(f"수위 API 응답을 JSON/XML로 해석하지 못했습니다: {exc}") from exc


def _normalise_water_sensor(item: dict[str, Any], prefix: str) -> dict[str, Any]:
    level_m = _water_value(
        item,
        "level_m", "water_level_m", "waterLevel", "water_level",
        "level", "wl", "wal", "wlvalue", "wl_value",
    )
    level_cm = _water_value(item, "level_cm", "water_level_cm")
    if level_m is None and level_cm is not None:
        level_m = level_cm / 100.0

    return {
        "id": _first_text(
            item,
            "id", "station_id", "stationId", "station_code", "stationCode",
            "wlobscd", "wlobsCd", "obscd", "code",
        ) or f"{prefix}-{abs(hash(json.dumps(item, ensure_ascii=False, sort_keys=True))) % 100000}",
        "name": _first_text(
            item,
            "name", "station_name", "stationName", "wlobsnm", "obsnm",
            "site_name", "siteName",
        ) or "이름 없는 수위계",
        "level_m": level_m,
        "flow_cms": _water_value(
            item,
            "flow_cms", "flow", "flux", "discharge", "qty",
        ),
        "warning_m": _water_value(
            item,
            "warning_m", "warningLevel", "warning_level", "attention_level",
        ),
        "danger_m": _water_value(
            item,
            "danger_m", "dangerLevel", "danger_level", "alarm_level",
        ),
        "lat": _water_value(item, "lat", "latitude", "y"),
        "lon": _water_value(item, "lon", "lng", "longitude", "x"),
        "observed_at": _first_text(
            item,
            "observed_at", "observedAt", "obs_time", "obsTime", "tm",
            "datetime", "dateTime", "obsrdate", "measure_time",
        ),
        "raw": item,
    }


def _river_request_url(template: str, station_code: str = "") -> tuple[str, dict[str, str]]:
    now = datetime.now(KST)
    replacements = {
        "service_key": quote(unquote((settings.river_api_key or "").strip()), safe=""),
        "station_code": quote(station_code, safe=""),
        "start_date": (now - timedelta(days=7)).strftime("%Y%m%d"),
        "end_date": now.strftime("%Y%m%d"),
        "today": now.strftime("%Y%m%d"),
    }
    url = template
    for key, value in replacements.items():
        url = url.replace("{" + key + "}", value)

    headers = {
        "User-Agent": "PohangFloodControl/8.4",
        "Accept": "application/json, application/xml, text/xml, */*",
    }
    api_key = unquote((settings.river_api_key or "").strip())
    header_name = (settings.river_api_key_header or "").strip()
    key_param = (settings.river_api_key_param or "serviceKey").strip()

    if api_key and header_name:
        headers[header_name] = api_key
    elif api_key and "{service_key}" not in template and key_param:
        separator = "&" if "?" in url else "?"
        url += separator + urlencode({key_param: api_key})

    return url, headers


def get_river_levels() -> dict:
    def load():
        template = (settings.river_api_url or "").strip()
        if template:
            station_codes = [
                code.strip()
                for code in str(settings.river_station_codes or "").split(",")
                if code.strip()
            ]
            if "{station_code}" in template and not station_codes:
                raise RuntimeError(
                    "RIVER_API_URL에 {station_code}가 있지만 "
                    "RIVER_STATION_CODES가 비어 있습니다."
                )

            requests = station_codes if "{station_code}" in template else [""]
            sensors: list[dict[str, Any]] = []
            for station_code in requests:
                url, headers = _river_request_url(template, station_code)
                with urlopen(Request(url, headers=headers), timeout=12) as response:
                    payload = _decode_api_payload(response.read())
                items = _find_item_list(payload)
                if not items and isinstance(payload, dict):
                    # Some single-station APIs return one flat record.
                    candidate = next(iter(payload.values()), payload)
                    if isinstance(candidate, dict):
                        items = [candidate]
                sensors.extend(
                    _normalise_water_sensor(item, "RV")
                    for item in items
                )

            # If a station history endpoint returns many rows, keep the newest row per station.
            latest: dict[str, dict[str, Any]] = {}
            for sensor in sensors:
                key = str(sensor.get("id") or sensor.get("name"))
                current = latest.get(key)
                if current is None or str(sensor.get("observed_at") or "") >= str(current.get("observed_at") or ""):
                    latest[key] = sensor

            return {
                "source": "api",
                "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
                "sensors": list(latest.values()),
                "message": f"하천 수위 API {len(latest)}개 지점",
            }

        t = time.time() / 120
        sensors = [
            {
                "id": "RV-PH-01",
                "name": "형산강 포항 구간",
                "level_m": round(0.82 + 0.12 * math.sin(t), 2),
                "flow_cms": round(21.0 + 3.0 * math.sin(t + 0.7), 1),
                "warning_m": 2.5,
                "danger_m": 3.5,
                "lat": 36.005,
                "lon": 129.361,
                "observed_at": datetime.now(KST).isoformat(timespec="minutes"),
            },
            {
                "id": "RV-PH-02",
                "name": "냉천 오천 구간",
                "level_m": round(0.48 + 0.08 * math.sin(t + 1.1), 2),
                "flow_cms": round(8.0 + 1.8 * math.sin(t + 1.6), 1),
                "warning_m": 1.8,
                "danger_m": 2.6,
                "lat": 35.963,
                "lon": 129.407,
                "observed_at": datetime.now(KST).isoformat(timespec="minutes"),
            },
            {
                "id": "RV-PH-03",
                "name": "곡강천 흥해 구간",
                "level_m": round(0.57 + 0.09 * math.sin(t + 2.0), 2),
                "flow_cms": round(10.0 + 2.1 * math.sin(t + 2.3), 1),
                "warning_m": 2.0,
                "danger_m": 2.8,
                "lat": 36.109,
                "lon": 129.346,
                "observed_at": datetime.now(KST).isoformat(timespec="minutes"),
            },
        ]
        return {
            "source": "demo",
            "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
            "sensors": sensors,
            "message": "RIVER_API_URL 미설정 — 데모값",
        }

    try:
        return _cached(
            "river",
            settings.river_refresh_seconds,
            load,
        )
    except Exception as exc:
        return {
            "source": "error",
            "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
            "sensors": [],
            "message": str(exc),
        }

def get_sewer_levels() -> dict:
    def load():
        if settings.sewer_api_url:
            headers = {"User-Agent": "FloodMonitor/1.0"}
            if settings.sewer_api_key:
                headers["Authorization"] = (
                    f"Bearer {settings.sewer_api_key}"
                )
            with urlopen(
                Request(
                    settings.sewer_api_url,
                    headers=headers,
                ),
                timeout=8,
            ) as response:
                raw = json.load(response)
            sensors = raw.get(
                "sensors",
                raw if isinstance(raw, list) else [],
            )
            return {
                "source": "api",
                "updated_at": datetime.now().isoformat(
                    timespec="minutes"
                ),
                "sensors": sensors,
            }

        t = time.time() / 90
        sensors = [
            {
                "id": "SW-PH-01",
                "name": "포항 도심 우수관",
                "level_m": round(
                    0.42 + 0.10 * math.sin(t),
                    2,
                ),
                "warning_m": 0.85,
                "danger_m": 1.10,
                "lat": 36.019,
                "lon": 129.343,
            },
            {
                "id": "SW-PH-02",
                "name": "형산강 유입관",
                "level_m": round(
                    0.58 + 0.13 * math.sin(t + 1.4),
                    2,
                ),
                "warning_m": 0.90,
                "danger_m": 1.20,
                "lat": 36.005,
                "lon": 129.355,
            },
            {
                "id": "SW-PH-03",
                "name": "흥해 간선관",
                "level_m": round(
                    0.35 + 0.08 * math.sin(t + 2.2),
                    2,
                ),
                "warning_m": 0.80,
                "danger_m": 1.05,
                "lat": 36.112,
                "lon": 129.343,
            },
        ]
        return {
            "source": "demo",
            "updated_at": datetime.now().isoformat(
                timespec="minutes"
            ),
            "sensors": sensors,
            "message": "SEWER_API_URL 미설정 — 데모값",
        }

    try:
        return _cached(
            "sewer",
            settings.sewer_refresh_seconds,
            load,
        )
    except Exception as exc:
        return {
            "source": "error",
            "updated_at": datetime.now().isoformat(
                timespec="minutes"
            ),
            "sensors": [],
            "message": str(exc),
        }
