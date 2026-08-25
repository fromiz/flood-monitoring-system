from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import heapq
import hashlib
import json
import math
from pathlib import Path
from threading import RLock
import time
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
from PIL import Image
import requests

from .config import settings
from .flood_map import depth_to_level, level_to_depth_cm


class DemUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class DemMetadata:
    template: str
    tile_size: int
    encoding: str
    minzoom: int
    maxzoom: int


class DemTileStore:
    """Mapbox Terrain-RGB/Terrarium DEM tile downloader and sampler."""

    def __init__(self) -> None:
        self.cache_dir = Path(settings.dem_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._metadata: DemMetadata | None = None
        self._arrays: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._failures: dict[tuple[int, int, int], float] = {}
        self._metadata_failure_until = 0.0
        self._last_error = ""
        self._download_count = 0
        self._cache_hit_count = 0
        self._zoom_fallback_count = 0
        self._last_requested_zoom: int | None = None
        self._last_effective_zoom: int | None = None

    @staticmethod
    def _tile_fraction(lon: float, lat: float, zoom: int) -> tuple[float, float]:
        lat = max(-85.05112878, min(85.05112878, float(lat)))
        n = 2.0 ** zoom
        x = (float(lon) + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        y = (
            1.0
            - math.asinh(math.tan(lat_rad)) / math.pi
        ) / 2.0 * n
        return x, y

    @staticmethod
    def _tile_bounds(zoom: int, x: int, y: int) -> tuple[float, float, float, float]:
        n = 2.0 ** zoom
        west = x / n * 360.0 - 180.0
        east = (x + 1) / n * 360.0 - 180.0

        def tile_lat(tile_y: int) -> float:
            value = math.pi * (1.0 - 2.0 * tile_y / n)
            return math.degrees(math.atan(math.sinh(value)))

        north = tile_lat(y)
        south = tile_lat(y + 1)
        return west, south, east, north

    @staticmethod
    def _source_host(template: str) -> str:
        return (urlparse(template).hostname or "").lower()

    @classmethod
    def _effective_maxzoom(
        cls,
        template: str,
        declared_maxzoom: int,
    ) -> int:
        """
        Pohang uses Mapterhorn's global 30 m terrain coverage, which is
        available through z12. Clamp this project to z12 even when stale
        environment settings request z13 or a TileJSON advertises a wider
        regional maximum.
        """
        if "mapterhorn.com" in cls._source_host(template):
            return min(int(declared_maxzoom), 12)
        return int(declared_maxzoom)

    def _metadata_from_settings(self) -> DemMetadata | None:
        template = settings.dem_tile_url_template.strip()
        if not template:
            return None
        template_host = (
            urlparse(template).hostname or ""
        ).lower()
        encoding = (
            "terrarium"
            if "mapterhorn.com" in template_host
            else settings.dem_encoding.strip().lower()
            or "terrarium"
        )
        return DemMetadata(
            template=template,
            tile_size=max(256, int(settings.dem_tile_size)),
            encoding=encoding,
            minzoom=0,
            maxzoom=self._effective_maxzoom(
                template,
                max(0, int(settings.dem_zoom)),
            ),
        )

    def metadata(self, force: bool = False) -> DemMetadata:
        with self._lock:
            if self._metadata is not None and not force:
                return self._metadata

        configured = self._metadata_from_settings()
        if configured is not None:
            with self._lock:
                self._metadata = configured
            return configured

        if time.time() < self._metadata_failure_until and not force:
            raise DemUnavailable(self._last_error or "DEM 메타데이터 재시도 대기 중")

        url = settings.dem_tilejson_url.strip()
        if not url:
            raise DemUnavailable("DEM_TILEJSON_URL 또는 DEM_TILE_URL_TEMPLATE가 필요합니다.")

        try:
            response = requests.get(
                url,
                timeout=max(5, settings.dem_request_timeout_seconds),
                headers={"User-Agent": "PohangFloodControl/8.3"},
            )
            response.raise_for_status()
            payload = response.json()
            tiles = payload.get("tiles") or []
            if not tiles:
                raise ValueError("tilejson에 tiles 항목이 없습니다.")
            template = str(tiles[0])
            if template.startswith("//"):
                template = "https:" + template
            elif not template.startswith(("http://", "https://")):
                template = urljoin(url, template)

            declared_encoding = str(
                payload.get("encoding") or ""
            ).strip().lower()
            template_host = (
                urlparse(template).hostname or ""
            ).lower()

            # Mapterhorn은 512px Terrarium RGB 고도 타일입니다.
            # 기존 .env에 DEM_ENCODING=mapbox가 남아 있어도 자동 보정합니다.
            if "mapterhorn.com" in template_host:
                resolved_encoding = "terrarium"
            elif declared_encoding in {
                "terrarium",
                "mapzen",
                "mapbox",
            }:
                resolved_encoding = declared_encoding
            else:
                resolved_encoding = (
                    settings.dem_encoding.strip().lower()
                    or "terrarium"
                )

            metadata = DemMetadata(
                template=template,
                tile_size=int(
                    payload.get("tileSize")
                    or payload.get("tile_size")
                    or settings.dem_tile_size
                ),
                encoding=resolved_encoding,
                minzoom=int(payload.get("minzoom") or 0),
                maxzoom=self._effective_maxzoom(
                    template,
                    int(
                        payload.get("maxzoom")
                        or settings.dem_zoom
                    ),
                ),
            )
            with self._lock:
                self._metadata = metadata
                self._last_error = ""
            return metadata
        except Exception as exc:
            self._last_error = f"DEM tilejson 조회 실패: {exc}"
            self._metadata_failure_until = time.time() + 30
            raise DemUnavailable(self._last_error) from exc

    def _tile_url(self, metadata: DemMetadata, zoom: int, x: int, y: int) -> str:
        return (
            metadata.template
            .replace("{z}", str(zoom))
            .replace("{x}", str(x))
            .replace("{y}", str(y))
            .replace("{-y}", str((1 << zoom) - 1 - y))
        )

    def _cache_path(self, metadata: DemMetadata, zoom: int, x: int, y: int) -> Path:
        parsed = urlparse(self._tile_url(metadata, zoom, x, y))
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in {".png", ".webp", ".jpg", ".jpeg"}:
            suffix = ".tile"
        return self.cache_dir / "tiles" / str(zoom) / str(x) / f"{y}{suffix}"

    @staticmethod
    def _decode(content: bytes, encoding: str) -> np.ndarray:
        image = Image.open(BytesIO(content)).convert("RGB")
        rgb = np.asarray(image, dtype=np.float32)
        if encoding in {"terrarium", "mapzen"}:
            return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
        return (
            rgb[..., 0] * 65536.0
            + rgb[..., 1] * 256.0
            + rgb[..., 2]
        ) * 0.1 - 10000.0

    def _load_tile(self, zoom: int, x: int, y: int) -> np.ndarray:
        key = (zoom, x, y)
        with self._lock:
            cached = self._arrays.get(key)
            if cached is not None:
                self._arrays.move_to_end(key)
                self._cache_hit_count += 1
                return cached
            failed_at = self._failures.get(key)
            if failed_at and time.time() - failed_at < 30:
                raise DemUnavailable(f"DEM 타일 {zoom}/{x}/{y} 재시도 대기 중")

        metadata = self.metadata()
        path = self._cache_path(metadata, zoom, x, y)
        content: bytes
        try:
            if path.exists() and path.stat().st_size > 100:
                content = path.read_bytes()
                self._cache_hit_count += 1
            else:
                url = self._tile_url(metadata, zoom, x, y)
                response = requests.get(
                    url,
                    timeout=max(5, settings.dem_request_timeout_seconds),
                    headers={"User-Agent": "PohangFloodControl/8.3"},
                )
                response.raise_for_status()
                content = response.content
                if len(content) < 100:
                    raise ValueError("빈 DEM 타일")
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix + ".tmp")
                temp.write_bytes(content)
                temp.replace(path)
                self._download_count += 1

            array = self._decode(content, metadata.encoding)
            if array.ndim != 2 or array.size == 0:
                raise ValueError("DEM 타일 디코딩 결과가 올바르지 않습니다.")
            with self._lock:
                self._arrays[key] = array
                self._arrays.move_to_end(key)
                while len(self._arrays) > max(8, settings.dem_memory_tile_count):
                    self._arrays.popitem(last=False)
                self._failures.pop(key, None)
                self._last_error = ""
            return array
        except Exception as exc:
            with self._lock:
                self._failures[key] = time.time()
                self._last_error = f"DEM 타일 {zoom}/{x}/{y} 조회 실패: {exc}"
            raise DemUnavailable(self._last_error) from exc

    def _sample_tile_value(
        self,
        lon: float,
        lat: float,
        zoom: int,
    ) -> float:
        tx, ty = self._tile_fraction(lon, lat, zoom)
        x = int(math.floor(tx))
        y = int(math.floor(ty))
        array = self._load_tile(zoom, x, y)

        height, width = array.shape
        px = (tx - x) * width - 0.5
        py = (ty - y) * height - 0.5
        px = max(0.0, min(width - 1.001, px))
        py = max(0.0, min(height - 1.001, py))

        x0 = int(math.floor(px))
        y0 = int(math.floor(py))
        x1 = min(width - 1, x0 + 1)
        y1 = min(height - 1, y0 + 1)
        fx = px - x0
        fy = py - y0

        value = (
            array[y0, x0] * (1 - fx) * (1 - fy)
            + array[y0, x1] * fx * (1 - fy)
            + array[y1, x0] * (1 - fx) * fy
            + array[y1, x1] * fx * fy
        )
        if not np.isfinite(value):
            raise DemUnavailable("DEM 고도값이 유효하지 않습니다.")
        return float(value)

    def elevation(
        self,
        lon: float,
        lat: float,
        zoom: int | None = None,
    ) -> float:
        if not settings.dem_enabled:
            raise DemUnavailable("DEM_ENABLED=false")

        metadata = self.metadata()
        requested = int(
            zoom if zoom is not None
            else settings.dem_zoom
        )
        requested = max(metadata.minzoom, requested)
        start_zoom = min(metadata.maxzoom, requested)

        self._last_requested_zoom = requested
        errors: list[str] = []

        for candidate_zoom in range(
            start_zoom,
            metadata.minzoom - 1,
            -1,
        ):
            try:
                value = self._sample_tile_value(
                    lon,
                    lat,
                    candidate_zoom,
                )
                self._last_effective_zoom = candidate_zoom
                if candidate_zoom < requested:
                    self._zoom_fallback_count += 1
                return value
            except DemUnavailable as exc:
                errors.append(str(exc))
                continue

        detail = errors[-1] if errors else "사용 가능한 DEM 타일이 없습니다."
        raise DemUnavailable(
            f"DEM 줌 {requested}~{metadata.minzoom} 조회 실패: {detail}"
        )

    def elevation_grid(
        self,
        lon_values: np.ndarray,
        lat_values: np.ndarray,
        zoom: int | None = None,
    ) -> np.ndarray:
        """Sample a complete grid tile-by-tile instead of cell-by-cell."""
        if not settings.dem_enabled:
            raise DemUnavailable("DEM_ENABLED=false")
        metadata = self.metadata()
        requested = int(zoom if zoom is not None else settings.dem_zoom)
        requested = max(metadata.minzoom, requested)
        start_zoom = min(metadata.maxzoom, requested)
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        clipped_lat = np.clip(lat_grid, -85.05112878, 85.05112878)
        errors: list[str] = []
        for candidate_zoom in range(start_zoom, metadata.minzoom - 1, -1):
            try:
                n = float(2 ** candidate_zoom)
                tx = (lon_grid + 180.0) / 360.0 * n
                ty = (
                    1.0
                    - np.arcsinh(np.tan(np.radians(clipped_lat))) / np.pi
                ) / 2.0 * n
                tile_x = np.floor(tx).astype(np.int64)
                tile_y = np.floor(ty).astype(np.int64)
                output = np.empty(lon_grid.shape, dtype=np.float32)
                pairs = np.unique(
                    np.column_stack((tile_x.ravel(), tile_y.ravel())),
                    axis=0,
                )
                for raw_x, raw_y in pairs:
                    x, y = int(raw_x), int(raw_y)
                    array = self._load_tile(candidate_zoom, x, y)
                    mask = (tile_x == x) & (tile_y == y)
                    height, width = array.shape
                    px = np.clip((tx[mask] - x) * width - 0.5, 0, width - 1.001)
                    py = np.clip((ty[mask] - y) * height - 0.5, 0, height - 1.001)
                    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
                    x1, y1 = np.minimum(width - 1, x0 + 1), np.minimum(height - 1, y0 + 1)
                    fx, fy = px - x0, py - y0
                    output[mask] = (
                        array[y0, x0] * (1 - fx) * (1 - fy)
                        + array[y0, x1] * fx * (1 - fy)
                        + array[y1, x0] * (1 - fx) * fy
                        + array[y1, x1] * fx * fy
                    ).astype(np.float32)
                if not np.all(np.isfinite(output)):
                    raise DemUnavailable("DEM 격자 고도값이 유효하지 않습니다.")
                self._last_requested_zoom = requested
                self._last_effective_zoom = candidate_zoom
                return output
            except DemUnavailable as exc:
                errors.append(str(exc))
        raise DemUnavailable(errors[-1] if errors else "DEM 격자 조회 실패")

    def clear_failures(self) -> None:
        """Clear transient tile and metadata retry locks."""
        with self._lock:
            self._failures.clear()
            self._metadata_failure_until = 0.0
            self._last_error = ""

    def prefetch(self, bounds: tuple[float, float, float, float] | None = None) -> dict:
        metadata = self.metadata(force=True)
        requested_zoom = int(settings.dem_zoom)
        z = max(
            metadata.minzoom,
            min(metadata.maxzoom, requested_zoom),
        )
        self.clear_failures()
        west, south, east, north = bounds or (
            settings.dem_pohang_west,
            settings.dem_pohang_south,
            settings.dem_pohang_east,
            settings.dem_pohang_north,
        )
        x0, y_north = self._tile_fraction(west, north, z)
        x1, y_south = self._tile_fraction(east, south, z)
        first_x, last_x = int(math.floor(min(x0, x1))), int(math.floor(max(x0, x1)))
        first_y, last_y = int(math.floor(min(y_north, y_south))), int(math.floor(max(y_north, y_south)))
        total = (last_x - first_x + 1) * (last_y - first_y + 1)
        downloaded = cached = failed = 0
        errors: list[str] = []
        for x in range(first_x, last_x + 1):
            for y in range(first_y, last_y + 1):
                path = self._cache_path(metadata, z, x, y)
                existed = path.exists() and path.stat().st_size > 100
                try:
                    self._load_tile(z, x, y)
                    if existed:
                        cached += 1
                    else:
                        downloaded += 1
                except Exception as exc:
                    failed += 1
                    if len(errors) < 10:
                        errors.append(str(exc))
        return {
            "requested_zoom": requested_zoom,
            "zoom": z,
            "bounds": [west, south, east, north],
            "total": total,
            "downloaded": downloaded,
            "cached": cached,
            "failed": failed,
            "errors": errors,
            "status": self.status(),
        }

    def status(self) -> dict:
        tile_files = list((self.cache_dir / "tiles").glob("*/*/*")) if (self.cache_dir / "tiles").exists() else []
        return {
            "enabled": bool(settings.dem_enabled),
            "source": settings.dem_tilejson_url or settings.dem_tile_url_template,
            "requested_zoom": settings.dem_zoom,
            "effective_maxzoom": (
                self._metadata.maxzoom
                if self._metadata is not None
                else min(settings.dem_zoom, 12)
                if "mapterhorn.com" in (
                    settings.dem_tilejson_url
                    or settings.dem_tile_url_template
                )
                else settings.dem_zoom
            ),
            "last_effective_zoom": self._last_effective_zoom,
            "zoom_fallback_count": self._zoom_fallback_count,
            "encoding": (
                self._metadata.encoding
                if self._metadata is not None
                else settings.dem_encoding
            ),
            "cache_dir": str(self.cache_dir),
            "cached_tile_count": len([p for p in tile_files if p.is_file()]),
            "memory_tile_count": len(self._arrays),
            "downloads": self._download_count,
            "cache_hits": self._cache_hit_count,
            "last_error": self._last_error or None,
        }


class VworldTerrainContextStore:
    """Cached VWorld roads, river network and coastline reader."""

    def __init__(self) -> None:
        self.cache_dir = Path(settings.dem_context_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._memory: OrderedDict[str, list[list[list[float]]]] = OrderedDict()
        self._requests = 0
        self._hits = 0
        self._last_error = ""

    @staticmethod
    def _geometry_lines(geometry: dict) -> list[list[list[float]]]:
        kind = str(geometry.get("type") or "")
        coordinates = geometry.get("coordinates") or []
        if kind == "LineString":
            raw_lines = [coordinates]
        elif kind == "MultiLineString":
            raw_lines = coordinates
        elif kind == "Polygon":
            raw_lines = coordinates
        elif kind == "MultiPolygon":
            raw_lines = [ring for polygon in coordinates for ring in polygon]
        else:
            raw_lines = []
        lines: list[list[list[float]]] = []
        for raw in raw_lines:
            line: list[list[float]] = []
            for point in raw or []:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                x, y = float(point[0]), float(point[1])
                if abs(x) > 180 or abs(y) > 90:
                    radius = 6378137.0
                    x, y = math.degrees(x / radius), math.degrees(math.atan(math.sinh(y / radius)))
                if -180 <= x <= 180 and -90 <= y <= 90:
                    line.append([round(x, 7), round(y, 7)])
            if len(line) >= 2:
                lines.append(line)
        return lines

    @staticmethod
    def _feature_collection(payload: dict) -> dict:
        response = payload.get("response") if isinstance(payload, dict) else None
        result = response.get("result") if isinstance(response, dict) else None
        collection = result.get("featureCollection") if isinstance(result, dict) else None
        if isinstance(collection, dict):
            return collection
        return payload if isinstance(payload, dict) and payload.get("type") == "FeatureCollection" else {"features": []}

    def lines(
        self,
        layer: str,
        bounds: tuple[float, float, float, float],
    ) -> list[list[list[float]]]:
        if not str(settings.vworld_api_key or "").strip():
            return []
        key_payload = {"layer": layer, "bounds": [round(float(v), 5) for v in bounds]}
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                self._hits += 1
                return json.loads(json.dumps(cached))
        path = self.cache_dir / f"{key}.json"
        try:
            if path.is_file():
                cached = json.loads(path.read_text(encoding="utf-8")).get("lines") or []
                with self._lock:
                    self._memory[key] = cached
                    self._hits += 1
                return cached
        except Exception:
            pass
        west, south, east, north = [float(v) for v in bounds]
        params = {
            "service": "data", "version": "2.0", "request": "GetFeature",
            "format": "json", "size": "1000", "page": "1", "data": layer,
            "geometry": "true", "attribute": "false", "crs": "EPSG:4326",
            "geomFilter": f"BOX({west},{south},{east},{north})",
            "key": str(settings.vworld_api_key),
            "domain": str(settings.vworld_referer or "http://localhost:8000"),
        }
        try:
            response = requests.get(
                str(settings.dem_context_data_url), params=params,
                timeout=max(1, int(settings.dem_context_timeout_seconds)),
                headers={"User-Agent": "PohangFloodControl/8.5.13"},
            )
            response.raise_for_status()
            collection = self._feature_collection(response.json())
            lines = []
            for feature in collection.get("features") or []:
                geometry = feature.get("geometry") if isinstance(feature, dict) else None
                if isinstance(geometry, dict):
                    lines.extend(self._geometry_lines(geometry))
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps({"lines": lines}, separators=(",", ":")), encoding="utf-8")
            temp.replace(path)
            with self._lock:
                self._requests += 1
                self._memory[key] = lines
                while len(self._memory) > 64:
                    self._memory.popitem(last=False)
                self._last_error = ""
            return lines
        except Exception as exc:
            with self._lock:
                self._requests += 1
                self._last_error = str(exc)[:300]
            return []

    def status(self) -> dict:
        with self._lock:
            return {
                "layers": [settings.dem_road_data_layer] + [
                    value.strip() for value in settings.dem_hydro_data_layers.split(",") if value.strip()
                ],
                "requests": self._requests,
                "cache_hits": self._hits,
                "last_error": self._last_error or None,
            }


class TerrainFloodModel:
    """DEM-connected, volume-limited screening model for dashboard visualization."""

    _neighbors = [
        (-1, -1, math.sqrt(2)), (-1, 0, 1), (-1, 1, math.sqrt(2)),
        (0, -1, 1),                         (0, 1, 1),
        (1, -1, math.sqrt(2)),  (1, 0, 1),  (1, 1, math.sqrt(2)),
    ]

    def __init__(self, store: DemTileStore, context: VworldTerrainContextStore | None = None) -> None:
        self.store = store
        self.context = context or VworldTerrainContextStore()

    @staticmethod
    def _source_priority(source: dict) -> float:
        return (
            float(source.get("stage") or 0) * 100.0
            + float(source.get("rain_mm") or 0) * 2.0
            + float(source.get("depth_cm") or 0)
        )

    @staticmethod
    def _distance_m(a: dict, b: dict) -> float:
        mean_lat = math.radians((float(a["lat"]) + float(b["lat"])) / 2.0)
        dx = (float(a["lon"]) - float(b["lon"])) * 111320.0 * math.cos(mean_lat)
        dy = (float(a["lat"]) - float(b["lat"])) * 110540.0
        return math.hypot(dx, dy)

    def select_sources(self, cameras: list[dict], rain_points: list[dict]) -> list[dict]:
        """Select actual CCTV flood detections as DEM flood seeds.

        Rain gauges no longer create independent flood polygons. Rainfall is
        already attached to each CCTV in main.py and is used only to support
        spread/volume around a CCTV whose AI stage is >= 1.
        """
        candidates: list[dict] = []
        for camera in cameras:
            ai_stage = max(
                0,
                min(
                    4,
                    int(
                        camera.get("ai_stage")
                        if camera.get("ai_stage") is not None
                        else camera.get("stage") or 0
                    ),
                ),
            )
            if ai_stage < 1:
                continue
            candidates.append({
                **camera,
                "stage": ai_stage,
                "ai_stage": ai_stage,
                "depth_cm": float(level_to_depth_cm(ai_stage)),
                "kind": "CCTV",
            })

        candidates.sort(key=self._source_priority, reverse=True)
        selected: list[dict] = []

        # The previous 280 m merge radius and 6-source cap could remove the
        # actual flooded CCTV when many rainfall-promoted Lev1 sources existed.
        # Keep distinct CCTV detections, merging only near-duplicate coordinates.
        merge_m = min(60.0, max(20.0, float(settings.dem_source_merge_m)))
        source_limit = max(
            int(settings.dem_max_sources),
            min(24, len(candidates)),
        )

        for candidate in candidates:
            if any(
                self._distance_m(candidate, prior) < merge_m
                for prior in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= source_limit:
                break

        return selected

    def _sample_grid(
        self,
        source: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        cell_m = max(8.0, min(30.0, float(settings.dem_flood_cell_m)))
        stage = max(1, min(4, int(source.get("stage") or 1)))
        rain_mm = max(0.0, float(source.get("rain_mm") or 0.0))
        kind = str(source.get("kind") or "").upper()

        if kind == "CCTV":
            radius_m = 160.0 + stage * 90.0 + min(120.0, rain_mm * 4.0)
        else:
            radius_m = 150.0 + stage * 115.0 + min(180.0, rain_mm * 4.0)

        radius_m = min(
            float(settings.dem_flood_max_radius_m),
            max(cell_m * 3.0, radius_m),
        )
        count = int(math.ceil(radius_m * 2.0 / cell_m)) + 1
        count = max(11, min(int(settings.dem_flood_max_grid_cells), count))
        if count % 2 == 0:
            count += 1
        radius_m = cell_m * (count - 1) / 2.0
        lat0 = float(source["lat"])
        lon0 = float(source["lon"])
        meters_per_lon = max(
            1000.0,
            111320.0 * math.cos(math.radians(lat0)),
        )
        dlat = cell_m / 110540.0
        dlon = cell_m / meters_per_lon
        rows = np.arange(count, dtype=np.float64) - count // 2
        cols = np.arange(count, dtype=np.float64) - count // 2
        lats = lat0 + rows * dlat
        lons = lon0 + cols * dlon
        dem = self.store.elevation_grid(lons, lats)
        return dem, lons, lats, cell_m, radius_m

    @staticmethod
    def _rasterise_lines(
        lines: list[list[list[float]]],
        lons: np.ndarray,
        lats: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        if len(lons) < 2 or len(lats) < 2:
            return mask
        dlon, dlat = float(lons[1] - lons[0]), float(lats[1] - lats[0])
        for line in lines:
            points = [[
                int(round((float(point[0]) - float(lons[0])) / dlon)),
                int(round((float(point[1]) - float(lats[0])) / dlat)),
            ] for point in line if len(point) >= 2]
            if len(points) >= 2:
                cv2.polylines(mask, [np.asarray(points, dtype=np.int32)], False, 255, 1, cv2.LINE_8)
        return mask

    def _context_masks(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.0
        dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.0
        bounds = (float(lons.min()) - dlon, float(lats.min()) - dlat,
                  float(lons.max()) + dlon, float(lats.max()) + dlat)
        hydro_layers = [value.strip() for value in settings.dem_hydro_data_layers.split(",") if value.strip()]
        layers = [str(settings.dem_road_data_layer)] + hydro_layers
        with ThreadPoolExecutor(max_workers=max(1, len(layers))) as executor:
            groups = list(executor.map(lambda layer: self.context.lines(layer, bounds), layers))
        road_lines = groups[0] if groups else []
        hydro_lines = [line for group in groups[1:] for line in group]
        road = self._rasterise_lines(road_lines, lons, lats, shape)
        hydro = self._rasterise_lines(hydro_lines, lons, lats, shape)
        return road, hydro, {
            "road_line_count": len(road_lines),
            "road_cell_count": int(np.count_nonzero(road)),
            "hydro_line_count": len(hydro_lines),
            "hydro_cell_count": int(np.count_nonzero(hydro)),
        }

    def _water_exclusion_mask(
        self,
        dem: np.ndarray,
        hydro_mask: np.ndarray | None,
        cell_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return land-exclusion masks for rivers/streams and the open sea.

        V8.6.2 used hydrography as a *positive* susceptibility factor, so the
        flood colour could be painted directly on the river or sea.  V8.6.3
        keeps hydrography useful for nearby-land susceptibility, but removes
        actual water cells from the inundated-land surface.
        """
        shape = dem.shape
        if not bool(settings.dem_water_exclusion_enabled):
            zero = np.zeros(shape, dtype=np.uint8)
            return zero, zero, zero, {
                "enabled": False,
                "hydro_core_cell_count": 0,
                "river_excluded_cell_count": 0,
                "sea_excluded_cell_count": 0,
                "water_excluded_cell_count": 0,
                "river_buffer_m": 0.0,
                "sea_level_max_m": None,
            }

        hydro_core = (
            (hydro_mask > 0).astype(np.uint8)
            if hydro_mask is not None and hydro_mask.shape == shape
            else np.zeros(shape, dtype=np.uint8)
        )
        river_exclusion = hydro_core.copy()
        river_buffer_m = max(0.0, float(settings.dem_river_exclusion_buffer_m))
        if np.any(hydro_core) and river_buffer_m > 0:
            radius_cells = max(1, int(round(river_buffer_m / max(1.0, cell_m))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (radius_cells * 2 + 1, radius_cells * 2 + 1),
            )
            river_exclusion = cv2.dilate(hydro_core, kernel, iterations=1)

        sea_level_max_m = float(settings.dem_sea_exclusion_max_elevation_m)
        sea_candidate = (dem <= sea_level_max_m).astype(np.uint8)
        sea_exclusion = np.zeros(shape, dtype=np.uint8)
        if np.any(sea_candidate):
            _, labels = cv2.connectedComponents(sea_candidate, connectivity=8)
            edge_values = np.concatenate(
                (labels[0], labels[-1], labels[:, 0], labels[:, -1])
            )
            edge_labels = {int(value) for value in edge_values if int(value) > 0}
            if edge_labels:
                sea_exclusion = np.isin(labels, list(edge_labels)).astype(np.uint8)

        water_exclusion = np.maximum(river_exclusion, sea_exclusion).astype(np.uint8)
        diagnostics = {
            "enabled": True,
            "hydro_core_cell_count": int(np.count_nonzero(hydro_core)),
            "river_excluded_cell_count": int(np.count_nonzero(river_exclusion)),
            "sea_excluded_cell_count": int(np.count_nonzero(sea_exclusion)),
            "water_excluded_cell_count": int(np.count_nonzero(water_exclusion)),
            "river_buffer_m": round(river_buffer_m, 1),
            "sea_level_max_m": round(sea_level_max_m, 2),
        }
        return water_exclusion, hydro_core, sea_exclusion, diagnostics

    @staticmethod
    def _history_grid(
        points: list[dict],
        lons: np.ndarray,
        lats: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        grid = np.zeros((len(lats), len(lons)), dtype=np.float32)
        if not points:
            return grid, 0
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        mean_lat = math.radians(float(np.mean(lats)))
        influence = max(30.0, float(settings.dem_history_influence_m))
        nearby = 0
        for point in points:
            try:
                lat, lon = float(point["lat"]), float(point["lon"])
                stage = max(1, min(4, int(point.get("stage") or 1)))
                recency = max(0.1, min(1.0, float(point.get("recency") or 1.0)))
            except Exception:
                continue
            dx = (lon_grid - lon) * 111320.0 * math.cos(mean_lat)
            dy = (lat_grid - lat) * 110540.0
            distance = np.hypot(dx, dy)
            if float(distance.min()) <= influence * 2:
                nearby += 1
            grid = np.maximum(grid, (0.35 + 0.65 * stage / 4.0) * recency * np.exp(-np.square(distance / influence)))
        return np.clip(grid, 0, 1), nearby

    def _snap_seed(
        self,
        source: dict,
        dem: np.ndarray,
        cell_m: float,
    ) -> tuple[int, int]:
        """Anchor CCTV to its actual coordinate; only rain points may snap downhill."""
        rows, cols = dem.shape
        center_r, center_c = rows // 2, cols // 2

        kind = str(source.get("kind") or "").upper()
        if kind == "CCTV":
            # A CCTV observation belongs to the camera coordinate. Moving the seed
            # to a remote local minimum makes the flood polygon appear at another block.
            return center_r, center_c

        snap_radius_m = (
            float(settings.dem_rain_snap_radius_m)
            if kind == "KMA"
            else float(settings.dem_source_snap_radius_m)
        )
        if snap_radius_m <= 0:
            return center_r, center_c

        snap_cells = max(1, int(round(snap_radius_m / cell_m)))
        r0, r1 = max(0, center_r - snap_cells), min(rows, center_r + snap_cells + 1)
        c0, c1 = max(0, center_c - snap_cells), min(cols, center_c + snap_cells + 1)
        gy, gx = np.gradient(dem, cell_m)
        slope = np.hypot(gx, gy)
        best = (center_r, center_c)
        best_score = float("inf")
        for r in range(r0, r1):
            for c in range(c0, c1):
                distance = math.hypot(r - center_r, c - center_c) * cell_m
                if distance > snap_radius_m + 1e-6:
                    continue
                # Distance penalty prevents a rain point from jumping to a distant basin.
                score = (
                    float(dem[r, c])
                    + float(slope[r, c]) * 25.0
                    + distance * 0.045
                )
                if score < best_score:
                    best_score = score
                    best = (r, c)
        return best

    def _spill_cost(self, dem: np.ndarray, seed: tuple[int, int], cell_m: float) -> np.ndarray:
        rows, cols = dem.shape
        spill = np.full((rows, cols), np.inf, dtype=np.float64)
        settled = np.zeros((rows, cols), dtype=bool)
        sr, sc = seed
        spill[sr, sc] = float(dem[sr, sc])
        heap: list[tuple[float, int, int]] = [(float(dem[sr, sc]), sr, sc)]
        friction = float(settings.dem_flow_friction_m_per_km) / 1000.0
        while heap:
            cost, r, c = heapq.heappop(heap)
            if settled[r, c]:
                continue
            settled[r, c] = True
            for dr, dc, scale in self._neighbors:
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= rows or nc >= cols or settled[nr, nc]:
                    continue
                proposed = max(
                    cost + friction * cell_m * scale,
                    float(dem[nr, nc]),
                )
                if proposed < spill[nr, nc]:
                    spill[nr, nc] = proposed
                    heapq.heappush(heap, (proposed, nr, nc))
        return spill.astype(np.float32)

    def _depth_grid(
        self,
        source: dict,
        dem: np.ndarray,
        cell_m: float,
        radius_m: float,
        road_mask: np.ndarray | None = None,
        hydro_mask: np.ndarray | None = None,
        history_grid: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[int, int],
        float,
        np.ndarray,
        dict,
    ]:
        seed = self._snap_seed(source, dem, cell_m)
        spill = self._spill_cost(dem, seed, cell_m)
        sr, sc = seed
        source_elevation = float(dem[sr, sc])
        rain_mm = max(0.0, float(source.get("rain_mm") or 0.0))
        source_stage = max(1, min(4, int(source.get("stage") or 1)))
        observed_depth_m = max(
            0.0,
            float(source.get("depth_cm") or 0.0) / 100.0,
        )
        stage_depth_cap_m = {
            1: 0.119,
            2: 0.349,
            3: 0.599,
            4: float(settings.dem_max_water_depth_m),
        }[source_stage]
        head_m = min(
            stage_depth_cap_m,
            max(observed_depth_m, 0.05 + rain_mm * 0.012),
        )
        water_surface = source_elevation + head_m
        yy, xx = np.indices(dem.shape)
        distance = np.hypot(yy - sr, xx - sc) * cell_m
        accessible = (
            (spill <= water_surface + 1e-5)
            & (distance <= radius_m)
        )
        raw_depth = np.where(
            accessible,
            np.maximum(0.0, water_surface - dem),
            0.0,
        )
        gy, gx = np.gradient(dem, cell_m)
        slope = np.hypot(gx, gy)
        slope_factor = np.exp(
            -np.square(
                slope
                / max(0.02, settings.dem_ponding_slope_scale)
            )
        )
        radial_decay = max(0.0, min(0.45, float(settings.dem_radial_depth_decay)))
        distance_factor = np.clip(
            1.0 - radial_decay * np.square(distance / max(cell_m, radius_m)),
            1.0 - radial_decay,
            1.0,
        )
        depth = raw_depth * slope_factor * distance_factor

        neighbourhood = cv2.GaussianBlur(
            dem.astype(np.float32), (0, 0),
            sigmaX=max(0.8, 45.0 / max(1.0, cell_m)),
            sigmaY=max(0.8, 45.0 / max(1.0, cell_m)),
            borderType=cv2.BORDER_REPLICATE,
        )
        relief = neighbourhood - dem
        valley_factor = np.clip((relief + 0.15) / 1.5, 0.0, 1.0)
        low, high = np.percentile(dem, [10, 90])
        lowland_factor = np.clip((float(high) - dem) / max(0.25, float(high - low)), 0.0, 1.0)
        road_factor = (
            cv2.dilate((road_mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
            if road_mask is not None and np.any(road_mask) else np.zeros(dem.shape, np.float32)
        )
        (
            water_exclusion,
            hydro_core,
            sea_exclusion,
            water_diagnostics,
        ) = self._water_exclusion_mask(dem, hydro_mask, cell_m)
        # Preserve the observed CCTV anchor even for bridge/river-edge cameras;
        # all surrounding water cells remain excluded from the flood polygon.
        seed_water_override = bool(water_exclusion[sr, sc])
        if seed_water_override:
            water_exclusion = water_exclusion.copy()
            water_exclusion[sr, sc] = 0

        hydro_reference = np.maximum(hydro_core, sea_exclusion).astype(np.uint8)
        if np.any(hydro_reference):
            hydro_distance = cv2.distanceTransform(
                (hydro_reference == 0).astype(np.uint8),
                cv2.DIST_L2,
                5,
            ) * cell_m
            hydro_factor = np.exp(
                -np.square(
                    hydro_distance
                    / max(cell_m, float(settings.dem_hydro_influence_m))
                )
            )
        else:
            hydro_distance = np.full(dem.shape, np.inf, np.float32)
            hydro_factor = np.zeros(dem.shape, np.float32)
        historical = (
            np.clip(history_grid, 0, 1).astype(np.float32)
            if history_grid is not None and history_grid.shape == dem.shape
            else np.zeros(dem.shape, np.float32)
        )
        susceptibility = np.clip(
            0.24 * slope_factor + 0.22 * valley_factor + 0.22 * lowland_factor
            + 0.11 * road_factor + 0.11 * hydro_factor
            + max(0.0, min(0.30, float(settings.dem_history_weight))) * historical,
            0.0, 1.0,
        )
        flood_prone = (
            susceptibility >= max(0.05, min(0.8, float(settings.dem_flood_prone_min_score)))
        ) | (distance <= max(45.0, cell_m * 2.5))
        # Flood colours describe inundated *land*.  Rivers/streams and open-sea
        # cells may guide nearby-land susceptibility, but can never themselves
        # become coloured flood polygons.
        accessible &= flood_prone
        accessible &= water_exclusion == 0
        depth = np.where(accessible, depth, 0.0)

        # Preserve a source-centred support footprint for an observed CCTV depth.
        # It is still clipped by the DEM minimax spill elevation, so it cannot cross
        # a high ridge, but it no longer collapses to a remote one-cell square.
        if observed_depth_m > 0:
            support_radius = max(
                cell_m * 1.5,
                min(
                    float(settings.dem_ai_support_radius_m),
                    radius_m * 0.45,
                ),
            )
            support_gate = (
                (
                    spill
                    <= source_elevation
                    + observed_depth_m
                    + float(settings.dem_fallback_spill_margin_m)
                )
                & (water_exclusion == 0)
            )
            support_depth = (
                observed_depth_m
                * np.exp(-np.square(distance / support_radius))
            )
            depth = np.maximum(
                depth,
                np.where(support_gate, support_depth, 0.0),
            )

        catchment_radius = min(
            radius_m,
            300.0 + int(source.get("stage") or 1) * 130.0,
        )
        rain_volume = (
            rain_mm / 1000.0
            * math.pi
            * catchment_radius**2
            * settings.dem_runoff_coefficient
        )
        ai_volume = (
            observed_depth_m
            * math.pi
            * settings.dem_ai_support_radius_m**2
            * 0.62
        )
        volume_budget = max(
            settings.dem_min_volume_m3,
            rain_volume + ai_volume,
        )
        # Keep a second depth field for colour classification.  The operational
        # volume budget may reduce the rendered *amount* of water, but it must
        # not flatten a Lev4 terrain surface into an all-Lev1 blue polygon.
        # Colour bands therefore use the DEM-derived water-surface-minus-ground
        # depth before global volume scaling, clipped to the final wet footprint.
        terrain_depth = depth.copy().astype(np.float32)

        raw_volume = float(depth.sum()) * cell_m * cell_m
        volume_scale = 1.0
        if raw_volume > volume_budget > 0:
            volume_scale = max(0.0, min(1.0, volume_budget / raw_volume))
            depth *= volume_scale

        # Rain and terrain may change the footprint, but they cannot create a
        # deeper colour band than the CCTV AI stage that seeded this surface.
        depth = np.clip(
            depth,
            0.0,
            stage_depth_cap_m,
        )
        terrain_depth = np.clip(
            terrain_depth,
            0.0,
            stage_depth_cap_m,
        )
        depth = np.where(water_exclusion == 0, depth, 0.0)
        terrain_depth = np.where(water_exclusion == 0, terrain_depth, 0.0)
        if observed_depth_m > 0:
            observed_clipped = min(
                observed_depth_m,
                settings.dem_max_water_depth_m,
            )
            depth[sr, sc] = max(float(depth[sr, sc]), observed_clipped)
            terrain_depth[sr, sc] = max(
                float(terrain_depth[sr, sc]),
                observed_clipped,
            )
        depth[depth < 0.01] = 0.0
        terrain_depth[terrain_depth < 0.01] = 0.0
        diagnostics = {
            "mean_slope_pct": round(float(np.mean(slope)) * 100.0, 2),
            "maximum_slope_pct": round(float(np.max(slope)) * 100.0, 2),
            "hydro_available": bool(np.any(hydro_reference)),
            "hydro_dem_sea_fallback": bool(np.any(sea_exclusion)),
            "nearest_hydro_m": round(float(np.min(hydro_distance)), 1) if np.isfinite(hydro_distance).any() else None,
            "water_exclusion": water_diagnostics,
            "seed_water_override": seed_water_override,
            "volume_scale": round(float(volume_scale), 4),
            "terrain_classification_max_depth_cm": round(float(terrain_depth.max()) * 100.0, 1),
            "maximum_influence_radius_m": round(float(radius_m), 1),
            "flood_prone_cell_count": int(np.count_nonzero(flood_prone)),
            "wet_cell_count": int(np.count_nonzero(depth >= 0.01)),
            "mean_susceptibility": round(float(np.mean(susceptibility)), 3),
        }
        return (
            depth.astype(np.float32),
            terrain_depth.astype(np.float32),
            water_exclusion.astype(np.uint8),
            seed,
            source_elevation,
            spill,
            diagnostics,
        )

    @staticmethod
    def _seed_component(
        depth: np.ndarray,
        seed: tuple[int, int],
    ) -> np.ndarray:
        wet = (depth >= 0.01).astype(np.uint8)
        sr, sc = seed
        if wet[sr, sc] == 0:
            wet[sr, sc] = 1
        _, labels = cv2.connectedComponents(wet, connectivity=8)
        label = int(labels[sr, sc])
        return labels == label

    @staticmethod
    def _ring_from_contour(
        contour: np.ndarray,
        dem: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        scale: int,
    ) -> list[list[float]]:
        dlon = float(lons[1] - lons[0]) if len(lons) > 1 else 0.0
        dlat = float(lats[1] - lats[0]) if len(lats) > 1 else 0.0
        ring: list[list[float]] = []
        for point in contour[:, 0, :]:
            # Upscaled raster contour coordinates describe cell edges. Convert
            # them to geographic edges rather than cell centres.
            col_f = float(point[0]) / scale - 0.5
            row_f = float(point[1]) / scale - 0.5
            lon = float(lons[0]) + col_f * dlon
            lat = float(lats[0]) + row_f * dlat
            col = int(max(0, min(len(lons) - 1, round(col_f))))
            row = int(max(0, min(len(lats) - 1, round(row_f))))
            elevation = (
                float(dem[row, col])
                + float(settings.dem_overlay_height_m)
            )
            ring.append([lon, lat, round(elevation, 2)])
        if len(ring) >= 3 and ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring

    @staticmethod
    def _smooth_ring_chaikin(
        ring: list[list[float]],
        iterations: int,
    ) -> list[list[float]]:
        """Round raster contour corners for display without changing the depth grid."""
        if len(ring) < 5 or iterations <= 0:
            return ring
        points = [list(map(float, point[:3])) for point in ring[:-1]]
        for _ in range(max(0, min(3, int(iterations)))):
            if len(points) < 3:
                break
            smoothed: list[list[float]] = []
            for index, point in enumerate(points):
                nxt = points[(index + 1) % len(points)]
                q = [0.75 * point[i] + 0.25 * nxt[i] for i in range(3)]
                r = [0.25 * point[i] + 0.75 * nxt[i] for i in range(3)]
                smoothed.extend([q, r])
            points = smoothed
        output = [
            [round(point[0], 7), round(point[1], 7), round(point[2], 2)]
            for point in points
        ]
        if output:
            output.append(output[0])
        return output

    def _features_from_depth(
        self,
        source: dict,
        depth: np.ndarray,
        dem: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        seed: tuple[int, int],
        cell_m: float,
        *,
        terrain_depth: np.ndarray | None = None,
        water_exclusion: np.ndarray | None = None,
    ) -> list[dict]:
        connected = self._seed_component(depth, seed)
        wet_footprint = connected & (depth >= 0.01)
        classification = (
            terrain_depth
            if terrain_depth is not None and terrain_depth.shape == depth.shape
            else depth
        )
        clean_depth = np.where(
            wet_footprint,
            classification,
            0.0,
        ).astype(np.float32)
        if water_exclusion is not None and water_exclusion.shape == depth.shape:
            clean_depth = np.where(water_exclusion == 0, clean_depth, 0.0)
            connected = connected & (water_exclusion == 0)
        thresholds = {
            1: 0.01,
            2: 0.12,
            3: 0.35,
            4: 0.60,
        }
        features: list[dict] = []
        scale = max(4, min(10, int(settings.dem_contour_upscale)))
        smooth_enabled = bool(settings.dem_depth_smoothing_enabled)
        sigma_cells = max(0.0, float(settings.dem_depth_smoothing_sigma_cells))
        boundary_iterations = max(0, min(3, int(settings.dem_boundary_smoothing_iterations)))

        # Commercial/operational flood viewers generally render an inundation
        # depth raster (water-surface elevation minus terrain), then derive
        # coloured depth bands / inundation boundaries from that grid. We keep
        # clean_depth as the authoritative calculation and only resample/smooth
        # the display raster before contour extraction.
        depth_display = cv2.resize(
            clean_depth,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC if smooth_enabled else cv2.INTER_NEAREST,
        )
        depth_display = np.maximum(depth_display, 0.0).astype(np.float32)
        if smooth_enabled and sigma_cells > 0:
            sigma_px = max(0.1, sigma_cells * scale)
            depth_display = cv2.GaussianBlur(
                depth_display,
                (0, 0),
                sigmaX=sigma_px,
                sigmaY=sigma_px,
                borderType=cv2.BORDER_REPLICATE,
            )

        # Prevent display smoothing from creating remote islands: the smoothed
        # pixels must remain close to the original terrain-connected wet cells.
        connected_up = cv2.resize(
            connected.astype(np.uint8),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        if smooth_enabled:
            support_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(3, scale // 2 * 2 + 1), max(3, scale // 2 * 2 + 1)),
            )
            connected_up = cv2.dilate(connected_up, support_kernel, iterations=1)
        depth_display = np.where(connected_up > 0, depth_display, 0.0)
        water_up: np.ndarray | None = None
        if water_exclusion is not None and water_exclusion.shape == depth.shape:
            water_up = cv2.resize(
                (water_exclusion > 0).astype(np.uint8),
                (depth_display.shape[1], depth_display.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            # One display-pixel guard stops cubic/Gaussian smoothing from bleeding
            # a coloured edge back across a river/coastline boundary.
            water_up = cv2.dilate(
                water_up,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            depth_display = np.where(water_up == 0, depth_display, 0.0)

        # Cumulative bands remain nested: Lev1 is the full footprint and deeper
        # thresholds are drawn above it.
        for level in range(1, 5):
            mask = np.where(
                depth_display >= thresholds[level],
                255,
                0,
            ).astype(np.uint8)
            if boundary_iterations > 0 and np.any(mask):
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (5, 5),
                )
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=boundary_iterations,
                )
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=1,
                )
            # Morphological smoothing is display-only and must never close a
            # river/sea gap that the land mask intentionally removed.
            if water_up is not None:
                mask = np.where(water_up > 0, 0, mask).astype(np.uint8)
            if not np.any(mask):
                continue

            contours, hierarchy = cv2.findContours(
                mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE,
            )
            if hierarchy is None:
                continue
            hierarchy = hierarchy[0]
            outer_index = 0
            for contour_index, contour in enumerate(contours):
                # CCOMP parent=-1 is an outer ring. Child contours become GeoJSON
                # holes, so an enclosed river/pond remains transparent instead of
                # being filled by the polygon renderer.
                if int(hierarchy[contour_index][3]) != -1:
                    continue
                outer_area_cells = cv2.contourArea(contour) / (scale * scale)
                if outer_area_cells < float(settings.dem_min_polygon_cells):
                    continue

                epsilon = max(
                    0.6,
                    0.0018 * cv2.arcLength(contour, True),
                )
                contour = cv2.approxPolyDP(contour, epsilon, True)
                outer_ring = self._ring_from_contour(
                    contour, dem, lons, lats, scale
                )
                outer_ring = self._smooth_ring_chaikin(
                    outer_ring, boundary_iterations
                )
                if len(outer_ring) < 4:
                    continue

                rings = [outer_ring]
                hole_area_cells = 0.0
                child_index = int(hierarchy[contour_index][2])
                while child_index != -1:
                    hole_contour = contours[child_index]
                    raw_hole_area = cv2.contourArea(hole_contour) / (scale * scale)
                    # Preserve even fairly small hydro holes; filtering them out
                    # would paint the flood layer back on narrow streams/ponds.
                    if raw_hole_area >= 0.20:
                        hole_epsilon = max(
                            0.6,
                            0.0018 * cv2.arcLength(hole_contour, True),
                        )
                        hole_contour = cv2.approxPolyDP(
                            hole_contour, hole_epsilon, True
                        )
                        hole_ring = self._ring_from_contour(
                            hole_contour, dem, lons, lats, scale
                        )
                        hole_ring = self._smooth_ring_chaikin(
                            hole_ring, boundary_iterations
                        )
                        if len(hole_ring) >= 4:
                            rings.append(hole_ring)
                            hole_area_cells += raw_hole_area
                    child_index = int(hierarchy[child_index][0])

                area_cells = max(0.0, outer_area_cells - hole_area_cells)
                level_mask = clean_depth >= thresholds[level]
                max_depth_cm = (
                    float(clean_depth[level_mask].max() * 100.0)
                    if np.any(level_mask)
                    else level_to_depth_cm(level)
                )
                band_ceiling_cm = {
                    1: 11.9,
                    2: 34.9,
                    3: 59.9,
                    4: max_depth_cm,
                }[level]
                display_depth_cm = min(max_depth_cm, band_ceiling_cm)
                sr, sc = seed
                features.append({
                    "type": "Feature",
                    "properties": {
                        "map_id": (
                            f"dem:{source.get('map_id', source.get('id'))}:"
                            f"{level}:{outer_index}"
                        ),
                        "level": level,
                        "depth_cm": round(display_depth_cm, 1),
                        "depth_m": round(display_depth_cm / 100.0, 3),
                        "terrain_max_depth_cm": round(max_depth_cm, 1),
                        "elevation_m": round(float(dem[sr, sc]), 1),
                        "source": (
                            source.get("kind")
                            or source.get("source")
                            or "DEM"
                        ),
                        "source_id": source.get("id"),
                        "source_name": source.get("name"),
                        "source_lon": float(source.get("lon")),
                        "source_lat": float(source.get("lat")),
                        "seed_lon": float(lons[sc]),
                        "seed_lat": float(lats[sr]),
                        "seed_offset_m": round(
                            math.hypot(
                                sr - len(lats) // 2,
                                sc - len(lons) // 2,
                            ) * cell_m,
                            1,
                        ),
                        "area_m2": round(area_cells * cell_m * cell_m, 1),
                        "hole_count": max(0, len(rings) - 1),
                        "rain_mm": round(
                            float(source.get("rain_mm") or 0.0),
                            2,
                        ),
                        "method": (
                            "DEM terrain-referenced depth bands + land-only "
                            "waterbody mask"
                        ),
                        "display_method": (
                            "terrain_depth_before_volume_scale_with_water_holes"
                        ),
                        "cell_m": cell_m,
                        "fallback": False,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": rings,
                    },
                })
                outer_index += 1
        return features

    @staticmethod
    def _minimum_cells_for_level(level: int) -> int:
        return {
            1: int(settings.dem_min_footprint_l1_cells),
            2: int(settings.dem_min_footprint_l2_cells),
            3: int(settings.dem_min_footprint_l3_cells),
            4: int(settings.dem_min_footprint_l4_cells),
        }.get(max(1, min(4, int(level))), 5)

    def _source_anchored_fallback_feature(
        self,
        source: dict,
        depth: np.ndarray,
        dem: np.ndarray,
        spill: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        seed: tuple[int, int],
        cell_m: float,
        *,
        water_exclusion: np.ndarray | None = None,
    ) -> dict | None:
        """Build an irregular connected patch that contains the CCTV coordinate."""
        if depth.size == 0 or not np.isfinite(dem).any():
            return None

        level = max(1, min(4, int(source.get("stage") or 1)))
        target_cells = self._minimum_cells_for_level(level)
        sr, sc = seed
        source_elevation = float(dem[sr, sc])
        observed_depth_m = max(
            0.01,
            float(source.get("depth_cm") or level_to_depth_cm(level))
            / 100.0,
        )
        max_spill = (
            source_elevation
            + observed_depth_m
            + float(settings.dem_fallback_spill_margin_m)
        )
        max_radius_m = max(
            cell_m * 1.5,
            min(
                float(settings.dem_ai_support_radius_m) * 1.35,
                55.0 + level * 28.0,
            ),
        )

        selected = np.zeros(dem.shape, dtype=np.uint8)
        visited = np.zeros(dem.shape, dtype=bool)
        heap: list[tuple[float, float, int, int]] = []
        heapq.heappush(
            heap,
            (float(spill[sr, sc]), 0.0, sr, sc),
        )

        while heap and int(selected.sum()) < target_cells:
            spill_cost, distance, r, c = heapq.heappop(heap)
            if visited[r, c]:
                continue
            visited[r, c] = True
            if distance > max_radius_m + 1e-6:
                continue
            if (
                water_exclusion is not None
                and water_exclusion.shape == dem.shape
                and bool(water_exclusion[r, c])
                and (r, c) != (sr, sc)
            ):
                continue
            # First pass respects terrain barriers. If too few cells are available,
            # the margin below still permits the immediate source neighbourhood.
            if spill_cost > max_spill and int(selected.sum()) >= 4:
                continue
            selected[r, c] = 1
            for dr, dc, scale_factor in self._neighbors:
                nr, nc = r + dr, c + dc
                if (
                    nr < 0
                    or nc < 0
                    or nr >= dem.shape[0]
                    or nc >= dem.shape[1]
                    or visited[nr, nc]
                    or (
                        water_exclusion is not None
                        and water_exclusion.shape == dem.shape
                        and bool(water_exclusion[nr, nc])
                    )
                ):
                    continue
                ndistance = math.hypot(nr - sr, nc - sc) * cell_m
                if ndistance > max_radius_m + 1e-6:
                    continue
                priority = (
                    float(spill[nr, nc])
                    + ndistance * 0.0025
                    + max(0.0, float(dem[nr, nc]) - source_elevation) * 0.15
                )
                heapq.heappush(
                    heap,
                    (priority, ndistance, nr, nc),
                )

        if int(selected.sum()) < 2:
            selected[sr, sc] = 1
            for dr, dc, _scale_factor in self._neighbors:
                nr, nc = sr + dr, sc + dc
                if nr < 0 or nc < 0 or nr >= selected.shape[0] or nc >= selected.shape[1]:
                    continue
                if (
                    water_exclusion is not None
                    and water_exclusion.shape == dem.shape
                    and bool(water_exclusion[nr, nc])
                ):
                    continue
                selected[nr, nc] = 1
                if int(selected.sum()) >= 2:
                    break

        scale = 4
        upscaled = cv2.resize(
            selected * 255,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            upscaled,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(
            contour,
            max(1.0, 0.006 * cv2.arcLength(contour, True)),
            True,
        )
        ring = self._ring_from_contour(
            contour,
            dem,
            lons,
            lats,
            scale,
        )
        if len(ring) < 4:
            return None

        area_cells = float(selected.sum())
        maximum_depth_m = max(
            float(depth.max()),
            observed_depth_m,
        )
        return {
            "type": "Feature",
            "properties": {
                "map_id": (
                    f"dem-source-anchored:"
                    f"{source.get('map_id', source.get('id'))}"
                ),
                "level": level,
                "depth_cm": round(maximum_depth_m * 100.0, 1),
                "depth_m": round(maximum_depth_m, 3),
                "elevation_m": round(float(dem[sr, sc]), 1),
                "source": (
                    source.get("kind")
                    or source.get("source")
                    or "DEM"
                ),
                "source_id": source.get("id"),
                "source_name": source.get("name"),
                "source_lon": float(source.get("lon")),
                "source_lat": float(source.get("lat")),
                "seed_lon": float(lons[sc]),
                "seed_lat": float(lats[sr]),
                "seed_offset_m": round(
                    math.hypot(
                        sr - len(lats) // 2,
                        sc - len(lons) // 2,
                    ) * cell_m,
                    1,
                ),
                "area_m2": round(area_cells * cell_m * cell_m, 1),
                "rain_mm": round(
                    float(source.get("rain_mm") or 0.0),
                    2,
                ),
                "method": (
                    "DEM source-anchored minimum connected footprint"
                ),
                "cell_m": cell_m,
                "fallback": True,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
        }


    def build_surface(
        self,
        cameras: list[dict],
        rain_points: list[dict],
        history_points: list[dict] | None = None,
    ) -> dict:
        selected = self.select_sources(cameras, rain_points)
        history_points = list(history_points or [])
        features: list[dict] = []
        errors: list[str] = []
        source_results: list[dict] = []
        for source in selected:
            try:
                dem, lons, lats, cell_m, radius_m = self._sample_grid(source)
                road_mask, hydro_mask, context_diagnostics = self._context_masks(
                    lons, lats, dem.shape
                )
                history_grid, nearby_history = self._history_grid(
                    history_points, lons, lats
                )
                (
                    depth,
                    terrain_depth,
                    water_exclusion,
                    seed,
                    source_elevation,
                    spill,
                    factor_diagnostics,
                ) = self._depth_grid(
                    source,
                    dem,
                    cell_m,
                    radius_m,
                    road_mask=road_mask,
                    hydro_mask=hydro_mask,
                    history_grid=history_grid,
                )
                source_features = self._features_from_depth(
                    source,
                    depth,
                    dem,
                    lons,
                    lats,
                    seed,
                    cell_m,
                    terrain_depth=terrain_depth,
                    water_exclusion=water_exclusion,
                )

                if not source_features:
                    fallback = self._source_anchored_fallback_feature(
                        source,
                        depth,
                        dem,
                        spill,
                        lons,
                        lats,
                        seed,
                        cell_m,
                        water_exclusion=water_exclusion,
                    )
                    if fallback is not None:
                        source_features = [fallback]

                features.extend(source_features)
                source_results.append({
                    "id": source.get("id"),
                    "name": source.get("name"),
                    "feature_count": len(source_features),
                    "source_elevation_m": round(source_elevation, 1),
                    "maximum_depth_cm": round(
                        max(
                            float(depth.max()) * 100.0,
                            float(source.get("depth_cm") or 0.0),
                        ),
                        1,
                    ),
                    "seed_row": int(seed[0]),
                    "seed_col": int(seed[1]),
                    "seed_lon": round(float(lons[seed[1]]), 7),
                    "seed_lat": round(float(lats[seed[0]]), 7),
                    "seed_offset_m": round(
                        math.hypot(
                            seed[0] - len(lats) // 2,
                            seed[1] - len(lons) // 2,
                        ) * cell_m,
                        1,
                    ),
                    "polygon_area_m2": round(
                        max(
                            (
                                float(feature["properties"].get("area_m2") or 0.0)
                                for feature in source_features
                            ),
                            default=0.0,
                        ),
                        1,
                    ),
                    "cell_m": round(float(cell_m), 2),
                    "analysis_radius_m": round(float(radius_m), 1),
                    "road_guidance": {
                        "connected": bool(context_diagnostics["road_cell_count"]),
                        "line_count": context_diagnostics["road_line_count"],
                        "road_cell_count": context_diagnostics["road_cell_count"],
                    },
                    "hydro_guidance": {
                        "line_count": context_diagnostics["hydro_line_count"],
                        "hydro_cell_count": context_diagnostics["hydro_cell_count"],
                    },
                    "historical_flood": {
                        "event_count": len(history_points),
                        "nearby_event_count": nearby_history,
                    },
                    "factor_diagnostics": factor_diagnostics,
                    "inputs_applied": {
                        "cctv_location": True,
                        "cctv_observed_stage": True,
                        "surrounding_elevation": True,
                        "terrain_slope": True,
                        "river_and_sea": bool(context_diagnostics["hydro_cell_count"] or factor_diagnostics["hydro_dem_sea_fallback"]),
                        "road_connectivity": bool(context_diagnostics["road_cell_count"]),
                        "maximum_influence_radius": True,
                        "actual_flood_prone_mask": True,
                        "historical_flood_records": bool(nearby_history),
                    },
                })
            except Exception as exc:
                message = f"{source.get('name') or source.get('id')}: {exc}"
                errors.append(message)
                # Keep the confirmed CCTV visible while DEM/context data is
                # unavailable. This temporary anchor is replaced by the
                # detailed cached result on a later request.
                level = max(1, min(4, int(source.get("stage") or 1)))
                lon, lat = float(source["lon"]), float(source["lat"])
                radius_m = 22.0 + level * 6.0
                meters_lon = max(1000.0, 111320.0 * math.cos(math.radians(lat)))
                ring = [[
                    lon + math.cos(2 * math.pi * index / 24) * radius_m / meters_lon,
                    lat + math.sin(2 * math.pi * index / 24) * radius_m / 110540.0,
                    round(float(settings.dem_overlay_height_m), 2),
                ] for index in range(25)]
                depth_cm = float(source.get("depth_cm") or level_to_depth_cm(level))
                features.append({
                    "type": "Feature",
                    "properties": {
                        "map_id": f"dem-pending:{source.get('id')}",
                        "level": level, "depth_cm": depth_cm,
                        "depth_m": round(depth_cm / 100.0, 3),
                        "source_id": source.get("id"), "source_name": source.get("name"),
                        "source_lon": lon, "source_lat": lat,
                        "surface_pending": True, "fallback": True,
                        "fallback_reason": message[:240],
                    },
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                })
                source_results.append({
                    "id": source.get("id"), "name": source.get("name"),
                    "feature_count": 1, "surface_pending": True,
                    "inputs_applied": {"cctv_location": True, "cctv_observed_stage": True},
                })
        maximum_depth = max((float(f["properties"]["depth_cm"]) for f in features), default=0.0)
        return {
            "geojson": {"type": "FeatureCollection", "features": features},
            "source_count": len(selected),
            "feature_count": len(features),
            "maximum_depth_cm": round(maximum_depth, 1),
            "sources": source_results,
            "errors": errors,
            "method": "DEM terrain-depth bands with river/sea exclusion: CCTV, stage, elevation, slope, hydro, roads, radius, susceptibility and history",
            "dem_status": self.store.status(),
            "context_status": self.context.status(),
            "history_point_count": len(history_points),
            "model_schema": "terrain-depth-watermask-v8.6.3",
            "surface_pending": bool(errors),
        }


dem_store = DemTileStore()
terrain_context_store = VworldTerrainContextStore()
terrain_flood_model = TerrainFloodModel(dem_store, terrain_context_store)
