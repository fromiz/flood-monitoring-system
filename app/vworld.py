from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import settings


ALLOWED_LAYERS = {
    "Base": ("png", "image/png"),
    "Satellite": ("jpeg", "image/jpeg"),
    "Hybrid": ("png", "image/png"),
}


@dataclass(frozen=True)
class TileResult:
    content: bytes
    media_type: str
    cache_status: str


def map_config() -> dict:
    return {
        "provider": "VWorld",
        "configured": bool(settings.vworld_api_key.strip()),
        "proxy": True,
        "layers": ["Base", "Satellite", "Hybrid"],
        "default_layer": "Hybrid",
        "attribution": "공간정보 오픈플랫폼(브이월드)",
        "tile_cache_seconds": settings.vworld_tile_cache_seconds,
        "message": (
            "브이월드 WMTS 프록시 정상"
            if settings.vworld_api_key.strip()
            else ".env의 VWORLD_API_KEY를 입력하세요."
        ),
    }


def _validate(layer: str, z: int, x: int, y: int) -> tuple[str, str]:
    if layer not in ALLOWED_LAYERS:
        raise ValueError("지원하지 않는 브이월드 레이어입니다.")
    if not 0 <= z <= 19:
        raise ValueError("브이월드 줌 레벨은 0~19만 지원합니다.")
    limit = 1 << z
    if not (0 <= x < limit and 0 <= y < limit):
        raise ValueError("잘못된 타일 좌표입니다.")
    return ALLOWED_LAYERS[layer]


def _cache_path(layer: str, z: int, x: int, y: int, ext: str) -> Path:
    return Path("data/vworld_tiles") / layer / str(z) / str(x) / f"{y}.{ext}"


def fetch_tile(
    layer: str,
    z: int,
    x: int,
    y: int,
    referer: str = "",
) -> TileResult:
    ext, media_type = _validate(layer, z, x, y)
    key = settings.vworld_api_key.strip()
    if not key:
        raise RuntimeError("VWORLD_API_KEY가 설정되지 않았습니다.")

    cache_path = _cache_path(layer, z, x, y, ext)
    now = time.time()
    if cache_path.exists():
        age = now - cache_path.stat().st_mtime
        if age <= max(60, settings.vworld_tile_cache_seconds):
            return TileResult(cache_path.read_bytes(), media_type, "HIT")

    # MapLibre의 XYZ 순서(z/x/y)를 브이월드 WMTS 순서(z/y/x)로 바꿉니다.
    url = (
        "https://api.vworld.kr/req/wmts/1.0.0/"
        f"{key}/{layer}/{z}/{y}/{x}.{ext}"
    )
    upstream_referer = settings.vworld_referer.strip() or referer
    headers = {
        "User-Agent": "PohangFloodControl/7.4",
        "Accept": media_type + ",image/*;q=0.8,*/*;q=0.5",
    }
    if upstream_referer:
        headers["Referer"] = upstream_referer

    try:
        request = Request(url, headers=headers)
        with urlopen(
            request,
            timeout=max(3, settings.vworld_request_timeout_seconds),
        ) as response:
            content = response.read()
            response_type = response.headers.get_content_type()

        if not content:
            raise RuntimeError("브이월드에서 빈 타일을 반환했습니다.")
        if response_type and not response_type.startswith("image/"):
            preview = content[:200].decode("utf-8", errors="replace")
            raise RuntimeError(f"브이월드 타일 응답 오류: {preview}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temp.write_bytes(content)
        temp.replace(cache_path)
        return TileResult(content, media_type, "MISS")
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        # 상류 API가 일시 중단되면 만료된 캐시라도 마지막 정상 타일을 제공합니다.
        if cache_path.exists():
            return TileResult(cache_path.read_bytes(), media_type, "STALE")
        raise RuntimeError(f"브이월드 타일 조회 실패: {exc}") from exc
