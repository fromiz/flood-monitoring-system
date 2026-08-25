from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


BASE_URL = "http://127.0.0.1:8000"


def get_json(path: str) -> dict:
    with urlopen(BASE_URL + path, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    try:
        status = get_json("/api/cctv/background-status")
        stages = get_json("/api/cctv/stages")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"서버 연결 실패: {exc}") from exc

    print("백그라운드 CCTV AI")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print()
    print(
        "현재 유효 단계:",
        len(stages.get("items") or []),
        "개 / 만료 기준:",
        stages.get("stale_after_seconds"),
        "초",
    )


if __name__ == "__main__":
    main()
