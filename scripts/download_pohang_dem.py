from __future__ import annotations

import json
import sys
from pathlib import Path

# scripts 폴더에서 직접 실행해도 프로젝트의 app 패키지를 찾도록
# 프로젝트 루트를 Python 모듈 검색 경로의 맨 앞에 추가합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dem_terrain import DemUnavailable, dem_store


def main() -> int:
    dem_store.clear_failures()

    try:
        result = dem_store.prefetch()
    except DemUnavailable as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "hint": (
                        "인터넷 연결과 "
                        "https://tiles.mapterhorn.com 접속을 확인하세요."
                    ),
                    "status": dem_store.status(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    result["ok"] = result.get("failed", 0) == 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
