from __future__ import annotations

import argparse
import json
from urllib.parse import urlencode
from urllib.request import urlopen


def get_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="V8.5.1 통합 침수/환경 이력 확인")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--camera-id", default="TEST-FLOOD-01")
    parser.add_argument("--hours", type=int, default=1)
    args = parser.parse_args()

    status = get_json(f"{args.base_url}/api/environment-history/status")
    print("환경 DB 실행:", status.get("running"))
    print("환경 DB 누계 저장:", status.get("stored"))
    print("최근 저장 건수:", status.get("last_counts"))
    print("최근 오류:", status.get("last_error"))

    query = urlencode({
        "camera_id": args.camera_id,
        "hours": max(1, args.hours),
        "bucket_minutes": 1,
    })
    history = get_json(f"{args.base_url}/api/history?{query}")
    print("침수 원본 행:", history.get("total_rows"))
    print("침수 구간:", len(history.get("points") or []))
    print("통합 구간:", len(history.get("combined_points") or []))
    for kind, label in (("rain", "강수"), ("sewer", "하수"), ("river", "하천")):
        series = (history.get("environment") or {}).get(kind) or {}
        print(
            f"{label}:",
            series.get("sensor_name"),
            "거리", series.get("distance_m"),
            "기록", len(series.get("points") or []),
        )

    combined = history.get("combined_points") or []
    if combined:
        print("최근 통합값:", combined[-1])


if __name__ == "__main__":
    main()
