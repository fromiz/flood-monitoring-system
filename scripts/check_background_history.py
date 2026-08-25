from __future__ import annotations

import argparse
import json
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="창을 열지 않은 CCTV의 백그라운드 저장·이력 조회 상태 확인"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--camera-id", default="TEST-FLOOD-01")
    parser.add_argument("--hours", type=int, default=1)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    status = fetch_json(f"{base}/api/cctv/background-status")
    query = urlencode(
        {
            "camera_id": args.camera_id,
            "hours": max(1, args.hours),
            "bucket_minutes": 1,
        }
    )
    history = fetch_json(f"{base}/api/history?{query}")

    camera_status = (status.get("camera_status") or {}).get(args.camera_id) or {}
    continuous = status.get("continuous_local") or {}
    print("백그라운드 실행:", status.get("running"))
    print("순환:", status.get("cycle"))
    print("이번 순환 저장:", status.get("stored"))
    print("테스트 지속 저장 실행:", continuous.get("running"))
    print("테스트 지속 저장 누계:", continuous.get("stored"))
    print("테스트 최근 저장:", continuous.get("last_recorded_at"))
    print("테스트 최근 단계:", continuous.get("last_stage"))
    print("카메라 원본 정상:", camera_status.get("stream_ok"))
    print("카메라 마지막 저장:", camera_status.get("last_recorded_at"))
    print("조회 원본 행:", history.get("total_rows"))
    print("출처별:", history.get("source_counts"))
    print("그래프 구간:", len(history.get("points") or []))

    points = history.get("points") or []
    if int(history.get("source_counts", {}).get("background") or 0) <= 0:
        raise SystemExit(
            "백그라운드 기록이 없습니다. 서버 시작 후 20초 뒤 다시 확인하세요."
        )

    for point in points[-10:]:
        print(
            point.get("time"),
            f"Lev{point.get('level')}",
            point.get("record_source"),
        )


if __name__ == "__main__":
    main()
