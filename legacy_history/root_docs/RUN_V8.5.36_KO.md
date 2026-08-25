# V8.5.36 GPU FULL 실행

1. 기존 서버를 `Ctrl+C`로 완전히 종료합니다.
2. 기존 CUDA 가상환경을 사용하거나 새 `.venv`를 활성화합니다.
3. `python -m app.main` 실행 후 `http://127.0.0.1:8000` 접속합니다.
4. 첫 실행에서 `Ctrl+F5`를 한 번 누릅니다.

## 이번 버전 핵심
- V8.5.35의 browser-only 차량 박스 렌더를 제거했습니다.
- 차량 박스는 서버 OpenCV JPEG에 **한 번만** 그립니다.
- WebSocket detection metadata는 진단용으로 유지하지만 브라우저는 두 번째 박스를 그리지 않습니다.
- 버퍼링이 해결된 HLS/전송 로직은 변경하지 않았습니다.
- FULL 파이프라인 `best.pt -> crop -> tire_level.pt -> (no tire only) car_flood_cls.pt`를 유지합니다.

## PowerShell 진단
AI가 활성화되면 5초 간격으로 아래 중 하나가 보입니다.

`CCTV BOX36 raw=... visible=... stage=... top=... floor=... ms=...`

WebSocket 쪽에서는:

`CCTV WS36 annotated=True ... detmeta=... result_age=...`

- `BOX36 visible > 0`이면 서버 JPEG에도 박스가 그려져야 합니다.
- `BOX36 visible = 0`이면 detector/filter 문제입니다.
- `WS36 annotated=True`인데 `BOX36`이 전혀 없으면 AI loop gating 문제입니다.
