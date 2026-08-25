# V8.6.4 GPU FULL SAFE MERGE 실행

기존 서버를 먼저 `Ctrl+C`로 종료하십시오.

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.main
```

브라우저: `http://127.0.0.1:8000`

업데이트 후 `Ctrl+F5`를 한 번 실행하십시오.

확인 포인트:
- 차량 박스가 기존 V8.6.3처럼 계속 표시되는지
- Lev 판정 뒤 새 best.pt 갱신 때 `DET`로 잠깐 돌아가지 않는지
- 차량 라벨이 SUV/NORMAL CAR/TRUCK 등으로 표시되는지
- 박스 위치 보정이 과하게 앞서가지 않는지(기본 최대 0.12초)
- 침수심 지도 고도별 색상과 바다/강 제외가 그대로 유지되는지

보정이 불편하면 `.env`에서 아래만 끄면 됩니다.
```env
VEHICLE_DISPLAY_PROJECTION_ENABLED=false
```
나머지 AI 구조에는 영향이 없습니다.
