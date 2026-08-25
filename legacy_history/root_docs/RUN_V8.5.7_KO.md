# V8.5.7 실행 방법

이 배포본에는 개인 API 키 보호를 위해 `.env`가 들어 있지 않습니다.

## Windows PowerShell

```powershell
cd "flood-monitor-v4-v8.5.7-live-stream-tracking-weather-fix"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m app.main
```

브라우저에서 `http://localhost:8000`을 열고 `Ctrl+F5`를 한 번 누릅니다.

## API 키 입력

생성된 `.env`에 본인이 사용하는 키만 입력합니다.

```env
KMA_SERVICE_KEY=
KMA_APIHUB_AUTH_KEY=
SEWER_API_KEY=
RIVER_API_KEY=
VWORLD_API_KEY=
```

`KMA_APIHUB_AUTH_KEY` 하나만 입력해도 AWS 매분자료를 먼저 사용하고,
기온·습도·풍속이 누락된 경우 같은 키로 APIHub 정시자료를 보완합니다.

## 확인 주소

- 대시보드: `http://localhost:8000`
- 상태: `http://localhost:8000/api/health`
- 실시간 날씨: `http://localhost:8000/api/weather/live`
- 모델: `http://localhost:8000/api/stage-model`
- CCTV 지점: `http://localhost:8000/api/cctv/pohang`
- 지도 오버레이: `http://localhost:8000/api/map/vworld-overlays`
- 작업자: `http://localhost:8000/api/cctv/background-status`

지도 오버레이 응답의 `camera_count`가 1 이상이면 지점 좌표가 전달된
것입니다. 침수심 면은 실제 AI 단계가 Lev1 이상인 지점이 있을 때만
생성됩니다. 모든 지점이 Lev0이면 지점 마커만 표시되고 침수면은 비어
있는 것이 정상입니다.
