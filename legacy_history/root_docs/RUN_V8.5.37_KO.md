# V8.5.37 GPU FULL 실행

기존 서버를 Ctrl+C로 종료한 뒤 새 폴더에서 실행합니다.

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.main
```

새 폴더에 `.venv`가 없다면 기존 CUDA Python 3.11 환경을 사용하거나 기존 설치 절차로 가상환경을 구성하세요.

브라우저에서 최초 한 번 Ctrl+F5를 누르세요.

정상적으로 CCTV 창을 열면 PowerShell에서 다음 진단이 보입니다.

```text
CCTV AI37 thread-start [...]
CCTV BOX37 raw=7 visible=4 stage=3 top=... floor=... ms=...
CCTV WS37 annotated=True ... ai_alive=True ai_state=published detmeta=4 result_age=...
```

박스가 나오지 않더라도 `AI37`, `BOX37`, `WS37` 줄만 보내면 AI thread 시작/프레임 대기/추론 시작/결과 publish 중 정확히 어디에서 멈췄는지 확인할 수 있습니다.
