# V8.6.1 GPU FULL 실행

기존 서버를 `Ctrl+C`로 완전히 종료한 뒤 실행하세요.

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.main
```

브라우저는 `http://127.0.0.1:8000` 이며 첫 실행에 `Ctrl+F5`를 한 번 권장합니다.

PowerShell에서 먼저 다음 줄을 확인하세요.

```text
CUDA61 tuning cudnn_benchmark=False ...
Central GPU scheduler started: single predict owner
```

정상적으로 박스가 갱신되면 `CCTV BOX61` 로그가 짧은 `ms` 값으로 반복됩니다.

2초 이상 지연이 생기면 `AI SCHED61 slow`와 `DET61 slow` 줄 전체를 보내주세요. `run_ms`, `queue_ms`, `full_ms`, `rescue_ms`, `clahe_ms`가 분리돼 있어 다음 병목을 특정할 수 있습니다.
