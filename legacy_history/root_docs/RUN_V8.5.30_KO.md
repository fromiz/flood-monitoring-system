# V8.5.31 GPU 실행

PowerShell에서 프로젝트 폴더로 이동한 뒤 기존 CUDA 가상환경을 사용하세요.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python -m app.main
```

브라우저: http://127.0.0.1:8000

기존 서버가 8000 포트를 쓰면 먼저 Ctrl+C로 종료하세요.
