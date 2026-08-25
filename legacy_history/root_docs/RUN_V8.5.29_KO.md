# V8.5.31 GPU 실행 방법

기존 서버를 먼저 `Ctrl+C`로 종료하세요.

PowerShell에서 프로젝트 폴더로 이동한 뒤 기존 가상환경이 있으면:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m app.main
```

새 가상환경이면:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

GPU 확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

RTX GPU에서 `True`가 나온 뒤:

```powershell
python -m app.main
```

브라우저: http://127.0.0.1:8000

버전 교체 후 처음 한 번 `Ctrl+F5`를 권장합니다.
