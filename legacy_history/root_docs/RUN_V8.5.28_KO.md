# V8.5.28 GPU 실행

사용 중인 RTX 5060 Laptop GPU 환경에서는 GPU ZIP을 사용합니다.

## 기존 GPU 가상환경을 재사용하는 경우

V8.5.28 폴더에서 기존 CUDA 가상환경을 활성화한 뒤 실행해도 됩니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
C:\기존프로젝트\.venv\Scripts\Activate.ps1
cd C:\flood-monitor-v8.5.28-GPU
python -m app.main
```

## 새 가상환경을 만드는 경우

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` 설치 후 PyTorch가 CPU 전용으로 설치됐다면 CUDA wheel을 다시 설치하고 아래 결과가 `True`인지 확인합니다.

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

현재 확인된 사용자 환경은 PyTorch 2.11.0+cu128 / CUDA 12.8 / RTX 5060 Laptop GPU입니다.

## 실행

```powershell
python -m app.main
```

브라우저: `http://127.0.0.1:8000`

이전 서버가 8000번 포트를 쓰고 있으면 이전 PowerShell 창에서 `Ctrl+C`로 먼저 종료합니다. 버전 교체 후에는 브라우저에서 `Ctrl+F5`를 한 번 실행합니다.
