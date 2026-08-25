# V8.5.26 실행 방법

## GPU 버전 (RTX 5060 Laptop GPU / CUDA 12.8 확인 환경)

PowerShell에서 프로젝트 폴더로 이동한 뒤:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python -m app.main
```

새 가상환경을 처음 만드는 경우 `SETUP_GPU.ps1`을 실행할 수 있습니다.

GPU 확인 결과는 `CUDA available=True`여야 합니다.

## CPU 버전

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m app.main
```

새 가상환경이면 `SETUP_CPU.ps1`을 사용하세요.

웹 주소: http://127.0.0.1:8000
