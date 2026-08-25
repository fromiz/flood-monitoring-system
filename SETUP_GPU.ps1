$ErrorActionPreference = "Stop"
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
# RTX 50-series/Windows: CUDA 12.8 PyTorch wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch=',torch.__version__); print('CUDA=',torch.version.cuda); print('Available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
Write-Host "설치 완료. 실행: python -m app.main"
