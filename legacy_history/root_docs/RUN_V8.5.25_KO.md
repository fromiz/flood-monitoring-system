# V8.5.25 실행 방법

## 공통

Python 3.11 가상환경을 사용합니다.

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m app.main
```

웹 주소: `http://127.0.0.1:8000`

## GPU ZIP 확인

GPU ZIP은 `.env`에 `AI_DEVICE=cuda`가 들어 있습니다. 실행 전에 아래 명령으로 CUDA가 True인지 확인하세요.

```powershell
python -c "import torch; print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

`cuda=False`라면 현재 가상환경의 PyTorch가 GPU를 사용하지 못하는 상태입니다. 그 경우 CPU ZIP을 사용하거나 사용 중인 NVIDIA 드라이버/CUDA에 맞는 PyTorch 설치를 먼저 구성해야 합니다.
