# V8.5.27 GPU 실행 방법

## 1. VS Code에서 프로젝트 폴더 열기
압축을 푼 폴더에서 `app`, `weights`, `requirements.txt`가 바로 보여야 합니다.

## 2. 가상환경
```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 3. GPU PyTorch
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```
`CUDA=True`가 나와야 합니다.

## 4. 기상청 인증키
`.env.local.example`을 `.env.local`로 복사하고 실제 키를 입력하세요.
APIHub 키가 URL 인코딩된 형태여도 V8.5.27에서 한 번만 정상 인코딩합니다.
`KMA_APIHUB_AUTH_KEY`가 실패하고 `KMA_SERVICE_KEY`도 설정되어 있으면 공공데이터포털 관측값으로 폴백합니다.

## 5. 실행
```powershell
python -m app.main
```
브라우저: `http://127.0.0.1:8000`
