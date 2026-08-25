# V8.5.15 실행 방법

## Windows PowerShell

프로젝트 최상위 폴더에서 실행합니다.

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

이미 가상환경과 패키지가 설치되어 있으면 다음만 실행합니다.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`에 접속하고 최초 1회 `Ctrl+F5`를 누릅니다.
기존 V8.5.14의 `.env`만 새 프로젝트 최상위 폴더로 복사하십시오. ZIP에는 `.env`, API 키, 실행 DB와 캐시가 포함되지 않습니다.

