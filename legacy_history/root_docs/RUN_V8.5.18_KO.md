# V8.5.18 실행 방법

이 ZIP에는 `.env`와 초기화 DB가 포함되어 있습니다.

## Windows PowerShell

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

이미 설치가 끝난 경우:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

접속 주소: `http://127.0.0.1:8000`

이전 서버 프로세스를 완전히 종료한 다음 V8.5.18을 실행하고 브라우저에서 `Ctrl+F5`를 누르십시오.

