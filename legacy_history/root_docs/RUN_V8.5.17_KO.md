# V8.5.17 실행 방법

이 배포본에는 `.env`와 초기화 DB가 포함되어 있어 별도 복사 없이 실행할 수 있습니다.

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

브라우저 접속: `http://127.0.0.1:8000`

서버를 완전히 재시작하고 `Ctrl+F5`를 누르십시오.

