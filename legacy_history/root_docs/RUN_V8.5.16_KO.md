# V8.5.16 실행 방법

기존 V8.5.15의 `.env`만 새 폴더 최상위로 복사합니다.

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

접속: `http://127.0.0.1:8000`

서버를 완전히 재시작하고 브라우저에서 `Ctrl+F5`를 누르십시오. CCTV 창을 여러 개 열어도 포커스된 창 하나만 실시간 재생되며, 다른 창은 그래프를 유지하고 영상이 일시정지됩니다.

