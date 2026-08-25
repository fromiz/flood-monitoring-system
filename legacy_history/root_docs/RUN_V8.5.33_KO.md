# V8.5.33 실행

기존 서버를 Ctrl+C로 종료한 뒤 PowerShell에서 프로젝트 폴더로 이동합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m app.main
```

브라우저: http://127.0.0.1:8000
처음 한 번 Ctrl+F5를 누르세요.
