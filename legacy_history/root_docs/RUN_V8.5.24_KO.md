# V8.5.24 실행 방법 (VS Code / PowerShell)

이번 버전은 사용자가 실제로 정상 실행한 `python -m app.main` 방식을 기준으로 합니다.

## 처음 한 번만

VS Code에서 이 폴더를 열고 PowerShell 터미널에서 실행합니다.

```powershell
py -3.11 -m venv .venv

Set-ExecutionPolicy `
  -Scope Process `
  -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 평소 실행

```powershell
Set-ExecutionPolicy `
  -Scope Process `
  -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
python -m app.main
```

브라우저 주소:

```text
http://127.0.0.1:8000
```

V8.5.23을 실행 중이었다면 먼저 기존 서버를 `Ctrl+C`로 완전히 종료한 뒤 V8.5.24를 실행합니다. 처음 접속할 때는 브라우저에서 `Ctrl+F5`를 한 번 눌러 V8.5.24 JavaScript를 새로 불러오십시오.

> `pip install -r requirements.txt`는 새 가상환경을 만든 첫 실행 때만 필요합니다. 평소에는 가상환경 활성화 후 `python -m app.main`만 실행하면 됩니다.
