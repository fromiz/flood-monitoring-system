# V8.5.34 실행

기존 V8.5.33 서버를 `Ctrl+C`로 종료한 뒤 이 폴더에서 실행합니다.

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.main
```

새 폴더라면 `SETUP_GPU.ps1` 또는 기존 방식으로 CUDA PyTorch 환경을 준비하세요.

브라우저:
`http://127.0.0.1:8000`

JavaScript가 바뀌었으므로 첫 실행 후 `Ctrl+F5`를 1회 누르세요.
