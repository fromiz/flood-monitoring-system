# V8.5.36 GPU 실행

1. 기존 서버를 Ctrl+C로 완전히 종료합니다.
2. 이 폴더의 `.venv`를 사용하거나 CUDA PyTorch 12.8 환경을 활성화합니다.
3. `python -m app.main`
4. 브라우저에서 `http://127.0.0.1:8000` 접속 후 최초 1회 Ctrl+F5.

박스가 보이지 않으면 PowerShell의 `CCTV best.pt ...` 경고 한 줄과 `/api/cctv/worker-status` 결과를 보내주세요.
