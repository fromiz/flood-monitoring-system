# V8.5.20 실행 방법

이 ZIP은 V8.5.19에서 확인된 **CCTV 창이 `CCTV 연결 중` 화면에 오래 가려지는 문제**와 실시간 HLS 전환 지연을 수정한 전체 파일입니다. `.env`, 초기화 DB, 테스트 CCTV 영상이 포함되어 있습니다.

## 처음 한 번 설치 (Windows PowerShell)

압축을 푼 **V8.5.20 폴더 안에서** 아래 명령을 실행합니다.

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

설치 후 `START_V8.5.20.bat`를 더블클릭합니다.

접속 주소: `http://127.0.0.1:8000`

## V8.5.19에서 교체할 때

1. 실행 중인 V8.5.19 서버 창을 완전히 종료합니다.
2. V8.5.20 ZIP을 **새 폴더**에 압축 해제합니다.
3. 기존 `.venv`를 새 폴더로 복사하지 않았다면 위의 최초 설치를 진행합니다.
4. `START_V8.5.20.bat`로 실행합니다.
5. 브라우저에서 `Ctrl+F5`를 한 번 눌러 V8.5.20 JavaScript/CSS를 새로 받습니다.
6. 먼저 `테스트 CCTV`를 열어 영상이 바로 표시되고 움직이는지 확인합니다.
7. 이후 실제 포항 CCTV를 열어 확인합니다.

## 이번 버전에서 CCTV가 보이는 방식

- CCTV 창을 열면 **원본(raw) 영상 연결을 계속 유지**합니다.
- 포커스된 창은 AI 영상이 준비될 때까지 raw 영상이 계속 보이고, AI 첫 프레임이 준비된 뒤에만 AI 화면으로 전환합니다.
- Chromium/Chrome의 MJPEG `img.onload`가 발생하지 않는 경우에도 `naturalWidth/naturalHeight`를 0.1초 간격으로 확인해 첫 JPEG가 디코딩되는 즉시 `CCTV 연결 중` 가림막을 제거합니다.
- HLS fallback이 한 번 성공하면 30초마다 OpenCV 직접 연결을 다시 시도해 영상을 끊던 동작을 하지 않고, 동작 중인 HLS 경로를 계속 사용합니다.
- HLS master playlist를 매 세그먼트마다 다시 받지 않고 확인된 media playlist를 재사용해 다음 영상 조각을 더 빨리 가져옵니다.

## 참고

실제 포항 CCTV 원본 서버 자체가 중단되거나 외부망에서 접근할 수 없는 카메라는 프로그램에서 영상을 생성할 수 없습니다. 이 경우에는 영상 영역 안에 재연결 상태 JPEG가 표시되고 자동으로 계속 재시도합니다.
