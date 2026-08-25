# V8.5.21 실행 방법

이 ZIP은 V8.5.20을 기반으로 **CCTV 다중 창 스트리밍 지연/미시작**, **자동 정렬 후 기록 스크롤 하단 잘림**, **침수 단계 차량 박스 누락/깜박임**을 수정한 전체 파일입니다. `.env`, 초기화 DB, 테스트 CCTV 영상이 포함되어 있습니다.

## 처음 한 번 설치 (Windows PowerShell)

압축을 푼 **V8.5.21 폴더 안에서** 아래 명령을 실행합니다.

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

설치 후 `START_V8.5.21.bat`를 더블클릭합니다.

접속 주소: `http://127.0.0.1:8000`

## V8.5.20에서 교체할 때

1. 실행 중인 V8.5.20 서버 창을 완전히 종료합니다.
2. V8.5.21 ZIP을 **새 폴더**에 압축 해제합니다.
3. 기존 `.venv`를 새 폴더로 복사하지 않았다면 위의 최초 설치를 진행합니다.
4. `START_V8.5.21.bat`로 실행합니다.
5. 브라우저에서 `Ctrl+F5`를 한 번 눌러 V8.5.21 JavaScript/CSS를 새로 받습니다.
6. `테스트 CCTV`와 실제 CCTV를 여러 개 열고 `CCTV 자동 정렬`을 눌러 확인합니다.

## 이번 버전의 CCTV 연결 방식

- **선택된 CCTV 한 개만** 지속형 AI MJPEG 연결을 사용합니다.
- 나머지 열린 CCTV는 `/api/cctv/frame-raw`의 짧은 JPEG 요청을 반복하여 화면을 갱신합니다. 따라서 여러 창이 브라우저의 지속 연결 슬롯을 전부 차지하지 않습니다.
- 선택된 CCTV의 AI 화면이 준비되기 전에는 raw JPEG가 아래에 계속 표시되어 검은 화면을 줄입니다.
- CCTV 선택이 바뀌면 이전 AI MJPEG를 먼저 종료하고 새 AI MJPEG를 시작합니다.
- 포항 HTTP CCTV는 OpenCV timeout보다 **HLS playlist/segment 경로를 먼저 빠르게 시도**합니다. HLS가 아닌 경우에만 제한된 직접 연결로 전환합니다.
- CCTV 창을 잠깐 닫았다 다시 열 때 재접속 시간이 줄도록 worker를 약 8초간 warm 상태로 유지합니다.

## 차량 박스 확인

- `best.pt` 차량 박스 검출은 `VEHICLE_DETECTION_IMGSZ=960` 설정을 실제로 사용합니다.
- 원거리/작은 차량 수가 적으면 3분할 tiled rescue를 주기적으로 다시 수행합니다.
- CPU 추론이 0.65초보다 오래 걸리더라도 박스가 바로 사라지지 않도록 detector 실제 지연시간에 맞춰 optical-flow 유지시간을 조절합니다.

## 참고

실제 포항 CCTV 원본 서버 자체가 중단되거나 외부망에서 접근할 수 없는 카메라는 프로그램에서 영상을 생성할 수 없습니다. 이 경우 영상 영역 안에 재연결 상태 JPEG가 표시되고 자동으로 계속 재시도합니다.
