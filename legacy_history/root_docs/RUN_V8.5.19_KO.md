# V8.5.19 실행 방법

이 ZIP은 V8.5.18을 기준으로 CCTV 영상 연결과 대시보드 `Failed to fetch` 복구를 보강한 전체 수정본입니다. `.env`와 초기화 DB가 포함되어 있습니다.

## 처음 한 번 설치 (Windows PowerShell)

압축을 푼 **V8.5.19 폴더 안에서** 아래 명령을 실행합니다.

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

설치 후에는 `START_V8.5.19.bat`를 더블클릭하면 프로젝트 폴더로 자동 이동한 뒤 서버를 실행합니다.

직접 실행하려면:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

접속 주소: `http://127.0.0.1:8000`

## 기존 V8.5.18에서 교체할 때

1. 기존 V8.5.18 서버 창을 완전히 종료합니다.
2. V8.5.19 ZIP을 새 폴더에 풀어 실행합니다. 이전 폴더 위에 덮어쓰는 방식은 피하십시오.
3. 브라우저에서 `Ctrl+F5`로 캐시를 한 번 비웁니다.
4. CCTV 팝업을 열어 실제 영상 프레임이 움직이는지 확인합니다.
5. 상단 상태의 `백그라운드 CCTV AI`, `환경 DB 기록`, `통합 침수 GeoJSON` 항목은 일시적인 통신 지연 때 재시도하며, 정상 응답을 받으면 상태가 자동 복구됩니다.

## 참고

- `.env`의 `BACKGROUND_CCTV_WORKERS=1`은 열린 CCTV 영상의 안정성과 응답성을 우선하기 위한 기본값입니다.
- 포항 공개 CCTV는 외부 제공 서버의 실제 접속 가능 여부에 영향을 받습니다. 외부 원본 자체가 중단된 경우 프로그램은 재연결 상태 화면을 표시하고 계속 재시도합니다.
- 프로그램을 다른 작업 폴더에서 실행하면 상대 경로 DB/정적 파일을 잘못 찾을 수 있으므로 제공된 `START_V8.5.19.bat` 사용을 권장합니다.
