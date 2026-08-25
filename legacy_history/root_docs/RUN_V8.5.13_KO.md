# V8.5.13 실행

1. `.env.example`을 복사해 `.env`를 만듭니다.
2. 기존에 사용하던 API 키를 새 `.env`에 직접 옮깁니다.
3. 프로젝트 폴더에서 기존 실행 방법 또는 `start.bat`을 사용합니다.
4. 브라우저 캐시를 무시하려면 `Ctrl+F5`로 한 번 새로고침합니다.

배포 ZIP에는 보안을 위해 `.env`, 데이터베이스, 지도 캐시와 API 키가 들어 있지 않습니다.

## 지도 입력 자료

- `VWORLD_API_KEY`: 브이월드 DEM/도로/수계 자료 요청
- `DEM_CONTEXT_DATA_URL`: 문맥 데이터 API 주소(기본 브이월드 Data API)
- `DEM_ROAD_DATA_LAYER`: 도로 레이어
- `DEM_HYDRO_DATA_LAYERS`: 하천·등고/수계 레이어 목록

외부 지도 API가 늦으면 먼저 CCTV 위치 기반 임시 면을 표시하고, 자료 준비가 끝날 때 자동 갱신합니다.
