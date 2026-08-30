# CCTV 기반 실시간 도시 침수 분석 및 침수 지도 생성 시스템

신규 수위계를 설치하지 않고 기존 교통 CCTV 영상만으로 도로 침수 단계를 실시간 판정하고,
3D 지도 위에 위험 분포와 침수 지역 알림을 제공하는 통합 관제 시스템입니다.

| 항목 | 내용 |
|---|---|
| 기간 | 2026.07 ~ 2026.08 (7주, KDT 13기 팀 차오름 5인) |
| 최종 성능 | 바퀴탐지 mAP50 99.3% · 차종탐지 mAP50 89.4% · 차량침수분류 Accuracy 81.2% |
| 학습 데이터 | 87,000장 직접 구축 · 연동 CCTV 106개 지정 |
| 핵심 스택 | Python, PyTorch, Ultralytics(YOLO26s), MapLibre, GeoJSON, DEM, FastAPI |

📄 **[포트폴리오 PDF 보기](./docs/portfolio.pdf)**

담당: **PM · 대시보드 개발** — MapLibre 3D 침수 분석 지도, 침수 지역 알림·시계열 조회,
DEM 고도 결합 위험 분포 시각화, 기상청·브이월드 외부 API 연동

---

## 실행 방법

1. `.env.example`을 복사해 `.env` 생성 후 API 키(기상청, 브이월드 등) 입력
2. `pip install -r requirements.txt`
3. `python app/main.py` (또는 `docker-compose up`)

모델 가중치(`weights/*.pt`)는 용량 문제로 레포에 포함하지 않았습니다.

---

## 버전 히스토리 — V8.6.4 GPU FULL · SAFE MERGE

V8.6.4는 V8.6.3의 안정본을 기준으로 팀원 코드에서 검증 가치가 높은 부분만 선택 병합한 버전입니다.

**유지한 기준**
- V8.6.3 침수심 지도: DEM 고도별 Lev1~Lev4 색상 + 바다/강/하천 수면 제외
- V8.6.2 중앙 GPU single-owner scheduler + micro-batch + adaptive cadence
- `cudnn.benchmark=False`
- 낮은 raw 차량 confidence 정책(.env의 0.001)
- GPU `STAGE_TRACKING_MAX_MISSED_AI=0`
- live CCTV가 열려 있을 때 background AI를 멈추는 기존 우선순위
- FULL 파이프라인: best.pt → 차량 crop → tire_level.pt → 타이어 미검출 시 car_flood_cls.pt

**선택 병합**
1. track별 Lev 상태를 canonical state에 보존해 새 best.pt geometry가 들어와도 `Lev ↔ DET/HOLD` 깜빡임을 줄임
2. geometry/stage 상태 갱신용 `track_state_lock` 추가. 모델 predict와 JPEG 렌더링 중에는 잡지 않음
3. `stage_conf`와 `vehicle_conf`를 분리해 표시 신뢰도 혼동 방지
4. 박스 라벨에 `SUV`, `NORMAL CAR`, `TRUCK` 등 best.pt 차량 유형 표시
5. 표시 전용 velocity 보정 최대 0.12초. canonical geometry/AI 입력에는 절대 피드백하지 않음

**의도적으로 제외**
- 차량 detector confidence 0.25 강제
- 0.85 즉시 표시 threshold
- GPU missed track 1~2회 carry
- live 중 background GPU opportunistic 실행
- PyTorch 2.10 강제 requirements
- 팀원 ZIP의 API 키/DB/cache/__pycache__

상세 실행 안내는 `RUN_V8.6.4_KO.md`, 상세 변경 내역은 `V8.6.4_CHANGELOG_KO.md`를 확인하십시오.
