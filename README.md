# 포항 침수 통합관제 V8.6.4 GPU FULL · SAFE MERGE

V8.6.4는 V8.6.3의 안정본을 기준으로 팀원 코드에서 검증 가치가 높은 부분만 선택 병합한 버전입니다.

유지한 기준:
- V8.6.3 침수심 지도: DEM 고도별 Lev1~Lev4 색상 + 바다/강/하천 수면 제외
- V8.6.2 중앙 GPU single-owner scheduler + micro-batch + adaptive cadence
- `cudnn.benchmark=False`
- 낮은 raw 차량 confidence 정책(.env의 0.001)
- GPU `STAGE_TRACKING_MAX_MISSED_AI=0`
- live CCTV가 열려 있을 때 background AI를 멈추는 기존 우선순위
- FULL 파이프라인: best.pt → 차량 crop → tire_level.pt → 타이어 미검출 시 car_flood_cls.pt

선택 병합:
1. track별 Lev 상태를 canonical state에 보존해 새 best.pt geometry가 들어와도 `Lev ↔ DET/HOLD` 깜빡임을 줄임
2. geometry/stage 상태 갱신용 `track_state_lock` 추가. 모델 predict와 JPEG 렌더링 중에는 잡지 않음
3. `stage_conf`와 `vehicle_conf`를 분리해 표시 신뢰도 혼동 방지
4. 박스 라벨에 `SUV`, `NORMAL CAR`, `TRUCK` 등 best.pt 차량 유형 표시
5. 표시 전용 velocity 보정 최대 0.12초. canonical geometry/AI 입력에는 절대 피드백하지 않음

의도적으로 제외:
- 차량 detector confidence 0.25 강제
- 0.85 즉시 표시 threshold
- GPU missed track 1~2회 carry
- live 중 background GPU opportunistic 실행
- PyTorch 2.10 강제 requirements
- 팀원 ZIP의 API 키/DB/cache/__pycache__

실행은 `RUN_V8.6.4_KO.md`, 상세 변경은 `V8.6.4_CHANGELOG_KO.md`를 확인하십시오.
