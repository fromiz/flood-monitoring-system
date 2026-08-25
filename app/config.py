from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


Point = Tuple[float, float]


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    name: str
    site_name: str
    address: str
    source: str
    roi: List[Point]
    lat: float
    lon: float


def parse_roi(value: str | None) -> List[Point]:
    if not value:
        return []
    points: List[Point] = []
    for pair in value.split(";"):
        x_str, y_str = pair.split(",", maxsplit=1)
        x, y = float(x_str), float(y_str)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("ROI 좌표는 0~1 범위여야 합니다.")
        points.append((x, y))
    if points and len(points) < 3:
        raise ValueError("ROI는 최소 3개 점이 필요합니다.")
    return points


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="allow")

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    database_url: str = "sqlite:///./data/flood_monitor.db"

    model_path: str = ""
    model_confidence: float = 0.60
    model_iou: float = 0.45
    device: str = ""
    demo_mode: bool = True

    alert_min_level: int = 2
    alert_consecutive_frames: int = 5
    event_cooldown_seconds: int = 30
    legacy_camera_workers_enabled: bool = False
    jpeg_quality: int = 80
    process_every_n_frames: int = 1
    ai_cpu_threads: int = 3
    # AI_DEVICE=auto|cpu|cuda. CPU/GPU 배포 ZIP에서 각각 고정합니다.
    ai_device: str = "auto"

    # V8.6.2 central inference scheduler. All CUDA predict() calls are owned by
    # one scheduler thread; these values tune fairness/micro-batching without
    # hard-coding timing thresholds in the worker implementation.
    ai_scheduler_vehicle_burst: int = 4
    ai_scheduler_micro_batch_wait_ms: float = 10.0
    ai_scheduler_max_vehicle_batch: int = 4

    # Scheduler-aware detector cadence. V8.6.1 proved the RTX 5060 usually
    # finishes best.pt in ~40-110 ms with qv almost always 0. Rather than
    # visually extrapolating stale boxes, V8.6.2 asks best.pt for fresh geometry
    # more often while the central queue is healthy, and automatically backs off
    # before the faster cadence can starve tire/body classification.
    ai_adaptive_cadence_enabled: bool = True
    ai_geometry_fast_interval_seconds: float = 0.18
    ai_geometry_busy_interval_seconds: float = 0.34
    ai_geometry_nonfocused_multiplier: float = 1.12
    ai_geometry_fast_max_vehicle_queue: int = 1
    ai_geometry_busy_vehicle_queue: int = 3
    ai_geometry_fast_max_stage_queue: int = 2
    ai_geometry_busy_stage_queue: int = 4
    ai_geometry_fast_max_vehicle_ms: float = 130.0
    ai_geometry_busy_vehicle_ms: float = 220.0
    ai_geometry_fast_max_queue_ms: float = 120.0
    ai_geometry_busy_queue_ms: float = 220.0

    # V8.6.4: optional display-only bbox latency compensation. This never feeds
    # projected geometry back into tracking or stage inference. Keep the horizon
    # deliberately short; fresh best.pt geometry always replaces it.
    vehicle_display_projection_enabled: bool = True
    vehicle_display_projection_max_seconds: float = 0.12

    # 공공데이터포털 단기예보 조회서비스
    kma_service_key: str = ""
    kma_nx: int = 102
    kma_ny: int = 94
    kma_station_name: str = "포항시청"
    weather_refresh_seconds: int = 600

    # 기상청 API허브: 포항 AWS 매분자료 + 500m 고해상도 강수 격자
    kma_apihub_auth_key: str = ""
    kma_aws_station_id: int = 138
    kma_aws_station_name: str = "포항"
    kma_aws_station_lat: float = 36.03201
    kma_aws_station_lon: float = 129.38002
    kma_aws_poll_seconds: int = 60
    kma_grid_poll_seconds: int = 300
    kma_forecast_poll_seconds: int = 600
    kma_weather_stale_seconds: int = 180

    # 브이월드 WMTS/TMS. 인증키는 백엔드 프록시에서만 사용합니다.
    vworld_api_key: str = ""
    vworld_referer: str = ""
    vworld_tile_cache_seconds: int = 86400
    vworld_request_timeout_seconds: int = 12

    # 브이월드 CCTV 화면 마커의 지면 고정·가림 판정
    vworld_cctv_ground_offset_m: float = 0.30
    vworld_cctv_occlusion_tolerance_m: float = 8.0
    vworld_cctv_occlusion_check_frames: int = 12
    vworld_cctv_far_scale_distance_m: float = 18000.0

    # 실제 지형 기반 침수 분석용 DEM 타일
    dem_enabled: bool = True
    dem_tilejson_url: str = "https://tiles.mapterhorn.com/tilejson.json"
    dem_tile_url_template: str = "https://tiles.mapterhorn.com/{z}/{x}/{y}.webp"
    dem_encoding: str = "terrarium"
    dem_tile_size: int = 512
    dem_zoom: int = 12
    dem_cache_dir: str = "data/dem"
    flood_surface_cache_path: str = "data/flood_surface_cache.json"
    dem_request_timeout_seconds: int = 20
    dem_context_data_url: str = "https://api.vworld.kr/req/data"
    dem_road_data_layer: str = "LT_L_MOCTLINK"
    dem_hydro_data_layers: str = "LT_C_WKMSTRM,LT_L_TOISDEPCNTAH"
    dem_context_cache_dir: str = "data/terrain_context"
    dem_context_timeout_seconds: int = 2
    dem_hydro_influence_m: float = 180.0
    # V8.6.3: flood polygons represent inundated land only. VWorld hydro
    # centerlines are buffered and DEM sea-level cells connected to the grid
    # edge are masked from the rendered flood surface.
    dem_water_exclusion_enabled: bool = True
    dem_river_exclusion_buffer_m: float = 35.0
    dem_sea_exclusion_max_elevation_m: float = 0.35
    dem_history_lookback_days: int = 365
    dem_history_influence_m: float = 220.0
    dem_history_weight: float = 0.14
    dem_flood_prone_min_score: float = 0.26
    dem_memory_tile_count: int = 48
    dem_pohang_west: float = 129.20
    dem_pohang_south: float = 35.88
    dem_pohang_east: float = 129.67
    dem_pohang_north: float = 36.32
    dem_flood_cell_m: float = 15.0
    dem_flood_base_radius_m: float = 520.0
    dem_flood_max_radius_m: float = 1000.0
    dem_flood_max_grid_cells: int = 101
    dem_terrain_valley_weight: float = 0.45
    dem_radial_depth_decay: float = 0.08
    dem_source_snap_radius_m: float = 90.0
    dem_cctv_snap_radius_m: float = 0.0
    dem_rain_snap_radius_m: float = 90.0
    dem_source_merge_m: float = 60.0
    dem_max_sources: int = 24
    dem_runoff_coefficient: float = 0.68
    dem_ai_support_radius_m: float = 115.0
    dem_min_volume_m3: float = 250.0
    dem_max_water_depth_m: float = 1.2
    dem_flow_friction_m_per_km: float = 0.12
    dem_ponding_slope_scale: float = 0.075
    dem_min_polygon_cells: float = 2.0
    dem_min_footprint_l1_cells: int = 5
    dem_min_footprint_l2_cells: int = 10
    dem_min_footprint_l3_cells: int = 18
    dem_min_footprint_l4_cells: int = 30
    dem_fallback_spill_margin_m: float = 0.30
    dem_overlay_height_m: float = 1.0

    # V8.5.0: HEC-RAS/USGS식 depth-grid 표현에 가깝게, DEM 수심 격자를
    # 먼저 부드럽게 한 뒤 등수심 경계를 벡터화합니다. 계산 레벨 자체는
    # CCTV AI 단계보다 올라가지 않습니다.
    dem_depth_smoothing_enabled: bool = True
    dem_depth_smoothing_sigma_cells: float = 0.85
    dem_boundary_smoothing_iterations: int = 2
    dem_contour_upscale: int = 6


    # 지도에 표시할 로컬 테스트 CCTV
    test_cctv_enabled: bool = True
    test_cctv_id: str = "TEST-FLOOD-01"
    test_cctv_name: str = "애니시 인근 침수 테스트 CCTV"
    test_cctv_address: str = "경상북도 포항시 북구 학전로 142 (장성동 1259-1) 애니시 인근"
    # 학전로 142 애니시 북측 도로 구간의 테스트 표시 좌표입니다. 현장 설치 시 실제 CCTV 측량 좌표로 보정하세요.
    test_cctv_lat: float = 36.06418
    test_cctv_lon: float = 129.37762
    test_cctv_video_path: str = "app/static/media/flood-test-01.mp4"
    # 이 영상은 침수 검증용으로 확정된 자료이므로 단일 프레임 오분류가
    # Lev0으로 내려가지 않도록 최소 침수등급을 적용할 수 있습니다.
    test_cctv_min_level: int = 1
    test_cctv_min_confidence: float = 0.70
    # V8.5.2: 다수결을 왜곡하지 않도록 테스트 영상도 기본적으로 강제 Lev1을
    # 적용하지 않습니다. 필요할 때만 true로 켤 수 있습니다.
    test_cctv_enforce_min_level: bool = False
    test_cctv_trusted_baseline: bool = True
    # 백그라운드 단일 프레임 분석 시 검은 시작 프레임을 피하고
    # 실제 침수 장면에서 프레임을 가져옵니다.
    test_cctv_sample_seconds: float = 20.0
    # 과거 로컬 테스트 카메라가 사용한 ID. 새 이벤트와 과거 이벤트를
    # TEST_CCTV_ID로 통일하기 위한 별칭입니다.
    test_cctv_legacy_ids: str = "e06_041,E06-041"

    sewer_api_url: str = ""
    sewer_api_key: str = ""

    # 하천 수위 API. 전체 요청 URL을 지정하며 아래 자리표시자를 사용할 수 있습니다.
    # {service_key}, {station_code}, {start_date}, {end_date}, {today}
    river_api_url: str = ""
    river_api_key: str = ""
    river_api_key_param: str = "serviceKey"
    river_api_key_header: str = ""
    river_station_codes: str = ""
    river_refresh_seconds: int = 300

    # 3-stage vehicle flood inference
    # 1) best.pt: vehicle detector
    # 2) tire_level.pt: tire flood-stage detector on vehicle crop
    # 3) car_flood_cls.pt: body flood-stage classifier fallback when tire is not detected
    stage2_model_path: str = "weights/best.pt"
    tire_level_model_path: str = "weights/tire_level.pt"
    car_flood_cls_model_path: str = "weights/car_flood_cls.pt"
    # Diagnostic switch: best.pt can be used only for box geometry while
    # suppressing its 7-class vehicle-type semantics.
    vehicle_type_labels_enabled: bool = True
    # best.pt는 차량 탐지 전용입니다. 기존 MODEL_CONFIDENCE와 분리해
    # 작은/원거리 차량을 놓치지 않도록 낮은 임계값과 큰 입력 크기를 사용합니다.
    vehicle_detection_confidence: float = 0.28
    vehicle_detection_iou: float = 0.45
    vehicle_detection_imgsz: int = 960
    vehicle_crop_pad_ratio: float = 0.12
    # Full-frame detector가 원거리/소형 차량을 적게 잡으면 좌우 중첩 타일로
    # 한 번 더 best.pt를 실행해 작은 차량을 보강합니다.
    vehicle_detection_rescue_enabled: bool = True
    vehicle_detection_rescue_min_count: int = 3
    vehicle_detection_rescue_confidence: float = 0.26
    vehicle_detection_rescue_imgsz: int = 640
    vehicle_detection_rescue_overlap: float = 0.16
    vehicle_detection_rescue_interval_seconds: float = 2.5

    # Live box visibility/stage-confirmation calibration. These defaults preserve
    # the V8.5.37 behaviour but are now explicit settings rather than scattered
    # magic numbers.
    vehicle_visible_min_confidence: float = 0.003
    vehicle_visible_max_floor: float = 0.035
    vehicle_visible_relative_floor: float = 0.12
    vehicle_stage_immediate_confidence: float = 0.12
    vehicle_stage_confirm_iou: float = 0.22
    vehicle_stage_confirm_center_ratio: float = 0.30
    vehicle_confirm_raw_min_confidence: float = 0.001
    vehicle_confirm_immediate_confidence: float = 0.12
    vehicle_confirm_two_hit_confidence: float = 0.03
    vehicle_confirm_three_hit_confidence: float = 0.008
    vehicle_confirm_four_hit_confidence: float = 0.0025
    vehicle_confirm_match_min_iou: float = 0.04
    vehicle_confirm_match_max_center_ratio: float = 0.62
    stage_max_vehicles_per_cycle: int = 4

    tire_model_confidence: float = 0.25
    tire_inference_imgsz: int = 416
    car_flood_cls_imgsz: int = 256

    pohang_cctv_enabled: bool = True
    pohang_cctv_refresh_seconds: int = 300
    stage_stream_interval_seconds: float = 1.80
    stage_box_detection_interval_seconds: float = 0.65
    stage_tracking_max_flow_seconds: float = 1.35
    # Only tire/body classifications at or above this confidence can vote for
    # or be stored as a CCTV flood stage.
    stage_min_confidence: float = 0.70
    positive_flood_confirmation_hits: int = 5
    positive_flood_confirmation_seconds: float = 3.0
    positive_flood_min_vehicles: int = 2
    positive_flood_min_ratio: float = 0.75
    stage_result_cache_seconds: float = 9.0
    stage_inference_imgsz: int = 416
    stage_max_width: int = 640
    # 기존 .env의 STAGE_MAX_WIDTH=640을 그대로 보존해도 V8.5.0 차량 추적은
    # 더 넓은 작업 프레임에서 수행합니다.
    stage_tracking_frame_width: int = 960
    stage_stale_seconds: int = 1200

    # V8.5.0: 객체 다수결 + 시간 EMA로 단발성 Lev1 오탐을 억제합니다.
    # 예: 한 프레임에서 Lev0 5대, Lev1 1대면 프레임 대표값은 Lev0입니다.
    stage_ema_alpha: float = 0.35
    stage_ema_reset_seconds: float = 8.0
    stage_majority_conf_weight: float = 0.0
    # 최종 단계는 반드시 "가장 많이 나온 차량 단계"를 우선합니다.
    # 차량 수 동률은 단계별 평균 신뢰도로 판정합니다. EMA는 진단용이며
    # 명확한 다수결이나 신뢰도 동률 판정을 뒤집지 못합니다.
    stage_strict_majority: bool = True

    # YOLO 추론 사이 프레임에서 차량 박스가 차를 따라가도록 optical-flow
    # 추적과 새 YOLO 박스 재연결/EMA 보정을 사용합니다.
    stage_tracking_enabled: bool = True
    stage_tracking_bbox_alpha: float = 1.00
    stage_tracking_min_iou: float = 0.01
    stage_tracking_max_center_ratio: float = 1.10
    stage_tracking_max_scale_change: float = 0.18
    stage_tracking_max_motion_ratio: float = 0.75
    stage_tracking_min_inlier_ratio: float = 0.35
    stage_tracking_history_frames: int = 96
    stage_tracking_max_missed_ai: int = 2
    stage_tracking_template_min_score: float = 0.42
    stage_tracking_lk_bottom_exclude_ratio: float = 0.18

    # 지도/목록은 최근 저장값이 충분히 촘촘할 때 최근 6개 기록의
    # 최빈값을 사용합니다. 5개의 Lev0 + 1개의 Lev1이면 Lev0입니다.
    # 5분 주기처럼 기록이 드문 CCTV는 최신값을 그대로 사용합니다.
    # 지도도 최신 한 건이 아니라 최근 기록/차량 투표를 합산한 최빈값만 사용합니다.
    stage_map_consensus_window_seconds: int = 120
    stage_map_consensus_max_records: int = 5
    stage_map_consensus_min_records: int = 2

    # 공개 CCTV를 화면에서 열지 않아도 순환 점검·추론·기록합니다.
    # 전체 카메라를 동시에 재생하는 방식이 아니라, 지정한 작업자 수로
    # 최신 프레임을 한 장씩 가져와 전체 목록을 반복 순회합니다.
    background_cctv_enabled: bool = True
    background_cctv_cycle_seconds: int = 300
    background_cctv_workers: int = 1
    background_cctv_start_delay_seconds: int = 8
    background_cctv_max_cameras: int = 0
    background_cctv_store_level0: bool = True
    # 로컬/테스트 CCTV는 전체 106개 순환과 별개로 짧은 주기로 계속
    # 추론·저장합니다. CCTV 창을 열지 않아도 이전 기록에 남습니다.
    background_local_enabled: bool = True
    background_local_interval_seconds: int = 15

    # 공개 CCTV 스트림 복구·표시 설정
    cctv_stream_fallback_seconds: float = 1.5
    cctv_stream_error_frame_seconds: float = 4.0
    cctv_hls_stall_reset_seconds: float = 3.5
    cctv_hls_hard_reconnect_seconds: float = 8.0
    cctv_raw_interest_seconds: float = 1.8

    # 얼굴·번호판 비식별화. 전용 YOLO 가중치가 없으면 OpenCV Haar 보조 탐지기를 사용합니다.
    anonymizer_enabled: bool = True
    anonymizer_model_path: str = "weights/anonymizer_best.pt"
    anonymizer_confidence: float = 0.10
    anonymizer_imgsz: int = 416
    anonymizer_interval_seconds: float = 0.70
    # 기존 .env에 0.70이 남아 있어도 이 신규 값으로 실제 갱신 주기를 제한합니다.
    anonymizer_refresh_seconds: float = 0.55
    anonymizer_box_scale: float = 1.05
    anonymizer_blocks: int = 10
    # 과도한 모자이크 방지: 얼굴/번호판으로 보기 어려운 큰 박스는 제거합니다.
    anonymizer_face_max_area_ratio: float = 0.12
    anonymizer_plate_max_area_ratio: float = 0.035
    anonymizer_plate_min_aspect: float = 1.20
    anonymizer_plate_max_aspect: float = 7.00
    # Haar 오탐이 벽/간판/도로를 모자이크하지 못하도록 코드에서 추가로 적용하는 hard guard.
    anonymizer_face_hard_max_area_ratio: float = 0.010
    anonymizer_plate_hard_max_area_ratio: float = 0.005
    anonymizer_plate_hard_min_aspect: float = 1.40
    anonymizer_plate_hard_max_aspect: float = 6.00
    anonymizer_plate_require_vehicle: bool = True
    anonymizer_disable_haar_flow: bool = True
    # Haar fallback은 2회 연속 비슷한 위치에서 검출된 경우에만 모자이크합니다.
    # 벽/간판을 한 프레임만 얼굴·번호판으로 오인하는 현상을 억제합니다.
    anonymizer_haar_confirm_iou: float = 0.20
    anonymizer_haar_confirm_hits: int = 3
    anonymizer_haar_require_motion: bool = True
    anonymizer_haar_motion_mean_threshold: float = 4.0
    anonymizer_haar_face_max_static_hits: int = 2

    sewer_refresh_seconds: int = 60

    # V8.5.1: 강수량/하수 수위/하천 수위를 SQLite에 시계열로 저장합니다.
    # 업스트림 관측시각이 같으면 UNIQUE 키로 중복 저장하지 않습니다.
    environment_history_enabled: bool = True
    environment_history_interval_seconds: int = 60
    environment_history_start_delay_seconds: int = 12
    environment_history_retention_days: int = 365

    @staticmethod
    def _camera_key(value: str | None) -> str:
        return re.sub(
            r"[^a-z0-9가-힣]+",
            "",
            str(value or "").strip().lower(),
        )

    def test_cctv_aliases(self) -> tuple[str, ...]:
        values = [
            self.test_cctv_id,
            *str(self.test_cctv_legacy_ids or "").split(","),
        ]
        aliases: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in aliases:
                aliases.append(text)
        return tuple(aliases)

    def is_test_cctv_alias(self, value: str | None) -> bool:
        key = self._camera_key(value)
        return bool(
            key
            and key in {
                self._camera_key(alias)
                for alias in self.test_cctv_aliases()
            }
        )

    def cameras(self) -> list[CameraConfig]:
        cameras: list[CameraConfig] = []
        extra = self.model_extra or {}
        for index in range(100):
            prefix = f"camera_{index}_"
            camera_id = extra.get(prefix + "id")
            if camera_id is None and index == 0:
                camera_id = "demo"
            if camera_id is None:
                continue
            source = str(extra.get(prefix + "source", "demo://flood"))
            raw_camera_id = str(camera_id)
            name = str(extra.get(prefix + "name", raw_camera_id))
            site_name = str(
                extra.get(prefix + "site_name")
                or extra.get(prefix + "location_name")
                or name
            )
            address = str(
                extra.get(prefix + "address")
                or site_name
            )
            lat = float(extra.get(prefix + "lat", 37.4947))
            lon = float(extra.get(prefix + "lon", 127.0631))

            # E06-041은 이 프로젝트에서 테스트 영상에 사용된 이전 ID입니다.
            # 이전 .env를 복사해도 오른쪽 CCTV 목록과 동일한 정식 테스트
            # ID·이름·좌표로 자동 통일합니다.
            if self.is_test_cctv_alias(raw_camera_id):
                camera_id = self.test_cctv_id
                name = self.test_cctv_name
                site_name = self.test_cctv_name
                address = self.test_cctv_address
                lat = float(self.test_cctv_lat)
                lon = float(self.test_cctv_lon)

            cameras.append(
                CameraConfig(
                    camera_id=str(camera_id),
                    name=name,
                    site_name=site_name,
                    address=address,
                    source=source,
                    roi=parse_roi(extra.get(prefix + "roi")),
                    lat=lat,
                    lon=lon,
                )
            )
        if not cameras:
            cameras.append(
                CameraConfig(
                    "demo",
                    "데모 침수 CCTV",
                    "데모 침수 지점",
                    "데모 침수 지점",
                    "demo://flood",
                    [],
                    37.4947,
                    127.0631,
                )
            )
        return cameras


settings = Settings()
