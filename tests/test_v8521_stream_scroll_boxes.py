from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
MAIN = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (ROOT / "app" / "static" / "style.css").read_text(encoding="utf-8")
ENV = (ROOT / ".env").read_text(encoding="utf-8")
PIPELINE = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")


def test_all_windows_use_short_annotated_snapshots_not_persistent_mjpeg():
    assert '@app.get("/api/cctv/frame-annotated")' in MAIN
    assert "def annotated_snapshot(" in WORKER
    assert "function annotatedSnapshotUrl" in FRONTEND
    assert "scheduleAnnotatedSnapshot" in FRONTEND
    select_block = FRONTEND.split("function selectAnnotatedWindow", 1)[1].split(
        "function ensureCctvWindowLayerOnBody", 1
    )[0]
    assert "cctvWindows.forEach" in select_block
    assert "ensureAnnotatedCctvStream(win)" in select_block
    assert "syncCctvWebSocketSubscriptions()" in select_block


def test_http_cctv_uses_hls_first_before_direct_opencv_timeout():
    grab = WORKER.split("def _grab_loop", 1)[1].split("def _get_latest_frame", 1)[0]
    assert "hls_first_attempt = bool(self.http_stream)" in grab
    assert 'hls_fallback_until = float("inf") if hls_first_attempt else 0.0' in grab
    assert "hls_first_attempt = False" in grab


def test_arranged_history_is_flex_shrinkable_and_scrollable_to_bottom():
    assert ".cctv-window.auto-arranged:not(.compact) .cctv-history" in STYLE
    assert "flex:1 1 55%" in STYLE
    assert "min-height:0" in STYLE
    assert "overflow-y:auto" in STYLE
    assert "scrollbar-gutter:stable" in STYLE


def test_vehicle_detector_uses_geometry_resolution_not_stage_resolution():
    assert "self.vehicle_detector_imgsz" in WORKER
    ai = WORKER.split("def _ai_loop", 1)[1].split("def _stage_loop", 1)[0]
    assert "vehicle_imgsz=self.vehicle_detector_imgsz" in ai
    assert "len(self.last_associated_detections) < rescue_target" in ai
    assert "geometry_flow_seconds" in WORKER
    assert "len(raw_boxes) < max(" in PIPELINE
    assert "settings.vehicle_detection_rescue_min_count" in PIPELINE


def test_packaged_box_defaults_enable_distant_vehicle_rescue():
    assert "VEHICLE_DETECTION_IMGSZ=960" in ENV
    assert "VEHICLE_DETECTION_RESCUE_MIN_COUNT=3" in ENV
    assert "VEHICLE_DETECTION_RESCUE_CONFIDENCE=0.26" in ENV
    assert "VEHICLE_DETECTION_RESCUE_IMGSZ=640" in ENV
    assert "VEHICLE_DETECTION_RESCUE_OVERLAP=0.16" in ENV
    assert "VEHICLE_DETECTION_RESCUE_INTERVAL_SECONDS=2.5" in ENV
    assert "STAGE_TRACKING_MAX_FLOW_SECONDS=1.35" in ENV
    assert "STAGE_TRACKING_MIN_IOU=0.08" in ENV
    assert "STAGE_TRACKING_MAX_CENTER_RATIO=0.70" in ENV
