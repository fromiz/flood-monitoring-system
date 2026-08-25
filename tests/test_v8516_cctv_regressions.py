from pathlib import Path


ROOT = Path(__file__).parents[1]
PIPELINE = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")


def test_cpu_models_share_one_prediction_lock_to_prevent_contention():
    assert "_PREDICT_LOCKS" in PIPELINE
    assert "_CPU_PREDICT_LOCK" in PIPELINE
    assert "if not torch.cuda.is_available()" in PIPELINE
    assert "with predict_lock:" in PIPELINE


def test_fast_detector_has_small_vehicle_tile_rescue():
    assert '"live_tile"' in PIPELINE
    assert "len(raw_boxes) < max(" in PIPELINE
    assert "settings.vehicle_detection_rescue_min_count" in PIPELINE
    assert "iou >= 0.38" in PIPELINE
    assert "overlap_smaller >= 0.72" in PIPELINE


def test_worker_has_explicit_stop_for_close_and_shutdown():
    assert "def stop(self)" in WORKER


def test_every_open_window_keeps_live_mjpeg():
    assert "function selectAnnotatedWindow(activeWin)" in FRONTEND
    assert "'/api/stream-raw'" in FRONTEND
