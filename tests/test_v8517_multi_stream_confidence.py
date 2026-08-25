from pathlib import Path

from app.stage_policy import qualifies_stage_confidence


ROOT = Path(__file__).parents[1]
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
PIPELINE = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")
CONFIG = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
MAIN = (ROOT / "app" / "main.py").read_text(encoding="utf-8")


def test_every_open_window_keeps_its_mjpeg_source():
    assert "function selectAnnotatedWindow(activeWin)" in FRONTEND
    assert "mode === 'annotated'" in FRONTEND
    assert "'/api/stream-raw'" in FRONTEND


def test_new_camera_does_not_stop_existing_workers():
    get_worker = WORKER.split("def _get_camera_worker", 1)[1].split(
        "def camera_worker_status", 1
    )[0]
    assert "existing.stop()" not in get_worker


def test_stage_threshold_is_seventy_percent():
    assert "stage_min_confidence: float = 0.70" in CONFIG
    assert "float(settings.stage_min_confidence)" in PIPELINE
    assert '"stage": int(stage) if stage_valid else None' in PIPELINE


def test_stage_threshold_is_inclusive_at_exact_boundary():
    assert not qualifies_stage_confidence(0.699, 0.70)
    assert qualifies_stage_confidence(0.700, 0.70)
    assert qualifies_stage_confidence(0.90, 0.70)
    assert not qualifies_stage_confidence(None, 0.70)


def test_no_qualified_vote_does_not_fabricate_level_zero():
    probability_function = WORKER.split(
        "def _frame_stage_probabilities", 1
    )[1].split("def _consensus_stage", 1)[0]
    assert "votes[0] = 1" not in probability_function
    assert 'return None, 0.0, diagnostics' in WORKER
    assert 'if final_stage is None:' in WORKER


def test_database_save_has_same_confidence_guard():
    save_function = MAIN.split("def _save_stage_event", 1)[1].split(
        "class BackgroundCctvMonitor", 1
    )[0]
    assert "qualifies_stage_confidence" in save_function


def test_local_recorder_uses_model_result_not_synthetic_baseline():
    run_function = MAIN.split("def _run(self)", 2)[2].split(
        "continuous_local_recorder", 1
    )[0]
    assert "result = background_cctv_monitor._process_camera(camera)" in run_function
    assert "result = self._store_baseline(camera)" not in run_function
