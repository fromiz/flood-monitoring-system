from pathlib import Path

from app.stage_policy import (
    PositiveFloodConfirmation,
    is_authoritative_stage_record,
)


ROOT = Path(__file__).parents[1]
MAIN = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
PIPELINE = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")


def test_single_public_false_positive_is_never_authoritative():
    gate = PositiveFloodConfirmation()
    result = gate.evaluate(
        "public-a", 4, 0.99,
        positive_votes=1, total_votes=1, now=0.0,
    )
    assert not result["accepted"]
    assert not result["pending"]
    low = gate.evaluate(
        "public-low", 2, 0.699,
        positive_votes=3, total_votes=3, now=0.0,
    )
    assert not low["accepted"]
    assert low["reason"] == "below_minimum_confidence"


def test_public_positive_requires_repeated_multi_vehicle_confirmation():
    gate = PositiveFloodConfirmation()
    first = gate.evaluate(
        "public-b", 2, 0.91,
        positive_votes=2, total_votes=2, now=0.0,
    )
    second = gate.evaluate(
        "public-b", 2, 0.93,
        positive_votes=2, total_votes=2, now=0.8,
    )
    third = gate.evaluate(
        "public-b", 2, 0.95,
        positive_votes=2, total_votes=2, now=1.6,
    )
    fourth = gate.evaluate(
        "public-b", 2, 0.94,
        positive_votes=2, total_votes=2, now=2.4,
    )
    fifth = gate.evaluate(
        "public-b", 2, 0.96,
        positive_votes=2, total_votes=2, now=3.2,
    )
    assert first["pending"] and not first["accepted"]
    assert second["pending"] and not second["accepted"]
    assert third["pending"] and not third["accepted"]
    assert fourth["pending"] and not fourth["accepted"]
    assert fifth["accepted"]


def test_normal_and_trusted_test_decisions_are_immediate():
    gate = PositiveFloodConfirmation()
    assert gate.evaluate(
        "normal", 0, 0.80, positive_votes=0, total_votes=3, now=0.0
    )["accepted"]
    assert gate.evaluate(
        "test", 1, 0.70, positive_votes=1, total_votes=1,
        trusted_test=True, now=0.0,
    )["accepted"]


def test_old_unconfirmed_positive_rows_are_filtered_from_map():
    assert not is_authoritative_stage_record(4, {})
    assert not is_authoritative_stage_record(1, {"positive_confirmed": False})
    assert is_authoritative_stage_record(0, {})
    assert is_authoritative_stage_record(2, {"positive_confirmed": True})
    assert "is_authoritative_stage_record(row.level, details)" in MAIN
    assert "if stage > 0 and not bool(result.get(\"positive_confirmed\"))" in MAIN
    assert "record?.positive_confirmed !== true" in FRONTEND


def test_all_windows_keep_ai_snapshots_and_focus_only_changes_cadence():
    assert "def raw_mjpeg(stream_url" in WORKER
    assert "def annotated_snapshot(" in WORKER
    assert "@app.get(\"/api/cctv/frame-annotated\")" in MAIN
    assert 'cctvWindows.forEach(win => {' in FRONTEND
    assert 'ensureAnnotatedCctvStream(win)' in FRONTEND
    assert 'syncCctvWebSocketSubscriptions()' in FRONTEND
    assert "annotatedSnapshotUrl(win.camera, win.isFocused)" in FRONTEND
    assert "/api/cctv/frame-raw" in FRONTEND


def test_raw_and_annotated_endpoints_share_capture_worker():
    raw = WORKER.split("def raw_mjpeg", 1)[1]
    annotated = WORKER.split("def annotated_mjpeg", 1)[1].split("def raw_mjpeg", 1)[0]
    assert "_get_camera_worker(stream_url)" in raw
    assert "_get_camera_worker(stream_url)" in annotated


def test_box_detector_is_fast_and_rescue_is_throttled():
    assert "stage_box_detection_interval_seconds" in WORKER
    assert "vehicle_detection_rescue_interval_seconds" in WORKER
    assert "max(1.0, float(settings.vehicle_detection_rescue_interval_seconds))" in WORKER
    assert "len(self.last_associated_detections) < rescue_target" in WORKER
    assert "allow_rescue=allow_rescue" in WORKER
    assert "and bool(settings.vehicle_detection_rescue_enabled)" in PIPELINE


def test_optical_flow_boxes_expire_before_long_drift():
    assert "stage_tracking_max_flow_seconds" in WORKER
    assert "tracked_detections = []" in WORKER
