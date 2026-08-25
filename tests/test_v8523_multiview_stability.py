from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
PIPELINE = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
MAIN = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
ENV = (ROOT / ".env").read_text(encoding="utf-8")


def test_hls_segments_are_prefetched_and_consumed_in_order():
    assert "class _HlsSegmentPrefetcher" in WORKER
    assert "queue.Queue(maxsize=2)" in WORKER
    chooser = WORKER.split("def _choose_unseen_hls_segment", 1)[1].split(
        "def _download_latest_hls_segment", 1
    )[0]
    assert "return newer[0]" in chooser
    assert "data_resp.close()" in WORKER


def test_all_open_windows_keep_annotations_with_focus_priority():
    assert '@app.get("/api/cctv/frame-annotated")' in MAIN
    assert "focus: bool = Query(False)" in MAIN
    assert "focused_interest_until" in WORKER
    assert "def has_focus_interest" in WORKER
    assert 'cctvWindows.forEach(win => {' in FRONTEND
    assert 'ensureAnnotatedCctvStream(win)' in FRONTEND
    assert 'syncCctvWebSocketSubscriptions()' in FRONTEND


def test_snapshot_endpoint_is_nonblocking_and_browser_keeps_old_frame_until_decode():
    snap = WORKER.split("def annotated_snapshot", 1)[1].split("def annotated_mjpeg", 1)[0]
    assert "wait_for_jpeg" not in snap
    assert "renew_annotated_interest" in snap
    assert "preloadCctvObjectUrl" in FRONTEND
    assert "CCTV JPEG decode timeout" in FRONTEND
    assert "X-CCTV-Frame-Seq" in MAIN


def test_box_filtering_rejects_one_frame_weak_false_positives():
    assert "def _confirm_vehicle_candidates" in WORKER
    confirm = WORKER.split("def _confirm_vehicle_candidates", 1)[1].split(
        "def _associate_detections", 1
    )[0]
    assert "immediate_conf = 0.55 if is_tile else 0.42" in confirm
    assert "required_hits = 3 if (is_tile or conf < 0.30) else 2" in confirm
    assert "next_pending.append(item)" in confirm
    assert "VEHICLE_DETECTION_CONFIDENCE=0.28" in ENV
    assert "VEHICLE_DETECTION_RESCUE_CONFIDENCE=0.26" in ENV


def test_low_confidence_tracking_is_more_conservative_and_boxes_are_held_between_ai_passes():
    tracker = WORKER.split("def _propagate_detections(", 1)[1].split(
        "def _draw_live_detections", 1
    )[0]
    assert "required_inlier = 0.72 if vehicle_conf < 0.45 else 0.58" in tracker
    assert "geometry_hold_seconds" in WORKER
    assert "max(3.0" in WORKER


def test_cpu_inference_is_serialized_and_background_cameras_are_throttled():
    assert "_CPU_PREDICT_LOCK" in PIPELINE
    assert "if not torch.cuda.is_available()" in PIPELINE
    ai = WORKER.split("def _ai_loop", 1)[1].split("def _stage_loop", 1)[0]
    assert "self.has_focus_interest()" in ai
    assert "max(1.10, self.ai_interval * 1.7)" in ai
    stage = WORKER.split("def _stage_loop", 1)[1].split("def _render_loop", 1)[0]
    assert "max(3.0, self.stage_interval * 1.8)" in stage


def test_reconnect_never_overwrites_a_real_frame_with_status_card():
    block = WORKER.split("def _publish_stream_error", 1)[1].split("def _grab_loop", 1)[0]
    assert "if self.last_stream_ok_at:" in block
    assert "return" in block
