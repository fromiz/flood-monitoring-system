from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
PIPELINE = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (ROOT / "app" / "static" / "style.css").read_text(encoding="utf-8")
ENV = (ROOT / ".env").read_text(encoding="utf-8")
MAIN = (ROOT / "app" / "main.py").read_text(encoding="utf-8")


def test_hls_stall_watchdog_re_resolves_and_hard_reconnects():
    grab = WORKER.split("def _grab_loop", 1)[1].split("def _get_latest_frame", 1)[0]
    assert "cctv_hls_stall_reset_seconds" in grab
    assert "cctv_hls_hard_reconnect_seconds" in grab
    assert "_HlsSegmentPrefetcher" in grab
    assert "hls_fallback_until = 0.0" in grab
    assert "hls_stall_resets" in grab
    assert "CCTV_HLS_STALL_RESET_SECONDS=3.5" in ENV
    assert "CCTV_HLS_HARD_RECONNECT_SECONDS=8.0" in ENV


def test_raw_snapshot_interest_lease_avoids_polling_race_and_long_waits():
    block = WORKER.split("def raw_snapshot", 1)[1].split("def stream_mjpeg", 1)[0]
    assert "raw_interest_until" in WORKER
    assert "cctv_raw_interest_seconds" in WORKER
    assert "age <= 0.22" in block
    assert "min(0.70" in block
    assert "timeout=0.45" in MAIN
    assert "CCTV_RAW_INTEREST_SECONDS=1.8" in ENV
    assert "self.raw_render_interval = 0.125" in WORKER
    assert "now >= next_raw_render_at" in WORKER


def test_frontend_uses_stateless_decoded_snapshots_without_blank_overlay():
    assert "function restartAnnotatedCctvStream" in FRONTEND
    assert "new WebSocket(cctvTransportUrl())" in FRONTEND
    assert "createImageBitmap(blob)" in FRONTEND
    assert "win.pendingWsFrame" in FRONTEND
    assert "header.mode === 'status' && win.hasRealFrame" in FRONTEND
    assert "ensureRawCctvStream(win)" in FRONTEND
    assert "/api/cctv/worker-status" not in FRONTEND.split("async function refreshWindowStreamStatus", 1)[1].split("function rawSnapshotUrl", 1)[0]
    assert "annotatedRenewTimer" not in FRONTEND


def test_stage_classifier_and_rescue_are_throttled_to_reduce_render_contention():
    assert "self.stage_interval" in WORKER
    stage = WORKER.split("def _stage_loop", 1)[1].split("def _render_loop", 1)[0]
    assert "next_stage_at" in stage
    ai = WORKER.split("def _ai_loop", 1)[1].split("def _stage_loop", 1)[0]
    assert "vehicle_detection_rescue_interval_seconds" in ai
    assert "VEHICLE_DETECTION_RESCUE_INTERVAL_SECONDS=2.5" in ENV


def test_vehicle_boxes_reject_implausible_geometry_and_cross_tile_duplicates():
    detector = PIPELINE.split("def detect_vehicle_boxes", 1)[1].split("def ", 1)[0]
    assert "area_ratio > 0.48" in detector
    assert "aspect < 0.32 or aspect > 5.8" in detector
    assert "iou >= 0.38" in detector
    assert "overlap_smaller >= 0.72" in detector
    assert "center_ratio <= 0.18" in detector


def test_live_tracker_uses_translation_only_flow_and_no_box_scale_wobble():
    block = WORKER.split("def _propagate_detections(", 1)[1].split("def _draw_live_detections", 1)[0]
    assert "lk-translate" in block
    assert "_flow_failures" in block
    assert "flow_inlier_ratio" in block
    assert "new_spread" not in block
    assert "scale =" not in block
    assert "box_width * 0.16" in block


def test_fresh_association_cleans_near_duplicate_boxes_and_labels_avoid_collisions():
    associate = WORKER.split("def _associate_detections", 1)[1].split("def _stage_color", 1)[0]
    assert "overlap_smaller >= 0.86" in associate
    assert "iou >= 0.62" in associate
    draw = WORKER.split("def _draw_live_detections", 1)[1].split("class CCTVWorker", 1)[0]
    assert "occupied_labels" in draw
    assert "candidate_tops" in draw


def test_css_isolates_high_frequency_video_repaint_from_history_layout():
    assert ".cctv-body{contain:paint}" in STYLE.replace(" ", "")
    assert "translateZ(0)" in STYLE


def test_dead_critical_worker_is_not_reused_forever():
    block = WORKER.split("def is_alive(self)", 1)[1].split("def stop(self)", 1)[0]
    assert '"cctv-grab", "cctv-render"' in block
    assert "thread.is_alive()" in block
    assert "latest_annotated_frame_at" in WORKER
    assert '"last_render_age_seconds"' in WORKER
