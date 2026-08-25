from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
PIPELINE = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
MAIN = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
ENV = (ROOT / ".env").read_text(encoding="utf-8")
INDEX = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_all_open_cctv_share_one_websocket_transport():
    assert '@app.websocket("/ws/cctv")' in MAIN
    assert "struct.pack(\">I\", len(header)) + header + jpeg" in MAIN
    assert "new WebSocket(cctvTransportUrl())" in FRONTEND
    assert "type: 'subscribe'" in FRONTEND
    assert "cameras," in FRONTEND
    assert "focused_key" in FRONTEND
    assert 'class="cctv-video cctv-video-canvas hidden"' in FRONTEND


def test_frontend_is_newest_only_and_holds_last_good_frame():
    transport = FRONTEND.split("async function drawCctvTransportFrame", 1)[1].split(
        "function ensureCctvWebSocket", 1
    )[0]
    assert "win.pendingWsFrame = null" in transport
    assert "If a newer packet arrived" in transport
    assert "createImageBitmap(blob)" in transport
    assert "header.mode === 'status' && win.hasRealFrame" in transport
    assert "win.hasRealFrame = true" in transport


def test_stream_status_no_longer_polls_worker_status_per_window():
    status = FRONTEND.split("async function refreshWindowStreamStatus", 1)[1].split(
        "function rawSnapshotUrl", 1
    )[0]
    assert "/api/cctv/worker-status" not in status
    assert "ensureCctvWebSocket()" in status
    assert "win.hasRealFrame" in status


def test_websocket_worker_renews_ai_for_every_camera_and_raw_only_when_needed():
    block = WORKER.split("def live_transport_packet", 1)[1].split(
        "def raw_snapshot", 1
    )[0]
    assert "renew_annotated_interest(2.4, focused=focused)" in block
    assert "renew_raw_interest(2.4)" in block
    assert '"mode": "annotated"' in block
    assert '"mode": "raw"' in block
    assert "annotated_age <= 0.85" in block
    assert "raw_age <= 0.75" in block


def test_weak_boxes_remain_confirmed_in_temporal_state_instead_of_flickering():
    block = WORKER.split("def _confirm_vehicle_candidates", 1)[1].split(
        "def _associate_detections", 1
    )[0]
    assert "immediate_conf = 0.55 if is_tile else 0.42" in block
    assert "required_hits = 3 if (is_tile or conf < 0.30) else 2" in block
    # This line is the V8.5.23 flicker fix: a confirmed weak box stays pending
    # so the next detector pass increments rather than restarting at hit=1.
    assert "next_pending.append(item)" in block


def test_tile_false_boxes_and_flow_drift_are_more_conservative():
    assert "touches_internal_edge" in PIPELINE
    assert "tile_x2 < width" in PIPELINE
    assert "box_conf < 0.50" in PIPELINE
    assert "aspect < 0.32 or aspect > 5.8 or area_ratio > 0.48" in PIPELINE
    tracker = WORKER.split("def _propagate_detections(", 1)[1].split(
        "def _draw_live_detections", 1
    )[0]
    assert "required_inlier = 0.72 if vehicle_conf < 0.45 else 0.58" in tracker
    assert "safe_tiny_motion" in tracker


def test_transport_cache_bust_and_vehicle_thresholds_are_v8524():
    assert "/static/app.js?v=8.5.24" in INDEX
    assert "VEHICLE_DETECTION_CONFIDENCE=0.28" in ENV
    assert "VEHICLE_DETECTION_RESCUE_CONFIDENCE=0.26" in ENV
