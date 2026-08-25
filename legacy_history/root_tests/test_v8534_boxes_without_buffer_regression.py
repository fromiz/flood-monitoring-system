from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = (ROOT / 'app' / 'vehicle_flood_pipeline.py').read_text(encoding='utf-8')
CCTV = (ROOT / 'app' / 'pohang_cctv.py').read_text(encoding='utf-8')
JS = (ROOT / 'app' / 'static' / 'app.js').read_text(encoding='utf-8')
ENV = (ROOT / '.env').read_text(encoding='utf-8')


def test_low_raw_yolo_candidates_are_not_pruned_at_three_percent():
    assert 'raw_conf = 0.001 if ai_uses_cuda() else 0.004' in PIPE
    assert 'raw_conf = max(0.03 if ai_uses_cuda()' not in PIPE
    assert 'VEHICLE_DETECTION_CONFIDENCE=0.001' in ENV


def test_weak_boxes_are_temporally_confirmed_not_immediately_published():
    assert 'elif conf >= 0.008:' in CCTV
    assert 'required_hits = 3' in CCTV
    assert 'elif conf >= 0.0025:' in CCTV
    assert 'required_hits = 4' in CCTV
    assert 'item["_provisional"] = not item["_confirmed"]' in CCTV


def test_gpu_does_not_carry_missed_vehicle_boxes():
    assert 'max_missed = 0 if ai_uses_cuda()' in CCTV


def test_live_jpeg_does_not_bake_vehicle_boxes():
    assert 'draw_boxes=False' in CCTV
    assert 'for detection in (ordered if draw_boxes else [])' in CCTV


def test_browser_is_single_authoritative_box_renderer():
    assert 'drawCctvVectorDetections(context, canvas, packet);' in JS
    assert "if (packet.mode !== 'annotated')" not in JS


def test_required_tire_then_body_fallback_pipeline_remains():
    assert 'tire_results' in PIPE
    assert 'fallback_indices' in PIPE
    assert PIPE.find('tire_results') < PIPE.find('fallback_indices')
    assert 'car_flood_cls' in PIPE


def test_hls_latest_only_prefetch_is_retained():
    assert 'queue.Queue(maxsize=1)' in CCTV
    assert 'Replace it with the newer live-edge segment' in CCTV

def test_worker_status_exposes_detection_diagnostics():
    assert '"latest_detection_count"' in CCTV
    assert '"annotated_interest_active"' in CCTV
    assert '"focused_interest_active"' in CCTV
    assert 'confirmed=0 pending=' in CCTV
