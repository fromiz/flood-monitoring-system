from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = (ROOT / 'app' / 'pohang_cctv.py').read_text(encoding='utf-8')
JS = (ROOT / 'app' / 'static' / 'app.js').read_text(encoding='utf-8')
PIPE = (ROOT / 'app' / 'vehicle_flood_pipeline.py').read_text(encoding='utf-8')

def test_gpu_visibility_is_separate_from_stage_confirmation():
    assert 'stage_confirmed_boxes, next_pending = _confirm_vehicle_candidates' in P
    assert 'visible_floor = max(0.003, min(0.035, top_conf * 0.12))' in P
    assert 'item["_stage_eligible"] = bool(stage_eligible)' in P

def test_weak_visible_boxes_do_not_enter_stage_until_confirmed():
    assert 'and bool(item.get("_stage_eligible", not item.get("_provisional")))' in P

def test_gpu_has_no_missed_box_carry():
    assert 'max_missed = 0 if ai_uses_cuda()' in P

def test_gpu_has_no_optical_flow_projection_after_detector():
    assert 'not ai_uses_cuda()\n                    and frames_behind > 0' in P

def test_raw_detector_floor_is_open_for_night_cctv():
    assert 'raw_conf = 0.001 if ai_uses_cuda() else 0.004' in PIPE

def test_browser_draws_vector_detections_after_jpeg():
    assert 'drawCctvVectorDetections(context, canvas, packet);' in JS

def test_diagnostic_is_visible_at_warning_level():
    assert 'CCTV best.pt raw=%s visible=%s confirmed_for_stage=0' in P
    ix = P.index('CCTV best.pt raw=%s visible=%s confirmed_for_stage=0')
    assert 'logger.warning(' in P[max(0, ix-600):ix]

def test_version_bumped():
    assert '8.5.36' in JS
