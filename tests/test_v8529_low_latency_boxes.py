from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PIPELINE = (ROOT/'app/vehicle_flood_pipeline.py').read_text(encoding='utf-8')
WORKER = (ROOT/'app/pohang_cctv.py').read_text(encoding='utf-8')
MAIN = (ROOT/'app/main.py').read_text(encoding='utf-8')
APPJS = (ROOT/'app/static/app.js').read_text(encoding='utf-8')
INDEX = (ROOT/'app/static/index.html').read_text(encoding='utf-8')
ENV = (ROOT/'.env').read_text(encoding='utf-8')

def test_cuda_preference_fails_open_not_models_off():
    assert 'CUDA를 사용할 수 없어 CPU로 자동 전환' in PIPELINE
    assert 'return "cpu"' in PIPELINE

def test_best_pt_requests_low_conf_candidates_then_temporal_filters():
    assert '0.025 if ai_uses_cuda() else 0.05' in PIPELINE
    assert 'if float(vehicle_conf) < 0.018' in PIPELINE
    assert 'return detections[:80]' in PIPELINE
    assert 'required_hits = 1 if conf >= immediate else (2 if conf >= 0.035 else 3)' in WORKER

def test_zero_box_scene_has_provisional_geometry_not_silent_drop():
    assert 'if not detected_boxes and raw_detected_boxes' in WORKER
    assert 'item["_provisional"] = True' in WORKER
    assert 'and not bool(item.get("_provisional"))' in WORKER
    assert 'CHECK {confidence * 100:.0f}%' in WORKER

def test_stage_pipeline_order_preserved():
    order = PIPELINE.index('tire_results =')
    fallback = PIPELINE.index('fallback_indices:', order)
    body = PIPELINE.index('body_results = _predict(', fallback)
    assert order < fallback < body
    assert 'vehicle_candidates=job["vehicle_candidates"]' in WORKER

def test_browser_transport_reduces_encode_decode_pressure():
    assert 'if jpeg_frame.shape[1] > 640' in WORKER
    assert 'ordered_subscriptions.sort' in MAIN
    assert 'next_due[key] = now + (0.080 if focused else 0.120)' in MAIN
    assert 'V8.5.29 loaded' in APPJS
    assert '/static/app.js?v=8.5.29' in INDEX

def test_gpu_env_keeps_requested_models_and_tuning():
    assert 'AI_DEVICE=cuda' in ENV
    assert 'STAGE2_MODEL_PATH=weights/best.pt' in ENV
    assert 'TIRE_LEVEL_MODEL_PATH=weights/tire_level.pt' in ENV
    assert 'CAR_FLOOD_CLS_MODEL_PATH=weights/car_flood_cls.pt' in ENV
