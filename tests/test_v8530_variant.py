from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENV=(ROOT/'.env').read_text(encoding='utf-8')
PIPE=(ROOT/'app/vehicle_flood_pipeline.py').read_text(encoding='utf-8')
CCTV=(ROOT/'app/pohang_cctv.py').read_text(encoding='utf-8')

def test_geometry_is_published_before_stage_models():
    assert 'detected_boxes = []' in CCTV
    assert 'item["_provisional"] = bool(conf < 0.04)' in CCTV
    assert 'CCTV best.pt raw boxes=0' in CCTV
    assert 'geometry_update": True' in CCTV

def test_low_conf_candidates_reach_ui_for_diagnosis():
    assert '0.008 if ai_uses_cuda() else 0.03' in PIPE
    assert 'if float(vehicle_conf) < 0.005' in PIPE
    assert 'aspect < 0.18 or aspect > 8.0' in PIPE

def test_best_model_is_always_box_geometry_source():
    assert 'STAGE2_MODEL_PATH=weights/best.pt' in ENV
    assert (ROOT/'weights/best.pt').is_file()

def test_body_fallback_path_exists():
    assert 'if fallback_indices and body_model is not None' in PIPE
    assert 'stage_source = "car_body"' in PIPE

def test_full_variant_keeps_type_tire_body():
    assert 'VEHICLE_TYPE_LABELS_ENABLED=true' in ENV
    assert 'TIRE_LEVEL_MODEL_PATH=weights/tire_level.pt' in ENV
    assert (ROOT/'weights/tire_level.pt').is_file()
    assert (ROOT/'weights/car_flood_cls.pt').is_file()
