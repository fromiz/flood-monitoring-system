from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_pipeline_no_low_conf_background_clamp():
    text=(ROOT/'app/vehicle_flood_pipeline.py').read_text(encoding='utf-8')
    assert 'max(0.24, min(0.60, float(settings.vehicle_detection_confidence)))' in text
    assert 'max_stage_vehicles = 32 if ai_uses_cuda() else 12' in text
    assert '"tire_level": _TIRE_PREDICT_LOCK' in text
    assert '"car_flood_cls": _BODY_PREDICT_LOCK' in text

def test_weather_key_not_double_encoded():
    text=(ROOT/'app/realtime_weather.py').read_text(encoding='utf-8')
    assert 'normalised_key = unquote(auth_key)' in text
    assert 'KMA_SERVICE_KEY 폴백' in text

def test_trusted_test_event_is_authoritative():
    text=(ROOT/'app/main.py').read_text(encoding='utf-8')
    assert 'and not test_floor_applied' in text
    assert '"positive_confirmed": True' in text

def test_live_cctv_pauses_background_scan():
    text=(ROOT/'app/main.py').read_text(encoding='utf-8')
    assert 'if has_live_cctv_clients():' in text
    assert '"skipped_live_priority": True' in text
