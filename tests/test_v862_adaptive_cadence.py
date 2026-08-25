from __future__ import annotations

from pathlib import Path


def test_v862_adaptive_cadence_is_configured_without_geometry_prediction():
    root = Path(__file__).resolve().parents[1]
    env = (root / '.env').read_text(encoding='utf-8')
    pohang = (root / 'app' / 'pohang_cctv.py').read_text(encoding='utf-8')

    assert 'AI_ADAPTIVE_CADENCE_ENABLED=true' in env
    assert 'AI_GEOMETRY_FAST_INTERVAL_SECONDS=0.18' in env
    assert 'AI_GEOMETRY_BUSY_INTERVAL_SECONDS=0.34' in env
    assert 'def _adaptive_cuda_ai_interval' in pohang
    assert 'last_ai_cadence_mode' in pohang
    # CUDA box geometry remains detector-owned; do not bring back GPU LK flow.
    assert 'not ai_uses_cuda()' in pohang
    assert 'and not applied_new_ai' in pohang


def test_v862_preserves_single_owner_scheduler_and_full_stage_pipeline():
    root = Path(__file__).resolve().parents[1]
    pipeline = (root / 'app' / 'vehicle_flood_pipeline.py').read_text(encoding='utf-8')
    env = (root / '.env').read_text(encoding='utf-8')

    assert 'class _CentralInferenceScheduler' in pipeline
    assert '_VEHICLE_PREDICT_LOCK' not in pipeline
    assert '_STAGE_PREDICT_LOCK' not in pipeline
    assert 'torch.backends.cudnn.benchmark = False' in pipeline
    assert 'TIRE_LEVEL_MODEL_PATH=weights/tire_level.pt' in env
    assert 'CAR_FLOOD_CLS_MODEL_PATH=weights/car_flood_cls.pt' in env


def test_v862_diagnostics_expose_cadence():
    root = Path(__file__).resolve().parents[1]
    pohang = (root / 'app' / 'pohang_cctv.py').read_text(encoding='utf-8')
    assert 'CCTV BOX62' in pohang
    assert 'CCTV WS62' in pohang
    assert 'cadence=%s/%.3f' in pohang
    assert '"ai_interval_seconds"' in pohang
