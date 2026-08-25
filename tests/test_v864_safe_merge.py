from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POHANG = ROOT / "app" / "pohang_cctv.py"


def _load_functions(*names: str):
    tree = ast.parse(POHANG.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_STAGE_STATE_KEYS" in targets:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
    namespace = {"Any": Any}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(POHANG), "exec"), namespace)
    return [namespace[name] for name in names]


def test_safe_merge_keeps_v863_detection_and_gpu_policies():
    env = (ROOT / ".env").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8-sig")
    pipeline = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")
    main = (ROOT / "app" / "main.py").read_text(encoding="utf-8")

    assert "VEHICLE_DETECTION_CONFIDENCE=0.001" in env
    assert "VEHICLE_CONFIRM_RAW_MIN_CONFIDENCE=0.001" in env
    assert "STAGE_TRACKING_MAX_MISSED_AI=0" in env
    assert "KMA_SERVICE_KEY=\n" in env
    assert "KMA_APIHUB_AUTH_KEY=\n" in env
    assert "VWORLD_API_KEY=\n" in env
    assert "torch==2.10.0" not in requirements
    assert "torch.backends.cudnn.benchmark = False" in pipeline
    assert "Central GPU scheduler started: single predict owner" in pipeline
    assert "if has_live_cctv_clients():" in main
    assert "opportunistic_live" not in main
    pohang = POHANG.read_text(encoding="utf-8")
    assert "canonical_by_track" in pohang
    assert "merged_fast.append(_merge_stage_state(item, canonical))" in pohang


def test_stage_merge_preserves_valid_stage_and_vehicle_identity():
    (_merge_stage_state,) = _load_functions("_merge_stage_state")
    base = {
        "bbox": [10, 20, 30, 40],
        "track_id": 7,
        "vehicle_label": "suv",
        "vehicle_conf": 0.81,
        "source_label": "suv",
        "class_id": 2,
        "stage": 2,
        "stage_valid": True,
        "stage_source": "tire",
        "stage_conf": 0.91,
    }
    rejected = {
        "stage": None,
        "stage_valid": False,
        "stage_rejected_low_confidence": True,
        "conf": 0.42,
        "source_label": "level_1",
        "class_id": 1,
    }
    kept = _merge_stage_state(base, rejected)
    assert kept["stage"] == 2
    assert kept["stage_conf"] == 0.91
    assert kept["vehicle_label"] == "suv"
    assert kept["source_label"] == "suv"
    assert kept["class_id"] == 2
    assert kept["stage_stale"] is True

    valid = {
        "stage": 3,
        "stage_valid": True,
        "stage_source": "car_body",
        "stage_model_label": "level_3",
        "conf": 0.88,
        "source_label": "level_3",
        "class_id": 3,
    }
    updated = _merge_stage_state(base, valid)
    assert updated["stage"] == 3
    assert updated["stage_conf"] == 0.88
    assert updated["vehicle_label"] == "suv"
    assert updated["source_label"] == "suv"
    assert updated["class_id"] == 2


def test_projection_is_display_only_bounded_and_non_mutating():
    _copy, _clip, project = _load_functions(
        "_copy_detections", "_clip_bbox", "_project_detections_by_velocity"
    )
    original = [{"bbox": [100.0, 100.0, 200.0, 160.0], "_velocity": [30.0, 10.0]}]
    out = project(
        original,
        delta_seconds=0.12,
        detector_interval_seconds=0.18,
        width=640,
        height=360,
    )
    assert original[0]["bbox"] == [100.0, 100.0, 200.0, 160.0]
    assert out[0]["bbox"][0] > 100.0
    assert out[0]["bbox"][1] > 100.0
    # 0.12/0.18 of 30px ~= 20px; well under the safety cap.
    assert out[0]["bbox"][0] <= 121.0
    assert out[0]["bbox"][1] <= 107.0


def test_full_tire_then_body_fallback_pipeline_is_unchanged():
    text = (ROOT / "app" / "vehicle_flood_pipeline.py").read_text(encoding="utf-8")
    tire_pos = text.index("tire_stages")
    fallback_pos = text.index("fallback_indices", tire_pos)
    body_pos = text.index("body_stages", fallback_pos)
    assert tire_pos < fallback_pos < body_pos
    assert "if tire_stage is None:" in text
    assert "body_model" in text
