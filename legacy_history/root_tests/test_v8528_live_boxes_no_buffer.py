from pathlib import Path

ROOT = Path(__file__).resolve().parent


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_video_and_ai_are_separate_threads():
    source = text("app/pohang_cctv.py")
    for target in (
        "target=self._grab_loop",
        "target=self._ai_loop",
        "target=self._stage_loop",
        "target=self._render_loop",
    ):
        assert target in source


def test_gpu_renderer_and_transport_are_not_artificially_5fps():
    worker = text("app/pohang_cctv.py")
    server = text("app/main.py")
    assert "self.render_fps = 15.0" in worker
    assert "0.065 if focused else 0.085" in server


def test_browser_does_not_discard_every_decoded_frame_under_load():
    js = text("app/static/app.js")
    assert "Always paint the JPEG that has already finished decoding" in js
    assert "pendingWsFrame remains a" in js
    assert "if (win.pendingWsFrame && Number(win.pendingWsFrame.receivedAt) > packet.receivedAt)" not in js


def test_hls_prefetch_is_latest_only():
    source = text("app/pohang_cctv.py")
    assert "queue.Queue(maxsize=1)" in source
    assert "return segments[-1]" in source


def test_successful_vehicle_result_is_not_dropped_as_stale():
    source = text("app/pohang_cctv.py")
    assert "Never throw away a successful best.pt result" in source
    assert '"detector_lag_seconds"' in source


def test_vehicle_threshold_allows_distant_custom_model_boxes():
    pipeline = text("app/vehicle_flood_pipeline.py")
    env = text(".env")
    assert "confidence = max(\n        0.06," in pipeline
    assert (
        "VEHICLE_DETECTION_CONFIDENCE=0.12" in env
        or "VEHICLE_DETECTION_CONFIDENCE=0.16" in env
    )
    assert (
        "VEHICLE_DETECTION_RESCUE_CONFIDENCE=0.08" in env
        or "VEHICLE_DETECTION_RESCUE_CONFIDENCE=0.12" in env
    )


def test_flood_pipeline_order_is_vehicle_then_tire_then_body_fallback():
    pipeline = text("app/vehicle_flood_pipeline.py")
    tire_pos = pipeline.index("tire_results =")
    fallback_pos = pipeline.index("fallback_indices: list[int]", tire_pos)
    body_pos = pipeline.index("if fallback_indices and body_model is not None", fallback_pos)
    assert tire_pos < fallback_pos < body_pos
    assert '"track_id": vehicle.get("track_id")' in pipeline


def test_gpu_vehicle_and_stage_have_separate_scheduling_lanes():
    pipeline = text("app/vehicle_flood_pipeline.py")
    assert "_VEHICLE_PREDICT_LOCK = threading.RLock()" in pipeline
    assert "_STAGE_PREDICT_LOCK = threading.RLock()" in pipeline
    assert '"tire_level": _STAGE_PREDICT_LOCK' in pipeline
    assert '"car_flood_cls": _STAGE_PREDICT_LOCK' in pipeline


def test_ai_errors_are_visible_in_console_without_spamming():
    source = text("app/pohang_cctv.py")
    assert "CCTV vehicle detector failed [%s]: %s" in source
    assert "CCTV tire/body stage pipeline failed [%s]: %s" in source
