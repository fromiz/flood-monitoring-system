from pathlib import Path

ROOT = Path(__file__).resolve().parent
POHANG = (ROOT / 'app' / 'pohang_cctv.py').read_text(encoding='utf-8')
MAIN = (ROOT / 'app' / 'main.py').read_text(encoding='utf-8')
JS = (ROOT / 'app' / 'static' / 'app.js').read_text(encoding='utf-8')
ENV = (ROOT / '.env').read_text(encoding='utf-8')
PIPE = (ROOT / 'app' / 'vehicle_flood_pipeline.py').read_text(encoding='utf-8')


def test_ai_activation_is_sticky_and_box_thread_is_critical():
    assert 'self.ai_activation = threading.Event()' in POHANG
    assert 'self.ai_activation.set()' in POHANG
    assert '"cctv-box-detector"' in POHANG
    assert 'CCTV AI37 thread-start' in POHANG
    assert 'CCTV BOX37 raw=' in POHANG
    assert 'CCTV WS37 annotated=' in POHANG


def test_first_live_pass_skips_rescue_until_first_publish():
    assert 'self.last_ai_success_at > 0.0' in POHANG
    assert 'self.last_ai_success_at = self.latest_result_at' in POHANG


def test_bottom_board_requests_stage_changes_only():
    assert 'changes_only: bool = Query(default=False)' in MAIN
    assert 'changes_only=true' in JS
    assert 'int(row.level) == int(older.level)' in MAIN


def test_background_startup_does_not_run_trusted_test_ai():
    assert 'if bool(camera.get("trusted_baseline")):' in MAIN
    assert 'result = self._store_baseline(camera)' in MAIN
    assert 'BACKGROUND_CCTV_START_DELAY_SECONDS=20' in ENV


def test_full_pipeline_order_is_preserved():
    tire_pos = PIPE.index('tire_level')
    body_pos = PIPE.index('car_flood_cls')
    assert tire_pos < body_pos
    assert 'vehicle_candidates' in PIPE
