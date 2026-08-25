from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np


def test_source_has_single_owner_scheduler_and_no_busy_wait_locks():
    root = Path(__file__).resolve().parents[1]
    pipeline = (root / 'app' / 'vehicle_flood_pipeline.py').read_text(encoding='utf-8')
    pohang = (root / 'app' / 'pohang_cctv.py').read_text(encoding='utf-8')

    assert 'class _CentralInferenceScheduler' in pipeline
    assert 'single-owner-priority-scheduler' in pipeline
    assert '_VEHICLE_PREDICT_LOCK' not in pipeline
    assert '_STAGE_PREDICT_LOCK' not in pipeline
    assert '_VEHICLE_WAITERS' not in pipeline
    assert 'def _vehicle_waiter_count' not in pipeline
    assert 'def _load_model()' not in pohang  # removed dead legacy best.pt loader


def test_scheduler_serializes_and_micro_batches_vehicle_calls():
    # Import only after source invariants. In the real project ultralytics is a
    # runtime dependency, so this exercises the exact scheduler implementation.
    from app.vehicle_flood_pipeline import _CentralInferenceScheduler

    class FakeModel:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()
            self.batch_sizes: list[int] = []

        def predict(self, source=None, **kwargs):
            batch_size = len(source) if isinstance(source, list) else 1
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.batch_sizes.append(batch_size)
            time.sleep(0.015)
            with self.lock:
                self.active -= 1
            return [object() for _ in range(batch_size)]

    scheduler = _CentralInferenceScheduler()
    model = FakeModel()
    errors: list[BaseException] = []

    def submit(kind: str, delay: float = 0.0):
        time.sleep(delay)
        try:
            result = scheduler.submit(
                kind,
                model,
                np.zeros((32, 32, 3), dtype=np.uint8),
                {'conf': 0.1, 'imgsz': 32, 'verbose': False, 'device': 'cpu'},
                live_priority=True,
                timeout_seconds=5.0,
            )
            assert len(result) == 1
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=submit, args=('vehicle', i * 0.001))
        for i in range(4)
    ] + [
        threading.Thread(target=submit, args=('tire_level', 0.002 + i * 0.001))
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert not errors
    assert model.max_active == 1
    # The four nearly simultaneous full-frame vehicle requests should be eligible
    # for a micro-batch, while stage calls stay serialized behind the same owner.
    assert max(model.batch_sizes) >= 2
    status = scheduler.status()
    assert status['failed'] == 0
    assert status['completed'] == 6
    assert status['vehicle_queue'] == 0
    assert status['stage_queue'] == 0


def test_full_pipeline_and_change_only_alerts_remain_configured():
    root = Path(__file__).resolve().parents[1]
    env = (root / '.env').read_text(encoding='utf-8')
    app_js = (root / 'app' / 'static' / 'app.js').read_text(encoding='utf-8')

    assert 'STAGE2_MODEL_PATH=weights/best.pt' in env
    assert 'TIRE_LEVEL_MODEL_PATH=weights/tire_level.pt' in env
    assert 'CAR_FLOOD_CLS_MODEL_PATH=weights/car_flood_cls.pt' in env
    assert 'changes_only=true' in app_js
