import numpy as np

from app.inference import FloodDetector


def test_demo_predict():
    detector = FloodDetector("", demo_mode=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.predict(frame)
    assert 0 <= result.max_level <= 4
    assert result.detections
