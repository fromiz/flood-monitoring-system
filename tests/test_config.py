from app.config import parse_roi


def test_parse_roi():
    roi = parse_roi("0.1,0.2;0.9,0.2;0.9,0.8")
    assert roi == [(0.1, 0.2), (0.9, 0.2), (0.9, 0.8)]
