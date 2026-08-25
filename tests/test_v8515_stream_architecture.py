from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "app" / "pohang_cctv.py").read_text(
    encoding="utf-8"
)


def test_box_and_stage_workers_are_independent():
    assert 'name="cctv-box-detector"' in SOURCE
    assert 'name="cctv-stage-classifier"' in SOURCE
    assert "def _stage_loop(self)" in SOURCE


def test_stage_and_geometry_use_separate_mailboxes():
    assert "self.latest_ai_packet" in SOURCE
    assert "self.latest_stage_packet" in SOURCE
    assert '"geometry_update": False' in SOURCE


def test_hls_fallback_prefetches_without_repeating_direct_open():
    grab = SOURCE.split("def _grab_loop", 1)[1].split("def _get_latest_frame", 1)[0]
    assert "_HlsSegmentPrefetcher" in grab
    assert "hls_fallback_until" in grab
    assert "now < hls_fallback_until" in grab
    assert grab.count("capture = self._open_capture()") == 1

