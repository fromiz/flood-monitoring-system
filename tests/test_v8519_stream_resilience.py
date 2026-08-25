from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
MAIN = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
ENV = (ROOT / ".env").read_text(encoding="utf-8")


def test_extensionless_hls_segments_are_supported():
    assert "def _playlist_uri_lines" in WORKER
    parser = WORKER.split("def _parse_hls", 1)[1].split("def _download_latest_hls_segment", 1)[0]
    assert "media_extensions" not in parser
    assert "#EXT-X-STREAM-INF" in parser


def test_stream_failure_frame_reaches_raw_and_annotated_clients():
    error_block = WORKER.split("def _publish_stream_error", 1)[1].split("def _grab_loop", 1)[0]
    assert "self.latest_jpeg = jpeg" in error_block
    assert "self.latest_raw_jpeg = jpeg" in error_block


def test_mjpeg_generators_have_keepalive_yields():
    annotated = WORKER.split("def annotated_mjpeg", 1)[1].split("def raw_mjpeg", 1)[0]
    raw = WORKER.split("def raw_mjpeg", 1)[1]
    assert "timeout=1.0" in annotated
    assert "jpeg_seq == last_seq" not in annotated
    assert "timeout=1.0" in raw
    assert "jpeg_seq == last_seq" not in raw


def test_lightweight_status_routes_bypass_sync_threadpool():
    assert '@app.get("/api/environment-history/status")\nasync def environment_history_status' in MAIN
    assert '@app.get("/api/cctv/background-status")\nasync def cctv_background_status' in MAIN
    assert '@app.get("/api/cctv/worker-status")\nasync def cctv_stream_worker_status' in MAIN


def test_dashboard_recovers_transient_status_and_geojson_failures():
    assert "fetchJsonWithRetry" in FRONTEND
    assert "통합 DEM 침수 GeoJSON 정상" in FRONTEND
    assert "통합 침수 GeoJSON 재연결 중" in FRONTEND


def test_background_scan_defaults_to_one_worker_in_packaged_env():
    assert "BACKGROUND_CCTV_WORKERS=1" in ENV
