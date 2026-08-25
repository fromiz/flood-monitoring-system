from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKER = (ROOT / "app" / "pohang_cctv.py").read_text(encoding="utf-8")
FRONTEND = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (ROOT / "app" / "static" / "style.css").read_text(encoding="utf-8")
INDEX = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_mjpeg_first_frame_does_not_depend_on_img_onload_only():
    assert "function cctvImageReady" in FRONTEND
    assert "naturalWidth > 0" in FRONTEND
    assert "frameWatchTimer" in FRONTEND
    assert "250\n  );" in FRONTEND or "250\r\n  );" in FRONTEND


def test_raw_layer_remains_as_nonpersistent_fallback_while_ai_changes():
    assert 'class="cctv-video cctv-video-canvas hidden"' in FRONTEND
    assert "new WebSocket(cctvTransportUrl())" in FRONTEND
    assert "header.mode === 'status' && win.hasRealFrame" in FRONTEND
    assert "function stopAnnotatedCctvStream" in FRONTEND


def test_loading_overlay_does_not_cover_a_decoded_frame():
    status_block = FRONTEND.split("async function refreshWindowStreamStatus", 1)[1].split(
        "function rawSnapshotUrl", 1
    )[0]
    assert "win.hasRealFrame" in status_block
    assert "loading?.classList.add('hidden')" in status_block
    assert "ensureCctvWebSocket()" in status_block


def test_hls_worker_reuses_media_playlist_and_stays_on_working_fallback():
    assert "def _media_playlist_segments" in WORKER
    assert "media_playlist_url" in WORKER
    assert 'hls_fallback_until = float("inf")' in WORKER
    assert "_stream_http_local = threading.local()" in WORKER


def test_generic_stream_requests_do_not_force_origin_header():
    generic_header_block = WORKER.split("_session.headers.update({", 1)[1].split("})", 1)[0]
    assert '"Origin"' not in generic_header_block
    assert '"Referer"' in generic_header_block


def test_v8522_assets_are_cache_busted_and_dual_layer_css_exists():
    assert "/static/app.js?v=8.5.24" in INDEX
    assert ".cctv-video-raw" in STYLE
    assert ".cctv-video-ai" in STYLE
