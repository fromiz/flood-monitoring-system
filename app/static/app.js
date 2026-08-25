console.info('POHANG FLOOD CONTROL V8.6.4 loaded');
const $ = id => document.getElementById(id);

function waitMs(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchJsonWithRetry(url, options = {}, attempts = 2, timeoutMs = 6000) {
  let lastError = null;
  const totalAttempts = Math.max(1, Number(attempts) || 1);
  for (let attempt = 1; attempt <= totalAttempts; attempt += 1) {
    const controller = timeoutMs > 0 ? new AbortController() : null;
    const timer = controller
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;
    try {
      const response = await fetch(url, {
        cache: 'no-store',
        ...options,
        ...(controller ? { signal: controller.signal } : {})
      });
      if (!response.ok) {
        let message = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          message = body.detail || message;
        } catch (error) { }
        throw new Error(message);
      }
      return await response.json();
    } catch (error) {
      lastError = error?.name === 'AbortError'
        ? new Error(`응답 시간 초과 (${Math.round(timeoutMs / 1000)}초)`)
        : error;
      if (attempt < totalAttempts) {
        await waitMs(350 * attempt);
      }
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
  throw lastError || new Error('요청 실패');
}

const LEVEL_COLORS = window.POHANG_LEVEL_COLORS || Object.freeze([
  '#42c889',
  '#315cff',
  '#7136d9',
  '#bd2caf',
  '#ff4298'
]);

// Existing functions continue to use COLORS, but COLORS and the flood
// polygons now reference the exact same immutable palette.
const COLORS = LEVEL_COLORS;
const DEPTHS = [0, 8, 25, 48, 70];
const FLOOD_COLORS = [
  'rgba(0,0,0,0)',
  ...LEVEL_COLORS.slice(1)
];

LEVEL_COLORS.forEach((color, level) => {
  document.documentElement.style.setProperty(
    `--lv${level}`,
    color
  );
});
const POHANG_BOUNDS = [[128.72, 35.63], [129.78, 36.48]];
let map, realCameras = [], realStages = {}, cctvWindows = new Map(), windowZ = 40;
let cctvSocket = null, cctvSocketReconnectTimer = null, cctvSocketGeneration = 0;
let cctvSocketLastMessageAt = 0, cctvSocketLastSyncAt = 0;
const cctvTransportDecoder = new TextDecoder('utf-8');
let selectedCameraId = null;
let pendingVworldFocus = null;
let vworldFocusSequence = 0;
let vworldFocusTimers = [];
let surfaceVisible = true, markersVisible = true, rainfallVisible = true, buildingsVisible = true, is3d = true, terrainReady = false, surfaceTimer = null;
let latestWeather = null, latestRainGrid = [], rainChart = null, weatherEventSource = null, weatherFallbackTimer = null;
let mapProvider = { configured: false, provider: 'VWorld' }, activeBaseMap = 'vworld-hybrid', vworldErrorCount = 0;
let mapEngine = 'vworld3d';
let vworld3dReady = false;
let latestFloodSurface = { type: 'FeatureCollection', features: [] };
let latestUnifiedOverlayPayload = null;
let unifiedFloodTestEnabled = false;
let unifiedOverlayLoading = false;
let unifiedOverlayPending = false;
let unifiedOverlaySequence = 0;
let recentFloodEvents = [], eventReloadTimer = null;
let elevationMarker = null;

function setStatus(id, text, type = '') { const el = $(id); if (!el) return; el.textContent = text; el.className = type; }
function clock() { const el = $('clock'); if (el) el.textContent = new Date().toLocaleString('ko-KR'); }
function cameraMinimumStage(c) {
  if (!c?.local_test || !c?.trusted_baseline) return 0;
  return Math.max(1, Math.min(4, Number(c.minimum_stage ?? 1) || 1));
}
function stageRecordForCamera(c) {
  const id = String(c?.id ?? '');
  return (id && realStages[id]) || realStages[`name:${c?.name || ''}`] || null;
}
function stageForCamera(c) {
  const record = stageRecordForCamera(c);
  let stage = Math.max(0, Math.min(4, Number(record?.stage ?? 0) || 0));
  if (stage > 0 && !c?.local_test && record?.positive_confirmed !== true) {
    stage = 0;
  }
  return Math.max(stage, cameraMinimumStage(c));
}
function stageTimestamp(value) {
  const text = String(value || '');
  const time = Date.parse(/[zZ]$|[+-]\d\d:\d\d$/.test(text) ? text : `${text}Z`);
  return Number.isFinite(time) ? time : 0;
}
function setCameraStage(
  camera,
  stage,
  confidence = 0,
  detectedAt = null,
  positiveConfirmed = false
) {
  const id = String(camera?.id ?? '');
  if (!id) return 0;
  let resolved = Math.max(
    cameraMinimumStage(camera),
    Math.max(0, Math.min(4, Number(stage) || 0))
  );
  if (resolved > 0 && !camera?.local_test && positiveConfirmed !== true) {
    return stageForCamera(camera);
  }
  realStages[id] = {
    ...(realStages[id] || {}),
    stage: resolved,
    confidence: Number(confidence || 0),
    positive_confirmed: resolved === 0 || positiveConfirmed === true,
    detected_at: detectedAt || new Date().toISOString()
  };
  return resolved;
}
function updateSummary() { const levels = realCameras.map(stageForCamera); const max = levels.length ? Math.max(...levels) : 0; $('max-level').textContent = max; $('max-depth').textContent = DEPTHS[max]; $('open-count').textContent = cctvWindows.size; }

async function initMap() {
  try {
    const response = await fetch('/api/map/config', { cache: 'no-store' });
    if (response.ok) mapProvider = await response.json();
  } catch (error) {
    mapProvider = { configured: false, provider: 'VWorld', message: error.message };
  }

  activeBaseMap = mapProvider.configured ? 'vworld-hybrid' : 'fallback';
  setStatus(
    'vworld-status',
    mapProvider.configured ? '브이월드 WMTS 정상' : '브이월드 키 미설정 · 예비지도',
    mapProvider.configured ? 'ok' : 'warn'
  );

  map = new maplibregl.Map({
    container: 'map',
    center: [129.343, 36.019],
    zoom: 11.8,
    pitch: 62,
    bearing: -15,
    maxPitch: 85,
    maxZoom: 19,
    maxBounds: POHANG_BOUNDS,
    renderWorldCopies: false,
    canvasContextAttributes: { antialias: true },
    style: {
      version: 8,
      glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
      sources: {
        vworldBase: {
          type: 'raster',
          tiles: ['/api/map/vworld/Base/{z}/{x}/{y}.png'],
          tileSize: 256,
          maxzoom: 19,
          attribution: '공간정보 오픈플랫폼(브이월드)'
        },
        vworldSatellite: {
          type: 'raster',
          tiles: ['/api/map/vworld/Satellite/{z}/{x}/{y}.jpeg'],
          tileSize: 256,
          maxzoom: 19,
          attribution: '공간정보 오픈플랫폼(브이월드)'
        },
        vworldHybrid: {
          type: 'raster',
          tiles: ['/api/map/vworld/Hybrid/{z}/{x}/{y}.png'],
          tileSize: 256,
          maxzoom: 19,
          attribution: '공간정보 오픈플랫폼(브이월드)'
        },
        fallbackSatellite: {
          type: 'raster',
          tiles: [
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
          ],
          tileSize: 256,
          maxzoom: 19,
          attribution: 'Tiles © Esri'
        },
        terrainSource: {
          type: 'raster-dem',
          url: 'https://tiles.mapterhorn.com/tilejson.json',
          tileSize: 512,
          maxzoom: 14
        },
        hillshadeSource: {
          type: 'raster-dem',
          url: 'https://tiles.mapterhorn.com/tilejson.json',
          tileSize: 512,
          maxzoom: 14
        },
        openfreemap: {
          type: 'vector',
          url: 'https://tiles.openfreemap.org/planet',
          attribution: '© OpenStreetMap contributors'
        }
      },
      layers: [
        {
          id: 'vworld-base',
          type: 'raster',
          source: 'vworldBase',
          layout: { visibility: activeBaseMap === 'vworld-base' ? 'visible' : 'none' }
        },
        {
          id: 'vworld-satellite',
          type: 'raster',
          source: 'vworldSatellite',
          layout: {
            visibility: ['vworld-satellite', 'vworld-hybrid'].includes(activeBaseMap) ? 'visible' : 'none'
          },
          paint: {
            'raster-saturation': -.03,
            'raster-contrast': .06,
            'raster-brightness-min': .03,
            'raster-brightness-max': .96
          }
        },
        {
          id: 'fallback-base',
          type: 'raster',
          source: 'fallbackSatellite',
          layout: { visibility: activeBaseMap === 'fallback' ? 'visible' : 'none' },
          paint: {
            'raster-saturation': -.04,
            'raster-contrast': .08,
            'raster-brightness-min': .03,
            'raster-brightness-max': .93
          }
        },
        {
          id: 'hillshade',
          type: 'hillshade',
          source: 'hillshadeSource',
          paint: {
            'hillshade-exaggeration': .38,
            'hillshade-shadow-color': '#07121d',
            'hillshade-highlight-color': '#d7edf6',
            'hillshade-accent-color': '#345267'
          }
        },
        {
          id: 'vworld-hybrid-labels',
          type: 'raster',
          source: 'vworldHybrid',
          layout: { visibility: activeBaseMap === 'vworld-hybrid' ? 'visible' : 'none' },
          paint: { 'raster-opacity': .96 }
        }
      ]
    }
  });

  map.addControl(
    new maplibregl.NavigationControl({ visualizePitch: true }),
    'top-left'
  );
  map.addControl(
    new maplibregl.ScaleControl({ unit: 'metric' }),
    'bottom-right'
  );

  map.on('error', event => {
    const sourceId = String(event?.sourceId || event?.source?.id || '');
    if (sourceId.startsWith('vworld')) {
      vworldErrorCount += 1;
      if (vworldErrorCount >= 2 && activeBaseMap !== 'fallback') {
        setStatus('vworld-status', '브이월드 타일 오류 · 예비지도 전환', 'error');
        setBaseMap('fallback');
      }
    }
  });

  map.on('load', () => {
    try {
      map.setTerrain({
        source: 'terrainSource',
        exaggeration: Number($('terrain-exaggeration').value)
      });
      terrainReady = true;
    } catch (error) {
      terrainReady = false;
    }

    addMapSources();

    map.on('click', 'cctv-circles', event => {
      if (mapEngine === 'analysis') return;
      const id = event.features?.[0]?.properties?.camera_id;
      const idx = realCameras.findIndex(
        camera => String(camera.id) === String(id)
      );
      if (idx >= 0) {
        selectCameraByIndex(idx, { openWindow: true });
      }
    });

    map.on('mouseenter', 'cctv-circles', () => {
      map.getCanvas().style.cursor = mapEngine === 'analysis' ? 'crosshair' : 'pointer';
    });
    map.on('mouseleave', 'cctv-circles', () => {
      map.getCanvas().style.cursor = mapEngine === 'analysis' ? 'crosshair' : '';
    });

    ['flood-surface', 'flood-volume-3d'].forEach(layerId => {
      map.on('click', layerId, event => {
        if (mapEngine === 'analysis') return;
        const properties = event.features?.[0]?.properties;
        if (!properties) return;

        new maplibregl.Popup()
          .setLngLat(event.lngLat)
          .setHTML(
            `<b>Python DEM 통합 침수 추정</b><br>` +
            `추정수심 ${Number(properties.depth_cm || 0).toFixed(1)}cm · ` +
            `Lev${Number(properties.level || 0)}<br>` +
            `DEM 고도 ${Number(properties.elevation_m || 0).toFixed(1)}m · ` +
            `강수 ${Number(properties.rain_mm || 0).toFixed(1)}mm<br>` +
            `기준 CCTV ${escapeHtml(properties.source_name || properties.source_id || '-')}`
          )
          .addTo(map);
      });
    });

    map.on('click', 'rain-points', event => {
      if (mapEngine === 'analysis') return;
      const properties = event.features?.[0]?.properties;
      if (!properties) return;
      new maplibregl.Popup()
        .setLngLat(event.lngLat)
        .setHTML(
          `<b>${escapeHtml(properties.name)}</b><br>` +
          `1시간 강수 ${Number(properties.rain_mm || 0).toFixed(1)}mm<br>` +
          `기온 ${Number(properties.temperature_c || 0).toFixed(1)}℃ · ` +
          `습도 ${Number(properties.humidity_pct || 0).toFixed(0)}%`
        )
        .addTo(map);
    });

    updateMapData();
    updateRainMap();
    scheduleSurface();
  });

  // V8.5.0: 별도 ON/OFF 버튼 없이 침수 분석지도 클릭 자체가 DEM 고도 조회입니다.
  // CCTV/침수/강수 레이어를 눌러도 브이월드 3D로 전환하지 않습니다.
  map.on('click', event => {
    if (mapEngine === 'analysis') {
      showElevationPopup(event.lngLat);
    }
  });

  map.on('idle', () => {
    if (!terrainReady) {
      try {
        map.setTerrain({
          source: 'terrainSource',
          exaggeration: Number($('terrain-exaggeration').value)
        });
        terrainReady = true;
      } catch (error) { }
    }
  });
}

function addMapSources() {
  if (!map.getSource('rainfall')) {
    map.addSource('rainfall', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });
  }

  if (!map.getLayer('rain-heat')) {
    map.addLayer({
      id: 'rain-heat',
      type: 'heatmap',
      source: 'rainfall',
      maxzoom: 15,
      paint: {
        'heatmap-weight': [
          'interpolate', ['linear'], ['get', 'rain_mm'],
          0, 0,
          1, .15,
          5, .45,
          15, .8,
          30, 1
        ],
        'heatmap-intensity': [
          'interpolate', ['linear'], ['zoom'],
          8, .6,
          13, 1.6
        ],
        'heatmap-radius': [
          'interpolate', ['linear'], ['zoom'],
          8, 25,
          13, 70
        ],
        'heatmap-opacity': .72,
        'heatmap-color': [
          'interpolate', ['linear'], ['heatmap-density'],
          0, 'rgba(0,0,0,0)',
          .12, 'rgba(40,119,255,.25)',
          .35, 'rgba(55,83,255,.55)',
          .58, 'rgba(105,48,214,.70)',
          .78, 'rgba(190,40,172,.82)',
          1, 'rgba(255,55,139,.92)'
        ]
      }
    });
  }

  if (!map.getLayer('rain-points')) {
    map.addLayer({
      id: 'rain-points',
      type: 'circle',
      source: 'rainfall',
      minzoom: 10,
      paint: {
        'circle-radius': [
          'interpolate', ['linear'], ['get', 'rain_mm'],
          0, 4,
          5, 8,
          20, 14,
          40, 18
        ],
        'circle-color': [
          'interpolate', ['linear'], ['get', 'rain_mm'],
          0, '#61d3ff',
          5, '#376dff',
          15, '#7b3cdb',
          30, '#ed3aae'
        ],
        'circle-opacity': .85,
        'circle-stroke-color': '#e9f7ff',
        'circle-stroke-width': 1.5
      }
    });
  }

  if (!map.getSource('floodSurface')) {
    map.addSource('floodSurface', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });
  }

  if (!map.getLayer('flood-surface')) {
    map.addLayer({
      id: 'flood-surface',
      type: 'fill',
      source: 'floodSurface',
      paint: {
        'fill-color': [
          'match', ['get', 'level'],
          1, FLOOD_COLORS[1],
          2, FLOOD_COLORS[2],
          3, FLOOD_COLORS[3],
          4, FLOOD_COLORS[4],
          'rgba(0,0,0,0)'
        ],
        'fill-opacity': .36,
        'fill-outline-color': 'rgba(235,245,255,.22)'
      }
    });
  }

  if (!map.getLayer('flood-volume-3d')) {
    map.addLayer({
      id: 'flood-volume-3d',
      type: 'fill-extrusion',
      source: 'floodSurface',
      minzoom: 10,
      paint: {
        'fill-extrusion-color': [
          'match', ['get', 'level'],
          1, FLOOD_COLORS[1],
          2, FLOOD_COLORS[2],
          3, FLOOD_COLORS[3],
          4, FLOOD_COLORS[4],
          'rgba(0,0,0,0)'
        ],
        'fill-extrusion-height': [
          '*',
          ['get', 'depth_m'],
          Number($('flood-exaggeration').value)
        ],
        'fill-extrusion-base': 0,
        'fill-extrusion-opacity': .50,
        'fill-extrusion-vertical-gradient': true
      }
    });
  }

  if (!map.getLayer('pohang-3d-buildings')) {
    map.addLayer({
      id: 'pohang-3d-buildings',
      type: 'fill-extrusion',
      source: 'openfreemap',
      'source-layer': 'building',
      minzoom: 12.5,
      filter: ['!=', ['get', 'hide_3d'], true],
      paint: {
        'fill-extrusion-color': [
          'interpolate', ['linear'],
          ['coalesce', ['get', 'render_height'], 0],
          0, '#b9c4cb',
          30, '#d2d9de',
          100, '#e2e7ea'
        ],
        'fill-extrusion-height': [
          'interpolate', ['linear'], ['zoom'],
          12.5, 0,
          14, [
            'coalesce',
            ['get', 'render_height'],
            ['*', ['coalesce', ['get', 'levels'], 1], 3]
          ]
        ],
        'fill-extrusion-base': [
          'coalesce', ['get', 'render_min_height'], 0
        ],
        'fill-extrusion-opacity': .76
      }
    });
  }

  if (!map.getSource('cctv')) {
    map.addSource('cctv', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });
  }

  if (!map.getLayer('cctv-circles')) {
    map.addLayer({
      id: 'cctv-circles',
      type: 'circle',
      source: 'cctv',
      paint: {
        'circle-radius': 13,
        'circle-color': [
          'match', ['get', 'stage'],
          0, COLORS[0],
          1, COLORS[1],
          2, COLORS[2],
          3, COLORS[3],
          4, COLORS[4],
          COLORS[0]
        ],
        'circle-stroke-color': '#eaf6ff',
        'circle-stroke-width': 2,
        'circle-opacity': .96
      }
    });
  }

  if (!map.getLayer('cctv-labels')) {
    map.addLayer({
      id: 'cctv-labels',
      type: 'symbol',
      source: 'cctv',
      layout: {
        'text-field': [
          'concat', 'Lev', ['to-string', ['get', 'stage']]
        ],
        'text-size': 9,
        'text-font': ['Open Sans Bold'],
        'text-allow-overlap': true
      },
      paint: { 'text-color': '#07131d' }
    });
  }

  setMapLayerVisibility();
}

function setMapLayerVisibility() {
  const surfaceVisibility = surfaceVisible ? 'visible' : 'none';
  ['flood-surface', 'flood-volume-3d'].forEach(layerId => {
    if (map.getLayer(layerId)) {
      map.setLayoutProperty(
        layerId,
        'visibility',
        surfaceVisibility
      );
    }
  });

  const rainVisibility = rainfallVisible ? 'visible' : 'none';
  ['rain-heat', 'rain-points'].forEach(layerId => {
    if (map.getLayer(layerId)) {
      map.setLayoutProperty(
        layerId,
        'visibility',
        rainVisibility
      );
    }
  });

  // 분석지도에서는 단순 돌출 건물을 항상 숨깁니다.
  // 실제 건물은 브이월드 WebGL의 facility_build 레이어를 사용합니다.
  if (map.getLayer('pohang-3d-buildings')) {
    map.setLayoutProperty(
      'pohang-3d-buildings',
      'visibility',
      'none'
    );
  }

  ['cctv-circles', 'cctv-labels'].forEach(layerId => {
    if (map.getLayer(layerId)) {
      map.setLayoutProperty(
        layerId,
        'visibility',
        markersVisible ? 'visible' : 'none'
      );
    }
  });
}

function setBaseMap(mode) {
  const allowed = ['vworld-base', 'vworld-satellite', 'vworld-hybrid', 'fallback'];
  if (!allowed.includes(mode)) return;

  if (mode.startsWith('vworld') && !mapProvider.configured) {
    setStatus('vworld-status', '브이월드 인증키를 .env에 입력하세요.', 'warn');
    mode = 'fallback';
  }

  activeBaseMap = mode;
  const visibility = {
    'vworld-base': mode === 'vworld-base',
    'vworld-satellite': ['vworld-satellite', 'vworld-hybrid'].includes(mode),
    'vworld-hybrid-labels': mode === 'vworld-hybrid',
    'fallback-base': mode === 'fallback'
  };

  Object.entries(visibility).forEach(([layerId, visible]) => {
    if (map.getLayer(layerId)) {
      map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
    }
  });

  ['base-vworld-base', 'base-vworld-satellite', 'base-vworld-hybrid', 'base-fallback']
    .forEach(id => $(id)?.classList.remove('active'));
  const activeId = {
    'vworld-base': 'base-vworld-base',
    'vworld-satellite': 'base-vworld-satellite',
    'vworld-hybrid': 'base-vworld-hybrid',
    fallback: 'base-fallback'
  }[mode];
  $(activeId)?.classList.add('active');

  if (mode === 'fallback') {
    setStatus(
      'vworld-status',
      mapProvider.configured ? '예비 위성지도 사용 중' : '브이월드 키 미설정 · 예비지도',
      'warn'
    );
  } else {
    const names = {
      'vworld-base': '일반지도',
      'vworld-satellite': '영상지도',
      'vworld-hybrid': '영상+하이브리드'
    };
    setStatus('vworld-status', `브이월드 ${names[mode]} 정상`, 'ok');
  }
}


function sendToVworld3d(message) {
  const frame = $('vworld3d-frame');
  try {
    frame?.contentWindow?.postMessage(
      message,
      window.location.origin
    );
  } catch (error) {
    console.warn('브이월드 3D 메시지 전송 실패', error);
  }
}

function clearVworldFocusTimers() {
  vworldFocusTimers.forEach(timer => clearTimeout(timer));
  vworldFocusTimers = [];
}

function transmitPendingVworldFocus() {
  if (!pendingVworldFocus) return;
  sendToVworld3d(pendingVworldFocus);
}

function queueVworldFocus(message) {
  clearVworldFocusTimers();

  const requestId = (
    `cctv-focus-${Date.now()}-${++vworldFocusSequence}`
  );
  pendingVworldFocus = {
    ...message,
    requestId
  };

  // iframe 전환·Cesium 초기화 시점 차이에도 요청이 유실되지 않도록
  // 짧은 간격부터 2.2초까지 재전송합니다. iframe에서는 requestId로
  // 중복 이동을 제거합니다.
  [60, 220, 520, 950, 1500, 2200].forEach(delay => {
    const timer = setTimeout(
      transmitPendingVworldFocus,
      delay
    );
    vworldFocusTimers.push(timer);
  });
}

function cameraCoordinates(camera) {
  const lat = Number(camera?.lat);
  const lon = Number(camera?.lon);
  return {
    lat: Number.isFinite(lat) ? lat : null,
    lon: Number.isFinite(lon) ? lon : null
  };
}

function focusCameraOnMap(camera) {
  const position = cameraCoordinates(camera);

  if (position.lat === null || position.lon === null) {
    alert(
      `${camera?.name || '선택한 CCTV'}의 지도 좌표가 없습니다.`
    );
    return false;
  }

  setMapEngine('vworld3d');
  queueVworldFocus({
    type: 'focus-map-location',
    source: 'cctv-selection',
    lat: position.lat,
    lon: position.lon,
    label: camera.name || String(camera.id),
    address: camera.address || '포항 CCTV',
    level: stageForCamera(camera),
    cameraId: String(camera.id),
    range_m: 620,
    centerOnTarget: true
  });
  return true;
}

function selectCameraByIndex(index, { openWindow = true } = {}) {
  const camera = realCameras[index];
  if (!camera) return;

  selectedCameraId = String(camera.id);
  renderCameraList();
  focusCameraOnMap(camera);

  if (openWindow) {
    openCctvWindow(index);
  }
}

function vworldCameraPayload() {
  return realCameras
    .filter(camera =>
      Number.isFinite(Number(camera.lat)) &&
      Number.isFinite(Number(camera.lon))
    )
    .map(camera => {
      const stage = stageForCamera(camera);
      const record = realStages[String(camera.id)] || (!camera.id ? realStages[`name:${camera.name || ''}`] : null) || {};
      return {
        id: String(camera.id),
        map_id: `${camera.id}:${realCameras.indexOf(camera)}:${Number(camera.lat).toFixed(6)}:${Number(camera.lon).toFixed(6)}`,
        camera_index: realCameras.indexOf(camera),
        name: camera.name || String(camera.id),
        address: camera.address || '포항 CCTV',
        lat: Number(camera.lat),
        lon: Number(camera.lon),
        stage,
        depth_cm: DEPTHS[stage],
        confidence: Number(
          record.confidence ??
          record.conf ??
          0
        ) || 0,
        local_test: Boolean(camera.local_test)
      };
    });
}

function syncVworldCameras() {
  // Camera markers must not depend on DEM/flood-surface generation.  The
  // supplied replacement lost the markers when its overlay request failed or
  // had not completed yet. Always send the authoritative CCTV coordinates
  // immediately, while retaining the latest valid flood GeoJSON separately.
  const cameras = vworldCameraPayload();
  latestUnifiedOverlayPayload = {
    ...(latestUnifiedOverlayPayload || {}),
    source: latestUnifiedOverlayPayload?.source || 'dashboard CCTV fallback V8.6.4',
    camera_count: cameras.length,
    cameras,
    flood: normaliseFloodGeoJson(
      latestUnifiedOverlayPayload?.flood || latestFloodSurface
    )
  };
  syncVworldFlood();
}

function syncVworldFlood() {
  if (!vworld3dReady || !latestUnifiedOverlayPayload) return;

  sendToVworld3d({
    type: 'set-vworld-overlay-payload',
    payload: latestUnifiedOverlayPayload
  });
}

function syncVworld3dLayers() {
  if (!vworld3dReady) return;

  sendToVworld3d({
    type: 'set-vworld-buildings',
    visible: buildingsVisible
  });
  sendToVworld3d({
    type: 'set-vworld-cctv-visible',
    visible: markersVisible
  });
  sendToVworld3d({
    type: 'set-vworld-flood-visible',
    visible: surfaceVisible
  });
}

function setMapEngine(mode) {
  if (!['vworld3d', 'analysis'].includes(mode)) return;

  mapEngine = mode;
  const wrap = $('map-wrap');
  const vworldMode = mode === 'vworld3d';

  wrap.classList.toggle('vworld-engine', vworldMode);
  wrap.classList.toggle('analysis-engine', !vworldMode);

  $('engine-vworld3d').classList.toggle('active', vworldMode);
  $('engine-analysis').classList.toggle('active', !vworldMode);

  document.querySelectorAll('.analysis-control').forEach(element => {
    element.classList.toggle('engine-hidden', vworldMode);
  });

  if (map?.getCanvas()) {
    map.getCanvas().style.cursor = vworldMode ? '' : 'crosshair';
  }
  if ($('elevation-readout') && !vworldMode) {
    $('elevation-readout').textContent =
      '침수 분석지도에서 지점을 클릭하면 해당 위치의 DEM 고도가 표시됩니다.';
  }

  if (vworldMode) {
    setStatus(
      'vworld-status',
      mapProvider.configured
        ? '브이월드 실제 3D 건물 모드'
        : '브이월드 인증키 확인 필요',
      mapProvider.configured ? 'ok' : 'warn'
    );
    syncVworld3dLayers();
  } else {
    window.setTimeout(() => {
      map?.resize();
      setMapLayerVisibility();
      scheduleSurface();
    }, 50);
    setStatus(
      'vworld-status',
      '브이월드 WMTS 침수 분석지도',
      mapProvider.configured ? 'ok' : 'warn'
    );
  }
}

function openCameraById(cameraId, cameraIndex = null) {
  const requestedIndex = Number(cameraIndex);
  let index = (
    Number.isInteger(requestedIndex) &&
    requestedIndex >= 0 &&
    requestedIndex < realCameras.length
  )
    ? requestedIndex
    : -1;

  if (index < 0) {
    index = realCameras.findIndex(camera =>
      String(camera.id) === String(cameraId)
    );
  }

  if (index < 0) {
    alert('CCTV 목록을 아직 불러오는 중입니다.');
    return;
  }

  selectCameraByIndex(index, { openWindow: true });
}

window.addEventListener('message', event => {
  if (event.origin !== window.location.origin) return;

  if (
    event.data?.type === 'open-test-camera' ||
    event.data?.type === 'open-camera'
  ) {
    openCameraById(
      event.data.cameraId || 'TEST-FLOOD-01',
      event.data.cameraIndex
    );
  }

  if (event.data?.type === 'vworld3d-ready') {
    vworld3dReady = true;
    setStatus(
      'vworld-status',
      '브이월드 ws3d.viewer · 통합 DEM 침수 GeoJSON 정상',
      'ok'
    );
    syncVworld3dLayers();
    syncVworldFlood();
    scheduleSurface();
    transmitPendingVworldFocus();
  }

  if (event.data?.type === 'vworld-refresh-overlay') {
    if (typeof event.data.test === 'boolean') {
      unifiedFloodTestEnabled = Boolean(event.data.test);
    }
    clearTimeout(surfaceTimer);
    surfaceTimer = null;
    buildElevationFloodSurface();
  }

  if (event.data?.type === 'focus-map-location-complete') {
    const requestId = String(event.data.requestId || '');
    if (
      pendingVworldFocus &&
      requestId &&
      requestId === String(pendingVworldFocus.requestId)
    ) {
      pendingVworldFocus = null;
      clearVworldFocusTimers();
    }
  }

  if (event.data?.type === 'vworld-elevation-picked') {
    const lat = Number(event.data.lat);
    const lon = Number(event.data.lon);
    const elevation = Number(event.data.elevation_m);
    if (
      Number.isFinite(lat) &&
      Number.isFinite(lon) &&
      Number.isFinite(elevation)
    ) {
      $('elevation-readout').textContent =
        `선택 지점 · 경도 ${lon.toFixed(5)} · ` +
        `위도 ${lat.toFixed(5)} · DEM 고도 ${elevation.toFixed(1)}m`;
    }
  }
});

function updateMapData() {
  const features = realCameras
    .filter(camera =>
      Number.isFinite(Number(camera.lon)) &&
      Number.isFinite(Number(camera.lat))
    )
    .map(camera => ({
      type: 'Feature',
      properties: {
        camera_id: String(camera.id),
        name: camera.name,
        stage: stageForCamera(camera)
      },
      geometry: {
        type: 'Point',
        coordinates: [
          Number(camera.lon),
          Number(camera.lat)
        ]
      }
    }));

  if (map?.getSource('cctv')) {
    map.getSource('cctv').setData({
      type: 'FeatureCollection',
      features
    });
    setMapLayerVisibility();
  }

  renderCameraList();
  updateSummary();
  syncVworldCameras();
}

function updateRainMap() {
  if (!map?.getSource('rainfall')) return;

  const features = (latestRainGrid || [])
    .filter(point =>
      Number.isFinite(Number(point.lon)) &&
      Number.isFinite(Number(point.lat)) &&
      point.rain_1h_mm !== null &&
      point.rain_1h_mm !== undefined
    )
    .map(point => ({
      type: 'Feature',
      properties: {
        id: point.id,
        name: point.name,
        rain_mm: Number(point.rain_1h_mm) || 0,
        temperature_c: Number(point.temperature_c) || 0,
        humidity_pct: Number(point.humidity_pct) || 0,
        wind_ms: Number(point.wind_ms) || 0,
        observed_at: point.observed_at || ''
      },
      geometry: {
        type: 'Point',
        coordinates: [
          Number(point.lon),
          Number(point.lat)
        ]
      }
    }));

  map.getSource('rainfall').setData({
    type: 'FeatureCollection',
    features
  });

  setMapLayerVisibility();
}

function fitMap() {
  map.fitBounds(
    [[128.86, 35.78], [129.58, 36.25]],
    {
      padding: {
        top: 35,
        bottom: 35,
        left: 35,
        right: 330
      },
      duration: 700
    }
  );
}

function nearestRainfall(lngLat) {
  let best = null;
  let bestDistance = Infinity;

  for (const point of latestRainGrid || []) {
    if (
      point.rain_1h_mm === null ||
      point.rain_1h_mm === undefined
    ) continue;

    const dx = (lngLat.lng - Number(point.lon)) * 89000;
    const dy = (lngLat.lat - Number(point.lat)) * 111000;
    const distance = Math.hypot(dx, dy);

    if (distance < bestDistance) {
      bestDistance = distance;
      best = point;
    }
  }
  return best;
}

function showElevation(lngLat) {
  let elevation = null;
  try {
    elevation = map.queryTerrainElevation(
      lngLat,
      { exaggerated: false }
    );
  } catch (error) { }

  const rain = nearestRainfall(lngLat);
  const rainText = rain
    ? ` · 인근 강수 ${Number(rain.rain_1h_mm || 0).toFixed(1)}mm`
    : '';

  $('elevation-readout').textContent =
    `경도 ${lngLat.lng.toFixed(5)} · ` +
    `위도 ${lngLat.lat.toFixed(5)} · ` +
    `고도 ${Number.isFinite(elevation)
      ? elevation.toFixed(1) + 'm'
      : 'DEM 로딩 중'}${rainText}`;
}

async function loadModelStatus() {
  try {
    const d = await fetch('/api/stage-model', { cache: 'no-store' }).then(r => r.json());
    const models = d.models || {};
    const vehicleOk = Boolean(models.vehicle?.loaded);
    const tireOk = Boolean(models.tire_level?.loaded);
    const bodyOk = Boolean(models.car_flood_cls?.loaded);
    const states = `차량 ${vehicleOk ? 'OK' : 'X'} · 타이어 ${tireOk ? 'OK' : 'X'} · 차체 ${bodyOk ? 'OK' : 'X'}`;
    const detail = d.loaded
      ? (d.degraded
          ? `침수 AI 보조모드 · ${states} · 차체 대체판정 사용`
          : `침수 AI 3모델 정상 · ${d.device}`)
      : `침수 AI 중단 · ${states}`;
    const status = $('model-status');
    setStatus('model-status', detail, d.loaded ? (d.degraded ? 'warn' : 'ok') : 'error');
    if (status) status.title = String(d.warning || d.error || detail);
  } catch (e) {
    setStatus('model-status', '차량·타이어·차체 모델 상태 오류', 'error');
  }
}

async function loadEnvironmentHistoryStatus() {
  try {
    const data = await fetchJsonWithRetry(
      '/api/environment-history/status',
      {},
      3,
      4500
    );
    if (!data.enabled) {
      setStatus('environment-history-status', '환경 DB 기록 꺼짐', 'warn');
      return;
    }
    const counts = data.last_counts || {};
    const stored = Number(data.stored || 0);
    const detail = data.running
      ? `환경 DB 저장 정상 · 누계 ${stored} · 최근 강수 ${Number(counts.rain || 0)} / 하수 ${Number(counts.sewer || 0)} / 하천 ${Number(counts.river || 0)}`
      : `환경 DB 저장 대기 · 누계 ${stored}`;
    setStatus('environment-history-status', detail, data.last_error ? 'warn' : (data.running ? 'ok' : 'warn'));
  } catch (error) {
    setStatus(
      'environment-history-status',
      `환경 DB 기록 재연결 중 · ${error.message}`,
      'warn'
    );
  }
}

async function loadBackgroundAiStatus() {
  try {
    const data = await fetchJsonWithRetry(
      '/api/cctv/background-status',
      {},
      3,
      4500
    );

    if (!data.enabled) {
      setStatus(
        'background-ai-status',
        '백그라운드 CCTV AI 꺼짐',
        'warn'
      );
      return;
    }

    if (!data.running) {
      setStatus(
        'background-ai-status',
        '백그라운드 CCTV AI 대기',
        'warn'
      );
      return;
    }

    const processed = Number(data.processed || 0);
    const total = Number(data.total || 0);
    const failed = Number(data.failed || 0);
    const stored = Number(data.stored || 0);
    const cycle = Number(data.cycle || 0);
    const local = data.continuous_local || {};
    const localStored = Number(local.stored || 0);
    const localStage = Number(local.last_stage);
    const localText = local.running
      ? ` · 테스트 지속저장 ${localStored}` +
        `${Number.isFinite(localStage) ? ` · Lev${localStage}` : ''}`
      : '';
    const state = data.scanning
      ? `순환 ${cycle} · ${processed}/${total}`
      : `순환 ${cycle} 완료 · ${Number(data.success || 0)}/${total}`;

    setStatus(
      'background-ai-status',
      `백그라운드 CCTV AI · ${state}` +
        `${stored ? ` · 저장 ${stored}` : ''}` +
        localText +
        `${failed ? ` · 원본 실패 ${failed}` : ''}`,
      failed ? 'warn' : 'ok'
    );
  } catch (error) {
    setStatus(
      'background-ai-status',
      `백그라운드 CCTV AI 재연결 중 · ${error.message}`,
      'warn'
    );
  }
}

async function loadPrivacyStatus() {
  try {
    const response = await fetch(
      '/api/privacy-model',
      { cache: 'no-store' }
    );
    const text = await response.text();
    let data = {};

    try {
      data = text ? JSON.parse(text) : {};
    } catch (parseError) {
      throw new Error(
        `HTTP ${response.status} · JSON 아님: ${text.slice(0, 160)}`
      );
    }

    const operational = Boolean(
      data.operational ?? (
        data.enabled && data.backend !== 'disabled'
      )
    );
    const backend = String(data.backend || 'disabled');

    if (operational) {
      const label = backend === 'yolo'
        ? '비식별화 YOLO · 얼굴/번호판'
        : '비식별화 OpenCV · 얼굴/번호판';
      setStatus(
        'privacy-status',
        label,
        backend === 'yolo' ? 'ok' : 'warn'
      );
    } else {
      setStatus(
        'privacy-status',
        data.enabled
          ? '비식별화 탐지기 준비 실패'
          : '비식별화 꺼짐',
        'error'
      );
    }

    const statusElement = $('privacy-status');
    if (statusElement) {
      statusElement.title = [
        data.warning,
        data.error,
        `backend=${backend}`,
        `fallback_ready=${Boolean(data.fallback_ready)}`,
        `device=${String(data.device || 'unknown')}`
      ].filter(Boolean).join('\n');
    }

    if (!response.ok) {
      console.warn(
        'privacy status HTTP error',
        response.status,
        data
      );
    } else if (data.error) {
      console.warn(
        'privacy status diagnostic',
        data
      );
    }
  } catch (error) {
    setStatus(
      'privacy-status',
      '비식별화 상태 조회 실패',
      'error'
    );
    const statusElement = $('privacy-status');
    if (statusElement) {
      statusElement.title = String(
        error?.message || error
      );
    }
    console.error(
      'privacy status request failed',
      error
    );
  }
}
async function health() { try { await fetch('/api/health'); setStatus('system-status', '시스템 정상', 'ok'); } catch (e) { setStatus('system-status', '연결 오류', 'error'); } }
async function loadRealCameras() { try { const r = await fetch('/api/cctv/pohang'); if (!r.ok) throw new Error((await r.json()).detail || 'CCTV API 오류'); realCameras = await r.json(); $('camera-count').textContent = realCameras.length; const hasTest = realCameras.some(camera => camera.local_test); setStatus('cctv-api-status', `포항 CCTV · ${realCameras.length}개소${hasTest ? ' · 테스트 영상 포함' : ''}`, 'ok'); await loadLatestStages(); updateMapData(); setTimeout(fitMap, 400); } catch (e) { setStatus('cctv-api-status', `포항 CCTV 오류 · ${e.message}`, 'error'); $('camera-list').innerHTML = '<p class="empty">CCTV 목록을 불러오지 못했습니다.</p>'; } }
async function loadLatestStages() {
  try {
    const response = await fetch('/api/cctv/stages', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const next = {};
    (data.items || []).forEach(item => {
      const key = item.camera_id
        ? String(item.camera_id)
        : `name:${String(item.camera_name || '')}`;
      const current = realStages[key];
      next[key] = current && stageTimestamp(current.detected_at) > stageTimestamp(item.detected_at)
        ? current
        : item;
    });
    realStages = next;
    realCameras.forEach(camera => {
      if (cameraMinimumStage(camera) > 0) {
        setCameraStage(
          camera,
          stageForCamera(camera),
          stageRecordForCamera(camera)?.confidence || 0,
          stageRecordForCamera(camera)?.detected_at || new Date().toISOString(),
          stageRecordForCamera(camera)?.positive_confirmed === true
        );
      }
    });
  } catch (error) {
    console.warn('CCTV 단계 갱신 지연', error);
  }
  updateMapData();
  scheduleSurface();
}
function renderCameraList() {
  const q = '';
  $('camera-list').innerHTML = realCameras.map((c, i) => {
    const lv = stageForCamera(c);
    const open = cctvWindows.has(String(c.id));
    const selected = String(c.id) === String(selectedCameraId);
    return `<button class="camera-row ${open ? 'open' : ''} ${selected ? 'selected' : ''}" data-camera-index="${i}" data-local-test="${Boolean(c.local_test)}" aria-label="${escapeHtml(c.name)} CCTV 열기 및 지도 이동"><i style="background:${COLORS[lv]}"></i><span><b>${escapeHtml(c.name)}</b><small>${escapeHtml(c.address || '포항 CCTV')}</small></span><em style="color:${COLORS[lv]}">Lev${lv} · ${DEPTHS[lv]}cm</em></button>`;
  }).join('') || '<p class="empty">CCTV 데이터가 없습니다.</p>';

  $('camera-list')
    .querySelectorAll('[data-camera-index]')
    .forEach(element => {
      element.addEventListener('click', () => {
        selectCameraByIndex(
          Number(element.dataset.cameraIndex),
          { openWindow: true }
        );
      });
    });
}
function escapeHtml(v) { return String(v ?? '').replace(/[&<>'"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[ch])); }


function cctvTransportUrl() {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${location.host}/ws/cctv`;
}

function scheduleCctvSocketReconnect(delay = 350) {
  if (!cctvWindows.size || cctvSocketReconnectTimer) return;
  cctvSocketReconnectTimer = window.setTimeout(() => {
    cctvSocketReconnectTimer = null;
    ensureCctvWebSocket();
  }, Math.max(120, Number(delay) || 350));
}

function syncCctvWebSocketSubscriptions() {
  if (!cctvSocket || cctvSocket.readyState !== WebSocket.OPEN) return;
  const cameras = Array.from(cctvWindows.values()).map(win => ({
    key: win.key,
    url: win.camera.stream_url
  }));
  const focused = Array.from(cctvWindows.values()).find(win => win.isFocused);
  try {
    cctvSocket.send(JSON.stringify({
      type: 'subscribe',
      cameras,
      focused_key: focused?.key || ''
    }));
    cctvSocketLastSyncAt = Date.now();
  } catch (_) {
    try { cctvSocket.close(); } catch (_) {}
  }
}

function drawCctvVectorDetections(context, canvas, packet) {
  const detections = Array.isArray(packet?.detections) ? packet.detections : [];
  if (!detections.length) return;
  const sourceWidth = Number(packet.detectionFrameWidth || canvas.width || 1);
  const sourceHeight = Number(packet.detectionFrameHeight || canvas.height || 1);
  if (!(sourceWidth > 0 && sourceHeight > 0)) return;
  const sx = canvas.width / sourceWidth;
  const sy = canvas.height / sourceHeight;
  const stageColors = ['#42c889', '#315cff', '#7136d9', '#bd2caf', '#ff4298'];

  context.save();
  context.textBaseline = 'top';
  detections.forEach(det => {
    const b = Array.isArray(det?.bbox) ? det.bbox : [];
    if (b.length !== 4) return;
    const x1 = Math.max(0, Math.min(canvas.width - 1, Number(b[0]) * sx));
    const y1 = Math.max(0, Math.min(canvas.height - 1, Number(b[1]) * sy));
    const x2 = Math.max(x1 + 1, Math.min(canvas.width, Number(b[2]) * sx));
    const y2 = Math.max(y1 + 1, Math.min(canvas.height, Number(b[3]) * sy));
    const stageValid = Boolean(det.stage_valid) && Number.isFinite(Number(det.stage));
    const stage = stageValid ? Math.max(0, Math.min(4, Number(det.stage))) : null;
    const provisional = Boolean(det.provisional);
    const color = stageValid ? stageColors[stage] : (provisional ? '#ffd75a' : '#5fe7ff');
    const conf = Math.max(0, Math.min(1, Number(det.conf || 0)));
    const track = det.track_id == null ? '' : ` #${det.track_id}`;
    const source = det.stage_source === 'tire' ? ' TIRE' : (det.stage_source === 'car_body' ? ' BODY' : '');
    const label = stageValid
      ? `VEHICLE${track} Lev${stage} ${Math.round(conf * 100)}%${source}`
      : `VEHICLE${track} ${provisional ? 'CHECK' : 'DET'} ${Math.round(conf * 100)}%`;

    context.strokeStyle = color;
    context.lineWidth = Math.max(2, Math.round(canvas.width / 480));
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const fontSize = Math.max(11, Math.min(16, Math.round(canvas.width / 48)));
    context.font = `600 ${fontSize}px Arial, sans-serif`;
    const metrics = context.measureText(label);
    const pad = 4;
    const textHeight = fontSize + pad * 2;
    const labelY = Math.max(0, y1 - textHeight);
    context.fillStyle = 'rgba(3,10,17,.84)';
    context.fillRect(x1, labelY, Math.min(canvas.width - x1, metrics.width + pad * 2), textHeight);
    context.fillStyle = color;
    context.fillText(label, x1 + pad, labelY + pad);
  });
  context.restore();
}

async function drawCctvTransportFrame(win) {
  if (!cctvWindows.has(win.key) || win.wsDecodeBusy) return;
  const packet = win.pendingWsFrame;
  if (!packet) return;
  win.pendingWsFrame = null;
  win.wsDecodeBusy = true;
  try {
    const blob = new Blob([packet.jpeg], { type: 'image/jpeg' });
    const bitmap = await createImageBitmap(blob);
    try {
      if (!cctvWindows.has(win.key)) return;
      // Always paint the JPEG that has already finished decoding.  V8.6.4
      // discarded it whenever a newer packet arrived during createImageBitmap().
      // Under sustained traffic that could repeat forever: decode -> newer packet
      // -> discard -> decode -> newer packet -> discard, which looked like CCTV
      // buffering even though frames were arriving.  pendingWsFrame remains a
      // latest-only slot, so after this paint the finally block immediately
      // decodes the newest waiting frame without building a backlog.
      const canvas = win.el.querySelector('.cctv-video-canvas');
      if (!canvas) return;
      if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
      if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
      const context = canvas.getContext('2d', { alpha: false });
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      // V8.6.4: vehicle geometry has exactly one renderer: the server JPEG.
      // V8.5.35's browser-only vector path could receive valid detections yet
      // still show no rectangles. Metadata remains in the packet for diagnostics,
      // but the browser must not draw a second copy of the server boxes.
      canvas.classList.remove('hidden');
      win.lastWsFrameAt = Date.now();
      if (packet.ready) {
        win.hasRealFrame = true;
        win.lastRealFrameAt = Date.now();
        if (packet.mode === 'annotated') win.lastAiFrameAt = Date.now();
      }
      const loading = win.el.querySelector('.cctv-loading');
      if (win.hasRealFrame || packet.ready) loading?.classList.add('hidden');
      else if (loading) {
        loading.textContent = 'CCTV 첫 화면 연결 중';
        loading.classList.remove('hidden');
      }
    } finally {
      bitmap.close?.();
    }
  } catch (_) {
    // Keep the last good canvas frame. A single damaged JPEG must never blank
    // a long-running CCTV window.
  } finally {
    win.wsDecodeBusy = false;
    if (win.pendingWsFrame) queueMicrotask(() => drawCctvTransportFrame(win));
  }
}

function handleCctvBinaryPacket(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 6) return;
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, false);
  if (headerLength <= 1 || 4 + headerLength >= buffer.byteLength) return;
  let header;
  try {
    header = JSON.parse(cctvTransportDecoder.decode(
      new Uint8Array(buffer, 4, headerLength)
    ));
  } catch (_) {
    return;
  }
  const win = cctvWindows.get(String(header.key || ''));
  if (!win) return;
  // Once real video has been displayed, reconnect/status cards are never
  // allowed to replace it. The canvas simply holds the last good frame.
  if (header.mode === 'status' && win.hasRealFrame) return;
  win.pendingWsFrame = {
    mode: String(header.mode || 'status'),
    ready: Boolean(header.ready),
    seq: Number(header.seq || 0),
    receivedAt: Date.now(),
    detections: Array.isArray(header.detections) ? header.detections : [],
    detectionFrameWidth: Number(header.detection_frame_width || 0),
    detectionFrameHeight: Number(header.detection_frame_height || 0),
    detectorMs: Number(header.detector_ms || 0),
    stage: header.stage,
    jpeg: new Uint8Array(buffer, 4 + headerLength)
  };
  drawCctvTransportFrame(win);
}

function ensureCctvWebSocket() {
  if (!cctvWindows.size) return;
  if (cctvSocket && (
    cctvSocket.readyState === WebSocket.OPEN ||
    cctvSocket.readyState === WebSocket.CONNECTING
  )) return;

  const generation = ++cctvSocketGeneration;
  const socket = new WebSocket(cctvTransportUrl());
  socket.binaryType = 'arraybuffer';
  cctvSocket = socket;

  socket.onopen = () => {
    if (generation !== cctvSocketGeneration) return;
    cctvSocketLastMessageAt = Date.now();
    syncCctvWebSocketSubscriptions();
  };
  socket.onmessage = event => {
    if (generation !== cctvSocketGeneration) return;
    cctvSocketLastMessageAt = Date.now();
    if (typeof event.data === 'string') return; // heartbeat
    handleCctvBinaryPacket(event.data);
  };
  socket.onerror = () => {
    try { socket.close(); } catch (_) {}
  };
  socket.onclose = () => {
    if (generation !== cctvSocketGeneration) return;
    if (cctvSocket === socket) cctvSocket = null;
    scheduleCctvSocketReconnect(300);
  };
}

function restartCctvWebSocket() {
  if (!cctvWindows.size) return;
  const socket = cctvSocket;
  cctvSocket = null;
  cctvSocketGeneration += 1;
  try { socket?.close(); } catch (_) {}
  scheduleCctvSocketReconnect(120);
}

// One transport watchdog for every CCTV window. It does not issue HTTP status
// requests, so opening many cameras cannot exhaust the browser connection pool.
window.setInterval(() => {
  if (!cctvWindows.size) return;
  const now = Date.now();
  if (!cctvSocket || cctvSocket.readyState > WebSocket.OPEN ||
      (cctvSocket.readyState === WebSocket.OPEN && now - cctvSocketLastMessageAt > 5500)) {
    restartCctvWebSocket();
    return;
  }
  if (cctvSocket.readyState === WebSocket.OPEN && now - cctvSocketLastSyncAt > 2500) {
    syncCctvWebSocketSubscriptions();
  }
}, 1000);

document.addEventListener('visibilitychange', () => {
  if (document.hidden || !cctvWindows.size) return;
  // Chromium may suspend timers/network work for a background tab. On return,
  // force the single transport to resubscribe (or reconnect if its heartbeat
  // went stale) while the canvases keep their last good image.
  if (!cctvSocket || cctvSocket.readyState !== WebSocket.OPEN ||
      Date.now() - cctvSocketLastMessageAt > 4500) {
    restartCctvWebSocket();
  } else {
    syncCctvWebSocketSubscriptions();
  }
});

function focusWindow(win) {
  windowZ += 1;
  document.querySelectorAll('.cctv-window').forEach(x => x.classList.remove('focused'));
  cctvWindows.forEach(item => { item.isFocused = item.key === win.key; });
  win.el.classList.add('focused');
  win.el.style.zIndex = String(windowZ);
  ensureCctvWebSocket();
  syncCctvWebSocketSubscriptions();
}

function cctvImageReady(img) {
  return Boolean(img && img.naturalWidth > 0 && img.naturalHeight > 0);
}

function preloadCctvObjectUrl(objectUrl, timeoutMs = 900) {
  // Decode off-screen first. Replacing the visible <img> before a Blob is
  // decoded can momentarily blank the CCTV and, on long-running Chromium
  // sessions, occasionally leaves the visible element stuck. The old frame
  // stays on screen until this promise succeeds.
  return new Promise((resolve, reject) => {
    const probe = new Image();
    let done = false;
    const finish = (error = null) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      probe.onload = null;
      probe.onerror = null;
      if (error) reject(error);
      else resolve();
    };
    const timer = window.setTimeout(
      () => finish(new Error('CCTV JPEG decode timeout')),
      Math.max(300, Number(timeoutMs) || 900)
    );
    probe.onload = () => finish();
    probe.onerror = () => finish(new Error('CCTV JPEG decode failed'));
    probe.src = objectUrl;
  });
}

function stopRawSnapshotLoop(win) {
  win.rawSnapshotActive = false;
  win.rawStreamActive = false;
  win.rawSnapshotLoading = false;
  if (win.rawSnapshotTimer) {
    clearTimeout(win.rawSnapshotTimer);
    win.rawSnapshotTimer = null;
  }
}

function scheduleRawSnapshot(win, delay = 0) {
  if (!cctvWindows.has(win.key) || !win.rawSnapshotActive) return;
  if (win.rawSnapshotTimer) clearTimeout(win.rawSnapshotTimer);
  win.rawSnapshotTimer = window.setTimeout(() => {
    win.rawSnapshotTimer = null;
    requestRawSnapshot(win);
  }, Math.max(0, Number(delay) || 0));
}

async function requestRawSnapshot(win) {
  if (!cctvWindows.has(win.key) || !win.rawSnapshotActive) return;
  if (document.hidden) {
    scheduleRawSnapshot(win, 900);
    return;
  }
  if (win.rawSnapshotLoading) return;
  const raw = win.el.querySelector('.cctv-video-raw');
  if (!raw) return;
  win.rawSnapshotLoading = true;
  const controller = new AbortController();
  win.rawFetchController = controller;
  const timeout = window.setTimeout(() => controller.abort(), 1400);
  try {
    const response = await fetch(rawSnapshotUrl(win.camera), {
      cache: 'no-store', signal: controller.signal
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const sequence = response.headers.get('X-CCTV-Frame-Seq') || '';
    const ready = response.headers.get('X-CCTV-Frame-Ready') === '1';
    const blob = await response.blob();
    if (!cctvWindows.has(win.key) || !win.rawSnapshotActive) return;

    if ((!ready || sequence === win.lastRawServerSeq) && cctvImageReady(raw)) {
      win.rawSnapshotLoading = false;
      scheduleRawSnapshot(win, 160);
      return;
    }
    if (sequence) win.lastRawServerSeq = sequence;
    const objectUrl = URL.createObjectURL(blob);
    try {
      await preloadCctvObjectUrl(objectUrl, 850);
    } catch (decodeError) {
      URL.revokeObjectURL(objectUrl);
      throw decodeError;
    }
    if (!cctvWindows.has(win.key) || !win.rawSnapshotActive) {
      URL.revokeObjectURL(objectUrl);
      return;
    }
    const oldUrl = win.rawObjectUrl;
    win.rawObjectUrl = objectUrl;
    win.rawPendingRevoke = oldUrl || null;
    raw.src = objectUrl;
    win.rawSnapshotLoading = false;
    scheduleRawSnapshot(win, 150);
  } catch (error) {
    win.rawSnapshotLoading = false;
    if (cctvWindows.has(win.key) && win.rawSnapshotActive) {
      scheduleRawSnapshot(win, error?.name === 'AbortError' ? 180 : 320);
    }
  } finally {
    window.clearTimeout(timeout);
    if (win.rawFetchController === controller) win.rawFetchController = null;
  }
}

function stopAnnotatedSnapshotLoop(win) {
  win.annotatedStreamActive = false;
  win.aiSnapshotLoading = false;
  if (win.aiSnapshotTimer) {
    clearTimeout(win.aiSnapshotTimer);
    win.aiSnapshotTimer = null;
  }
}

function scheduleAnnotatedSnapshot(win, delay = 0) {
  if (!cctvWindows.has(win.key) || !win.annotatedStreamActive) return;
  if (win.aiSnapshotTimer) clearTimeout(win.aiSnapshotTimer);
  win.aiSnapshotTimer = window.setTimeout(() => {
    win.aiSnapshotTimer = null;
    requestAnnotatedSnapshot(win);
  }, Math.max(0, Number(delay) || 0));
}

async function requestAnnotatedSnapshot(win) {
  if (!cctvWindows.has(win.key) || !win.annotatedStreamActive) return;
  if (document.hidden) {
    scheduleAnnotatedSnapshot(win, 900);
    return;
  }
  if (win.aiSnapshotLoading) return;
  const ai = win.el.querySelector('.cctv-video-ai');
  if (!ai) return;
  win.aiSnapshotLoading = true;
  const controller = new AbortController();
  win.aiFetchController = controller;
  const timeout = window.setTimeout(() => controller.abort(), 1500);
  try {
    const response = await fetch(annotatedSnapshotUrl(win.camera, win.isFocused), {
      cache: 'no-store', signal: controller.signal
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const sequence = response.headers.get('X-CCTV-Frame-Seq') || '';
    const ready = response.headers.get('X-CCTV-Frame-Ready') === '1';
    const blob = await response.blob();
    if (!cctvWindows.has(win.key) || !win.annotatedStreamActive) return;

    if (sequence && sequence === win.lastAiServerSeq && cctvImageReady(ai)) {
      win.aiSnapshotLoading = false;
      if (Date.now() - Number(win.lastAiServerFrameAt || 0) > 2600) {
        ensureRawCctvStream(win);
      }
      scheduleAnnotatedSnapshot(win, win.isFocused ? 85 : 210);
      return;
    }
    if (!ready) {
      // Never promote a CONNECTING/RETRY card to the AI layer. The raw layer
      // remains visible until the first genuinely rendered annotated frame.
      win.aiSnapshotLoading = false;
      ensureRawCctvStream(win);
      scheduleAnnotatedSnapshot(win, 180);
      return;
    }

    if (sequence) win.lastAiServerSeq = sequence;
    if (ready) win.lastAiServerFrameAt = Date.now();
    const objectUrl = URL.createObjectURL(blob);
    try {
      await preloadCctvObjectUrl(objectUrl, 850);
    } catch (decodeError) {
      URL.revokeObjectURL(objectUrl);
      throw decodeError;
    }
    if (!cctvWindows.has(win.key) || !win.annotatedStreamActive) {
      URL.revokeObjectURL(objectUrl);
      return;
    }
    const oldUrl = win.aiObjectUrl;
    win.aiObjectUrl = objectUrl;
    win.aiPendingRevoke = oldUrl || null;
    ai.src = objectUrl;
    win.aiSnapshotLoading = false;
    win.lastAiFrameAt = Date.now();
    refreshCctvImageVisibility(win);
    scheduleAnnotatedSnapshot(win, win.isFocused ? 90 : 220);
  } catch (error) {
    win.aiSnapshotLoading = false;
    if (cctvWindows.has(win.key) && win.annotatedStreamActive) {
      ensureRawCctvStream(win);
      scheduleAnnotatedSnapshot(win, error?.name === 'AbortError' ? 180 : 320);
    }
  } finally {
    window.clearTimeout(timeout);
    if (win.aiFetchController === controller) win.aiFetchController = null;
  }
}

function refreshCctvImageVisibility(win) {
  if (!cctvWindows.has(win.key)) return;
  const canvas = win.el.querySelector('.cctv-video-canvas');
  const loading = win.el.querySelector('.cctv-loading');
  if (win.hasRealFrame) {
    canvas?.classList.remove('hidden');
    loading?.classList.add('hidden');
  }
}

function ensureRawCctvStream(win) {
  if (!cctvWindows.has(win.key)) return;
  win.streamActive = true;
  win.rawSnapshotActive = false;
  win.rawStreamActive = true;
  ensureCctvWebSocket();
  syncCctvWebSocketSubscriptions();
}

function ensureAnnotatedCctvStream(win) {
  if (!cctvWindows.has(win.key)) return;
  win.streamActive = true;
  win.streamMode = 'annotated';
  win.annotatedStreamActive = true;
  ensureCctvWebSocket();
  syncCctvWebSocketSubscriptions();
}

function stopAnnotatedCctvStream(win) {
  if (!win) return;
  win.annotatedStreamActive = false;
  syncCctvWebSocketSubscriptions();
}

function restartAnnotatedCctvStream(win, reason = 'watchdog') {
  if (!cctvWindows.has(win.key)) return;
  win.lastAnnotatedRestartAt = Date.now();
  if (!cctvSocket || cctvSocket.readyState !== WebSocket.OPEN ||
      Date.now() - cctvSocketLastMessageAt > 4500) {
    restartCctvWebSocket();
  } else {
    syncCctvWebSocketSubscriptions();
  }
}

function setCctvStreamMode(win, mode) {
  if (!cctvWindows.has(win.key)) return;
  win.streamMode = 'annotated';
  ensureAnnotatedCctvStream(win);
}

function selectAnnotatedWindow(activeWin) {
  cctvWindows.forEach(win => {
    win.isFocused = Boolean(activeWin && win.key === activeWin.key);
    ensureAnnotatedCctvStream(win);
  });
  syncCctvWebSocketSubscriptions();
}
function ensureCctvWindowLayerOnBody() {
  const layer = $('cctv-window-layer');
  if (layer && layer.parentElement !== document.body) document.body.appendChild(layer);
  return layer;
}
function makeDraggable(win) {
  const head = win.el.querySelector('.cctv-head');
  let drag = null;
  head.addEventListener('pointerdown', e => {
    if (e.target.closest('button') || win.el.classList.contains('maximized')) return;
    win.el.classList.remove('auto-arranged', 'compact', 'micro');
    focusWindow(win);
    const rect = win.el.getBoundingClientRect();
    drag = { dx: e.clientX - rect.left, dy: e.clientY - rect.top };
    head.setPointerCapture(e.pointerId);
  });
  head.addEventListener('pointermove', e => {
    if (!drag) return;
    const maxX = Math.max(0, window.innerWidth - win.el.offsetWidth);
    const maxY = Math.max(0, window.innerHeight - win.el.offsetHeight);
    win.el.style.left = Math.max(0, Math.min(maxX, e.clientX - drag.dx)) + 'px';
    win.el.style.top = Math.max(0, Math.min(maxY, e.clientY - drag.dy)) + 'px';
  });
  const stop = () => { drag = null; };
  head.addEventListener('pointerup', stop);
  head.addEventListener('pointercancel', stop);
}

function toLocalInputValue(date) {
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function historyPresetLabel(mode) {
  return ({
    live: '실시간 30분',
    h1: '최근 1시간',
    h6: '최근 6시간',
    h24: '최근 24시간',
    d7: '최근 7일',
    d30: '최근 30일',
    custom: '사용자 지정'
  })[mode] || '이전 기록';
}

function setWindowHistoryInputs(win, start, end) {
  const startInput = win.el.querySelector('[data-history-start]');
  const endInput = win.el.querySelector('[data-history-end]');
  if (startInput) startInput.value = toLocalInputValue(start);
  if (endInput) endInput.value = toLocalInputValue(end);
}

function setHistoryButtons(win) {
  win.el.querySelectorAll('[data-history-preset]').forEach(button => {
    button.classList.toggle(
      'active',
      button.dataset.historyPreset === win.historyMode
    );
  });
}

function applyHistoryPreset(win, mode) {
  if (!cctvWindows.has(win.key)) return;

  const hoursByMode = {
    live: .5,
    h1: 1,
    h6: 6,
    h24: 24,
    d7: 24 * 7,
    d30: 24 * 30
  };
  const hours = hoursByMode[mode];
  if (!hours) return;

  const end = new Date();
  const start = new Date(end.getTime() - hours * 60 * 60 * 1000);

  win.historyMode = mode;
  win.historyHours = hours;
  win.historyStart = start;
  win.historyEnd = end;

  setWindowHistoryInputs(win, start, end);
  setHistoryButtons(win);
  loadWindowHistory(win);
}

function openCctvWindow(idx) {
  const c = realCameras[idx];
  if (!c) return;

  const key = String(c.id);
  if (cctvWindows.has(key)) {
    focusWindow(cctvWindows.get(key));
    return;
  }

  const n = cctvWindows.size;
  const el = document.createElement('section');
  el.className = 'cctv-window';
  el.style.left = (18 + (n % 12) * 34) + 'px';
  el.style.top = (60 + (n % 10) * 36) + 'px';

  el.innerHTML = `
    <div class="cctv-head">
      <div>
        <p>LIVE · AI FLOOD · PRIVACY · HISTORY V8.6.4</p>
        <h3>${escapeHtml(c.name)}</h3>
      </div>
      <div class="window-actions">
        <button data-act="min" title="최소화">—</button>
        <button data-act="max" title="최대화">□</button>
        <button data-act="close" title="닫기">×</button>
      </div>
    </div>

    <div class="cctv-body">
      <img class="cctv-video cctv-video-raw hidden" decoding="async" draggable="false" alt="${escapeHtml(c.name)} CCTV">
      <img class="cctv-video cctv-video-ai hidden" decoding="async" draggable="false" alt="${escapeHtml(c.name)} AI CCTV">
      <canvas class="cctv-video cctv-video-canvas hidden" aria-label="${escapeHtml(c.name)} 실시간 CCTV"></canvas>
      <div class="cctv-loading">CCTV 첫 화면 수신 중</div>
      <div class="privacy-badge">얼굴·번호판 모자이크</div>
      <div class="cctv-badge">
        <b>분석 중</b>
        <span>Lev0</span>
        <small>${escapeHtml(c.address || '')}</small>
      </div>
    </div>

    <div class="cctv-history">
      <div class="cctv-history-head">
        <strong>침수 레벨 기록</strong>
        <span class="cctv-history-status">기록 불러오는 중</span>
      </div>

      <div class="cctv-history-presets" aria-label="침수 기록 조회 기간">
        <button type="button" data-history-preset="live" class="active">실시간</button>
        <button type="button" data-history-preset="h1">1시간</button>
        <button type="button" data-history-preset="h6">6시간</button>
        <button type="button" data-history-preset="h24">24시간</button>
        <button type="button" data-history-preset="d7">7일</button>
        <button type="button" data-history-preset="d30">30일</button>
      </div>

      <div class="cctv-history-range">
        <label>
          <span>시작</span>
          <input type="datetime-local" data-history-start>
        </label>
        <label>
          <span>종료</span>
          <input type="datetime-local" data-history-end>
        </label>
        <button type="button" data-history-search>조회</button>
      </div>

      <div class="cctv-history-env" data-history-env-summary>
        <span>강수 —</span><span>하수 —</span><span>하천 —</span>
      </div>

      <div class="cctv-chart-wrap">
        <canvas></canvas>
        <div class="cctv-chart-empty">선택한 기간의 기록이 없습니다.</div>
      </div>

      <div class="cctv-history-table-wrap">
        <table class="cctv-history-table">
          <thead><tr><th>시간</th><th>침수</th><th>강수</th><th>하수</th><th>하천</th></tr></thead>
          <tbody data-history-table-body><tr><td colspan="5">기록 불러오는 중</td></tr></tbody>
        </table>
      </div>
    </div>`;

  const windowLayer = ensureCctvWindowLayerOnBody();
  windowLayer.appendChild(el);

  const end = new Date();
  const start = new Date(end.getTime() - 30 * 60 * 1000);

  const initialStage = stageForCamera(c);
  const initialConfidence = Number(stageRecordForCamera(c)?.confidence || 0);
  const win = {
    key,
    idx,
    camera: c,
    el,
    timer: null,
    historyTimer: null,
    streamStatusTimer: null,
    chart: null,
    previousRect: null,
    storedHistoryPoints: [],
    combinedHistoryPoints: [],
    environmentHistory: {},
    // Two boundary points render a visible horizontal graph immediately.
    // A single current point looked like an empty chart until history arrived.
    liveHistoryPoints: [
      {
        time: start.toISOString(),
        level: initialStage,
        confidence: initialConfidence,
        instant_seed: true
      },
      {
        time: end.toISOString(),
        level: initialStage,
        confidence: initialConfidence,
        instant_seed: true
      }
    ],
    historyRequestSequence: 0,
    historyAbortController: null,
    historyMode: 'live',
    historyHours: .5,
    historyStart: start,
    historyEnd: end,
    streamActive: false,
    streamMode: 'annotated',
    rawStreamActive: false,
    rawSnapshotActive: false,
    rawSnapshotLoading: false,
    rawSnapshotTimer: null,
    rawFetchController: null,
    rawObjectUrl: null,
    rawPendingRevoke: null,
    lastRawServerSeq: '',
    annotatedStreamActive: false,
    aiSnapshotLoading: false,
    aiSnapshotTimer: null,
    aiFetchController: null,
    aiObjectUrl: null,
    aiPendingRevoke: null,
    lastAiServerSeq: '',
    lastAiServerFrameAt: 0,
    lastAiFrameAt: 0,
    pendingWsFrame: null,
    wsDecodeBusy: false,
    lastWsFrameAt: 0,
    lastRealFrameAt: 0,
    hasRealFrame: false,
    isFocused: false,
    openedAt: Date.now(),
    lastAnnotatedRestartAt: 0,
    frameWatchTimer: null
  };

  cctvWindows.set(key, win);
  makeDraggable(win);
  setWindowHistoryInputs(win, start, end);
  setHistoryButtons(win);

  const rawImg = el.querySelector('.cctv-video-raw');
  const aiImg = el.querySelector('.cctv-video-ai');
  const loading = el.querySelector('.cctv-loading');

  rawImg.onload = () => {
    win.rawSnapshotLoading = false;
    if (win.rawPendingRevoke) {
      URL.revokeObjectURL(win.rawPendingRevoke);
      win.rawPendingRevoke = null;
    }
    refreshCctvImageVisibility(win);
    if (!cctvWindows.has(key) || !win.rawSnapshotActive) return;
    if (Number(win.lastAiFrameAt || 0) > 0) {
      stopRawSnapshotLoop(win);
      return;
    }
    scheduleRawSnapshot(win, 150);
  };

  aiImg.onload = () => {
    win.aiSnapshotLoading = false;
    win.lastAiFrameAt = Date.now();
    if (win.aiPendingRevoke) {
      URL.revokeObjectURL(win.aiPendingRevoke);
      win.aiPendingRevoke = null;
    }
    refreshCctvImageVisibility(win);
    if (!cctvWindows.has(key) || !win.annotatedStreamActive) return;
    // Regular JPEG snapshots avoid the long-lived MJPEG connection that
    // disappeared after extended use. The focused window is a little faster,
    // while every tiled window still receives continuous AI boxes.
    scheduleAnnotatedSnapshot(win, win.isFocused ? 90 : 220);
  };

  rawImg.onerror = () => {
    win.rawSnapshotLoading = false;
    if (!win.streamActive || !win.rawSnapshotActive) return;
    if (!cctvImageReady(aiImg)) {
      loading.textContent = 'CCTV 재연결 중';
      loading.classList.remove('hidden');
    }
    scheduleRawSnapshot(win, 320);
  };

  aiImg.onerror = () => {
    win.aiSnapshotLoading = false;
    aiImg.classList.add('hidden');
    if (cctvWindows.has(key) && win.annotatedStreamActive) {
      ensureRawCctvStream(win);
      scheduleAnnotatedSnapshot(win, 300);
    }
  };

  win.frameWatchTimer = setInterval(
    () => refreshCctvImageVisibility(win),
    250
  );

  focusWindow(win);

  setTimeout(() => refreshWindowStreamStatus(win), 1200);
  win.streamStatusTimer = setInterval(
    () => refreshWindowStreamStatus(win),
    5000
  );

  el.addEventListener('pointerdown', () => focusWindow(win));
  el.querySelector('[data-act="close"]')
    .addEventListener('click', () => closeWindow(key));
  el.querySelector('[data-act="min"]')
    .addEventListener('click', () => { el.classList.remove('auto-arranged', 'compact', 'micro'); el.classList.toggle('minimized'); });
  el.querySelector('[data-act="max"]')
    .addEventListener('click', () => { el.classList.remove('auto-arranged', 'compact', 'micro'); el.classList.toggle('maximized'); });

  el.querySelectorAll('[data-history-preset]').forEach(button => {
    button.addEventListener('click', () => {
      applyHistoryPreset(win, button.dataset.historyPreset);
    });
  });

  el.querySelector('[data-history-search]').addEventListener('click', () => {
    const startValue = el.querySelector('[data-history-start]')?.value;
    const endValue = el.querySelector('[data-history-end]')?.value;

    if (!startValue || !endValue) {
      alert('조회할 시작 시간과 종료 시간을 모두 선택하세요.');
      return;
    }

    const customStart = new Date(startValue);
    const customEnd = new Date(endValue);

    // datetime-local has minute precision. Include the complete end minute.
    customStart.setSeconds(0, 0);
    customEnd.setSeconds(59, 999);

    if (
      !Number.isFinite(customStart.getTime()) ||
      !Number.isFinite(customEnd.getTime())
    ) {
      alert('날짜 형식이 올바르지 않습니다.');
      return;
    }

    if (customEnd <= customStart) {
      alert('종료 시간은 시작 시간보다 뒤여야 합니다.');
      return;
    }

    if (customEnd - customStart > 365 * 24 * 60 * 60 * 1000) {
      alert('최대 조회 범위는 1년입니다.');
      return;
    }

    win.historyMode = 'custom';
    win.historyStart = customStart;
    win.historyEnd = customEnd;
    win.historyHours = (customEnd - customStart) / (60 * 60 * 1000);

    setHistoryButtons(win);
    loadWindowHistory(win);
  });

  drawWindowHistory(win);
  loadWindowHistory(win);
  refreshWindowStage(win);

  win.timer = setInterval(() => refreshWindowStage(win), 10000);
  win.historyTimer = setInterval(() => {
    if (win.historyMode === 'live') {
      applyHistoryPreset(win, 'live');
    }
  }, 60000);

  new ResizeObserver(() => win.chart?.resize()).observe(el);
  renderCameraList();
  updateSummary();
}


function arrangeCctvWindows() {
  const layer = $('cctv-window-layer');
  const windows = Array.from(cctvWindows.values());
  if (!layer || !windows.length) {
    alert('정렬할 열린 CCTV가 없습니다.');
    return;
  }

  const gap = 8;
  const width = Math.max(320, layer.clientWidth);
  const height = Math.max(260, layer.clientHeight);
  const count = windows.length;
  const columns = Math.max(
    1,
    Math.min(
      count,
      Math.ceil(Math.sqrt(count * (width / height) * 0.78))
    )
  );
  const rows = Math.ceil(count / columns);
  // Arrangement must never enlarge a CCTV window. It only shrinks windows
  // when several of them need to fit in the viewport.
  const naturalWidth = Math.min(590, Math.max(320, width - gap * 2));
  const naturalHeight = Math.min(620, Math.max(260, height - gap * 2));
  const cellWidth = Math.max(48, Math.min(
    naturalWidth,
    (width - gap * (columns + 1)) / columns
  ));
  const cellHeight = Math.max(52, Math.min(
    naturalHeight,
    (height - gap * (rows + 1)) / rows
  ));
  const usedWidth = columns * cellWidth + (columns - 1) * gap;
  const usedHeight = rows * cellHeight + (rows - 1) * gap;
  const originX = Math.max(gap, (width - usedWidth) / 2);
  const originY = Math.max(gap, (height - usedHeight) / 2);
  const compact = cellHeight < 430 || cellWidth < 430;
  const micro = cellHeight < 145 || cellWidth < 190;

  windows.forEach((win, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const el = win.el;
    el.classList.remove('maximized', 'minimized');
    el.classList.add('auto-arranged');
    el.classList.toggle('compact', compact);
    el.classList.toggle('micro', micro);
    el.style.left = `${originX + column * (cellWidth + gap)}px`;
    el.style.top = `${originY + row * (cellHeight + gap)}px`;
    el.style.width = `${cellWidth}px`;
    el.style.height = `${cellHeight}px`;
    el.style.zIndex = String(40 + index);
    win.chart?.resize();
  });

  const button = $('arrange-cctv');
  button?.classList.add('active');
  window.setTimeout(() => button?.classList.remove('active'), 700);
}

async function fetchElevationAt(lngLat) {
  const params = new URLSearchParams({
    lat: String(Number(lngLat.lat)),
    lon: String(Number(lngLat.lng))
  });
  const response = await fetch('/api/dem/elevation?' + params, {
    cache: 'no-store'
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || 'DEM 고도 조회 실패');
  }
  return body;
}

async function showElevationPopup(lngLat) {
  try {
    const data = await fetchElevationAt(lngLat);
    $('elevation-readout').textContent =
      `선택 지점 · 경도 ${Number(data.lon).toFixed(5)} · ` +
      `위도 ${Number(data.lat).toFixed(5)} · ` +
      `DEM 고도 ${Number(data.elevation_m).toFixed(1)}m`;
    if (elevationMarker) elevationMarker.remove();
    elevationMarker = new maplibregl.Marker({ color: '#ffffff', scale: 0.82 })
      .setLngLat([Number(data.lon), Number(data.lat)])
      .addTo(map);
    new maplibregl.Popup()
      .setLngLat([Number(data.lon), Number(data.lat)])
      .setHTML(
        `<b>선택 지점 고도</b><br>` +
        `위도 ${Number(data.lat).toFixed(6)}<br>` +
        `경도 ${Number(data.lon).toFixed(6)}<br>` +
        `DEM 고도 ${Number(data.elevation_m).toFixed(1)}m`
      )
      .addTo(map);
  } catch (error) {
    setStatus('vworld-status', `고도 조회 오류 · ${error.message}`, 'error');
  }
}

async function refreshWindowStreamStatus(win) {
  if (!cctvWindows.has(win.key)) return;
  const loading = win.el.querySelector('.cctv-loading');
  refreshCctvImageVisibility(win);
  ensureCctvWebSocket();

  if (win.hasRealFrame) {
    // Never cover a good frame just because the upstream stream is repairing.
    loading?.classList.add('hidden');
    if (Date.now() - Number(win.lastRealFrameAt || 0) > 4500) {
      syncCctvWebSocketSubscriptions();
    }
    return;
  }

  const socketOpen = cctvSocket && cctvSocket.readyState === WebSocket.OPEN;
  if (loading) {
    loading.textContent = socketOpen
      ? 'CCTV 영상 수신 대기 중'
      : 'CCTV 전송 채널 재연결 중';
    loading.classList.remove('hidden');
  }
  if (!socketOpen || Date.now() - cctvSocketLastMessageAt > 4500) {
    restartCctvWebSocket();
  } else {
    syncCctvWebSocketSubscriptions();
  }
}


function rawSnapshotUrl(c) {
  return '/api/cctv/frame-raw?url=' + encodeURIComponent(c.stream_url) +
    '&t=' + Date.now();
}

function annotatedSnapshotUrl(c, focused = false) {
  return '/api/cctv/frame-annotated?url=' + encodeURIComponent(c.stream_url) +
    '&focus=' + (focused ? '1' : '0') + '&t=' + Date.now();
}

function streamUrl(c, mode = 'raw') {
  const endpoint = mode === 'annotated'
    ? '/api/stream-annotated'
    : '/api/stream-raw';
  return endpoint + '?url=' + encodeURIComponent(c.stream_url) + '&t=' + Date.now();
}
function closeWindow(key) {
  const win = cctvWindows.get(key);
  if (!win) return;
  if (win.timer) clearInterval(win.timer);
  if (win.historyTimer) clearInterval(win.historyTimer);
  if (win.streamStatusTimer) clearInterval(win.streamStatusTimer);
  if (win.frameWatchTimer) clearInterval(win.frameWatchTimer);
  if (win.aiSnapshotTimer) clearTimeout(win.aiSnapshotTimer);
  if (win.rawSnapshotTimer) clearTimeout(win.rawSnapshotTimer);
  win.historyAbortController?.abort();
  win.rawFetchController?.abort();
  win.aiFetchController?.abort();
  if (win.rawObjectUrl) URL.revokeObjectURL(win.rawObjectUrl);
  if (win.rawPendingRevoke) URL.revokeObjectURL(win.rawPendingRevoke);
  if (win.aiObjectUrl) URL.revokeObjectURL(win.aiObjectUrl);
  if (win.aiPendingRevoke) URL.revokeObjectURL(win.aiPendingRevoke);
  win.streamActive = false;
  win.rawSnapshotActive = false;
  win.rawSnapshotLoading = false;
  win.rawStreamActive = false;
  win.annotatedStreamActive = false;
  win.aiSnapshotLoading = false;
  win.el.querySelectorAll('.cctv-video').forEach(img => img.removeAttribute('src'));
  if (win.chart) win.chart.destroy();
  win.el.remove();
  cctvWindows.delete(key);
  const next = Array.from(cctvWindows.values()).at(-1);
  if (next) {
    focusWindow(next);
  } else {
    if (cctvSocketReconnectTimer) {
      clearTimeout(cctvSocketReconnectTimer);
      cctvSocketReconnectTimer = null;
    }
    const socket = cctvSocket;
    cctvSocket = null;
    cctvSocketGeneration += 1;
    try { socket?.close(); } catch (_) {}
  }
  renderCameraList();
  updateSummary();
}
function historyDate(value) {
  const text = String(value || '');
  if (!text) return new Date();
  return new Date(/[zZ]$|[+-]\d\d:\d\d$/.test(text) ? text : text + 'Z');
}

function addWindowHistoryPoint(win, stage, confidence) {
  if (!cctvWindows.has(win.key)) return;

  const point = {
    time: new Date().toISOString(),
    level: Math.max(0, Math.min(4, Number(stage) || 0)),
    confidence: Number(confidence || 0),
    live: true
  };

  const previous = win.liveHistoryPoints[
    win.liveHistoryPoints.length - 1
  ];

  if (
    !previous ||
    historyDate(point.time) - historyDate(previous.time) >= 8000 ||
    previous.level !== point.level
  ) {
    win.liveHistoryPoints.push(point);
    if (win.liveHistoryPoints.length > 720) {
      win.liveHistoryPoints.shift();
    }
  }

  if (win.historyMode === 'live') {
    const end = new Date();
    win.historyEnd = end;
    win.historyStart = new Date(end.getTime() - 30 * 60 * 1000);
    setWindowHistoryInputs(win, win.historyStart, win.historyEnd);
    drawWindowHistory(win);
  }
}

function mergedWindowHistory(win) {
  const start = win.historyStart instanceof Date
    ? win.historyStart
    : new Date(Date.now() - 30 * 60 * 1000);
  const end = win.historyEnd instanceof Date
    ? win.historyEnd
    : new Date();

  const source = [
    ...(win.storedHistoryPoints || []),
    ...(win.historyMode === 'custom' ? [] : (win.liveHistoryPoints || []))
  ];

  return source
    .filter(point => {
      const time = historyDate(point.time).getTime();
      return time >= start.getTime() && time <= end.getTime();
    })
    .sort((a, b) => historyDate(a.time) - historyDate(b.time))
    .filter((point, index, array) =>
      index === 0 ||
      Math.abs(
        historyDate(point.time) -
        historyDate(array[index - 1].time)
      ) > 1000 ||
      Number(point.level) !== Number(array[index - 1].level)
    );
}

function combinedWindowHistory(win) {
  const start = win.historyStart instanceof Date
    ? win.historyStart
    : new Date(Date.now() - 30 * 60 * 1000);
  const end = win.historyEnd instanceof Date
    ? win.historyEnd
    : new Date();

  let points = Array.isArray(win.combinedHistoryPoints)
    ? win.combinedHistoryPoints.map(point => ({ ...point }))
    : [];

  // Backward-compatible fallback if an older server only returns stage points.
  if (!points.length && Array.isArray(win.storedHistoryPoints)) {
    points = win.storedHistoryPoints.map(point => ({ ...point }));
  }

  if (win.historyMode !== 'custom' && win.liveHistoryPoints?.length) {
    const latestEnv = points.length ? points[points.length - 1] : {};
    points.push(...win.liveHistoryPoints.map(point => ({
      ...point,
      rain_mm: latestEnv.rain_mm ?? null,
      sewer_level_m: latestEnv.sewer_level_m ?? null,
      river_level_m: latestEnv.river_level_m ?? null
    })));
  }

  const byTime = new Map();
  points
    .filter(point => {
      const time = historyDate(point.time).getTime();
      return time >= start.getTime() && time <= end.getTime();
    })
    .sort((a, b) => historyDate(a.time) - historyDate(b.time))
    .forEach(point => {
      const key = historyDate(point.time).getTime();
      const previous = byTime.get(key) || {};
      byTime.set(key, { ...previous, ...point });
    });
  return Array.from(byTime.values())
    .sort((a, b) => historyDate(a.time) - historyDate(b.time));
}

function formatHistoryValue(value, digits = 2, suffix = '') {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(digits)}${suffix}` : '—';
}

function drawWindowHistory(win) {
  if (!cctvWindows.has(win.key)) return;

  const stagePoints = mergedWindowHistory(win);
  const points = combinedWindowHistory(win);
  const canvas = win.el.querySelector('.cctv-chart-wrap canvas');
  const empty = win.el.querySelector('.cctv-chart-empty');
  const status = win.el.querySelector('.cctv-history-status');
  const envSummary = win.el.querySelector('[data-history-env-summary]');
  const tableBody = win.el.querySelector('[data-history-table-body]');

  if (empty) empty.classList.toggle('hidden', points.length > 0);

  if (status) {
    const backgroundCount = Number(win.historySourceCounts?.background || 0);
    status.textContent = points.length
      ? `${historyPresetLabel(win.historyMode)} · 침수 ${stagePoints.length}구간 · 통합 ${points.length}구간` +
        (backgroundCount ? ` · 백그라운드 ${backgroundCount}건` : '')
      : `${historyPresetLabel(win.historyMode)} · 기록 없음`;
  }

  const latest = points.length ? points[points.length - 1] : {};
  const rainMeta = win.environmentHistory?.rain || {};
  const sewerMeta = win.environmentHistory?.sewer || {};
  const riverMeta = win.environmentHistory?.river || {};
  if (envSummary) {
    const distance = meta => Number.isFinite(Number(meta?.distance_m))
      ? ` · ${Math.round(Number(meta.distance_m))}m`
      : '';
    envSummary.innerHTML = `
      <span title="${escapeHtml(rainMeta.sensor_name || '강수 관측')} ${distance(rainMeta)}">강수 <b>${formatHistoryValue(latest.rain_mm, 1, 'mm')}</b></span>
      <span title="${escapeHtml(sewerMeta.sensor_name || '하수 수위')} ${distance(sewerMeta)}">하수 <b>${formatHistoryValue(latest.sewer_level_m, 2, 'm')}</b></span>
      <span title="${escapeHtml(riverMeta.sensor_name || '하천 수위')} ${distance(riverMeta)}">하천 <b>${formatHistoryValue(latest.river_level_m, 2, 'm')}</b></span>`;
  }

  if (tableBody) {
    const rows = points.slice(-160).reverse();
    tableBody.innerHTML = rows.length
      ? rows.map(point => {
          const date = historyDate(point.time);
          const level = Number(point.level);
          const levelText = Number.isFinite(level) ? `Lev${Math.max(0, Math.min(4, level))}` : '—';
          return `<tr>
            <td>${escapeHtml(date.toLocaleString('ko-KR', {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit'}))}</td>
            <td>${levelText}</td>
            <td>${formatHistoryValue(point.rain_mm, 1, 'mm')}</td>
            <td>${formatHistoryValue(point.sewer_level_m, 2, 'm')}</td>
            <td>${formatHistoryValue(point.river_level_m, 2, 'm')}</td>
          </tr>`;
        }).join('')
      : '<tr><td colspan="5">선택한 기간의 통합 기록이 없습니다.</td></tr>';
  }

  const rangeHours = Math.max(.01, (win.historyEnd - win.historyStart) / (60 * 60 * 1000));
  // V8.5.2: graph only the flood stage. Rain/sewer/river remain in the table
  // below, as requested, so different physical units do not clutter the chart.
  const graphPoints = stagePoints;
  const labels = graphPoints.map(point => {
    const date = historyDate(point.time);
    return rangeHours > 48
      ? date.toLocaleString('ko-KR', {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit'})
      : date.toLocaleTimeString('ko-KR', {hour:'2-digit', minute:'2-digit', second: rangeHours <= 6 ? '2-digit' : undefined});
  });

  const stageValues = graphPoints.map(point => {
    const value = Number(point.level);
    return Number.isFinite(value) ? Math.max(0, Math.min(4, value)) : null;
  });
  const pointColors = stageValues.map(level => level == null ? '#9eb6c8' : COLORS[level]);
  const datasets = [{
    label: '침수 단계', data: stageValues, yAxisID: 'yStage', stepped: 'after',
    borderColor: '#58c8ff', backgroundColor: 'rgba(88,200,255,.10)', fill: false,
    borderWidth: 2.2, pointRadius: 2.5, pointHoverRadius: 5,
    pointBackgroundColor: pointColors, pointBorderColor: '#eaf6ff', pointBorderWidth: 1,
    tension: 0, spanGaps: true
  }];

  if (!win.chart) {
    win.chart = new Chart(canvas, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false, normalized: true,
        layout: { padding: { top: 5, right: 4, bottom: 4, left: 2 } },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: items => {
                const point = graphPoints[items[0]?.dataIndex];
                return point ? historyDate(point.time).toLocaleString('ko-KR') : '';
              },
              label: context => {
                const point = graphPoints[context.dataIndex] || {};
                return point.level == null ? '침수 단계 —' : `침수 Lev${point.level}`;
              }
            }
          }
        },
        scales: {
          yStage: {
            position: 'left', min: -.5, max: 4.5,
            afterBuildTicks: axis => { axis.ticks = [0,1,2,3,4].map(value => ({value})); },
            grid: { color: 'rgba(130,174,202,.14)' },
            ticks: { color: '#9eb6c8', callback: value => Number.isInteger(Number(value)) ? `L${value}` : '' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#9eb6c8', maxTicksLimit: rangeHours > 48 ? 8 : 6, maxRotation: 0 }
          }
        }
      }
    });
    return;
  }

  win.chart.data.labels = labels;
  win.chart.data.datasets = datasets;
  win.chart.options.scales.x.ticks.maxTicksLimit = rangeHours > 48 ? 8 : 6;
  win.chart.update('none');
}

function historyBucketMinutes(hours) {
  if (hours <= 6) return 1;
  if (hours <= 48) return 5;
  if (hours <= 24 * 14) return 30;
  return 60;
}

async function loadWindowHistory(win) {
  if (!cctvWindows.has(win.key)) return;

  const requestSequence = ++win.historyRequestSequence;
  win.historyAbortController?.abort();
  const controller = new AbortController();
  win.historyAbortController = controller;
  const timeoutId = window.setTimeout(() => controller.abort(), 10000);
  const status = win.el.querySelector('.cctv-history-status');
  let fastStageLoaded = false;

  try {
    if (win.historyMode !== 'custom') {
      const end = new Date();
      win.historyEnd = end;
      win.historyStart = new Date(
        end.getTime() - win.historyHours * 60 * 60 * 1000
      );
      setWindowHistoryInputs(
        win,
        win.historyStart,
        win.historyEnd
      );

      const stage = stageForCamera(win.camera);
      const confidence = Number(stageRecordForCamera(win.camera)?.confidence || 0);
      const retainedLive = (win.liveHistoryPoints || []).filter(point => !point.instant_seed);
      win.liveHistoryPoints = [
        {
          time: win.historyStart.toISOString(), level: stage,
          confidence, instant_seed: true
        },
        ...retainedLive,
        {
          time: win.historyEnd.toISOString(), level: stage,
          confidence, instant_seed: true
        }
      ];
    }

    // Render the current stage before any network request. The graph is visible
    // as soon as the popup is created instead of waiting for SQLite or sensor
    // history.
    drawWindowHistory(win);
    if (status) status.textContent = `${historyPresetLabel(win.historyMode)} · 현재 단계 즉시 표시`;

    const params = new URLSearchParams({
      camera_id: win.camera.id,
      region: win.camera.name,
      bucket_minutes: String(
        historyBucketMinutes(win.historyHours)
      ),
      start: win.historyStart.toISOString(),
      end: win.historyEnd.toISOString()
    });
    if (Number.isFinite(Number(win.camera.lat))) {
      params.set('camera_lat', String(Number(win.camera.lat)));
    }
    if (Number.isFinite(Number(win.camera.lon))) {
      params.set('camera_lon', String(Number(win.camera.lon)));
    }

    // First request only the saved flood stages. Environmental history is
    // intentionally deferred so it cannot delay the graph.
    params.set('include_environment', 'false');
    const fastResponse = await fetch('/api/history?' + params, {
      cache: 'no-store',
      signal: controller.signal
    });
    if (!fastResponse.ok) {
      const detail = await fastResponse.json().catch(() => ({}));
      throw new Error(detail.detail || '이력 조회 실패');
    }

    const fastData = await fastResponse.json();
    if (!cctvWindows.has(win.key) || requestSequence !== win.historyRequestSequence) return;

    win.storedHistoryPoints = Array.isArray(fastData.points)
      ? fastData.points
      : [];
    win.combinedHistoryPoints = Array.isArray(fastData.combined_points)
      ? fastData.combined_points
      : [];
    win.historySourceCounts = fastData.source_counts || {};
    win.historyTotalRows = Number(fastData.total_rows || 0);
    fastStageLoaded = true;

    drawWindowHistory(win);
    if (status) status.textContent = `${historyPresetLabel(win.historyMode)} · 침수 그래프 표시 · 환경 이력 불러오는 중`;

    // Complete the table and summaries without blocking the stage graph.
    params.set('include_environment', 'true');
    const fullResponse = await fetch('/api/history?' + params, {
      cache: 'no-store',
      signal: controller.signal
    });
    if (!fullResponse.ok) {
      const detail = await fullResponse.json().catch(() => ({}));
      throw new Error(detail.detail || '환경 이력 조회 실패');
    }
    const data = await fullResponse.json();
    if (!cctvWindows.has(win.key) || requestSequence !== win.historyRequestSequence) return;

    win.storedHistoryPoints = Array.isArray(data.points) ? data.points : win.storedHistoryPoints;
    win.combinedHistoryPoints = Array.isArray(data.combined_points) ? data.combined_points : win.combinedHistoryPoints;
    win.environmentHistory = data.environment || {};
    win.historySourceCounts = data.source_counts || win.historySourceCounts;
    win.historyTotalRows = Number(data.total_rows || win.historyTotalRows || 0);

    drawWindowHistory(win);
  } catch (error) {
    if (requestSequence !== win.historyRequestSequence) return;
    if (status) {
      status.textContent = error?.name === 'AbortError'
        ? (fastStageLoaded ? '침수 그래프 표시 완료 · 환경 이력 응답 지연' : '이전 기록 응답 지연 · 현재 단계 표시 중')
        : (fastStageLoaded ? `침수 그래프 표시 완료 · 환경 이력 오류 · ${error.message}` : `이전 기록 조회 오류 · ${error.message}`);
    }
    if (!fastStageLoaded) {
      win.storedHistoryPoints = [];
      win.combinedHistoryPoints = [];
      win.environmentHistory = {};
    }
    drawWindowHistory(win);
  } finally {
    window.clearTimeout(timeoutId);
    if (win.historyAbortController === controller) {
      win.historyAbortController = null;
    }
  }
}

async function refreshWindowStage(win) {
  if (!cctvWindows.has(win.key)) return;

  const c = win.camera;
  const badge = win.el.querySelector('.cctv-badge');

  try {
    const params = new URLSearchParams({
      url: c.stream_url,
      camera_id: c.id,
      camera_name: c.name,
      camera_address: c.address || '포항 CCTV'
    });
    if (Number.isFinite(Number(c.lat))) {
      params.set('camera_lat', String(Number(c.lat)));
    }
    if (Number.isFinite(Number(c.lon))) {
      params.set('camera_lon', String(Number(c.lon)));
    }
    const response = await fetch('/api/stage?' + params);

    if (!response.ok) {
      throw new Error('AI 판정 요청 실패');
    }

    const data = await response.json();
    if (!cctvWindows.has(win.key)) return;
    if (data.stage === null || data.stage === undefined) {
      if (data.pending) {
        const detected = Array.isArray(data.detections)
          ? data.detections.length
          : 0;
        const small = badge.querySelector('small');
        if (small) {
          small.textContent = `차량 박스 ${detected}대 · 침수단계 분석 중`;
        }
        return;
      }
      throw new Error(
        data.error || data.label || 'AI 판정값 없음'
      );
    }

    const stage = setCameraStage(
      c,
      data.stage,
      data.conf,
      new Date().toISOString(),
      data.positive_confirmed === true
    );

    const votes = data.stage_votes || {};
    const voteText = [0,1,2,3,4].map(level => Number(votes[`Lev${level}`] || 0)).join('/');
    badge.innerHTML = `
      <b>${DEPTHS[stage]} cm</b>
      <span style="color:${COLORS[stage]}">
        Lev${stage} · ${((data.conf || 0) * 100).toFixed(1)}%
      </span>
      <small>
        차량 ${(data.detections || []).length}대 · 투표 ${voteText} ·
        ${data.inference_ms || '-'}ms
      </small>`;

    addWindowHistoryPoint(win, stage, data.conf);

    if (c.local_test) {
      sendToVworld3d({
        type: 'update-test-camera-stage',
        stage,
        confidence: Number(data.conf || 0),
        depthCm: DEPTHS[stage]
      });
    }

    updateMapData();
    scheduleSurface();
    scheduleEventReload();
  } catch (error) {
    const small = badge.querySelector('small');
    if (small) small.textContent = 'AI 판정 오류';
  }
}

function scheduleSurface() {
  clearTimeout(surfaceTimer);
  surfaceTimer = setTimeout(
    buildElevationFloodSurface,
    900
  );
}

function normaliseFloodGeoJson(value) {
  if (
    value &&
    value.type === 'FeatureCollection' &&
    Array.isArray(value.features)
  ) {
    return value;
  }
  return { type: 'FeatureCollection', features: [] };
}

function applyUnifiedOverlayPayload(payload) {
  const flood = normaliseFloodGeoJson(payload?.flood);
  const cameras = Array.isArray(payload?.cameras) && payload.cameras.length
    ? payload.cameras
    : vworldCameraPayload();
  latestUnifiedOverlayPayload = {
    ...(payload || {}),
    camera_count: cameras.length,
    cameras,
    flood
  };
  latestFloodSurface = flood;

  const floodSource = map?.getSource('floodSurface');
  if (floodSource) {
    floodSource.setData(latestFloodSurface);
    setMapLayerVisibility();
  }

  const maximumDepth = Number(payload?.maximum_depth_cm || 0);
  if ($('max-depth')) {
    $('max-depth').textContent = Number.isFinite(maximumDepth)
      ? Math.round(maximumDepth)
      : 0;
  }

  // 브이월드 3D도 이 payload 자체를 사용하므로 두 지도는 완전히
  // 동일한 Python DEM GeoJSON을 그립니다.
  syncVworldFlood();
  if (payload?.surface_pending) {
    clearTimeout(surfaceTimer);
    surfaceTimer = setTimeout(buildElevationFloodSurface, 900);
  }
}

async function buildElevationFloodSurface() {
  if (unifiedOverlayLoading) {
    unifiedOverlayPending = true;
    return;
  }

  unifiedOverlayPending = false;
  const requestId = ++unifiedOverlaySequence;
  unifiedOverlayLoading = true;

  try {
    const response = await fetch(
      `/api/map/vworld-overlays?test=${unifiedFloodTestEnabled ? 'true' : 'false'}`,
      { cache: 'no-store' }
    );

    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        message = body.detail || message;
      } catch (error) { }
      throw new Error(message);
    }

    const payload = await response.json();
    if (requestId !== unifiedOverlaySequence) return;

    applyUnifiedOverlayPayload(payload);
    setStatus(
      'vworld-status',
      mapEngine === 'vworld3d'
        ? '브이월드 실제 3D · 통합 DEM 침수 GeoJSON 정상'
        : '브이월드 침수 분석지도 · 통합 DEM 침수 GeoJSON 정상',
      'ok'
    );
  } catch (error) {
    // Fail closed: a stale inundation area is more dangerous than an empty map.
    console.error('unified DEM flood overlay failed', error);
    applyUnifiedOverlayPayload({
      ...(latestUnifiedOverlayPayload || {}),
      source: 'overlay-error-fail-closed V8.6.4',
      active_camera_count: 0,
      flood_feature_count: 0,
      maximum_depth_cm: 0,
      flood: { type: 'FeatureCollection', features: [] },
      cameras: vworldCameraPayload(),
      errors: [String(error?.message || error)]
    });
    const networkLike = error instanceof TypeError || /Failed to fetch|NetworkError|Load failed/i.test(String(error?.message || ''));
    setStatus(
      'vworld-status',
      networkLike
        ? `통합 침수 GeoJSON 재연결 중 · 오래된 영역 제거 · ${error.message}`
        : `통합 침수 GeoJSON 오류 · 오래된 영역 제거 · ${error.message}`,
      networkLike ? 'warn' : 'error'
    );
    if (networkLike) {
      clearTimeout(surfaceTimer);
      surfaceTimer = setTimeout(buildElevationFloodSurface, 1800);
    }
  } finally {
    if (requestId === unifiedOverlaySequence) {
      unifiedOverlayLoading = false;
      if (unifiedOverlayPending) {
        unifiedOverlayPending = false;
        scheduleSurface();
      }
    }
  }
}

function formatWeatherTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return '-';
  return date.toLocaleString('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}

function setWeatherStreamState(state, text) {
  const dot = $('weather-stream-dot');
  const label = $('weather-stream-state');
  if (dot) dot.className = state || '';
  if (label) label.textContent = text;
}

function renderWeatherCard(data) {
  const card = $('weather-card');
  const updated = $('weather-updated');
  if (!card) return;

  if (!data?.configured) {
    card.innerHTML = `
      <div class="weather-key-required">
        <b>기상청 실시간 인증키 필요</b>
        <p>KMA API허브 인증키를 넣으면 포항 AWS를 60초마다 자동 수집합니다.</p>
        <code>KMA_APIHUB_AUTH_KEY=발급받은_API허브_인증키</code>
        <p>초단기예보도 함께 사용하려면 KMA_SERVICE_KEY를 추가하세요.</p>
      </div>`;
    if (updated) updated.textContent = '실자료 미연동';
    setWeatherStreamState('warn', '인증키 대기');
    return;
  }

  const source = escapeHtml(data.source || '기상청');
  const age = Number(data.age_seconds);
  const stale = Boolean(data.stale);
  const errors = data.errors || {};
  const errorText = Object.values(errors).filter(Boolean).join(' · ');

  card.innerHTML = `
    <div class="weather-primary live-rain">
      <span>최근 1분 강수</span>
      <b>${data.rain_1m_mm == null ? '—' : Number(data.rain_1m_mm).toFixed(1)}</b>
      <em>mm</em>
      <small>${source}</small>
    </div>
    <div class="weather-primary hourly-rain">
      <span>최근 60분 누적</span>
      <b>${data.rain_60m_mm == null ? '—' : Number(data.rain_60m_mm).toFixed(1)}</b>
      <em>mm</em>
      <small>일 누적 ${data.rain_day_mm == null ? '—' : Number(data.rain_day_mm).toFixed(1) + 'mm'}</small>
    </div>
    <div class="weather-metrics">
      <article><span>기온</span><b>${data.temperature_c == null ? '—' : Number(data.temperature_c).toFixed(1) + '℃'}</b></article>
      <article><span>습도</span><b>${data.humidity_pct == null ? '—' : Number(data.humidity_pct).toFixed(0) + '%'}</b></article>
      <article><span>풍속</span><b>${data.wind_ms == null ? '—' : Number(data.wind_ms).toFixed(1) + 'm/s'}</b></article>
      <article><span>관측 시차</span><b>${Number.isFinite(age) ? age + '초' : '—'}</b></article>
    </div>
    ${errorText ? `<div class="weather-error-line">마지막 정상값 유지 · ${escapeHtml(errorText)}</div>` : ''}`;

  if (updated) {
    updated.textContent =
      `${data.station?.name || '포항'} · 관측 ${formatWeatherTime(data.observed_at)}`;
  }

  if (stale) {
    setWeatherStreamState('stale', '자료 지연');
  } else {
    setWeatherStreamState('ok', 'SSE 실시간');
  }
}

function renderRainTimeline(data) {
  const canvas = $('rainfall-chart');
  if (!canvas) return;

  const history = (data?.minute_history || []).slice(-60);
  const forecast = (data?.forecast || []).slice(0, 6);

  const labels = [];
  const observed = [];
  const predicted = [];

  history.forEach(point => {
    labels.push(
      new Date(point.observed_at).toLocaleTimeString(
        'ko-KR',
        { hour: '2-digit', minute: '2-digit' }
      )
    );
    observed.push(Number(point.rain_1m_mm) || 0);
    predicted.push(null);
  });

  forecast.forEach(point => {
    labels.push(
      new Date(point.forecast_at).toLocaleTimeString(
        'ko-KR',
        { hour: '2-digit', minute: '2-digit' }
      )
    );
    observed.push(null);
    predicted.push(Number(point.rain_1h_mm) || 0);
  });

  if (rainChart) {
    rainChart.destroy();
    rainChart = null;
  }

  rainChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          type: 'bar',
          label: 'AWS 1분 강수',
          data: observed,
          backgroundColor: 'rgba(88,200,255,.68)',
          borderColor: '#58c8ff',
          borderWidth: 1,
          borderRadius: 3,
          yAxisID: 'yMinute'
        },
        {
          type: 'line',
          label: '초단기예보 1시간 강수',
          data: predicted,
          borderColor: '#ff66bc',
          backgroundColor: 'rgba(255,102,188,.18)',
          pointBackgroundColor: '#ff66bc',
          pointRadius: 3,
          borderWidth: 2,
          tension: .25,
          spanGaps: false,
          yAxisID: 'yHour'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: '#9eb6c8',
            boxWidth: 12,
            font: { size: 9 }
          }
        },
        tooltip: {
          callbacks: {
            label: context => {
              const value = Number(context.raw || 0);
              return `${context.dataset.label}: ${value.toFixed(1)}mm`;
            }
          }
        }
      },
      scales: {
        yMinute: {
          position: 'left',
          beginAtZero: true,
          grid: { color: 'rgba(130,174,202,.14)' },
          ticks: {
            color: '#9eb6c8',
            callback: value => `${value}mm`
          },
          title: {
            display: true,
            text: '1분',
            color: '#7fcfff'
          }
        },
        yHour: {
          position: 'right',
          beginAtZero: true,
          grid: { drawOnChartArea: false },
          ticks: {
            color: '#dba1c5',
            callback: value => `${value}mm`
          },
          title: {
            display: true,
            text: '예보 1시간',
            color: '#ff9ad1'
          }
        },
        x: {
          grid: { display: false },
          ticks: {
            color: '#9eb6c8',
            maxTicksLimit: 10,
            maxRotation: 0
          }
        }
      }
    }
  });
}

function applyWeatherSnapshot(data) {
  latestWeather = data;
  latestRainGrid = Array.isArray(data?.grid?.points)
    ? data.grid.points
    : [];

  renderWeatherCard(data);
  renderRainTimeline(data);
  updateRainMap();

  const grid = data?.grid || {};
  const maxRain = Number(grid.max_rain_1h_mm || 0);
  const gridTime = formatWeatherTime(grid.updated_at);

  $('rainfall-readout').textContent = data?.configured
    ? `${grid.source || '기상청 강수격자'} · 최대 60분 ${maxRain.toFixed(1)}mm · ${latestRainGrid.length}지점 · ${gridTime}`
    : '기상청 실시간 강수: .env 인증키 설정 필요';

  if (data?.configured && !data?.stale) {
    setStatus(
      'weather-status',
      `기상청 실시간 정상 · 1분 ${Number(data.rain_1m_mm || 0).toFixed(1)}mm · 60분 ${Number(data.rain_60m_mm || 0).toFixed(1)}mm`,
      'ok'
    );
  } else if (data?.configured) {
    setStatus(
      'weather-status',
      `기상청 자료 지연 · 마지막 관측 ${formatWeatherTime(data.observed_at)}`,
      'warn'
    );
  } else {
    setStatus(
      'weather-status',
      '기상청 API 인증키 필요',
      'warn'
    );
  }

  scheduleSurface();
}

async function loadWeather() {
  try {
    const response = await fetch(
      '/api/weather/live',
      { cache: 'no-store' }
    );
    if (!response.ok) throw new Error('기상청 스냅샷 조회 실패');
    applyWeatherSnapshot(await response.json());
  } catch (error) {
    setWeatherStreamState('error', '연결 오류');
    setStatus(
      'weather-status',
      `기상청 데이터 오류 · ${error.message}`,
      'error'
    );
  }
}

function connectWeatherStream() {
  if (weatherEventSource) {
    weatherEventSource.close();
  }

  if (!('EventSource' in window)) {
    setWeatherStreamState('warn', '5분 폴링');
    weatherFallbackTimer = setInterval(loadWeather, 300000);
    return;
  }

  setWeatherStreamState('connecting', 'SSE 연결 중');
  weatherEventSource = new EventSource('/api/weather/stream');

  weatherEventSource.addEventListener('open', () => {
    setWeatherStreamState('ok', 'SSE 실시간');
  });

  weatherEventSource.addEventListener('weather', event => {
    try {
      applyWeatherSnapshot(JSON.parse(event.data));
    } catch (error) {
      console.error('기상청 SSE 파싱 오류', error);
    }
  });

  weatherEventSource.addEventListener('error', () => {
    setWeatherStreamState('error', '재연결 중');
    // EventSource가 자동 재접속하며, 별도 폴링은 안전망으로만 사용합니다.
    if (!weatherFallbackTimer) {
      weatherFallbackTimer = setInterval(loadWeather, 300000);
    }
  });
}

function sensorLevel(sensor) {
  const meters = Number(
    sensor.level_m ?? sensor.water_level_m ?? sensor.wal
  );
  if (Number.isFinite(meters)) {
    return { value: meters, text: `${meters.toFixed(2)} m` };
  }
  const centimeters = Number(
    sensor.level_cm ?? sensor.water_level_cm ?? sensor.value
  );
  if (Number.isFinite(centimeters)) {
    return { value: centimeters / 100, text: `${centimeters.toFixed(0)} cm` };
  }
  return { value: null, text: '—' };
}

function renderLevelSensors(containerId, data, emptyText) {
  const container = $(containerId);
  if (!container) return;
  const items = Array.isArray(data?.sensors)
    ? data.sensors
    : Array.isArray(data?.items)
      ? data.items
      : Array.isArray(data)
        ? data
        : [];

  container.innerHTML = items.map(sensor => {
    const level = sensorLevel(sensor);
    const warning = Number(sensor.warning_m);
    const danger = Number(sensor.danger_m);
    const state = Number.isFinite(level.value) && Number.isFinite(danger) && level.value >= danger
      ? 'danger'
      : Number.isFinite(level.value) && Number.isFinite(warning) && level.value >= warning
        ? 'warn'
        : '';
    const flow = Number(sensor.flow_cms ?? sensor.flux);
    const details = [
      Number.isFinite(flow) ? `유량 ${flow.toFixed(1)}㎥/s` : '',
      sensor.observed_at ? `관측 ${escapeHtml(sensor.observed_at)}` : '',
      sensor.id ? `ID ${escapeHtml(sensor.id)}` : ''
    ].filter(Boolean).join(' · ');
    return `<article class="${state}"><b>${escapeHtml(sensor.name || '수위계')}</b><span>${level.text}</span><small>${details || '관측 정보 없음'}</small></article>`;
  }).join('') || `<p class="empty">${emptyText}</p>`;
}

function setSensorTab(tab) {
  const sewer = tab === 'sewer';
  $('sewer-tab')?.classList.toggle('active', sewer);
  $('river-tab')?.classList.toggle('active', !sewer);
  $('sewer-tab')?.setAttribute('aria-selected', String(sewer));
  $('river-tab')?.setAttribute('aria-selected', String(!sewer));
  $('sewer-panel')?.classList.toggle('hidden', !sewer);
  $('river-panel')?.classList.toggle('hidden', sewer);
}

async function loadSewer() {
  try {
    const response = await fetch('/api/sewer-levels', { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '하수도 수위 조회 실패');
    renderLevelSensors('sewer-grid', data, '하수도 수위 데이터 없음');
    $('water-level-updated').textContent =
      `갱신 ${new Date().toLocaleTimeString('ko-KR')}`;
  } catch (error) {
    $('sewer-grid').innerHTML = `<p class="empty">하수도 수위 오류 · ${escapeHtml(error.message)}</p>`;
  }
}

async function loadRiver() {
  try {
    const response = await fetch('/api/river-levels', { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '하천 수위 조회 실패');
    renderLevelSensors('river-grid', data, '하천 수위 데이터 없음');
    $('water-level-updated').textContent =
      `갱신 ${new Date().toLocaleTimeString('ko-KR')}`;
  } catch (error) {
    $('river-grid').innerHTML = `<p class="empty">하천 수위 오류 · ${escapeHtml(error.message)}</p>`;
  }
}
function eventCamera(event) {
  const eventId = String(event.camera_id || '').trim();
  const eventName = String(event.camera_name || '').trim();

  const byId = eventId
    ? realCameras.find(camera =>
      String(camera.id) === eventId
    )
    : null;

  if (byId) return byId;

  return eventName
    ? realCameras.find(camera =>
      String(camera.name) === eventName
    ) || null
    : null;
}

function eventPosition(event) {
  const camera = eventCamera(event);

  // The current right-side CCTV list is authoritative.
  const lat = Number(camera?.lat ?? event.lat);
  const lon = Number(camera?.lon ?? event.lon);

  return {
    lat: Number.isFinite(lat) ? lat : null,
    lon: Number.isFinite(lon) ? lon : null
  };
}

function eventDisplayName(event) {
  const camera = eventCamera(event);

  // Display exactly the same name shown in the right-side list.
  return (
    camera?.name ||
    event.camera_name ||
    event.display_name ||
    event.site_name ||
    '포항 CCTV'
  );
}

function eventRegion(event) {
  const camera = eventCamera(event);
  return (
    camera?.address ||
    event.address ||
    event.region ||
    event.site_name ||
    eventDisplayName(event)
  );
}

function focusFloodEvent(event) {
  const position = eventPosition(event);

  if (position.lat === null || position.lon === null) {
    alert(
      `${eventDisplayName(event)}의 지도 좌표가 없습니다. ` +
      'CAMERA_n_LAT/LON 설정을 확인하세요.'
    );
    return;
  }

  setMapEngine('vworld3d');

  const message = {
    type: 'focus-map-location',
    lat: position.lat,
    lon: position.lon,
    label: eventDisplayName(event),
    address: eventRegion(event),
    level: Math.max(0, Math.min(4, Number(event.level) || 0)),
    cameraId: String(event.camera_id || ''),
    range_m: 620,
    centerOnTarget: true
  };

  queueVworldFocus(message);
}

function renderFloodEvents(events) {
  const list = $('event-list');
  const count = $('event-count');
  recentFloodEvents = Array.isArray(events) ? events : [];
  if (count) count.textContent = `${recentFloodEvents.length}건`;

  if (!recentFloodEvents.length) {
    list.innerHTML = '<p class="empty">최근 침수 감지 기록이 없습니다.</p>';
    return;
  }

  list.innerHTML = recentFloodEvents.map((event, index) => {
    const level = Math.max(1, Math.min(4, Number(event.level) || 1));
    const confidence = Math.max(0, Number(event.confidence) || 0);
    const depth = Number(event.depth_cm ?? DEPTHS[level]) || 0;
    const position = eventPosition(event);
    const coordinate = (position.lat !== null && position.lon !== null)
      ? `${position.lat.toFixed(5)}, ${position.lon.toFixed(5)}`
      : '좌표 정보 없음';
    const detected = historyDate(event.detected_at).toLocaleString('ko-KR');
    const displayName = eventDisplayName(event);
    const region = eventRegion(event);
    const subtitle = (
      region && region !== displayName
    )
      ? region
      : '침수 감지 지점';
    const deviceCode = String(event.camera_id || '').trim();
    const sourceDeviceCode = String(event.source_camera_id || '').trim();
    const legacyCode = (
      sourceDeviceCode && sourceDeviceCode !== deviceCode
    )
      ? ` · 이전 장비 ${escapeHtml(sourceDeviceCode)}`
      : '';

    return `
      <button type="button" class="event-card level-${level}"
        data-event-index="${index}"
        style="--event-level-color:${COLORS[level]}"
        aria-label="${escapeHtml(displayName)} Lev${level} 지도 위치로 이동">
        <div class="event-card-head">
          <span class="event-level-badge">Lev${level}</span>
          <time>${escapeHtml(detected)}</time>
        </div>
        <b>${escapeHtml(displayName)}</b>
        <p class="event-region">${escapeHtml(subtitle)}</p>
        <div class="event-metrics">
          <span>추정 침수심 <strong>${depth}cm</strong></span>
          <span>확신도 <strong>${(confidence * 100).toFixed(1)}%</strong></span>
        </div>
        <small>${deviceCode ? `장비 ${escapeHtml(deviceCode)}` : ''}${legacyCode} · 좌표 ${escapeHtml(coordinate)} · 클릭하면 지도 위치로 이동</small>
      </button>`;
  }).join('');

  list.querySelectorAll('[data-event-index]').forEach(button => {
    button.addEventListener('click', () => {
      const event = recentFloodEvents[Number(button.dataset.eventIndex)];
      if (event) focusFloodEvent(event);
    });
  });

  const testEvent = recentFloodEvents.find(event =>
    String(event.camera_id) === 'TEST-FLOOD-01'
  );
  if (testEvent) {
    sendToVworld3d({
      type: 'update-test-camera-stage',
      stage: Number(testEvent.level) || 0,
      confidence: Number(testEvent.confidence) || 0,
      depthCm: Number(testEvent.depth_cm) || 0
    });
  }
}

function scheduleEventReload() {
  clearTimeout(eventReloadTimer);
  eventReloadTimer = setTimeout(loadEvents, 450);
}

async function loadEvents() {
  try {
    const response = await fetch('/api/events?limit=12&min_level=1&changes_only=true', { cache: 'no-store' });
    if (!response.ok) throw new Error('침수 감지 기록 조회 실패');
    renderFloodEvents(await response.json());
  } catch (error) {
    $('event-list').innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
  }
}

$('fit-map').addEventListener('click', fitMap);

$('engine-vworld3d').addEventListener(
  'click',
  () => setMapEngine('vworld3d')
);
$('engine-analysis').addEventListener(
  'click',
  () => setMapEngine('analysis')
);
$('focus-test-camera').addEventListener('click', () => {
  setMapEngine('vworld3d');
  sendToVworld3d({
    type: 'focus-test-camera',
    openVideo: true
  });
});


$('base-vworld-base').addEventListener(
  'click',
  () => setBaseMap('vworld-base')
);
$('base-vworld-satellite').addEventListener(
  'click',
  () => setBaseMap('vworld-satellite')
);
$('base-vworld-hybrid').addEventListener(
  'click',
  () => setBaseMap('vworld-hybrid')
);
$('base-fallback').addEventListener(
  'click',
  () => setBaseMap('fallback')
);

$('terrain-mode').addEventListener('click', () => {
  is3d = !is3d;
  map.easeTo({
    pitch: is3d ? 62 : 0,
    bearing: is3d ? -15 : 0,
    duration: 600
  });
  $('terrain-mode').textContent =
    is3d ? '2D 전환' : '3D 전환';
});

$('terrain-exaggeration').addEventListener(
  'input',
  event => {
    try {
      map.setTerrain({
        source: 'terrainSource',
        exaggeration: Number(event.target.value)
      });
    } catch (error) { }
    scheduleSurface();
  }
);

$('flood-exaggeration').addEventListener(
  'input',
  event => {
    if (map.getLayer('flood-volume-3d')) {
      map.setPaintProperty(
        'flood-volume-3d',
        'fill-extrusion-height',
        [
          '*',
          ['get', 'depth_m'],
          Number(event.target.value)
        ]
      );
    }
  }
);

$('surface-toggle').addEventListener(
  'change',
  event => {
    surfaceVisible = event.target.checked;
    setMapLayerVisibility();
    sendToVworld3d({
      type: 'set-vworld-flood-visible',
      visible: surfaceVisible
    });
  }
);

$('rain-toggle').addEventListener(
  'change',
  event => {
    rainfallVisible = event.target.checked;
    setMapLayerVisibility();
  }
);

$('building-toggle').addEventListener(
  'change',
  event => {
    buildingsVisible = event.target.checked;
    sendToVworld3d({
      type: 'set-vworld-buildings',
      visible: buildingsVisible
    });
    setMapLayerVisibility();
  }
);

$('marker-toggle').addEventListener(
  'change',
  event => {
    markersVisible = event.target.checked;
    setMapLayerVisibility();
    sendToVworld3d({
      type: 'set-vworld-cctv-visible',
      visible: markersVisible
    });
  }
);

$('arrange-cctv').addEventListener('click', arrangeCctvWindows);
$('sewer-tab').addEventListener('click', () => setSensorTab('sewer'));
$('river-tab').addEventListener('click', () => setSensorTab('river'));

initMap();
setMapEngine('vworld3d');
clock();
setInterval(clock, 1000);
Promise.allSettled([
  loadModelStatus(),
  loadPrivacyStatus(),
  loadBackgroundAiStatus(),
  loadEnvironmentHistoryStatus(),
  loadRealCameras(),
  loadWeather(),
  loadSewer(),
  loadRiver(),
  loadEvents(),
  health()
]).finally(connectWeatherStream);
setInterval(() => {
  loadSewer();
  loadRiver();
  loadEvents();
  health();
  loadLatestStages();
  loadBackgroundAiStatus();
  loadEnvironmentHistoryStatus();
}, 60000);
setInterval(loadBackgroundAiStatus, 15000);
window.addEventListener('beforeunload', () => {
  weatherEventSource?.close();
  try { cctvSocket?.close(); } catch (_) {}
  if (weatherFallbackTimer) clearInterval(weatherFallbackTimer);
  if (eventReloadTimer) clearTimeout(eventReloadTimer);
});
