from __future__ import annotations

import numpy as np

from app.dem_terrain import TerrainFloodModel


class _NoopContext:
    def lines(self, _layer, _bounds):
        return []

    def status(self):
        return {"noop": True}


def _model() -> TerrainFloodModel:
    return TerrainFloodModel(store=None, context=_NoopContext())


def test_water_exclusion_masks_open_sea_and_hydro_line():
    model = _model()
    dem = np.full((31, 31), 2.0, dtype=np.float32)
    dem[:, :4] = 0.0  # open-sea candidate connected to the grid edge
    hydro = np.zeros(dem.shape, dtype=np.uint8)
    hydro[:, 23] = 255

    water, hydro_core, sea, diagnostics = model._water_exclusion_mask(
        dem, hydro, 30.0
    )

    assert diagnostics["enabled"] is True
    assert water[15, 1] == 1
    assert sea[15, 1] == 1
    assert hydro_core[15, 23] == 1
    assert water[15, 23] == 1
    assert water[15, 15] == 0
    assert diagnostics["water_excluded_cell_count"] > 0


def test_stage4_colour_classification_uses_terrain_depth_before_volume_scale():
    model = _model()
    size = 41
    dem = np.full((size, size), 2.0, dtype=np.float32)
    dem[:, :5] = 0.0
    for row in range(size):
        for col in range(5, size):
            dem[row, col] = (
                1.0
                + 0.02 * abs(col - size // 2)
                + 0.015 * abs(row - size // 2)
            )

    hydro = np.zeros_like(dem, dtype=np.uint8)
    hydro[:, 30] = 255
    source = {
        "id": "TEST",
        "name": "TEST",
        "stage": 4,
        "depth_cm": 70.0,
        "rain_mm": 0.0,
        "kind": "CCTV",
        "lat": 36.0,
        "lon": 129.0,
    }

    (
        depth,
        terrain_depth,
        water,
        seed,
        _source_elevation,
        _spill,
        diagnostics,
    ) = model._depth_grid(
        source,
        dem,
        30.0,
        600.0,
        road_mask=np.zeros_like(hydro),
        hydro_mask=hydro,
        history_grid=np.zeros_like(dem, dtype=np.float32),
    )

    # The old bug scaled almost every rendered cell below 12 cm.  The physical
    # footprint can still be volume-limited, but colour classification keeps the
    # terrain-derived depth gradient.
    assert diagnostics["volume_scale"] < 1.0
    assert float(terrain_depth.max()) >= 0.60
    assert np.count_nonzero(depth[water > 0]) == 0
    assert np.count_nonzero(terrain_depth[water > 0]) == 0

    lons = np.linspace(129.0, 129.01, size)
    lats = np.linspace(36.0, 36.01, size)
    features = model._features_from_depth(
        source,
        depth,
        dem,
        lons,
        lats,
        seed,
        30.0,
        terrain_depth=terrain_depth,
        water_exclusion=water,
    )
    levels = {int(feature["properties"]["level"]) for feature in features}
    assert {2, 3, 4}.issubset(levels)


def test_water_hole_is_preserved_in_geojson_polygon():
    model = _model()
    size = 31
    dem = np.full((size, size), 2.0, dtype=np.float32)
    depth = np.full((size, size), 0.25, dtype=np.float32)
    terrain_depth = depth.copy()
    water = np.zeros((size, size), dtype=np.uint8)
    water[12:19, 12:19] = 1
    depth[water > 0] = 0.0
    terrain_depth[water > 0] = 0.0
    source = {
        "id": "TEST",
        "name": "TEST",
        "stage": 2,
        "depth_cm": 25.0,
        "rain_mm": 0.0,
        "kind": "CCTV",
        "lat": 36.0,
        "lon": 129.0,
    }
    lons = np.linspace(129.0, 129.01, size)
    lats = np.linspace(36.0, 36.01, size)

    features = model._features_from_depth(
        source,
        depth,
        dem,
        lons,
        lats,
        (size // 2, 7),
        15.0,
        terrain_depth=terrain_depth,
        water_exclusion=water,
    )

    assert features
    assert any(
        int(feature["properties"].get("hole_count") or 0) >= 1
        and len(feature["geometry"]["coordinates"]) >= 2
        for feature in features
    )


def test_vworld_svg_renderer_preserves_geojson_holes():
    from pathlib import Path

    html = (Path(__file__).parents[1] / "app" / "static" / "vworld3d.html").read_text(
        encoding="utf-8"
    )
    assert "function floodCoordinateRings" in html
    assert "'path'" in html
    assert "fill-rule', 'evenodd'" in html
    assert "polygon._cartesianRings" in html
    assert "polygon.setAttribute('d', pathData)" in html
