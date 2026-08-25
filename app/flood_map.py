from __future__ import annotations
import math

LEVEL_DEPTH_CM = {0: 0.0, 1: 8.0, 2: 25.0, 3: 48.0, 4: 70.0}

def level_to_depth_cm(level: int) -> float:
    return LEVEL_DEPTH_CM.get(max(0, min(4, int(level))), 0.0)

def depth_to_level(depth: float) -> int:
    if depth < 1: return 0
    if depth < 12: return 1
    if depth < 35: return 2
    if depth < 60: return 3
    return 4

def build_depth_surface(cameras: list[dict], grid_size: int = 26) -> dict:
    if not cameras:
        return {"type":"FeatureCollection","features":[]}
    lats=[float(c["lat"]) for c in cameras]; lons=[float(c["lon"]) for c in cameras]
    lat_pad=max(.003,(max(lats)-min(lats))*.8); lon_pad=max(.004,(max(lons)-min(lons))*.8)
    lat0,lat1=min(lats)-lat_pad,max(lats)+lat_pad; lon0,lon1=min(lons)-lon_pad,max(lons)+lon_pad
    dy=(lat1-lat0)/grid_size; dx=(lon1-lon0)/grid_size; features=[]
    for r in range(grid_size):
        for c in range(grid_size):
            lat=lat0+(r+.5)*dy; lon=lon0+(c+.5)*dx; num=den=0.0; nearest=1e9
            for cam in cameras:
                my=(lat-float(cam["lat"]))*111000; mx=(lon-float(cam["lon"]))*88000
                dist=max(15.0,math.hypot(mx,my)); nearest=min(nearest,dist); w=1/(dist**2)
                num += level_to_depth_cm(cam["current_level"])*w; den += w
            depth=(num/den if den else 0)*math.exp(-max(0,nearest-120)/1100)
            if depth < .8: continue
            x0,x1=lon0+c*dx,lon0+(c+1)*dx; y0,y1=lat0+r*dy,lat0+(r+1)*dy
            features.append({"type":"Feature","properties":{"depth_cm":round(depth,1),"level":depth_to_level(depth)},"geometry":{"type":"Polygon","coordinates":[[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]}})
    return {"type":"FeatureCollection","bbox":[lon0,lat0,lon1,lat1],"features":features,"method":"CCTV level IDW prototype"}
