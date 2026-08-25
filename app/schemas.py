from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    camera_name: str
    source_camera_id: str | None = None
    source_camera_name: str | None = None
    matched_cctv: bool = False
    level: int
    confidence: float
    detected_at: datetime
    image_path: str | None
    details: str | None

    display_name: str | None = None
    site_name: str | None = None
    address: str | None = None
    region: str | None = None
    lat: float | None = None
    lon: float | None = None
    level_label: str | None = None
    depth_cm: int | None = None
    is_flooded: bool = False


class CameraOut(BaseModel):
    camera_id: str
    name: str
    site_name: str | None = None
    address: str | None = None
    source: str
    lat: float
    lon: float
    running: bool
    current_level: int
    current_confidence: float
    fps: float
    last_error: str | None
