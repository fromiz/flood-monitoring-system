from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FloodEvent(Base):
    __tablename__ = "flood_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(100), index=True)
    camera_name: Mapped[str] = mapped_column(String(200))
    level: Mapped[int] = mapped_column(Integer, index=True)
    confidence: Mapped[float] = mapped_column(Float)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class EnvironmentalObservation(Base):
    """Time-series observation for rain / sewer / river history.

    ``value`` is the primary value shown together with CCTV flood stages:
      - rain  -> 60-minute rainfall (mm)
      - sewer -> water level (m)
      - river -> water level (m)

    Source-specific fields (1-minute/day rainfall, flow, warning levels, raw
    payload metadata, etc.) stay in ``details`` so schema changes are not
    required whenever an upstream API adds a field.
    """

    __tablename__ = "environmental_observations"
    __table_args__ = (
        UniqueConstraint(
            "sensor_type",
            "sensor_id",
            "observed_at",
            name="uq_environment_sensor_observed_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sensor_type: Mapped[str] = mapped_column(String(20), index=True)
    sensor_id: Mapped[str] = mapped_column(String(120), index=True)
    sensor_name: Mapped[str] = mapped_column(String(240))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(30))
    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
