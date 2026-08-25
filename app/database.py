from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import Base


if settings.database_url.startswith("sqlite:///"):
    db_path = settings.database_url.removeprefix("sqlite:///")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

connect_args = {"check_same_thread": False, "timeout": 30} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    # 조회 후 context manager가 commit/close되어도 ORM 객체의 값이
    # 만료되지 않게 합니다. /api/history가 저장된 행을 세션 밖에서
    # 직렬화할 때 DetachedInstanceError가 나던 문제를 방지합니다.
    expire_on_commit=False,
)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    if settings.database_url.startswith("sqlite"):
        with engine.begin() as connection:
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_flood_events_camera_time "
                "ON flood_events (camera_id, detected_at)"
            ))
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_flood_events_name_time "
                "ON flood_events (camera_name, detected_at)"
            ))
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_environment_type_sensor_time "
                "ON environmental_observations (sensor_type, sensor_id, observed_at)"
            ))


@contextmanager
def session_scope():
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
