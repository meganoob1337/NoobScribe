"""Sync SQLAlchemy engine and session for RQ workers."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.db import Base, ensure_sqlite_schema_sync

_sync_engine = None
_sync_session_factory: sessionmaker[Session] | None = None


def async_url_to_sync(url: str) -> str:
    """Convert aiosqlite URL to sync sqlite URL for worker use."""
    if url.startswith("sqlite+aiosqlite:///"):
        return url.replace("sqlite+aiosqlite:///", "sqlite:///", 1)
    return url


def get_sync_database_url() -> str:
    try:
        from config import get_config

        return async_url_to_sync(get_config().database_url)
    except Exception:
        return async_url_to_sync(
            os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/noobscribe.db")
        )


def init_sync_engine() -> None:
    """Create sync engine and session factory (call once at worker startup)."""
    global _sync_engine, _sync_session_factory
    url = get_sync_database_url()
    if "sqlite" in url:
        if url.startswith("sqlite:///"):
            path_part = url.replace("sqlite:///", "", 1)
            if path_part != ":memory:" and not path_part.startswith("///"):
                db_path = Path(path_part)
                if not db_path.is_absolute():
                    db_path = Path.cwd() / db_path
                db_path.parent.mkdir(parents=True, exist_ok=True)
    _sync_engine = create_engine(
        url,
        echo=os.environ.get("SQL_ECHO", "0") == "1",
        connect_args={"check_same_thread": False} if "sqlite" in url else {},
    )
    _sync_session_factory = sessionmaker(
        bind=_sync_engine,
        class_=Session,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


def init_sync_db() -> None:
    """Create tables if they do not exist (sync, for workers)."""
    if _sync_engine is None:
        init_sync_engine()
    from database import models as _orm  # noqa: F401

    with _sync_engine.begin() as conn:
        Base.metadata.create_all(conn)
        url = str(_sync_engine.url)
        if "sqlite" in url:
            ensure_sqlite_schema_sync(conn)


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Yield a sync session with commit/rollback."""
    if _sync_session_factory is None:
        init_sync_engine()
    session = _sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_sync_engine():
    if _sync_engine is None:
        init_sync_engine()
    return _sync_engine
