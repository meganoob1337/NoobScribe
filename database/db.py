"""Async SQLAlchemy engine and session for SQLite."""
from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for ORM models."""

    pass


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    try:
        from config import get_config

        return get_config().database_url
    except Exception:
        return os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/noobscribe.db")


def init_engine() -> None:
    """Create async engine and session factory (sync, call once at startup)."""
    global _engine, _session_factory
    url = get_database_url()
    if "sqlite" in url:
        # Ensure parent directory exists for file-based SQLite
        if url.startswith("sqlite+aiosqlite:///"):
            path_part = url.replace("sqlite+aiosqlite:///", "", 1)
            if path_part != ":memory:" and not path_part.startswith("///"):
                db_path = Path(path_part)
                if not db_path.is_absolute():
                    db_path = Path.cwd() / db_path
                db_path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_async_engine(url, echo=os.environ.get("SQL_ECHO", "0") == "1")
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


def _ensure_hide_in_recordings_column_sync(sync_conn) -> None:
    """SQLite: add hide_in_recordings if missing (existing DBs default 0 = visible)."""
    r = sync_conn.execute(text("PRAGMA table_info(recordings)"))
    cols = [row[1] for row in r.fetchall()]
    if cols and "hide_in_recordings" not in cols:
        sync_conn.execute(
            text(
                "ALTER TABLE recordings ADD COLUMN hide_in_recordings "
                "BOOLEAN NOT NULL DEFAULT 0"
            )
        )


async def init_db() -> None:
    """Create tables if they do not exist."""
    if _engine is None:
        init_engine()
    # Import ORM models so they register with Base.metadata
    from database import models as _orm  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        url = str(_engine.url)
        if "sqlite" in url:
            await conn.run_sync(_ensure_hide_in_recordings_column_sync)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield an async session."""
    if _session_factory is None:
        init_engine()
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_engine():
    if _engine is None:
        init_engine()
    return _engine
