from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, declarative_base, sessionmaker


Base = declarative_base()

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _default_database_url() -> str:
    configured_url = os.getenv("DATABASE_URL")
    if configured_url:
        return configured_url.replace("postgres://", "postgresql://", 1)

    if os.getenv("VERCEL"):
        return "sqlite:////tmp/study_assistant.db"

    database_path = Path(__file__).resolve().parent / "study_assistant.db"
    return f"sqlite:///{database_path}"


def configure_database(database_url: str | None = None) -> None:
    global _engine, _SessionLocal

    database_url = database_url or _default_database_url()
    connect_args = {"check_same_thread": False, "timeout": 30} if database_url.startswith("sqlite") else {}

    if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
        database_path = Path(database_url.replace("sqlite:///", "", 1))
        database_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
    )

    if database_url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _enable_sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # pragma: no cover - exercised via runtime
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    _SessionLocal = sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def get_engine():
    if _engine is None:
        configure_database()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        configure_database()
    assert _SessionLocal is not None
    return _SessionLocal


def init_database() -> None:
    Base.metadata.create_all(bind=get_engine())


def get_db() -> Iterator[Session]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
