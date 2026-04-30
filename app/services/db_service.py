"""SQLAlchemy engine, session factory, and transactional helpers.

ORM model classes live in `app.models.db_models`; this module owns the
database connectivity surface used by the rest of the service.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_settings = get_settings()

engine = create_engine(
    _settings.pg_dsn,
    pool_size=_settings.pg_pool_size,
    max_overflow=_settings.pg_pool_max_overflow,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping() -> bool:
    """Cheap connectivity check used by `/readyz`."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
