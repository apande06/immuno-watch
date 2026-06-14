"""Alembic migration environment for ImmunoWatch.

Clinical purpose:
    A clinical system's schema *will* evolve — new sensors, new alert fields,
    regulatory audit columns. Every such change to patient data must be a
    reviewed, reversible, version-controlled migration rather than an ad-hoc
    ``ALTER TABLE``. This file is the Alembic entry point that makes that possible.

Migration workflow (reference):
    1. Edit the ORM models in ``data/database.py``.
    2. Autogenerate a migration that diffs the models against the live DB:
           alembic revision --autogenerate -m "add neutrophil_count column"
    3. Review the generated script in ``alembic/versions/`` — autogenerate is a
       starting point, not a substitute for human review (it misses some changes,
       e.g. server defaults and certain type changes).
    4. Apply it:        alembic upgrade head
       Roll back one:    alembic downgrade -1

Note:
    The runtime app uses SQLAlchemy's async engine + ``create_all`` for zero-config
    local dev. Alembic runs migrations synchronously here, which is the standard
    pattern: a sync engine derived from the same URL drives DDL, while the async
    engine serves requests.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make the project importable when Alembic runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import constants as C  # noqa: E402
from data.database import Base  # noqa: E402

config = context.config

# Alembic uses a *synchronous* DBAPI; strip the async driver from the URL.
sync_url = C.DATABASE_URL.replace("+aiosqlite", "")
config.set_main_option("sqlalchemy.url", sync_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a live connection)."""
    context.configure(
        url=sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
