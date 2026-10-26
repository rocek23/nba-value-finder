"""Archivio remoto opzionale (Postgres) per far sopravvivere dati e storico quote ai riavvii di un hosting gratuito.

Si attiva SOLO se la variabile d'ambiente DATABASE_URL e' impostata (su Streamlit Cloud: pagina «Secrets»). Senza, tutto resta
sui file locali come prima. Il database contiene dati di mercato e statistiche, MAI chiavi API. L'URL non viene mai mostrato.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Protocol

_backend = None


class Backend(Protocol):
    def put(self, key: str, value: str) -> None: ...
    def get(self, key: str) -> tuple[str, datetime] | None: ...
    def append(self, stream: str, value: str) -> None: ...
    def read(self, stream: str) -> list[str]: ...


def enabled() -> bool:
    return get_backend() is not None


def set_backend(b: Backend | None) -> None:
    """Usato dai test (backend in memoria)."""
    global _backend
    _backend = b


def get_backend() -> Backend | None:
    global _backend
    if _backend is not None:
        return _backend
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return None
    _backend = PostgresBackend(url)
    return _backend


class PostgresBackend:
    def __init__(self, url: str):
        self._url = url
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS nvf_kv (key text PRIMARY KEY, value text NOT NULL, updated_at timestamptz NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS nvf_stream (id bigserial PRIMARY KEY, stream text NOT NULL, value text NOT NULL, created_at timestamptz NOT NULL)")
            c.execute("CREATE INDEX IF NOT EXISTS nvf_stream_ix ON nvf_stream(stream, id)")

    def _conn(self):
        import psycopg  # import tardivo: serve solo con DATABASE_URL

        return psycopg.connect(self._url, autocommit=True, connect_timeout=15)

    def put(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO nvf_kv VALUES (%s, %s, now()) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (key, value))

    def get(self, key: str):
        with self._conn() as c:
            row = c.execute("SELECT value, updated_at FROM nvf_kv WHERE key = %s", (key,)).fetchone()
        return (row[0], row[1]) if row else None

    def append(self, stream: str, value: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO nvf_stream (stream, value, created_at) VALUES (%s, %s, now())", (stream, value))

    def read(self, stream: str) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT value FROM nvf_stream WHERE stream = %s ORDER BY id", (stream,)).fetchall()]


class MemoryBackend:
    def __init__(self):
        self.kv: dict[str, tuple[str, datetime]] = {}
        self.streams: dict[str, list[str]] = {}

    def put(self, key, value):
        self.kv[key] = (value, datetime.now(timezone.utc))

    def get(self, key):
        return self.kv.get(key)

    def append(self, stream, value):
        self.streams.setdefault(stream, []).append(value)

    def read(self, stream):
        return list(self.streams.get(stream, []))
