"""SQLite storage (one file, WAL mode, no external services).

Personal data lives only in ``events`` (pseudonymous shopper/session HMACs), ``impressions``
(the unit HMAC of experiment participants) and ``suppression``. Erasure deletes from all three.
``processing_log`` records what was processed and when, never payloads (Rule 8(3)).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS merchants(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, secret BLOB NOT NULL,
  public_key TEXT NOT NULL, private_key TEXT NOT NULL,
  rules TEXT NOT NULL DEFAULT '{}', graph TEXT NOT NULL DEFAULT '[]', created_at INTEGER NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS m_pub ON merchants(public_key);
CREATE UNIQUE INDEX IF NOT EXISTS m_priv ON merchants(private_key);
CREATE TABLE IF NOT EXISTS categories(
  merchant TEXT NOT NULL, idx INTEGER NOT NULL, path TEXT NOT NULL, name TEXT NOT NULL,
  parent INTEGER NOT NULL, PRIMARY KEY(merchant, idx), UNIQUE(merchant, path));
CREATE TABLE IF NOT EXISTS items(
  merchant TEXT NOT NULL, idx INTEGER NOT NULL, ext_id TEXT NOT NULL, title TEXT NOT NULL,
  description TEXT NOT NULL, category INTEGER NOT NULL, brand TEXT NOT NULL, price REAL,
  stock REAL, regions TEXT, launched_at INTEGER, active INTEGER NOT NULL, added_at INTEGER NOT NULL,
  image TEXT, PRIMARY KEY(merchant, idx), UNIQUE(merchant, ext_id));
CREATE TABLE IF NOT EXISTS events(
  merchant TEXT NOT NULL, shopper TEXT, session TEXT NOT NULL, item INTEGER NOT NULL,
  type INTEGER NOT NULL, ts INTEGER NOT NULL, value REAL, channel TEXT, device TEXT);
CREATE INDEX IF NOT EXISTS ev_shopper ON events(merchant, shopper, ts);
CREATE INDEX IF NOT EXISTS ev_session ON events(merchant, session, ts);
CREATE INDEX IF NOT EXISTS ev_ts ON events(merchant, ts);
CREATE TABLE IF NOT EXISTS suppression(
  merchant TEXT NOT NULL, shopper TEXT NOT NULL, erased_at INTEGER NOT NULL,
  PRIMARY KEY(merchant, shopper));
CREATE TABLE IF NOT EXISTS impressions(
  id TEXT PRIMARY KEY, merchant TEXT NOT NULL, placement TEXT NOT NULL, unit TEXT,
  experiment TEXT, arm TEXT, items TEXT NOT NULL, teams TEXT, reasons TEXT NOT NULL,
  ts INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS imp_exp ON impressions(experiment, ts);
CREATE TABLE IF NOT EXISTS engagements(
  merchant TEXT NOT NULL, impression TEXT NOT NULL, item INTEGER NOT NULL, type TEXT NOT NULL,
  ts INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS eng_imp ON engagements(impression);
CREATE TABLE IF NOT EXISTS exposures(
  experiment TEXT NOT NULL, unit TEXT NOT NULL, arm TEXT NOT NULL, first_ts INTEGER NOT NULL,
  PRIMARY KEY(experiment, unit));
CREATE TABLE IF NOT EXISTS experiments(
  id TEXT PRIMARY KEY, merchant TEXT NOT NULL, spec TEXT NOT NULL, status TEXT NOT NULL,
  started_at INTEGER NOT NULL, stopped_at INTEGER);
CREATE TABLE IF NOT EXISTS models(
  merchant TEXT NOT NULL, version INTEGER NOT NULL, path TEXT NOT NULL, info TEXT NOT NULL,
  created_at INTEGER NOT NULL, active INTEGER NOT NULL, PRIMARY KEY(merchant, version));
CREATE TABLE IF NOT EXISTS processing_log(
  merchant TEXT NOT NULL, ts INTEGER NOT NULL, action TEXT NOT NULL, count INTEGER NOT NULL);
"""


class DB:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._con = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._con.execute("PRAGMA foreign_keys=ON")
            self._con.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Additive column changes for databases created by earlier versions."""
        cols = {r["name"] for r in self._con.execute("PRAGMA table_info(items)")}
        if "image" not in cols:
            self._con.execute("ALTER TABLE items ADD COLUMN image TEXT")

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                yield self._con
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise

    def q(self, sql: str, args: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._con.execute(sql, args).fetchall()

    def one(self, sql: str, args: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args: Sequence[Any] = ()) -> int:
        with self._lock:
            return self._con.execute(sql, args).rowcount

    def many(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self.tx() as c:
            c.executemany(sql, rows)

    def log(self, merchant: str, action: str, count: int = 1) -> None:
        self.x(
            "INSERT INTO processing_log VALUES (?, ?, ?, ?)",
            (merchant, int(time.time()), action, count),
        )

    @staticmethod
    def dumps(x: Any) -> str:
        return json.dumps(x, separators=(",", ":"))

    def close(self) -> None:
        with self._lock:
            self._con.close()
