"""SQLite-backed FX tick/bar store (client-side cache).

The ingest backend (ingest.py) writes here; the GUI (fx_monitor.py) and any
other consumer reads from here, so restarting the GUI does NOT re-request
anything from IBKR -- the backend keeps streaming and the GUI reads the cache.

Schema:
  ticks(sym, ts_ms, bid, ask, mid)        raw ticks, optional, prunable
  bars(sym, t_s, interval_s, o, h, l, c)  OHLC bars; base interval = 1s

The backend always writes 1-second bars. Readers request any bar size that is
a whole-second multiple of 1 by aggregating the stored 1-second bars on the
fly (Store.get_bars), so the displayed bar interval is a pure view choice and
does not have to match what the backend stores.

WAL mode is enabled so the single writer (backend) and concurrent readers
(GUI) operate on their own connections without blocking each other.
"""
from __future__ import annotations

import fcntl
import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class StoredBar:
    t: int       # bar start, epoch seconds
    o: float
    h: float
    l: float
    c: float


def reset_db(db_path: str | Path) -> bool:
    """Delete the SQLite database file and its WAL/SHM sidecars.

    A full file wipe (vs truncating tables) also clears WAL state and any
    autoincrement, so the next writer starts from a pristine store. For the
    sim this means the clock returns to SIM_EPOCH. Returns True if anything
    was removed.
    """
    p = Path(db_path)
    candidates = [p, p.parent / (p.name + "-wal"), p.parent / (p.name + "-shm")]
    removed = False
    for f in candidates:
        if f.exists():
            f.unlink()
            removed = True
    return removed


SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    sym  TEXT    NOT NULL,
    ts   INTEGER NOT NULL,     -- epoch ms
    bid  REAL,
    ask  REAL,
    mid  REAL    NOT NULL,
    PRIMARY KEY (sym, ts)
);
CREATE INDEX IF NOT EXISTS ticks_sym_ts ON ticks(sym, ts);

CREATE TABLE IF NOT EXISTS bars (
    sym        TEXT    NOT NULL,
    t          INTEGER NOT NULL,  -- bar start, epoch seconds
    interval_s INTEGER NOT NULL,
    o          REAL    NOT NULL,
    h          REAL    NOT NULL,
    l          REAL    NOT NULL,
    c          REAL    NOT NULL,
    PRIMARY KEY (sym, t, interval_s)
);
CREATE INDEX IF NOT EXISTS bars_sym_t ON bars(sym, t);
"""

BASE_INTERVAL = 1  # seconds; the backend always stores 1-second bars


class Store:
    """SQLite-backed FX store.

    One instance owns one connection. The backend opens one instance for
    writing; each reader (e.g. the GUI) opens its own instance for reading.
    WAL mode lets them run concurrently.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; transactions managed explicitly
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=30000;")
        with self._lock:
            self._conn.executescript(SCHEMA)
        # Exclusive file lock prevents two writer processes from using the
        # same DB simultaneously (which would corrupt OHLC bars via the
        # upsert merge — two random walks writing to the same timestamps
        # produce artificially wide high/low ranges). Readers don't hold
        # this lock.
        self._lock_fd: int | None = None

    # ----- writers (backend) --------------------------------------------

    def acquire_writer_lock(self) -> bool:
        """Try to acquire an exclusive OS-level lock on the database file.

        Returns True if the lock was acquired, False if another process
        already holds it. The lock is released on close() or process exit.
        This prevents two sim/ingest backends from writing simultaneously,
        which would corrupt OHLC bars via the upsert merge (two random
        walks writing to the same timestamps produce artificially wide
        high/low ranges).
        """
        if self._lock_fd is not None:
            return True  # already held
        lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            os.close(fd)
            return False
        self._lock_fd = fd
        os.write(fd, f"{os.getpid()}\n".encode())
        return True

    def release_writer_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._lock_fd)
            self._lock_fd = None

    def begin(self) -> None:
        with self._lock:
            self._conn.execute("BEGIN;")

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def rollback(self) -> None:
        with self._lock:
            self._conn.rollback()

    def insert_tick(
        self, sym: str, ts_ms: int,
        bid: float | None, ask: float | None, mid: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO ticks(sym, ts, bid, ask, mid) "
                "VALUES(?,?,?,?,?)",
                (sym, ts_ms, bid, ask, mid),
            )

    def upsert_bar(
        self, sym: str, t: int, interval_s: int,
        o: float, h: float, l: float, c: float,
    ) -> None:
        # On conflict (same bar already written): keep the first open, take
        # the running max high / min low, and update the close. This makes a
        # mid-bar backend restart non-destructive for high/low.
        with self._lock:
            self._conn.execute(
                "INSERT INTO bars(sym, t, interval_s, o, h, l, c) "
                "VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(sym, t, interval_s) DO UPDATE SET "
                "h = MAX(excluded.h, bars.h), "
                "l = MIN(excluded.l, bars.l), "
                "c = excluded.c",
                (sym, t, interval_s, o, h, l, c),
            )

    def prune_ticks_before(self, ts_ms: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM ticks WHERE ts < ?", (ts_ms,))
            return cur.rowcount

    # ----- readers (GUI / anything) --------------------------------------

    def get_bars(self, sym: str, interval_s: int, limit: int) -> list[StoredBar]:
        """Return up to `limit` most recent bars for `sym` at `interval_s`.

        Bars are returned oldest-first. `interval_s` must be a whole-second
        multiple of the stored base interval (1s). Bars at interval_s == 1 are
        read directly; larger intervals are aggregated from 1-second bars.
        """
        if interval_s < 1:
            raise ValueError("interval_s must be >= 1 second")
        if interval_s % BASE_INTERVAL != 0:
            raise ValueError(
                f"interval_s must be a whole-second multiple of {BASE_INTERVAL}s"
            )
        if interval_s == BASE_INTERVAL:
            rows = self._conn.execute(
                "SELECT t, o, h, l, c FROM bars "
                "WHERE sym=? AND interval_s=? "
                "ORDER BY t DESC LIMIT ?",
                (sym, BASE_INTERVAL, limit),
            ).fetchall()
            rows = list(reversed(rows))
            return [StoredBar(r["t"], r["o"], r["h"], r["l"], r["c"]) for r in rows]

        # Aggregate 1-second bars up to interval_s. Pull a generous window so we
        # still have `limit` complete bars after grouping (the newest group may
        # be partial, the oldest group may be partial -> fetch a few extra).
        need = (limit + 2) * interval_s
        rows = self._conn.execute(
            "SELECT t, o, h, l, c FROM bars "
            "WHERE sym=? AND interval_s=? "
            "ORDER BY t DESC LIMIT ?",
            (sym, BASE_INTERVAL, need),
        ).fetchall()
        rows = list(reversed(rows))
        return _aggregate(rows, interval_s, limit)

    def has_any_bars(self, sym: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM bars WHERE sym=? AND interval_s=? LIMIT 1",
            (sym, BASE_INTERVAL),
        ).fetchone()
        return row is not None

    def get_bars_range(
        self, sym: str, interval_s: int, t_start: float, t_end: float,
    ) -> list[StoredBar]:
        """Return bars for `sym` at `interval_s` overlapping [t_start, t_end].

        Bars are returned oldest-first. Unlike get_bars (which returns the most
        recent N bars), this returns every bar whose period intersects the
        half-open window [t_start, t_end], so a viewer can fetch exactly the
        bars visible in a panned/zoomed view rect.
        """
        if interval_s < 1:
            raise ValueError("interval_s must be >= 1 second")
        if interval_s % BASE_INTERVAL != 0:
            raise ValueError(
                f"interval_s must be a whole-second multiple of {BASE_INTERVAL}s"
            )
        if t_end < t_start:
            return []

        # Snap the fetch window outwards to bar boundaries so the bars
        # straddling t_start and t_end are fully captured.
        key_lo = int(math.floor(t_start / interval_s) * interval_s)
        key_hi = int(math.floor(t_end / interval_s) * interval_s) + interval_s

        if interval_s == BASE_INTERVAL:
            rows = self._conn.execute(
                "SELECT t, o, h, l, c FROM bars "
                "WHERE sym=? AND interval_s=? AND t>=? AND t<? "
                "ORDER BY t",
                (sym, BASE_INTERVAL, key_lo, key_hi),
            ).fetchall()
            return [StoredBar(r["t"], r["o"], r["h"], r["l"], r["c"]) for r in rows]

        rows = self._conn.execute(
            "SELECT t, o, h, l, c FROM bars "
            "WHERE sym=? AND interval_s=? AND t>=? AND t<? "
            "ORDER BY t",
            (sym, BASE_INTERVAL, key_lo, key_hi),
        ).fetchall()
        return _aggregate_range(rows, interval_s)

    def max_bar_time(self, syms: list[str], interval_s: int = BASE_INTERVAL) -> int | None:
        """Most recent bar start time across `syms`, or None if no bars exist."""
        if not syms:
            return None
        placeholders = ",".join("?" * len(syms))
        row = self._conn.execute(
            f"SELECT MAX(t) FROM bars WHERE sym IN ({placeholders}) AND interval_s=?",
            (*syms, interval_s),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        self.release_writer_lock()


def _aggregate(rows: list[sqlite3.Row], interval_s: int, limit: int) -> list[StoredBar]:
    """Group 1-second bars into `interval_s`-second OHLC bars."""
    out: list[StoredBar] = []
    cur_key: int | None = None
    o = h = l = c = 0.0
    for r in rows:
        key = (r["t"] // interval_s) * interval_s
        if cur_key is None or key != cur_key:
            if cur_key is not None:
                out.append(StoredBar(cur_key, o, h, l, c))
            cur_key = key
            o, h, l, c = r["o"], r["h"], r["l"], r["c"]
        else:
            if r["h"] > h:
                h = r["h"]
            if r["l"] < l:
                l = r["l"]
            c = r["c"]
    if cur_key is not None:
        out.append(StoredBar(cur_key, o, h, l, c))
    return out[-limit:]


def _aggregate_range(rows: list[sqlite3.Row], interval_s: int) -> list[StoredBar]:
    """Group 1-second bars into `interval_s`-second OHLC bars (no limit slice).

    Unlike _aggregate this returns every group present in `rows` (oldest-first),
    so a viewer can fetch exactly the bars overlapping a given time range.
    """
    out: list[StoredBar] = []
    cur_key: int | None = None
    o = h = l = c = 0.0
    for r in rows:
        key = (r["t"] // interval_s) * interval_s
        if cur_key is None or key != cur_key:
            if cur_key is not None:
                out.append(StoredBar(cur_key, o, h, l, c))
            cur_key = key
            o, h, l, c = r["o"], r["h"], r["l"], r["c"]
        else:
            if r["h"] > h:
                h = r["h"]
            if r["l"] < l:
                l = r["l"]
            c = r["c"]
    if cur_key is not None:
        out.append(StoredBar(cur_key, o, h, l, c))
    return out