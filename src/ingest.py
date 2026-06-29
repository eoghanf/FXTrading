"""IBKR FX ingestion backend.

Connects to TWS / IB Gateway, subscribes to N FX pairs, and writes every tick
plus 1-second OHLC bars into a SQLite database. The GUI (fx_monitor.py) reads
bars from the same database, so restarting the GUI does NOT re-request anything
from IBKR -- this backend keeps streaming and the GUI just reads the cache.

Two databases, kept strictly separate:
    fxsim.db   -- simulated data (the --simulate mode writes here)
    fxreal.db  -- live IBKR data (the default when --simulate is NOT passed)

Architecture:
    TWS / IB Gateway  --(ib_insync)-->  ingest.py  --(SQLite/WAL)-->  fxreal.db
                                                                         |
                                             fx_monitor.py (reader) <---+

    random walk  --(sim.run_sim)-->  ingest.py --(SQLite/WAL)-->  fxsim.db

Run (live, writes fxreal.db):
    uv run python src/ingest.py --port 7497
    uv run python src/ingest.py --pairs EUR/USD,GBP/USD,USD/TRY,USD/CAD --db fxreal.db

Simulated (writes fxsim.db; starts at 2026-01-01T00:00:00Z and resumes on
restart). For viewing + populating in one launch, prefer fx_monitor.py --simulate:
    uv run python src/ingest.py --simulate --quit-after 8

IBKR prerequisites:
  - A running TWS or IB Gateway with the API socket enabled
      (Configure -> API -> "Enable ActiveX and Socket Clients"; socket port
       7497 paper / 7496 live for TWS; same for IB Gateway).
  - Market-data permission for the FX pairs on your IBKR account.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path

import yaml

from pairs import Pair, load_pairs, parse_pairs
from sim import BarBuilder, flush, run_sim
from store import Store, reset_db

CONFIG_PATH = Path(__file__).resolve().parent.parent / "ibkr_connection.yaml"


def _load_connection_config() -> dict:
    """Read ibkr_connection.yaml. Returns {} if missing/invalid."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open() as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        print(f"[ingest] warning: could not parse {CONFIG_PATH}: {e}",
              file=sys.stderr)
        return {}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def _mid(ticker) -> float | None:
    """Best-effort mid price from an ib_insync Ticker (bid/ask, else last)."""
    bid, ask, last = ticker.bid, ticker.ask, ticker.last
    if _finite(bid) and _finite(ask):
        return (bid + ask) / 2.0
    if _finite(last):
        return float(last)
    return None


# --------------------------------------------------------------------------- #
#  Live IBKR session (with reconnect)
# --------------------------------------------------------------------------- #
def _run_session(args, store: Store, pairs: list[Pair], stop: threading.Event) -> None:
    from ib_insync import IB, Forex

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
    print(f"[ingest] connected to {args.host}:{args.port} as client {args.client_id}")

    builders = {p.name: BarBuilder(p.name) for p in pairs}
    tick_buf: list = []
    disconnected = threading.Event()

    def on_disconnect() -> None:
        disconnected.set()
    ib.disconnectedEvent += on_disconnect

    def on_tick(pair: Pair, ticker) -> None:
        mid = _mid(ticker)
        if mid is None:
            return
        if pair.invert:
            mid = 1.0 / mid
        now = time.time()
        bid = float(ticker.bid) if _finite(ticker.bid) else None
        ask = float(ticker.ask) if _finite(ticker.ask) else None
        tick_buf.append((pair.name, int(now * 1000), bid, ask, mid))
        builders[pair.name].on_price(now, mid)

    for p in pairs:
        contract = Forex(p.base + p.quote)
        try:
            ticker = ib.reqMktData(contract, "", False, False)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] {p.name}: reqMktData failed: {e}", file=sys.stderr)
            continue
        ticker.updateEvent += partial(on_tick, p)
        print(f"[ingest] subscribed {p.name}")

    flush_period = args.flush_ms / 1000.0
    last_flush = time.time()
    last_prune = time.time()
    try:
        while not stop.is_set() and not disconnected.is_set():
            ib.sleep(0.05)
            now = time.time()
            if now - last_flush >= flush_period:
                flush(store, tick_buf, builders)
                last_flush = now
            if args.tick_retention_days > 0 and now - last_prune >= 60.0:
                cutoff = int((now - args.tick_retention_days * 86400) * 1000)
                n = store.prune_ticks_before(cutoff)
                if n:
                    print(f"[ingest] pruned {n} ticks older than "
                          f"{args.tick_retention_days}d")
                last_prune = now
    finally:
        flush(store, tick_buf, builders)
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001 / S110
            pass
        print("[ingest] session ended")


def _real_loop(args, store: Store, pairs: list[Pair], stop: threading.Event) -> None:
    """Run IBKR sessions with automatic reconnect on disconnect/error."""
    while not stop.is_set():
        try:
            _run_session(args, store, pairs, stop)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] session error: {e}", file=sys.stderr)
        if stop.is_set():
            break
        print("[ingest] reconnecting in 5s...")
        for _ in range(50):
            if stop.is_set():
                break
            time.sleep(0.1)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    cfg = _load_connection_config()
    ap = argparse.ArgumentParser(description="IBKR FX ingestion backend.")
    ap.add_argument("--host", default=cfg.get("host", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(cfg.get("port", 7497)),
                    help="TWS/IBG socket port (default from ibkr_connection.yaml; "
                         "IBG paper 4002 / live 4001; TWS paper 7497 / live 7496)")
    ap.add_argument("--client-id", type=int,
                    default=int(cfg.get("client_id", 1)))
    ap.add_argument("--pairs", default=None,
                    help="comma list BASE/QUOTE (default: read from fx_pairs.yaml)")
    ap.add_argument("--db", default=None,
                    help="SQLite database path (default: fxsim.db if --simulate, else fxreal.db)")
    ap.add_argument("--flush-ms", type=int, default=200,
                    help="how often to batch-write to the DB, in ms (default 200)")
    ap.add_argument("--tick-retention-days", type=float, default=7.0,
                    help="delete raw ticks older than N days (0 = keep forever; default 7)")
    ap.add_argument("--simulate", action="store_true",
                    help="random-walk price source instead of IBKR (writes fxsim.db; no TWS needed)")
    ap.add_argument("--reset", action="store_true",
                    help="wipe the sim database and reset the clock to 2026-01-01T00:00:00Z "
                         "(defaults to fxsim.db unless --db is given)")
    ap.add_argument("--quit-after", type=float, default=0.0,
                    help="auto-stop after N seconds (0=disabled; for testing)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.db is None:
        if args.reset:
            args.db = "fxsim.db"   # reset never defaults to the live db
        else:
            args.db = "fxsim.db" if args.simulate else "fxreal.db"
    if args.reset:
        removed = reset_db(args.db)
        print(f"[ingest] reset {args.db}: {'wiped' if removed else 'already empty'}")
    pairs = parse_pairs(args.pairs) if args.pairs else load_pairs()
    store = Store(args.db)
    if not store.acquire_writer_lock():
        print(f"[ingest] ERROR: another writer is already using {args.db}. "
              f"Close the previous instance before starting.")
        store.close()
        raise SystemExit(1)
    stop = threading.Event()

    def handle_sig(_signum, _frame) -> None:
        stop.set()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    if args.quit_after > 0:
        def quitter() -> None:
            stop.wait(args.quit_after)
            stop.set()
        threading.Thread(target=quitter, daemon=True).start()

    print(f"[ingest] db={args.db} pairs={[p.name for p in pairs]} "
          f"simulate={args.simulate}")
    try:
        if args.simulate:
            run_sim(pairs, store, stop, flush_ms=args.flush_ms)
        else:
            _real_loop(args, store, pairs, stop)
    finally:
        store.close()
    print("[ingest] stopped")


if __name__ == "__main__":
    main()