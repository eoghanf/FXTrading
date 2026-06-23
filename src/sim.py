"""Simulation engine: random-walk FX prices advancing at wall-clock rate from
a fixed simulated epoch (2026-01-01T00:00:00 UTC).

Both `ingest.py --simulate` (headless population) and `fx_monitor.py --simulate`
(launch + view in one process) call run_sim(). The sim resumes from the last bar
in the DB on restart, so simulated time is contiguous across stop/start cycles:
run for 30s (sim time = 00:00:30), stop for an hour, restart, and the next bar is
00:00:31 -- no gap, no jump to "now".

Sim time advances 1 second per 1 second of wall clock. Bars are always 1 second
wide at the storage layer (BASE_INTERVAL); the viewer aggregates on read.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from pairs import Pair, SEED_PRICES
from store import BASE_INTERVAL, Store

# 2026-01-01T00:00:00Z as a Unix epoch second. Stored bar timestamps are offset
# from this; the viewer renders them with a UTC DateAxisItem so the axis reads
# "2026-01-01 00:00:00" at the start regardless of the host's local timezone.
SIM_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()


@dataclass(slots=True)
class BarBuilder:
    """Accumulates one bar interval in memory; flush() upserts it to the Store.

    On conflict (a bar already exists for this t, e.g. after a restart that
    lands in the same second), Store.upsert_bar keeps the first open and merges
    high/low/close, so a mid-bar restart is non-destructive.
    """
    sym: str
    interval_s: int = BASE_INTERVAL
    cur_t: int | None = None
    o: float = 0.0
    h: float = 0.0
    l: float = 0.0
    c: float = 0.0

    def on_price(self, now: float, price: float) -> None:
        t = int(now // self.interval_s) * self.interval_s
        if self.cur_t is None or t != self.cur_t:
            self.cur_t = t
            self.o = self.h = self.l = self.c = price
        else:
            if price > self.h:
                self.h = price
            if price < self.l:
                self.l = price
            self.c = price

    def upsert(self, store: Store) -> None:
        if self.cur_t is None:
            return
        store.upsert_bar(self.sym, self.cur_t, self.interval_s,
                         self.o, self.h, self.l, self.c)


def flush(store: Store, tick_buf: list, builders: dict[str, BarBuilder]) -> None:
    """Batch-write buffered ticks and every builder's current bar in one txn."""
    if not tick_buf and not any(b.cur_t is not None for b in builders.values()):
        return
    store.begin()
    try:
        for (sym, ts_ms, bid, ask, mid) in tick_buf:
            store.insert_tick(sym, ts_ms, bid, ask, mid)
        for b in builders.values():
            b.upsert(store)
        store.commit()
    except Exception:
        store.rollback()
        raise
    tick_buf.clear()


def sim_start_time(store: Store, pairs: list[Pair],
                   interval_s: int = BASE_INTERVAL) -> float:
    """Resume point: last stored bar t + interval_s, or SIM_EPOCH if empty."""
    syms = [p.name for p in pairs]
    last = store.max_bar_time(syms, interval_s)
    if last is None:
        return float(SIM_EPOCH)
    return float(last + interval_s)


class SimClock:
    """Simulated epoch seconds, advancing at wall-clock rate from a start point."""

    def __init__(self, start_t: float) -> None:
        self._start_t = start_t
        self._wall0 = time.time()

    def now(self) -> float:
        return self._start_t + (time.time() - self._wall0)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def run_sim(
    pairs: list[Pair],
    store: Store,
    stop: threading.Event,
    flush_ms: int = 200,
    tick_hz: int = 20,
) -> None:
    """Generate ticks + 1-second bars, advancing sim time at wall-clock rate.

    Resumes from the last stored bar if the DB already has data; otherwise starts
    at SIM_EPOCH. Price walks continue from each pair's last close so the series
    is continuous across restarts.
    """
    start_t = sim_start_time(store, pairs)
    clock = SimClock(start_t)
    rng = random.Random(0)
    prices = {p.name: SEED_PRICES.get(p.name, 1.0) for p in pairs}
    # If resuming, seed each pair's price from its last stored close so the
    # random walk is continuous rather than jumping back to the hardcoded seed.
    for p in pairs:
        bars = store.get_bars(p.name, BASE_INTERVAL, 1)
        if bars:
            prices[p.name] = bars[-1].c
    builders = {p.name: BarBuilder(p.name) for p in pairs}
    tick_buf: list = []
    tick_period = 1.0 / tick_hz
    flush_period = flush_ms / 1000.0
    next_tick = clock.now()
    last_flush = time.time()
    print(f"[sim] start t={int(start_t)} ({_iso(start_t)}) "
          f"pairs={[p.name for p in pairs]}")
    while not stop.is_set():
        now = clock.now()
        if now >= next_tick:
            for p in pairs:
                old = prices[p.name]
                prices[p.name] *= 1.0 + rng.gauss(0.0, 0.0004)
                mid = prices[p.name]
                tick_buf.append((p.name, int(now * 1000), None, None, mid))
                builders[p.name].on_price(now, mid)
                # Guard: log abnormally large relative price moves so we can
                # trace any future bad-data episode to its exact tick.
                # sigma=0.0004 -> a 10-sigma move (0.4%) is statistically
                # impossible and would indicate a real bug.
                rel = abs(mid / old - 1.0)
                if rel > 0.004:
                    print(f"[sim] WARN large move {p.name}: "
                          f"{old:.5f} -> {mid:.5f} (rel={rel*100:.3f}%) "
                          f"t={int(now)} ({_iso(now)})")
            next_tick = now + tick_period
        if time.time() - last_flush >= flush_period:
            flush(store, tick_buf, builders)
            last_flush = time.time()
        time.sleep(0.005)
    flush(store, tick_buf, builders)
    print(f"[sim] stopped at t={int(clock.now())} ({_iso(clock.now())})")