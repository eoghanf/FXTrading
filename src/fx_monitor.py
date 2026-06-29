"""Live FX monitor: reads OHLC bars from the local SQLite cache and draws
scrolling candlestick charts with a real datetime x-axis.

Default pairs: EUR/USD, GBP/USD, USD/TRY, USD/CAD (override with --pairs).

Click any subplot to pop it out into its own window with zoom/drag/scroll; the
datetime x-axis auto-reformats as you zoom (years -> days -> HH:MM:SS).

Two databases, kept strictly separate:
    fxsim.db   -- simulated data (written by the sim; starts 2026-01-01 UTC,
                  resumes from the last bar on restart)
    fxreal.db  -- live IBKR data (written by src/ingest.py against TWS/IBG)

This viewer never talks to IBKR directly; it reads whatever a backend wrote.

Sim environment (one launch both populates fxsim.db AND shows the graphs):
    uv run python src/fx_monitor.py --simulate
    uv run python src/fx_monitor.py --simulate --bar-seconds 5 --window 120

Live environment (run the backend once, then the viewer; or many viewers):
    uv run python src/ingest.py --port 7497
    uv run python src/fx_monitor.py --db fxreal.db

Smoke test (no display needed; auto-quits):
    QT_QPA_PLATFORM=offscreen uv run python src/fx_monitor.py --simulate --quit-after 3
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

# Pin pyqtgraph onto PySide6 so the Qt binding is deterministic.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pairs import Pair, load_pairs, parse_pairs
from sim import run_sim
from store import Store, reset_db

UP_COLOR = "#26a69a"
DN_COLOR = "#ef5350"

# Admissible candlestick widths (seconds), ascending. The pop-out window picks
# one of these based on the visible time span (see choose_bar_seconds).
ADMISSIBLE_BAR_SECONDS: list[int] = [1, 10, 20, 30, 60, 300, 600, 3600, 86400]
# Below this visible span (seconds) the pop-out always shows 1s candles.
SMALL_WINDOW_THRESHOLD: float = 100.0
# Above SMALL_WINDOW_THRESHOLD, the widest admissible width is chosen such that
# the window still spans at least this many candles.
MIN_BARS_IN_VIEW: int = 30


def choose_bar_seconds(window_span: float) -> int:
    """Pick the candlestick width (seconds) for a visible window span.

    Spans <= 100s use 1s candles. Otherwise the largest admissible width that
    still yields at least MIN_BARS_IN_VIEW candles across the window is chosen.
    """
    if window_span <= SMALL_WINDOW_THRESHOLD:
        return 1
    cap = window_span / MIN_BARS_IN_VIEW
    chosen = 1
    for s in ADMISSIBLE_BAR_SECONDS:
        if s <= cap:
            chosen = s
        else:
            break
    return chosen


def _fmt_bar_seconds(s: int) -> str:
    """Human-readable width for the pop-out readout (1s, 10s, 1m, 5m, 1h, 1d)."""
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, r = divmod(s, 60)
        return f"{m}m" if r == 0 else f"{m}m{r}s"
    if s < 86400:
        h, r = divmod(s, 3600)
        return f"{h}h" if r == 0 else f"{h}h{r // 60}m"
    return f"{s // 86400}d"


DAY_SECONDS = 86400


class TimeAxisItem(pg.AxisItem):
    """Bottom x-axis whose tick spacing is a multiple of the candlestick width.

    Ticks therefore land on bar boundaries and the spacing coarsens with the
    bar width. Labels are time-of-day (HH:MM:SS / HH:MM) for sub-day bars and
    dates (%Y-%m-%d) for 1-day bars.
    """

    def __init__(self, orientation="bottom", **kw) -> None:
        super().__init__(orientation, **kw)
        self.bar_seconds = 1

    def set_bar_seconds(self, s: int) -> None:
        self.bar_seconds = max(int(s), 1)

    def tickValues(self, minVal, maxVal, size):
        b = self.bar_seconds
        span = maxVal - minVal
        if span <= 0 or size <= 0:
            return [(float(b), [])]
        # ~8 ticks, but never denser than one label per ~60 px.
        ideal = max(span / 8.0, span * 60.0 / size)
        step = float(b)
        for m in (1, 2, 3, 5, 6, 10, 15, 30, 60, 120, 300, 600,
                  1800, 3600, 7200, 14400, 43200, 86400):
            cand = float(b) * m
            if cand >= ideal:
                step = cand
                break
        start = math.floor(minVal / step) * step + step
        ticks = []
        t = start
        while t <= maxVal + 1e-6:
            ticks.append(float(t))
            t += step
        return [(step, ticks)]

    def tickStrings(self, values, scale, spacing):
        b = self.bar_seconds
        if b >= DAY_SECONDS:
            fmt = "%Y-%m-%d"
        elif spacing < 60:
            fmt = "%H:%M:%S"
        else:
            fmt = "%H:%M"
        out = []
        for v in values:
            try:
                out.append(datetime.fromtimestamp(v, tz=timezone.utc).strftime(fmt))
            except (OSError, ValueError, OverflowError):
                out.append("")
        return out


class DayAxisItem(pg.AxisItem):
    """Top x-axis that divides the visible range into days.

    Ticks land on UTC midnights (multiples of DAY_SECONDS). Shown only for
    sub-day candlestick widths; for a single-day window no tick falls inside
    the view, so the owning Popout sets a date label instead.
    """

    def __init__(self, orientation="top", **kw) -> None:
        super().__init__(orientation, **kw)

    def tickValues(self, minVal, maxVal, size):
        span = maxVal - minVal
        if span <= 0 or size <= 0:
            return [(float(DAY_SECONDS), [])]
        ideal = max(span / 8.0, span * 60.0 / size)
        step = float(DAY_SECONDS)
        for m in (1, 2, 3, 5, 7, 14, 30, 60, 90, 180, 365):
            cand = DAY_SECONDS * m
            if cand >= ideal:
                step = float(cand)
                break
        start = math.floor(minVal / step) * step + step
        ticks = []
        t = start
        while t <= maxVal + 1e-6:
            ticks.append(float(t))
            t += step
        return [(step, ticks)]

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            try:
                out.append(datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d"))
            except (OSError, ValueError, OverflowError):
                out.append("")
        return out


@dataclass(slots=True)
class Bar:
    t: float   # bar start, epoch seconds (sim time for fxsim, wall time for fxreal)
    o: float
    h: float
    l: float
    c: float


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


# --------------------------------------------------------------------------- #
#  Data source: reads pre-built bars from the SQLite cache
# --------------------------------------------------------------------------- #
class DbSource:
    """Reads bars from a SQLite database written by src/ingest.py or the sim.

    The viewer owns its own read connection (WAL mode -> never blocks the
    writer), so restarting the viewer re-reads the cache instead of
    re-requesting data from IBKR / regenerating sim history.
    """

    def __init__(self, db_path: str, pairs: list[Pair],
                 bar_seconds: int, window: int) -> None:
        self._pairs = pairs
        self._bar_seconds = bar_seconds
        self._window = window
        self._db_path = db_path
        self._store = Store(db_path)
        self._has_data = {p.name: self._store.has_any_bars(p.name) for p in pairs}

    def get_bars(self, sym: str) -> list[Bar]:
        stored = self._store.get_bars(sym, self._bar_seconds, self._window)
        if not self._has_data[sym] and stored:
            self._has_data[sym] = True
        return [Bar(b.t, b.o, b.h, b.l, b.c) for b in stored]

    def get_bars_at(self, sym: str, interval_s: int, limit: int) -> list[Bar]:
        """Return up to `limit` recent bars for `sym` at an arbitrary interval.

        Used by the pop-out window, whose candlestick width adapts to the zoom
        level and therefore needs to fetch bars at a view-chosen interval
        rather than the fixed display interval.
        """
        stored = self._store.get_bars(sym, interval_s, limit)
        if not self._has_data[sym] and stored:
            self._has_data[sym] = True
        return [Bar(b.t, b.o, b.h, b.l, b.c) for b in stored]

    def get_bars_range(self, sym: str, interval_s: int,
                       t_start: float, t_end: float) -> list[Bar]:
        """Return bars for `sym` at `interval_s` overlapping [t_start, t_end].

        Used by the pop-out window to fetch exactly the bars visible in the
        current view rect (which may be panned away from the most recent data),
        rather than the N most recent bars.
        """
        stored = self._store.get_bars_range(sym, interval_s, t_start, t_end)
        if stored and not self._has_data[sym]:
            self._has_data[sym] = True
        return [Bar(b.t, b.o, b.h, b.l, b.c) for b in stored]

    def status(self) -> str:
        if not any(self._has_data.values()):
            return f"db: {self._db_path} (no data yet)"
        return f"db: {self._db_path}"

    def close(self) -> None:
        self._store.close()


# --------------------------------------------------------------------------- #
#  Candlestick graphics object
# --------------------------------------------------------------------------- #
class CandlestickItem(pg.GraphicsObject):
    """Paints OHLC candles (wick line + open/close body), green up / red down."""

    def __init__(self) -> None:
        super().__init__()
        self._t = np.empty(0)
        self._o = np.empty(0)
        self._h = np.empty(0)
        self._l = np.empty(0)
        self._c = np.empty(0)
        self._w = 0.3
        self._up_pen = pg.mkPen(UP_COLOR)
        self._up_brush = pg.mkBrush(UP_COLOR)
        self._dn_pen = pg.mkPen(DN_COLOR)
        self._dn_brush = pg.mkBrush(DN_COLOR)

    def setData(self, t, o, h, l, c, bar_seconds=None) -> None:
        self._t = np.asarray(t, dtype=float)
        self._o = np.asarray(o, dtype=float)
        self._h = np.asarray(h, dtype=float)
        self._l = np.asarray(l, dtype=float)
        self._c = np.asarray(c, dtype=float)
        if bar_seconds is not None:
            self._w = bar_seconds / 3.0
        elif len(self._t) > 1:
            self._w = (self._t[1] - self._t[0]) / 3.0
        self.prepareGeometryChange()
        self.update()

    def body_width(self) -> float:
        """Full candle-body width (2 * half-width). Marker uses 2x this."""
        return 2.0 * self._w

    def boundingRect(self) -> QtCore.QRectF:
        if len(self._t) == 0:
            return QtCore.QRectF()
        dh = float(self._h.max() - self._l.min())
        if dh == 0:
            dh = 1e-6
        w = self._w
        return QtCore.QRectF(
            float(self._t.min()) - w - 1,
            float(self._l.min()),
            float(self._t.max() - self._t.min()) + 2 * w + 2,
            dh,
        )

    def paint(self, p: QtGui.QPainter, opt, widget) -> None:
        n = len(self._t)
        if n == 0:
            return
        w = self._w
        for i in range(n):
            t = float(self._t[i]); o = float(self._o[i]); h = float(self._h[i])
            l = float(self._l[i]); c = float(self._c[i])
            if c >= o:
                pen, brush = self._up_pen, self._up_brush
            else:
                pen, brush = self._dn_pen, self._dn_brush
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(t, l), QtCore.QPointF(t, h))
            p.setBrush(brush)
            top = o if o > c else c
            bot = c if o > c else o
            body = top - bot
            if body < 1e-12:
                p.drawLine(QtCore.QPointF(t - w, o), QtCore.QPointF(t + w, o))
            else:
                p.drawRect(QtCore.QRectF(t - w, bot, 2 * w, body))


# --------------------------------------------------------------------------- #
#  Pop-out window (click a subplot to open)
# --------------------------------------------------------------------------- #
class Popout(QtWidgets.QWidget):
    """Dedicated, interactive window for one pair: zoom/drag/scroll enabled.

    The datetime axis (UTC) reformats automatically as you zoom in/out.

    Zoom-driven candlestick width: the bar interval is chosen from the visible
    time span -- 1s for windows <= 100s, otherwise the widest admissible width
    (see ADMISSIBLE_BAR_SECONDS) that still shows at least MIN_BARS_IN_VIEW
    candles. Candles are centred on the midpoint of the period they cover. The
    active width is shown in the top-right corner.
    """

    def __init__(self, name: str, source: DbSource, bar_seconds: int,
                 window: int, on_close) -> None:
        super().__init__()
        self.name = name
        self.source = source
        self.bar_seconds = bar_seconds        # default interval (seeds initial view)
        self._window = window                 # default bar count (seeds initial view)
        self._on_close = on_close
        # Currently applied candlestick width; updated as the view span changes.
        self._bar_seconds = bar_seconds
        self._fitted = False                  # one-shot initial view seeding
        self._y_set = False                   # one-shot initial Y fit
        self._in_refresh = False              # reentrancy guard
        self.setWindowTitle(f"FX Monitor — {name}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        # Two-tier x-axis: bottom shows time-of-day (or dates for 1d bars) with
        # tick spacing matched to the candlestick width; top divides the range
        # into days when the bars are sub-day.
        self._time_axis = TimeAxisItem(orientation="bottom")
        self._time_axis.set_bar_seconds(bar_seconds)
        self._day_axis = DayAxisItem(orientation="top")
        self.plot = pg.PlotWidget(
            title=name,
            axisItems={"bottom": self._time_axis, "top": self._day_axis},
        )
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "price")
        self.plot.setLabel("bottom", "time (UTC)")
        # Full mouse interaction: left-drag pan, wheel zoom, right-drag zoom-box,
        # right-click menu (view all / export / axis options).
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.setMenuEnabled(True)
        # We manage the view range ourselves (initial fit + span-driven width),
        # so disable auto-range: item-bound changes must not override user zoom.
        self.plot.enableAutoRange(x=False, y=False)
        p = self.plot.getPlotItem()
        p.getAxis("left").setWidth(60)
        layout.addWidget(self.plot)
        self.resize(820, 520)

        self.candle = CandlestickItem()
        p.addItem(self.candle)
        self.line = pg.PlotDataItem(pen=pg.mkPen("#ffd166", width=2))
        p.addItem(self.line)
        self.label = pg.TextItem(anchor=(0, 0.5), color="#e6e8ee")
        p.addItem(self.label)
        # Candlestick-width readout, pinned to the top-right of the viewport.
        self.width_label = pg.TextItem(anchor=(1, 0), color="#ffd166")
        self.width_label.setZValue(50)
        p.addItem(self.width_label, ignoreBounds=True)
        self.width_label.setText(_fmt_bar_seconds(self._bar_seconds))

        p.getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)
        self.show()

    # ----- view helpers --------------------------------------------------

    def _visible_span(self) -> float:
        return float(self.plot.getPlotItem().getViewBox().viewRect().width())

    def _pin_width_label(self) -> None:
        """Anchor the width readout to the top-right of the current viewport."""
        vr = self.plot.getPlotItem().getViewBox().viewRect()
        self.width_label.setPos(vr.right(), vr.top())

    def _on_x_range_changed(self, *_args) -> None:
        # Keep the readout pinned on every pan/zoom (cheap; no fetch).
        self._pin_width_label()
        if self._in_refresh:
            return
        # Refresh immediately on any pan/zoom: refresh now fetches by visible
        # range, so a pan into a new region must reload candles there rather than
        # waiting for the periodic timer (avoids missing-candle flicker).
        self.refresh()

    # ----- data update ---------------------------------------------------

    def refresh(self) -> None:
        if self._in_refresh:
            return
        self._in_refresh = True
        try:
            p = self.plot.getPlotItem()
            if not self._fitted:
                # One-shot: seed the initial view to the default window span so
                # the first span-driven width selection is meaningful (not the
                # [0, 1] default range). The right edge is anchored to the last
                # bar's period-right-edge at the chosen interval so the most
                # recent candle has spacing (matching the base display, which
                # ends the view at cur_t + interval).
                seed_span = max(self._window * self.bar_seconds, 1)
                seed_interval = choose_bar_seconds(seed_span)
                probe = self.source.get_bars_at(self.name, seed_interval, 4)
                if not probe:
                    self.width_label.setText(_fmt_bar_seconds(seed_interval))
                    self._pin_width_label()
                    return
                right = float(probe[-1].t) + seed_interval
                p.setXRange(right - seed_span, right + seed_interval * 10, padding=0)
                self._fitted = True
                # setXRange fires sigXRangeChanged; the reentrancy guard makes
                # the recursive _on_x_range_changed a no-op for the fetch.

            span = self._visible_span()
            interval = choose_bar_seconds(span)
            self._bar_seconds = interval
            self.width_label.setText(_fmt_bar_seconds(interval))
            # Drive the two-tier axis: bottom tick spacing matches the bar
            # width; top day axis (with dividing ticks / date label) is shown
            # only for sub-day bar widths.
            self._time_axis.set_bar_seconds(interval)
            self._time_axis.setLabel("date (UTC)" if interval >= DAY_SECONDS
                                     else "time (UTC)")
            self._time_axis.update()
            vr = p.getViewBox().viewRect()
            if interval < DAY_SECONDS:
                p.showAxis("top", True)
                multi_day = (math.floor(vr.left() / DAY_SECONDS)
                             != math.floor(vr.right() / DAY_SECONDS))
                if multi_day:
                    self._day_axis.setLabel("")
                else:
                    day0 = math.floor(vr.left() / DAY_SECONDS) * DAY_SECONDS
                    self._day_axis.setLabel(
                        datetime.fromtimestamp(day0, tz=timezone.utc).strftime("%Y-%m-%d")
                    )
            else:
                p.showAxis("top", False)
            # Fetch exactly the bars overlapping the visible view rect (not the
            # N most recent): panning/zooming into an older region, or the
            # latest bars scrolling past a fixed view, must not starve the plot.
            vr = p.getViewBox().viewRect()
            bars = self.source.get_bars_range(
                self.name, interval, vr.left(), vr.right(),
            )
            if not bars:
                self.candle.setData(np.empty(0), np.empty(0), np.empty(0),
                                     np.empty(0), np.empty(0),
                                     bar_seconds=interval)
                self._pin_width_label()
                return

            # Centre each candle on the midpoint of the period it covers
            # (bar t is the start; the period runs [t, t + interval)).
            half = interval / 2.0
            t0 = np.fromiter((b.t + half for b in bars), dtype=float)
            o = np.fromiter((b.o for b in bars), dtype=float)
            h = np.fromiter((b.h for b in bars), dtype=float)
            l = np.fromiter((b.l for b in bars), dtype=float)
            c = np.fromiter((b.c for b in bars), dtype=float)
            self.candle.setData(t0, o, h, l, c, bar_seconds=interval)

            cur_centre = float(bars[-1].t) + half
            price = float(c[-1])
            hw = self.candle.body_width()  # 2x candle body width => half-width
            self.line.setData(x=np.array([cur_centre - hw, cur_centre + hw]),
                              y=np.array([price, price]))
            delta = price - float(o[0])
            self.label.setText(f"{price:.5f}   Δ{delta:+.5f}")
            self.label.setPos(cur_centre + hw, price)

            if not self._y_set:
                hi = float(h.max())
                lo = float(l.min())
                pad = max((hi - lo) * 0.08, abs(hi) * 1e-5)
                p.setYRange(lo - pad, hi + pad, padding=0)
                self._y_set = True

            self._pin_width_label()
        finally:
            self._in_refresh = False

    def closeEvent(self, ev) -> None:
        try:
            self._on_close(self.name)
        finally:
            super().closeEvent(ev)


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Live FX OHLC monitor (SQLite cache).")
    ap.add_argument("--db", default=None,
                    help="SQLite database (default: fxsim.db if --simulate, else fxreal.db)")
    ap.add_argument("--bar-seconds", type=int, default=5,
                    help="displayed OHLC bar interval in whole seconds (default 5)")
    ap.add_argument("--window", type=int, default=80,
                    help="number of visible bars (default 80)")
    ap.add_argument("--refresh-ms", type=int, default=100,
                    help="GUI refresh interval in ms (default 100)")
    ap.add_argument("--pairs", default=None,
                    help="comma list BASE/QUOTE (default: read from fx_pairs.yaml)")
    ap.add_argument("--simulate", action="store_true",
                    help="launch the sim (populates fxsim.db) AND view it in one process")
    ap.add_argument("--reset", action="store_true",
                    help="wipe the sim database and reset the clock to 2026-01-01T00:00:00Z "
                         "(defaults to fxsim.db unless --db is given)")
    ap.add_argument("--quit-after", type=float, default=0.0,
                    help="auto-quit after N seconds (0=disabled; for testing)")
    ap.add_argument("--screenshot", default="",
                    help="save a PNG to PATH right before auto-quitting (testing)")
    args = ap.parse_args()

    if args.db is None:
        if args.reset:
            args.db = "fxsim.db"   # reset never defaults to the live db
        else:
            args.db = "fxsim.db" if args.simulate else "fxreal.db"
    if args.reset:
        removed = reset_db(args.db)
        print(f"[fx] reset {args.db}: {'wiped' if removed else 'already empty'}")
    pairs = parse_pairs(args.pairs) if args.pairs else load_pairs()

    # Reader connection (the GUI polls this every refresh interval).
    source = DbSource(args.db, pairs, args.bar_seconds, args.window)

    # When simulating, also run the sim engine in-process as the DB writer. The
    # viewer and the sim share the file via WAL: separate connections, no
    # blocking. On restart, run_sim resumes from the last stored bar.
    writer_store: Store | None = None
    sim_thread: threading.Thread | None = None
    sim_stop: threading.Event | None = None
    if args.simulate:
        writer_store = Store(args.db)
        if not writer_store.acquire_writer_lock():
            print(f"[fx] ERROR: another writer is already using {args.db}. "
                  f"Close the previous instance before starting a new sim.")
            writer_store.close()
            raise SystemExit(1)
        sim_stop = threading.Event()
        sim_thread = threading.Thread(
            target=run_sim, args=(pairs, writer_store, sim_stop), daemon=True,
        )
        sim_thread.start()

    pg.setConfigOptions(antialias=True, background="#0e1116", foreground="#c7ccd6")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="FX Monitor")
    win.resize(1200, 820)
    win.setBackground("#0e1116")
    win.show()

    plots: dict[str, pg.PlotItem] = {}
    candles: dict[str, CandlestickItem] = {}
    lines: dict[str, pg.PlotDataItem] = {}
    labels: dict[str, pg.TextItem] = {}

    for i, p in enumerate(pairs):
        name = p.name
        axis = DateAxisItem(orientation="bottom", utcOffset=0)
        plot = win.addPlot(row=i // 2, col=i % 2, title=name,
                           axisItems={"bottom": axis})
        plot.showGrid(x=False, y=True, alpha=0.25)
        plot.setLabel("left", "price")
        plot.setLabel("bottom", "time (UTC)")
        plot.setMouseEnabled(x=False, y=False)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.getAxis("bottom").setStyle(showValues=True)
        plot.getAxis("left").setWidth(60)
        plot.enableAutoRange(axis="x", enable=False)
        plot.enableAutoRange(axis="y", enable=False)

        c = CandlestickItem()
        plot.addItem(c)
        line = pg.PlotDataItem(pen=pg.mkPen("#ffd166", width=2))
        plot.addItem(line)
        lbl = pg.TextItem(anchor=(0, 0.5), color="#e6e8ee")
        plot.addItem(lbl)

        plots[name] = plot
        candles[name] = c
        lines[name] = line
        labels[name] = lbl

    status_lbl = pg.LabelItem(justify="left", size="9pt", color="#9aa0aa")
    # Pairs are placed in a 2-column grid (row = i//2); the status label goes
    # in the first free row below them, so it never collides with a plot.
    status_row = (len(pairs) + 1) // 2
    win.addItem(status_lbl, row=status_row, col=0, colspan=2)
    mode = "sim" if args.simulate else "live"
    status_lbl.setText(f"{mode}: {args.db}  starting...")

    popouts: dict[str, Popout] = {}

    def toggle_popout(name: str) -> None:
        if name in popouts:
            popouts[name].close()  # closeEvent removes it from the dict
            return
        popouts[name] = Popout(name, source, args.bar_seconds, args.window,
                               lambda n: popouts.pop(n, None))

    def on_scene_clicked(ev) -> None:
        for name, plot in plots.items():
            vb = plot.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                toggle_popout(name)
                ev.accept()
                break
    win.scene().sigMouseClicked.connect(on_scene_clicked)

    interval = args.bar_seconds

    def tick() -> None:
        status = source.status()
        latest_t: float | None = None
        for p in pairs:
            name = p.name
            bars = source.get_bars(name)
            plot = plots[name]
            if not bars:
                lines[name].setVisible(False)
                labels[name].setText("waiting for data...")
                labels[name].setPos(args.window - 1, 1.0)
                continue

            t = np.fromiter((b.t + interval / 2.0 for b in bars), dtype=float)
            o = np.fromiter((b.o for b in bars), dtype=float)
            h = np.fromiter((b.h for b in bars), dtype=float)
            l = np.fromiter((b.l for b in bars), dtype=float)
            c = np.fromiter((b.c for b in bars), dtype=float)
            candles[name].setData(t, o, h, l, c, bar_seconds=interval)

            cur_t = float(bars[-1].t)
            latest_t = cur_t if latest_t is None else max(latest_t, cur_t)
            # Follow the latest bar: keep a fixed-width window of `window` bars,
            # with extra space on the right so the spot-price label isn't clipped.
            right_pad = interval * 10
            plot.setXRange(cur_t - (args.window - 1) * interval,
                           cur_t + interval + right_pad, padding=0)

            hi = float(h.max()); lo = float(l.min())
            pad = max((hi - lo) * 0.08, abs(hi) * 1e-5)
            plot.setYRange(lo - pad, hi + pad, padding=0)

            price = float(c[-1])
            hw = candles[name].body_width()  # 2x candle body => half-width
            cur_centre = cur_t + interval / 2.0
            lines[name].setData(x=np.array([cur_centre - hw, cur_centre + hw]),
                                y=np.array([price, price]))
            lines[name].setVisible(True)
            delta = price - float(o[0])
            labels[name].setText(f"{price:.5f}   Δ{delta:+.5f}")
            labels[name].setPos(cur_centre + hw, price)

        if latest_t is not None:
            status_lbl.setText(f"{mode}: {args.db}  t={_iso(latest_t)}")
        else:
            status_lbl.setText(f"{mode}: {status}")
        for pop in list(popouts.values()):
            pop.refresh()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(args.refresh_ms)
    tick()

    def cleanup() -> None:
        timer.stop()
        source.close()
        if sim_stop is not None:
            sim_stop.set()
        if sim_thread is not None:
            sim_thread.join(timeout=3.0)
        if writer_store is not None:
            writer_store.close()
    app.aboutToQuit.connect(cleanup)

    if args.quit_after > 0:
        def grab_and_quit() -> None:
            timer.stop()
            if args.screenshot:
                win.grab().save(args.screenshot, "PNG")
            for p in pairs:
                bars = source.get_bars(p.name)
                if bars:
                    print(f"{p.name}: {len(bars)} bars "
                          f"O={bars[0].o:.5f} H={max(b.h for b in bars):.5f} "
                          f"L={min(b.l for b in bars):.5f} C={bars[-1].c:.5f} "
                          f"t0={_iso(bars[0].t)} tN={_iso(bars[-1].t)}")
                else:
                    print(f"{p.name}: no bars")
            app.quit()
        QtCore.QTimer.singleShot(int(args.quit_after * 1000), grab_and_quit)

    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()