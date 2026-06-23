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

from pairs import DEFAULT_PAIRS, Pair, parse_pairs
from sim import run_sim
from store import Store, reset_db

UP_COLOR = "#26a69a"
DN_COLOR = "#ef5350"


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

    def setData(self, t, o, h, l, c) -> None:
        self._t = np.asarray(t, dtype=float)
        self._o = np.asarray(o, dtype=float)
        self._h = np.asarray(h, dtype=float)
        self._l = np.asarray(l, dtype=float)
        self._c = np.asarray(c, dtype=float)
        if len(self._t) > 1:
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
        return QtCore.QRectF(
            float(self._t.min()) - 1,
            float(self._l.min()),
            float(self._t.max() - self._t.min()) + 2,
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
    """

    def __init__(self, name: str, source: DbSource, bar_seconds: int,
                 on_close) -> None:
        super().__init__()
        self.name = name
        self.source = source
        self.bar_seconds = bar_seconds
        self._on_close = on_close
        self.setWindowTitle(f"FX Monitor — {name}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        # UTC datetime axis: shows "2026-01-01 00:00:00" at the sim start
        # regardless of the host's local timezone.
        axis = DateAxisItem(orientation="bottom", utcOffset=0)
        self.plot = pg.PlotWidget(title=name, axisItems={"bottom": axis})
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "price")
        self.plot.setLabel("bottom", "time (UTC)")
        # Full mouse interaction: left-drag pan, wheel zoom, right-drag zoom-box,
        # right-click menu (view all / export / axis options).
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.setMenuEnabled(True)
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
        self.show()

    def refresh(self) -> None:
        bars = self.source.get_bars(self.name)
        if not bars:
            return
        t = np.fromiter((b.t for b in bars), dtype=float)
        o = np.fromiter((b.o for b in bars), dtype=float)
        h = np.fromiter((b.h for b in bars), dtype=float)
        l = np.fromiter((b.l for b in bars), dtype=float)
        c = np.fromiter((b.c for b in bars), dtype=float)
        self.candle.setData(t, o, h, l, c)
        cur_t = bars[-1].t
        price = float(c[-1])
        hw = self.candle.body_width()  # 2x candle body width => half-width
        self.line.setData(x=np.array([cur_t - hw, cur_t + hw]),
                          y=np.array([price, price]))
        delta = price - float(o[0])
        self.label.setText(f"{price:.5f}   Δ{delta:+.5f}")
        self.label.setPos(cur_t + hw, price)

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
                    help="comma list BASE/QUOTE (default: EUR/USD,GBP/USD,USD/TRY,USD/CAD)")
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
    pairs = parse_pairs(args.pairs) if args.pairs else DEFAULT_PAIRS

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
    win.addItem(status_lbl, row=2, col=0, colspan=2)
    mode = "sim" if args.simulate else "live"
    status_lbl.setText(f"{mode}: {args.db}  starting...")

    popouts: dict[str, Popout] = {}

    def toggle_popout(name: str) -> None:
        if name in popouts:
            popouts[name].close()  # closeEvent removes it from the dict
            return
        popouts[name] = Popout(name, source, args.bar_seconds,
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

            t = np.fromiter((b.t for b in bars), dtype=float)
            o = np.fromiter((b.o for b in bars), dtype=float)
            h = np.fromiter((b.h for b in bars), dtype=float)
            l = np.fromiter((b.l for b in bars), dtype=float)
            c = np.fromiter((b.c for b in bars), dtype=float)
            candles[name].setData(t, o, h, l, c)

            cur_t = float(bars[-1].t)
            latest_t = cur_t if latest_t is None else max(latest_t, cur_t)
            # Follow the latest bar: keep a fixed-width window of `window` bars.
            plot.setXRange(cur_t - (args.window - 1) * interval,
                           cur_t + interval, padding=0)

            hi = float(h.max()); lo = float(l.min())
            pad = max((hi - lo) * 0.08, abs(hi) * 1e-5)
            plot.setYRange(lo - pad, hi + pad, padding=0)

            price = float(c[-1])
            hw = candles[name].body_width()  # 2x candle body => half-width
            lines[name].setData(x=np.array([cur_t - hw, cur_t + hw]),
                                y=np.array([price, price]))
            lines[name].setVisible(True)
            delta = price - float(o[0])
            labels[name].setText(f"{price:.5f}   Δ{delta:+.5f}")
            labels[name].setPos(cur_t + hw, price)

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