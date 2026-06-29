"""Shared FX pair definitions used by the ingest backend and the GUI viewer.

The pair list is read from fx_pairs.yaml at the project root; DEFAULT_PAIRS
below is the fallback used only when that file is missing or unparseable.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "fx_pairs.yaml"


@dataclass(slots=True)
class Pair:
    name: str            # display name, e.g. "EUR/USD"
    base: str            # contract base currency, e.g. "EUR"
    quote: str           # contract quote currency, e.g. "USD"
    invert: bool = False  # if True, displayed price = 1 / (contract mid)


DEFAULT_PAIRS: list[Pair] = [
    Pair("EUR/USD", "EUR", "USD"),
    Pair("GBP/USD", "GBP", "USD"),
    Pair("USD/TRY", "USD", "TRY"),
    Pair("USD/CAD", "USD", "CAD"),
]

# Indicative starting prices; only used by the --simulate random walk.
SEED_PRICES: dict[str, float] = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2690,
    "USD/TRY": 32.50,
    "USD/CAD": 1.3600,
}


def load_pairs() -> list[Pair]:
    """Read fx_pairs.yaml. Falls back to DEFAULT_PAIRS only if the file is
    missing or unparseable; an empty `pairs:` list is respected as-is.
    """
    if not CONFIG_PATH.exists():
        return list(DEFAULT_PAIRS)
    try:
        with CONFIG_PATH.open() as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        print(f"[pairs] warning: could not parse {CONFIG_PATH}: {e}",
              file=sys.stderr)
        return list(DEFAULT_PAIRS)
    items = data.get("pairs") or []
    return [
        Pair(
            name=it["name"],
            base=it["base"],
            quote=it["quote"],
            invert=it.get("invert", False),
        )
        for it in items
    ]


def parse_pairs(spec: str) -> list[Pair]:
    """Parse 'EUR/USD,GBP/USD' into a list of Pair (invert defaults to False)."""
    out: list[Pair] = []
    for item in spec.split(","):
        item = item.strip()
        if "/" not in item:
            raise SystemExit(f"bad pair {item!r}; use BASE/QUOTE, e.g. EUR/USD")
        base, quote = item.split("/")
        out.append(Pair(item.upper(), base.upper(), quote.upper()))
    return out