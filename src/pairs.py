"""Shared FX pair definitions used by the ingest backend and the GUI viewer."""
from __future__ import annotations

from dataclasses import dataclass


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