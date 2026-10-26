"""Conversioni di quote. Tutte le funzioni sono pure e coperte da test."""

from __future__ import annotations

import math


def american_to_decimal(american: float) -> float:
    """+150 -> 2.50, -200 -> 1.50. Le quote americane valide hanno modulo >= 100."""
    value = float(american)
    if not math.isfinite(value) or abs(value) < 100:
        raise ValueError(f"quota americana non valida: {american!r}")
    if value > 0:
        return 1.0 + value / 100.0
    return 1.0 + 100.0 / abs(value)


def parse_decimal_odds(value: float | str) -> float:
    """Valida una quota decimale (> 1.0). Accetta la virgola decimale italiana."""
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        odds = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"quota decimale non valida: {value!r}") from None
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError(f"la quota decimale deve essere > 1.0: {value!r}")
    return odds
