"""Frequenze storiche osservate (hit rate) per una linea: controllo di buon senso, NON una previsione."""

from __future__ import annotations

from typing import Sequence

from ..models import Market, PlayerGameLog

WINDOWS = (5, 10, 20, None)  # None = tutto lo storico disponibile


def stat_value(log: PlayerGameLog, market: Market) -> int:
    return {
        Market.POINTS: log.pts,
        Market.REBOUNDS: log.reb,
        Market.ASSISTS: log.ast,
        Market.THREES: log.fg3m,
        Market.PRA: log.pra,
    }[market]


def hit_rates(logs: Sequence[PlayerGameLog], market: Market, line: float) -> dict[str, dict[str, float | int]]:
    """Per ogni finestra (ultime 5/10/20/tutte): n, over, rimborsi (push), under come conteggi e frequenze."""
    out: dict[str, dict[str, float | int]] = {}
    for w in WINDOWS:
        part = list(logs)[-w:] if w else list(logs)
        n = len(part)
        vals = [stat_value(l, market) for l in part]
        over = sum(v > line for v in vals)
        push = sum(v == line for v in vals)
        under = sum(v < line for v in vals)
        key = f"ultime_{w}" if w else "tutte"
        out[key] = {
            "n": n, "over": over, "push": push, "under": under,
            "freq_over": over / n if n else float("nan"),
            "freq_push": push / n if n else float("nan"),
            "freq_under": under / n if n else float("nan"),
        }
    return out
