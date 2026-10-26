"""Aggiustamenti di contesto, applicati SOLO se supportati da verifica storica del giocatore stesso.

Fattore gestito: seconda partita in giorni consecutivi ("back-to-back", il giocatore ha giocato il giorno
prima secondo il suo storico). Effetti stimati come RAPPORTI (condizionato / base):
    - minuti:            media minuti b2b / media minuti non-b2b
    - tassi per minuto:  (somma y / somma minuti) b2b / non-b2b, per pts, reb, ast, fg3a

Un effetto viene applicato solo se TUTTE queste condizioni sono vere:
    1. almeno ``MIN_COND`` partite nel gruppo condizionato e ``MIN_BASE`` nel gruppo base;
    2. l'intervallo bootstrap al 99% del rapporto esclude 1;
    3. coerenza di segno tra le due meta' cronologiche del gruppo condizionato (>= ``MIN_HALF`` partite per meta');
e viene poi ridotto (shrinkage) con esponente n_cond / (n_cond + SHRINK_N). Altrimenti l'effetto e' registrato
con il motivo del rifiuto e NON e' usato.

Non esistono coefficienti per avversari specifici: l'avversario e' solo registrato nel contesto.
Le assenze dei compagni NON sono modellate: l'unico modo di tenerne conto e' ``minutes_override`` manuale,
sempre segnalato e mai stimato.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..models import PlayerGameLog
from .config import RATE_KEYS, ModelConfig

MIN_COND = 10
MIN_BASE = 10
MIN_HALF = 4
CI_LEVEL = 0.99
SHRINK_N = 10.0
N_BOOT = 400
TARGETS = ("minutes",) + RATE_KEYS


@dataclass(frozen=True)
class ContextEffect:
    factor: str
    target: str
    n_cond: int
    n_base: int
    ratio: float | None
    ci: tuple[float, float] | None
    applied: bool
    shrunk_ratio: float  # 1.0 se non applicato
    reason: str


def rest_days(logs: Sequence[PlayerGameLog]) -> list[int | None]:
    """Giorni dalla partita precedente del giocatore (None per la prima)."""
    out: list[int | None] = [None]
    for prev, cur in zip(logs[:-1], logs[1:]):
        out.append((cur.game_date - prev.game_date).days)
    return out


def b2b_flags(logs: Sequence[PlayerGameLog]) -> np.ndarray:
    rd = rest_days(logs)
    return np.array([d == 1 for d in rd], dtype=bool)


def _stat(target: str, m: np.ndarray, y: np.ndarray | None) -> float:
    if target == "minutes":
        return float(m.mean())
    keep = m >= 5.0
    return float(y[keep].sum() / max(m[keep].sum(), 1e-9))


def _ratio(target: str, mc, yc, mb, yb) -> float:
    return _stat(target, mc, yc) / max(_stat(target, mb, yb), 1e-12)


def estimate_effects(
    logs: Sequence[PlayerGameLog], cfg: ModelConfig, flags: np.ndarray | None = None, seed: int | None = None
) -> dict[str, ContextEffect]:
    """Stima (e verifica) l'effetto back-to-back su minuti e tassi. ``logs`` in ordine cronologico."""
    flags = b2b_flags(logs) if flags is None else np.asarray(flags, dtype=bool)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    m = np.array([l.minutes for l in logs], dtype=float)
    cols = {
        "pts": np.array([l.pts for l in logs], dtype=float),
        "reb": np.array([l.reb for l in logs], dtype=float),
        "ast": np.array([l.ast for l in logs], dtype=float),
        "fg3a": np.array([l.fg3a for l in logs], dtype=float),
    }
    cond_idx = np.flatnonzero(flags)
    base_idx = np.flatnonzero(~flags)
    n_c, n_b = len(cond_idx), len(base_idx)
    out: dict[str, ContextEffect] = {}

    for target in TARGETS:
        y = None if target == "minutes" else cols[target]

        def sub(idx):
            return m[idx], (None if y is None else y[idx])

        if n_c < MIN_COND or n_b < MIN_BASE:
            out[target] = ContextEffect(
                "back_to_back", target, n_c, n_b, None, None, False, 1.0,
                f"campione insufficiente (condizionate {n_c} < {MIN_COND} o base {n_b} < {MIN_BASE})",
            )
            continue
        mc, yc = sub(cond_idx)
        mb, yb = sub(base_idx)
        ratio = _ratio(target, mc, yc, mb, yb)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            ic = rng.integers(0, n_c, n_c)
            ib = rng.integers(0, n_b, n_b)
            boots[b] = _ratio(target, mc[ic], None if yc is None else yc[ic], mb[ib], None if yb is None else yb[ib])
        lo, hi = np.quantile(boots, [(1 - CI_LEVEL) / 2, 1 - (1 - CI_LEVEL) / 2])
        ci = (float(lo), float(hi))
        if lo <= 1.0 <= hi:
            out[target] = ContextEffect("back_to_back", target, n_c, n_b, ratio, ci, False, 1.0,
                                        "intervallo al 99% include 1: nessun effetto dimostrabile")
            continue
        half = n_c // 2
        ok = half >= MIN_HALF
        if ok:
            signs = []
            for part in (cond_idx[:half], cond_idx[half:]):
                mp, yp = sub(part)
                signs.append(np.sign(np.log(_ratio(target, mp, yp, mb, yb))))
            ok = signs[0] == signs[1] == np.sign(np.log(ratio))
        if not ok:
            out[target] = ContextEffect("back_to_back", target, n_c, n_b, ratio, ci, False, 1.0,
                                        "segno non coerente tra le due meta' dello storico")
            continue
        shrunk = float(np.exp(np.log(ratio) * n_c / (n_c + SHRINK_N)))
        out[target] = ContextEffect("back_to_back", target, n_c, n_b, ratio, ci, True, shrunk,
                                    "verificato sullo storico del giocatore (CI 99% esclude 1, segno coerente)")
    return out
