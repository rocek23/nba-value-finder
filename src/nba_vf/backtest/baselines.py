"""Baseline semplici contro cui confrontare il modello. Usano SOLO le partite precedenti alla gara.

moneta               p = 0.5 sulle mezze linee (nessuna abilita')
frequenza_ultime20   frequenza osservata nelle ultime 20 partite, con lisciamento di Laplace
poisson_ultime10     Poisson con media delle ultime 10 partite (ignora minuti, sovradispersione e correlazioni)
mercato_consenso     probabilita' media dei libri senza margine (solo dove esistono quote storiche reali)

Le soglie sintetiche (``neutral_lines``) sono generate dallo storico del giocatore, NON dal modello: cosi' tutti i
modelli e le baseline sono valutati sulle stesse linee e il modello non sceglie le linee su cui viene giudicato.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import poisson

from ..model.odds_math import classify_line
from ..model.errors import InvalidLineError

BASELINES = ("moneta", "frequenza_ultime20", "poisson_ultime10")
N_EMP = 20
N_POIS = 10
K_STEPS = (-2, -1, 0, 1, 2)


def neutral_lines(values: Sequence[float], ymax: int) -> list[float]:
    """Mezze linee attorno alla mediana delle ultime partite del giocatore, passo ~ meta' della deviazione standard."""
    v = np.asarray(values[-N_EMP:], float)
    if len(v) == 0:
        return []
    med = int(np.floor(np.median(v)))
    step = max(1, int(round(float(v.std()) / 2)))
    lines = {med + 0.5 + step * k for k in K_STEPS}
    return sorted(x for x in lines if 0.5 <= x <= ymax - 3)


def _split_counts(vals: np.ndarray, line: float) -> tuple[float, float, float]:
    base, is_int = classify_line(line, 10**6)
    lt = float((vals < line).sum())
    eq = float((vals == line).sum()) if is_int else 0.0
    gt = float((vals > line).sum())
    return lt, eq, gt


def p_empirical(values: Sequence[float], line: float) -> tuple[float, float, float]:
    """(P(Y<L), P(Y=L), P(Y>L)) dalle ultime 20 partite con lisciamento di Laplace (+1 per esito possibile)."""
    vals = np.asarray(values[-N_EMP:], float)
    lt, eq, gt = _split_counts(vals, line)
    k = 3 if abs(line - round(line)) < 1e-9 else 2
    n = len(vals) + k
    return (lt + 1) / n, ((eq + 1) / n if k == 3 else 0.0), (gt + 1) / n


def p_poisson(values: Sequence[float], line: float) -> tuple[float, float, float]:
    vals = np.asarray(values[-N_POIS:], float)
    mu = max(float(vals.mean()), 1e-3) if len(vals) else 1.0
    base, is_int = classify_line(line, 10**6)
    if is_int:
        lt = float(poisson.cdf(base - 1, mu)) if base >= 1 else 0.0
        eq = float(poisson.pmf(base, mu))
    else:
        lt, eq = float(poisson.cdf(base, mu)), 0.0
    return lt, eq, max(0.0, 1.0 - lt - eq)


def p_coin(values: Sequence[float], line: float) -> tuple[float, float, float]:
    """Nessuna abilita': vittoria e perdita equiprobabili; il rimborso (linee intere) dalla frequenza storica."""
    if abs(line - round(line)) > 1e-9:
        return 0.5, 0.0, 0.5
    _, eq, _ = p_empirical(values, line)
    return (1 - eq) / 2, eq, (1 - eq) / 2


def baseline_probs(name: str, values: Sequence[float], line: float) -> tuple[float, float, float]:
    """(p_lt, p_eq, p_gt) per la baseline, dato lo storico del giocatore sul mercato."""
    try:
        return {"moneta": p_coin, "frequenza_ultime20": p_empirical, "poisson_ultime10": p_poisson}[name](values, line)
    except InvalidLineError:
        raise
