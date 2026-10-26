"""Metriche di previsione e di rendimento, con intervalli di incertezza.

Gli intervalli sono bootstrap PER GIORNATA (si ricampionano giornate intere): le scommesse dello stesso giorno
sono correlate (stessa partita, PRA e punti dello stesso giocatore), quindi ricampionare le singole righe
darebbe intervalli troppo stretti. Il bootstrap copre l'errore di campionamento, non i cambi di regime futuri.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-12


def clip_p(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def log_loss(p, y) -> float:
    p, y = clip_p(p), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")


def wilson(k: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Intervallo di Wilson per una proporzione."""
    if n <= 0:
        return float("nan"), float("nan")
    z = norm.ppf(0.5 + level / 2)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    n: int  # osservazioni
    clusters: int  # giornate

    def excludes(self, x: float) -> bool:
        return bool(np.isfinite(self.lo) and (self.lo > x or self.hi < x))

    def as_dict(self) -> dict[str, float]:
        return {"valore": self.value, "lo": self.lo, "hi": self.hi, "n": self.n, "giornate": self.clusters}


def cluster_bootstrap(
    clusters, arrays: Mapping[str, np.ndarray], stat: Callable[[Mapping[str, np.ndarray]], np.ndarray | float],
    *, B: int = 2000, seed: int = 0, level: float = 0.90,
) -> Estimate:
    """Stima puntuale e intervallo percentile di una statistica funzione di somme per giornata."""
    clusters = np.asarray(clusters)
    n = len(clusters)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, 0)
    codes, inv = np.unique(clusters, return_inverse=True)
    k = len(codes)
    sums = {name: np.bincount(inv, weights=np.asarray(v, float), minlength=k) for name, v in arrays.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        point = float(stat({name: np.asarray(s.sum()) for name, s in sums.items()}))
        if k < 2:
            return Estimate(point, float("nan"), float("nan"), n, k)
        idx = np.random.default_rng(seed).integers(0, k, (B, k))
        boots = np.asarray(stat({name: s[idx].sum(axis=1) for name, s in sums.items()}), float)
    boots = boots[np.isfinite(boots)]
    if len(boots) < 10:
        return Estimate(point, float("nan"), float("nan"), n, k)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return Estimate(point, float(lo), float(hi), n, k)


def mean_ci(values, clusters, **kw) -> Estimate:
    v = np.asarray(values, float)
    return cluster_bootstrap(clusters, {"s": v, "n": np.ones(len(v))}, lambda s: s["s"] / s["n"], **kw)


# --------------------------------------------------------------------------------------------------
# Calibrazione
# --------------------------------------------------------------------------------------------------
def reliability(p, y, bins: int = 10, level: float = 0.95) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        lo, hi = wilson(float(y[sel].sum()), n, level)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": n, "p_predetta": float(p[sel].mean()),
                     "freq_osservata": float(y[sel].mean()), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def expected_calibration_error(p, y, bins: int = 10) -> float:
    t = reliability(p, y, bins)
    if t.empty:
        return float("nan")
    return float(np.sum(t["n"] * np.abs(t["p_predetta"] - t["freq_osservata"])) / t["n"].sum())


def calibration_slope_intercept(p, y) -> tuple[float, float]:
    """Regressione logistica y ~ a + b*logit(p) senza penalita': b=1, a=0 se calibrato (b<1: troppo sicuro di se')."""
    from .calibration import fit_logistic

    return fit_logistic(p, y)


# --------------------------------------------------------------------------------------------------
# Scommesse simulate
# --------------------------------------------------------------------------------------------------
def max_drawdown(profits) -> tuple[float, int]:
    """(massimo drawdown in unita', durata massima in scommesse sotto il massimo precedente). Il capitale parte da 0."""
    x = np.asarray(profits, float)
    if len(x) == 0:
        return 0.0, 0
    cum = np.concatenate([[0.0], np.cumsum(x)])
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    under = dd > 1e-12
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return float(dd.max()), int(longest)


def drawdown_ci(daily_pnl, *, B: int = 2000, seed: int = 0, level: float = 0.90) -> tuple[float, float, float]:
    """Max drawdown osservato e intervallo bootstrap (si ricampionano le giornate di P&L, ordine casuale)."""
    d = np.asarray(daily_pnl, float)
    if len(d) < 5:
        return max_drawdown(d)[0], float("nan"), float("nan")
    obs = max_drawdown(d)[0]
    idx = np.random.default_rng(seed).integers(0, len(d), (B, len(d)))
    cum = np.cumsum(d[idx], axis=1)
    cum = np.concatenate([np.zeros((B, 1)), cum], axis=1)
    dd = (np.maximum.accumulate(cum, axis=1) - cum).max(axis=1)
    lo, hi = np.quantile(dd, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(obs), float(lo), float(hi)


def summarize_bets(df: pd.DataFrame, *, bankroll: float = 100.0, level: float = 0.90, B: int = 2000, seed: int = 0) -> dict:
    """Metriche di un registro di scommesse (colonne: stake, profit, result, date). Stake costante = 1 unita'.

    yield           = profitto / somma delle puntate (rendimento per unita' puntata; coincide con il ROI atteso della Fase 2)
    ROI capitale    = profitto / capitale iniziale (``bankroll``)
    hit rate        = vinte / (vinte + perse), i rimborsi (push) sono esclusi
    """
    n = len(df)
    if n == 0:
        return {"n_scommesse": 0, "giornate": 0}
    df = df.sort_values(["commence_time", "player", "market", "line"], kind="stable") if "commence_time" in df else df
    stake = df["stake"].to_numpy(float)
    profit = df["profit"].to_numpy(float)
    day = df["date"].astype(str).to_numpy()
    res = df["result"].to_numpy()
    wins, losses, pushes = int((res == "win").sum()), int((res == "loss").sum()), int((res == "push").sum())
    y = cluster_bootstrap(day, {"p": profit, "s": stake}, lambda s: s["p"] / s["s"], B=B, seed=seed, level=level)
    p_tot = cluster_bootstrap(day, {"p": profit}, lambda s: s["p"] / bankroll, B=B, seed=seed, level=level)
    hl, hh = wilson(wins, wins + losses, level)
    daily = pd.Series(profit, index=day).groupby(level=0).sum().sort_index()
    dd_obs, dd_lo, dd_hi = drawdown_ci(daily.to_numpy(), B=B, seed=seed, level=level)
    _, longest = max_drawdown(profit)
    return {
        "n_scommesse": n, "giornate": int(len(daily)), "puntato": float(stake.sum()), "profitto": float(profit.sum()),
        "yield": y.value, "yield_lo": y.lo, "yield_hi": y.hi,
        "roi_capitale": p_tot.value, "roi_capitale_lo": p_tot.lo, "roi_capitale_hi": p_tot.hi,
        "hit_rate": wins / (wins + losses) if wins + losses else float("nan"), "hit_lo": hl, "hit_hi": hh,
        "vinte": wins, "perse": losses, "rimborsate": pushes,
        "quota_media": float(df["odds"].mean()),
        "max_drawdown": dd_obs, "max_drawdown_lo": dd_lo, "max_drawdown_hi": dd_hi,
        "durata_max_drawdown_scommesse": longest, "livello_ic": level,
    }
