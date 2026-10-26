"""Distribuzioni predittive per mercato a partire da un ``FittedPlayer``.

- punti / rimbalzi / assist : mistura sui minuti di NegBin2(media = M * rho, alpha)   (analitica)
- triple                    : mistura sui tentativi (NegBin2 sui minuti) di Binomiale o Beta-binomiale (analitica)
- PRA                       : simulazione con copula gaussiana + minuti condivisi        (Monte Carlo)
                              piu' un controllo analitico a componenti condizionatamente indipendenti
Tutte le pmf sono su 0..ymax, con la coda accorpata nell'ultimo bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .config import STATS, ModelConfig
from .distributions import MINUTES_GRID, betabinom_pmf_matrix, cdf_table, nb_pmf_matrix, sample_rows
from .fit import FittedPlayer

ATTEMPTS_MAX = 40  # supporto dei tentativi da tre usato nel modello delle triple
YKEY = {"pts": "points", "reb": "rebounds", "ast": "assists"}


def stat_pmf(fit: FittedPlayer, key: str, ymax: int) -> np.ndarray:
    """P(stat = y) = somma_m P(M = m) * NB(y; media = m * rho, alpha)."""
    mat = nb_pmf_matrix(fit.rho[key] * MINUTES_GRID, fit.alpha[key], ymax)
    pmf = fit.minutes_pmf @ mat
    return pmf / pmf.sum()


def threes_pmf(fit: FittedPlayer, ymax: int) -> np.ndarray | None:
    """Triple segnate: tentativi ~ mistura NB sui minuti, poi segnate | tentativi ~ (beta-)binomiale."""
    if not fit.three_ok or fit.p3 is None:
        return None
    att = stat_pmf(fit, "fg3a", ATTEMPTS_MAX)  # (ATTEMPTS_MAX + 1,)
    cond = betabinom_pmf_matrix(ATTEMPTS_MAX, ymax, fit.p3, fit.icc3 if fit.use_bb else 0.0)
    pmf = att @ cond
    return pmf / pmf.sum()


def _psd_cholesky(corr: np.ndarray) -> np.ndarray:
    """Cholesky di una matrice di correlazione, resa definita positiva (autovalori >= 1e-6)."""
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2.0)
    vals = np.clip(vals, 1e-6, None)
    m = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(m))
    return np.linalg.cholesky(m / np.outer(d, d))


@dataclass
class PraDraws:
    """Numeri casuali comuni (CRN): gli stessi per ogni replica bootstrap, cosi' le differenze tra repliche
    riflettono i parametri e non il rumore Monte Carlo."""

    u_minutes: np.ndarray  # (N,)
    eps: np.ndarray  # (N, 3) normali standard indipendenti

    @classmethod
    def make(cls, n: int, seed: int) -> "PraDraws":
        rng = np.random.default_rng(seed)
        return cls(rng.random(n), rng.standard_normal((n, 3)))


def pra_samples(fit: FittedPlayer, cfg: ModelConfig, draws: PraDraws) -> np.ndarray:
    """Campioni (N, 3) di (pts, reb, ast): minuti condivisi + copula gaussiana sui residui dato M."""
    cdf_m = np.cumsum(fit.minutes_pmf)
    cdf_m[-1] = 1.0
    midx = np.minimum(np.searchsorted(cdf_m, draws.u_minutes, side="left"), len(MINUTES_GRID) - 1)
    u = norm.cdf(draws.eps @ _psd_cholesky(fit.R).T)
    out = np.empty((len(midx), 3), dtype=np.int64)
    for j, key in enumerate(STATS):
        mat = nb_pmf_matrix(fit.rho[key] * MINUTES_GRID, fit.alpha[key], cfg.ymax[YKEY[key]])
        table, ncols = cdf_table(mat)
        out[:, j] = sample_rows(table, ncols, midx, u[:, j])
    return out


def pra_pmf_sim(fit: FittedPlayer, cfg: ModelConfig, draws: PraDraws) -> np.ndarray:
    pra = pra_samples(fit, cfg, draws).sum(axis=1)
    ymax = cfg.ymax["pra"]
    counts = np.bincount(np.minimum(pra, ymax), minlength=ymax + 1).astype(float)
    return counts / counts.sum()


def pra_pmf_independent(fit: FittedPlayer, cfg: ModelConfig) -> np.ndarray:
    """PRA analitica assumendo pts, reb, ast INDIPENDENTI dato M (convoluzione). Serve come controllo:
    differisce dalla simulazione solo per la correlazione residua (la dipendenza via minuti e' inclusa)."""
    ymax = cfg.ymax["pra"]
    mats = [nb_pmf_matrix(fit.rho[k] * MINUTES_GRID, fit.alpha[k], cfg.ymax[YKEY[k]]) for k in STATS]
    out = np.zeros(ymax + 1)
    for i, pm in enumerate(fit.minutes_pmf):
        if pm < 1e-12:
            continue
        conv = np.convolve(np.convolve(mats[0][i], mats[1][i]), mats[2][i])
        if len(conv) > ymax + 1:
            conv = np.concatenate([conv[:ymax], [conv[ymax:].sum()]])
        out[: len(conv)] += pm * conv
    return out / out.sum()


def all_pmfs(fit: FittedPlayer, cfg: ModelConfig, draws: PraDraws) -> dict[str, np.ndarray | None]:
    return {
        "points": stat_pmf(fit, "pts", cfg.ymax["points"]),
        "rebounds": stat_pmf(fit, "reb", cfg.ymax["rebounds"]),
        "assists": stat_pmf(fit, "ast", cfg.ymax["assists"]),
        "threes": threes_pmf(fit, cfg.ymax["threes"]),
        "pra": pra_pmf_sim(fit, cfg, draws),
    }
