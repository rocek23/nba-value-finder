"""Stima dei parametri del giocatore a partire dalle sue partite storiche.

Struttura del modello (vedi docs/MODEL.md):

    M            ~ distribuzione dei minuti (KDE pesata dei minuti storici, media proiettata)
    stat | M     ~ NegBin2(media = M * rho_stat, dispersione alpha_stat)       stat in {pts, reb, ast}
    3PA  | M     ~ NegBin2(media = M * rho_3a,   dispersione alpha_3a)
    3PM  | 3PA   ~ Binomiale(3PA, p3)   oppure Beta-binomiale(3PA, p3, icc3) se il test LR lo giustifica
    (pts, reb, ast) | M  : dipendenza residua da una copula gaussiana con matrice R (shrinkage verso il prior/0)

rho = tasso per minuto con shrinkage; alpha = sovradispersione condizionata ai minuti.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Sequence

import numpy as np

from ..models import PlayerGameLog
from .config import RATE_KEYS, STATS, LeaguePrior, ModelConfig
from .distributions import (
    MINUTES_GRID,
    effective_n,
    fit_icc,
    minutes_pmf,
    nb_alpha_moment,
    recency_weights,
    weighted_corr,
)
from .errors import InsufficientDataError


@dataclass
class FitInputs:
    """Array paralleli (ordine cronologico) per un campione di partite, con i pesi gia' associati."""

    m: np.ndarray
    pts: np.ndarray
    reb: np.ndarray
    ast: np.ndarray
    fg3m: np.ndarray
    fg3a: np.ndarray
    w_min: np.ndarray
    w_rate: np.ndarray
    w_pct: np.ndarray

    def take(self, idx: np.ndarray) -> "FitInputs":
        return FitInputs(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def __len__(self) -> int:
        return len(self.m)


def build_inputs(logs: Sequence[PlayerGameLog], cfg: ModelConfig) -> FitInputs:
    """Costruisce gli array dalle partite (devono essere ordinate per data crescente e con minuti > 0)."""
    n = len(logs)
    return FitInputs(
        m=np.array([l.minutes for l in logs], dtype=float),
        pts=np.array([l.pts for l in logs], dtype=float),
        reb=np.array([l.reb for l in logs], dtype=float),
        ast=np.array([l.ast for l in logs], dtype=float),
        fg3m=np.array([l.fg3m for l in logs], dtype=float),
        fg3a=np.array([l.fg3a for l in logs], dtype=float),
        w_min=recency_weights(n, cfg.half_life_minutes),
        w_rate=recency_weights(n, cfg.half_life_rates),
        w_pct=recency_weights(n, cfg.half_life_pct),
    )


@dataclass
class FittedPlayer:
    minutes_pmf: np.ndarray
    minutes_mean: float
    minutes_sd: float
    rho: dict[str, float]
    alpha: dict[str, float]
    p3: float | None
    icc3: float
    use_bb: bool
    icc_pvalue: float | None
    three_ok: bool
    R: np.ndarray
    n_games: int
    n_rate_games: int
    n_eff: float
    prior_source: str
    flags: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float | str | None]:
        out: dict[str, float | str | None] = {
            "minuti_proiettati": round(self.minutes_mean, 2),
            "minuti_sd_predittiva": round(self.minutes_sd, 2),
            "prior": self.prior_source,
            "partite": self.n_games,
            "partite_efficaci": round(self.n_eff, 1),
            "p3": None if self.p3 is None else round(self.p3, 4),
            "icc3": round(self.icc3, 4),
            "modello_triple": "beta-binomiale" if self.use_bb else "binomiale",
            "icc_pvalue": None if self.icc_pvalue is None else round(self.icc_pvalue, 4),
        }
        for key in RATE_KEYS:
            out[f"rho_{key}"] = round(self.rho[key], 5)
            out[f"alpha_{key}"] = round(self.alpha[key], 4)
        return out


def ratio_with_prior(y: np.ndarray, m: np.ndarray, w: np.ndarray, rho0: float, strength: float) -> float:
    """Stimatore a rapporto pesato con prior Gamma: (sum w y + rho0*T0) / (sum w m + T0).

    ``strength`` (T0) e' il prior espresso in minuti equivalenti di dati; T0 = 0 -> nessuno shrinkage."""
    return float((np.sum(w * y) + rho0 * strength) / (np.sum(w * m) + strength))


def fit_arrays(
    inp: FitInputs,
    cfg: ModelConfig,
    prior: LeaguePrior,
    *,
    minutes_scale: float = 1.0,
    minutes_override: float | None = None,
    rate_scales: dict[str, float] | None = None,
    use_bb: bool | None = None,
) -> FittedPlayer:
    """Stima il modello su un campione (anche un ricampionamento bootstrap).

    ``minutes_scale`` / ``rate_scales`` sono gli aggiustamenti contestuali verificati (context.py), 1.0 =
    modello base. ``minutes_override`` e' un valore MANUALE (es. minutaggio atteso per un'assenza nota):
    non e' un coefficiente stimato. ``use_bb`` fissa la scelta binomiale/beta-binomiale (usato dal bootstrap
    per non rifare il test a ogni replica)."""
    n = len(inp)
    flags: list[str] = []

    # ---- minuti ---------------------------------------------------------------------------------
    wm = inp.w_min
    mean_w = float(np.average(inp.m, weights=wm))
    var_w = float(np.average((inp.m - mean_w) ** 2, weights=wm))
    se2 = var_w / max(effective_n(wm), 1.0)  # varianza della media proiettata
    target = float(minutes_override) if minutes_override is not None else mean_w * minutes_scale
    pm = minutes_pmf(inp.m, wm, target, se2, MINUTES_GRID, cfg.minutes_bandwidth_floor)
    m_mean = float(np.sum(pm * MINUTES_GRID))
    m_sd = float(np.sqrt(np.sum(pm * (MINUTES_GRID - m_mean) ** 2)))

    # ---- tassi per minuto, dispersione, residui --------------------------------------------------
    mask = inp.m >= cfg.min_minutes_rate
    if int(mask.sum()) < 5:
        raise InsufficientDataError("meno di 5 partite con minuti sufficienti per stimare i tassi")
    mr = inp.m[mask]
    wr = inp.w_rate[mask]
    n_eff = effective_n(wr)
    ys = {"pts": inp.pts[mask], "reb": inp.reb[mask], "ast": inp.ast[mask], "fg3a": inp.fg3a[mask]}

    rho: dict[str, float] = {}
    alpha: dict[str, float] = {}
    resid_cols: list[np.ndarray] = []
    for key in RATE_KEYS:
        y = ys[key]
        if prior.is_pool and prior.rho0 is not None and prior.rate_strength_minutes is not None:
            rho0, strength = prior.rho0[key], prior.rate_strength_minutes[key]
        else:  # nessun numero esterno: ritorno verso il tasso non pesato del giocatore
            rho0, strength = float(y.sum() / mr.sum()), cfg.fallback_rate_prior_minutes
        r_hat = ratio_with_prior(y, mr, wr, rho0, strength)
        mu = r_hat * mr
        a_hat = nb_alpha_moment(y, mu, wr)
        if prior.is_pool and prior.alpha0 is not None:
            kappa = cfg.alpha_prior_games
            a = (n_eff * a_hat + kappa * prior.alpha0[key]) / (n_eff + kappa)
        else:
            a = a_hat
        rho[key] = r_hat * (rate_scales or {}).get(key, 1.0)
        alpha[key] = float(a)
        if key in STATS:
            var = mu + a * mu * mu
            resid_cols.append(np.where(var > 0, (y - mu) / np.sqrt(np.where(var > 0, var, 1.0)), 0.0))

    # ---- dipendenza residua tra pts, reb, ast (data M) -------------------------------------------
    r_hat_mat = weighted_corr(np.column_stack(resid_cols), wr)
    target_corr = prior.resid_corr if (prior.is_pool and prior.resid_corr is not None) else np.eye(3)
    lam = n_eff / (n_eff + cfg.corr_shrink_games)
    corr = lam * r_hat_mat + (1.0 - lam) * target_corr
    corr = (corr + corr.T) / 2.0

    # ---- triple: percentuale e scelta binomiale / beta-binomiale ---------------------------------
    x = np.minimum(inp.fg3m[mask], inp.fg3a[mask])
    att = inp.fg3a[mask]
    wp = inp.w_pct[mask]
    three_ok = bool(att.sum() >= cfg.min_attempts_3pt)
    p3: float | None = None
    icc = 0.0
    use = False
    pvalue: float | None = None
    if three_ok:
        a0, b0 = prior.p3_prior if (prior.is_pool and prior.p3_prior is not None) else (0.5, 0.5)
        p3 = float(np.clip((np.sum(wp * x) + a0) / (np.sum(wp * att) + a0 + b0), 1e-4, 1.0 - 1e-4))
        res = fit_icc(x, att, p3, cfg.icc_max)
        pvalue = res.p_value
        use = bool(use_bb) if use_bb is not None else bool(res.p_value < cfg.icc_test_pvalue and res.rho > 0)
        if use:
            icc = res.rho * res.n_games / (res.n_games + cfg.icc_shrink_games)
    else:
        flags.append("triple_non_prezzate_tentativi_insufficienti")

    if n < cfg.low_sample_games:
        flags.append("campione_ridotto")
    if not prior.is_pool:
        flags.append("prior_fallback_nessun_pool")
    if minutes_override is not None:
        flags.append("minuti_manuali")

    return FittedPlayer(
        minutes_pmf=pm,
        minutes_mean=m_mean,
        minutes_sd=m_sd,
        rho=rho,
        alpha=alpha,
        p3=p3,
        icc3=float(icc),
        use_bb=use and icc > 0,
        icc_pvalue=pvalue,
        three_ok=three_ok,
        R=corr,
        n_games=n,
        n_rate_games=int(mask.sum()),
        n_eff=float(n_eff),
        prior_source="pool" if prior.is_pool else "fallback",
        flags=flags,
    )
