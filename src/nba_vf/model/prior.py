"""Stima dei prior di popolazione (empirical Bayes) a partire da un pool di giocatori.

Serve a rendere lo shrinkage ancorato ai dati: la media e la forza del prior sono stimate da come i giocatori
differiscono davvero tra loro, non fissate a mano. Serve un pool reale (molti giocatori della stessa lega);
con meno di ``min_players`` giocatori solleva ValueError e il chiamante deve usare ``LeaguePrior.fallback()``.

Metodo, per ogni tasso per minuto (pts, reb, ast, fg3a):
    rho_p  = somma(y) / somma(minuti)                         tasso del giocatore p
    v_p    = varianza di campionamento del rapporto (stimatore a linearizzazione, include la sovradispersione)
    tau^2  = max(Var_p(rho_p) - media_p(v_p), eps)            varianza vera tra giocatori
    prior Gamma con media rho0 = somma(y)/somma(minuti) e varianza tau^2  ->  T0 = rho0 / tau^2 minuti
Per la percentuale da tre: analogo con prior Beta, n0 + 1 = p0 (1 - p0) / tau^2.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..models import PlayerGameLog
from .config import RATE_KEYS, STATS, LeaguePrior, ModelConfig
from .distributions import nb_alpha_moment, weighted_corr
from .fit import build_inputs

T0_BOUNDS = (30.0, 5000.0)  # minuti equivalenti: limiti cautelativi sul prior stimato
N0_BOUNDS = (2.0, 5000.0)  # tentativi equivalenti per la percentuale da tre
MIN_ATTEMPTS_FOR_POOL_3PT = 20


def estimate_league_prior(
    pool: Mapping[int, Sequence[PlayerGameLog]], cfg: ModelConfig | None = None, min_players: int = 20
) -> LeaguePrior:
    cfg = cfg or ModelConfig()
    per_player: list[dict] = []
    for logs in pool.values():
        logs = [l for l in logs if l.minutes > 0]
        if len(logs) < cfg.min_games:
            continue
        inp = build_inputs(logs, cfg)
        mask = inp.m >= cfg.min_minutes_rate
        if int(mask.sum()) < cfg.min_games:
            continue
        mr = inp.m[mask]
        ones = np.ones(len(mr))
        entry: dict = {"minutes": float(mr.sum()), "n": int(len(mr))}
        resid_cols = []
        ys = {"pts": inp.pts[mask], "reb": inp.reb[mask], "ast": inp.ast[mask], "fg3a": inp.fg3a[mask]}
        for key in RATE_KEYS:
            y = ys[key]
            rate = float(y.sum() / mr.sum())
            resid = y - rate * mr
            n = len(mr)
            var = float((n / (n - 1)) * np.sum(resid**2) / mr.sum() ** 2)
            mu = rate * mr
            a_hat = nb_alpha_moment(y, mu, ones)
            entry[key] = {"sum_y": float(y.sum()), "rate": rate, "var": var, "alpha": a_hat}
            if key in STATS:
                v = mu + a_hat * mu * mu
                resid_cols.append(np.where(v > 0, (y - mu) / np.sqrt(np.where(v > 0, v, 1.0)), 0.0))
        entry["corr"] = weighted_corr(np.column_stack(resid_cols), ones)
        entry["sum_x"] = float(np.minimum(inp.fg3m[mask], inp.fg3a[mask]).sum())
        entry["sum_a"] = float(inp.fg3a[mask].sum())
        per_player.append(entry)

    if len(per_player) < min_players:
        raise ValueError(
            f"pool insufficiente: {len(per_player)} giocatori idonei, ne servono almeno {min_players}. "
            "Usa LeaguePrior.fallback()."
        )

    rho0: dict[str, float] = {}
    strength: dict[str, float] = {}
    alpha0: dict[str, float] = {}
    for key in RATE_KEYS:
        total_y = sum(p[key]["sum_y"] for p in per_player)
        total_m = sum(p["minutes"] for p in per_player)
        rates = np.array([p[key]["rate"] for p in per_player])
        sampling = np.array([p[key]["var"] for p in per_player])
        base = total_y / total_m
        tau2 = max(float(np.var(rates, ddof=1) - sampling.mean()), 1e-12)
        rho0[key] = float(base)
        strength[key] = float(np.clip(base / tau2, *T0_BOUNDS))
        alpha0[key] = float(np.median([p[key]["alpha"] for p in per_player]))

    with_att = [p for p in per_player if p["sum_a"] >= MIN_ATTEMPTS_FOR_POOL_3PT]
    p3_prior = None
    if len(with_att) >= min_players:
        tot_x = sum(p["sum_x"] for p in with_att)
        tot_a = sum(p["sum_a"] for p in with_att)
        p0 = tot_x / tot_a
        pj = np.array([p["sum_x"] / p["sum_a"] for p in with_att])
        sampling = np.array([p0 * (1 - p0) / p["sum_a"] for p in with_att])
        tau2 = max(float(np.var(pj, ddof=1) - sampling.mean()), 1e-8)
        n0 = float(np.clip(p0 * (1 - p0) / tau2 - 1.0, *N0_BOUNDS))
        p3_prior = (p0 * n0, (1 - p0) * n0)

    corr = np.mean([p["corr"] for p in per_player], axis=0)
    return LeaguePrior(
        source="pool",
        n_players=len(per_player),
        rho0=rho0,
        rate_strength_minutes=strength,
        alpha0=alpha0,
        p3_prior=p3_prior,
        resid_corr=corr,
    )
