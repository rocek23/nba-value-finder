"""Generatore di dati SINTETICI, usato SOLO per test di recupero dei parametri ed esempi riproducibili.

I numeri qui dentro sono i "parametri veri" di un mondo simulato: servono a verificare che il motore li
recuperi e NON sono prior del modello ne' statistiche NBA reali. Le righe prodotte hanno
``source = "SINTETICO"`` e non vanno mai mescolate a dati live (la modalita' demo deve mostrarlo chiaramente).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
from scipy.stats import nbinom, norm

from ..models import PlayerGameLog

SOURCE = "SINTETICO"


@dataclass(frozen=True)
class TruePlayer:
    player_id: int = 1
    name: str = "Giocatore Sintetico"
    minutes_mean: float = 32.0
    minutes_sd: float = 4.5
    rho: dict[str, float] = field(default_factory=lambda: {"pts": 0.55, "reb": 0.20, "ast": 0.15, "fg3a": 0.14})
    alpha: dict[str, float] = field(default_factory=lambda: {"pts": 0.06, "reb": 0.08, "ast": 0.12, "fg3a": 0.10})
    corr: tuple[tuple[float, ...], ...] = ((1.0, 0.10, 0.15), (0.10, 1.0, 0.0), (0.15, 0.0, 1.0))
    p3: float = 0.37
    icc3: float = 0.0
    b2b_minutes_ratio: float = 1.0  # effetto VERO (simulato) della seconda partita in due giorni sui minuti
    b2b_rate_ratio: float = 1.0
    b2b_share: float = 0.18


def _nb_draw(mu: np.ndarray, alpha: float, u: np.ndarray) -> np.ndarray:
    mu = np.maximum(mu, 1e-9)
    if alpha < 1e-6:
        from scipy.stats import poisson

        return poisson.ppf(u, mu).astype(int)
    k = 1.0 / alpha
    return nbinom.ppf(u, k, k / (k + mu)).astype(int)


def generate_player(
    truth: TruePlayer, n_games: int, seed: int, start: date = date(2025, 10, 22), season: int = 2025
) -> list[PlayerGameLog]:
    rng = np.random.default_rng(seed)
    gaps = np.where(rng.random(n_games) < truth.b2b_share, 1, rng.integers(2, 5, n_games))
    gaps[0] = 0
    dates = [start + timedelta(days=int(c)) for c in np.cumsum(gaps)]
    b2b = gaps == 1

    minutes = rng.normal(truth.minutes_mean, truth.minutes_sd, n_games)
    blowout = rng.random(n_games) < 0.10  # coda sinistra (garbage time)
    minutes = np.where(blowout, minutes - rng.uniform(4, 10, n_games), minutes)
    minutes = np.where(b2b, minutes * truth.b2b_minutes_ratio, minutes)
    minutes = np.clip(np.round(minutes), 8, 48)

    chol = np.linalg.cholesky(np.array(truth.corr))
    u = norm.cdf(rng.standard_normal((n_games, 3)) @ chol.T)
    rate_mult = np.where(b2b, truth.b2b_rate_ratio, 1.0)
    pts = _nb_draw(truth.rho["pts"] * minutes * rate_mult, truth.alpha["pts"], u[:, 0])
    reb = _nb_draw(truth.rho["reb"] * minutes * rate_mult, truth.alpha["reb"], u[:, 1])
    ast = _nb_draw(truth.rho["ast"] * minutes * rate_mult, truth.alpha["ast"], u[:, 2])
    fg3a = _nb_draw(truth.rho["fg3a"] * minutes, truth.alpha["fg3a"], rng.random(n_games))
    if truth.icc3 > 1e-9:
        s = (1.0 - truth.icc3) / truth.icc3
        pg = rng.beta(truth.p3 * s, (1.0 - truth.p3) * s, n_games)
    else:
        pg = np.full(n_games, truth.p3)
    fg3m = rng.binomial(fg3a, pg)

    return [
        PlayerGameLog(
            source=SOURCE, player_id=truth.player_id, player_name=truth.name, game_id=f"S{truth.player_id}-{i}",
            game_date=dates[i], season=season, team="SYN", opponent="OPP", is_home=bool(i % 2),
            minutes=float(minutes[i]), pts=int(pts[i]), reb=int(reb[i]), ast=int(ast[i]),
            fg3m=int(fg3m[i]), fg3a=int(fg3a[i]), fgm=int(fg3m[i]), fga=int(fg3a[i]),  # fgm/fga non usati dal modello
        )
        for i in range(n_games)
    ]


def generate_pool(n_players: int, n_games: int, seed: int) -> dict[int, list[PlayerGameLog]]:
    """Pool di giocatori sintetici con eterogeneita' vera tra giocatori (serve a testare l'empirical Bayes)."""
    rng = np.random.default_rng(seed)
    pool: dict[int, list[PlayerGameLog]] = {}
    for pid in range(1, n_players + 1):
        def g(mean: float, cv: float) -> float:
            shape = 1.0 / cv**2
            return float(rng.gamma(shape, mean / shape))

        truth = TruePlayer(
            player_id=pid,
            name=f"Sintetico {pid}",
            minutes_mean=float(rng.uniform(18, 36)),
            rho={"pts": g(0.45, 0.30), "reb": g(0.17, 0.50), "ast": g(0.12, 0.60), "fg3a": g(0.12, 0.50)},
            p3=float(rng.beta(0.36 * 300, 0.64 * 300)),
        )
        pool[pid] = generate_player(truth, n_games, seed=int(rng.integers(1, 2**31 - 1)))
    return pool
