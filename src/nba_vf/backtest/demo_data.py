"""Dati SINTETICI multi-stagione per provare il backtest (modalita' demo). Niente qui e' reale.

Storico: giocatori sintetici (nba_vf.model.synthetic) su 3 stagioni con lieve deriva dei parametri veri.
Quote storiche: prezzate da un "mercato" che conosce la distribuzione vera del giocatore (campione grande indipendente)
con rumore per libro e margine del 6%: e' un mercato efficiente, quindi il modello NON dovrebbe battere le quote.
Una piccola quota di prelievi e' deliberatamente POST-INIZIO (prezzata sapendo l'esito) per mostrare che il filtro
anti-leakage li scarta. Ogni quota ha source = "SINTETICO".
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta, timezone

import numpy as np

from ..model.empirical import stat_value
from ..model.synthetic import SOURCE, TruePlayer, generate_player
from ..models import Market, OddsQuote, PlayerGameLog, Side
from ..ui.labels import normalize_name

SEASON_STARTS = {2023: date(2023, 10, 24), 2024: date(2024, 10, 22), 2025: date(2025, 10, 21)}
BOOKS = ("demo_a", "demo_b", "demo_c")
MARGIN = 0.06
DEMO_BT_NOTICE = ("DATI SINTETICI: storico e quote sono inventati e il «mercato» è sintetico. "
                  "I risultati verificano il codice, non la NBA.")


def _truth(rng: np.random.Generator, pid: int, drift: float = 0.0) -> TruePlayer:
    def gam(mean: float, cv: float) -> float:
        k = 1.0 / cv**2
        return float(rng.gamma(k, mean / k))
    return TruePlayer(player_id=pid, name=f"Giocatore Demo {pid:02d}", minutes_mean=float(rng.uniform(22, 36)),
                      rho={"pts": gam(0.45, 0.3), "reb": gam(0.17, 0.5), "ast": gam(0.12, 0.6), "fg3a": gam(0.12, 0.5)},
                      p3=float(rng.beta(0.36 * 300, 0.64 * 300)))


def build_pool(n_players: int = 24, games_per_season: int = 40, seasons=(2023, 2024, 2025), seed: int = 7):
    rng = np.random.default_rng(seed)
    pool: dict[str, list[PlayerGameLog]] = {}
    truths: dict[str, TruePlayer] = {}
    for pid in range(1, n_players + 1):
        truth = _truth(rng, pid)
        logs: list[PlayerGameLog] = []
        for s in seasons:
            t = dataclasses.replace(truth, rho={k: v * float(np.exp(rng.normal(0, 0.04))) for k, v in truth.rho.items()})
            part = generate_player(t, games_per_season, seed=int(rng.integers(1, 2**31 - 1)), start=SEASON_STARTS.get(s, date(s, 10, 22)), season=s)
            logs += [dataclasses.replace(l, game_id=f"BT{pid:02d}-{s}-{i:03d}", team=f"T{pid % 8}", opponent="OPP") for i, l in enumerate(part)]
        key = normalize_name(truth.name)
        pool[key] = logs
        truths[key] = truth
    return pool, truths


def _true_pmf(truth: TruePlayer, market: Market, seed: int, n: int = 6000) -> np.ndarray:
    sample = generate_player(truth, n, seed=seed)
    vals = np.array([stat_value(l, market) for l in sample])
    return np.bincount(vals, minlength=int(vals.max()) + 3) / len(vals)


def build_quotes(pool, truths, *, from_date: date, seed: int = 11, leak_share: float = 0.03, edge: float = 0.0, true_sample: int = 4000) -> list[OddsQuote]:
    """Quote sintetiche per le partite con data >= from_date. ``edge`` > 0 sposta le probabilita' del mercato lontano dal vero (per i test)."""
    rng = np.random.default_rng(seed)
    quotes: list[OddsQuote] = []
    for key, logs in pool.items():
        truth = truths[key]
        pmfs = {m: _true_pmf(truth, m, seed=int(rng.integers(1, 2**31 - 1)), n=true_sample) for m in Market}
        for l in logs:
            if l.game_date < from_date or not l.minutes:
                continue
            start = datetime(l.game_date.year, l.game_date.month, l.game_date.day, 23, 30, tzinfo=timezone.utc)  # 19:30 ET circa
            for m in Market:
                pmf = pmfs[m]
                med = int(np.searchsorted(np.cumsum(pmf), 0.5))
                line = float(med) if rng.random() < 0.2 else med + 0.5
                p_win_over = float(pmf[int(line) + 1:].sum()) if line == int(line) else float(pmf[int(line) + 1:].sum())
                p_push = float(pmf[int(line)]) if line == int(line) else 0.0
                p_over_nv = p_win_over / max(1e-9, 1 - p_push)  # senza rimborso
                y = stat_value(l, m)
                for book in BOOKS:
                    if rng.random() < 0.15:
                        continue
                    shades = [(start - timedelta(minutes=int(rng.integers(90, 240))), False)]
                    if rng.random() < 0.3:
                        shades.append((start - timedelta(minutes=int(rng.integers(10, 60))), False))
                    if rng.random() < leak_share:
                        shades.append((start + timedelta(minutes=int(rng.integers(5, 90))), True))
                    for t, leaky in shades:
                        p = float(np.clip(p_over_nv + rng.normal(0, 0.02) + edge, 0.03, 0.97))
                        if leaky:  # prezzata conoscendo l'esito: se entrasse nel backtest gonfierebbe il ROI
                            p = 0.93 if y > line else 0.07 if y < line else p
                        for side, ps in ((Side.OVER, p), (Side.UNDER, 1 - p)):
                            q = ps * (1 + MARGIN)
                            odds = round(max(1.0 / q, 1.01), 2)
                            quotes.append(OddsQuote(provider="demo", bookmaker=book, event_id=f"EV-{l.game_id}", player_name=l.player_name,
                                                    player_id=None, market=m, line=line, side=side, odds_decimal=odds, fetched_at=t,
                                                    quote_time=t - timedelta(minutes=2), commence_time=start, is_alternate=False, source=SOURCE))
    return quotes


def build_backtest_demo(n_players: int = 24, games_per_season: int = 40, seed: int = 7):
    pool, truths = build_pool(n_players, games_per_season, seed=seed)
    dates = sorted({l.game_date for v in pool.values() for l in v})
    quotes = build_quotes(pool, truths, from_date=dates[int(len(dates) * 0.5)], seed=seed + 4)
    return pool, quotes
