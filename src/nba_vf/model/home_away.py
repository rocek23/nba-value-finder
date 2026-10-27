"""Effetto casa/trasferta: stimato su TUTTI i giocatori caricati e usato SOLO se migliora la previsione fuori campione.

Procedura (per ognuna delle statistiche pts, reb, ast, fg3a):
1. Si ordinano le partite nel tempo e si usa il primo ``TRAIN_FRAC`` come periodo di stima, il resto come verifica.
2. Nel periodo di stima, per ogni giocatore con almeno ``MIN_SIDE`` partite in casa e in trasferta, si calcola il rapporto
   media in casa / media in trasferta. Il rapporto complessivo R e' la media pesata dei logaritmi, con intervallo al 99% da bootstrap
   SUI GIOCATORI (non sulle partite: le partite dello stesso giocatore non sono indipendenti).
3. Nel periodo di verifica si confronta l'errore quadratico di due previsioni per ogni partita: la media storica del giocatore
   (base) e la stessa media moltiplicata per R^(1/2) in casa o R^(-1/2) in trasferta. Il test e' sui giocatori: t di Student sulle
   differenze medie per giocatore.
4. L'effetto e' APPLICATO solo se: (a) l'intervallo di R esclude 1, (b) l'errore fuori campione diminuisce, (c) la diminuzione e'
   statisticamente significativa (t >= ``T_MIN``, unilaterale ~95%). Altrimenti non cambia nulla e si registra il motivo.
5. Se applicato, R viene ristimato su tutto lo storico e usato come ``R^(+-1/2)`` (la media storica del giocatore e' circa a meta'
   tra casa e trasferta).

Con pochi giocatori o poche partite il test non ha potenza: in quel caso l'esito corretto e' «non dimostrato», non «applicato».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..models import PlayerGameLog

TARGETS = ("pts", "reb", "ast", "fg3a")
MIN_SIDE = 8
MIN_PLAYERS = 20
TRAIN_FRAC = 0.6
CI_LEVEL = 0.99
T_MIN = 1.645
N_BOOT = 400


@dataclass(frozen=True)
class HomeAwayEffect:
    target: str
    ratio: float | None  # casa / trasferta (ristimato su tutto lo storico se applicato)
    ci: tuple[float, float] | None
    n_players: int
    n_test_games: int
    gain_pct: float | None  # riduzione % dell'errore quadratico fuori campione (positiva = migliora)
    t_stat: float | None
    applied: bool
    reason: str

    @property
    def factor_home(self) -> float:
        return float(self.ratio ** 0.5) if self.applied and self.ratio else 1.0

    @property
    def factor_away(self) -> float:
        return float(self.ratio ** -0.5) if self.applied and self.ratio else 1.0


def _val(log: PlayerGameLog, target: str) -> float:
    return float(getattr(log, target))


def _pooled_log_ratio(per_player: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, np.ndarray]:
    """per_player: [(y, home_flag)] -> (log R pesato, log-rapporti e pesi per il bootstrap come array Nx2)."""
    rows = []
    for y, h in per_player:
        yh, ya = y[h], y[~h]
        if len(yh) < MIN_SIDE or len(ya) < MIN_SIDE or ya.mean() <= 0 or yh.mean() <= 0:
            continue
        w = 1.0 / (1.0 / len(yh) + 1.0 / len(ya))
        rows.append((np.log(yh.mean() / ya.mean()), w))
    arr = np.array(rows, dtype=float).reshape(-1, 2)
    if len(arr) == 0:
        return float("nan"), arr
    return float((arr[:, 0] * arr[:, 1]).sum() / arr[:, 1].sum()), arr


def estimate_home_away(pool: Mapping[object, Sequence[PlayerGameLog]], seed: int = 7,
                       min_players: int = MIN_PLAYERS) -> dict[str, HomeAwayEffect]:
    rng = np.random.default_rng(seed)
    players = []
    for logs in pool.values():
        clean = sorted((l for l in logs if l.minutes and l.minutes > 0 and l.is_home is not None), key=lambda l: (l.game_date, l.game_id))
        if len(clean) >= 2 * MIN_SIDE:
            players.append(clean)
    out: dict[str, HomeAwayEffect] = {}
    if len(players) < min_players:
        reason = f"troppo pochi giocatori con storico casa/trasferta ({len(players)} < {min_players})"
        return {t: HomeAwayEffect(t, None, None, len(players), 0, None, None, False, reason) for t in TARGETS}

    dates = sorted(l.game_date for logs in players for l in logs)
    cutoff = dates[int(TRAIN_FRAC * len(dates))]
    for target in TARGETS:
        train, test = [], []
        for logs in players:
            tr = [l for l in logs if l.game_date < cutoff]
            te = [l for l in logs if l.game_date >= cutoff]
            if len(tr) and len(te):
                train.append(tr)
                test.append(te)
        per_train = [(np.array([_val(l, target) for l in tr]), np.array([bool(l.is_home) for l in tr])) for tr in train]
        log_r, arr = _pooled_log_ratio(per_train)
        n_used = len(arr)
        if n_used < min_players:
            out[target] = HomeAwayEffect(target, None, None, n_used, 0, None, None, False,
                                         f"troppi pochi giocatori con abbastanza partite in casa e in trasferta ({n_used} < {min_players})")
            continue
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n_used, n_used)
            s = arr[idx]
            boots[b] = (s[:, 0] * s[:, 1]).sum() / s[:, 1].sum()
        lo, hi = np.exp(np.quantile(boots, [(1 - CI_LEVEL) / 2, 1 - (1 - CI_LEVEL) / 2]))
        ci = (float(lo), float(hi))
        ratio = float(np.exp(log_r))
        fh, fa = ratio ** 0.5, ratio ** -0.5

        diffs, base_err, n_games = [], [], 0
        for tr, te in zip(train, test):
            base = float(np.mean([_val(l, target) for l in tr]))
            ytest = np.array([_val(l, target) for l in te])
            home = np.array([bool(l.is_home) for l in te])
            adj = base * np.where(home, fh, fa)
            d = (ytest - base) ** 2 - (ytest - adj) ** 2
            diffs.append(float(d.mean()))
            base_err.append(float(((ytest - base) ** 2).mean()))
            n_games += len(te)
        dm = np.array(diffs)
        sd = float(dm.std(ddof=1)) if len(dm) > 1 else 0.0
        t = float(dm.mean() / (sd / np.sqrt(len(dm)))) if sd > 1e-12 else None
        gain = float(100 * dm.mean() / max(np.mean(base_err), 1e-12))

        if ci[0] <= 1.0 <= ci[1]:
            out[target] = HomeAwayEffect(target, ratio, ci, n_used, n_games, gain, t, False,
                                         "l'intervallo al 99% include 1: nessun effetto dimostrabile")
        elif gain <= 0 or t is None or t < T_MIN:
            out[target] = HomeAwayEffect(target, ratio, ci, n_used, n_games, gain, t, False,
                                         "l'effetto sembra esistere ma non migliora la previsione fuori campione in modo significativo: non applicato")
        else:
            full_log_r, _ = _pooled_log_ratio([(np.array([_val(l, target) for l in logs]), np.array([bool(l.is_home) for l in logs])) for logs in players])
            out[target] = HomeAwayEffect(target, float(np.exp(full_log_r)), ci, n_used, n_games, gain, t, True,
                                         "verificato: migliora la previsione fuori campione (errore minore, t >= 1,645)")
    return out
