"""Backtest walk-forward e calibrazione.

Per ogni partita giocata dal giocatore nel periodo di test, il modello vede SOLO le partite con data
strettamente precedente (prior di pool incluso: ristimato a blocchi usando solo partite anteriori al blocco).
Le soglie sono linee SINTETICHE x.5 attorno alla mediana predetta: misurano se le probabilita' del modello
sono calibrate rispetto agli esiti osservati, NON se il modello batte le quote dei bookmaker (servono quote
storiche reali per questo, e non sono disponibili). Nessun risultato qui e' una promessa di rendimento.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..models import Market, PlayerGameLog
from .config import LeaguePrior, ModelConfig
from .empirical import stat_value
from .engine import GameContext, ValueEngine
from .errors import InsufficientDataError
from .odds_math import split_probs
from .prior import estimate_league_prior

DEFAULT_MARKETS = (Market.POINTS, Market.REBOUNDS, Market.ASSISTS, Market.THREES, Market.PRA)
EPS = 1e-12


def synthetic_half_lines(pmf: np.ndarray, k_values=(-2, -1, 0, 1, 2)) -> list[float]:
    """Linee x.5 attorno alla mediana, con passo ~ meta' deviazione standard (almeno 1)."""
    s = np.arange(len(pmf))
    mean = float(pmf @ s)
    sd = float(np.sqrt(pmf @ (s - mean) ** 2))
    median = int(np.searchsorted(np.cumsum(pmf), 0.5 - 1e-12))
    step = max(1, int(round(sd / 2)))
    base = median + 0.5
    lines = [base + step * k for k in k_values]
    return sorted({ln for ln in lines if 0.5 <= ln < len(pmf) - 2})


def walk_forward(
    pool: Mapping[int, Sequence[PlayerGameLog]],
    cfg: ModelConfig | None = None,
    *,
    start_date: date,
    end_date: date | None = None,
    markets: Sequence[Market] = DEFAULT_MARKETS,
    block_days: int = 30,
    min_players: int = 20,
    seed: int = 12345,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restituisce (righe per linea, righe per partita/mercato con PIT). Usa cfg.n_boot = 0 per velocita'."""
    cfg = cfg or ModelConfig(n_boot=0, n_sims_pra=20_000)
    if cfg.n_boot != 0:
        cfg = ModelConfig(**{**cfg.__dict__, "n_boot": 0})
    rng = np.random.default_rng(seed)
    sorted_pool = {pid: sorted(logs, key=lambda l: (l.game_date, l.game_id)) for pid, logs in pool.items()}
    last = end_date or max(l.game_date for logs in sorted_pool.values() for l in logs)
    line_rows: list[dict] = []
    pit_rows: list[dict] = []

    block_start = start_date
    while block_start <= last:
        block_end = block_start + timedelta(days=block_days)
        past = {pid: [l for l in logs if l.game_date < block_start] for pid, logs in sorted_pool.items()}
        try:
            prior = estimate_league_prior(past, cfg, min_players)
        except ValueError:
            prior = LeaguePrior.fallback()
        engine = ValueEngine(cfg, prior)
        for pid, logs in sorted_pool.items():
            for i, game in enumerate(logs):
                if not (block_start <= game.game_date < block_end) or game.game_date > last:
                    continue
                history = [l for l in logs[:i] if l.game_date < game.game_date]
                try:
                    pred = engine.predict(history, GameContext(), intervals=False)
                except InsufficientDataError:
                    continue
                for mk in markets:
                    if mk not in pred.markets:
                        continue
                    pmf = pred.markets[mk].pmf
                    y = int(stat_value(game, mk))
                    cdf = np.cumsum(pmf)
                    yy = min(y, len(pmf) - 1)
                    below = cdf[yy - 1] if yy >= 1 else 0.0
                    pit = below + rng.random() * pmf[yy]  # PIT randomizzata (uniforme se calibrato)
                    pit_rows.append({"mercato": mk.value, "giocatore": pid, "data": game.game_date, "y": y,
                                     "media": float(pmf @ np.arange(len(pmf))), "pit": float(pit),
                                     "partite_storico": len(history)})
                    for ln in synthetic_half_lines(pmf):
                        _, _, p_gt = split_probs(pmf, ln)
                        line_rows.append({"mercato": mk.value, "giocatore": pid, "data": game.game_date,
                                          "linea": ln, "p_over": float(p_gt), "esito_over": int(y > ln), "y": y})
        block_start = block_end
    return pd.DataFrame(line_rows), pd.DataFrame(pit_rows)


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if sel.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(sel.sum()),
                         "p_media_predetta": float(p[sel].mean()), "freq_osservata": float(y[sel].mean())})
    return pd.DataFrame(rows)


def calibration_report(lines_df: pd.DataFrame, pit_df: pd.DataFrame, bins: int = 10) -> dict[str, dict]:
    """Metriche per mercato: Brier, log-loss, tabella di affidabilita', PIT e copertura dell'intervallo 80%."""
    out: dict[str, dict] = {}
    for mk in sorted(lines_df["mercato"].unique()) if len(lines_df) else []:
        d = lines_df[lines_df["mercato"] == mk]
        p = np.clip(d["p_over"].to_numpy(), EPS, 1 - EPS)
        y = d["esito_over"].to_numpy().astype(float)
        pit = pit_df[pit_df["mercato"] == mk]["pit"].to_numpy()
        out[mk] = {
            "n_linee": int(len(d)),
            "n_partite": int(len(pit)),
            "brier": float(np.mean((p - y) ** 2)),
            "brier_riferimento_0.5": 0.25,
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "p_media_predetta": float(p.mean()),
            "freq_osservata": float(y.mean()),
            "pit_media": float(pit.mean()),
            "pit_varianza": float(pit.var()),  # uniforme: 0.5 e 1/12 = 0.0833
            "copertura_intervallo_80": float(np.mean((pit >= 0.1) & (pit <= 0.9))),  # nominale 0.80
            "affidabilita": reliability_table(p, y, bins),
        }
    return out


@dataclass
class LogisticCalibrator:
    """Ricalibrazione logistica logit(p') = a + b * logit(p), con ridge verso (a, b) = (0, 1) (identita').

    Va addestrata su dati fuori campione (walk-forward) e solo su mezze linee (esito binario, niente push)."""

    ridge: float = 5.0
    a: float = 0.0
    b: float = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "LogisticCalibrator":
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, float)
        z = np.log(p / (1 - p))

        def nll(theta):
            a, b = theta
            q = np.clip(1 / (1 + np.exp(-(a + b * z))), EPS, 1 - EPS)
            return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)) + 0.5 * self.ridge * (a * a + (b - 1) ** 2)

        res = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(self.a + self.b * np.log(p / (1 - p)))))
