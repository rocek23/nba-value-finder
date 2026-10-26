"""Predizione per mercato: pmf centrale + repliche bootstrap, con prezzatura (quota equa, ROI, EV)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models import Market, Side
from .odds_math import (
    SideProbs,
    breakeven_win_prob,
    expected_roi,
    expected_value,
    fair_odds,
    implied_probability,
    side_probs_arrays,
)

CONDITION_NOTE = "probabilita' condizionate al fatto che il giocatore scenda in campo"


def _interval(values: np.ndarray, ci: float) -> tuple[float, float]:
    lo, hi = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    q = np.quantile(values, [lo, hi], method="nearest")  # 'nearest': gestisce anche valori infiniti
    return float(q[0]), float(q[1])


@dataclass(frozen=True)
class Valuation:
    market: str
    side: str
    line: float
    decimal_odds: float
    p_win: float
    p_push: float
    p_loss: float
    p_win_ci: tuple[float, float] | None
    implied_prob_raw: float  # 1 / quota (grezza)
    breakeven_p_win: float  # (1 - p_push) / quota
    fair_odds: float  # (1 - p_push) / p_win
    fair_odds_ci: tuple[float, float] | None
    roi: float  # per unita' di stake
    roi_ci: tuple[float, float] | None
    edge: float  # p_win - breakeven_p_win
    ci_level: float
    flags: tuple[str, ...] = ()

    def ev(self, stake: float) -> float:
        """Valore atteso in denaro per lo stake indicato (stake * ROI)."""
        return float(expected_value(stake, self.roi))

    @property
    def verdict(self) -> str:
        """Etichetta prudente. Non indica mai certezza: copre solo l'incertezza di stima dei parametri."""
        if self.roi_ci is None:
            return "EV stimato positivo, intervallo non calcolato" if self.roi > 0 else "EV stimato non positivo"
        if self.roi_ci[0] > 0:
            return "EV stimato positivo anche nel limite inferiore dell'intervallo"
        if self.roi > 0:
            return "EV stimato positivo ma l'intervallo di incertezza include lo zero"
        return "EV stimato non positivo"

    def as_dict(self, stake: float | None = None) -> dict:
        d = {
            "mercato": self.market,
            "lato": self.side,
            "linea": self.line,
            "quota": self.decimal_odds,
            "p_vittoria": self.p_win,
            "p_rimborso": self.p_push,
            "p_perdita": self.p_loss,
            "p_vittoria_ci": self.p_win_ci,
            "prob_implicita_grezza": self.implied_prob_raw,
            "p_vittoria_pareggio": self.breakeven_p_win,
            "quota_equa": self.fair_odds,
            "quota_equa_ci": self.fair_odds_ci,
            "roi_atteso": self.roi,
            "roi_ci": self.roi_ci,
            "edge": self.edge,
            "esito": self.verdict,
        }
        if stake is not None:
            d["ev_denaro"] = self.ev(stake)
        return d


@dataclass
class MarketPrediction:
    market: Market
    pmf: np.ndarray  # (K,) pmf centrale
    reps: np.ndarray | None  # (B, K) pmf delle repliche bootstrap (None se n_boot = 0)
    ci_level: float
    flags: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=lambda: [CONDITION_NOTE])

    # ---- sintesi ------------------------------------------------------------------------------
    @property
    def support(self) -> np.ndarray:
        return np.arange(len(self.pmf))

    @property
    def mean(self) -> float:
        return float(self.pmf @ self.support)

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.pmf @ (self.support - self.mean) ** 2))

    def quantile(self, q: float) -> int:
        """Il piu' piccolo y con P(Y <= y) >= q."""
        return int(np.searchsorted(np.cumsum(self.pmf), q - 1e-12, side="left"))

    @property
    def median(self) -> int:
        return self.quantile(0.5)

    def mean_ci(self) -> tuple[float, float] | None:
        if self.reps is None:
            return None
        return _interval(self.reps @ self.support, self.ci_level)

    # ---- probabilita' per linea ---------------------------------------------------------------
    def side_probs(self, line: float, side: Side | str) -> SideProbs:
        w, p, l = side_probs_arrays(self.pmf, line, side)
        return SideProbs(float(w), float(p), float(l))

    def line_table(self, lines: list[float] | None = None) -> pd.DataFrame:
        """Probabilita' Over/Under (con push separato) per ogni soglia."""
        if lines is None:
            top = int(np.searchsorted(np.cumsum(self.pmf), 0.9995)) + 1
            lines = [x / 2.0 for x in range(1, 2 * min(top, len(self.pmf) - 2) + 1)]
        rows = []
        for ln in lines:
            ov = self.side_probs(ln, Side.OVER)
            row = {
                "linea": ln,
                "p_over": ov.p_win,
                "p_push": ov.p_push,
                "p_under": ov.p_loss,
                "quota_equa_over": float(fair_odds(ov.p_win, ov.p_push)),
                "quota_equa_under": float(fair_odds(ov.p_loss, ov.p_push)),
            }
            if self.reps is not None:
                w, _, _ = side_probs_arrays(self.reps, ln, "over")
                row["p_over_lo"], row["p_over_hi"] = _interval(w, self.ci_level)
            rows.append(row)
        return pd.DataFrame(rows)

    # ---- prezzatura -----------------------------------------------------------------------------
    def price(self, line: float, side: Side | str, decimal_odds: float) -> Valuation:
        sp = self.side_probs(line, side)
        d = float(decimal_odds)
        p_win_ci = fo_ci = roi_ci = None
        if self.reps is not None:
            w, p, _ = side_probs_arrays(self.reps, line, side)
            p_win_ci = _interval(w, self.ci_level)
            fo_ci = _interval(fair_odds(w, p), self.ci_level)
            roi_ci = _interval(expected_roi(w, p, d), self.ci_level)
        be = float(breakeven_win_prob(d, sp.p_push))
        return Valuation(
            market=self.market.value,
            side=getattr(side, "value", side),
            line=float(line),
            decimal_odds=d,
            p_win=sp.p_win,
            p_push=sp.p_push,
            p_loss=sp.p_loss,
            p_win_ci=p_win_ci,
            implied_prob_raw=float(implied_probability(d)),
            breakeven_p_win=be,
            fair_odds=float(fair_odds(sp.p_win, sp.p_push)),
            fair_odds_ci=fo_ci,
            roi=float(expected_roi(sp.p_win, sp.p_push, d)),
            roi_ci=roi_ci,
            edge=sp.p_win - be,
            ci_level=self.ci_level,
            flags=self.flags,
        )
