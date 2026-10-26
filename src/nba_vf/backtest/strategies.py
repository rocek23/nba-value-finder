"""Strategie simulate sulle SOLE righe con quota reale pre-partita. Puntata fissa, nessun reinvestimento.

valore:<modello>   per ogni (giocatore, partita, mercato) si sceglie la riga (linea, lato, libro) con EV massimo secondo
                   il modello e si scommette se EV >= soglia. EV = p_vittoria*quota + p_rimborso - 1 (push incluso),
                   con la quota REALE del libro, quindi il margine e' gia' pagato. Nessun libro e' scelto guardando l'esito.
sempre_over/under  strategia ingenua senza modello: lato fisso sulla linea con quota piu' vicina a 2.0 (proxy della linea
                   principale). Misura quanto costa il margine a chi scommette a caso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model.odds_math import expected_roi

BET_COLUMNS = ["strategy", "phase", "player", "game_id", "date", "season", "commence_time", "market", "line", "side", "book", "odds",
               "p_win", "p_push", "ev", "stake", "result", "profit"]
GROUP = ["phase", "player", "game_id", "market"]


def _profit(result: pd.Series, odds: pd.Series, stake: float) -> pd.Series:
    return np.where(result == "win", stake * (odds - 1.0), np.where(result == "push", 0.0, -stake))


def _finish(sel: pd.DataFrame, strategy: str, stake: float) -> pd.DataFrame:
    out = pd.DataFrame({
        "strategy": strategy, "phase": sel["phase"], "player": sel["player"], "game_id": sel["game_id"], "date": sel["date"], "season": sel["season"],
        "commence_time": sel["commence_time"], "market": sel["market"], "line": sel["line"], "side": sel["side"], "book": sel["book"],
        "odds": sel["odds"], "p_win": sel["p_win"], "p_push": sel["p_push"], "ev": sel["ev"], "stake": stake, "result": sel["result"],
    })
    out["profit"] = _profit(out["result"], out["odds"], stake)
    return out[BET_COLUMNS].reset_index(drop=True)


def value_bets(rows: pd.DataFrame, model: str, threshold: float, stake: float = 1.0, strategy: str | None = None, phase: str | None = None) -> pd.DataFrame:
    d = rows[(rows["model"] == model) & (rows["line_source"] == "quota_reale")]
    if phase is not None:
        d = d[d["phase"] == phase]
    if d.empty:
        return pd.DataFrame(columns=BET_COLUMNS)
    d = d.copy()
    d["ev"] = expected_roi(d["p_win"].to_numpy(), d["p_push"].to_numpy(), d["odds"].to_numpy())
    d = d.sort_values(["ev", "odds", "book", "line", "side"], ascending=[False, False, True, True, True], kind="stable")
    best = d.drop_duplicates(GROUP, keep="first")
    best = best[best["ev"] >= threshold]
    return _finish(best, strategy or f"valore:{model}", stake)


def naive_bets(rows: pd.DataFrame, side: str, stake: float = 1.0, phase: str | None = None, model: str = "modello") -> pd.DataFrame:
    d = rows[(rows["model"] == model) & (rows["line_source"] == "quota_reale") & (rows["side"] == side)]
    if phase is not None:
        d = d[d["phase"] == phase]
    if d.empty:
        return pd.DataFrame(columns=BET_COLUMNS)
    d = d.copy()
    d["ev"] = np.nan
    d["_dist"] = (d["odds"] - 2.0).abs()
    d = d.sort_values(["_dist", "odds", "book", "line"], ascending=[True, False, True, True], kind="stable")
    return _finish(d.drop_duplicates(GROUP, keep="first"), f"sempre_{side}", stake)


def all_bets(rows: pd.DataFrame, model: str, threshold: float, stake: float, phase: str, with_baselines: bool = True) -> pd.DataFrame:
    parts = [value_bets(rows, model, threshold, stake, "valore:" + ("modello" if model.startswith("modello") and not model.endswith("calibrato") else model), phase)]
    cal = model + "_calibrato"
    if (rows["model"] == cal).any():
        parts.append(value_bets(rows, cal, threshold, stake, "valore:modello_calibrato", phase))
    if with_baselines:
        for b in ("frequenza_ultime20", "poisson_ultime10"):
            parts.append(value_bets(rows, b, threshold, stake, f"valore:{b}", phase))
        parts.append(naive_bets(rows, "over", stake, phase, model))
        parts.append(naive_bets(rows, "under", stake, phase, model))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BET_COLUMNS)
