"""Migliori giocate: classifica leggibile delle quote con valore atteso positivo, senza gergo tecnico.

Logica pura (testabile). Parte dal tabellone gia' calcolato: non stima nulla e non inventa nulla. Una giocata entra
solo se il modello l'ha prezzata (storico sufficiente) e il suo valore atteso stimato e' positivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import game_title, team_label, to_local
from .tables import best_per_prop

HIGH, MEDIUM, LOW = "Alta", "Media", "Bassa"
RELIABILITY_HELP = {
    HIGH: "campione di almeno 25 partite e guadagno atteso positivo anche nello scenario pessimistico dell'intervallo di incertezza",
    MEDIUM: "campione di almeno 25 partite, ma l'intervallo di incertezza include anche il caso di guadagno nullo o negativo",
    LOW: "meno di 25 partite di storico: la stima è molto incerta",
}
PICK_COLUMNS = ["giocatore", "squadra", "partita", "inizio", "giocata", "bookmaker", "quota", "prob", "quota_equa",
                "guadagno", "vincita", "affidabilita", "partite", "eta_min"]


def reliability(n_games: float, roi: float, roi_lo: float) -> str:
    if n_games is None or np.isnan(n_games) or n_games < 25:
        return LOW
    return HIGH if (roi_lo == roi_lo and roi_lo > 0) else MEDIUM


def build_picks(rows: pd.DataFrame, stake: float, *, include_low: bool = False, fresh_only: bool = True,
                markets: list[str] | None = None) -> pd.DataFrame:
    """Una riga per giocata (prop + lato) con la migliore quota disponibile e solo valore atteso > 0."""
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["key"] + PICK_COLUMNS + ["prop_id", "lato"])
    d = rows[rows["priced"] & (rows["roi"] > 0)]
    if markets:
        d = d[d["mercato"].isin(markets)]
    if d.empty:
        return pd.DataFrame(columns=["key"] + PICK_COLUMNS + ["prop_id", "lato"])
    d = best_per_prop(d, include_stale=not fresh_only)
    if d.empty:
        return pd.DataFrame(columns=["key"] + PICK_COLUMNS + ["prop_id", "lato"])
    out = pd.DataFrame({
        "key": d["key"].values, "prop_id": d["prop_id"].values, "lato": d["lato"].values,
        "giocatore": d["giocatore"].values,
        "squadra": [team_label(x) for x in d["squadra"]],
        "partita": [_match(r) for r in d.itertuples()],
        "inizio": [to_local(t).strftime("%d/%m %H:%M") if t is not None and t == t else "—" for t in d["inizio"]],
        "giocata": [f"{m} {l} {x:g}" for m, l, x in zip(d["mercato"], d["lato"], d["linea"])],
        "bookmaker": d["bookmaker"].values, "quota": d["quota"].values,
        "prob": 100 * d["p_win"].values, "quota_equa": d["quota_equa"].values,
        "guadagno": stake * d["roi"].values, "vincita": stake * (d["quota"].values - 1),
        "affidabilita": [reliability(n, r, lo) for n, r, lo in zip(d["n_partite"], d["roi"], d["roi_lo"])],
        "partite": d["n_partite"].values, "eta_min": d["eta_min"].values,
    })
    if not include_low:
        out = out[out["affidabilita"] != LOW]
    return out.reset_index(drop=True)


def _match(r) -> str:
    title = str(r.evento)
    if " @ " in title:
        away, home = title.split(" @ ", 1)
        return game_title(away, home)
    return title


def most_probable(picks: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Le giocate con valore atteso positivo che hanno la probabilita' di riuscita piu' alta."""
    return picks.sort_values(["prob", "guadagno"], ascending=False).head(n).reset_index(drop=True)


def most_profitable(picks: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Le giocate con il guadagno atteso piu' alto sullo stake scelto."""
    return picks.sort_values(["guadagno", "prob"], ascending=False).head(n).reset_index(drop=True)
