"""Giocata del giorno: 1-2 quote vicine a 2 scelte SENZA usare il modello, dal confronto tra bookmaker.

Per ogni giocata (stessa linea) si toglie il margine di ogni bookmaker che quota sia Over sia Under; la probabilita' equa e' la media
degli ALTRI bookmaker (quello considerato non influenza il proprio confronto). Se un bookmaker paga piu' di quanto dice quel consenso,
il prezzo e' migliore del dovuto. Non e' una garanzia: la giocata puo' perdere (a quota ~2 riesce circa una volta su due).
Un vantaggio troppo grande (oltre MAX_EDGE) e' quasi sempre una quota vecchia o una linea diversa: viene scartato, non proposto.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .injuries import OUT, injury_level

ODDS_RANGE = (1.80, 2.20)
MIN_OTHER_BOOKS = 3
MAX_EDGE = 0.10
COLUMNS = ["giocatore", "squadra", "partita", "inizio", "giocata", "bookmaker", "quota", "p_equa", "quota_equa", "vantaggio", "n_altri",
           "min_altri", "max_altri", "eta_min", "nota", "prop_id", "lato", "game"]


def consensus_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Una riga per quota con la probabilita' equa del consenso degli altri bookmaker (margine tolto)."""
    need = {"prop_id", "lato", "book", "quota"}
    if rows is None or rows.empty or not need <= set(rows.columns):
        return pd.DataFrame()
    d = rows[["prop_id", "lato", "book", "quota"]]
    d = d.sort_values("quota", ascending=False).drop_duplicates(["prop_id", "lato", "book"])
    piv = d.pivot_table(index=["prop_id", "book"], columns="lato", values="quota", aggfunc="max")
    if not {"Over", "Under"} <= set(piv.columns):
        return pd.DataFrame()
    piv = piv.dropna(subset=["Over", "Under"])
    inv_o, inv_u = 1 / piv["Over"], 1 / piv["Under"]
    nv = pd.DataFrame({"Over": inv_o / (inv_o + inv_u), "Under": inv_u / (inv_o + inv_u)}, index=piv.index)
    out = []
    for pid, g in nv.groupby(level=0):
        books = g.index.get_level_values(1)
        for b in books:
            others = g[books != b]
            for side in ("Over", "Under"):
                if len(others) >= 1:
                    out.append((pid, side, b, float(others[side].mean()), len(others), float(others[side].min()), float(others[side].max())))
    return pd.DataFrame(out, columns=["prop_id", "lato", "book", "p_equa", "n_altri", "p_min", "p_max"])


def daily_picks(rows: pd.DataFrame, now: datetime, *, include_under: bool = False, odds_range=ODDS_RANGE, min_books: int = MIN_OTHER_BOOKS,
                max_edge: float = MAX_EDGE, n: int = 2) -> pd.DataFrame:
    """Fino a ``n`` quote (una per partita) con quota nella finestra e prezzo migliore del consenso degli altri bookmaker."""
    empty = pd.DataFrame(columns=COLUMNS)
    cons = consensus_table(rows)
    if cons.empty:
        return empty
    d = rows.merge(cons, on=["prop_id", "lato", "book"], how="inner")
    d = d[~d["manuale"] & ~d["obsoleta"] & d["inizio"].notna()]
    d = d[[pd.Timestamp(t).to_pydatetime() > now for t in d["inizio"]]] if len(d) else d
    if "infortunio" in d.columns:
        d = d[[injury_level(s) != OUT for s in d["infortunio"].fillna("")]]
    if not include_under:
        d = d[d["lato"] == "Over"]
    d = d[(d["quota"] >= odds_range[0]) & (d["quota"] <= odds_range[1]) & (d["n_altri"] >= min_books)]
    if d.empty:
        return empty
    d = d.assign(vantaggio=d["p_equa"] * d["quota"] - 1)
    d = d[(d["vantaggio"] > 0) & (d["vantaggio"] <= max_edge)]
    if d.empty:
        return empty
    d = d.assign(game=d["gday"].astype(str) + "|" + d["evento"].astype(str)).sort_values("vantaggio", ascending=False)
    d = d.drop_duplicates("game").head(n)
    inj = d["infortunio"].fillna("") if "infortunio" in d.columns else pd.Series([""] * len(d), index=d.index)
    out = pd.DataFrame({
        "giocatore": d["giocatore"].values, "squadra": d["squadra"].values, "partita": d["evento"].values, "inizio": [pd.Timestamp(t) for t in d["inizio"]],
        "giocata": [f"{m} {l} {x:g}" for m, l, x in zip(d["mercato"], d["lato"], d["linea"])], "bookmaker": d["bookmaker"].values,
        "quota": d["quota"].values, "p_equa": d["p_equa"].values, "quota_equa": 1 / d["p_equa"].values, "vantaggio": d["vantaggio"].values,
        "n_altri": d["n_altri"].values, "min_altri": d["p_min"].values, "max_altri": d["p_max"].values, "eta_min": d["eta_min"].values,
        "nota": [f"segnalato «{s}» (infortuni)" if injury_level(s) else "" for s in inj], "prop_id": d["prop_id"].values, "lato": d["lato"].values,
        "game": d["game"].values,
    })
    return out.reset_index(drop=True)


def market_legs(rows: pd.DataFrame, now: datetime, *, include_under: bool = False, odds_max: float = 1.60, p_min: float = 0.60,
                min_books: int = MIN_OTHER_BOOKS) -> list:
    """Selezioni a quota bassa con probabilita' del MERCATO (media degli altri bookmaker, margine tolto), senza usare il modello.

    p_lo / p_hi sono la probabilita' piu' bassa e piu' alta tra gli altri bookmaker: misurano quanto i bookmaker sono in disaccordo.
    Linee intere escluse (il margine a due vie non gestisce il rimborso).
    """
    from .builder import Leg

    cons = consensus_table(rows)
    if cons.empty:
        return []
    d = rows.merge(cons, on=["prop_id", "lato", "book"], how="inner")
    d = d[~d["manuale"] & ~d["obsoleta"] & d["inizio"].notna()]
    d = d[[pd.Timestamp(t).to_pydatetime() > now for t in d["inizio"]]] if len(d) else d
    if "infortunio" in d.columns:
        d = d[[injury_level(s) == "" for s in d["infortunio"].fillna("")]]
    if not include_under:
        d = d[d["lato"] == "Over"]
    d = d[(d["linea"] % 1).abs() > 1e-9]
    d = d[(d["quota"] <= odds_max) & (d["quota"] > 1.0) & (d["p_equa"] >= p_min) & (d["n_altri"] >= min_books)]
    legs = []
    for r in d.itertuples():
        legs.append(Leg(id=f"{r.prop_id}|{r.lato}|{r.book}", key=r.key, giocatore=r.giocatore, squadra=str(r.squadra or ""), game=f"{r.gday}|{r.evento}",
                        evento=r.evento, inizio=pd.Timestamp(r.inizio), market=r.market, mercato=r.mercato, linea=float(r.linea), lato=r.lato,
                        book=r.bookmaker, odds=float(r.quota), p_win=float(r.p_equa), p_push=0.0, p_lo=float(min(r.p_min, r.p_equa)),
                        p_hi=float(max(r.p_max, r.p_equa)), n_games=int(r.n_partite) if r.n_partite == r.n_partite else 0,
                        eta_min=float(r.eta_min) if r.eta_min == r.eta_min else float("nan"), quote_known=bool(r.ts_nota), prop_id=r.prop_id))
    return legs
