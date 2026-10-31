"""Probabilita' di mercato (senza modello): confronto tra bookmaker per le multiple a quota bassa.

Per ogni giocata (stessa linea) si toglie il margine di ogni bookmaker che quota sia Over sia Under; la probabilita' equa e' la media
degli ALTRI bookmaker (quello considerato non influenza il proprio confronto). Se un bookmaker paga piu' di quanto dice quel consenso,
il prezzo e' migliore del dovuto. Non e' una garanzia: la giocata puo' perdere (a quota ~2 riesce circa una volta su due).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .injuries import mask

MIN_OTHER_BOOKS = 3


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


def market_legs(rows: pd.DataFrame, now: datetime, *, include_under: bool = False, odds_max: float = 1.60, p_min: float = 0.60,
                min_books: int = MIN_OTHER_BOOKS, min_lines: dict | None = None) -> list:
    """Selezioni a quota bassa con probabilita' del MERCATO (media degli altri bookmaker, margine tolto), senza usare il modello.

    p_lo / p_hi sono la probabilita' piu' bassa e piu' alta tra gli altri bookmaker: misurano quanto i bookmaker sono in disaccordo.
    Linee intere escluse (il margine a due vie non gestisce il rimborso).
    """
    from .builder import Leg, apply_min_lines

    cons = consensus_table(rows)
    if cons.empty:
        return []
    d = rows.merge(cons, on=["prop_id", "lato", "book"], how="inner")
    d = d[~d["manuale"] & ~d["obsoleta"] & d["inizio"].notna()]
    d = d[[pd.Timestamp(t).to_pydatetime() > now for t in d["inizio"]]] if len(d) else d
    if "infortunio" in d.columns:
        d = d[mask(d["infortunio"].fillna(""), lambda l: l == "")]
    if not include_under:
        d = d[d["lato"] == "Over"]
    d = d[(d["linea"] % 1).abs() > 1e-9]
    d = d[(d["quota"] <= odds_max) & (d["quota"] > 1.0) & (d["p_equa"] >= p_min) & (d["n_altri"] >= min_books)]
    d = apply_min_lines(d, min_lines)
    legs = []
    for r in d.itertuples():
        legs.append(Leg(id=f"{r.prop_id}|{r.lato}|{r.book}", key=r.key, giocatore=r.giocatore, squadra=str(r.squadra or ""), game=f"{r.gday}|{r.evento}",
                        evento=r.evento, inizio=pd.Timestamp(r.inizio), market=r.market, mercato=r.mercato, linea=float(r.linea), lato=r.lato,
                        book=r.bookmaker, odds=float(r.quota), p_win=float(r.p_equa), p_push=0.0, p_lo=float(min(r.p_min, r.p_equa)),
                        p_hi=float(max(r.p_max, r.p_equa)), n_games=int(r.n_partite) if r.n_partite == r.n_partite else 0,
                        eta_min=float(r.eta_min) if r.eta_min == r.eta_min else float("nan"), quote_known=bool(r.ts_nota), prop_id=r.prop_id))
    return legs


MIN_TOP_PROB = 0.62      # probabilita' minima del mercato per entrare tra le "Top del giorno"
TOP_PRICE_FLOOR = -0.02  # la quota migliore non puo' pagare piu' del 2% sotto la quota equa del consenso
TOP_MODEL_GAP = 0.05     # il modello non deve stimare >5 punti meno del mercato (se lo contraddice, la giocata e' scartata)


def top_of_day(rows: pd.DataFrame, now: datetime, *, n: int = 3, include_under: bool = False, min_lines: dict | None = None,
               min_prob: float = MIN_TOP_PROB, min_books: int = MIN_OTHER_BOOKS, funnel: list | None = None) -> pd.DataFrame:
    """Le giocate con la probabilita' (di mercato) piu' alta, a prezzo corretto, una per partita e una per giocatore.

    Probabilita' = media degli ALTRI bookmaker con il margine tolto. Si tiene la quota migliore di ogni giocata solo se non paga
    meno del consenso (+ tolleranza), il modello non la contraddice, il giocatore non e' «Out» e la linea esiste nel proprio bookmaker.
    Nessuna promessa: a quota bassa si vince piu' spesso, ma il valore atteso resta al piu' vicino a zero.
    """
    from .builder import apply_min_lines

    cols = ["giocatore", "squadra", "partita", "inizio", "giocata", "bookmaker", "quota", "prob_mercato", "prob_modello", "n_altri",
            "prezzo", "range", "partite", "nota", "prop_id", "lato", "gday"]
    empty = pd.DataFrame(columns=cols)

    def step(label, df):
        if funnel is not None:
            funnel.append((label, int(len(df[["prop_id", "lato"]].drop_duplicates())) if len(df) else 0))

    if rows is not None and len(rows):
        step("Giocate (prop + lato) disponibili", rows)
    cons = consensus_table(rows)
    if cons.empty:
        step("Con almeno un confronto Over/Under tra bookmaker", cons)
        return empty
    d = rows.merge(cons, on=["prop_id", "lato", "book"], how="inner")
    step("Con un confronto Over/Under tra bookmaker", d)
    d = d[~d["manuale"] & ~d["obsoleta"] & d["inizio"].notna()]
    step("Quota recente, non inserita a mano", d)
    d = d[d["priced"]]
    step("Con stima del modello (storico sufficiente)", d)
    d = d[[pd.Timestamp(t).to_pydatetime() > now for t in d["inizio"]]] if len(d) else d
    step("Partita non ancora iniziata", d)
    if "infortunio" in d.columns:
        d = d[mask(d["infortunio"].fillna(""), lambda l: l == "")]
    if not include_under:
        d = d[d["lato"] == "Over"]
    step("Giocatore non segnalato «Out» e lato ammesso", d)
    d = d[(d["linea"] % 1).abs() > 1e-9]
    step("Linea non intera", d)
    d = d[(d["n_altri"] >= min_books) & (d["quota"] > 1.0) & (d["n_partite"] >= 25)]
    step(f"Almeno {min_books} altri bookmaker e 25 partite di storico", d)
    d = apply_min_lines(d, min_lines)
    step("Linea almeno pari al minimo del tuo bookmaker", d)
    if len(d) == 0:
        return empty
    d = d.assign(prezzo=d["quota"] * d["p_equa"] - 1)
    d = d[d["p_equa"] >= min_prob]
    step(f"Probabilità di mercato almeno {100 * min_prob:.0f}%", d)
    d = d[d["prezzo"] >= TOP_PRICE_FLOOR]
    step("Prezzo non peggiore del consenso (−2%)", d)
    d = d[d["p_win"] >= d["p_equa"] - TOP_MODEL_GAP]
    step("Modello d'accordo con il mercato", d)
    if len(d) == 0:
        return empty
    d = d.sort_values("quota", ascending=False).drop_duplicates(["prop_id", "lato"])
    d = d.sort_values(["p_equa", "prezzo"], ascending=False)
    d = d.drop_duplicates("key").drop_duplicates(["gday", "evento"]).head(n)
    out = pd.DataFrame({
        "giocatore": d["giocatore"].values, "squadra": d["squadra"].astype(str).values, "partita": d["evento"].values,
        "inizio": [pd.Timestamp(t) for t in d["inizio"]], "giocata": [f"{m} {l} {x:g}" for m, l, x in zip(d["mercato"], d["lato"], d["linea"])],
        "bookmaker": d["bookmaker"].values, "quota": d["quota"].values.astype(float), "prob_mercato": 100 * d["p_equa"].values.astype(float),
        "prob_modello": 100 * d["p_win"].values.astype(float), "n_altri": d["n_altri"].values.astype(int), "prezzo": 100 * d["prezzo"].values.astype(float),
        "range": [f"{100 * lo:.0f}–{100 * hi:.0f}%" for lo, hi in zip(d["p_min"], d["p_max"])], "partite": d["n_partite"].values,
        "nota": ["" for _ in range(len(d))], "prop_id": d["prop_id"].values, "lato": d["lato"].values, "gday": d["gday"].values,
    })
    return out.reset_index(drop=True)
