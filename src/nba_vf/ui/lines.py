"""Linee «principali»: i bookmaker americani pubblicano anche scale di linee alternative molto basse (Over 4.5 punti per un giocatore da 25),
con quote da 1.03-1.20, che i bookmaker italiani non quotano. Si tengono solo la linea principale e quelle adiacenti."""
from __future__ import annotations

import numpy as np
import pandas as pd

NEEDED = {"gday", "key", "mercato", "linea", "quota", "lato"}


def near_main(rows: pd.DataFrame, steps: int = 1) -> np.ndarray:
    """Maschera booleana (stessa lunghezza di rows): True se la linea e' quella principale (probabilita' Over implicita piu' vicina al 50%)
    o a `steps` linee di distanza. Le giocate senza quote Over di riferimento restano ammesse."""
    n = 0 if rows is None else len(rows)
    out = np.ones(n, dtype=bool)
    if n == 0 or not NEEDED <= set(rows.columns):
        return out
    over = rows[rows["lato"] == "Over"]
    if over.empty:
        return out
    t = over.assign(_p=1.0 / over["quota"].astype(float)).groupby(["gday", "key", "mercato", "linea"], sort=False)["_p"].median().reset_index()
    allowed: set = set()
    for (gday, key, merc), g in t.groupby(["gday", "key", "mercato"], sort=False):
        g = g.sort_values("linea").reset_index(drop=True)
        i = int((g["_p"] - 0.5).abs().to_numpy().argmin())
        for line in g["linea"].iloc[max(0, i - steps): i + steps + 1]:
            allowed.add((gday, key, merc, float(line)))
    has_over = set(zip(t["gday"], t["key"], t["mercato"]))
    for k, (gday, key, merc, line) in enumerate(zip(rows["gday"], rows["key"], rows["mercato"], rows["linea"].astype(float))):
        if (gday, key, merc) in has_over:
            out[k] = (gday, key, merc, float(line)) in allowed
    return out
