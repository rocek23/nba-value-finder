"""Migliori giocate: classifica leggibile delle quote con valore atteso positivo, senza gergo tecnico.

Logica pura (testabile). Parte dal tabellone gia' calcolato: non stima nulla e non inventa nulla. Una giocata entra
solo se il modello l'ha prezzata (storico sufficiente) e il suo valore atteso stimato e' positivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import fmt_age, game_title, team_label, to_local
from .tables import best_per_prop

HIGH, MEDIUM, LOW = "Alta", "Media", "Bassa"
RELIABILITY_HELP = {
    HIGH: "campione di almeno 25 partite e guadagno atteso positivo anche nello scenario pessimistico dell'intervallo di incertezza",
    MEDIUM: "campione di almeno 25 partite, ma l'intervallo di incertezza include anche il caso di guadagno nullo o negativo",
    LOW: "meno di 25 partite di storico: la stima è molto incerta",
}
def quote_age_text(eta_min: float, known: bool, max_age: float | None = None) -> str:
    """Quanto e' aggiornata la quota, dichiarando quando l'orario non e' certo."""
    if eta_min is None or eta_min != eta_min:
        return "da verificare"
    age = fmt_age(float(eta_min))
    if max_age is not None and eta_min > max_age:
        return f"obsoleta ({age})"
    return f"aggiornata {age}" if known else f"ottenuta {age} (orario del bookmaker da verificare)"


def data_note(n_games: float, eta_min: float, known: bool, max_age: float | None = None) -> str:
    """Riga di affidabilita' del dato: «Storico: 30 partite · Quota aggiornata 5 min fa»."""
    n = "da verificare" if n_games is None or n_games != n_games else f"{int(n_games)} partite"
    return f"Storico: {n} · Quota {quote_age_text(eta_min, known, max_age)}"


PICK_COLUMNS = ["giocatore", "squadra", "partita", "inizio", "giocata", "bookmaker", "quota", "prob", "quota_equa",
                "guadagno", "vincita", "affidabilita", "partite", "eta_min", "nota_dato"]


def reliability(n_games: float, roi: float, roi_lo: float) -> str:
    if n_games is None or np.isnan(n_games) or n_games < 25:
        return LOW
    return HIGH if (roi_lo == roi_lo and roi_lo > 0) else MEDIUM


def build_picks(rows: pd.DataFrame, stake: float, *, include_low: bool = False, fresh_only: bool = True,
                markets: list[str] | None = None, include_under: bool = True) -> pd.DataFrame:
    """Una riga per giocata (prop + lato) con la migliore quota disponibile e solo valore atteso > 0."""
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["key"] + PICK_COLUMNS + ["prop_id", "lato"])
    d = rows[rows["priced"] & (rows["roi"] > 0)]
    if not include_under:
        d = d[d["lato"] == "Over"]
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
        "inizio_ts": [pd.Timestamp(t) if t is not None and t == t else pd.NaT for t in d["inizio"]],
        "inizio": [to_local(t).strftime("%d/%m %H:%M") if t is not None and t == t else "—" for t in d["inizio"]],
        "giocata": [f"{m} {l} {x:g}" for m, l, x in zip(d["mercato"], d["lato"], d["linea"])],
        "bookmaker": d["bookmaker"].values, "quota": d["quota"].values,
        "prob": 100 * d["p_win"].values, "quota_equa": d["quota_equa"].values,
        "guadagno": stake * d["roi"].values, "vincita": stake * (d["quota"].values - 1),
        "affidabilita": [reliability(n, r, lo) for n, r, lo in zip(d["n_partite"], d["roi"], d["roi_lo"])],
        "partite": d["n_partite"].values, "eta_min": d["eta_min"].values,
        "nota_dato": [data_note(n, e, bool(k)) for n, e, k in zip(d["n_partite"], d["eta_min"], d["ts_nota"])],
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


def by_game(picks: pd.DataFrame) -> pd.DataFrame:
    """Riepilogo per partita: quante giocate con valore e quale e' la migliore."""
    cols = ["partita", "inizio", "giocate", "alta", "migliore"]
    if picks is None or picks.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (match, start), g in picks.groupby(["partita", "inizio"], sort=False):
        top = g.sort_values("guadagno", ascending=False).iloc[0]
        out.append({"partita": match, "inizio": start, "giocate": len(g), "alta": int((g["affidabilita"] == HIGH).sum()),
                    "migliore": f"{top['giocatore']} · {top['giocata']} @ {top['quota']:.2f}", "_t": g["inizio_ts"].min()})
    res = pd.DataFrame(out, columns=cols + ["_t"])
    return res.sort_values(["_t", "partita"], na_position="last").drop(columns="_t").reset_index(drop=True)


def for_game(picks: pd.DataFrame, partita: str, inizio: str | None = None) -> pd.DataFrame:
    """Le giocate di una sola partita, dalla piu' redditizia."""
    d = picks[picks["partita"] == partita]
    if inizio is not None:
        d = d[d["inizio"] == inizio]
    return d.sort_values(["guadagno", "prob"], ascending=False).reset_index(drop=True)
