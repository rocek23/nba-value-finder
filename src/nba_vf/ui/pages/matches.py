"""Partite NBA: calendario con props disponibili e storico delle partite analizzate."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, shell, tables
from ..labels import fmt_dt


def render() -> None:
    ctx = shell.begin("Partite NBA", "Partite in programma con le props disponibili e storico delle partite analizzate")
    b, board, rows = ctx.bundle, ctx.board, ctx.rows
    tab1, tab2 = st.tabs(["In programma", "Storico analizzato"])

    with tab1:
        if rows.empty:
            st.info("Nessuna partita in programma con quote nei dati caricati.")
            for level, text in b.messages:
                if not text.startswith("MODALITÀ DEMO"):
                    getattr(st, level if level in ("info", "warning", "error") else "info")(text)
        else:
            g = (rows.groupby(["gday", "evento"]).agg(
                inizio=("inizio", "min"), props=("prop_id", "nunique"), quote=("quota", "size"), bookmaker=("book", "nunique"),
                giocatori=("key", "nunique"), prezzate=("priced", "mean"), con_avvisi=("n_avvisi", lambda s: int((s > 0).sum())),
            ).reset_index().sort_values("inizio", na_position="last"))
            show = pd.DataFrame({
                "Inizio (Roma)": [fmt_dt(x) if pd.notna(x) else "—" for x in g["inizio"]], "Partita": g["evento"],
                "Props": g["props"], "Quote": g["quote"], "Bookmaker": g["bookmaker"], "Giocatori": g["giocatori"],
                "Quote prezzate": 100 * g["prezzate"], "Quote con avvisi": g["con_avvisi"],
            })
            st.dataframe(show.style.format({"Quote prezzate": "{:.0f}%"}), hide_index=True, width="stretch")
            ev = st.selectbox("Mostra le props di una partita", list(g["evento"]), key="mt_event")
            sub = board.groups[board.groups["evento"] == ev].copy()
            if sub.empty:
                st.info("Nessuna prop per questa partita.")
            else:
                sub["delta"] = 100 * sub["delta_modello_mercato"]
                out = pd.DataFrame({
                    "Giocatore": sub["giocatore"], "Mercato": sub["mercato"], "Linea": sub["linea"], "Book": sub["n_book"],
                    "Quota Over media": sub["quota_over_media"], "Quota Under media": sub["quota_under_media"],
                    "P Over modello": 100 * sub["p_over_modello"], "P Over mercato": 100 * sub["p_over_mercato"],
                    "Differenza (pt)": sub["delta"],
                }).sort_values(["Giocatore", "Mercato"])
                st.dataframe(out.style.format({"Linea": "{:g}", "Quota Over media": "{:.2f}", "Quota Under media": "{:.2f}",
                                               "P Over modello": "{:.1f}%", "P Over mercato": "{:.1f}%", "Differenza (pt)": "{:+.1f}"}, na_rep="—"),
                             hide_index=True, width="stretch", height=420)
                st.caption("«P Over mercato» è la media tra bookmaker della probabilità implicita con il margine rimosso "
                           "(metodo proporzionale; per linee intere è un'approssimazione perché ignora il rimborso).")

    with tab2:
        games = b.games_df()
        if games.empty:
            st.warning("Nessuno storico di partite caricato: senza storico non si possono calcolare le stime.")
            return
        c1, c2 = st.columns([2, 1])
        lo, hi = games["data"].min(), games["data"].max()
        rng = c1.date_input("Periodo", value=(lo, hi), min_value=lo, max_value=hi, key="mt_range")
        team = c2.text_input("Squadra contiene", key="mt_team", placeholder="es. DMA")
        d = games
        if isinstance(rng, tuple) and len(rng) == 2:
            d = d[(d["data"] >= rng[0]) & (d["data"] <= rng[1])]
        if team:
            d = d[d["squadre"].str.contains(team, case=False)]
        st.dataframe(d.rename(columns={"partita_id": "ID partita", "data": "Data", "squadre": "Squadre", "giocatori": "Giocatori nel dataset", "stagione": "Stagione"}),
                     hide_index=True, width="stretch", height=380)
        st.plotly_chart(charts.fig_games_per_day(d), width="stretch")
        st.subheader("Giocatori con storico")
        pl = pd.DataFrame([{
            "Giocatore": b.player_name(k), "Squadra": (b.players[k].team if k in b.players else None), "Partite": len(v),
            "Ultima partita": v[-1].game_date, "Minuti medi": sum(l.minutes for l in v) / len(v),
        } for k, v in b.logs.items() if v]).sort_values("Giocatore")
        st.dataframe(pl.style.format({"Minuti medi": "{:.1f}"}), hide_index=True, width="stretch", height=320)
