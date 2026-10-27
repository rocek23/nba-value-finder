"""Partite NBA: calendario con props disponibili e storico delle partite analizzate."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, shell, tables
from ..labels import fmt_dt


def render() -> None:
    ctx = shell.begin("Partite NBA", "Partite in programma con le props disponibili e storico delle partite analizzate")
    b, board, rows = ctx.bundle, ctx.board, ctx.rows
    from datetime import timedelta

    from ...models import utc_now
    from .. import widgets as W
    from ..labels import to_local

    today = to_local(utc_now()).date()
    g = pd.DataFrame()
    if not rows.empty:
        g = (rows.groupby(["gday", "evento"]).agg(
            inizio=("inizio", "min"), props=("prop_id", "nunique"), quote=("quota", "size"), bookmaker=("book", "nunique"),
            giocatori=("key", "nunique"), prezzate=("priced", "sum"),
        ).reset_index().sort_values("inizio", na_position="last"))
        g["data"] = [to_local(x).date() if pd.notna(x) else None for x in g["inizio"]]
    days = {"Oggi": today, "Domani": today + timedelta(days=1)}
    n_today = int((g["data"] == days["Oggi"]).sum()) if len(g) else 0
    n_tom = int((g["data"] == days["Domani"]).sum()) if len(g) else 0
    n_other = len(g) - n_today - n_tom if len(g) else 0
    st.caption(f"Partite con player props disponibili · ora di Roma · oggi è {today.strftime('%d/%m/%Y')}")
    tabs = st.tabs([f"Oggi ({n_today})", f"Domani ({n_tom})", f"Altre date ({n_other})", "Storico analizzato"])

    def day_view(sel: pd.DataFrame, key: str) -> None:
        if sel.empty:
            st.info("Nessuna partita con quote in questa data nei dati caricati.")
            for level, text in b.messages:
                if not text.startswith("MODALITÀ DEMO"):
                    getattr(st, level if level in ("info", "warning", "error") else "info")(text)
            return
        body = []
        for r in sel.itertuples():
            away, _, home = str(r.evento).partition(" @ ")
            when = to_local(r.inizio).strftime("%H:%M") if pd.notna(r.inizio) else "—"
            if r.data and r.data not in days.values():
                when = f"{r.data.strftime('%d/%m')} {when}"
            body.append([when, W.team_chip(home), W.team_chip(away), str(r.props), str(r.bookmaker), f"{int(r.prezzate)}/{r.quote}"])
        st.markdown('<div class="nvf-panel">' + W.table(["Ora (Roma)", "Casa", "Trasferta", "Props", "Bookmaker", "Quote prezzate"], body,
                                                        ["left", "left", "left", "right", "right", "right"]) + "</div>", unsafe_allow_html=True)
        ev = st.selectbox("Mostra le props di una partita", list(sel["evento"]), key=f"mt_event_{key}")
        prefixes = tuple(f"{d}|" for d in sel.loc[sel["evento"] == ev, "gday"])
        sub = board.groups[(board.groups["evento"] == ev) & board.groups["prop_id"].str.startswith(prefixes)].copy()
        if sub.empty:
            st.info("Nessuna prop per questa partita.")
            return
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

    with tabs[0]:
        day_view(g[g["data"] == days["Oggi"]] if len(g) else g, "oggi")
    with tabs[1]:
        day_view(g[g["data"] == days["Domani"]] if len(g) else g, "domani")
    with tabs[2]:
        day_view(g[~g["data"].isin(list(days.values()))] if len(g) else g, "altre")
    tab2 = tabs[3]

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
