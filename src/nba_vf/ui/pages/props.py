"""Player Props: tutte le props disponibili con quote medie, migliori e confronto con il modello."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import shell


def render() -> None:
    ctx = shell.begin("Player Props", "Tutte le props disponibili: quote medie tra bookmaker, probabilità del modello e del mercato")
    shell.need_quotes(ctx)
    g = ctx.board.groups
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    markets = c1.multiselect("Mercato", sorted(g["mercato"].unique()), key="pp_mkt", placeholder="Tutti")
    events = c2.multiselect("Partita", sorted(g["evento"].unique()), key="pp_ev", placeholder="Tutte")
    q = c3.text_input("Giocatore contiene", key="pp_q")
    min_books = c4.slider("Bookmaker minimi", 1, int(max(g["n_book"].max(), 2)), 1, key="pp_books")
    c5, c6 = st.columns([2, 2])
    only_priced = c5.toggle("Solo props con stima del modello", value=False, key="pp_priced")
    teams = sorted(t for t in g["squadra"].dropna().unique())
    team_sel = c6.multiselect("Squadra", teams, key="pp_team", placeholder="Tutte")
    d = g.copy()
    if markets:
        d = d[d["mercato"].isin(markets)]
    if events:
        d = d[d["evento"].isin(events)]
    if team_sel:
        d = d[d["squadra"].isin(team_sel)]
    if q:
        d = d[d["giocatore"].str.contains(q, case=False)]
    d = d[d["n_book"] >= min_books]
    if only_priced:
        d = d[d["p_over_modello"].notna()]
    st.caption(f"{len(d)} props su {len(g)}")
    out = pd.DataFrame({
        "Giocatore": d["giocatore"], "Sq.": d["squadra"], "Partita": d["evento"], "Mercato": d["mercato"], "Linea": d["linea"],
        "Book": d["n_book"], "Over media": d["quota_over_media"], "Over migliore": d["quota_over_max"], "Under media": d["quota_under_media"],
        "Under migliore": d["quota_under_max"], "Margine medio": 100 * d["margine_medio"], "P Over modello": 100 * d["p_over_modello"],
        "P Over mercato": 100 * d["p_over_mercato"], "Differenza (pt)": 100 * d["delta_modello_mercato"],
        "Quota equa Over": d["quota_equa_over"], "Quota equa Under": d["quota_equa_under"],
    })
    fmt = {"Linea": "{:g}", "Over media": "{:.2f}", "Over migliore": "{:.2f}", "Under media": "{:.2f}", "Under migliore": "{:.2f}",
           "Margine medio": "{:.1f}%", "P Over modello": "{:.1f}%", "P Over mercato": "{:.1f}%", "Differenza (pt)": "{:+.1f}",
           "Quota equa Over": "{:.2f}", "Quota equa Under": "{:.2f}"}
    st.dataframe(out.style.format(fmt, na_rep="—"), hide_index=True, width="stretch", height=520)
    st.download_button("Scarica CSV", out.to_csv(index=False).encode("utf-8"), "player_props.csv", "text/csv")
    st.caption("«—» = dato non disponibile (es. prop non prezzabile dal modello o un solo lato quotato). "
               "«Differenza» = P Over del modello − P Over del mercato senza margine.")
    if not d.empty:
        opts = {f"{r.giocatore} · {r.mercato} {r.linea:g} ({r.evento})": r.prop_id for r in d.head(300).itertuples()}
        sel = st.selectbox("Apri l'analisi dettagliata", list(opts), key="pp_sel")
        if st.button("Apri analisi dettagliata", type="primary", key="pp_open"):
            shell.open_analysis(opts[sel])
