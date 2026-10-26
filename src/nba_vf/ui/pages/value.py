"""Value Finder: quote con valore atteso stimato positivo, con filtri e indicatori di affidabilità."""

from __future__ import annotations

import streamlit as st

from .. import charts, shell, tables


def render() -> None:
    ctx = shell.begin("Value Finder", "Quote il cui valore atteso stimato è positivo, con filtri di affidabilità")
    shell.need_quotes(ctx)
    rows = ctx.rows
    stake = float(st.session_state["stake"])
    st.info("«Valore» qui significa **EV stimato positivo secondo il modello**, non un risultato assicurato: le stime hanno "
            "incertezza (colonne ROI min/max) e il mercato può avere informazioni che il modello non vede (infortuni, rotazioni).", icon="ℹ️")

    c1, c2, c3, c4 = st.columns(4)
    markets = c1.multiselect("Mercato", sorted(rows["mercato"].unique()), key="vf_mkt", placeholder="Tutti")
    sides = c2.multiselect("Lato", ["Over", "Under"], key="vf_side", placeholder="Entrambi")
    books = c3.multiselect("Bookmaker", sorted(rows["bookmaker"].unique()), key="vf_book", placeholder="Tutti")
    min_roi = c4.slider("ROI atteso minimo (%)", -20, 60, 0, key="vf_roi")
    c5, c6, c7, c8 = st.columns(4)
    min_p = c5.slider("P vittoria minima (%)", 0, 90, 0, key="vf_p")
    min_games = c6.slider("Partite di storico minime", 10, 82, 10, key="vf_games")
    robust = c7.toggle("Solo ROI min (IC) > 0", value=False, key="vf_robust",
                       help="Richiede che anche il limite inferiore dell'intervallo di incertezza sia positivo.")
    fresh = c8.toggle("Escludi quote obsolete", value=True, key="vf_fresh")
    c9, c10, c11 = st.columns(3)
    no_small = c9.toggle("Escludi campione ridotto", value=True, key="vf_small")
    no_push = c10.toggle("Escludi linee con rimborso", value=False, key="vf_push")
    best = c11.toggle("Migliore quota per prop", value=True, key="vf_best")

    d = rows[rows["priced"]]
    if markets:
        d = d[d["mercato"].isin(markets)]
    if sides:
        d = d[d["lato"].isin(sides)]
    if books:
        d = d[d["bookmaker"].isin(books)]
    d = d[(100 * d["roi"] >= min_roi) & (100 * d["p_win"] >= min_p) & (d["n_partite"] >= min_games)]
    if robust:
        d = d[d["roi_lo"] > 0]
    if no_small:
        d = d[~d["avvisi"].str.contains("Campione ridotto")]
    if no_push:
        d = d[d["p_push"] == 0]
    if best:
        d = tables.best_per_prop(d, include_stale=not fresh)
    elif fresh:
        d = d[~d["obsoleta"]]
    d = d.sort_values("roi", ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Quote che passano i filtri", len(d))
    k2.metric("Con ROI atteso > 0", int((d["roi"] > 0).sum()) if len(d) else 0)
    k3.metric("Con ROI min (IC) > 0", int((d["roi_lo"] > 0).sum()) if len(d) else 0)
    if d.empty:
        st.warning("Nessuna quota soddisfa i filtri. Prova ad allentarli: con filtri severi è normale non trovare nulla.")
    else:
        tables.show_table(d, stake, height=520)
        tables.csv_download(d, stake, "value_finder.csv")
        opts = {f"{r.giocatore} · {r.mercato} {r.lato} {r.linea:g} @ {r.quota:.2f} ({r.bookmaker})": (r.prop_id, r.lato) for r in d.head(200).itertuples()}
        sel = st.selectbox("Apri l'analisi dettagliata", list(opts), key="vf_sel")
        if st.button("Apri analisi dettagliata", type="primary", key="vf_open"):
            pid, lato = opts[sel]
            shell.open_analysis(pid, side=lato)
    st.plotly_chart(charts.fig_ev_scatter(d if len(d) else rows.head(0)), width="stretch")
    with st.expander("Come leggere la tabella"):
        st.markdown(
            "- **P vittoria / P rimborso**: probabilità stimate; con linee intere il rimborso (push) è calcolato a parte.\n"
            "- **Quota equa** = (1 − P rimborso) / P vittoria: la quota a cui il valore atteso sarebbe zero.\n"
            "- **ROI atteso** = P vittoria × quota + P rimborso − 1; **EV** = stake × ROI.\n"
            "- **ROI min/max (IC)**: intervallo di incertezza sulla *stima dei parametri*; non copre un modello sbagliato.\n"
            "- **Avvisi**: campione ridotto, quota obsoleta o manuale, linea con rimborso, ecc."
        )
