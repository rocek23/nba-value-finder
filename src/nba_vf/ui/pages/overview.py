"""Panoramica: numeri chiave, stato dei dati, avvisi, migliori valori stimati e collegamenti."""

from __future__ import annotations

import streamlit as st

from ...models import utc_now
from .. import charts, shell, tables
from .. import theme as T
from ..labels import fmt_age, fmt_dt


OV_COLS = ["giocatore", "evento", "mercato", "linea", "lato", "bookmaker", "quota", "p_win", "quota_equa", "roi", "ev", "esito", "avvisi"]


def _books_sub(rows) -> str:
    if not len(rows):
        return "nessuno"
    labels = sorted(rows["bookmaker"].unique())
    return ", ".join(labels[:4]) + ("…" if len(labels) > 4 else "")


def render() -> None:
    ctx = shell.begin("Panoramica", "Stato dei dati, avvisi sulle stime e quote con il maggiore valore atteso stimato")
    b, board, rows = ctx.bundle, ctx.board, ctx.rows
    pages = st.session_state["_pages"]

    if b.is_empty:
        for level, text in b.messages:
            getattr(st, level if level in ("info", "warning", "error") else "info")(text)
        st.markdown("#### Come iniziare in modalità live")
        st.markdown("1. Copia `.env.example` in `.env` e inserisci le chiavi **solo nel tuo computer** (mai in chat).\n"
                    "2. Apri «Impostazioni e API», verifica le connessioni e premi **Aggiorna dati live**.\n"
                    "3. In alternativa importa un CSV con le quote dei bookmaker non coperti dalle API.")
        st.page_link(pages["impostazioni"], label="Impostazioni e API", icon=":material/settings:")
        st.page_link(pages["partite"], label="Partite NBA", icon=":material/sports_basketball:")
        return

    age = (utc_now() - b.updated_at).total_seconds() / 60 if b.updated_at else None
    if b.is_demo:
        stato, color = "Dati demo", T.AMBER
    elif b.status != "ok":
        stato, color = "Parziale", T.AMBER
    elif age is not None and age > 60:
        stato, color = "Dati datati", T.AMBER
    else:
        stato, color = "Aggiornato", T.GREEN
    n_priced = int(rows["priced"].sum()) if len(rows) else 0
    upcoming = len({r for r in rows["gday"]}) if len(rows) else 0
    T.kpi_row([
        ("Partite NBA analizzate", f"{b.n_games_analyzed}", f"nello storico caricato · {upcoming} giorno/i in programma", None),
        ("Player props disponibili", f"{b.n_props}", f"{n_priced} quote prezzate su {len(rows)}", None),
        ("Bookmaker collegati", f"{b.n_bookmakers}", _books_sub(rows), None),
        ("Ultimo aggiornamento", stato, f"{fmt_dt(b.updated_at)} · {fmt_age(age)} · stato {b.status}", color),
    ])
    st.write("")

    if b.quotes:
        from ..coverage import COMPLETE, INSUFFICIENT, NONE, PARTIAL, build_coverage

        cov = build_coverage(b, board)
        sc = cov.state_counts
        st.caption(f"Copertura dello storico: {sc[COMPLETE]} giocatori completi · {sc[PARTIAL]} parziali · "
                   f"{sc[INSUFFICIENT] + sc[NONE]} insufficienti o senza storico · {cov.n_evaluable} quote valutabili su {cov.n_quotes}, "
                   f"{cov.n_priced if cov.n_priced is not None else 0} prezzate.")
        st.page_link(pages["copertura"], label="Dettaglio copertura dati", icon=":material/fact_check:")

    # ---- avvisi sulle stime -----------------------------------------------------------------
    warnings = [m for m in b.messages if not m[1].startswith("MODALITÀ DEMO")] + (board.warnings if board else [])
    with st.expander(f"Avvisi sulle stime ({len(warnings)})", expanded=bool(warnings)):
        if not warnings:
            st.success("Nessun avviso: tutti i giocatori hanno storico sufficiente.")
        for level, text in warnings:
            getattr(st, level if level in ("info", "warning", "error") else "info")(text)
        if b.collection_issues:
            st.caption(f"Problemi di raccolta dati per {len(b.collection_issues)} giocatori: dettagli in «Analisi giocatore».")

    if rows.empty:
        st.warning("Ci sono dati ma nessuna quota: impossibile calcolare quote eque ed EV.")
        return

    # ---- tabella quote ---------------------------------------------------------------------
    st.subheader("Quote con probabilità, quota equa ed EV")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    markets = c1.multiselect("Mercato", sorted(rows["mercato"].unique()), key="ov_mkt", placeholder="Tutti i mercati")
    fresh = c2.toggle("Solo quote aggiornate", value=True, key="ov_fresh")
    no_small = c3.toggle("Escludi campione ridotto", value=False, key="ov_small")
    best = c4.toggle("Migliore quota per prop", value=True, key="ov_best")
    d = rows[rows["priced"]]
    if markets:
        d = d[d["mercato"].isin(markets)]
    if no_small:
        d = d[~d["avvisi"].str.contains("Campione ridotto")]
    if best:
        d = tables.best_per_prop(d, include_stale=not fresh)
    elif fresh:
        d = d[~d["obsoleta"]]
    d = d.sort_values("roi", ascending=False)
    if d.empty:
        st.info("Nessuna quota corrisponde ai filtri.")
    else:
        tables.show_table(d.head(25), float(st.session_state["stake"]), columns=OV_COLS, height=520)
        st.caption("Clicca l'intestazione di una colonna per ordinare. ROI e EV sono **stime** con incertezza: "
                   "gli intervalli (ROI min/max) mostrano quanto possono variare.")
        opts = {f"{r.giocatore} · {r.mercato} {r.lato} {r.linea:g} @ {r.quota:.2f} ({r.bookmaker})": (r.prop_id, r.lato)
                for r in d.head(25).itertuples()}
        sel = st.selectbox("Apri l'analisi dettagliata di una quota", list(opts), key="ov_sel")
        if st.button("Apri analisi dettagliata", type="primary", key="ov_open"):
            pid, lato = opts[sel]
            shell.open_analysis(pid, side=lato)

    st.write("")
    t1, t2 = st.tabs(["Probabilità vs ROI atteso", "Distribuzione del ROI atteso"])
    t1.plotly_chart(charts.fig_ev_scatter(rows), width="stretch")
    t2.plotly_chart(charts.fig_ev_hist(rows), width="stretch")

    st.subheader("Vai a")
    c = st.columns(4)
    c[0].page_link(pages["value"], label="Value Finder", icon=":material/search:")
    c[1].page_link(pages["props"], label="Player Props", icon=":material/list:")
    c[2].page_link(pages["giocatore"], label="Analisi giocatore", icon=":material/person:")
    c[3].page_link(pages["bookmaker"], label="Confronto bookmaker", icon=":material/compare_arrows:")
