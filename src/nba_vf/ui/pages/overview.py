"""Panoramica: numeri chiave, stato dei dati, avvisi, migliori valori stimati e collegamenti."""

from __future__ import annotations

import streamlit as st

from ...models import utc_now
from .. import charts, shell, tables
from .. import theme as T
from ..best import build_picks, most_profitable
from ..labels import fmt_age, fmt_dt
from .. import widgets as W


OV_COLS = ["giocatore", "evento", "mercato", "linea", "lato", "bookmaker", "quota", "p_win", "quota_equa", "roi", "ev", "esito", "avvisi"]


def _books_sub(rows) -> str:
    if not len(rows):
        return "nessuno"
    labels = sorted(rows["bookmaker"].unique())
    return ", ".join(labels[:4]) + ("…" if len(labels) > 4 else "")


def render() -> None:
    ctx = shell.begin("Panoramica", show_title=False)
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
    stake = float(st.session_state["stake"])
    n_priced = int(rows["priced"].sum()) if len(rows) else 0
    n_games = len(rows.drop_duplicates(["gday", "evento"])) if len(rows) else 0
    when = f"{fmt_dt(b.updated_at)} · {fmt_age(age)}" if b.updated_at else "—"

    st.markdown(
        f'<div class="nvf-hello"><div><h1>Benvenuto, Ale</h1><div class="sub">Panoramica delle opportunità NBA di oggi</div></div>'
        f'<div class="upd">Ultimo aggiornamento<br><b>{when}</b></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="nvf-cards">' + "".join([
        W.card_kpi("calendar_month", f"{n_games}", "Partite con quote", f"{b.n_games_analyzed} partite nello storico"),
        W.card_kpi("bar_chart", f"{b.n_props}", "Player props", "con almeno una quota"),
        W.card_kpi("database", f"{b.n_bookmakers}", "Bookmaker", _books_sub(rows)),
        W.card_kpi("trending_up", f"{n_priced}", "Quote analizzate", f"su {len(rows):,}".replace(",", ".")),
    ]) + "</div>", unsafe_allow_html=True)

    picks = build_picks(rows, stake, include_low=True, fresh_only=True) if len(rows) else None
    h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
    h1.markdown("### Top 5 giocate con valore")
    h2.page_link(pages["migliori"], label="Vedi tutte", icon=":material/arrow_forward:")
    if picks is None or picks.empty:
        st.info("Ora nessuna quota ha un valore atteso positivo con quote aggiornate, oppure mancano i dati di storico. "
                "Dettagli in «Copertura dati».")
    else:
        top = most_profitable(picks, 5)
        body = []
        for i, r in enumerate(top.itertuples(), 1):
            body.append([str(i), W.player_cell(r.giocatore), W.escape(r.partita), W.escape(r.giocata), W.escape(r.bookmaker),
                         f"{r.quota:.2f}", f"{r.prob:.0f}%", W.pill(f"{100 * r.guadagno / stake:+.1f}%"), W.reliability_pill(r.affidabilita),
                         f'<span class="nvf-muted">{W.escape(r.nota_dato)}</span>'])
        st.markdown('<div class="nvf-panel">' + W.table(
            ["#", "Giocatore", "Partita", "Giocata", "Book", "Quota", "Prob. stimata", "Valore atteso", "Affidabilità", "Dati"], body,
            ["left", "left", "left", "left", "left", "right", "right", "right", "left", "left"]) + "</div>", unsafe_allow_html=True)
        st.caption("Probabilità e valore atteso sono stime del modello, non garanzie.")
    st.write("")

    tab_all, tab_chart, tab_warn = st.tabs(["Tutte le quote", "Grafici", "Avvisi e copertura"])
    with tab_warn:
        if b.quotes:
            from ..coverage import COMPLETE, INSUFFICIENT, NONE, PARTIAL, build_coverage

            cov = build_coverage(b, board)
            sc = cov.state_counts
            st.caption(f"Copertura dello storico: {sc[COMPLETE]} giocatori completi · {sc[PARTIAL]} parziali · "
                       f"{sc[INSUFFICIENT] + sc[NONE]} insufficienti o senza storico · {cov.n_evaluable} quote valutabili su {cov.n_quotes}, "
                       f"{cov.n_priced if cov.n_priced is not None else 0} prezzate.")
            st.page_link(pages["copertura"], label="Dettaglio copertura dati", icon=":material/rule:")
        warnings = [m for m in b.messages if not m[1].startswith("MODALITÀ DEMO")] + (board.warnings if board else [])
        st.markdown(f"**Avvisi sulle stime ({len(warnings)})**")
        if not warnings:
            st.success("Nessun avviso: tutti i giocatori hanno storico sufficiente.")
        for level, text in warnings:
            getattr(st, level if level in ("info", "warning", "error") else "info")(text)
        if b.collection_issues:
            st.caption(f"Problemi di raccolta dati per {len(b.collection_issues)} giocatori: dettagli in «Copertura dati».")

    if rows.empty:
        st.warning("Ci sono dati ma nessuna quota: impossibile calcolare quote eque ed EV.")
        return

    with tab_all:
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


    with tab_chart:
        t1, t2 = st.tabs(["Probabilità vs ROI atteso", "Distribuzione del ROI atteso"])
        t1.plotly_chart(charts.fig_ev_scatter(rows), width="stretch")
        t2.plotly_chart(charts.fig_ev_hist(rows), width="stretch")
