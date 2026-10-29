"""Migliori giocate: schermata iniziale, in linguaggio semplice."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import shell
from .. import widgets as W
from .. import theme as T
from .. import builder as B
from ..daily import market_legs
from ..best import HIGH, LOW, MEDIUM, RELIABILITY_HELP, build_picks, by_game, for_game, most_probable, most_profitable
from ..coverage import INSUFFICIENT, NONE, build_coverage
from ..labels import fmt_age


def _card(r, stake: float) -> None:
    with st.container(border=True):
        left, right = st.columns([3, 2])
        left.markdown(f"**{r.giocatore}** · {r.squadra}")
        right.markdown(f'<div style="text-align:right">{W.reliability_pill(r.affidabilita, long=True)}</div>', unsafe_allow_html=True)
        st.caption(f"{r.partita} · {r.inizio}")
        st.markdown(f"### {r.giocata}")
        st.markdown(f"Quota **{r.quota:.2f}** su **{r.bookmaker}** · quota equa {r.quota_equa:.2f}")
        c1, c2 = st.columns(2)
        c1.metric("Probabilità", f"{r.prob:.0f}%", help="Probabilità di riuscita stimata dal modello. È una stima, non una certezza.")
        c2.metric("Guadagno atteso", f"{r.guadagno:+.2f} €", help=f"Media per ogni {stake:g} € puntati se la giocata fosse ripetuta molte volte, secondo il modello. Se vince: +{r.vincita:.2f} €.")
        st.caption(r.nota_dato)


def _table(d, stake: float, key: str) -> None:
    cols = {
        "giocatore": st.column_config.TextColumn("Giocatore"),
        "squadra": st.column_config.TextColumn("Squadra"),
        "partita": st.column_config.TextColumn("Partita (inizio, ora di Roma)"),
        "giocata": st.column_config.TextColumn("Giocata"),
        "bookmaker": st.column_config.TextColumn("Bookmaker"),
        "quota": st.column_config.NumberColumn("Quota", format="%.2f"),
        "prob": st.column_config.NumberColumn("Probabilità", format="%.0f%%"),
        "quota_equa": st.column_config.NumberColumn("Quota equa", format="%.2f"),
        "guadagno": st.column_config.NumberColumn(f"Guadagno atteso ({stake:g} €)", format="%+.2f €"),
        "vincita": st.column_config.NumberColumn(f"Se vince ({stake:g} €)", format="+%.2f €"),
        "affidabilita": st.column_config.TextColumn("Affidabilità"),
        "nota_dato": st.column_config.TextColumn("Affidabilità del dato"),
    }
    show = d[["giocatore", "squadra", "partita", "giocata", "bookmaker", "quota", "prob", "quota_equa",
              "guadagno", "vincita", "affidabilita", "nota_dato"]].copy()
    show["partita"] = d["partita"] + " · " + d["inizio"]
    st.dataframe(show, hide_index=True, width="stretch", column_config=cols, key=key, height=min(480, 38 + 35 * len(show)))


MIN_PRICE_EDGE = -0.06  # sotto questo prezzo (piu' di 6% peggio della quota equa) la multipla non viene proposta


def _doubles(ctx) -> None:
    """2-3 multiple vicino a quota 2 fatte con quote basse. Probabilita' del MERCATO (media degli altri bookmaker), non del modello."""
    from ...models import utc_now
    from ..labels import to_local
    st.subheader("Multiple vicino a quota 2")
    legs = market_legs(ctx.rows, utc_now(), include_under=bool(st.session_state.get("show_under", False)))
    res = B.search(legs, 2.0, tolerance=0.10, min_sel=2, max_sel=3, min_ev=MIN_PRICE_EDGE)
    if not res.proposals:
        st.info("Oggi non ci sono abbastanza selezioni a quota bassa, con un confronto credibile tra bookmaker e un prezzo ragionevole "
                "(non più del 6% peggiore della quota equa), per comporre una multipla vicino a 2. Meglio nessuna multipla che una forzata.")
    for p in res.proposals:
        with st.container(border=True):
            st.markdown("".join(f'<span style="display:inline-block;padding:.12rem .6rem;border-radius:999px;background:#1b2540;border:1px solid #2b3655;'
                                f'font-size:.78rem;margin-right:.35rem">{l}</span>' for l in p.labels), unsafe_allow_html=True)
            c = st.columns(4)
            c[0].metric("Quota totale", f"{p.odds:.2f}", help=f"{p.n} selezioni su {p.book}")
            c[1].metric("Probabilità che vincano tutte", f"{100 * p.p_all:.0f}%", help="Prodotto delle probabilità del mercato (margine tolto) di ogni selezione, una per partita. Non è una certezza.")
            c[2].metric("Quota equa", f"{1 / p.p_all:.2f}")
            c[3].metric("Prezzo rispetto al mercato", f"{100 * p.ev:+.1f}%", help="Positivo = paga più della quota equa; negativo = paga meno (il margine del bookmaker). Non è un guadagno garantito.")
            lines = pd.DataFrame([{"Giocatore": l.giocatore, "Partita": f"{l.evento} · {to_local(l.inizio).strftime('%d/%m %H:%M')}", "Giocata": l.descr,
                                   "Quota": l.odds, "Probabilità": f"{100 * l.p_win:.0f}%"} for l in p.legs])
            st.dataframe(lines, hide_index=True, width="stretch", column_config={"Quota": st.column_config.NumberColumn("Quota", format="%.2f")})
            st.caption(f"Tutte le quote sono di {p.book}.")
    st.warning("Nessuna multipla è sicura: ogni selezione può perdere e basta una per perdere tutto. Le probabilità qui sono quelle del **mercato** (media degli altri bookmaker, "
               "margine tolto), perché quelle del modello finora non hanno battuto il mercato. Una multipla a quota 2 riesce circa una volta su due e paga il margine più volte "
               "rispetto a una singola. Controlla sempre su Sisal che giocate e quote esistano.")
    st.divider()


def render() -> None:
    ctx = shell.begin("Migliori giocate", "Le quote con il miglior rapporto tra probabilità di riuscita e guadagno atteso, secondo il modello")
    shell.need_quotes(ctx)
    b, rows = ctx.bundle, ctx.rows
    stake = float(st.session_state["stake"])
    st.markdown("<style>[data-testid='stMetricValue']{font-size:1.7rem}</style>", unsafe_allow_html=True)

    if not st.session_state.get("show_under", False):
        st.caption("Sono mostrate solo giocate **Over**. Per vedere anche gli Under attiva «Mostra anche gli Under» nella barra laterale.")
    c1, c2, c3 = st.columns([2, 1, 1])
    markets = c1.multiselect("Mercato", sorted(rows["mercato"].unique()), key="bp_mkt", placeholder="Tutti i mercati")
    fresh = c2.toggle("Solo quote aggiornate", value=True, key="bp_fresh",
                      help="Esclude le quote più vecchie del limite scelto nella barra laterale.")
    low = c3.toggle("Includi campione ridotto", value=False, key="bp_low", help="Giocatori con meno di 25 partite: stime più incerte.")
    n_show = st.slider("Quante giocate mostrare", 3, 30, 10, key="bp_n")

    under = bool(st.session_state.get("show_under", False))
    picks = build_picks(rows, stake, include_low=low, fresh_only=fresh, markets=markets or None, include_under=under)
    cov = build_coverage(b, ctx.board)
    n_priced = cov.n_priced or 0
    T.kpi_row([
        ("Giocate con valore", f"{len(picks)}", "quota migliore per ogni giocata", T.GREEN if len(picks) else T.AMBER),
        ("Affidabilità alta", f"{int((picks['affidabilita'] == HIGH).sum()) if len(picks) else 0}", "guadagno positivo anche nel caso pessimistico", None),
        ("Quote valutate", f"{n_priced} / {cov.n_quotes}", f"{cov.n_players - cov.state_counts[INSUFFICIENT] - cov.state_counts[NONE]} giocatori su {cov.n_players} con storico sufficiente", None),
    ])
    st.write("")

    if picks.empty:
        _doubles(ctx)
        if n_priced == 0:
            st.warning("Nessuna quota è stata valutata: i giocatori non hanno abbastanza partite di storico. Nessuna probabilità viene inventata. "
                       "Dettagli nella pagina «Copertura dati».")
        elif fresh and len(build_picks(rows, stake, include_low=low, fresh_only=False, markets=markets or None, include_under=under)):
            st.warning("Ci sono giocate con valore, ma le quote sono più vecchie del limite scelto. Premi «Aggiorna dati live» nella barra laterale "
                       "oppure disattiva «Solo quote aggiornate».")
        else:
            st.info("In questo momento nessuna quota ha un valore atteso positivo con questi filtri. È un risultato normale: il mercato di solito è efficiente."
                    + ("" if under else " Gli Under sono nascosti: puoi mostrarli con «Mostra anche gli Under» nella barra laterale."))
        st.page_link(st.session_state["_pages"]["copertura"], label="Copertura dati", icon=":material/fact_check:")
        return

    _doubles(ctx)
    st.subheader("Le 3 giocate migliori")
    top = most_profitable(picks, 3)
    for col, (_, r) in zip(st.columns(len(top)), top.iterrows()):
        with col:
            _card(r, stake)

    t1, t2, t3 = st.tabs(["Più redditizie", "Più probabili", "Per partita"])
    with t1:
        st.caption("Ordinate per guadagno atteso sullo stake scelto nella barra laterale.")
        d = most_profitable(picks, n_show)
        _table(d, stake, "bp_t1")
    with t2:
        st.caption("Tra le giocate con valore positivo, quelle con la probabilità di riuscita più alta.")
        d2 = most_probable(picks, n_show)
        _table(d2, stake, "bp_t2")
    with t3:
        games = by_game(picks)
        st.caption("Le giocate con valore raggruppate per partita, in ordine di inizio (ora di Roma).")
        st.dataframe(games.rename(columns={"partita": "Partita", "inizio": "Inizio", "giocate": "Giocate con valore",
                                           "alta": "Di cui affidabilità alta", "migliore": "La più redditizia"}),
                     hide_index=True, width="stretch", height=min(360, 38 + 35 * len(games)))
        labels = {f"{r.partita} · {r.inizio} ({r.giocate} giocate)": (r.partita, r.inizio) for r in games.itertuples()}
        pick_game = st.selectbox("Scegli una partita", list(labels), key="bp_game")
        match, start = labels[pick_game]
        g = for_game(picks, match, start)
        st.markdown(f"#### {match}")
        _table(g, stake, "bp_t3")

    opts = {f"{r.giocatore} · {r.giocata} @ {r.quota:.2f} ({r.bookmaker})": (r.prop_id, r.lato) for r in most_profitable(picks, 60).itertuples()}
    sel = st.selectbox("Apri l'analisi dettagliata di una giocata", list(opts), key="bp_sel")
    if st.button("Apri analisi dettagliata", type="primary", key="bp_open"):
        pid, lato = opts[sel]
        shell.open_analysis(pid, side=lato)

    with st.expander("Come leggere questa pagina"):
        st.markdown(
            "- **Probabilità**: quanto spesso il modello stima che la giocata riesca. Non è una certezza.\n"
            "- **Quota equa**: la quota a cui la giocata sarebbe a pareggio. Se quella offerta è più alta, la giocata ha *valore*.\n"
            f"- **Guadagno atteso**: quanto si guadagna in media per ogni {stake:g} € puntati, se la stessa giocata fosse ripetuta moltissime volte. Su una singola scommessa si può perdere.\n"
            "- **Affidabilità**:\n"
            + "\n".join(f"    - **{k}**: {v}." for k, v in RELIABILITY_HELP.items()) + "\n"
            "- Il modello non conosce infortuni, rotazioni o notizie dell'ultimo minuto: controlla sempre prima di giocare.\n"
            "- Nessuna giocata è sicura o garantita. L'app non piazza scommesse."
        )
    age = ctx.bundle.updated_at
    if age is not None:
        from ...models import utc_now
        st.caption(f"Dati aggiornati {fmt_age((utc_now() - age).total_seconds() / 60)}.")
