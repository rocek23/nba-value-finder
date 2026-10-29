"""Bet Builder AI: multiple costruite dal modello per una quota obiettivo (analisi statistica, nessuna scommessa automatica)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...models import utc_now
from .. import builder as B
from .. import shell
from .. import tracker as K
from ..best import quote_age_text
from ..labels import to_local

CSS = """
<style>
.bb-head{display:flex;align-items:center;gap:.9rem;margin:.2rem 0 .8rem 0}
.bb-head .t{font-size:2rem;font-weight:700;line-height:1.1}
.bb-head .s{opacity:.65;font-size:1.02rem}
.bb-h{font-size:1.25rem;font-weight:600;margin:1.1rem 0 .5rem 0}
.st-key-bb_targets button{height:3.6rem;border-radius:18px;font-size:1.5rem;font-weight:600;background:#1b2540;border:1px solid #2b3655;color:#e5e9f2}
.st-key-bb_targets button[kind="primary"]{background:#2c4fd6;border:2px solid #6b9bff;color:#fff}
.st-key-bb_go button{height:3.2rem;border-radius:999px;background:#eeeeee;border:0;color:#111;font-size:1.1rem;font-weight:600}
.st-key-bb_targets button p{font-size:1.45rem;font-weight:600}
.st-key-bb_go button p{font-size:1.05rem;font-weight:600}
.st-key-bb_go button:hover{background:#fff;color:#000}
.bb-pill{display:inline-block;padding:.12rem .6rem;border-radius:999px;background:#1b2540;border:1px solid #2b3655;font-size:.78rem;margin-right:.35rem}
.bb-metric-fix{}
[data-testid='stMetricValue']{font-size:1.6rem}
.bb-odds{font-size:2.3rem;font-weight:700;line-height:1}
</style>
"""


def _pct(p: float) -> str:
    v = 100 * p
    return f"{v:.2f}%" if v < 1 else f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def _once_every(p: float) -> str:
    return f"circa 1 volta su {1 / p:,.0f}".replace(",", ".") if p > 0 else "praticamente mai"


def _settled_count(ctx) -> int | None:
    try:
        df = K.track_frame(K.load_picks(), K.load_results(), ctx.bundle.logs)
        return int(len(K.settled(df)))
    except Exception:
        return None


def _validation_note(ctx) -> str:
    n = _settled_count(ctx)
    base = "Le probabilità del modello **non sono ancora sufficientemente validate**"
    if n is None:
        return base + ": non è stato possibile leggere il registro delle giocate chiuse."
    return base + f": il registro ha {n} giocate chiuse e ne servono almeno {K.MIN_FOR_VERDICT} per un primo verdetto (pagina «Registro e taratura»)."


def _details(ctx, leg: B.Leg) -> None:
    stats = B.leg_stats(ctx.bundle.logs.get(leg.key, []), leg)
    st.markdown(B.simple_reason(leg, stats))
    rows = []
    for w, v in stats["finestre"].items():
        rows.append({"Periodo": f"Ultime {w}", "Partite": v["n"],
                     f"Giocata riuscita ({leg.lato} {leg.linea:g})": "—" if v["riuscita"] is None else f"{100 * v['riuscita']:.0f}%",
                     "Media": "—" if v["media"] is None else f"{v['media']:.1f}", "Minuti medi": "—" if v["minuti"] is None else f"{v['minuti']:.1f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pred = ctx.board.predictions.get(leg.key) if ctx.board is not None else None
    if pred is not None:
        st.markdown(f"**Minuti previsti dal modello:** {pred.fit.minutes_mean:.1f}")
    else:
        st.markdown("**Minuti previsti:** non disponibili")
    vs = stats["contro_avversario"]
    if vs is None:
        st.markdown("**Contro l'avversario:** non disponibile (squadra o avversario non identificabili con certezza).")
    elif vs["n"] == 0:
        st.markdown(f"**Contro {vs['avversario']}:** nessuna partita nello storico caricato.")
    else:
        st.markdown(f"**Contro {vs['avversario']}:** {vs['n']} partite, media {vs['media']:.1f}, giocata riuscita {100 * vs['riuscita']:.0f}% "
                    "(campione piccolo; **non** usato nel calcolo del modello).")
    if leg.infortunio:
        st.markdown(f"**Assenze:** ⚠ segnalato «{leg.infortunio}» nell'elenco infortuni. Il modello **non** ne tiene conto: controlla prima di giocare.")
    elif ctx.bundle.injuries is None:
        st.markdown("**Assenze:** elenco infortuni non scaricato (premi «Aggiorna dati live»). Controllale prima di giocare.")
    else:
        st.markdown("**Assenze:** nessuna segnalazione per questo giocatore nell'elenco infortuni BALLDONTLIE. Controlla comunque prima di giocare.")


def _proposal(ctx, p: B.Proposal, stake: float, idx: int) -> None:
    with st.container(border=True):
        st.markdown("".join(f'<span class="bb-pill">{l}</span>' for l in p.labels), unsafe_allow_html=True)
        c = st.columns([1.2, 1, 1, 1, 1])
        c[0].markdown(f'<div class="bb-odds">{p.odds:.2f}</div><div style="opacity:.65;font-size:.85rem">quota totale · {p.n} selezioni</div>',
                      unsafe_allow_html=True)
        c[1].metric("Probabilità totale", _pct(p.p_all), help=f"Prodotto delle probabilità di ogni selezione (una per partita). Intervallo: {_pct(p.p_lo)}–{_pct(p.p_hi)}. "
                    "È una stima, non una certezza.")
        c[2].metric("EV centrale", f"{100 * p.ev:+.1f}%", help="Guadagno atteso medio per euro puntato secondo il modello, ripetendo molte volte la stessa multipla.")
        c[3].metric("EV prudente", f"{100 * p.ev_prudent:+.1f}%", help="Come l'EV centrale, ma con ogni probabilità al limite basso del proprio intervallo di incertezza.")
        c[4].metric("Incertezza", p.uncertainty, help="Convenzione semplice: rapporto tra stima alta e stima bassa della probabilità congiunta "
                    f"({p.p_hi / p.p_lo:.1f}×). Bassa ≤1,5× · Media ≤2,5× · Alta oltre.")
        st.caption(f"Con {stake:g} € la vincita potenziale sarebbe {stake * p.odds:,.2f} € (profitto {stake * (p.odds - 1):,.2f} €); "
                   f"la multipla riesce {_once_every(p.p_all)} secondo il modello.")
        lines = pd.DataFrame([{
            "Giocatore": l.giocatore, "Partita": f"{l.evento} · {to_local(l.inizio).strftime('%d/%m %H:%M')}", "Giocata": l.descr,
            "Quota": l.odds, "Bookmaker": l.book, "Orario quota": quote_age_text(l.eta_min, l.quote_known)} for l in p.legs])
        st.dataframe(lines, hide_index=True, width="stretch", column_config={"Quota": st.column_config.NumberColumn("Quota", format="%.2f")})
        if p.mixed:
            st.warning(f"Le quote vengono da bookmaker diversi ({', '.join(p.books)}): la multipla potrebbe **non essere disponibile** presso un solo operatore.")
        else:
            st.caption(f"Tutte le quote sono di {p.book}.")
        flagged = [l for l in p.legs if l.infortunio]
        if flagged:
            st.warning("Possibile assenza: " + ", ".join(f"{l.giocatore} ({l.infortunio})" for l in flagged) + ". Il modello non ne tiene conto.")
        if p.has_push_risk:
            st.caption("Una o più linee sono intere: in caso di rimborso la selezione esce dalla multipla e la quota totale scende.")
        if p.target >= B.HIGH_TARGET:
            st.warning(f"Quota alta: la probabilità di vincita è molto bassa ({_pct(p.p_all)}). Nella gran parte dei casi la puntata va persa.")
        for l in p.legs:
            with st.expander(f"{l.giocatore} · {l.descr} @ {l.odds:.2f} ({l.book})"):
                _details(ctx, l)


def render() -> None:
    ctx = shell.begin("Bet Builder AI", None, show_title=False)
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="bb-head"><span style="font-size:1.9rem;color:#5B8CFF">☰</span><div><div class="t">Bet Builder AI</div>'
                '<div class="s">Costruisci una multipla basata sulle statistiche NBA</div></div></div>', unsafe_allow_html=True)
    shell.need_quotes(ctx)
    rows = ctx.rows
    stake = float(st.session_state["stake"])
    show_under = bool(st.session_state.get("show_under", False))
    st.session_state.setdefault("bb_target", 2)

    st.markdown('<div class="bb-h">1. Scegli la quota desiderata</div>', unsafe_allow_html=True)
    with st.container(key="bb_targets"):
        for row in (B.TARGETS[:3], B.TARGETS[3:]):
            for col, t in zip(st.columns(3), row):
                if col.button(f"x{t}", key=f"bb_t{t}", type="primary" if st.session_state["bb_target"] == t else "secondary", use_container_width=True):
                    st.session_state["bb_target"] = t
                    st.rerun()
    target = int(st.session_state["bb_target"])

    st.markdown('<div class="bb-h">2. Strategia di selezione</div>', unsafe_allow_html=True)
    strat = st.radio("Strategia", ["Miglior valore statistico stimato", "Maggiore probabilità di successo", "Meno selezioni possibili"],
                     key="bb_strat", label_visibility="collapsed",
                     help="Le tre proposte vengono sempre cercate: questa scelta decide quale viene mostrata per prima.")
    first = {"Miglior valore statistico stimato": B.LABEL_EV, "Maggiore probabilità di successo": B.LABEL_PROB, "Meno selezioni possibili": B.LABEL_FEW}[strat]

    st.markdown('<div class="bb-h">3. Mercati da includere</div>', unsafe_allow_html=True)
    mk = st.selectbox("Mercati", ["Tutti i mercati"] + sorted(rows["mercato"].unique()), key="bb_mkt", label_visibility="collapsed")
    markets = None if mk == "Tutti i mercati" else [mk]

    with st.expander("Filtri avanzati"):
        a1, a2 = st.columns(2)
        sel_min, sel_max = a1.slider("Numero di selezioni (min–max)", 1, 12, (2, 8), key="bb_sel",
                                     help="Una sola selezione per partita: il massimo reale dipende dalle partite del giorno.")
        tol = a2.slider("Tolleranza rispetto alla quota obiettivo", 5, 30, 15, step=5, format="±%d%%", key="bb_tol")
        books = st.multiselect("Bookmaker", sorted(rows["bookmaker"].unique()), key="bb_books", placeholder="Tutti i bookmaker disponibili")
        mixed = st.toggle("Consenti bookmaker diversi nella stessa multipla", value=False, key="bb_mixed",
                          help="Spento (consigliato): ogni multipla usa un solo bookmaker. Acceso: la quota totale potrebbe non esistere presso nessun operatore.")
        prud = st.toggle("Mostra solo multiple con EV prudente positivo", value=False, key="bb_prud")
        st.caption("Gli Under sono inclusi solo se attivi «Mostra anche gli Under» nella barra laterale.")

    now = utc_now()
    legs, funnel = B.eligible_legs(rows, now, markets=markets, books=books or None, include_under=show_under)
    n_games = len({l.game for l in legs})
    n_books = len({l.book for l in legs})
    st.caption(f"Selezioni valide: **{len(legs)}** · partite: **{n_games}** · bookmaker: **{n_books}**. "
               "Si usano solo quote reali recenti di partite non ancora iniziate, con almeno 25 partite di storico.")
    with st.expander("Controllo dati"):
        st.dataframe(pd.DataFrame(funnel, columns=["Passaggio", "Quote rimaste"]), hide_index=True, width="stretch")
        st.caption("Una selezione resta solo se supera tutti i passaggi. Nessuna probabilità, quota o partita viene inventata: ciò che manca resta mancante.")

    go = st.container(key="bb_go")
    if go.button(f"Genera multipla quota {target}", key="bb_gen", icon=":material/auto_awesome:", use_container_width=True):
        with st.spinner("Cerco le combinazioni migliori…"):
            res = B.search(legs, float(target), tolerance=tol / 100, min_sel=sel_min, max_sel=sel_max, mixed_books=mixed,
                           require_prudent=prud, books_filter=books or None)
        st.session_state["bb_result"] = {"target": target, "res": res}
    st.caption("Strumento di analisi statistica: non piazza scommesse e non garantisce alcun risultato.")

    saved = st.session_state.get("bb_result")
    if not saved:
        return
    res: B.BuildResult = saved["res"]
    if saved["target"] != target:
        st.info(f"Il risultato qui sotto è per la quota {saved['target']}: premi «Genera multipla quota {target}» per aggiornarlo.")
    st.divider()
    if not res.proposals:
        st.error(f"Nessuna multipla affidabile per la quota {saved['target']}.")
        for r in res.reasons:
            st.markdown(f"- {r}")
        return
    st.warning(_validation_note(ctx))
    if saved["target"] >= B.HIGH_TARGET:
        st.warning(f"Quota {saved['target']}: la probabilità di vincita può essere molto bassa. Usa questa pagina per capire i rischi, non per inseguire vincite.")
    props = sorted(res.proposals, key=lambda p: first not in p.labels)
    for i, p in enumerate(props):
        _proposal(ctx, p, stake, i)
    st.caption("Come si calcola: quota totale = prodotto delle quote · probabilità che vincano tutte = prodotto delle probabilità, ammesso perché c'è una sola "
               "selezione per partita (indipendenza tra partite diverse: **Da verificare**) · EV = prodotto di (probabilità × quota + probabilità di rimborso) − 1.")
