"""Copertura dati: storico per giocatore, quote valutabili e prezzate, motivi di esclusione, stato del recupero."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import shell
from .. import theme as T
from ..coverage import COMPLETE, INSUFFICIENT, NONE, PARTIAL, PLAYER_COLUMNS, STATE_HELP, STATES, build_coverage
from ..labels import fmt_dt


def _retrieval(cov, b) -> None:
    r = cov.retrieval
    st.subheader("Stato del recupero dati")
    if b.is_demo:
        st.info("In modalità Demo lo storico è sintetico e non viene scaricato da nessuna API.")
        return
    if not r:
        st.info("Nessun recupero registrato: premi «Aggiorna dati live» nella barra laterale.")
        return
    seasons = ", ".join(f"{x}-{str(x + 1)[2:]}" for x in sorted(r.get("seasons", []), reverse=True)) or "—"
    pending = r.get("pending", [])
    status, color = ("Completo", T.GREEN) if r.get("complete") else ("Incompleto", T.AMBER)
    T.kpi_row([
        ("Stato", status, f"ultimo recupero: {fmt_dt(b.updated_at)}", color),
        ("Stagioni richieste", seasons, "storico regolare, solo partite terminate", None),
        ("Giocatori richiesti", f"{r.get('players_requested', 0)} / {r.get('players_in_quotes', 0)}",
         f"{r.get('with_logs', 0)} con almeno una partita", None),
        ("Da cache / incrementali", f"{r.get('reused_from_cache', 0)} / {r.get('incremental', 0)}",
         f"{r.get('requests_bdl') if r.get('requests_bdl') is not None else '—'} richieste a BALLDONTLIE · {r.get('elapsed_s', 0)} s", None),
    ])
    if pending:
        st.warning(f"{len(pending)} giocatori sono ancora in attesa (limite di tempo): premi di nuovo «Aggiorna dati live», "
                   "il recupero riparte da dove si era fermato senza ripetere quanto già scaricato.")
    if r.get("tier_blocked"):
        st.error("BALLDONTLIE ha rifiutato le statistiche per il piano attuale: nessuno storico può essere scaricato da lì.")
    if r.get("nba_api_disabled"):
        st.warning("nba_api non ha risposto ed è stata disattivata per questo aggiornamento.")


def render() -> None:
    ctx = shell.begin("Copertura dati", "Quanto storico c'è per ogni giocatore, quante quote sono valutabili e perché alcune non vengono prezzate")
    b, board = ctx.bundle, ctx.board
    if b.is_empty or not b.quotes:
        for level, text in b.messages:
            getattr(st, level if level in ("info", "warning", "error") else "info")(text)
        st.info("Nessuna quota disponibile: la copertura si calcola sui giocatori che compaiono nelle quote.")
        _retrieval(build_coverage(b, None), b)
        return
    cov = build_coverage(b, board)
    sc = cov.state_counts
    T.kpi_row([
        ("Giocatori nelle quote", f"{cov.n_players}", f"{cov.n_with_history} con storico utilizzabile", None),
        ("Storico completo", f"{sc[COMPLETE]}", f"≥ {cov.low_sample} partite", T.GREEN if sc[COMPLETE] else None),
        ("Storico parziale", f"{sc[PARTIAL]}", f"{cov.min_games}–{cov.low_sample - 1} partite (campione ridotto)", T.AMBER if sc[PARTIAL] else None),
        ("Insufficiente / senza storico", f"{sc[INSUFFICIENT]} / {sc[NONE]}", f"meno di {cov.min_games} partite · non valutabili",
         T.RED if sc[INSUFFICIENT] + sc[NONE] else None),
    ])
    st.write("")
    T.kpi_row([
        ("Quote totali", f"{cov.n_quotes}", "tutte le quote caricate", None),
        ("Quote valutabili", f"{cov.n_evaluable}", "giocatori con storico sufficiente", None),
        ("Quote prezzate", f"{cov.n_priced if cov.n_priced is not None else '—'}",
         "con probabilità, quota equa ed EV", T.GREEN if cov.n_priced else T.AMBER),
        ("Quote non prezzate", f"{cov.n_quotes - (cov.n_priced or 0)}", "motivi nella tabella sotto", None),
    ])
    st.write("")
    if cov.n_priced == 0:
        st.warning("Nessuna quota è prezzata: nessun giocatore ha abbastanza partite. Nessuna probabilità viene inventata per colmare il vuoto.")

    _retrieval(cov, b)

    st.subheader("Motivi di esclusione")
    if cov.reasons.empty:
        st.success("Tutte le quote sono state prezzate.")
    else:
        st.dataframe(cov.reasons.rename(columns={"motivo": "Motivo", "quote": "Quote", "giocatori": "Giocatori"}),
                     hide_index=True, width="stretch")
    with st.expander("Come si classifica lo storico"):
        for s in STATES:
            st.markdown(f"- **{s}**: {STATE_HELP[s]}")

    st.subheader("Giocatori")
    chosen = st.multiselect("Stato dello storico", STATES, key="cov_state", placeholder="Tutti")
    d = cov.players if not chosen else cov.players[cov.players["stato"].isin(chosen)]
    search = st.text_input("Cerca giocatore", key="cov_search")
    if search:
        d = d[d["giocatore"].str.contains(search, case=False, na=False)]
    st.dataframe(d[PLAYER_COLUMNS].rename(columns={"giocatore": "Giocatore", "squadra": "Squadra", "stato": "Stato", "partite": "Partite",
                                                   "ultima_partita": "Ultima partita", "stagioni": "Stagioni", "quote": "Quote",
                                                   "prezzate": "Prezzate", "nota": "Nota"}),
                 hide_index=True, width="stretch", height=480)
    st.caption(f"{len(d)} giocatori mostrati. Le partite contate sono quelle con minuti giocati, precedenti alla gara quotata.")
