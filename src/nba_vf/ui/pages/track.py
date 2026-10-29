"""Registro e taratura: le giocate consigliate in passato e come sono andate davvero."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, shell
from .. import theme as T
from ..live import settle_now
from ..session import settings_or_none
from .. import tracker as K


@st.cache_data(ttl=60, show_spinner=False)
def _load(_nonce: int) -> tuple[list[dict], dict[str, dict]]:
    return K.load_picks(), K.load_results()


def _bump() -> None:
    st.session_state["trk_nonce"] = st.session_state.get("trk_nonce", 0) + 1
    st.cache_data.clear()


def render() -> None:
    ctx = shell.begin("Registro e taratura", "Come sono andate davvero le giocate consigliate: l'unico modo per sapere se le percentuali sono credibili")
    if ctx.bundle.is_demo:
        st.info("Il registro funziona solo con i dati **Live**: i dati demo sono inventati e non vengono mai registrati. "
                "Passa a «Live (dati API)» nella barra laterale.")
        return
    stake = float(st.session_state["stake"])
    records, results = _load(st.session_state.get("trk_nonce", 0))
    df = K.track_frame(records, results, ctx.bundle.logs)

    c1, c2 = st.columns([3, 1])
    c1.caption("Ogni volta che apri l'app dopo un aggiornamento, le giocate con valore di partite non ancora iniziate vengono salvate (una sola volta, "
               "con la quota di quel momento). A partita finita vengono chiuse con la statistica reale.")
    if c2.button("Aggiorna risultati", use_container_width=True, help="Chiude le giocate delle partite finite (poche richieste alle API)."):
        settings, err = settings_or_none()
        if settings is None:
            st.error(f"Configurazione non valida: {err}")
        else:
            with st.spinner("Controllo i risultati…"):
                rep = settle_now(settings)
            if rep is None:
                st.error("Non sono riuscito a controllare i risultati. Riprova tra poco.")
            else:
                st.success(f"Chiuse {rep.settled} giocate (annullate: {rep.void}; ancora in attesa: {rep.waiting}).")
            _bump()
            st.rerun()

    if df.empty:
        st.info("Il registro è ancora vuoto. Appena ci sono giocate con valore in «Migliori giocate» e la partita non è iniziata, "
                "compaiono qui da sole. Dopo la partita premi «Aggiorna risultati».")
        return

    st.download_button("Scarica il registro (CSV)", K.export_csv(df), file_name="registro_giocate.csv", mime="text/csv", key="trk_csv",
                       help="Tutte le giocate registrate con quota, probabilità del modello ed esito: utile per analizzare dove il modello sbaglia.")
    sm = K.summarize(df)
    hit = f"{100 * sm.wins / sm.n_settled:.0f}%" if sm.n_settled else "—"
    exp_hit = f"{100 * sm.expected_wins / sm.n_settled:.0f}%" if sm.n_settled else "—"
    T.kpi_row([
        ("Giocate registrate", f"{sm.n_logged}", f"{sm.n_settled} chiuse · {sm.n_pending} in attesa · {sm.n_void} annullate", None),
        ("Vinte (previsto)", f"{hit} ({exp_hit})", f"{sm.wins} vinte, {sm.losses} perse, {sm.pushes} rimborsi", None),
        (f"Risultato su {stake:g} € a giocata", f"{stake * sm.profit_units:+.2f} €", f"atteso dal modello {stake * sm.expected_units:+.2f} €",
         (T.GREEN if sm.profit_units >= 0 else T.RED) if sm.n_settled else None),
    ])
    st.write("")
    level, text = K.verdict(sm)
    getattr(st, level)(text)

    t1, t2 = st.tabs(["Registro", "Taratura"])
    with t1:
        f1, f2 = st.columns(2)
        esiti = f1.multiselect("Esito", [K.WIN, K.LOSS, K.PUSH, K.PENDING, K.VOID], key="trk_esito", placeholder="Tutti")
        rel = f2.multiselect("Affidabilità", ["Alta", "Media", "Bassa"], key="trk_rel", placeholder="Tutte")
        d = df
        if esiti:
            d = d[d["esito"].isin(esiti)]
        if rel:
            d = d[d["affidabilita"].isin(rel)]
        show = d[K.TRACK_COLUMNS].copy()
        show["profitto"] = stake * show["profitto"].astype(float)
        st.dataframe(show, hide_index=True, width="stretch", height=min(520, 38 + 35 * len(show)), column_config={
            "registrata": st.column_config.DatetimeColumn("Registrata", format="DD/MM HH:mm"),
            "giocatore": "Giocatore", "squadra": "Squadra", "partita": "Partita",
            "inizio": st.column_config.DatetimeColumn("Inizio (Roma)", format="DD/MM HH:mm"),
            "giocata": "Giocata", "bookmaker": "Bookmaker", "quota": st.column_config.NumberColumn("Quota", format="%.2f"),
            "prob": st.column_config.NumberColumn("Probabilità prevista", format="%.0f%%"),
            "affidabilita": "Affidabilità", "esito": "Esito",
            "risultato": st.column_config.NumberColumn("Valore reale", format="%d"),
            "profitto": st.column_config.NumberColumn(f"Risultato ({stake:g} €)", format="%+.2f €"),
        })
        st.caption("«Annullata»: il giocatore non è sceso in campo o la statistica non è disponibile; non conta nelle percentuali.")
    with t2:
        if sm.n_settled == 0:
            st.info("La taratura compare appena almeno una giocata è stata chiusa.")
        else:
            tab = K.calibration_table(df)
            st.plotly_chart(charts.fig_pick_calibration(tab), width="stretch")
            st.caption("Se i punti stanno sulla linea tratteggiata, le percentuali sono ben tarate. Le barre sono l'intervallo di incertezza al 90% "
                       "sul valore realizzato: con poche giocate sono molto larghe, ed è normale.")
            nice = tab.rename(columns={"gruppo": "Fascia di probabilità", "giocate": "Giocate", "previsto": "Previsto %", "realizzato": "Realizzato %",
                                       "da": "Realizzato min %", "a": "Realizzato max %"})
            st.dataframe(nice, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(c, format="%.0f") for c in ("Previsto %", "Realizzato %", "Realizzato min %", "Realizzato max %")})
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Per affidabilità**")
                st.dataframe(K.calibration_table(df, "affidabilita").rename(columns={"gruppo": "Affidabilità", "giocate": "Giocate", "previsto": "Previsto %",
                             "realizzato": "Realizzato %"})[["Affidabilità", "Giocate", "Previsto %", "Realizzato %"]],
                             hide_index=True, width="stretch", column_config={
                    c: st.column_config.NumberColumn(c, format="%.0f") for c in ("Previsto %", "Realizzato %")})
            with g2:
                st.markdown("**Per mercato**")
                st.dataframe(K.calibration_table(df, "mercato").rename(columns={"gruppo": "Mercato", "giocate": "Giocate", "previsto": "Previsto %",
                             "realizzato": "Realizzato %"})[["Mercato", "Giocate", "Previsto %", "Realizzato %"]],
                             hide_index=True, width="stretch", column_config={
                    c: st.column_config.NumberColumn(c, format="%.0f") for c in ("Previsto %", "Realizzato %")})
            st.caption("Il registro contiene solo le giocate che il modello ha consigliato (quelle con valore positivo): misura quindi proprio le stime "
                       "che userai. I rimborsi non contano come vittorie.")
