"""Impostazioni e API: stato chiavi (mai i valori), test di connessione, aggiornamento live, import CSV, parametri."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from ...adapters.balldontlie import BalldontlieAdapter
from ...adapters.csv_odds import import_odds_csv
from ...adapters.nba_api_adapter import NbaApiAdapter
from ...adapters.the_odds_api import DEFAULT_MARKETS, TheOddsApiAdapter, estimate_credits
from ...model.config import ModelConfig
from .. import shell
from ..labels import fmt_dt
from ..live import CollectOptions, add_manual_quotes
from ..session import settings_or_none
from ..snapshot import cache_dir, snapshot_path

ROOT = Path(__file__).resolve().parents[4]


def _test_connections(settings) -> None:
    results = []
    with st.spinner("Test delle connessioni…"):
        if settings.balldontlie_api_key:
            a = BalldontlieAdapter(settings.balldontlie_api_key, requests_per_min=settings.balldontlie_requests_per_min,
                                   timeout=settings.http_timeout, max_retries=0)
            try:
                results.append(a.test_connection())
            finally:
                a.close()
        if settings.odds_api_key:
            o = TheOddsApiAdapter(settings.odds_api_key, regions=settings.odds_regions, timeout=settings.http_timeout, max_retries=0)
            try:
                results.append(o.test_connection())
            finally:
                o.close()
        results.append(NbaApiAdapter(sleep_seconds=0).test_connection())
    for r in results:
        (st.success if r.ok else st.warning)(f"**{r.provider}** — {r.status}: {r.detail}")


def render() -> None:
    ctx = shell.begin("Impostazioni e API", "Chiavi, connessioni, aggiornamento dei dati live, import CSV e parametri del modello")
    b = ctx.bundle
    settings, err = settings_or_none()
    t1, t2, t3, t4 = st.tabs(["API e chiavi", "Dati live", "Import CSV", "Parametri e dati salvati"])

    with t1:
        if err:
            st.error(f"Configurazione non valida: {err}")
        else:
            st.markdown("**Stato delle chiavi** (i valori non vengono mai mostrati)")
            st.dataframe(pd.DataFrame([{"Variabile": k, "Stato": v} for k, v in settings.describe().items()]), hide_index=True, width="stretch")
            if st.button("Testa le connessioni", key="st_test"):
                _test_connections(settings)
        with st.expander("Come inserire le chiavi in modo sicuro", expanded=not (settings and settings.secrets())):
            st.markdown(
                "1. Nella cartella del progetto copia `.env.example` in `.env`.\n"
                "2. Apri `.env` con un editor e scrivi le chiavi dopo il segno `=` (es. `THE_ODDS_API_KEY=...`).\n"
                "3. Riavvia l'app. Il file `.env` è escluso da Git: **non incollare mai le chiavi in chat né nel codice**.\n\n"
                "Piani e limiti: vedi `docs/API_PROVIDERS.md` e `docs/SETUP_API.md`."
            )

    with t2:
        if b.is_demo:
            st.info("Sei in modalità Demo: il prelievo live non è attivo. Passa a «Live (dati API)» dalla barra laterale.")
        else:
            c1, c2, c3 = st.columns(3)
            days = c1.number_input("Giorni in avanti", 0, 7, 1, key="lv_days")
            events = c2.number_input("Partite al massimo", 1, 15, 8, key="lv_events")
            players = c3.number_input("Giocatori con storico (max)", 5, 120, 30, key="lv_players")
            u1, u2 = st.columns(2)
            use_odds = u1.toggle("The Odds API (quote)", value=True, key="lv_odds")
            use_bdl = u2.toggle("BALLDONTLIE (props, piano GOAT)", value=True, key="lv_bdl")
            if settings:
                est = estimate_credits(DEFAULT_MARKETS, settings.odds_regions) * int(events)
                st.caption(f"Stima crediti The Odds API: ~{est} (mercati × regioni × partite, secondo la documentazione: **Da verificare** sul tuo piano). "
                           "Il prelievo parte solo quando premi il pulsante: le pagine non chiamano mai le API da sole.")
            if st.button("Aggiorna dati live", type="primary", key="lv_go"):
                shell.run_refresh(CollectOptions(days_ahead=int(days), max_events=int(events), max_players=int(players),
                                                 use_odds_api=use_odds, use_balldontlie_props=use_bdl))
                st.rerun()
        st.markdown("**Stato dei provider (ultimo prelievo)**")
        if b.providers:
            st.dataframe(pd.DataFrame([{"Provider": p.name, "Stato": p.status, "Quote": p.n_quotes, "Dettaglio": p.detail} for p in b.providers]),
                         hide_index=True, width="stretch")
        else:
            st.info("Nessun prelievo effettuato.")
        for level, text in b.messages:
            if not text.startswith("MODALITÀ DEMO"):
                getattr(st, level if level in ("info", "warning", "error") else "info")(text)

    with t3:
        if b.is_demo:
            st.info("L'import CSV è disponibile solo in modalità Live: in demo non si mescolano mai dati reali e sintetici.")
        else:
            st.markdown("Importa quote di bookmaker non coperti dalle API (Snai, Sisal, Eurobet, Lottomatica, Goldbet…). "
                        "Colonne richieste: `bookmaker, event_date, home_team, away_team, player, market, line, side, odds, timestamp`.")
            template = ROOT / "data" / "templates" / "odds_template.csv"
            if template.exists():
                st.download_button("Scarica il modello CSV", template.read_bytes(), "odds_template.csv", "text/csv")
            up = st.file_uploader("File CSV", type=["csv"], key="csv_up")
            if up is not None:
                with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as tmp:
                    tmp.write(up.getvalue())
                res = import_odds_csv(tmp.name)
                Path(tmp.name).unlink(missing_ok=True)
                st.success(f"{len(res.quotes)} quote valide, {len(res.errors)} righe scartate.")
                if res.naive_timestamps:
                    st.warning(f"{res.naive_timestamps} timestamp senza fuso orario: interpretati come Europe/Rome.")
                if res.errors:
                    st.dataframe(pd.DataFrame([{"Riga": e.line, "Motivo": e.message} for e in res.errors]), hide_index=True, width="stretch")
                if res.quotes and st.button("Aggiungi alle quote live", type="primary", key="csv_add"):
                    add_manual_quotes(res.quotes)
                    st.success("Quote aggiunte. Aggiorna i dati live per scaricare anche lo storico dei giocatori coinvolti.")
                    st.rerun()

    with t4:
        c1, c2 = st.columns(2)
        c1.number_input("Repliche bootstrap (intervalli)", 0, 300, key="n_boot", step=10,
                        help="Più repliche = intervalli più stabili ma calcolo più lento. 0 = nessun intervallo.")
        c2.select_slider("Livello degli intervalli", [0.80, 0.90, 0.95], key="ci_level")
        st.caption("Stake e soglia di obsolescenza si impostano nella barra laterale.")
        st.markdown("**Iperparametri del modello** (scelte dichiarate, non ottimizzate su dati reali)")
        cfg = {k: v for k, v in asdict(ModelConfig()).items() if k != "ymax"}
        st.dataframe(pd.DataFrame([{"Parametro": k, "Valore": str(v)} for k, v in cfg.items()]), hide_index=True, width="stretch", height=300)
        st.markdown("**Dati salvati in locale**")
        sp = snapshot_path()
        if sp.exists():
            st.write(f"Snapshot live: `{sp}` — {sp.stat().st_size / 1024:.0f} KB, aggiornato {fmt_dt(b.updated_at) if b.mode == 'live' else '—'}")
        else:
            st.write("Nessuno snapshot live salvato.")
        st.caption(f"Cartella cache: `{cache_dir()}` (esclusa da Git).")
        if st.button("Svuota la cache di calcolo", key="clear_cache"):
            st.cache_resource.clear()
            st.success("Cache svuotata.")
