"""Struttura comune delle pagine: barra laterale, banner modalita', intestazione, navigazione verso il dettaglio."""

from __future__ import annotations

import streamlit as st

from ..models import utc_now
from . import theme as T
from . import tracker
from .labels import fmt_age, fmt_dt
from .live import CollectOptions, refresh_live
from .session import MODES, Ctx, get_board, get_bundle, init_state, rows_view, settings_or_none

APP_NAME = "NBA Value Finder"
DISCLAIMER = (
    "Strumento di analisi statistica: non inserisce scommesse e non garantisce alcun risultato. "
    "Le stime hanno incertezza; valuta sempre in autonomia."
)


def run_refresh(opts: CollectOptions) -> None:
    """Aggiorna lo snapshot live dalle API (unica via per chiamare la rete). Mostra l'esito, non lo nasconde."""
    settings, err = settings_or_none()
    if settings is None:
        st.error(f"Configurazione non valida: {err}")
        return
    bar = st.progress(0.0, text="Avvio…")
    try:
        snap = refresh_live(settings, opts, progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
        st.success(f"Aggiornamento completato: {len(snap.quotes)} quote, {len(snap.logs)} giocatori con storico.")
    except Exception as exc:  # mostra l'errore, senza chiavi (gli adapter le oscurano)
        st.error(f"Aggiornamento non riuscito: {type(exc).__name__}: {exc}")
    finally:
        bar.empty()


def _sidebar(bundle) -> None:
    with st.sidebar:
        st.markdown(f"### 🏀 {APP_NAME}")
        st.radio("Modalità dati", list(MODES), key="mode_label",
                 help="Demo: dati sintetici inventati. Live: ultimo prelievo dalle API reali (nessun dato inventato).")
        age = (utc_now() - bundle.updated_at).total_seconds() / 60 if bundle.updated_at else None
        color = T.GREEN if bundle.status == "ok" else T.AMBER if bundle.status == "parziale" else T.RED
        st.markdown(T.badge(bundle.status.upper(), color) + T.badge(bundle.source_label, T.MUTED), unsafe_allow_html=True)
        st.caption(f"Aggiornato: {fmt_dt(bundle.updated_at)} ({fmt_age(age)})")
        if bundle.mode == "live":
            if st.button("Aggiorna dati live", use_container_width=True, type="primary",
                         help="Chiama le API configurate (consuma crediti/richieste)."):
                run_refresh(CollectOptions())
                st.rerun()
        st.divider()
        st.markdown("**Parametri di visualizzazione**")
        st.number_input("Stake per l'EV (€)", min_value=1.0, max_value=100000.0, step=1.0, key="stake")
        st.number_input("Quota obsoleta dopo (min)", min_value=1.0, max_value=1440.0, step=5.0, key="max_age")
        st.divider()
        st.caption(DISCLAIMER)


def begin(title: str, subtitle: str | None = None, *, show_title: bool = True) -> Ctx:
    """Inizializza la pagina e restituisce i dati. Per i pagina che non hanno nulla da mostrare usare ``need``."""
    init_state()
    T.inject_css()
    bundle = get_bundle()
    _sidebar(bundle)
    if bundle.is_demo:
        st.markdown('<div class="nvf-demo">🧪 <b>MODALITÀ DEMO</b> — tutti i dati sono <b>sintetici</b> e inventati; '
                    'nessun valore corrisponde alla NBA reale.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="nvf-live">● <b>MODALITÀ LIVE</b> — dati provenienti dalle API configurate; '
                    'i dati mancanti restano mancanti (nessun dato inventato).</div>', unsafe_allow_html=True)
    if show_title:
        st.title(title)
        if subtitle:
            st.caption(subtitle)
    board = get_board(bundle) if bundle.quotes else None
    rows = rows_view(bundle, board) if board is not None else None
    if rows is None:
        import pandas as pd

        rows = pd.DataFrame()
    _track(bundle, rows)
    return Ctx(bundle, board, rows)


def _track(bundle, rows) -> None:
    """Salva nel registro le giocate con valore viste in Live (una sola volta per aggiornamento dei dati). Mai in Demo."""
    if bundle.is_demo or rows is None or rows.empty or st.session_state.get("_tracked_fp") == bundle.fingerprint:
        return
    st.session_state["_tracked_fp"] = bundle.fingerprint
    if tracker.record_picks(rows, bundle, utc_now()):
        st.cache_data.clear()


def need_quotes(ctx: Ctx) -> None:
    """Interrompe la pagina con un messaggio chiaro se non ci sono quote (mai dati sostitutivi)."""
    if ctx.bundle.quotes:
        return
    if ctx.bundle.is_demo:
        st.error("Dati demo non disponibili (errore interno).")
    else:
        for level, text in ctx.bundle.messages:
            getattr(st, level if level in ("info", "warning", "error") else "info")(text)
        if not ctx.bundle.messages:
            st.warning("Nessuna quota live disponibile.")
        st.page_link(st.session_state["_pages"]["impostazioni"], label="Vai a «Impostazioni e API»", icon=":material/settings:")
    st.stop()


def open_analysis(prop_id: str | None = None, *, key: str | None = None, side: str = "Over") -> None:
    """Prepara la selezione e passa alla pagina «Analisi giocatore»."""
    st.session_state["sel_prop"] = {"prop_id": prop_id, "key": key, "side": side}
    st.session_state["ap_nonce"] = st.session_state.get("ap_nonce", 0) + 1
    st.switch_page(st.session_state["_pages"]["giocatore"])
