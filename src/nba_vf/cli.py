"""Aggiornamento automatico da riga di comando (usato da GitHub Actions, senza aprire l'app).

    python -m nba_vf.cli refresh   quote + storico + chiusura dei risultati + registrazione delle giocate con valore
    python -m nba_vf.cli settle    solo chiusura dei risultati (poche richieste)

Le chiavi arrivano dalle variabili d'ambiente (Secrets del repository): non vengono mai stampate. Se DATABASE_URL e' impostato,
snapshot, storico quote e registro vengono salvati nel database e l'app li trova gia' aggiornati.
"""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, Settings, redact
from .models import utc_now


def _say(msg: str) -> None:
    print(msg, flush=True)


def run_refresh(settings: Settings, args) -> int:
    from .ui import live, tracker
    from .ui.board import BoardParams, annotate, build_board
    from .ui.snapshot import sync_from_remote

    sync_from_remote()                       # riparte dall'ultimo snapshot (storico gia' scaricato, richieste risparmiate)
    snap = live.refresh_live(settings, live.CollectOptions(days_ahead=args.days_ahead, max_events=args.max_events, time_budget_s=args.time_budget))
    bundle = live.bundle_from_snapshot(snap)
    _say(f"Quote: {len(snap.quotes)} · giocatori con storico: {len(snap.logs)} · infortuni: "
         f"{'non scaricati' if snap.injuries is None else len(snap.injuries)}")
    for p in snap.providers:
        _say(f"  {p['name']}: {p['status']}")
    new = 0
    if bundle.quotes:
        board = build_board(bundle, BoardParams())
        rows = annotate(board.rows, utc_now(), settings.max_quote_age_minutes, 10.0)
        new = tracker.record_picks(rows, bundle, utc_now())
    _say(f"Giocate nuove registrate: {new}")
    return 0


def run_settle(settings: Settings, args) -> int:
    from .ui import live
    from .ui.snapshot import sync_from_remote

    sync_from_remote()
    rep = live.settle_now(settings)
    if rep is None:
        _say("Chiusura risultati non riuscita.")
        return 1
    _say(f"Chiuse {rep.settled} giocate (annullate: {rep.void}; in attesa: {rep.waiting}; errori: {rep.errors})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nba_vf.cli", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["refresh", "settle"])
    ap.add_argument("--days-ahead", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=8, help="partite massime per The Odds API (consumo di crediti)")
    ap.add_argument("--time-budget", type=float, default=900.0, help="secondi massimi per il recupero dello storico")
    args = ap.parse_args(argv)
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        _say(f"Configurazione non valida: {exc}")
        return 2
    if not (settings.balldontlie_api_key or settings.odds_api_key):
        _say("Nessuna chiave API configurata (BALLDONTLIE_API_KEY / THE_ODDS_API_KEY): niente da aggiornare.")
        return 2
    try:
        return run_refresh(settings, args) if args.command == "refresh" else run_settle(settings, args)
    except Exception as exc:  # mai mostrare chiavi nei log
        _say(f"Errore: {type(exc).__name__}: {redact(str(exc), settings.secrets())}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
