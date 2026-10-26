"""Testi di spiegazione e elenco dei dati mancanti (funzioni pure, testabili)."""

from __future__ import annotations

import numpy as np

from ..model.engine import PlayerPrediction
from ..model.fit import FittedPlayer
from ..model.prediction import Valuation
from ..models import Market
from .labels import MARKET_IT, pct

RATE_KEY = {Market.POINTS: ("pts",), Market.REBOUNDS: ("reb",), Market.ASSISTS: ("ast",), Market.PRA: ("pts", "reb", "ast")}


def explain(pred: PlayerPrediction, market: Market, line: float, side: str, v: Valuation, *, season_avg: float | None,
            last10_avg: float | None, hit_all: float | None, hit_20: float | None, market_p_over: float | None) -> list[str]:
    fit: FittedPlayer = pred.fit
    mp = pred.markets[market]
    out = [
        f"Minuti proiettati: {fit.minutes_mean:.1f} (deviazione {fit.minutes_sd:.1f}), media pesata che dà più peso alle partite recenti.",
    ]
    if market in RATE_KEY:
        rate = sum(fit.rho[k] for k in RATE_KEY[market])
        base = ("verso la media dei giocatori caricati" if fit.prior_source == "pool" else "verso la media storica del giocatore")
        out.append(f"Tasso stimato {MARKET_IT[market].lower()}: {rate:.3f} per minuto (shrinkage {base}, perché i campioni piccoli sono rumorosi).")
    elif market is Market.THREES and fit.p3 is not None:
        out.append(f"Tentativi da tre previsti {fit.rho['fg3a'] * fit.minutes_mean:.1f} a partita, percentuale stimata {pct(fit.p3)} "
                   f"({'beta-binomiale: la percentuale varia da partita a partita' if fit.use_bb else 'binomiale: nessuna evidenza di variazione extra tra partite'}).")
    ref = []
    if season_avg is not None:
        ref.append(f"media stagione {season_avg:.1f}")
    if last10_avg is not None:
        ref.append(f"ultime 10 {last10_avg:.1f}")
    out.append(f"Valore atteso del modello: {mp.mean:.1f} (mediana {mp.median})" + (f", contro {', '.join(ref)}." if ref else "."))
    ci = f" [IC {v.ci_level:.0%}: {pct(v.p_win_ci[0])}–{pct(v.p_win_ci[1])}]" if v.p_win_ci else ""
    push = f", rimborso {pct(v.p_push)}" if v.p_push > 0 else ""
    out.append(f"{side.capitalize()} {line:g}: probabilità di vincita {pct(v.p_win)}{ci}{push}. Quota equa {v.fair_odds:.2f}; "
               f"la quota considerata ({v.decimal_odds:.2f}) richiede almeno il {pct(v.breakeven_p_win)} di probabilità di vincita.")
    if hit_all is not None:
        out.append(f"Storico: la linea è stata superata nel {pct(hit_all, 0)} delle partite disponibili"
                   + (f" e nel {pct(hit_20, 0)} delle ultime 20." if hit_20 is not None else "."))
    if market_p_over is not None:
        p_model_over = v.p_win if side == "over" else v.p_loss
        diff = 100 * (p_model_over - market_p_over)
        out.append(f"Mercato (media dei bookmaker, margine rimosso): Over {pct(market_p_over)}; il modello è {abs(diff):.1f} punti "
                   f"{'sopra' if diff > 0 else 'sotto'}. Una differenza grande è più spesso un limite del modello (informazioni mancanti) che un'occasione.")
    if "campione_ridotto" in fit.flags:
        out.append(f"Campione ridotto ({fit.n_games} partite): la stima è più incerta e l'intervallo più largo.")
    out.append("L'intervallo copre solo l'incertezza di stima dei parametri, non un modello sbagliato. Nessuna stima garantisce un risultato.")
    return out


def missing_data(pred: PlayerPrediction | None, *, n_logs: int, skipped_reason: str | None, issues: list[str], market: Market,
                 n_books: int, n_stale: int, has_opponent: bool, mode: str) -> list[tuple[str, str]]:
    """Elenco (livello, testo). Livelli: 'ok', 'warn', 'missing'."""
    items: list[tuple[str, str]] = []
    if n_logs == 0:
        items.append(("missing", "Storico partite: assente. Senza storico non si può stimare nulla."))
    elif skipped_reason:
        items.append(("missing", f"Stima non calcolabile: {skipped_reason}."))
    else:
        items.append(("ok", f"Storico partite: {n_logs} partite usate."))
    for i in issues:
        items.append(("warn", f"Raccolta dati: {i}"))
    if pred is not None:
        flags = pred.fit.flags
        if "campione_ridotto" in flags:
            items.append(("warn", "Campione ridotto: meno di 25 partite."))
        if "prior_fallback_nessun_pool" in flags:
            items.append(("warn", "Prior di popolazione non disponibile (pochi giocatori caricati): shrinkage solo verso il giocatore stesso."))
        if market is Market.THREES and not pred.fit.three_ok:
            items.append(("missing", "Triple non prezzabili: tentativi da tre insufficienti."))
        if pred.adjustments:
            items.append(("ok", "Aggiustamento back-to-back applicato: verificato sullo storico del giocatore."))
        elif pred.context.back_to_back:
            items.append(("warn", "Partita in back-to-back, ma l'effetto non è dimostrabile dallo storico: nessun aggiustamento."))
    items.append(("warn" if has_opponent else "missing", "Avversario: registrato ma NON usato nel calcolo (nessun coefficiente specifico per avversario)."))
    items.append(("missing", "Assenze dei compagni e infortuni: non modellati. Se un titolare è assente, i minuti possono cambiare (usa il campo minuti manuali)."))
    items.append(("missing", "Quote storiche: non disponibili (BALLDONTLIE non le conserva). Nessun backtest dell'EV contro il mercato."))
    items.append(("ok" if n_books >= 3 else "warn", f"Bookmaker che quotano questa prop: {n_books}" + (" (pochi: la media tra book è poco rappresentativa)." if n_books < 3 else ".")))
    if n_stale:
        items.append(("warn", f"{n_stale} quote obsolete rispetto alla soglia impostata."))
    if mode == "demo":
        items.append(("warn", "Tutti i dati sono sintetici (modalità demo)."))
    return items
