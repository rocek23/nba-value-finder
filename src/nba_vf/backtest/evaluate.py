"""Costruzione delle righe di valutazione: ogni riga e' (modello, partita, mercato, linea, lato) con probabilita' e esito.

Due tipi di linee:
  * ``sintetica``   mezze linee generate dallo storico del giocatore (non dal modello): servono a misurare la qualita'
                    predittiva e la calibrazione; NON hanno quota, quindi NON producono ROI;
  * ``quota_reale`` linee e quote di un libro catturate prima dell'inizio (vedi quotes.py): le uniche su cui si calcola il ROI.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..model.empirical import stat_value
from ..model.errors import InvalidLineError
from ..model.odds_math import side_probs_arrays, split_probs
from ..models import Market, Side
from .baselines import BASELINES, baseline_probs, neutral_lines
from .calibration import Recalibrator
from .quotes import SelectedQuote
from .records import PredRecord

MARKET_BASELINE = "mercato_consenso"
CAL_SUFFIX = "_calibrato"
ROW_COLUMNS = ["model", "phase", "player", "game_id", "date", "season", "market", "line", "side", "line_source", "book", "odds",
               "capture_time", "commence_time", "minutes_before_start", "p_win", "p_push", "p_loss", "y", "result", "obs_id"]


def result_of(y: float, line: float, side: str) -> str:
    if abs(y - line) < 1e-9:
        return "push"
    over = y > line
    return "win" if (over if side == "over" else not over) else "loss"


def _row(model, rec: PredRecord, market: Market, line, side, source, probs, y, *, book="", odds=np.nan, capture="", commence="", mins=np.nan) -> dict:
    pw, pp, pl = probs
    return {
        "model": model, "phase": rec.phase, "player": rec.player, "game_id": rec.log.game_id, "date": rec.log.game_date,
        "season": rec.log.season, "market": market.value, "line": float(line), "side": side, "line_source": source, "book": book,
        "odds": odds, "capture_time": capture, "commence_time": commence, "minutes_before_start": mins,
        "p_win": float(pw), "p_push": float(pp), "p_loss": float(pl), "y": float(y), "result": result_of(y, line, side),
        "obs_id": f"{rec.player}|{rec.log.game_id}|{market.value}|{float(line)}|{side}|{book}",
    }


def _calibrated(raw_win: float, side: str, cal: Recalibrator) -> tuple[float, float, float]:
    p_over = raw_win if side == "over" else 1.0 - raw_win
    q = float(cal.predict(np.array([p_over]))[0])
    pw = q if side == "over" else 1.0 - q
    return pw, 0.0, 1.0 - pw


def synthetic_rows(models: Mapping[str, Mapping[tuple, PredRecord]], cal: Mapping[str, Recalibrator] | None = None) -> list[dict]:
    cal = cal or {}
    names = list(models)
    ref = models[names[0]]
    rows: list[dict] = []
    for key, rec in ref.items():
        recs = {n: models[n].get(key) for n in names}
        if any(r is None for r in recs.values()):
            continue
        markets = set.intersection(*[set(r.pmfs) for r in recs.values()])
        for m in sorted(markets, key=lambda x: x.value):
            vals = np.array([stat_value(l, m) for l in rec.history], float)
            y = stat_value(rec.log, m)
            ymax = len(rec.pmfs[m]) - 1
            for ln in neutral_lines(vals, ymax):
                for n, r in recs.items():
                    _, _, pg = split_probs(r.pmfs[m], ln)
                    rows.append(_row(n, rec, m, ln, "over", "sintetica", (pg, 0.0, 1 - pg), y))
                    if n in cal:
                        rows.append(_row(n + CAL_SUFFIX, rec, m, ln, "over", "sintetica", _calibrated(float(pg), "over", cal[n]), y))
                for b in BASELINES:
                    lt, eq, gt = baseline_probs(b, vals, ln)
                    rows.append(_row(b, rec, m, ln, "over", "sintetica", (gt, eq, lt), y))
    return rows


def market_consensus(matched: Sequence[tuple[SelectedQuote, str, object]]) -> dict[tuple, float]:
    """Probabilita' Over senza margine, media tra libri, per (giocatore, data, mercato, linea). Solo coppie dello stesso prelievo."""
    acc: dict[tuple, dict[str, float]] = defaultdict(dict)
    for sq, _k, _l in matched:
        if sq.novig_p_over is not None:
            acc[(sq.player_key, sq.game_date, sq.quote.market, float(sq.quote.line))][sq.quote.bookmaker] = sq.novig_p_over
    return {k: float(np.mean(list(v.values()))) for k, v in acc.items()}


def quote_rows(
    models: Mapping[str, Mapping[tuple, PredRecord]], matched: Sequence[tuple[SelectedQuote, str, object]],
    cal: Mapping[str, Recalibrator] | None = None, phases: Sequence[str] = ("validazione", "test"),
) -> tuple[list[dict], Counter]:
    cal = cal or {}
    names = list(models)
    cons = market_consensus(matched)
    rows: list[dict] = []
    rej: Counter = Counter()
    for sq, pool_key, log in matched:
        q = sq.quote
        key = (pool_key, log.game_id)
        recs = {n: models[n].get(key) for n in names}
        if any(r is None for r in recs.values()):
            rej["fuori_dalle_fasi_valutate_o_storico_insufficiente"] += 1
            continue
        ref = next(iter(recs.values()))
        if ref.phase not in phases:
            continue
        if any(q.market not in r.pmfs for r in recs.values()):
            rej["mercato_non_prezzabile"] += 1
            continue
        side = q.side.value
        y = stat_value(log, q.market)
        try:
            base_probs = {n: tuple(float(x) for x in side_probs_arrays(r.pmfs[q.market], q.line, side)) for n, r in recs.items()}
        except InvalidLineError:
            rej["linea_fuori_griglia"] += 1
            continue
        meta = dict(book=q.bookmaker, odds=q.odds_decimal, capture=sq.capture_time.isoformat(),
                    commence=q.commence_time.isoformat(), mins=sq.minutes_before_start)
        half = abs(q.line - round(q.line)) > 1e-9
        for n, r in recs.items():
            pw, pp, pl = base_probs[n]
            rows.append(_row(n, r, q.market, q.line, side, "quota_reale", (pw, pp, pl), y, **meta))
            if n in cal and half:
                rows.append(_row(n + CAL_SUFFIX, r, q.market, q.line, side, "quota_reale", _calibrated(pw, side, cal[n]), y, **meta))
        vals = np.array([stat_value(l, q.market) for l in ref.history], float)
        for b in BASELINES:
            lt, eq, gt = baseline_probs(b, vals, q.line)
            pw, pp, pl = (gt, eq, lt) if side == "over" else (lt, eq, gt)
            rows.append(_row(b, ref, q.market, q.line, side, "quota_reale", (pw, pp, pl), y, **meta))
        c = cons.get((sq.player_key, sq.game_date, q.market, float(q.line)))
        if c is not None and half:
            pw = c if side == "over" else 1 - c
            rows.append(_row(MARKET_BASELINE, ref, q.market, q.line, side, "quota_reale", (pw, 0.0, 1 - pw), y, **meta))
    return rows, rej


NUMERIC_COLUMNS = ["line", "odds", "minutes_before_start", "p_win", "p_push", "p_loss", "y"]


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Colonne numeriche sempre float (un DataFrame vuoto le rende 'object' e rompe np.log dopo la concatenazione)."""
    df = df.copy()
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return coerce_numeric(pd.DataFrame(rows, columns=ROW_COLUMNS))


# --------------------------------------------------------------------------------------------------
# PIT, log score e CRPS sulla pmf intera (indipendenti dalle linee)
# --------------------------------------------------------------------------------------------------
def _poisson_pmf(mean: float, k: int) -> np.ndarray:
    pmf = poisson.pmf(np.arange(k), max(mean, 1e-3))
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def crps_pmf(pmf: np.ndarray, y: int) -> float:
    cdf = np.cumsum(pmf)
    ind = (np.arange(len(pmf)) >= min(y, len(pmf) - 1)).astype(float)
    return float(np.sum((cdf - ind) ** 2))


def pmf_score_rows(models: Mapping[str, Mapping[tuple, PredRecord]], seed: int = 0) -> list[dict]:
    """Per ogni partita/mercato: log score negativo e CRPS del modello e del Poisson (ultime 10), PIT randomizzata."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    first = next(iter(models), None)
    for name, recs in models.items():
        for key, rec in recs.items():
            for m, pmf in rec.pmfs.items():
                y = min(int(stat_value(rec.log, m)), len(pmf) - 1)
                cdf = np.cumsum(pmf)
                pit = (cdf[y - 1] if y >= 1 else 0.0) + rng.random() * pmf[y]
                vals = [stat_value(l, m) for l in rec.history[-10:]]
                pois = _poisson_pmf(float(np.mean(vals)), len(pmf))
                for label, p in ((name, pmf),) + ((("poisson_ultime10", pois),) if name == first else ()):
                    rows.append({"model": label, "phase": rec.phase, "player": rec.player, "game_id": rec.log.game_id, "date": rec.log.game_date,
                                 "season": rec.log.season, "market": m.value, "y": y, "media": float(p @ np.arange(len(p))),
                                 "log_score": float(-np.log(max(p[y], 1e-12))), "crps": crps_pmf(p, y),
                                 "pit": float(pit) if label == name else np.nan})
    return rows
