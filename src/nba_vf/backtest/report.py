"""Tabelle di risultati, segmentazioni e verdetto prudente a partire dalle righe di valutazione e dal registro scommesse."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import (Estimate, calibration_slope_intercept, cluster_bootstrap, expected_calibration_error, log_loss, brier,
                      reliability, summarize_bets)
from .strategies import value_bets

MIN_GROUP_N = 30  # sotto questa numerosita' non si mostrano intervalli per i sottogruppi


def binary_view(evals: pd.DataFrame, phase: str, source: str, models: Sequence[str] | None = None, common: bool = True) -> pd.DataFrame:
    """Righe binarie (mezze linee, lato Over) con p_over e esito_over. Se ``common`` tiene solo le osservazioni presenti in tutti i modelli."""
    d = evals[(evals["phase"] == phase) & (evals["line_source"] == source) & (evals["side"] == "over")]
    d = d.assign(p_push=pd.to_numeric(d["p_push"], errors="coerce"), line=pd.to_numeric(d["line"], errors="coerce"))
    d = d[(d["p_push"] == 0) & ((d["line"] % 1).abs() > 1e-9)]
    if models is not None:
        d = d[d["model"].isin(models)]
    if d.empty:
        return d.assign(p=[], yb=[])
    if common:
        present = d.groupby("obs_id")["model"].nunique()
        k = d["model"].nunique()
        d = d[d["obs_id"].isin(present[present == k].index)]
    return d.assign(p=pd.to_numeric(d["p_win"], errors="coerce").astype(float), yb=(d["result"] == "win").astype(float))


def _ci(values, clusters, level, B, seed) -> Estimate:
    v = np.asarray(values, float)
    return cluster_bootstrap(clusters, {"s": v, "n": np.ones(len(v))}, lambda s: s["s"] / s["n"], B=B, seed=seed, level=level)


def predictive_table(evals: pd.DataFrame, phase: str, source: str, models: Sequence[str] | None = None, *, level=0.90, B=1000, seed=0) -> pd.DataFrame:
    d = binary_view(evals, phase, source, models)
    rows = []
    for m, g in d.groupby("model", sort=False):
        p = np.clip(g["p"].to_numpy(), 1e-12, 1 - 1e-12)
        y = g["yb"].to_numpy()
        days = g["date"].astype(str).to_numpy()
        bl = (p - y) ** 2
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        eb, el = _ci(bl, days, level, B, seed), _ci(ll, days, level, B, seed)
        rows.append({"modello": m, "n_osservazioni": len(g), "giornate": eb.clusters, "brier": eb.value, "brier_lo": eb.lo, "brier_hi": eb.hi,
                     "log_loss": el.value, "ll_lo": el.lo, "ll_hi": el.hi, "brier_skill_vs_moneta": 1 - eb.value / 0.25,
                     "p_media": float(p.mean()), "freq_osservata": float(y.mean())})
    return pd.DataFrame(rows)


def paired_vs_baselines(evals: pd.DataFrame, phase: str, source: str, model: str = "modello", baselines: Sequence[str] | None = None,
                        *, level=0.90, B=1000, seed=0) -> pd.DataFrame:
    """Differenza di perdita (modello - baseline) sulle STESSE osservazioni; negativo = modello migliore. IC bootstrap per giornata."""
    if baselines is None:
        baselines = [m for m in evals["model"].unique() if m != model and not m.startswith("modello")]
    rows = []
    for b in baselines:
        d = binary_view(evals, phase, source, [model, b])
        if d.empty or d["model"].nunique() < 2:
            continue
        a = d[d["model"] == model].set_index("obs_id")
        c = d[d["model"] == b].set_index("obs_id").reindex(a.index)
        pa, pb = np.clip(a["p"].to_numpy(), 1e-12, 1 - 1e-12), np.clip(c["p"].to_numpy(), 1e-12, 1 - 1e-12)
        y = a["yb"].to_numpy()
        days = a["date"].astype(str).to_numpy()
        db = (pa - y) ** 2 - (pb - y) ** 2
        dl = -(y * np.log(pa) + (1 - y) * np.log(1 - pa)) + (y * np.log(pb) + (1 - y) * np.log(1 - pb))
        eb, el = _ci(db, days, level, B, seed), _ci(dl, days, level, B, seed)
        rows.append({"baseline": b, "n": len(a), "delta_brier": eb.value, "d_brier_lo": eb.lo, "d_brier_hi": eb.hi,
                     "delta_log_loss": el.value, "d_ll_lo": el.lo, "d_ll_hi": el.hi,
                     "modello_migliore": bool(np.isfinite(el.hi) and el.hi < 0 and np.isfinite(eb.hi) and eb.hi < 0)})
    return pd.DataFrame(rows)


def reliability_view(evals: pd.DataFrame, phase: str, source: str, model: str = "modello", bins: int = 10) -> pd.DataFrame:
    d = binary_view(evals, phase, source, [model], common=False)
    return reliability(d["p"].to_numpy(), d["yb"].to_numpy(), bins) if len(d) else pd.DataFrame()


def calibration_summary(evals: pd.DataFrame, phase: str, source: str, model: str = "modello", *, B: int = 300, seed: int = 0, level=0.90) -> dict:
    d = binary_view(evals, phase, source, [model], common=False)
    if len(d) < 30:
        return {"n": len(d)}
    p, y = d["p"].to_numpy(), d["yb"].to_numpy()
    a, b = calibration_slope_intercept(p, y)
    days = d["date"].astype(str).to_numpy()
    codes, inv = np.unique(days, return_inverse=True)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(codes), len(codes))
        mask = np.concatenate([np.flatnonzero(inv == c) for c in pick])
        bs.append(calibration_slope_intercept(p[mask], y[mask])[1])
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"n": len(d), "intercetta": a, "pendenza": b, "pendenza_lo": float(lo), "pendenza_hi": float(hi),
            "ece": expected_calibration_error(p, y), "brier": brier(p, y), "log_loss": log_loss(p, y)}


# --------------------------------------------------------------------------------------------------
def bet_table(bets: pd.DataFrame, phase: str, *, bankroll=100.0, level=0.90, B=1000, seed=0) -> pd.DataFrame:
    rows = []
    d = bets[bets["phase"] == phase] if len(bets) else bets
    for strat, g in d.groupby("strategy", sort=False):
        s = summarize_bets(g, bankroll=bankroll, level=level, B=B, seed=seed)
        rows.append({"strategia": strat, **s})
    return pd.DataFrame(rows)


def bet_breakdown(bets: pd.DataFrame, strategy: str, phase: str, by: str, *, level=0.90, B=500, seed=0) -> pd.DataFrame:
    d = bets[(bets["strategy"] == strategy) & (bets["phase"] == phase)]
    rows = []
    for k, g in d.groupby(by, sort=True):
        n = len(g)
        row = {by: k, "n_scommesse": n, "profitto": float(g["profit"].sum()), "yield": float(g["profit"].sum() / g["stake"].sum()),
               "hit_rate": float((g["result"] == "win").sum() / max(1, ((g["result"] != "push").sum()))), "quota_media": float(g["odds"].mean())}
        if n >= MIN_GROUP_N:
            est = cluster_bootstrap(g["date"].astype(str).to_numpy(), {"p": g["profit"].to_numpy(float), "s": g["stake"].to_numpy(float)},
                                    lambda s: s["p"] / s["s"], B=B, seed=seed, level=level)
            row.update({"yield_lo": est.lo, "yield_hi": est.hi, "campione": "ok"})
        else:
            row.update({"yield_lo": np.nan, "yield_hi": np.nan, "campione": f"insufficiente (n<{MIN_GROUP_N})"})
        rows.append(row)
    return pd.DataFrame(rows)


def bets_by_book(evals: pd.DataFrame, model: str, threshold: float, phase: str, stake: float = 1.0, **kw) -> pd.DataFrame:
    """Ogni libro valutato DA SOLO (le sue quote soltanto). Le scommesse si sovrappongono tra libri: non sommare le righe."""
    d = evals[(evals["phase"] == phase) & (evals["line_source"] == "quota_reale") & (evals["model"] == model)]
    parts = []
    for b in sorted(d["book"].unique()):
        sub = value_bets(evals[evals["book"] == b], model, threshold, stake, "valore:" + model, phase)
        if not sub.empty:
            sub = sub.assign(book=b)
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    allb = pd.concat(parts, ignore_index=True)
    return bet_breakdown(allb, "valore:" + model, phase, "book", **kw)


def predictive_breakdown(evals: pd.DataFrame, phase: str, source: str, by: str, model: str = "modello", level=0.90, B=500, seed=0) -> pd.DataFrame:
    d = binary_view(evals, phase, source, [model], common=False)
    rows = []
    for k, g in d.groupby(by, sort=True):
        n = len(g)
        row = {by: k, "n_osservazioni": n, "giocatori": g["player"].nunique(), "brier": brier(g["p"], g["yb"]), "log_loss": log_loss(g["p"], g["yb"]),
               "p_media": float(g["p"].mean()), "freq_osservata": float(g["yb"].mean())}
        if n >= MIN_GROUP_N:
            e = _ci((g["p"].to_numpy() - g["yb"].to_numpy()) ** 2, g["date"].astype(str).to_numpy(), level, B, seed)
            row.update({"brier_lo": e.lo, "brier_hi": e.hi, "campione": "ok"})
        else:
            row.update({"brier_lo": np.nan, "brier_hi": np.nan, "campione": f"insufficiente (n<{MIN_GROUP_N})"})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Verdetto
# --------------------------------------------------------------------------------------------------
def verdict(
    *, source: str, n_quote_rows_test: int, test_bets: dict | None, val_bets: dict | None, vs_baselines_quotes: pd.DataFrame,
    vs_market: pd.DataFrame, calibration: dict, min_bets: int, n_test_runs: int = 1,
) -> dict:
    """Esito prudente. 'Verifica superata' richiede TUTTE le condizioni sul test (fuori campione) con quote reali."""
    base = {"checks": [], "level": "info"}
    if source in ("demo", "sintetico"):
        return {**base, "code": "sintetico", "level": "warning", "title": "Dati sintetici: nessuna conclusione sulla NBA",
                "text": "Il backtest gira su dati e quote inventati (demo). Serve a verificare che il codice funzioni e che le regole anti-leakage valgano, "
                        "non a stabilire se il modello sia profittevole."}
    test_bets = test_bets or {}
    if n_quote_rows_test == 0:
        return {**base, "code": "solo_predittivo", "level": "warning", "title": "ROI non calcolabile: nessuna quota storica reale nel periodo di test",
                "text": "Senza quote storiche con timestamp precedente all'inizio delle partite non si calcola alcun ROI (non si inventano quote). "
                        "È valutata solo la qualità predittiva. BALLDONTLIE non conserva quote storiche: ogni «Aggiorna dati live» ne salva una copia "
                        "locale, oppure puoi importarle da CSV con timestamp."}
    checks = []
    n = int(test_bets.get("n_scommesse", 0))
    checks.append(("Numerosità: scommesse nel test ≥ %d" % min_bets, n >= min_bets, f"{n} scommesse"))
    ylo = test_bets.get("yield_lo", np.nan)
    yv = test_bets.get("yield", np.nan)
    checks.append(("Yield nel test con limite inferiore dell'IC > 0", bool(np.isfinite(ylo) and ylo > 0), f"yield {yv:+.1%} (IC {ylo:+.1%}…{test_bets.get('yield_hi', np.nan):+.1%})" if np.isfinite(yv) else "n/d"))
    vy = (val_bets or {}).get("yield", np.nan)
    checks.append(("Yield positivo anche nella validazione", bool(np.isfinite(vy) and vy > 0), f"{vy:+.1%}" if np.isfinite(vy) else "n/d"))
    beat = (not vs_baselines_quotes.empty) and bool(vs_baselines_quotes["modello_migliore"].all())
    checks.append(("Batte le baseline semplici (Brier e log-loss, IC esclude 0)", beat, "non verificabile" if vs_baselines_quotes.empty else
                   ", ".join(f"{r.baseline}: {'sì' if r.modello_migliore else 'no'}" for r in vs_baselines_quotes.itertuples())))
    mk_ok = (not vs_market.empty) and int(vs_market.iloc[0]["n"]) >= 100 and bool(vs_market["modello_migliore"].all())
    checks.append(("Batte il consenso di mercato senza margine (n ≥ 100)", mk_ok, "non verificabile (poche coppie Over/Under)" if vs_market.empty or int(vs_market.iloc[0]["n"]) < 100 else "sì" if mk_ok else "no"))
    slope = calibration.get("pendenza", np.nan)
    checks.append(("Calibrazione: pendenza tra 0.8 e 1.2", bool(np.isfinite(slope) and 0.8 <= slope <= 1.2), f"pendenza {slope:.2f}" if np.isfinite(slope) else "n/d"))
    checks.append(("Test usato una sola volta su questi dati", n_test_runs <= 1, f"{n_test_runs} configurazioni testate"))
    ok = all(c[1] for c in checks)
    if ok:
        code, level, title = "verifica_superata", "success", "Verifica fuori campione SUPERATA (preliminare)"
        text = ("Tutte le condizioni sono soddisfatte sul periodo di test. Non è una garanzia di rendimenti futuri: il campione è limitato, le quote "
                "storiche sono quelle raccolte da te e il mercato cambia. Da verificare con nuovi dati prima di qualsiasi uso.")
    elif np.isfinite(test_bets.get("yield_hi", np.nan)) and test_bets["yield_hi"] < 0 and n >= min_bets:
        code, level, title = "fallita", "error", "Verifica fuori campione NON superata: rendimento negativo"
        text = "Nel periodo di test l'intero intervallo di incertezza dello yield è sotto zero. Il modello NON va considerato profittevole."
    elif n < min_bets:
        code, level, title = "campione_insufficiente", "warning", "Campione insufficiente per concludere"
        text = "Troppe poche scommesse simulate nel test per distinguere un vantaggio dal caso. Nessuna conclusione di profittabilità."
    else:
        code, level, title = "non_dimostrato", "warning", "Vantaggio NON dimostrato fuori campione"
        text = "Non tutte le condizioni sono soddisfatte: il modello non va presentato come profittevole."
    return {"code": code, "level": level, "title": title, "text": text, "checks": checks}
