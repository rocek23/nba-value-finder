"""Orchestrazione del backtest: split temporale, walk-forward, selezione SOLO su validazione, test una volta, salvataggio."""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..model.config import ModelConfig
from ..models import Market, OddsQuote, PlayerGameLog
from . import evaluate as ev
from .calibration import Recalibrator
from .metrics import log_loss, summarize_bets
from .quotes import margins_by_book, match_to_games, select_pre_game_quotes
from .records import PredRecord, config_dict, generate_records, sort_pool, stable_hash
from .report import (bet_table, calibration_summary, paired_vs_baselines, predictive_table, verdict)
from .splits import TEST, VALIDATION, TemporalSplit, make_split
from .strategies import all_bets, value_bets

ALL_MARKETS = tuple(m.value for m in Market)


@dataclass(frozen=True)
class BacktestConfig:
    train_frac: float = 0.5
    val_frac: float = 0.25
    block_days: int = 30
    min_players: int = 20
    n_sims_pra: int = 4000
    use_context: bool = True
    seed: int = 12345
    candidates: tuple[str, ...] = ("base", "recenza_corta", "senza_recenza")
    thresholds: tuple[float, ...] = (0.0, 0.03, 0.05, 0.08)
    default_threshold: float = 0.05
    min_val_bets_for_threshold: int = 30
    lead_minutes: float = 0.0
    stake: float = 1.0
    bankroll: float = 100.0
    ci_level: float = 0.90
    n_boot: int = 1000
    min_bets_verdict: int = 200
    markets: tuple[str, ...] = ALL_MARKETS
    max_players: int | None = None

    def hash(self) -> str:
        return stable_hash(dataclasses.asdict(self))


def candidate_configs(cfg: BacktestConfig) -> dict[str, ModelConfig]:
    base = ModelConfig(n_boot=0, n_sims_pra=cfg.n_sims_pra)
    all_c = {
        "base": base,
        "recenza_corta": dataclasses.replace(base, half_life_rates=10.0, half_life_minutes=6.0),
        "senza_recenza": dataclasses.replace(base, half_life_rates=None, half_life_minutes=None),
    }
    unknown = [c for c in cfg.candidates if c not in all_c]
    if unknown:
        raise ValueError(f"candidati sconosciuti: {unknown}")
    return {c: all_c[c] for c in cfg.candidates}


def data_fingerprint(pool: Mapping[str, Sequence[PlayerGameLog]], quotes: Sequence[OddsQuote]) -> str:
    h = hashlib.sha1()
    for k in sorted(pool):
        for l in sorted(pool[k], key=lambda l: (l.game_date, l.game_id)):
            h.update(f"{k}|{l.game_id}|{l.game_date}|{l.minutes}|{l.pts}|{l.reb}|{l.ast}|{l.fg3m}|{l.fg3a}\n".encode())
    h.update(f"Q{len(quotes)}".encode())
    if quotes:
        h.update(f"{min(q.fetched_at for q in quotes)}|{max(q.fetched_at for q in quotes)}".encode())
    return h.hexdigest()[:12]


def make_run_id(pool: Mapping[str, Sequence[PlayerGameLog]], quotes: Sequence[OddsQuote], cfg: BacktestConfig, source: str) -> tuple[str, str]:
    """(run_id, impronta dei dati): stesso dato + stessa configurazione + stessa fonte = stesso run."""
    pool = sort_pool(pool)
    if cfg.max_players:
        pool = dict(sorted(pool.items(), key=lambda kv: -len(kv[1]))[: cfg.max_players])
    fp = data_fingerprint(pool, list(quotes))
    return stable_hash({"fp": fp, "cfg": dataclasses.asdict(cfg), "source": source}), fp


def _by_key(recs: Iterable[PredRecord]) -> dict[tuple, PredRecord]:
    return {r.key: r for r in recs}


@dataclass
class BacktestResult:
    run_id: str
    label: str
    source: str
    cfg: BacktestConfig
    split: TemporalSplit
    data_fingerprint: str
    selection: dict
    recalibrator: Recalibrator
    records: list[PredRecord]
    priors: dict
    evals: pd.DataFrame
    bets: pd.DataFrame
    pmf_scores: pd.DataFrame
    quote_info: dict
    pool: dict
    threshold: float
    threshold_grid: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    matched: list = field(default_factory=list)
    verdict: dict = field(default_factory=dict)
    n_test_runs: int = 1


def run_backtest(
    pool: Mapping[str, Sequence[PlayerGameLog]], quotes: Sequence[OddsQuote] | None = None, cfg: BacktestConfig | None = None, *,
    source: str = "live", label: str = "", progress: Callable[[float, str], None] | None = None, n_test_runs: int = 1,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    quotes = list(quotes or [])
    say = progress or (lambda f, m: None)
    pool = sort_pool(pool)
    if cfg.max_players:
        pool = dict(sorted(pool.items(), key=lambda kv: -len(kv[1]))[: cfg.max_players])
    markets = [Market(m) for m in cfg.markets]
    split = make_split((l.game_date for v in pool.values() for l in v), cfg.train_frac, cfg.val_frac)
    run_id, fp = make_run_id(pool, quotes, cfg, source)
    notes: list[str] = []

    # ---- quote storiche reali: solo pre-partita, abbinate alla partita dell'evento -------------------
    selected, rej_sel = select_pre_game_quotes(quotes, cfg.lead_minutes)
    matched, rej_match = match_to_games(selected, pool)
    matched = [m for m in matched if split.phase(m[0].game_date) in (VALIDATION, TEST) and m[0].quote.market in markets]
    quote_info = {
        "quote_ricevute": len(quotes), "quote_pre_partita_selezionate": len(selected), "quote_abbinate_a_partite": len(matched),
        "scartate": {**dict(rej_sel), **dict(rej_match)}, "margini_per_libro": margins_by_book([m[0] for m in matched]),
        "libri": sorted({m[0].quote.bookmaker for m in matched}),
    }

    # ---- validazione: tutti i candidati, walk-forward ------------------------------------------------
    cands = candidate_configs(cfg)
    val_recs: dict[str, list[PredRecord]] = {}
    priors: dict = {}
    for i, (name, mcfg) in enumerate(cands.items()):
        say(0.05 + 0.4 * i / len(cands), f"Validazione: candidato «{name}»")
        recs, pri = generate_records(pool, mcfg, name, split.phase, split.train_end, split.val_end, block_days=cfg.block_days,
                                     min_players=cfg.min_players, use_context=cfg.use_context, markets=markets)
        val_recs[name] = recs
        priors.update({(name, d): v for d, v in pri.items()})
    if not any(val_recs.values()):
        raise ValueError("nessuna previsione nella validazione: storico troppo corto o giocatori insufficienti")
    val_models = {f"modello/{c}": _by_key(r) for c, r in val_recs.items()}
    val_syn = ev.to_frame(ev.synthetic_rows(val_models))

    # ---- scelta del candidato: SOLO log-loss sulla validazione (linee neutrali) ---------------------
    table = []
    for c in cands:
        d = val_syn[(val_syn["model"] == f"modello/{c}") & (val_syn["side"] == "over")]
        per_market = [log_loss(g["p_win"], (g["result"] == "win").astype(float)) for _, g in d.groupby("market")]
        table.append({"candidato": c, "log_loss_validazione": float(np.mean(per_market)) if per_market else float("inf"), "n": len(d)})
    sel_table = pd.DataFrame(table)
    chosen = str(sel_table.loc[sel_table["log_loss_validazione"].idxmin(), "candidato"])  # a parita' vince il primo
    selection = {"scelto": chosen, "criterio": "log-loss medio per mercato sulla validazione (linee neutrali), nessun dato di test", "tabella": sel_table.to_dict("records")}

    # ---- ricalibrazione e soglia: stimate sulla validazione -----------------------------------------
    chosen_val = _by_key(val_recs[chosen])
    d = val_syn[(val_syn["model"] == f"modello/{chosen}") & (val_syn["side"] == "over")]
    cal = Recalibrator().fit(d["p_win"].to_numpy(), (d["result"] == "win").astype(float).to_numpy())
    say(0.5, "Validazione: quote reali e soglia")
    val_q_rows, rej_q_val = ev.quote_rows({"modello": chosen_val}, matched, None)
    val_frame = ev.to_frame(ev.synthetic_rows({"modello": chosen_val}) + val_q_rows)
    grid = []
    for thr in cfg.thresholds:
        b = value_bets(val_frame, "modello", thr, cfg.stake, phase=VALIDATION)
        s = summarize_bets(b, bankroll=cfg.bankroll, B=200) if len(b) else {"n_scommesse": 0}
        grid.append({"soglia_ev": thr, "n_scommesse": s["n_scommesse"], "yield": s.get("yield", np.nan)})
    grid_df = pd.DataFrame(grid)
    ok = grid_df[grid_df["n_scommesse"] >= cfg.min_val_bets_for_threshold]
    if ok.empty:
        thr = cfg.default_threshold
        notes.append(f"Soglia EV non ottimizzata (meno di {cfg.min_val_bets_for_threshold} scommesse in validazione): usato il valore predefinito {thr:.0%}.")
    else:
        thr = float(ok.sort_values(["yield", "soglia_ev"], ascending=[False, False]).iloc[0]["soglia_ev"])
    selection["soglia_ev"] = thr

    # ---- test: UNA sola valutazione con tutto congelato ---------------------------------------------
    say(0.6, "Test fuori campione")
    test_recs, pri_t = generate_records(pool, cands[chosen], chosen, split.phase, split.val_end, split.last, block_days=cfg.block_days,
                                        min_players=cfg.min_players, use_context=cfg.use_context, markets=markets)
    priors.update({(chosen, d): v for d, v in pri_t.items()})
    test_by_key = _by_key(test_recs)
    test_models = {"modello": test_by_key}
    test_syn = ev.synthetic_rows(test_models, {"modello": cal})
    test_q, rej_q_test = ev.quote_rows(test_models, matched, {"modello": cal}, phases=(TEST,))
    quote_info["scartate"].update({k: quote_info["scartate"].get(k, 0) + v for k, v in {**rej_q_val, **rej_q_test}.items() if k == "linea_fuori_griglia"})
    # le righe di validazione del modello scelto sono rinominate come nel test per un registro unico
    evals = pd.concat([val_syn, ev.to_frame(test_syn + test_q), ev.to_frame(val_q_rows)], ignore_index=True)

    say(0.8, "Scommesse simulate e metriche")
    bets = pd.concat([all_bets(evals, "modello", thr, cfg.stake, TEST), all_bets(val_frame, "modello", thr, cfg.stake, VALIDATION, with_baselines=True)], ignore_index=True)
    pmf = ev.pmf_score_rows({f"modello/{c}": _by_key(r) for c, r in val_recs.items()}, cfg.seed) + ev.pmf_score_rows(test_models, cfg.seed)
    pmf_df = pd.DataFrame(pmf)

    res = BacktestResult(
        run_id=run_id, label=label or source, source=source, cfg=cfg, split=split, data_fingerprint=fp, selection=selection, recalibrator=cal,
        records=[r for rs in val_recs.values() for r in rs] + test_recs, priors=priors, evals=evals, bets=bets, pmf_scores=pmf_df,
        quote_info=quote_info, pool=pool, threshold=thr, threshold_grid=grid_df, notes=notes, n_test_runs=n_test_runs, matched=matched,
    )
    res.verdict = compute_verdict(res)
    say(1.0, "Completato")
    return res


def compute_verdict(res: BacktestResult) -> dict:
    cfg, ev_df, bets = res.cfg, res.evals, res.bets
    n_q_test = int(((ev_df["phase"] == TEST) & (ev_df["line_source"] == "quota_reale") & (ev_df["model"] == "modello")).sum())
    tb = bet_table(bets, TEST, bankroll=cfg.bankroll, level=cfg.ci_level, B=cfg.n_boot)
    vb = bet_table(bets, VALIDATION, bankroll=cfg.bankroll, level=cfg.ci_level, B=cfg.n_boot)
    pick = lambda t: (t[t["strategia"] == "valore:modello"].iloc[0].to_dict() if len(t) and (t["strategia"] == "valore:modello").any() else None)
    vs_base = paired_vs_baselines(ev_df, TEST, "quota_reale", "modello", ["frequenza_ultime20", "poisson_ultime10"], level=cfg.ci_level, B=cfg.n_boot)
    vs_mkt = paired_vs_baselines(ev_df, TEST, "quota_reale", "modello", [ev.MARKET_BASELINE], level=cfg.ci_level, B=cfg.n_boot)
    cal = calibration_summary(ev_df, TEST, "quota_reale", "modello") if n_q_test else calibration_summary(ev_df, TEST, "sintetica", "modello")
    return verdict(source=res.source, n_quote_rows_test=n_q_test, test_bets=pick(tb), val_bets=pick(vb), vs_baselines_quotes=vs_base,
                   vs_market=vs_mkt, calibration=cal, min_bets=cfg.min_bets_verdict, n_test_runs=res.n_test_runs)
