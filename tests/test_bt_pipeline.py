"""Backtest end-to-end su dati SINTETICI: anti-leakage, test usato una volta, nessun ROI senza quote, persistenza e verifica."""

import dataclasses
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_vf.backtest import BacktestConfig, run_backtest
from nba_vf.backtest.demo_data import build_pool, build_quotes
from nba_vf.backtest.quotes import match_to_games, select_pre_game_quotes
from nba_vf.backtest.records import LeakageError, assert_no_leak, generate_records
from nba_vf.backtest.report import bet_table, paired_vs_baselines
from nba_vf.backtest.splits import make_split
from nba_vf.backtest.store import BacktestStore
from nba_vf.model.config import ModelConfig

CFG = BacktestConfig(n_boot=200, candidates=("base",), min_players=8, n_sims_pra=1500, block_days=30)


@pytest.fixture(scope="module")
def world():
    pool, truths = build_pool(n_players=12, games_per_season=26, seasons=(2024, 2025), seed=3)
    dates = sorted({l.game_date for v in pool.values() for l in v})
    quotes = build_quotes(pool, truths, from_date=dates[int(len(dates) * 0.5)], seed=9, leak_share=0.10, true_sample=2500)
    return pool, truths, quotes


@pytest.fixture(scope="module")
def run(world):
    pool, _, quotes = world
    return run_backtest(pool, quotes, CFG, source="live-test", label="test")


# ---------------------------------------------------------------- anti-leakage
def _perturb(pool, after: date, factor: float = 3.0):
    out = {}
    for k, logs in pool.items():
        out[k] = [dataclasses.replace(l, pts=int(l.pts * factor) + 7, reb=l.reb + 9, ast=l.ast + 5, fg3m=l.fg3m, minutes=l.minutes) if l.game_date >= after else l for l in logs]
    return out


def test_predictions_do_not_depend_on_the_target_game_or_the_future(world):
    pool = world[0]
    split = make_split({l.game_date for v in pool.values() for l in v})
    cfg = ModelConfig(n_boot=0, n_sims_pra=1500)
    kw = dict(block_days=30, min_players=8, use_context=True)
    base, _ = generate_records(pool, cfg, "c", split.phase, split.train_end, split.last, **kw)
    cut = split.val_end  # si alterano le statistiche di tutte le partite dal confine in poi (esiti inclusi)
    changed, _ = generate_records(_perturb(pool, cut + timedelta(days=1)), cfg, "c", split.phase, split.train_end, split.last, **kw)
    a = {r.key: r for r in base if r.log.game_date <= cut}
    b = {r.key: r for r in changed if r.log.game_date <= cut}
    assert a.keys() == b.keys() and len(a) > 50
    for k in a:
        for m in a[k].pmfs:
            np.testing.assert_array_equal(a[k].pmfs[m], b[k].pmfs[m])
    # le partite alterate hanno previsioni che cambiano solo perche' cambia il passato, mai perche' cambia il proprio esito
    first_day = cut + timedelta(days=1)
    t1 = {r.key: r for r in base if r.log.game_date == first_day or (cut < r.log.game_date <= cut + timedelta(days=2))}
    t2 = {r.key: r for r in changed if r.key in t1}
    assert t1 and all(np.array_equal(t1[k].pmfs[m], t2[k].pmfs[m]) for k in t1 if t1[k].history[-1].game_date <= cut for m in t1[k].pmfs)


def test_own_outcome_never_changes_own_prediction(world):
    pool = world[0]
    split = make_split({l.game_date for v in pool.values() for l in v})
    cfg = ModelConfig(n_boot=0, n_sims_pra=1500)
    base, _ = generate_records(pool, cfg, "c", split.phase, split.val_end, split.last, block_days=30, min_players=8)
    target = base[len(base) // 2]
    mod = {k: [dataclasses.replace(l, pts=l.pts + 40) if (k == target.player and l.game_id == target.log.game_id) else l for l in v] for k, v in pool.items()}
    again, _ = generate_records(mod, cfg, "c", split.phase, split.val_end, split.last, block_days=30, min_players=8)
    same = [r for r in again if r.key == target.key][0]
    for m in target.pmfs:
        np.testing.assert_array_equal(target.pmfs[m], same.pmfs[m])


def test_assert_no_leak_guard_raises(world):
    pool = world[0]
    logs = next(iter(pool.values()))
    with pytest.raises(LeakageError):
        assert_no_leak(logs[:10], logs[5])
    assert_no_leak(logs[:5], logs[5])


def test_selection_and_calibration_do_not_depend_on_test_data(world, run):
    """Scelta del candidato, soglia e ricalibrazione si fanno SOLO sulla validazione: alterare il test non li cambia."""
    pool, _, quotes = world
    altered = _perturb(pool, run.split.val_end + timedelta(days=1), factor=1.7)
    cfg = dataclasses.replace(CFG, candidates=("base", "senza_recenza"))
    r1 = run_backtest(pool, quotes, cfg, source="live-test")
    r2 = run_backtest(altered, quotes, cfg, source="live-test")
    assert r1.selection["scelto"] == r2.selection["scelto"] and r1.threshold == r2.threshold
    assert (r1.recalibrator.a, r1.recalibrator.b) == (r2.recalibrator.a, r2.recalibrator.b)
    assert r1.selection["tabella"] == r2.selection["tabella"]


def test_post_start_quotes_cannot_change_results(world):
    pool, _, quotes = world
    clean = [q for q in quotes if q.fetched_at < q.commence_time]
    assert len(clean) < len(quotes)  # il generatore ha messo quote post-inizio "prezzate sapendo l'esito"
    r_all = run_backtest(pool, quotes, CFG, source="live-test")
    r_clean = run_backtest(pool, clean, CFG, source="live-test")
    pd.testing.assert_frame_equal(r_all.bets.reset_index(drop=True), r_clean.bets.reset_index(drop=True))
    assert r_all.quote_info["scartate"]["dopo_inizio"] == len(quotes) - len(clean)


def test_current_quotes_are_never_applied_retroactively(world):
    """Quote con evento in una data diversa da quella della partita non vengono abbinate a partite passate."""
    pool, _, quotes = world
    later = [dataclasses.replace(q, commence_time=q.commence_time + timedelta(days=400), fetched_at=q.fetched_at + timedelta(days=400),
                                 quote_time=q.quote_time + timedelta(days=400)) for q in quotes[:2000]]
    sel, _ = select_pre_game_quotes(later)
    matched, rej = match_to_games(sel, pool)
    assert matched == [] and rej["partita_non_trovata"] == len(sel)


# ---------------------------------------------------------------- struttura e onesta' dei risultati
def test_phases_are_ordered_and_test_uses_frozen_choices(run):
    s = run.split
    ev = run.evals
    val_dates = pd.to_datetime(ev[ev["phase"] == "validazione"]["date"].astype(str))
    test_dates = pd.to_datetime(ev[ev["phase"] == "test"]["date"].astype(str))
    assert val_dates.max().date() <= s.val_end < test_dates.min().date()
    assert set(ev[ev["phase"] == "test"]["model"]) >= {"modello", "moneta", "frequenza_ultime20", "poisson_ultime10"}
    assert not ev[ev["phase"] == "test"]["model"].str.contains("/").any()  # nel test c'e' solo il candidato scelto


def test_all_baselines_and_market_are_evaluated_on_same_observations(run):
    t = paired_vs_baselines(run.evals, "test", "sintetica", baselines=["moneta", "frequenza_ultime20", "poisson_ultime10"], B=100)
    assert len(t) == 3 and (t["n"] > 100).all()
    assert (paired_vs_baselines(run.evals, "test", "quota_reale", baselines=["mercato_consenso"], B=100)["n"] > 100).all()


def test_roi_only_from_real_quotes_and_with_margin(run):
    bets = run.bets
    assert len(bets) > 100 and (bets["odds"] > 1.0).all()
    ev = run.evals[run.evals["line_source"] == "quota_reale"]
    assert ev["odds"].notna().all() and (ev["minutes_before_start"][ev["model"] == "modello"] > 0).all()
    # i margini misurati dai libri sintetici sono ~6%
    margins = [v["margine_medio"] for v in run.quote_info["margini_per_libro"].values()]
    assert margins and all(0.05 < m < 0.07 for m in margins)
    # chi scommette sempre Over paga il margine: yield negativo atteso (~ -margine/2 .. -margine)
    t = bet_table(bets, "test", B=200).set_index("strategia")
    assert t.loc["sempre_over", "yield"] < 0.02


def test_no_roi_without_historical_quotes(world):
    pool = world[0]
    r = run_backtest(pool, [], CFG, source="live-test")
    assert (r.evals["line_source"] == "sintetica").all()
    assert r.bets.empty
    assert r.verdict["code"] == "solo_predittivo"
    assert "ROI non calcolabile" in r.verdict["title"]
    assert r.evals[r.evals["phase"] == "test"]["odds"].isna().all()


def test_demo_source_never_concludes(world):
    pool, _, quotes = world
    assert run_backtest(pool, quotes, CFG, source="demo").verdict["code"] == "sintetico"


def test_efficient_market_is_not_declared_profitable(run):
    # il mercato sintetico conosce la distribuzione vera: il modello non puo' essere dichiarato superato
    assert run.verdict["code"] in {"non_dimostrato", "fallita", "campione_insufficiente"}
    assert run.verdict["code"] != "verifica_superata"


def test_planted_edge_is_detected(world):
    """Se il mercato sbaglia davvero (Over sovrastimato), la procedura deve vedere un rendimento positivo: il test ha potenza."""
    pool, truths, _ = world
    dates = sorted({l.game_date for v in pool.values() for l in v})
    biased = build_quotes(pool, truths, from_date=dates[int(len(dates) * 0.5)], seed=9, leak_share=0.0, edge=0.18, true_sample=2500)
    r = run_backtest(pool, biased, CFG, source="live-test")
    t = bet_table(r.bets, "test", B=300).set_index("strategia")
    assert t.loc["valore:modello", "yield"] > 0.10
    assert t.loc["valore:modello", "yield_lo"] > 0  # con ~600 scommesse un vantaggio del ~30% e' distinguibile dal caso
    # con ~500 scommesse l'errore standard dello yield e' ~5%: vantaggi piccoli NON sono dimostrabili (l'IC deve essere largo)
    assert t.loc["valore:modello", "yield_hi"] - t.loc["valore:modello", "yield_lo"] > 0.10
    assert t.loc["sempre_under", "yield"] > t.loc["sempre_over", "yield"]


# ---------------------------------------------------------------- persistenza e verificabilita'
@pytest.fixture(scope="module")
def stored(run, world, tmp_path_factory):
    st = BacktestStore(tmp_path_factory.mktemp("db") / "bt.sqlite")
    st.save(run)
    return st


def test_every_prediction_and_its_data_are_saved(run, stored):
    n_pred = len(stored.frame("predictions", run.run_id))
    assert n_pred == sum(len(r.pmfs) for r in run.records)
    inputs = stored.frame("inputs", run.run_id)
    assert len(inputs) == len({r.key for r in run.records})
    assert len(stored.frame("game_logs", run.run_id)) == sum(len(v) for v in run.pool.values())
    assert len(stored.frame("evals", run.run_id)) == len(run.evals)
    assert len(stored.frame("bets", run.run_id)) == len(run.bets)
    assert stored.meta(run.run_id)["config"]["n_sims_pra"] == CFG.n_sims_pra


def test_verification_passes_on_clean_run(run, stored):
    res = stored.verify(run.run_id, sample=15)
    assert all(ok for _, ok, _ in res), res


def test_verification_detects_tampering(run, stored, tmp_path):
    import shutil

    path = tmp_path / "t.sqlite"
    shutil.copy(stored.path, path)
    st = BacktestStore(path)
    with sqlite3.connect(path) as c:  # 1) una partita futura infilata nello storico
        row = c.execute("SELECT player, game_id FROM inputs WHERE run_id=? LIMIT 1 OFFSET 40", (run.run_id,)).fetchone()
        fut = c.execute("SELECT game_id FROM game_logs WHERE run_id=? AND player=? ORDER BY game_date DESC LIMIT 1", (run.run_id, row[0])).fetchone()[0]
        c.execute("UPDATE inputs SET history_ids = ? WHERE run_id=? AND player=? AND game_id=?", (f'["{fut}"]', run.run_id, row[0], row[1]))
        c.commit()
    res = dict((n, ok) for n, ok, _ in st.verify(run.run_id, sample=5))
    assert not res["Nessuna previsione usa partite non anteriori (storico)"]
    shutil.copy(stored.path, path)
    with sqlite3.connect(path) as c:  # 2) una quota catturata dopo l'inizio
        c.execute("UPDATE evals SET capture_time = '2099-01-01T00:00:00+00:00' WHERE rowid IN (SELECT rowid FROM evals WHERE model='modello' AND line_source='quota_reale' LIMIT 3)")
        c.commit()
    res = dict((n, ok) for n, ok, _ in BacktestStore(path).verify(run.run_id, sample=5))
    assert not res["Tutte le quote usate sono state catturate prima dell'inizio"]
    shutil.copy(stored.path, path)
    with sqlite3.connect(path) as c:  # 3) un esito manomesso
        c.execute("UPDATE predictions SET y = y + 50 WHERE rowid IN (SELECT rowid FROM predictions LIMIT 500)")
        c.commit()
    res = dict((n, ok) for n, ok, _ in BacktestStore(path).verify(run.run_id, sample=5))
    assert not res["Gli esiti salvati coincidono con i box score salvati"]


def test_run_is_deterministic_and_resave_is_idempotent(run, world, stored):
    pool, _, quotes = world
    again = run_backtest(pool, quotes, CFG, source="live-test", label="test")
    assert again.run_id == run.run_id
    pd.testing.assert_frame_equal(again.bets.reset_index(drop=True), run.bets.reset_index(drop=True))
    n = len(stored.frame("evals", run.run_id))
    stored.save(again)
    assert len(stored.frame("evals", run.run_id)) == n and len(stored.runs()) == 1


def test_multiple_testing_counter(run, stored):
    assert stored.n_test_configs(run.data_fingerprint) == 1
