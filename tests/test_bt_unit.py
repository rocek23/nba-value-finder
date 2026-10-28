"""Backtest: split temporali, filtro delle quote pre-partita, metriche, drawdown, ROI con push e margine, verdetto."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nba_vf.backtest import baselines as BL
from nba_vf.backtest import metrics as M
from nba_vf.backtest import quotes as Q
from nba_vf.backtest.report import verdict
from nba_vf.backtest.splits import SplitError, TemporalSplit, make_split
from nba_vf.backtest.strategies import naive_bets, value_bets
from nba_vf.models import Market, OddsQuote, PlayerGameLog, Side

UTC = timezone.utc
START = datetime(2026, 1, 11, 0, 30, tzinfo=UTC)  # 19:30 ET del 10 gennaio


def quote(minutes_before=120, side=Side.OVER, odds=1.9, line=24.5, book="b1", source="api", quote_age=1, commence=START, event="E1", name="Mario Rossi"):
    cap = (commence - timedelta(minutes=minutes_before)) if commence else START - timedelta(minutes=minutes_before)
    return OddsQuote(provider="p", bookmaker=book, event_id=event, player_name=name, player_id=None, market=Market.POINTS, line=line, side=side,
                     odds_decimal=odds, fetched_at=cap, quote_time=cap - timedelta(minutes=quote_age), commence_time=commence, source=source)


# ---------------------------------------------------------------- split
def test_split_is_chronological_and_disjoint():
    days = [date(2025, 10, 1) + timedelta(days=i) for i in range(100)]
    s = make_split(days)
    assert s.first <= s.train_end < s.val_end < s.last
    phases = [s.phase(d) for d in days]
    order = {"addestramento": 0, "validazione": 1, "test": 2}
    assert [order[p] for p in phases] == sorted(order[p] for p in phases)  # mai rimescolato
    assert phases.count("test") >= 5 and phases.count("validazione") >= 5


def test_split_rejects_bad_inputs():
    with pytest.raises(SplitError):
        make_split([date(2025, 10, 1) + timedelta(days=i) for i in range(10)])
    with pytest.raises(SplitError):
        TemporalSplit(date(2025, 1, 5), date(2025, 1, 4), date(2025, 1, 9), date(2025, 1, 10))
    with pytest.raises(SplitError):
        make_split([date(2025, 10, 1) + timedelta(days=i) for i in range(100)], train_frac=0.8, val_frac=0.2)


# ---------------------------------------------------------------- quote pre-partita
def test_quote_after_start_is_rejected():
    sel, rej = Q.select_pre_game_quotes([quote(-5)])
    assert sel == [] and rej["dopo_inizio"] == 1


def test_quote_at_exact_start_is_rejected():
    assert Q.reject_reason(quote(0)) == "dopo_inizio"


def test_quote_without_start_time_is_rejected():
    q = quote()
    q = OddsQuote(**{**q.__dict__, "commence_time": None})
    assert Q.reject_reason(q) == "senza_orario_inizio"


def test_lead_minutes_excludes_late_quotes():
    assert Q.reject_reason(quote(20), lead_minutes=30) == "troppo_vicina_all_inizio"
    assert Q.reject_reason(quote(40), lead_minutes=30) is None


def test_incoherent_quote_time_rejected():
    q = quote()
    q = OddsQuote(**{**q.__dict__, "quote_time": q.fetched_at + timedelta(hours=3)})
    assert Q.reject_reason(q) == "timestamp_incoerente"


def test_invalid_odds_and_lines_rejected():
    assert Q.reject_reason(quote(odds=1.0)) == "quota_non_valida"
    assert Q.reject_reason(quote(line=24.3)) == "linea_non_supportata"


def test_last_quote_before_start_is_used_and_later_ones_ignored():
    qs = [quote(240, odds=1.80), quote(60, odds=1.95), quote(10, odds=2.05), quote(-30, odds=9.0)]
    sel, rej = Q.select_pre_game_quotes(qs)
    assert len(sel) == 1 and sel[0].quote.odds_decimal == 2.05
    assert rej["dopo_inizio"] == 1


def test_margin_only_from_same_snapshot_pairs():
    over, under = quote(60, Side.OVER, 1.90), quote(60, Side.UNDER, 1.90)
    sel, _ = Q.select_pre_game_quotes([over, under])
    assert all(abs(s.overround - (2 / 1.9 - 1)) < 1e-12 for s in sel)
    assert all(abs(s.novig_p_over - 0.5) < 1e-12 for s in sel)
    # Under catturato in un altro momento: nessuna coppia -> margine ignoto (non stimato)
    sel2, _ = Q.select_pre_game_quotes([quote(60, Side.OVER, 1.90), quote(30, Side.UNDER, 1.90)])
    assert all(s.overround is None for s in sel2)


def test_csv_capture_time_is_user_declared_timestamp_not_import_time():
    q = quote(60, source="csv")
    q = OddsQuote(**{**q.__dict__, "fetched_at": datetime(2026, 10, 26, tzinfo=UTC)})  # importato molto dopo
    sel, rej = Q.select_pre_game_quotes([q])
    assert len(sel) == 1 and not rej
    # senza timestamp dichiarato verrebbe usato il momento dell'import, quindi scartata
    q2 = OddsQuote(**{**q.__dict__, "quote_time": None})
    assert Q.reject_reason(q2) == "dopo_inizio"


def test_game_date_uses_us_eastern_calendar_day():
    assert Q.game_date_of(datetime(2026, 1, 11, 0, 30, tzinfo=UTC)) == date(2026, 1, 10)


def _log(pid, gid, d, pts=20, minutes=30.0):
    return PlayerGameLog(source="t", player_id=pid, player_name="Mario Rossi", game_id=gid, game_date=d, season=2025, team="A", opponent="B",
                         is_home=True, minutes=minutes, pts=pts, reb=5, ast=5, fg3m=2, fg3a=5, fgm=8, fga=15)


def test_match_requires_game_on_event_date_never_other_dates():
    sel, _ = Q.select_pre_game_quotes([quote(60)])
    pool = {"mario rossi": [_log(1, "g1", date(2026, 1, 10)), _log(1, "g2", date(2026, 1, 12))]}
    matched, rej = Q.match_to_games(sel, pool)
    assert len(matched) == 1 and matched[0][2].game_id == "g1"
    matched, rej = Q.match_to_games(sel, {"mario rossi": [_log(1, "g2", date(2026, 1, 12))]})
    assert matched == [] and rej["partita_non_trovata"] == 1
    # non ha giocato (0 minuti): nessun esito
    matched, rej = Q.match_to_games(sel, {"mario rossi": [_log(1, "g1", date(2026, 1, 10), minutes=0.0)]})
    assert matched == []


# ---------------------------------------------------------------- metriche
def test_brier_and_logloss_known_values():
    assert M.brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert M.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(np.log(2))
    assert M.brier([1.0, 0.0], [1, 0]) == 0.0
    assert M.log_loss([1.0, 0.0], [1, 0]) < 1e-9  # clip, nessun inf


def test_wilson_interval():
    lo, hi = M.wilson(50, 100, 0.95)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert M.wilson(0, 0)[0] != M.wilson(0, 0)[0]  # nan


def test_cluster_bootstrap_contains_truth_and_narrows_with_n():
    rng = np.random.default_rng(0)

    def width(n_days):
        days = np.repeat(np.arange(n_days), 10)
        x = rng.normal(0.0, 1.0, len(days))
        e = M.mean_ci(x, days, B=800, seed=1, level=0.90)
        assert e.lo <= e.value <= e.hi
        return e.hi - e.lo

    assert width(400) < width(25)


def test_cluster_bootstrap_is_wider_when_clustered():
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(30), 20)
    shock = np.repeat(rng.normal(0, 1, 30), 20)  # forte correlazione intra-giornata
    x = shock + rng.normal(0, 0.1, len(days))
    clustered = M.mean_ci(x, days, B=800, seed=2)
    naive = M.mean_ci(x, np.arange(len(x)), B=800, seed=2)
    assert (clustered.hi - clustered.lo) > 2 * (naive.hi - naive.lo)


def test_max_drawdown_hand_computed():
    # cumulata: +1, +3, 0, -1, +1 -> picco 3, minimo -1 => drawdown 4
    dd, longest = M.max_drawdown([1, 2, -3, -1, 2])
    assert dd == pytest.approx(4.0) and longest == 3
    assert M.max_drawdown([1, 1, 1]) == (0.0, 0)
    assert M.max_drawdown([-2, -1])[0] == pytest.approx(3.0)  # il capitale parte da 0


def _bets(results, odds, stake=1.0):
    n = len(results)
    prof = [stake * (o - 1) if r == "win" else 0.0 if r == "push" else -stake for r, o in zip(results, odds)]
    return pd.DataFrame({"stake": stake, "profit": prof, "result": results, "odds": odds,
                         "date": [date(2026, 1, 1) + timedelta(days=i // 2) for i in range(n)],
                         "commence_time": [str(i) for i in range(n)], "player": "p", "market": "points", "line": 1.5})


def test_summarize_bets_roi_yield_hitrate_with_push():
    df = _bets(["win", "loss", "push", "win"], [2.0, 1.9, 1.8, 1.5])
    s = M.summarize_bets(df, bankroll=100, B=200)
    assert s["profitto"] == pytest.approx(1.0 - 1.0 + 0.0 + 0.5)
    assert s["yield"] == pytest.approx(0.5 / 4)
    assert s["roi_capitale"] == pytest.approx(0.5 / 100)
    assert s["hit_rate"] == pytest.approx(2 / 3)  # il rimborso e' escluso
    assert (s["vinte"], s["perse"], s["rimborsate"]) == (2, 1, 1)


def test_empty_bets_summary_is_safe():
    assert M.summarize_bets(pd.DataFrame(columns=["stake", "profit", "result", "date", "odds"]))["n_scommesse"] == 0


# ---------------------------------------------------------------- baseline
def test_empirical_baseline_laplace_and_push():
    vals = [10] * 20
    lt, eq, gt = BL.p_empirical(vals, 10.5)
    assert (lt, eq) == (pytest.approx(21 / 22), 0.0) and gt == pytest.approx(1 / 22)
    lt, eq, gt = BL.p_empirical(vals, 10.0)
    assert lt + eq + gt == pytest.approx(1.0) and eq == pytest.approx(21 / 23)


def test_poisson_baseline_normalised_and_coin():
    for line in (3.5, 4.0):
        assert sum(BL.p_poisson([3, 4, 5, 4, 3], line)) == pytest.approx(1.0)
    assert BL.p_coin([1, 2, 3], 2.5) == (0.5, 0.0, 0.5)


def test_neutral_lines_are_half_lines_from_history_only():
    lines = BL.neutral_lines([20, 22, 18, 25, 19, 21] * 3, ymax=90)
    assert lines and all(abs(l % 1 - 0.5) < 1e-9 for l in lines)


# ---------------------------------------------------------------- scommesse: margine, push, nessuna scelta guardando l'esito
def _rows(specs):
    base = dict(model="modello", phase="test", player="p", game_id="g", date=date(2026, 1, 10), season=2025, market="points", line_source="quota_reale",
                capture_time="x", commence_time="y", minutes_before_start=60.0, y=25.0, obs_id="o")
    out = []
    for i, sp in enumerate(specs):
        r = {**base, "game_id": sp.get("game", "g"), "obs_id": f"o{i}", **sp}
        r["p_loss"] = 1 - r["p_win"] - r["p_push"]
        out.append(r)
    return pd.DataFrame(out)


def test_value_bet_includes_margin_via_real_odds_and_pushes():
    # p_win .55 p_push .10 quota 1.90: EV = .55*1.9 + .10 - 1 = 0.145 ; con quota 1.60 EV = -0.02 -> nessuna scommessa
    rows = _rows([dict(line=25.0, side="over", book="a", odds=1.90, p_win=.55, p_push=.10, result="push"),
                  dict(line=25.0, side="over", book="b", odds=1.60, p_win=.55, p_push=.10, result="push")])
    b = value_bets(rows, "modello", 0.0)
    assert len(b) == 1 and b.iloc[0]["book"] == "a"
    assert b.iloc[0]["ev"] == pytest.approx(0.145) and b.iloc[0]["profit"] == 0.0  # push: rimborso
    assert len(value_bets(rows[rows["book"] == "b"], "modello", 0.0)) == 0


def test_value_bet_choice_ignores_outcome():
    a = _rows([dict(line=24.5, side="over", book="a", odds=1.9, p_win=.6, p_push=0.0, result="win"),
               dict(line=24.5, side="under", book="a", odds=1.9, p_win=.4, p_push=0.0, result="loss")])
    flipped = a.copy()
    flipped["result"] = ["loss", "win"]
    assert value_bets(a, "modello", 0.0).iloc[0]["side"] == value_bets(flipped, "modello", 0.0).iloc[0]["side"] == "over"


def test_profit_arithmetic_win_loss():
    rows = _rows([dict(line=24.5, side="over", book="a", odds=2.10, p_win=.6, p_push=0.0, result="win", game="g1"),
                  dict(line=24.5, side="over", book="a", odds=2.10, p_win=.6, p_push=0.0, result="loss", game="g2")])
    b = value_bets(rows, "modello", 0.0, stake=2.0)
    assert sorted(b["profit"]) == [-2.0, pytest.approx(2.2)]


def test_naive_bets_pay_the_margin_on_efficient_prices():
    # prezzi equi con margine 6%: puntare sempre un lato ha EV = -margine/(1+margine) per costruzione
    p = 0.5
    odds = 1 / (p * 1.06)
    res = ["win", "loss"] * 500
    rows = _rows([dict(line=24.5, side="over", book="a", odds=odds, p_win=p, p_push=0.0, result=r, game=f"g{i}") for i, r in enumerate(res)])
    s = M.summarize_bets(naive_bets(rows, "over"), B=100)
    assert s["yield"] == pytest.approx(odds / 2 - 1, abs=1e-9) and s["yield"] < 0


# ---------------------------------------------------------------- verdetto
def _ok_tb(**kw):
    return {"n_scommesse": 400, "yield": 0.06, "yield_lo": 0.01, "yield_hi": 0.11, **kw}


def _vs(ok=True, n=500):
    return pd.DataFrame([{"baseline": "frequenza_ultime20", "n": n, "modello_migliore": ok}])


def _verdict(**over):
    args = dict(source="live", n_quote_rows_test=1000, test_bets=_ok_tb(), val_bets={"yield": 0.03}, vs_baselines_quotes=_vs(), vs_market=_vs(),
                calibration={"pendenza": 1.0}, min_bets=200, n_test_runs=1)
    args.update(over)
    return verdict(**args)


def test_verdict_demo_never_concludes():
    assert _verdict(source="demo")["code"] == "sintetico"


def test_verdict_without_real_quotes_does_not_compute_roi_claims():
    v = _verdict(n_quote_rows_test=0, test_bets=None)
    assert v["code"] == "solo_predittivo" and "non si inventano" in v["text"]


def test_verdict_passes_only_when_all_checks_pass():
    assert _verdict()["code"] == "verifica_superata"
    for kw in (dict(test_bets=_ok_tb(yield_lo=-0.01)), dict(test_bets=_ok_tb(n_scommesse=50)), dict(val_bets={"yield": -0.02}),
               dict(vs_baselines_quotes=_vs(False)), dict(vs_market=_vs(False)), dict(vs_market=_vs(True, n=20)),
               dict(calibration={"pendenza": 0.6}), dict(n_test_runs=3)):
        assert _verdict(**kw)["code"] != "verifica_superata", kw


def test_verdict_negative_and_small_sample_wording():
    assert _verdict(test_bets={**_ok_tb(), "yield": -0.08, "yield_lo": -0.15, "yield_hi": -0.01})["code"] == "fallita"
    assert _verdict(test_bets=_ok_tb(n_scommesse=40))["code"] == "campione_insufficiente"
    assert _verdict(test_bets={**_ok_tb(), "yield": 0.05, "yield_lo": -0.02})["code"] == "non_dimostrato"


def test_paired_vs_baselines_survives_empty_frame_concat():
    """Regressione: in Live una parte vuota rendeva 'object' le colonne e np.log falliva."""
    import numpy as np, pandas as pd
    from nba_vf.backtest import evaluate as ev
    from nba_vf.backtest.report import paired_vs_baselines
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        y = float(rng.random() < 0.5)
        for m, p in (("modello", 0.55), ("frequenza_ultime20", 0.5)):
            rows.append({c: "x" for c in ev.ROW_COLUMNS} | {
                "model": m, "phase": "test", "line_source": "quota_reale", "side": "over", "line": 20.5, "p_win": p, "p_push": 0.0,
                "p_loss": 1 - p, "y": y, "result": "win" if y else "loss", "obs_id": f"o{i}", "date": f"2025-01-{i % 28 + 1:02d}"})
    frame = pd.concat([ev.to_frame(rows), ev.to_frame([])], ignore_index=True)
    out = paired_vs_baselines(ev.coerce_numeric(frame), "test", "quota_reale", "modello", ["frequenza_ultime20"], B=50)
    assert len(out) == 1 and np.isfinite(out["delta_log_loss"].iloc[0])
    out2 = paired_vs_baselines(frame, "test", "quota_reale", "modello", ["frequenza_ultime20"], B=50)  # anche senza coerce a monte
    assert len(out2) == 1
