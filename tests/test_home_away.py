"""Effetto casa/trasferta: entra nelle stime SOLO se migliora la previsione fuori campione. Dati SINTETICI."""

from datetime import date, timedelta

import numpy as np
import pytest

from nba_vf.model.config import ModelConfig
from nba_vf.model.engine import GameContext, ValueEngine
from nba_vf.model.home_away import MIN_PLAYERS, TARGETS, estimate_home_away
from nba_vf.models import Market, PlayerGameLog, Side


def make_pool(n_players=60, n_games=80, home_effect=0.0, seed=1):
    rng = np.random.default_rng(seed)
    pool = {}
    for p in range(n_players):
        base_pts, base_reb, base_ast, base_3 = rng.uniform(10, 26), rng.uniform(3, 9), rng.uniform(2, 7), rng.uniform(1, 3.5)
        logs = []
        for g in range(n_games):
            home = bool(rng.integers(0, 2))
            f = 1 + home_effect if home else 1.0
            pts, reb, ast, f3 = (int(rng.poisson(b * f)) for b in (base_pts, base_reb, base_ast, base_3))
            logs.append(PlayerGameLog("synthetic", p, f"P{p}", f"{p}-{g}", date(2026, 1, 1) + timedelta(days=2 * g), 2025, "AAA", "BBB", home,
                                      32.0, pts, reb, ast, f3, f3 * 2 + 2, pts // 2, pts, 3, 4))
        pool[p] = logs
    return pool


def test_real_effect_is_detected_validated_and_applied():
    res = estimate_home_away(make_pool(home_effect=0.12))
    assert set(res) == set(TARGETS)
    pts = res["pts"]
    assert pts.applied and pts.ci[0] > 1.0 and pts.gain_pct > 0 and pts.t_stat >= 1.645
    assert 1.05 < pts.ratio < 1.20  # ristimato su tutto lo storico, vicino al valore vero (1.12)
    assert pts.factor_home > 1 > pts.factor_away and pts.factor_home * pts.factor_away == pytest.approx(1.0)


def test_no_effect_is_not_applied():
    res = estimate_home_away(make_pool(home_effect=0.0, seed=3))
    assert not any(e.applied for e in res.values())
    assert all(e.factor_home == 1.0 and e.factor_away == 1.0 for e in res.values())
    assert all(e.reason for e in res.values())  # il motivo e' sempre dichiarato


def test_too_few_players_never_applies():
    res = estimate_home_away(make_pool(n_players=MIN_PLAYERS - 5, home_effect=0.3))
    assert not any(e.applied for e in res.values()) and "pochi giocatori" in res["pts"].reason


def test_missing_home_flag_is_ignored_not_guessed():
    pool = make_pool(home_effect=0.3)
    from dataclasses import replace

    pool = {k: [replace(l, is_home=None) for l in v] for k, v in pool.items()}
    assert not any(e.applied for e in estimate_home_away(pool).values())


def p_over(engine, logs, ctx, line=19.5):
    pred = engine.predict(logs, ctx)
    return pred.markets[Market.POINTS].price(line, Side.OVER, 1.9).p_win, pred


CFG = ModelConfig(n_boot=10, n_sims_pra=2000, n_sims_pra_boot=500)


def test_engine_uses_home_away_only_when_verified():
    pool = make_pool(home_effect=0.15)
    eng = ValueEngine(CFG)
    eng.fit_prior(pool)
    logs = pool[0]
    before = {h: p_over(eng, logs, GameContext(is_home=h))[0] for h in (True, False, None)}
    assert before[True] == before[False] == before[None]  # senza verifica non cambia nulla
    eng.fit_home_away(pool)
    assert eng.home_away["pts"].applied
    home, pred = p_over(eng, logs, GameContext(is_home=True))
    away, _ = p_over(eng, logs, GameContext(is_home=False))
    none, pred_none = p_over(eng, logs, GameContext(is_home=None))
    assert home > none > away
    assert any(e.factor == "casa_trasferta" for e in pred.adjustments) and not pred_none.adjustments
    assert any("casa_trasferta" in f for f in pred.flags)


def test_engine_unchanged_when_effect_not_proven():
    pool = make_pool(home_effect=0.0, seed=3)
    eng = ValueEngine(CFG)
    eng.fit_prior(pool)
    eng.fit_home_away(pool)
    a = p_over(eng, pool[0], GameContext(is_home=True))[0]
    b = p_over(eng, pool[0], GameContext(is_home=None))[0]
    assert a == b


def test_context_sets_is_home_from_team_and_event():
    from nba_vf.ui.board import context_for
    from nba_vf.ui.bundle import DataBundle, PlayerInfo
    from nba_vf.models import OddsQuote, utc_now

    b = DataBundle(mode="live", source_label="x", status="ok", updated_at=None, players={"p": PlayerInfo("p", "P", "LAL")})
    q = lambda home, away: OddsQuote("balldontlie", "bk", "1", "P", 1, Market.POINTS, 20.5, Side.OVER, 1.9, utc_now(), None, home, away, None)
    assert context_for(b, "p", q("LAL", "BOS")).is_home is True
    assert context_for(b, "p", q("BOS", "LAL")).is_home is False
    assert context_for(b, "p", q("Los Angeles Lakers", "Boston Celtics")).is_home is None  # nomi diversi: non si indovina
