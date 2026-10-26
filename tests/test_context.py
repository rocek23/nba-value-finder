from datetime import date, timedelta

import numpy as np
import pytest

from nba_vf.model.config import ModelConfig
from nba_vf.model.context import MIN_COND, b2b_flags, estimate_effects, rest_days
from nba_vf.model.engine import GameContext
from nba_vf.model.synthetic import TruePlayer, generate_player
from nba_vf.models import Market

CFG = ModelConfig()


def test_rest_days_and_flags():
    logs = generate_player(TruePlayer(), 30, 1)
    rd = rest_days(logs)
    assert rd[0] is None and all(d >= 1 for d in rd[1:])
    assert b2b_flags(logs).tolist() == [d == 1 for d in rd]


def test_no_true_effect_not_applied():
    rejected = 0
    for seed in range(6):
        eff = estimate_effects(generate_player(TruePlayer(), 400, seed), CFG)
        rejected += sum(not e.applied for e in eff.values())
    assert rejected >= 28  # su 30 test (6 giocatori x 5 target) al massimo 2 falsi positivi


def test_true_effect_is_detected_and_shrunk():
    truth = TruePlayer(b2b_minutes_ratio=0.80, b2b_share=0.3)
    eff = estimate_effects(generate_player(truth, 500, 3), CFG)
    m = eff["minutes"]
    assert m.applied and m.n_cond >= MIN_COND
    assert m.ratio == pytest.approx(0.80, abs=0.06)
    assert m.ci[1] < 1.0
    assert m.ratio < m.shrunk_ratio < 1.0  # lo shrinkage riduce l'effetto, non lo inverte
    assert "verificato" in m.reason


def test_small_sample_is_rejected_with_reason():
    eff = estimate_effects(generate_player(TruePlayer(b2b_minutes_ratio=0.7), 40, 2), CFG)
    assert all(not e.applied for e in eff.values())
    assert any("insufficiente" in e.reason for e in eff.values())
    assert all(e.shrunk_ratio == 1.0 for e in eff.values())


def test_engine_applies_only_verified_effects_and_only_when_requested(engine):
    logs = generate_player(TruePlayer(b2b_minutes_ratio=0.80, b2b_share=0.3), 300, 3)
    plain = engine.predict(logs, intervals=False)
    assert not plain.adjustments and plain.effects == {}
    b2b = engine.predict(logs, GameContext(back_to_back=True), intervals=False)
    assert any(a.target == "minutes" for a in b2b.adjustments)
    assert "aggiustamento_contesto_applicato:back_to_back" in b2b.flags
    assert b2b.fit.minutes_mean < plain.fit.minutes_mean - 1.5
    assert b2b.market(Market.POINTS).mean < b2b.base_markets[Market.POINTS].mean  # base e aggiustata distinte
    assert b2b.base_markets[Market.POINTS].mean == pytest.approx(plain.market(Market.POINTS).mean)
    not_b2b = engine.predict(logs, GameContext(back_to_back=False), intervals=False)
    assert not not_b2b.adjustments
    assert not_b2b.market(Market.POINTS).mean == pytest.approx(plain.market(Market.POINTS).mean)


def test_unverified_context_leaves_prediction_unchanged(engine, logs):
    plain = engine.predict(logs, intervals=False)
    ctx = engine.predict(logs, GameContext(back_to_back=True), intervals=False)  # effetto non dimostrabile
    assert not ctx.adjustments
    assert np.array_equal(plain.market(Market.POINTS).pmf, ctx.market(Market.POINTS).pmf)


def test_opponent_is_recorded_but_has_no_numeric_effect(engine, logs):
    a = engine.predict(logs, GameContext(opponent="BOS"), intervals=False)
    b = engine.predict(logs, GameContext(opponent="DET"), intervals=False)
    c = engine.predict(logs, GameContext(), intervals=False)
    for mk in Market:
        assert np.array_equal(a.market(mk).pmf, b.market(mk).pmf)
        assert np.array_equal(a.market(mk).pmf, c.market(mk).pmf)
    assert "avversario_registrato_non_usato_nel_calcolo" in a.flags


def test_manual_minutes_override_is_flagged(engine, logs):
    plain = engine.predict(logs, intervals=False)
    ov = engine.predict(logs, GameContext(minutes_override=38.0), intervals=False)
    assert "minuti_manuali" in ov.flags
    assert ov.fit.minutes_mean == pytest.approx(38.0, abs=0.3)
    assert ov.market(Market.POINTS).mean > plain.market(Market.POINTS).mean
    assert ov.market(Market.POINTS).mean / plain.market(Market.POINTS).mean == pytest.approx(
        38.0 / plain.fit.minutes_mean, rel=0.05
    )
