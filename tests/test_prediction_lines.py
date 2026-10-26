import re
from pathlib import Path

import numpy as np
import pytest

from nba_vf.model.errors import InvalidLineError
from nba_vf.model.odds_math import fair_odds
from nba_vf.models import Market, Side

ALL = list(Market)


def test_all_markets_priced_and_pmfs_valid(prediction):
    assert set(prediction.markets) == set(ALL)
    for mk, mp in prediction.markets.items():
        assert mp.pmf.sum() == pytest.approx(1.0)
        assert (mp.pmf >= 0).all() and (mp.pmf <= 1).all()
        assert mp.pmf[-1] < 1e-4, f"{mk}: massa di coda non trascurabile, aumentare ymax"


@pytest.mark.parametrize("market", ALL)
def test_probabilities_in_unit_interval_and_complementary_for_every_threshold(prediction, market):
    mp = prediction.market(market)
    tab = mp.line_table()
    assert len(tab) > 5
    assert tab[["p_over", "p_push", "p_under"]].min().min() >= 0.0
    assert tab[["p_over", "p_push", "p_under"]].max().max() <= 1.0
    assert (tab.p_over + tab.p_push + tab.p_under).sub(1.0).abs().max() < 1e-9
    # Over decrescente nella linea, Under crescente
    assert (np.diff(tab.p_over.to_numpy()) <= 1e-12).all() and (np.diff(tab.p_under.to_numpy()) >= -1e-12).all()
    # push solo sulle linee intere
    half = (tab.linea * 2) % 2 == 1
    assert (tab.loc[half, "p_push"] == 0).all()
    assert (tab.loc[~half, "p_push"] > 0).any()


@pytest.mark.parametrize("market", ALL)
def test_integer_and_half_thresholds_agree(prediction, market):
    mp = prediction.market(market)
    L = max(2, mp.median)
    over_int = mp.side_probs(L, Side.OVER)
    assert over_int.p_win == pytest.approx(mp.side_probs(L + 0.5, Side.OVER).p_win)
    assert over_int.p_win + over_int.p_push == pytest.approx(mp.side_probs(L - 0.5, Side.OVER).p_win)
    assert over_int.p_push == pytest.approx(mp.pmf[L])
    under_int = mp.side_probs(L, Side.UNDER)
    assert under_int.p_win == pytest.approx(mp.side_probs(L - 0.5, Side.UNDER).p_win)
    assert under_int.p_win == pytest.approx(over_int.p_loss) and under_int.p_loss == pytest.approx(over_int.p_win)


def test_invalid_lines_rejected(prediction):
    mp = prediction.market(Market.POINTS)
    for bad in (20.3, -1, float("nan")):
        with pytest.raises(InvalidLineError):
            mp.side_probs(bad, Side.OVER)
    with pytest.raises(InvalidLineError):
        mp.side_probs(10_000.5, Side.OVER)


def test_price_push_aware_vs_binary(prediction):
    mp = prediction.market(Market.POINTS)
    L, d = mp.median, 1.91
    v = mp.price(L, Side.OVER, d)
    assert v.p_push > 0.02
    assert v.p_win + v.p_push + v.p_loss == pytest.approx(1.0)
    assert v.roi == pytest.approx(v.p_win * d + v.p_push - 1.0)
    assert v.fair_odds == pytest.approx((1.0 - v.p_push) / v.p_win)
    assert v.fair_odds != pytest.approx(1.0 / v.p_win)  # la formula binaria sarebbe sbagliata
    assert v.breakeven_p_win == pytest.approx((1.0 - v.p_push) / d)
    assert v.edge == pytest.approx(v.p_win - v.breakeven_p_win)
    assert v.ev(25.0) == pytest.approx(25.0 * v.roi)
    # alla quota equa il ROI e' zero
    assert mp.price(L, Side.OVER, v.fair_odds).roi == pytest.approx(0.0, abs=1e-9)


def test_over_under_sum_of_rois_reflects_margin(prediction):
    mp = prediction.market(Market.ASSISTS)
    L = mp.median
    o, u = mp.price(L, Side.OVER, 1.9), mp.price(L, Side.UNDER, 1.9)
    # stessa linea, stessa quota: ROI_over + ROI_under = d*(p_w_o + p_w_u) + 2 p_push - 2 = (d-1)(1-p_push)*... > -0.2
    assert o.p_win == pytest.approx(u.p_loss) and o.p_push == pytest.approx(u.p_push)
    assert o.roi + u.roi == pytest.approx(1.9 * (o.p_win + u.p_win) + 2 * o.p_push - 2.0)


def test_intervals_present_ordered_and_contain_point(prediction):
    mp = prediction.market(Market.POINTS)
    L = mp.median + 0.5
    v = mp.price(L, Side.OVER, 1.9)
    for ci, point in ((v.p_win_ci, v.p_win), (v.roi_ci, v.roi), (v.fair_odds_ci, v.fair_odds)):
        assert ci is not None and ci[0] <= ci[1]
        assert ci[0] - 0.05 <= point <= ci[1] + 0.05
    assert v.ci_level == 0.90
    lo, hi = mp.mean_ci()
    assert lo < mp.mean < hi
    assert "intervalli_coprono_solo_incertezza_di_stima" in v.flags


def test_intervals_wider_with_less_data(engine):
    from nba_vf.model.synthetic import TruePlayer, generate_player

    short = engine.predict(generate_player(TruePlayer(), 15, 9)).market(Market.POINTS)
    long = engine.predict(generate_player(TruePlayer(), 100, 9)).market(Market.POINTS)
    w = lambda mp: np.subtract(*mp.mean_ci()[::-1])
    assert w(short) > w(long)


def test_no_intervals_option(engine, logs):
    p = engine.predict(logs, intervals=False)
    v = p.market(Market.POINTS).price(15.5, Side.OVER, 1.9)
    assert v.p_win_ci is None and v.roi_ci is None
    assert "intervallo non calcolato" in v.verdict or "non positivo" in v.verdict


def test_unpriceable_threes_when_few_attempts(engine):
    from nba_vf.model.synthetic import TruePlayer, generate_player

    truth = TruePlayer(rho={"pts": 0.5, "reb": 0.2, "ast": 0.1, "fg3a": 0.004})
    p = engine.predict(generate_player(truth, 30, 3), intervals=False)
    assert Market.THREES not in p.markets
    with pytest.raises(KeyError):
        p.market(Market.THREES)


def test_fair_odds_vectorised_with_zero_win():
    assert fair_odds(np.array([0.5, 0.0]), np.array([0.0, 0.2])).tolist() == [2.0, float("inf")]


def test_verdict_wording_never_claims_certainty(prediction):
    mp = prediction.market(Market.POINTS)
    verdicts = {mp.price(L, s, d).verdict for L in (12.5, 17.5, 22.5) for s in Side for d in (1.5, 1.9, 3.0)}
    assert verdicts
    forbidden = re.compile(r"sicur|garantit|certo|infallibil", re.I)
    assert not any(forbidden.search(v) for v in verdicts)


def test_source_code_never_calls_opportunities_safe_or_guaranteed():
    forbidden = re.compile(r"sicur[aeio]|garantit", re.I)
    root = Path(__file__).resolve().parents[1] / "src" / "nba_vf" / "model"
    for f in root.glob("*.py"):
        assert not forbidden.search(f.read_text(encoding="utf-8")), f.name


def test_reproducible(engine, logs):
    a = engine.predict(logs, intervals=False).market(Market.PRA).pmf
    b = engine.predict(logs, intervals=False).market(Market.PRA).pmf
    assert np.array_equal(a, b)
