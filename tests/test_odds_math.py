import math

import numpy as np
import pytest

from nba_vf.model.errors import InvalidLineError, ModelError
from nba_vf.model.odds_math import (
    SideProbs,
    breakeven_win_prob,
    classify_line,
    expected_roi,
    expected_value,
    fair_odds,
    implied_probability,
    side_probs_arrays,
    split_probs,
)

PMF = np.array([0.10, 0.20, 0.30, 0.25, 0.15])  # esempio fatto a mano


def test_hand_computed_integer_line_has_separate_push():
    lt, eq, gt = split_probs(PMF, 2)  # linea intera 2
    assert (lt, eq, gt) == pytest.approx((0.30, 0.30, 0.40))


def test_hand_computed_half_lines_have_no_push():
    assert split_probs(PMF, 2.5) == pytest.approx((0.60, 0.0, 0.40))
    assert split_probs(PMF, 1.5) == pytest.approx((0.30, 0.0, 0.70))
    assert split_probs(PMF, 0.5) == pytest.approx((0.10, 0.0, 0.90))


def test_over_and_under_sides_and_complementarity():
    w, p, l = side_probs_arrays(PMF, 2, "over")
    uw, up, ul = side_probs_arrays(PMF, 2, "under")
    assert (w, p, l) == pytest.approx((0.40, 0.30, 0.30))
    assert (uw, up, ul) == pytest.approx((0.30, 0.30, 0.40))
    assert w == pytest.approx(ul) and l == pytest.approx(uw) and p == pytest.approx(up)


@pytest.mark.parametrize("line", [0.5, 1, 1.5, 2, 2.5, 3, 3.5])
def test_complementarity_all_lines(line):
    w, p, l = side_probs_arrays(PMF, line, "over")
    uw, up, ul = side_probs_arrays(PMF, line, "under")
    assert w + p + l == pytest.approx(1.0)
    assert uw + up + ul == pytest.approx(1.0)
    assert w == pytest.approx(ul) and l == pytest.approx(uw)
    assert 0 <= w <= 1 and 0 <= p <= 1 and 0 <= l <= 1


def test_integer_and_half_thresholds_are_consistent():
    # Over 2 vince con Y>=3 (come Over 2.5); Under 2 vince con Y<=1 (come Under 1.5)
    assert side_probs_arrays(PMF, 2, "over")[0] == pytest.approx(side_probs_arrays(PMF, 2.5, "over")[0])
    assert side_probs_arrays(PMF, 2, "under")[0] == pytest.approx(side_probs_arrays(PMF, 1.5, "under")[0])
    # il push di una linea intera e' esattamente la differenza tra le due mezze linee adiacenti
    assert side_probs_arrays(PMF, 2, "over")[1] == pytest.approx(PMF[2])
    assert side_probs_arrays(PMF, 1.5, "over")[0] - side_probs_arrays(PMF, 2.5, "over")[0] == pytest.approx(PMF[2])


def test_line_zero_integer():
    lt, eq, gt = split_probs(PMF, 0)
    assert (lt, eq, gt) == pytest.approx((0.0, 0.10, 0.90))


@pytest.mark.parametrize("bad", [2.3, 2.25, -1, -0.5, float("nan"), float("inf"), "abc", None, 4, 5, 100])
def test_invalid_lines_raise(bad):
    with pytest.raises(InvalidLineError):
        split_probs(PMF, bad)  # PMF ha 5 celle: le linee 4 e oltre cadono sulla coda accorpata


def test_classify_line_tolerates_float_noise():
    assert classify_line(2.0000000001, 10) == (2, True)
    assert classify_line(2.5, 10) == (2, False)


def test_pmf_must_sum_to_one():
    with pytest.raises(ModelError):
        split_probs(np.array([0.5, 0.2, 0.1]), 1.5)


def test_stack_of_pmfs():
    stack = np.vstack([PMF, PMF[::-1]])
    lt, eq, gt = split_probs(stack, 2)
    assert lt.shape == (2,) and gt[0] == pytest.approx(0.40) and gt[1] == pytest.approx(0.30)


# ---- quota equa, ROI, EV ----------------------------------------------------------------------
def test_roi_and_fair_odds_with_push_hand_computed():
    p_win, p_push, p_loss, d = 0.45, 0.10, 0.45, 1.90
    assert expected_roi(p_win, p_push, d) == pytest.approx(0.45 * 1.9 + 0.10 - 1.0)  # -0.045
    assert expected_roi(p_win, p_push, d) == pytest.approx(p_win * (d - 1) - p_loss)
    assert fair_odds(p_win, p_push) == pytest.approx(0.90 / 0.45)  # 2.00
    assert breakeven_win_prob(d, p_push) == pytest.approx(0.90 / 1.90)


def test_roi_is_zero_at_fair_odds():
    for p_win, p_push in [(0.30, 0.0), (0.42, 0.08), (0.10, 0.25), (0.6, 0.1)]:
        assert expected_roi(p_win, p_push, fair_odds(p_win, p_push)) == pytest.approx(0.0, abs=1e-12)


def test_push_changes_the_binary_formula_result():
    # con il push, la formula binaria 1/p e p*d-1 darebbe valori SBAGLIATI
    p_win, p_push = 0.45, 0.10
    assert fair_odds(p_win, p_push) != pytest.approx(1 / p_win)
    assert fair_odds(p_win, 0.0) == pytest.approx(1 / p_win)  # caso particolare: nessun push
    assert expected_roi(p_win, 0.0, 1.9) == pytest.approx(p_win * 1.9 - 1.0)


def test_p_push_is_required_no_implicit_binary_formula():
    with pytest.raises(TypeError):
        expected_roi(0.5, 1.9)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        fair_odds(0.5)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        breakeven_win_prob(1.9)  # type: ignore[call-arg]


def test_expected_value_money():
    roi = expected_roi(0.55, 0.0, 1.95)
    assert expected_value(50.0, roi) == pytest.approx(50.0 * (0.55 * 1.95 - 1.0))
    with pytest.raises(ModelError):
        expected_value(-1.0, roi)


def test_fair_odds_infinite_when_win_impossible():
    assert math.isinf(float(fair_odds(0.0, 0.2)))


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -2.0, float("nan"), float("inf")])
def test_invalid_decimal_odds(bad):
    with pytest.raises(ModelError):
        expected_roi(0.5, 0.0, bad)
    with pytest.raises(ModelError):
        implied_probability(bad)


def test_implied_probability_is_raw():
    assert implied_probability(2.0) == pytest.approx(0.5)


def test_sideprobs_validation():
    SideProbs(0.5, 0.1, 0.4)
    with pytest.raises(ModelError):
        SideProbs(0.5, 0.5, 0.5)
    with pytest.raises(ModelError):
        SideProbs(1.2, 0.0, -0.2)


def test_expected_roi_matches_monte_carlo_of_outcomes():
    rng = np.random.default_rng(0)
    y = rng.choice(len(PMF), size=400_000, p=PMF)
    d, line = 1.9, 2
    profit = np.where(y > line, d - 1.0, np.where(y == line, 0.0, -1.0))
    w, p, _ = side_probs_arrays(PMF, line, "over")
    assert profit.mean() == pytest.approx(float(expected_roi(w, p, d)), abs=0.01)
