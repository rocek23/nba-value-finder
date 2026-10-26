import dataclasses

import numpy as np
import pytest

from nba_vf.model.config import ModelConfig
from nba_vf.model.fit import build_inputs, fit_arrays
from nba_vf.model.predict import PraDraws, pra_pmf_independent, pra_pmf_sim, pra_samples, stat_pmf
from nba_vf.model.synthetic import TruePlayer, generate_player

CFG = ModelConfig()


@pytest.fixture(scope="module")
def fit(engine, logs):
    return fit_arrays(build_inputs(logs, CFG), CFG, engine.prior)


def moments(pmf):
    s = np.arange(len(pmf))
    m = pmf @ s
    return m, pmf @ (s - m) ** 2


def with_corr(fit, r):
    R = np.array([[1, r, r], [r, 1, r], [r, r, 1]], dtype=float)
    return dataclasses.replace(fit, R=R)


def test_simulated_marginals_match_analytic_pmfs(fit):
    draws = PraDraws.make(200_000, 1)
    s = pra_samples(fit, CFG, draws)
    for j, (key, ymax) in enumerate((("pts", 90), ("reb", 45), ("ast", 35))):
        analytic = stat_pmf(fit, key, ymax)
        emp = np.bincount(np.minimum(s[:, j], ymax), minlength=ymax + 1) / len(s)
        assert 0.5 * np.abs(emp - analytic).sum() < 0.02  # TV
        assert s[:, j].mean() == pytest.approx(analytic @ np.arange(ymax + 1), rel=0.01)


def test_pra_is_sum_of_components_and_mean_adds_up(fit):
    draws = PraDraws.make(100_000, 2)
    s = pra_samples(fit, CFG, draws)
    pmf = pra_pmf_sim(fit, CFG, draws)
    assert pmf.sum() == pytest.approx(1.0)
    assert moments(pmf)[0] == pytest.approx(s.sum(axis=1).mean(), abs=1e-9)
    expected = sum(stat_pmf(fit, k, y) @ np.arange(y + 1) for k, y in (("pts", 90), ("reb", 45), ("ast", 35)))
    assert moments(pmf)[0] == pytest.approx(expected, rel=0.01)


def test_zero_residual_correlation_matches_analytic_convolution(fit):
    f0 = with_corr(fit, 0.0)
    sim = pra_pmf_sim(f0, CFG, PraDraws.make(400_000, 3))
    ana = pra_pmf_independent(f0, CFG)
    assert 0.5 * np.abs(sim - ana).sum() < 0.015
    assert moments(sim)[1] == pytest.approx(moments(ana)[1], rel=0.02)


def test_positive_dependence_widens_pra_and_negative_narrows(fit):
    d = PraDraws.make(200_000, 4)
    v0 = moments(pra_pmf_sim(with_corr(fit, 0.0), CFG, d))[1]
    vp = moments(pra_pmf_sim(with_corr(fit, 0.4), CFG, d))[1]
    vn = moments(pra_pmf_sim(with_corr(fit, -0.3), CFG, d))[1]
    assert vp > 1.15 * v0 > 1.15 * vn
    # la media non dipende dalla correlazione
    assert moments(pra_pmf_sim(with_corr(fit, 0.4), CFG, d))[0] == pytest.approx(moments(pra_pmf_sim(fit, CFG, d))[0], rel=0.01)


def test_shared_minutes_create_dependence_even_with_identity_copula(fit):
    d = PraDraws.make(150_000, 5)
    s = pra_samples(with_corr(fit, 0.0), CFG, d)
    assert np.corrcoef(s[:, 0], s[:, 2])[0, 1] > 0.05  # i minuti condivisi inducono correlazione positiva
    ind_var = sum(np.var(s[:, j]) for j in range(3))
    assert np.var(s.sum(axis=1)) > ind_var


def test_non_psd_correlation_is_handled(fit):
    bad = dataclasses.replace(fit, R=np.array([[1, 0.95, -0.95], [0.95, 1, 0.95], [-0.95, 0.95, 1.0]]))
    pmf = pra_pmf_sim(bad, CFG, PraDraws.make(20_000, 6))
    assert pmf.sum() == pytest.approx(1.0) and np.isfinite(pmf).all()


def test_common_random_numbers_and_seed_reproducibility(fit):
    a = pra_pmf_sim(fit, CFG, PraDraws.make(50_000, 7))
    b = pra_pmf_sim(fit, CFG, PraDraws.make(50_000, 7))
    c = pra_pmf_sim(fit, CFG, PraDraws.make(50_000, 8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    # con CRN una piccola variazione dei parametri produce una piccola variazione della pmf
    f2 = dataclasses.replace(fit, rho={**fit.rho, "pts": fit.rho["pts"] * 1.01})
    d = PraDraws.make(50_000, 7)
    assert 0.5 * np.abs(pra_pmf_sim(f2, CFG, d) - a).sum() < 0.03


def test_monte_carlo_error_shrinks_with_more_simulations(fit):
    def spread(n):
        return np.std([pra_pmf_sim(fit, CFG, PraDraws.make(n, s)).cumsum()[27] for s in range(12)])

    assert spread(100_000) < spread(5_000)


def test_pra_tail_mass_negligible(fit):
    assert pra_pmf_sim(fit, CFG, PraDraws.make(50_000, 9))[-1] < 1e-4
