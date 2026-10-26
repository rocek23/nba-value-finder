import numpy as np
import pytest
from scipy import stats

from nba_vf.model.distributions import (
    MINUTES_GRID,
    betabinom_pmf_matrix,
    cdf_table,
    effective_n,
    fit_icc,
    minutes_pmf,
    nb_alpha_moment,
    nb_pmf_matrix,
    recency_weights,
    sample_rows,
    weighted_corr,
)


def test_nb_matches_scipy_and_sums_to_one():
    mu = np.array([0.5, 4.0, 17.3, 30.0])
    alpha = 0.12
    mat = nb_pmf_matrix(mu, alpha, 90)
    k = 1 / alpha
    for i, m in enumerate(mu):
        ref = stats.nbinom.pmf(np.arange(91), k, k / (k + m))
        assert np.allclose(mat[i, :-1], ref[:-1], atol=1e-7)
    assert np.allclose(mat.sum(axis=1), 1.0)
    assert (mat >= 0).all() and (mat <= 1).all()


def test_nb_mean_variance():
    mu, alpha = 12.0, 0.2
    pmf = nb_pmf_matrix(np.array([mu]), alpha, 200)[0]
    s = np.arange(201)
    assert pmf @ s == pytest.approx(mu, rel=1e-6)
    assert pmf @ (s - mu) ** 2 == pytest.approx(mu + alpha * mu**2, rel=1e-4)


def test_nb_poisson_limit():
    mat = nb_pmf_matrix(np.array([6.0]), 0.0, 40)[0]
    assert np.allclose(mat[:40], stats.poisson.pmf(np.arange(40), 6.0), atol=1e-12)
    near = nb_pmf_matrix(np.array([6.0]), 1e-5, 40)[0]
    assert np.allclose(near, mat, atol=1e-3)


def test_nb_tail_mass_goes_to_last_bin():
    mat = nb_pmf_matrix(np.array([30.0]), 0.3, 10)[0]
    assert mat.sum() == pytest.approx(1.0) and mat[-1] > 0.5


def test_betabinom_matches_scipy_and_binomial_limit():
    p, rho = 0.37, 0.1
    s = (1 - rho) / rho
    mat = betabinom_pmf_matrix(15, 15, p, rho)
    for a in (1, 5, 12):
        ref = stats.betabinom.pmf(np.arange(16), a, p * s, (1 - p) * s)
        assert np.allclose(mat[a], ref, atol=1e-10)
    binm = betabinom_pmf_matrix(15, 15, p, 0.0)
    assert np.allclose(binm[7], stats.binom.pmf(np.arange(16), 7, p), atol=1e-12)
    assert np.allclose(mat.sum(axis=1), 1.0)


def test_betabinom_has_more_variance_than_binomial():
    p, a = 0.37, 10
    v_bin = betabinom_pmf_matrix(12, 12, p, 0.0)[a]
    v_bb = betabinom_pmf_matrix(12, 12, p, 0.15)[a]
    x = np.arange(13)
    var = lambda pm: pm @ (x - pm @ x) ** 2
    assert var(v_bb) > var(v_bin)
    assert v_bb @ x == pytest.approx(v_bin @ x)  # stessa media


def test_icc_test_accepts_binomial_and_detects_overdispersion():
    rng = np.random.default_rng(1)
    a = rng.poisson(6, 400) + 1
    x_bin = rng.binomial(a, 0.37)
    assert fit_icc(x_bin, a, 0.37).p_value > 0.05
    s = (1 - 0.15) / 0.15
    x_bb = rng.binomial(a, rng.beta(0.37 * s, 0.63 * s, 400))
    res = fit_icc(x_bb, a, float(x_bb.sum() / a.sum()))
    assert res.p_value < 0.001 and res.rho == pytest.approx(0.15, abs=0.05)


def test_icc_degenerate_inputs():
    assert fit_icc(np.array([1, 2]), np.array([3, 4]), 0.4).p_value == 1.0  # troppo pochi dati
    assert fit_icc(np.zeros(20), np.zeros(20), 0.4).p_value == 1.0  # nessun tentativo


def test_alpha_moment_estimator():
    rng = np.random.default_rng(2)
    mu = rng.uniform(5, 20, 20_000)
    k = 1 / 0.15
    y = rng.negative_binomial(k, k / (k + mu))
    assert nb_alpha_moment(y, mu, np.ones_like(mu)) == pytest.approx(0.15, abs=0.02)
    yp = rng.poisson(mu)
    assert nb_alpha_moment(yp, mu, np.ones_like(mu)) < 0.02


def test_recency_weights_and_effective_n():
    w = recency_weights(5, 2.0)
    assert w[-1] == 1.0 and w[-3] == pytest.approx(0.5)
    assert np.all(np.diff(w) > 0)
    assert recency_weights(4, None).tolist() == [1, 1, 1, 1]
    assert effective_n(np.ones(10)) == pytest.approx(10.0)
    assert effective_n(w) < 5


def test_weighted_corr_properties():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((500, 3))
    x[:, 1] += 0.8 * x[:, 0]
    c = weighted_corr(x, np.ones(500))
    assert np.allclose(np.diag(c), 1.0) and np.allclose(c, c.T)
    assert c[0, 1] == pytest.approx(np.corrcoef(x.T)[0, 1], abs=1e-9)
    assert weighted_corr(np.column_stack([x[:, 0], np.ones(500)]), np.ones(500))[0, 1] == 0.0


def test_minutes_pmf_normalised_and_hits_target_mean():
    rng = np.random.default_rng(4)
    m = np.clip(rng.normal(30, 5, 80), 8, 46)
    w = recency_weights(80, 10)
    for target in (24.0, 30.0, 35.5):
        pmf = minutes_pmf(m, w, target, 0.4)
        assert pmf.sum() == pytest.approx(1.0) and (pmf >= 0).all()
        assert pmf @ MINUTES_GRID == pytest.approx(target, abs=0.15)


def test_sample_rows_reproduces_pmf():
    rng = np.random.default_rng(5)
    pmf = nb_pmf_matrix(np.array([3.0, 9.0]), 0.2, 30)
    table, ncols = cdf_table(pmf)
    n = 200_000
    rows = rng.integers(0, 2, n)
    y = sample_rows(table, ncols, rows, rng.random(n))
    for r in (0, 1):
        emp = np.bincount(y[rows == r], minlength=31) / (rows == r).sum()
        assert 0.5 * np.abs(emp - pmf[r]).sum() < 0.02  # distanza di variazione totale
