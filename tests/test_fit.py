import dataclasses

import numpy as np
import pytest

from nba_vf.model.config import LeaguePrior, ModelConfig
from nba_vf.model.errors import InsufficientDataError
from nba_vf.model.fit import build_inputs, fit_arrays, ratio_with_prior
from nba_vf.model.prior import estimate_league_prior
from nba_vf.model.synthetic import TruePlayer, generate_player, generate_pool


def fit_player(truth, n, seed, prior=None, cfg=None, **kw):
    cfg = cfg or ModelConfig()
    inp = build_inputs(generate_player(truth, n, seed), cfg)
    return fit_arrays(inp, cfg, prior or LeaguePrior.fallback(), **kw)


def test_parameter_recovery_large_sample():
    truth = TruePlayer()
    cfg = ModelConfig(half_life_minutes=None, half_life_rates=None)  # senza recenza: i dati sono stazionari
    f = fit_player(truth, 1500, 11, cfg=cfg)
    for k in ("pts", "reb", "ast", "fg3a"):
        assert f.rho[k] == pytest.approx(truth.rho[k], rel=0.05)
        assert f.alpha[k] == pytest.approx(truth.alpha[k], abs=0.05)
    assert f.p3 == pytest.approx(truth.p3, abs=0.02)
    assert f.minutes_mean == pytest.approx(30.5, abs=1.5)  # 32 con il 10% di partite 'blowout' piu' corte
    assert f.R[0, 2] == pytest.approx(0.15, abs=0.08)


def test_binomial_vs_beta_binomial_selection():
    cfg = ModelConfig()
    adopted_bin = [fit_player(TruePlayer(icc3=0.0), 200, s, cfg=cfg).use_bb for s in range(10)]
    adopted_bb = [fit_player(TruePlayer(icc3=0.12), 200, s, cfg=cfg).use_bb for s in range(10)]
    assert sum(adopted_bin) <= 3  # falsi positivi: ~ soglia 10%
    assert sum(adopted_bb) >= 8
    f = fit_player(TruePlayer(icc3=0.12), 400, 3, cfg=cfg)
    assert f.use_bb and 0.03 < f.icc3 < 0.2


def test_threes_not_priced_with_too_few_attempts():
    truth = TruePlayer(rho={"pts": 0.5, "reb": 0.2, "ast": 0.1, "fg3a": 0.005})
    f = fit_player(truth, 15, 2)
    assert not f.three_ok and f.p3 is None
    assert "triple_non_prezzate_tentativi_insufficienti" in f.flags


def test_shrinkage_pulls_small_samples_toward_prior():
    y = np.array([30.0, 28.0, 33.0, 25.0, 31.0])
    m = np.array([30.0, 28.0, 32.0, 30.0, 31.0])
    w = np.ones(5)
    raw = y.sum() / m.sum()
    shrunk = ratio_with_prior(y, m, w, rho0=0.45, strength=100.0)
    assert 0.45 < shrunk < raw or raw < shrunk < 0.45
    assert ratio_with_prior(y, m, w, 0.45, 0.0) == pytest.approx(raw)  # strength 0 = nessuno shrinkage
    assert abs(ratio_with_prior(y, m, w, 0.45, 1e9) - 0.45) < 1e-4  # strength enorme = solo prior


def test_pool_prior_shrinks_more_when_sample_is_small():
    pool = generate_pool(40, 60, 1)
    prior = estimate_league_prior(pool)
    truth = TruePlayer(rho={"pts": 0.80, "reb": 0.20, "ast": 0.15, "fg3a": 0.14})  # tasso estremo
    cfg = ModelConfig(half_life_rates=None)
    weights = {}
    for n in (12, 400):
        logs = generate_player(truth, n, 4)
        f = fit_arrays(build_inputs(logs, cfg), cfg, prior)
        raw = sum(l.pts for l in logs) / sum(l.minutes for l in logs)
        # f.rho = (1 - w) * raw + w * prior  ->  peso del prior stimato dal risultato
        weights[n] = (f.rho["pts"] - raw) / (prior.rho0["pts"] - raw)
        t0 = prior.rate_strength_minutes["pts"]
        assert weights[n] == pytest.approx(t0 / (sum(l.minutes for l in logs) + t0), rel=1e-6)
        assert ("campione_ridotto" in f.flags) == (n < cfg.low_sample_games)
    assert weights[12] > 3 * weights[400] > 0  # a parita' di prior, il campione piccolo e' piu' regolarizzato
    big = fit_arrays(build_inputs(generate_player(truth, 400, 4), cfg), cfg, prior)
    assert big.rho["pts"] == pytest.approx(0.80, rel=0.08)


def test_fallback_prior_is_flagged_and_uses_no_external_numbers():
    f = fit_player(TruePlayer(), 40, 6)
    assert f.prior_source == "fallback" and "prior_fallback_nessun_pool" in f.flags
    f2 = fit_player(TruePlayer(), 40, 6, prior=estimate_league_prior(generate_pool(30, 50, 2)))
    assert f2.prior_source == "pool" and "prior_fallback_nessun_pool" not in f2.flags


def test_too_few_games_raises():
    with pytest.raises(InsufficientDataError):
        fit_player(TruePlayer(), 3, 1)


def test_minutes_override_is_manual_and_flagged():
    f0 = fit_player(TruePlayer(), 60, 8)
    f1 = fit_player(TruePlayer(), 60, 8, minutes_override=20.0)
    assert f1.minutes_mean == pytest.approx(20.0, abs=0.2) and "minuti_manuali" in f1.flags
    assert f0.minutes_mean > f1.minutes_mean
    assert f1.rho == f0.rho  # i tassi per minuto non cambiano: solo la proiezione dei minuti


def test_minutes_scale_scales_projection():
    f0 = fit_player(TruePlayer(), 60, 8)
    f1 = fit_player(TruePlayer(), 60, 8, minutes_scale=0.9)
    assert f1.minutes_mean == pytest.approx(0.9 * f0.minutes_mean, abs=0.3)


def test_estimate_league_prior_recovers_population_and_requires_pool():
    pool = generate_pool(60, 70, 7)
    prior = estimate_league_prior(pool)
    assert prior.is_pool and prior.n_players == 60
    assert prior.rho0["pts"] == pytest.approx(0.45, rel=0.12)
    assert prior.p3_prior is not None
    a, b = prior.p3_prior
    assert a / (a + b) == pytest.approx(0.36, abs=0.03)
    assert prior.resid_corr.shape == (3, 3)
    with pytest.raises(ValueError):
        estimate_league_prior({k: v for k, v in list(pool.items())[:5]})


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(min_games=2)
    with pytest.raises(ValueError):
        ModelConfig(ci_level=1.0)
    with pytest.raises(ValueError):
        ModelConfig(half_life_rates=-3)
    assert dataclasses.is_dataclass(ModelConfig())
