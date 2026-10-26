"""Fixture condivise per i test del motore. Tutti i dati sono SINTETICI (nba_vf.model.synthetic)."""

import pytest

from nba_vf.model.config import ModelConfig
from nba_vf.model.engine import ValueEngine
from nba_vf.model.synthetic import TruePlayer, generate_player, generate_pool


@pytest.fixture(scope="session")
def pool():
    return generate_pool(40, 60, seed=1)


@pytest.fixture(scope="session")
def cfg_fast():
    return ModelConfig(n_boot=60, n_sims_pra=40_000, n_sims_pra_boot=8_000)


@pytest.fixture(scope="session")
def engine(pool, cfg_fast):
    eng = ValueEngine(cfg_fast)
    eng.fit_prior(pool)
    assert eng.prior.is_pool
    return eng


@pytest.fixture(scope="session")
def logs():
    return generate_player(TruePlayer(), 70, seed=5)


@pytest.fixture(scope="session")
def prediction(engine, logs):
    return engine.predict(logs)
