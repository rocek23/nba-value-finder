from datetime import date

import numpy as np
import pytest

from nba_vf.model.backtest import (
    LogisticCalibrator,
    calibration_report,
    reliability_table,
    synthetic_half_lines,
    walk_forward,
)
from nba_vf.model.config import ModelConfig
from nba_vf.model.synthetic import generate_pool

CFG = ModelConfig(n_boot=0, n_sims_pra=8_000)


@pytest.fixture(scope="module")
def wf():
    pool = generate_pool(25, 80, seed=3)
    lines, pits = walk_forward(pool, CFG, start_date=date(2026, 1, 5), end_date=date(2026, 2, 28))
    return pool, lines, pits


def test_walk_forward_outputs_valid(wf):
    _, lines, pits = wf
    assert len(lines) > 1000 and len(pits) > 300
    assert lines.p_over.between(0, 1).all() and pits.pit.between(0, 1).all()
    assert (lines.linea * 2 % 2 == 1).all()  # soltanto mezze linee: esito binario, nessun push
    assert set(lines.esito_over.unique()) <= {0, 1}
    assert set(pits.mercato) == {"points", "rebounds", "assists", "threes", "pra"}


def test_calibration_on_well_specified_synthetic_world(wf):
    _, lines, pits = wf
    rep = calibration_report(lines, pits)
    for mk, r in rep.items():
        assert r["brier"] < 0.25, mk  # meglio di una moneta
        assert abs(r["p_media_predetta"] - r["freq_osservata"]) < 0.05, mk
        assert 0.70 < r["copertura_intervallo_80"] < 0.88, mk
        assert 0.42 < r["pit_media"] < 0.58, mk
        assert r["affidabilita"].n.sum() == r["n_linee"]


def test_synthetic_lines_helper():
    pmf = np.zeros(40)
    pmf[8:14] = 1 / 6
    lines = synthetic_half_lines(pmf)
    assert all(ln % 1 == 0.5 and 0.5 <= ln < 38 for ln in lines) and lines == sorted(lines)
    assert any(8 < ln < 14 for ln in lines)


def test_reliability_table_counts():
    p = np.array([0.05, 0.15, 0.15, 0.95])
    y = np.array([0, 1, 0, 1])
    t = reliability_table(p, y, 10)
    assert t.n.sum() == 4
    assert t.loc[t.bin == "0.1-0.2", "freq_osservata"].iloc[0] == 0.5


def test_calibrator_identity_on_calibrated_data_and_corrects_distortion():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.random(20_000) < p).astype(float)
    cal = LogisticCalibrator().fit(p, y)
    assert cal.a == pytest.approx(0.0, abs=0.08) and cal.b == pytest.approx(1.0, abs=0.08)
    # modello sovra-confidente: probabilita' vere piu' vicine a 0.5 di quelle dichiarate
    z = np.log(p / (1 - p))
    q_true = 1 / (1 + np.exp(-0.6 * z))
    y2 = (rng.random(20_000) < q_true).astype(float)
    cal2 = LogisticCalibrator().fit(p, y2)
    assert cal2.b == pytest.approx(0.6, abs=0.08)
    brier = lambda q: np.mean((q - y2) ** 2)
    assert brier(cal2.predict(p)) < brier(p)
    out = cal2.predict(p)
    assert (out > 0).all() and (out < 1).all()


def test_calibrator_ridge_keeps_identity_with_tiny_sample():
    cal = LogisticCalibrator(ridge=5.0).fit(np.array([0.3, 0.7]), np.array([1.0, 0.0]))
    assert abs(cal.b - 1.0) < 1.0 and abs(cal.a) < 1.0  # ridge limita la deriva dall'identita'
