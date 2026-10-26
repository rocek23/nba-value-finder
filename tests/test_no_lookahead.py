"""Il modello non deve mai usare informazioni successive alla partita da prevedere."""

import dataclasses
from datetime import date

import numpy as np

from nba_vf.model.backtest import walk_forward
from nba_vf.model.config import ModelConfig
from nba_vf.model.engine import ValueEngine
from nba_vf.model.synthetic import TruePlayer, generate_player, generate_pool
from nba_vf.models import Market

CFG = ModelConfig(n_boot=0, n_sims_pra=8_000)


def test_future_games_cannot_change_past_predictions():
    pool = generate_pool(22, 80, seed=5)
    end = date(2026, 1, 31)
    a_lines, a_pit = walk_forward(pool, CFG, start_date=date(2026, 1, 5), end_date=end)
    corrupted = {
        pid: [
            dataclasses.replace(l, pts=l.pts * 3 + 11, reb=l.reb + 9, ast=l.ast * 2 + 5, fg3m=l.fg3a, minutes=48.0)
            if l.game_date > end else l
            for l in logs
        ]
        for pid, logs in pool.items()
    }
    b_lines, b_pit = walk_forward(corrupted, CFG, start_date=date(2026, 1, 5), end_date=end)
    assert len(a_lines) == len(b_lines) > 0
    assert np.array_equal(a_lines.p_over.to_numpy(), b_lines.p_over.to_numpy())
    assert np.array_equal(a_pit.pit.to_numpy(), b_pit.pit.to_numpy())


def test_prediction_uses_only_the_logs_it_is_given(engine):
    logs = generate_player(TruePlayer(), 60, 4)
    head = engine.predict(logs[:40], intervals=False)
    # aggiungere partite in coda (il "futuro") cambia la predizione: e' l'unico canale di informazione
    full = engine.predict(logs, intervals=False)
    assert not np.array_equal(head.market(Market.POINTS).pmf, full.market(Market.POINTS).pmf)
    # e l'ordine di input non conta (il motore ordina per data)
    shuffled = engine.predict(list(reversed(logs[:40])), intervals=False)
    assert np.array_equal(head.market(Market.POINTS).pmf, shuffled.market(Market.POINTS).pmf)


def test_synthetic_rows_are_labelled():
    assert {l.source for l in generate_player(TruePlayer(), 12, 1)} == {"SINTETICO"}
