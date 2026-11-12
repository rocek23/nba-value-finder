import pandas as pd

from nba_vf.ui.lines import near_main


def _rows(lines_odds):
    return pd.DataFrame([dict(gday="d", key="k", mercato="Punti", linea=l, quota=q, lato="Over") for l, q in lines_odds])


def test_near_main_keeps_main_and_adjacent_lines_only():
    rows = _rows([(10.5, 1.05), (14.5, 1.20), (18.5, 1.45), (22.5, 1.85), (24.5, 2.40), (28.5, 4.0)])
    keep = near_main(rows).tolist()
    # principale = quota 1.85 (22.5), adiacenti 18.5 e 24.5; le linee molto basse escono
    assert keep == [False, False, True, True, True, False]


def test_near_main_independent_per_player_and_keeps_rows_without_over():
    a = _rows([(5.5, 1.80)])
    b = pd.DataFrame([dict(gday="d", key="k2", mercato="Punti", linea=20.5, quota=1.9, lato="Under")])
    assert near_main(pd.concat([a, b], ignore_index=True)).tolist() == [True, True]
    assert near_main(pd.DataFrame()).tolist() == []
