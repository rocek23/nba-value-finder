"""Test di nba_api senza rete: solo funzioni pure e DataFrame sintetico (non dati reali)."""

from datetime import date

import pandas as pd
import pytest

from nba_vf.adapters.nba_api_adapter import (
    NbaApiAdapter,
    current_season_start_year,
    parse_game_date,
    parse_game_log_frame,
    parse_matchup,
    season_string,
)


@pytest.mark.parametrize("year, text", [(2025, "2025-26"), (2026, "2026-27"), (1999, "1999-00"), (2009, "2009-10")])
def test_season_string(year, text):
    assert season_string(year) == text


@pytest.mark.parametrize(
    "today, expected",
    [
        (date(2026, 10, 26), 2026),  # a ottobre la nuova stagione e' iniziata
        (date(2026, 12, 31), 2026),
        (date(2027, 1, 15), 2026),
        (date(2027, 6, 20), 2026),  # finali: ancora stagione 2026-27
        (date(2027, 9, 30), 2026),  # offseason: ultima stagione conclusa
    ],
)
def test_current_season_start_year(today, expected):
    assert current_season_start_year(today) == expected


def test_parse_matchup():
    assert parse_matchup("BOS vs. NYK") == ("BOS", "NYK", True)
    assert parse_matchup("BOS @ NYK") == ("BOS", "NYK", False)
    assert parse_matchup("garbage") == (None, None, None)


def test_parse_game_date_formats():
    assert parse_game_date("OCT 22, 2025") == date(2025, 10, 22)
    assert parse_game_date("2025-10-22T00:00:00") == date(2025, 10, 22)
    assert parse_game_date("2025-10-22") == date(2025, 10, 22)
    with pytest.raises(ValueError):
        parse_game_date("ieri")


def test_parse_game_log_frame_sorts_and_maps_columns():
    frame = pd.DataFrame(
        [
            {"Game_ID": "2", "GAME_DATE": "OCT 25, 2025", "MATCHUP": "BOS @ MIA", "MIN": 35, "PTS": 30, "REB": 8,
             "AST": 5, "FG3M": 4, "FG3A": 10, "FGM": 10, "FGA": 20, "FTM": 6, "FTA": 7},
            {"Game_ID": "1", "GAME_DATE": "OCT 22, 2025", "MATCHUP": "BOS vs. NYK", "MIN": 30, "PTS": 20, "REB": 6,
             "AST": 4, "FG3M": 2, "FG3A": 7, "FGM": 7, "FGA": 15, "FTM": 4, "FTA": 4},
        ]
    )
    logs = parse_game_log_frame(frame, 2025, 123)
    assert [l.game_id for l in logs] == ["1", "2"]
    assert logs[0].is_home is True and logs[0].opponent == "NYK"
    assert logs[1].is_home is False and logs[1].opponent == "MIA"
    assert logs[1].pra == 43 and logs[1].season == 2025 and logs[0].fg3a == 7


def test_find_player_is_offline_and_returns_known_player():
    found = NbaApiAdapter.find_player("LeBron James")
    assert found and found[0]["full_name"] == "LeBron James"
