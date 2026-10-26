"""Test BALLDONTLIE con risposte simulate (MockTransport).

La riga di props e' copiata dall'esempio del tutorial ufficiale BALLDONTLIE; le righe di stats seguono lo
schema della reference ufficiale. Sono fixture di test, non dati reali di partite.
"""

from datetime import date

import httpx
import pytest

from nba_vf.adapters.balldontlie import BalldontlieAdapter, parse_minutes, parse_prop_row, parse_stat_row
from nba_vf.http import TierError
from nba_vf.models import Market, Side, utc_now

PROP_ROW = {
    "id": 1256660847,
    "game_id": 18447232,
    "player_id": 322,
    "vendor": "draftkings",
    "prop_type": "points",
    "line_value": "29.5",
    "market": {"type": "over_under", "over_odds": -115, "under_odds": -105},
    "updated_at": "2025-12-24T19:30:15.903Z",
}

STAT_ROW = {
    "min": "30",
    "fgm": 7, "fga": 18, "fg3m": 5, "fg3a": 9, "ftm": 4, "fta": 4,
    "reb": 7, "ast": 1, "pts": 23,
    "player": {"id": 70, "first_name": "Test", "last_name": "Player"},
    "team": {"id": 2, "abbreviation": "BOS"},
    "game": {"id": 99, "date": "2024-10-22", "season": 2024, "home_team_id": 2, "visitor_team_id": 20},
}


def adapter_with(handler, **kwargs):
    return BalldontlieAdapter(
        "SECRET-BDL-KEY", transport=httpx.MockTransport(handler), sleep=lambda s: None, requests_per_min=600, **kwargs
    )


def test_parse_minutes():
    assert parse_minutes("30") == 30.0
    assert parse_minutes("30:30") == pytest.approx(30.5)
    assert parse_minutes(None) == 0.0 and parse_minutes("") == 0.0


def test_parse_stat_row_home_game_and_opponent():
    log = parse_stat_row(STAT_ROW, {20: "NYK", 2: "BOS"})
    assert log.game_date == date(2024, 10, 22)
    assert log.is_home is True and log.opponent == "NYK" and log.team == "BOS"
    assert (log.pts, log.reb, log.ast, log.pra) == (23, 7, 1, 31)
    assert (log.fg3m, log.fg3a, log.minutes) == (5, 9, 30.0)


def test_parse_stat_row_missing_game_returns_none():
    assert parse_stat_row({"player": {"id": 1}, "game": {}}) is None


def test_parse_prop_row_converts_american_to_decimal():
    quotes, reason = parse_prop_row(PROP_ROW, utc_now())
    assert reason is None and len(quotes) == 2
    over = next(q for q in quotes if q.side is Side.OVER)
    under = next(q for q in quotes if q.side is Side.UNDER)
    assert over.market is Market.POINTS and over.line == 29.5 and over.bookmaker == "draftkings"
    assert over.odds_decimal == pytest.approx(1 + 100 / 115)
    assert under.odds_decimal == pytest.approx(1 + 100 / 105)
    assert over.quote_time is not None and over.is_alternate is None


def test_parse_prop_row_skips_milestone_and_unknown_types():
    milestone = {**PROP_ROW, "market": {"type": "milestone", "over_odds": 300}}
    assert parse_prop_row(milestone, utc_now()) == ([], "milestone")
    steals = {**PROP_ROW, "prop_type": "steals"}
    assert parse_prop_row(steals, utc_now()) == ([], "unsupported_type")
    assert parse_prop_row({**PROP_ROW, "line_value": "abc"}, utc_now())[1] == "invalid"


def test_pra_prop_type_is_mapped():
    row = {**PROP_ROW, "prop_type": "points_rebounds_assists"}
    quotes, _ = parse_prop_row(row, utc_now())
    assert quotes[0].market is Market.PRA


def test_connection_ok_and_authorization_header_sent():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"id": 1, "abbreviation": "ATL"}]})

    result = adapter_with(handler).test_connection()
    assert result.ok and seen["auth"] == "SECRET-BDL-KEY"
    assert "SECRET-BDL-KEY" not in result.detail


def test_connection_failure_on_401_does_not_leak_key():
    result = adapter_with(lambda r: httpx.Response(401)).test_connection()
    assert result.status == "failed" and "SECRET-BDL-KEY" not in result.detail


def test_prefix_fallback_when_v1_returns_404():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/v1/teams":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": 1, "abbreviation": "ATL"}]})

    result = adapter_with(handler).test_connection()
    assert result.ok and paths == ["/v1/teams", "/nba/v1/teams"]
    assert result.extra["prefix"] == "/nba/v1"


def test_game_logs_follow_cursor_pagination_and_sort():
    later = {**STAT_ROW, "game": {**STAT_ROW["game"], "id": 100, "date": "2024-10-25"}}

    def handler(request):
        if request.url.path.endswith("/teams"):
            return httpx.Response(200, json={"data": [{"id": 20, "abbreviation": "NYK"}, {"id": 2, "abbreviation": "BOS"}]})
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(200, json={"data": [later], "meta": {"next_cursor": 5}})
        assert cursor == "5"
        return httpx.Response(200, json={"data": [STAT_ROW], "meta": {}})

    logs = adapter_with(handler).get_player_game_logs(70, 2024)
    assert [l.game_date.isoformat() for l in logs] == ["2024-10-22", "2024-10-25"]


def test_game_logs_on_free_plan_raise_tier_error():
    def handler(request):
        if request.url.path.endswith("/teams"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(401)

    with pytest.raises(TierError):
        adapter_with(handler).get_player_game_logs(70, 2024)


def test_player_props_counts_skipped_rows():
    milestone = {**PROP_ROW, "market": {"type": "milestone", "over_odds": 300}}
    steals = {**PROP_ROW, "prop_type": "steals"}

    def handler(request):
        assert request.url.path == "/nba/v2/odds/player_props"
        assert request.url.params["game_id"] == "18447232"
        return httpx.Response(200, json={"data": [PROP_ROW, milestone, steals]})

    result = adapter_with(handler).get_player_props(18447232)
    assert len(result.quotes) == 2 and result.raw_count == 3
    assert result.skipped_milestone == 1 and result.skipped_unsupported_type == 1
