"""Test The Odds API.

ATTENZIONE: FIXTURE SINTETICA. Segue la struttura che ASSUMIAMO per la risposta event-odds
(outcome con name Over/Under, description = giocatore, point, price). Non e' una risposta reale e non
dimostra nulla sulla copertura: serve solo a testare il parser. La struttura va confermata con
scripts/capability_probe.py odds-api --yes.
"""

from datetime import datetime, timezone

import httpx
import pytest

from nba_vf.adapters.the_odds_api import (
    TheOddsApiAdapter,
    estimate_credits,
    is_italian_bookmaker,
    map_market_key,
    parse_event_odds,
    parse_quota,
)
from nba_vf.models import Market, Side

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

SYNTHETIC_PAYLOAD = {
    "id": "evt-1",
    "sport_key": "basketball_nba",
    "commence_time": "2026-01-02T00:30:00Z",
    "home_team": "Home FC",
    "away_team": "Away FC",
    "bookmakers": [
        {
            "key": "bookie_us",
            "last_update": "2026-01-01T11:55:00Z",
            "markets": [
                {
                    "key": "player_points",
                    "last_update": "2026-01-01T11:58:00Z",
                    "outcomes": [
                        {"name": "Over", "description": "Giocatore Uno", "price": 1.90, "point": 24.5},
                        {"name": "Under", "description": "Giocatore Uno", "price": 1.91, "point": 24.5},
                    ],
                },
                {
                    "key": "player_points_alternate",
                    "outcomes": [{"name": "Over", "description": "Giocatore Uno", "price": 2.6, "point": 29.5}],
                },
                {"key": "player_steals", "outcomes": [{"name": "Over", "description": "X", "price": 2.0, "point": 1.5}]},
                {
                    "key": "player_threes",
                    "outcomes": [
                        {"name": "Over", "description": "Giocatore Uno", "price": 1.0, "point": 2.5},  # quota non valida
                        {"name": "Yes", "description": "Giocatore Uno", "price": 2.0, "point": 2.5},  # lato sconosciuto
                        {"name": "Over", "price": 2.0, "point": 2.5},  # senza giocatore
                    ],
                },
            ],
        },
        {
            "key": "unibet_it",
            "last_update": "2026-01-01T11:00:00Z",
            "markets": [
                {
                    "key": "player_points_rebounds_assists",
                    "outcomes": [{"name": "Over", "description": "Giocatore Due", "price": 1.8, "point": 38.5}],
                }
            ],
        },
    ],
}


def test_map_market_key():
    assert map_market_key("player_points") == (Market.POINTS, False)
    assert map_market_key("player_points_rebounds_assists_alternate") == (Market.PRA, True)
    assert map_market_key("player_steals") is None
    assert map_market_key(None) is None


def test_estimate_credits_matches_documented_formula():
    assert estimate_credits(["a", "b", "c"], ["us"]) == 3  # esempio ufficiale: 3 mercati x 1 regione
    assert estimate_credits(["a"] * 5, ["eu", "us"]) == 10
    assert estimate_credits(["a"] * 5, ["eu", "us"], include_alternates=True) == 20


def test_parse_event_odds_extracts_quotes_and_counts_rejects():
    parsed = parse_event_odds(SYNTHETIC_PAYLOAD, NOW)
    assert len(parsed.quotes) == 4  # 2 points + 1 alternate + 1 PRA
    assert parsed.skipped_outcomes == 3
    assert parsed.skipped_markets == {"player_steals"}

    main_over = next(q for q in parsed.quotes if q.market is Market.POINTS and q.side is Side.OVER and not q.is_alternate)
    assert main_over.player_name == "Giocatore Uno" and main_over.line == 24.5
    assert main_over.odds_decimal == 1.90 and main_over.home_team == "Home FC"
    assert main_over.quote_time == datetime(2026, 1, 1, 11, 58, tzinfo=timezone.utc)

    alt = next(q for q in parsed.quotes if q.is_alternate)
    assert alt.line == 29.5
    # senza last_update del mercato si usa quello del bookmaker
    assert alt.quote_time == datetime(2026, 1, 1, 11, 55, tzinfo=timezone.utc)


def test_italian_bookmaker_flag():
    assert is_italian_bookmaker("unibet_it") and is_italian_bookmaker("codere_it")
    assert not is_italian_bookmaker("draftkings")


def test_stale_quote_detection():
    parsed = parse_event_odds(SYNTHETIC_PAYLOAD, NOW)
    pra = next(q for q in parsed.quotes if q.market is Market.PRA)  # quote_time 11:00
    assert pra.is_stale(30, now=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    assert not pra.is_stale(90, now=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))


def test_parse_quota_headers_tolerates_missing():
    q = parse_quota({"x-requests-remaining": "490", "x-requests-used": "10", "x-requests-last": "10"})
    assert (q.remaining, q.used, q.last_call_cost) == (490, 10, 10)
    empty = parse_quota({})
    assert empty.remaining is None


def adapter_with(handler):
    return TheOddsApiAdapter(
        "SECRET-ODDS-KEY", transport=httpx.MockTransport(handler), sleep=lambda s: None, min_interval=0
    )


def test_get_event_props_builds_expected_request():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=SYNTHETIC_PAYLOAD, headers={"x-requests-remaining": "490", "x-requests-last": "10"})

    result = adapter_with(handler).get_event_props("evt-1")
    assert seen["path"] == "/v4/sports/basketball_nba/events/evt-1/odds"
    assert seen["params"]["regions"] == "eu,us"
    assert seen["params"]["oddsFormat"] == "decimal"
    assert seen["params"]["markets"].split(",") == [
        "player_points", "player_rebounds", "player_assists", "player_threes", "player_points_rebounds_assists",
    ]
    assert seen["params"]["apiKey"] == "SECRET-ODDS-KEY"
    assert result.estimated_cost == 10 and result.quota.remaining == 490
    assert len(result.parsed.quotes) == 4


def test_get_event_props_with_alternates_requests_alternate_markets():
    seen = {}

    def handler(request):
        seen["markets"] = request.url.params["markets"]
        return httpx.Response(200, json={"id": "e", "bookmakers": []})

    result = adapter_with(handler).get_event_props("e", include_alternates=True)
    assert "player_points_alternate" in seen["markets"].split(",")
    assert result.estimated_cost == 20 and result.parsed.quotes == []


def test_connection_ok_requires_nba_in_sports_list():
    ok = adapter_with(lambda r: httpx.Response(200, json=[{"key": "basketball_nba"}], headers={"x-requests-remaining": "499"}))
    result = ok.test_connection()
    assert result.ok and "499" in result.detail

    missing = adapter_with(lambda r: httpx.Response(200, json=[{"key": "soccer_epl"}]))
    assert missing.test_connection().status == "failed"


def test_connection_errors_never_leak_key():
    result = adapter_with(lambda r: httpx.Response(401)).test_connection()
    assert result.status == "failed" and "SECRET-ODDS-KEY" not in result.detail
