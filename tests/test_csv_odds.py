from datetime import datetime, timedelta, timezone

import pytest

from nba_vf.adapters.csv_odds import import_odds_csv
from nba_vf.models import Market, Side

HEADER = "bookmaker,event_date,home_team,away_team,player,market,line,side,odds,odds_format,timestamp,is_alternate\n"


def write(tmp_path, body, header=HEADER, name="odds.csv"):
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return path


def test_valid_rows_import(tmp_path):
    path = write(
        tmp_path,
        "snai,2026-01-01,Home,Away,Giocatore Uno,points,24.5,over,1.85,decimal,2026-01-01T12:00:00+01:00,false\n"
        "snai,2026-01-01,Home,Away,Giocatore Uno,points,24.5,under,-110,american,2026-01-01T12:00:00+01:00,\n",
    )
    result = import_odds_csv(path)
    assert result.errors == [] and len(result.quotes) == 2
    over, under = result.quotes
    assert over.side is Side.OVER and over.market is Market.POINTS and over.odds_decimal == pytest.approx(1.85)
    assert under.odds_decimal == pytest.approx(1 + 100 / 110)
    assert over.is_alternate is False and under.is_alternate is None
    assert over.source == "csv" and over.bookmaker == "snai"
    assert over.event_id == "csv:2026-01-01:home:away"
    assert over.quote_time == datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)


def test_semicolon_delimiter_italian_aliases_and_decimal_comma(tmp_path):
    header = HEADER.replace(",", ";")
    path = write(
        tmp_path,
        "Sisal;2026-01-01;Home;Away;Giocatore Due;rimbalzi;9,5;o;1,90;;2026-01-01T12:00:00+01:00;\n",
        header=header,
    )
    result = import_odds_csv(path)
    assert result.errors == []
    q = result.quotes[0]
    assert q.market is Market.REBOUNDS and q.line == 9.5 and q.odds_decimal == pytest.approx(1.9)


def test_naive_timestamp_is_interpreted_as_rome_time(tmp_path):
    path = write(tmp_path, "snai,2026-07-01,H,A,P,points,20.5,over,1.8,,2026-07-01T12:00:00,\n")
    result = import_odds_csv(path)
    assert result.naive_timestamps == 1
    assert result.quotes[0].quote_time == datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)  # CEST = UTC+2


@pytest.mark.parametrize(
    "row, fragment",
    [
        ("snai,2026-01-01,H,A,P,steals,1.5,over,1.8,,2026-01-01T12:00:00+01:00,", "mercato"),
        ("snai,2026-01-01,H,A,P,points,20.5,boh,1.8,,2026-01-01T12:00:00+01:00,", "lato"),
        ("snai,2026-01-01,H,A,P,points,20.5,over,1.0,,2026-01-01T12:00:00+01:00,", "quota"),
        ("snai,2026-01-01,H,A,P,points,20.5,over,1.8,,", "timestamp"),
        ("snai,01/01/2026,H,A,P,points,20.5,over,1.8,,2026-01-01T12:00:00+01:00,", "event_date"),
        ("snai,2026-01-01,H,A,P,points,abc,over,1.8,,2026-01-01T12:00:00+01:00,", "line"),
        ("snai,2026-01-01,H,A,P,points,20.5,over,1.8,,ieri alle 12,", "timestamp"),
    ],
)
def test_invalid_rows_are_reported_not_imported(tmp_path, row, fragment):
    good = "snai,2026-01-01,H,A,P,points,20.5,over,1.8,,2026-01-01T12:00:00+01:00,\n"
    result = import_odds_csv(write(tmp_path, good + row + "\n"))
    assert len(result.quotes) == 1
    assert len(result.errors) == 1
    assert result.errors[0].line == 3
    assert fragment in result.errors[0].message


def test_missing_required_columns(tmp_path):
    path = write(tmp_path, "x,y\n", header="bookmaker,player\n")
    result = import_odds_csv(path)
    assert result.quotes == [] and "colonne mancanti" in result.errors[0].message


def test_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert import_odds_csv(path).errors[0].message == "file vuoto"


def test_csv_quote_staleness_uses_row_timestamp(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    path = write(tmp_path, f"snai,2026-01-01,H,A,P,points,20.5,over,1.8,,{old},\n")
    quote = import_odds_csv(path).quotes[0]
    assert quote.is_stale(30) and not quote.is_stale(600)


def test_shipped_template_is_importable():
    from pathlib import Path

    template = Path(__file__).resolve().parent.parent / "data" / "templates" / "odds_template.csv"
    result = import_odds_csv(template)
    assert result.errors == [] and len(result.quotes) == 2
