"""Importazione manuale di quote da CSV, per i bookmaker non disponibili via API.

Colonne obbligatorie: bookmaker, event_date, home_team, away_team, player, market, line, side, odds, timestamp
Colonne opzionali:    odds_format (decimal|american, default decimal), is_alternate (true/false),
                      commence_time (ISO-8601, orario d'inizio della partita: NECESSARIO per usare la quota nel backtest)

- Separatore ',' oppure ';' (rilevato dall'intestazione); la virgola decimale italiana ("1,85") e' accettata.
- ``timestamp`` e' obbligatorio: indica quando hai letto la quota. Senza di esso non si puo' sapere se e'
  obsoleta. Senza fuso orario viene interpretato come Europe/Rome.
- Le righe non valide non vengono importate: sono restituite con numero di riga e motivo.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..convert import american_to_decimal, parse_decimal_odds
from ..models import Market, OddsQuote, Side, utc_now

PROVIDER = "csv"
LOCAL_TZ = ZoneInfo("Europe/Rome")

REQUIRED_COLUMNS = (
    "bookmaker",
    "event_date",
    "home_team",
    "away_team",
    "player",
    "market",
    "line",
    "side",
    "odds",
    "timestamp",
)

MARKET_ALIASES = {
    "points": Market.POINTS,
    "punti": Market.POINTS,
    "rebounds": Market.REBOUNDS,
    "rimbalzi": Market.REBOUNDS,
    "assists": Market.ASSISTS,
    "assist": Market.ASSISTS,
    "threes": Market.THREES,
    "triple": Market.THREES,
    "pra": Market.PRA,
    "points_rebounds_assists": Market.PRA,
}
SIDE_ALIASES = {"over": Side.OVER, "o": Side.OVER, "under": Side.UNDER, "u": Side.UNDER}
TRUE_VALUES = {"1", "true", "yes", "si", "sì", "y"}


@dataclass(frozen=True)
class RowError:
    line: int
    message: str


@dataclass
class CsvImportResult:
    quotes: list[OddsQuote] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)
    naive_timestamps: int = 0


def _parse_timestamp(text: str) -> tuple[datetime, bool]:
    """Restituisce (datetime aware, era_naive)."""
    cleaned = text.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=LOCAL_TZ), True
    return parsed, False


def _event_id(event_date: date, home: str, away: str) -> str:
    return f"csv:{event_date.isoformat()}:{home.strip().lower()}:{away.strip().lower()}"


def import_odds_csv(path: str | Path) -> CsvImportResult:
    result = CsvImportResult()
    fetched_at = utc_now()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        header_line = handle.readline()
        if not header_line.strip():
            result.errors.append(RowError(1, "file vuoto"))
            return result
        delimiter = ";" if header_line.count(";") > header_line.count(",") else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = [c.strip().lower() for c in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            result.errors.append(RowError(1, f"colonne mancanti: {', '.join(missing)}"))
            return result

        for row in reader:
            line_no = reader.line_num
            data = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            try:
                quote, naive = _parse_row(data, fetched_at)
            except ValueError as exc:
                result.errors.append(RowError(line_no, str(exc)))
                continue
            result.quotes.append(quote)
            result.naive_timestamps += int(naive)
    return result


def _parse_row(data: dict[str, str], fetched_at: datetime) -> tuple[OddsQuote, bool]:
    for column in REQUIRED_COLUMNS:
        if not data.get(column):
            raise ValueError(f"campo '{column}' vuoto")

    market = MARKET_ALIASES.get(data["market"].lower())
    if market is None:
        raise ValueError(f"mercato non riconosciuto: {data['market']!r}")
    side = SIDE_ALIASES.get(data["side"].lower())
    if side is None:
        raise ValueError(f"lato non riconosciuto (usa over/under): {data['side']!r}")

    try:
        event_date = date.fromisoformat(data["event_date"])
    except ValueError:
        raise ValueError(f"event_date non valida (formato AAAA-MM-GG): {data['event_date']!r}") from None
    try:
        line = float(data["line"].replace(",", "."))
    except ValueError:
        raise ValueError(f"line non valida: {data['line']!r}") from None
    if line < 0:
        raise ValueError("line deve essere >= 0")

    fmt = (data.get("odds_format") or "decimal").lower()
    try:
        if fmt == "decimal":
            odds = parse_decimal_odds(data["odds"])
        elif fmt == "american":
            odds = american_to_decimal(float(data["odds"].replace(",", ".")))
        else:
            raise ValueError(f"odds_format non riconosciuto: {fmt!r}")
    except ValueError as exc:
        raise ValueError(f"quota non valida: {exc}") from None

    try:
        quote_time, naive = _parse_timestamp(data["timestamp"])
    except ValueError:
        raise ValueError(f"timestamp non valido (ISO-8601): {data['timestamp']!r}") from None

    commence = None
    if data.get("commence_time"):
        try:
            commence, _ = _parse_timestamp(data["commence_time"])
        except ValueError:
            raise ValueError(f"commence_time non valido (ISO-8601): {data['commence_time']!r}") from None

    is_alt = data.get("is_alternate", "").lower() in TRUE_VALUES if data.get("is_alternate") else None
    quote = OddsQuote(
        provider=PROVIDER,
        bookmaker=data["bookmaker"].lower(),
        event_id=_event_id(event_date, data["home_team"], data["away_team"]),
        player_name=data["player"],
        player_id=None,
        market=market,
        line=line,
        side=side,
        odds_decimal=odds,
        fetched_at=fetched_at,
        quote_time=quote_time,
        home_team=data["home_team"],
        away_team=data["away_team"],
        commence_time=commence,
        is_alternate=is_alt,
        source="csv",
    )
    return quote, naive
