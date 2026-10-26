"""Modelli dati condivisi da tutti gli adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(value: str | None) -> datetime | None:
    """Parsa un timestamp ISO-8601 (anche con 'Z'). Restituisce sempre un datetime aware (UTC se naive)."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class Market(str, Enum):
    POINTS = "points"
    REBOUNDS = "rebounds"
    ASSISTS = "assists"
    THREES = "threes"
    PRA = "pra"


class Side(str, Enum):
    OVER = "over"
    UNDER = "under"


@dataclass(frozen=True)
class OddsQuote:
    """Una quota (un solo lato, Over oppure Under) per un giocatore, in formato decimale."""

    provider: str
    bookmaker: str
    event_id: str
    player_name: str | None
    player_id: int | None
    market: Market
    line: float
    side: Side
    odds_decimal: float
    fetched_at: datetime
    quote_time: datetime | None = None
    home_team: str | None = None
    away_team: str | None = None
    commence_time: datetime | None = None
    # None = la fonte non distingue linea principale e alternativa
    is_alternate: bool | None = None
    source: str = "api"

    @property
    def reference_time(self) -> datetime:
        """Istante a cui la quota si riferisce: quello dichiarato dalla fonte, altrimenti il fetch."""
        return self.quote_time or self.fetched_at

    def age_minutes(self, now: datetime | None = None) -> float:
        now = now or utc_now()
        return (now - self.reference_time).total_seconds() / 60.0

    def is_stale(self, max_age_minutes: float, now: datetime | None = None) -> bool:
        return self.age_minutes(now) > max_age_minutes


@dataclass(frozen=True)
class PlayerGameLog:
    """Una riga di statistiche di un giocatore in una partita."""

    source: str
    player_id: int
    player_name: str | None
    game_id: str
    game_date: date
    season: int  # anno di inizio stagione: 2025 = stagione 2025-26
    team: str | None
    opponent: str | None
    is_home: bool | None
    minutes: float
    pts: int
    reb: int
    ast: int
    fg3m: int
    fg3a: int
    fgm: int
    fga: int
    ftm: int = 0
    fta: int = 0

    @property
    def pra(self) -> int:
        return self.pts + self.reb + self.ast


@dataclass
class ConnectionResult:
    provider: str
    status: str  # "ok" | "failed" | "not_configured"
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"
