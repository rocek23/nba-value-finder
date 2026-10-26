"""Adapter The Odds API v4 (https://the-odds-api.com).

Verificato nelle pagine ufficiali:
- base https://api.the-odds-api.com/v4, sport key ``basketball_nba``, chiave in query (?apiKey=).
- Le props si interrogano per singola partita: /v4/sports/basketball_nba/events/{eventId}/odds.
- Costo = numero di mercati x numero di regioni (crediti). Piano gratuito: 500 crediti al mese.
- Chiavi mercato: player_points, player_rebounds, player_assists, player_threes,
  player_points_rebounds_assists; varianti linea alternativa con suffisso ``_alternate``.
- Le props NBA sono coperte per i bookmaker US/AU: nessun bookmaker italiano confermato per le props.
- Bookmaker italiani nell'elenco EU: codere_it, unibet_it (copertura delle props NBA: Da verificare).

Struttura del JSON degli outcome (name Over/Under, description = giocatore, point, price) e header di quota
(x-requests-remaining / x-requests-used / x-requests-last): NON presenti nelle pagine lette, da confermare
con scripts/capability_probe.py. Il parser e' difensivo e conta gli outcome scartati.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

import httpx

from ..convert import parse_decimal_odds
from ..http import ApiClient, ApiError
from ..models import ConnectionResult, Market, OddsQuote, Side, parse_iso, utc_now

BASE_URL = "https://api.the-odds-api.com"
PROVIDER = "the_odds_api"
SPORT_KEY = "basketball_nba"

MARKET_KEYS: dict[str, Market] = {
    "player_points": Market.POINTS,
    "player_rebounds": Market.REBOUNDS,
    "player_assists": Market.ASSISTS,
    "player_threes": Market.THREES,
    "player_points_rebounds_assists": Market.PRA,
}
DEFAULT_MARKETS = tuple(MARKET_KEYS)
ALTERNATE_SUFFIX = "_alternate"

ITALIAN_BOOKMAKERS = frozenset({"codere_it", "unibet_it"})


def is_italian_bookmaker(key: str) -> bool:
    return key in ITALIAN_BOOKMAKERS


def map_market_key(key: str | None) -> tuple[Market, bool] | None:
    """'player_points' -> (POINTS, False); 'player_points_alternate' -> (POINTS, True); altro -> None."""
    if not key:
        return None
    if key in MARKET_KEYS:
        return MARKET_KEYS[key], False
    if key.endswith(ALTERNATE_SUFFIX) and key[: -len(ALTERNATE_SUFFIX)] in MARKET_KEYS:
        return MARKET_KEYS[key[: -len(ALTERNATE_SUFFIX)]], True
    return None


def estimate_credits(markets: Iterable[str], regions: Iterable[str], include_alternates: bool = False) -> int:
    """Costo documentato per chiamata: mercati x regioni. Le alternative raddoppiano i mercati richiesti."""
    n_markets = len(list(markets)) * (2 if include_alternates else 1)
    return n_markets * len(list(regions))


@dataclass
class Quota:
    remaining: int | None = None
    used: int | None = None
    last_call_cost: int | None = None


def parse_quota(headers: Mapping[str, str]) -> Quota:
    def _num(name: str) -> int | None:
        try:
            return int(float(headers.get(name)))
        except (TypeError, ValueError):
            return None

    return Quota(_num("x-requests-remaining"), _num("x-requests-used"), _num("x-requests-last"))


@dataclass
class ParsedOdds:
    quotes: list[OddsQuote] = field(default_factory=list)
    skipped_markets: set[str] = field(default_factory=set)
    skipped_outcomes: int = 0


def parse_event_odds(payload: Mapping[str, Any], fetched_at: datetime) -> ParsedOdds:
    result = ParsedOdds()
    event_id = str(payload.get("id"))
    home, away = payload.get("home_team"), payload.get("away_team")
    commence = parse_iso(payload.get("commence_time"))

    for bookmaker in payload.get("bookmakers") or []:
        book_key = str(bookmaker.get("key"))
        book_time = parse_iso(bookmaker.get("last_update"))
        for market in bookmaker.get("markets") or []:
            mapped = map_market_key(market.get("key"))
            if mapped is None:
                result.skipped_markets.add(str(market.get("key")))
                continue
            market_enum, is_alt = mapped
            quote_time = parse_iso(market.get("last_update")) or book_time
            for outcome in market.get("outcomes") or []:
                side = {"over": Side.OVER, "under": Side.UNDER}.get(str(outcome.get("name")).strip().lower())
                player = outcome.get("description")
                point, price = outcome.get("point"), outcome.get("price")
                if side is None or not player or point is None or price is None:
                    result.skipped_outcomes += 1
                    continue
                try:
                    odds = parse_decimal_odds(price)
                    line = float(point)
                except ValueError:
                    result.skipped_outcomes += 1
                    continue
                result.quotes.append(
                    OddsQuote(
                        provider=PROVIDER,
                        bookmaker=book_key,
                        event_id=event_id,
                        player_name=str(player),
                        player_id=None,
                        market=market_enum,
                        line=line,
                        side=side,
                        odds_decimal=odds,
                        fetched_at=fetched_at,
                        quote_time=quote_time,
                        home_team=home,
                        away_team=away,
                        commence_time=commence,
                        is_alternate=is_alt,
                    )
                )
    return result


@dataclass
class OddsFetchResult:
    parsed: ParsedOdds
    quota: Quota
    estimated_cost: int
    raw: Any = None  # payload originale (non contiene la chiave API), utile per il report del probe


class TheOddsApiAdapter:
    provider = PROVIDER

    def __init__(
        self,
        api_key: str,
        *,
        regions: tuple[str, ...] = ("eu", "us"),
        timeout: float = 15.0,
        max_retries: int = 3,
        min_interval: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep=None,
    ) -> None:
        self.regions = regions
        kwargs = {} if sleep is None else {"sleep": sleep}
        self._client = ApiClient(
            BASE_URL,
            provider=PROVIDER,
            default_params={"apiKey": api_key},
            secrets=[api_key],
            timeout=timeout,
            max_retries=max_retries,
            min_interval=min_interval,
            transport=transport,
            **kwargs,
        )

    def close(self) -> None:
        self._client.close()

    def test_connection(self) -> ConnectionResult:
        """Chiama /v4/sports e controlla che basketball_nba sia presente.

        Se questa chiamata consumi crediti: Da verificare (confronta x-requests-remaining prima/dopo).
        """
        try:
            response = self._client.get("/v4/sports")
        except ApiError as exc:
            return ConnectionResult(PROVIDER, "failed", str(exc))
        sports = response.data if isinstance(response.data, list) else []
        keys = {s.get("key") for s in sports if isinstance(s, dict)}
        quota = parse_quota(response.headers)
        if SPORT_KEY not in keys:
            return ConnectionResult(
                PROVIDER,
                "failed",
                f"risposta ricevuta ma '{SPORT_KEY}' non e' elencato (fuori stagione o sport non disponibile)",
                {"quota_remaining": quota.remaining},
            )
        return ConnectionResult(
            PROVIDER,
            "ok",
            f"/sports ha risposto, {SPORT_KEY} presente. Crediti residui: "
            f"{quota.remaining if quota.remaining is not None else 'sconosciuti (header assente)'}",
            {"quota_remaining": quota.remaining, "quota_used": quota.used},
        )

    def list_events(self) -> list[dict[str, Any]]:
        """Partite NBA imminenti. Se consuma crediti: Da verificare."""
        data = self._client.get(f"/v4/sports/{SPORT_KEY}/events").data
        return data if isinstance(data, list) else []

    def get_event_props(
        self,
        event_id: str,
        markets: tuple[str, ...] = DEFAULT_MARKETS,
        regions: tuple[str, ...] | None = None,
        include_alternates: bool = False,
    ) -> OddsFetchResult:
        regions = regions or self.regions
        requested = list(markets)
        if include_alternates:
            requested += [m + ALTERNATE_SUFFIX for m in markets]
        response = self._client.get(
            f"/v4/sports/{SPORT_KEY}/events/{event_id}/odds",
            {"regions": ",".join(regions), "markets": ",".join(requested), "oddsFormat": "decimal"},
        )
        payload = response.data if isinstance(response.data, dict) else {}
        return OddsFetchResult(
            parsed=parse_event_odds(payload, utc_now()),
            quota=parse_quota(response.headers),
            estimated_cost=estimate_credits(markets, regions, include_alternates),
            raw=payload,
        )
