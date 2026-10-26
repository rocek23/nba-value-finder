"""Adapter BALLDONTLIE (https://docs.balldontlie.io).

Verificato nella documentazione ufficiale:
- header di autenticazione: ``Authorization: <API_KEY>``
- /v1/teams, /v1/players, /v1/games: piano Free; /v1/stats e /v1/player_injuries: da ALL-STAR;
  player props: solo GOAT. Paginazione a cursore (per_page max 100).
- 401 = chiave mancante OPPURE piano senza accesso all'endpoint.
- Limiti di richieste al minuto: Free 5, ALL-STAR 60, GOAT 600.

Basato solo sul tutorial ufficiale (non sulla reference): percorso e forma delle player props
(/nba/v2/odds/player_props, parametri game_id e prop_type). Da confermare con scripts/capability_probe.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Sequence

import httpx

from ..convert import american_to_decimal
from ..http import ApiClient, ApiError, TierError
from ..models import (
    ConnectionResult,
    Market,
    OddsQuote,
    PlayerGameLog,
    Side,
    parse_iso,
    utc_now,
)

BASE_URL = "https://api.balldontlie.io"
PROVIDER = "balldontlie"

# La documentazione ufficiale mostra /v1/...; il tutorial ufficiale usa /nba/v1/...
V1_PREFIXES = ("/v1", "/nba/v1")
PROPS_PATH = "/nba/v2/odds/player_props"

PROP_TYPE_TO_MARKET = {
    "points": Market.POINTS,
    "rebounds": Market.REBOUNDS,
    "assists": Market.ASSISTS,
    "threes": Market.THREES,
    "points_rebounds_assists": Market.PRA,
}


def parse_minutes(value: Any) -> float:
    """'30' -> 30.0, '30:12' -> 30.2, None/'' -> 0.0."""
    if value is None or value == "":
        return 0.0
    text = str(value).strip()
    if ":" in text:
        minutes, _, seconds = text.partition(":")
        try:
            return int(minutes) + int(seconds) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_stat_row(row: dict[str, Any], teams_by_id: dict[int, str] | None = None) -> PlayerGameLog | None:
    """Converte una riga di /stats in PlayerGameLog. None se mancano i campi essenziali."""
    player = row.get("player") or {}
    team = row.get("team") or {}
    game = row.get("game") or {}
    try:
        game_date = date.fromisoformat(str(game["date"])[:10])
        player_id = int(player["id"])
        season = int(game["season"])
    except (KeyError, TypeError, ValueError):
        return None

    team_id = team.get("id")
    home_id, visitor_id = game.get("home_team_id"), game.get("visitor_team_id")
    is_home: bool | None = None
    opponent_id = None
    if team_id is not None and home_id is not None and visitor_id is not None:
        is_home = team_id == home_id
        opponent_id = visitor_id if is_home else home_id
    opponent = (teams_by_id or {}).get(opponent_id) if opponent_id is not None else None

    status = str(game.get("status") or "").strip()
    if status and not status.lower().startswith("final"):
        return None  # partita non terminata: le statistiche sarebbero parziali e non vanno usate come storico
    name = " ".join(p for p in (player.get("first_name"), player.get("last_name")) if p) or None
    return PlayerGameLog(
        source=PROVIDER,
        player_id=player_id,
        player_name=name,
        game_id=str(game.get("id")),
        game_date=game_date,
        season=season,
        team=team.get("abbreviation"),
        opponent=opponent or (str(opponent_id) if opponent_id is not None else None),
        is_home=is_home,
        minutes=parse_minutes(row.get("min")),
        pts=_int(row.get("pts")),
        reb=_int(row.get("reb")),
        ast=_int(row.get("ast")),
        fg3m=_int(row.get("fg3m")),
        fg3a=_int(row.get("fg3a")),
        fgm=_int(row.get("fgm")),
        fga=_int(row.get("fga")),
        ftm=_int(row.get("ftm")),
        fta=_int(row.get("fta")),
    )


@dataclass
class PropsResult:
    quotes: list[OddsQuote] = field(default_factory=list)
    skipped_unsupported_type: int = 0
    skipped_milestone: int = 0
    skipped_invalid: int = 0
    raw_count: int = 0


def parse_prop_row(row: dict[str, Any], fetched_at: datetime) -> tuple[list[OddsQuote], str | None]:
    """Restituisce (quote, motivo_scarto). Le quote americane vengono convertite in decimali."""
    market = PROP_TYPE_TO_MARKET.get(str(row.get("prop_type")))
    if market is None:
        return [], "unsupported_type"
    market_obj = row.get("market") or {}
    if market_obj.get("type") != "over_under":
        return [], "milestone"
    try:
        line = float(row["line_value"])
        player_id = int(row["player_id"])
    except (KeyError, TypeError, ValueError):
        return [], "invalid"

    quotes: list[OddsQuote] = []
    for side, key in ((Side.OVER, "over_odds"), (Side.UNDER, "under_odds")):
        raw = market_obj.get(key)
        if raw is None:
            continue
        try:
            decimal = american_to_decimal(raw)
        except ValueError:
            continue
        quotes.append(
            OddsQuote(
                provider=PROVIDER,
                bookmaker=str(row.get("vendor")),
                event_id=str(row.get("game_id")),
                player_name=None,
                player_id=player_id,
                market=market,
                line=line,
                side=side,
                odds_decimal=decimal,
                fetched_at=fetched_at,
                quote_time=parse_iso(row.get("updated_at")),
                is_alternate=None,
            )
        )
    return (quotes, None) if quotes else ([], "invalid")


class BalldontlieAdapter:
    provider = PROVIDER

    def __init__(
        self,
        api_key: str,
        *,
        requests_per_min: int = 60,
        timeout: float = 15.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep=None,
    ) -> None:
        kwargs = {} if sleep is None else {"sleep": sleep}
        self._client = ApiClient(
            BASE_URL,
            provider=PROVIDER,
            headers={"Authorization": api_key},
            secrets=[api_key],
            timeout=timeout,
            max_retries=max_retries,
            min_interval=60.0 / max(1, requests_per_min),
            transport=transport,
            **kwargs,
        )
        self._prefix_index = 0
        self._teams_cache: dict[int, str] | None = None

    def close(self) -> None:
        self._client.close()

    @property
    def request_count(self) -> int:
        return self._client.request_count

    # -- accesso agli endpoint v1 con fallback sul prefisso ------------------------------------
    def _get_v1(self, endpoint: str, params: dict[str, Any] | None = None):
        last_404: ApiError | None = None
        for offset in range(len(V1_PREFIXES)):
            index = (self._prefix_index + offset) % len(V1_PREFIXES)
            try:
                response = self._client.get(f"{V1_PREFIXES[index]}/{endpoint}", params)
            except ApiError as exc:
                if exc.status == 404:
                    last_404 = exc
                    continue
                raise
            self._prefix_index = index
            return response
        assert last_404 is not None
        raise last_404

    # -- API pubbliche ----------------------------------------------------------------------------
    def test_connection(self) -> ConnectionResult:
        """Chiama /teams (piano Free). Verifica rete e validita' della chiave, non il piano."""
        try:
            response = self._get_v1("teams", {"per_page": 1})
        except TierError as exc:
            return ConnectionResult(PROVIDER, "failed", str(exc))
        except ApiError as exc:
            return ConnectionResult(PROVIDER, "failed", str(exc))
        count = len(response.data.get("data", [])) if isinstance(response.data, dict) else 0
        return ConnectionResult(
            PROVIDER,
            "ok",
            f"/teams ha risposto ({count} record). Il piano non e' verificato da questo test: "
            "usa capability_probe.py per stats e props.",
            {"prefix": V1_PREFIXES[self._prefix_index]},
        )

    def get_teams(self) -> dict[int, str]:
        if self._teams_cache is None:
            data = self._get_v1("teams").data.get("data", [])
            self._teams_cache = {int(t["id"]): t.get("abbreviation") for t in data if "id" in t}
        return self._teams_cache

    def get_games(self, day: date | str) -> list[dict[str, Any]]:
        value = day.isoformat() if isinstance(day, date) else str(day)
        return self._paginate("games", {"dates[]": [value], "per_page": 100})

    def get_player_game_logs(
        self,
        player_id: int,
        season_start_year: int | Sequence[int],
        season_type: str = "regular",
        *,
        start_date: date | str | None = None,
    ) -> list[PlayerGameLog]:
        """Storico partita per partita di UN giocatore (ID ufficiale BALLDONTLIE) in una o piu' stagioni.

        Richiede il piano ALL-STAR o superiore (TierError con piano Free). ``season_start_year`` puo' essere
        una lista (es. [2026, 2025]): una sola richiesta paginata copre tutte le stagioni, e il giocatore
        trasferito durante l'anno e' gia' corretto perche' ogni riga porta la squadra di quella partita.
        ``start_date`` scarica solo le partite da quella data in poi (aggiornamento incrementale). Le righe
        sono deduplicate per partita; partite non terminate e playoff (con season_type="regular") sono escluse.
        """
        teams = self.get_teams()
        seasons = [int(season_start_year)] if isinstance(season_start_year, (int, float)) else [int(x) for x in season_start_year]
        params: dict[str, Any] = {
            "player_ids[]": [player_id],
            "seasons[]": seasons,
            "season_type": season_type,
            "per_page": 100,
        }
        if start_date is not None:
            params["start_date"] = start_date.isoformat() if isinstance(start_date, date) else str(start_date)
        rows = self._paginate("stats", params)
        if season_type == "regular":
            rows = [r for r in rows if not (r.get("game") or {}).get("postseason")]
        by_game: dict[str, PlayerGameLog] = {}
        for r in rows:
            log = parse_stat_row(r, teams)
            if log is not None:
                by_game[log.game_id] = log
        return sorted(by_game.values(), key=lambda l: (l.game_date, l.game_id))

    def get_injuries(self) -> list[dict[str, Any]]:
        """Richiede ALL-STAR o superiore."""
        out = []
        for row in self._paginate("player_injuries", {"per_page": 100}):
            player = row.get("player") or {}
            out.append(
                {
                    "player_id": player.get("id"),
                    "player_name": " ".join(
                        p for p in (player.get("first_name"), player.get("last_name")) if p
                    ),
                    "status": row.get("status"),
                    "return_date": row.get("return_date"),
                    "description": row.get("description"),
                }
            )
        return out

    def get_player_props(self, game_id: int, prop_type: str | None = None) -> PropsResult:
        """Richiede il piano GOAT. Le quote sono live: BALLDONTLIE non conserva lo storico."""
        params: dict[str, Any] = {"game_id": game_id}
        if prop_type:
            params["prop_type"] = prop_type
        fetched_at = utc_now()
        result = PropsResult()
        cursor = None
        for _ in range(50):
            page_params = dict(params)
            if cursor is not None:
                page_params["cursor"] = cursor
            body = self._client.get(PROPS_PATH, page_params).data
            rows = body.get("data", []) if isinstance(body, dict) else []
            for row in rows:
                result.raw_count += 1
                quotes, reason = parse_prop_row(row, fetched_at)
                result.quotes.extend(quotes)
                if reason == "unsupported_type":
                    result.skipped_unsupported_type += 1
                elif reason == "milestone":
                    result.skipped_milestone += 1
                elif reason == "invalid":
                    result.skipped_invalid += 1
            cursor = (body.get("meta") or {}).get("next_cursor") if isinstance(body, dict) else None
            if cursor is None:
                break
        return result

    def resolve_player_names(self, player_ids: list[int]) -> dict[int, str]:
        names: dict[int, str] = {}
        for start in range(0, len(player_ids), 100):
            chunk = player_ids[start : start + 100]
            for row in self._paginate("players", {"player_ids[]": chunk, "per_page": 100}):
                names[int(row["id"])] = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
        return names

    def _paginate(self, endpoint: str, params: dict[str, Any], max_pages: int = 50) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor = None
        for _ in range(max_pages):
            page_params = dict(params)
            if cursor is not None:
                page_params["cursor"] = cursor
            body = self._get_v1(endpoint, page_params).data
            if not isinstance(body, dict):
                break
            rows.extend(body.get("data", []))
            cursor = (body.get("meta") or {}).get("next_cursor")
            if cursor is None:
                break
        return rows
