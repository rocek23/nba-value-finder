"""Adapter nba_api (https://pypi.org/project/nba-api/): wrapper community degli endpoint di NBA.com.

Verificato nella pagina PyPI: nessuna chiave richiesta, licenza MIT, i dati restano soggetti ai Termini d'Uso
di NBA.com, stats.nba.com limita le richieste e non risponde da molti IP cloud (AWS, GCP, Azure, Colab).
Da verificare al primo run reale: nomi dei parametri di PlayerGameLog e delle colonne restituite.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime
from typing import Any

from ..models import ConnectionResult, PlayerGameLog

PROVIDER = "nba_api"
_MATCHUP = re.compile(r"^\s*([A-Z]{2,4})\s+(vs\.|@)\s+([A-Z]{2,4})\s*$")


def season_string(start_year: int) -> str:
    """2025 -> '2025-26'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season_start_year(today: date) -> int:
    """La stagione NBA inizia a ottobre: da ottobre in poi vale l'anno corrente."""
    return today.year if today.month >= 10 else today.year - 1


def parse_matchup(matchup: str) -> tuple[str | None, str | None, bool | None]:
    """'BOS vs. NYK' -> ('BOS','NYK',True); 'BOS @ NYK' -> ('BOS','NYK',False)."""
    match = _MATCHUP.match(matchup or "")
    if not match:
        return None, None, None
    team, marker, opponent = match.groups()
    return team, opponent, marker == "vs."


def parse_game_date(value: Any) -> date:
    text = str(value).strip()
    for fmt in ("%b %d, %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.title() if fmt == "%b %d, %Y" else text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"data partita non riconosciuta: {value!r}")


def parse_game_log_frame(frame, season_start_year: int, player_id: int, player_name: str | None = None):
    """Converte il DataFrame di PlayerGameLog in PlayerGameLog. Funzione pura (testabile senza rete)."""
    logs: list[PlayerGameLog] = []
    for _, row in frame.iterrows():
        team, opponent, is_home = parse_matchup(str(row.get("MATCHUP", "")))
        logs.append(
            PlayerGameLog(
                source=PROVIDER,
                player_id=player_id,
                player_name=player_name,
                game_id=str(row.get("Game_ID")),
                game_date=parse_game_date(row.get("GAME_DATE")),
                season=season_start_year,
                team=team,
                opponent=opponent,
                is_home=is_home,
                minutes=float(row.get("MIN") or 0),
                pts=int(row.get("PTS") or 0),
                reb=int(row.get("REB") or 0),
                ast=int(row.get("AST") or 0),
                fg3m=int(row.get("FG3M") or 0),
                fg3a=int(row.get("FG3A") or 0),
                fgm=int(row.get("FGM") or 0),
                fga=int(row.get("FGA") or 0),
                ftm=int(row.get("FTM") or 0),
                fta=int(row.get("FTA") or 0),
            )
        )
    return sorted(logs, key=lambda l: l.game_date)


class NbaApiAdapter:
    provider = PROVIDER

    def __init__(self, *, sleep_seconds: float = 1.5, timeout: float = 20.0, sleep=time.sleep) -> None:
        self._sleep_seconds = sleep_seconds
        self._timeout = timeout
        self._sleep = sleep
        self._last_call: float | None = None

    def _pause(self) -> None:
        if self._last_call is not None:
            remaining = self._sleep_seconds - (time.monotonic() - self._last_call)
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = time.monotonic()

    @staticmethod
    def find_player(full_name: str) -> list[dict[str, Any]]:
        """Ricerca offline nell'elenco incluso nel pacchetto (nessuna chiamata di rete)."""
        from nba_api.stats.static import players

        return players.find_players_by_full_name(full_name)

    def get_player_game_logs(
        self, player_id: int, season_start_year: int, season_type: str = "Regular Season"
    ) -> list[PlayerGameLog]:
        from nba_api.stats.endpoints import playergamelog

        self._pause()
        endpoint = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season_string(season_start_year),
            season_type_all_star=season_type,
            timeout=self._timeout,
        )
        frame = endpoint.get_data_frames()[0]
        return parse_game_log_frame(frame, season_start_year, player_id)

    def test_connection(self) -> ConnectionResult:
        try:
            import nba_api  # noqa: F401
            from nba_api.stats.static import players  # noqa: F401
        except ImportError:
            return ConnectionResult(PROVIDER, "failed", "pacchetto nba_api non installato (pip install nba_api)")

        found = self.find_player("LeBron James")
        if not found:
            return ConnectionResult(PROVIDER, "failed", "elenco giocatori offline vuoto: installazione anomala")
        try:
            start = current_season_start_year(date.today())
            self.get_player_game_logs(found[0]["id"], start - 1)
        except Exception as exc:  # nba_api puo' sollevare errori di rete di varia natura
            return ConnectionResult(
                PROVIDER,
                "failed",
                f"libreria OK, ma stats.nba.com non ha risposto ({type(exc).__name__}). "
                "Succede spesso da IP cloud o dopo troppe richieste ravvicinate.",
            )
        return ConnectionResult(PROVIDER, "ok", "stats.nba.com ha risposto (PlayerGameLog)")
