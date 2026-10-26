"""Interfacce minime degli adapter."""

from __future__ import annotations

from typing import Protocol

from ..models import ConnectionResult, OddsQuote, PlayerGameLog


class StatsProvider(Protocol):
    provider: str

    def test_connection(self) -> ConnectionResult: ...

    def get_player_game_logs(self, player_id: int, season_start_year: int) -> list[PlayerGameLog]: ...


class OddsProvider(Protocol):
    provider: str

    def test_connection(self) -> ConnectionResult: ...


__all__ = ["StatsProvider", "OddsProvider", "OddsQuote"]
