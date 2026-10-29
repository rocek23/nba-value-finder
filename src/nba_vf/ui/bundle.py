"""Contenitore dei dati mostrati dall'interfaccia (demo o live) con le sue statistiche di sintesi."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..models import OddsQuote, PlayerGameLog
from .labels import normalize_name


@dataclass(frozen=True)
class PlayerInfo:
    key: str
    name: str
    team: str | None = None


@dataclass(frozen=True)
class EventInfo:
    event_id: str
    home: str | None
    away: str | None
    commence_time: datetime | None
    source: str = ""

    @property
    def title(self) -> str:
        return f"{self.away or '?'} @ {self.home or '?'}"


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    status: str  # "ok" | "non_configurato" | "errore" | "non_disponibile" | "manuale"
    detail: str = ""
    n_quotes: int = 0


@dataclass
class DataBundle:
    mode: str  # "demo" | "live"
    source_label: str
    status: str  # "ok" | "parziale" | "vuoto"
    updated_at: datetime | None
    quotes: list[OddsQuote] = field(default_factory=list)
    logs: dict[str, list[PlayerGameLog]] = field(default_factory=dict)
    players: dict[str, PlayerInfo] = field(default_factory=dict)
    events: dict[str, EventInfo] = field(default_factory=dict)
    providers: list[ProviderStatus] = field(default_factory=list)
    messages: list[tuple[str, str]] = field(default_factory=list)  # (livello, testo): info | warning | error
    collection_issues: dict[str, list[str]] = field(default_factory=dict)  # chiave giocatore -> problemi nella raccolta
    fingerprint: str = ""
    retrieval: dict = field(default_factory=dict)  # stato del recupero dello storico (solo live)
    injuries: dict | None = None  # elenco infortuni BALLDONTLIE; None = non scaricato (non significa "nessun infortunio")

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @property
    def is_empty(self) -> bool:
        return not self.quotes and not self.logs

    # ---- numeri di sintesi (schermata principale) ---------------------------------------------
    @property
    def n_games_analyzed(self) -> int:
        return len({l.game_id for logs in self.logs.values() for l in logs})

    @property
    def n_props(self) -> int:
        return len({(q.event_id, normalize_name(q.player_name), q.market, q.line) for q in self.quotes})

    @property
    def bookmakers(self) -> list[str]:
        return sorted({q.bookmaker for q in self.quotes})

    @property
    def n_bookmakers(self) -> int:
        return len(self.bookmakers)

    def player_name(self, key: str) -> str:
        info = self.players.get(key)
        return info.name if info else key

    def games_df(self) -> pd.DataFrame:
        """Partite presenti nello storico caricato (una riga per partita)."""
        rows: dict[str, dict] = {}
        for logs in self.logs.values():
            for l in logs:
                row = rows.setdefault(
                    l.game_id,
                    {"partita_id": l.game_id, "data": l.game_date, "squadre": set(), "giocatori": 0, "stagione": l.season},
                )
                if l.team:
                    row["squadre"].add(l.team)
                row["giocatori"] += 1
        out = []
        for r in rows.values():
            teams = sorted(r["squadre"])
            out.append({**r, "squadre": " / ".join(teams) if teams else "—"})
        df = pd.DataFrame(out, columns=["partita_id", "data", "squadre", "giocatori", "stagione"])
        return df.sort_values("data", ascending=False).reset_index(drop=True) if len(df) else df
