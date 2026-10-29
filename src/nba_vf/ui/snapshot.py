"""Salvataggio su disco (JSON) dell'ultimo prelievo live, cosi' l'interfaccia non chiama le API a ogni ricarica.

Il file contiene dati di mercato e statistiche, MAI chiavi API. Sta in data/cache/ (escluso da Git).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .. import storage
from ..models import Market, OddsQuote, PlayerGameLog, Side

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 1


def cache_dir() -> Path:
    return Path(os.environ.get("NBA_VF_CACHE_DIR") or ROOT / "data" / "cache")


def snapshot_path() -> Path:
    return cache_dir() / "live_snapshot.json"


@dataclass
class Snapshot:
    fetched_at: datetime
    quotes: list[OddsQuote] = field(default_factory=list)
    logs: dict[str, list[PlayerGameLog]] = field(default_factory=dict)
    players: dict[str, dict[str, Any]] = field(default_factory=dict)  # key -> {name, team}
    events: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> {home, away, commence_time, source}
    providers: list[dict[str, Any]] = field(default_factory=list)
    issues: dict[str, list[str]] = field(default_factory=dict)
    messages: list[list[str]] = field(default_factory=list)
    logs_fetched_at: dict[str, datetime] = field(default_factory=dict)
    logs_seasons: dict[str, list[int]] = field(default_factory=dict)  # stagioni effettivamente richieste per ogni giocatore
    retrieval: dict[str, Any] = field(default_factory=dict)  # stato del recupero dello storico (vedi LiveCollector)
    injuries: dict[str, dict[str, Any]] | None = None  # chiave giocatore -> {status, description, return_date}; None = elenco non scaricato


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def quote_to_dict(q: OddsQuote) -> dict[str, Any]:
    return {
        "provider": q.provider, "bookmaker": q.bookmaker, "event_id": q.event_id, "player_name": q.player_name,
        "player_id": q.player_id, "market": q.market.value, "line": q.line, "side": q.side.value,
        "odds_decimal": q.odds_decimal, "fetched_at": _dt(q.fetched_at), "quote_time": _dt(q.quote_time),
        "home_team": q.home_team, "away_team": q.away_team, "commence_time": _dt(q.commence_time),
        "is_alternate": q.is_alternate, "source": q.source,
    }


def quote_from_dict(d: dict[str, Any]) -> OddsQuote:
    return OddsQuote(
        provider=d["provider"], bookmaker=d["bookmaker"], event_id=d["event_id"], player_name=d.get("player_name"),
        player_id=d.get("player_id"), market=Market(d["market"]), line=float(d["line"]), side=Side(d["side"]),
        odds_decimal=float(d["odds_decimal"]), fetched_at=_parse_dt(d["fetched_at"]),
        quote_time=_parse_dt(d.get("quote_time")), home_team=d.get("home_team"), away_team=d.get("away_team"),
        commence_time=_parse_dt(d.get("commence_time")), is_alternate=d.get("is_alternate"),
        source=d.get("source", "api"),
    )


def log_to_dict(l: PlayerGameLog) -> dict[str, Any]:
    d = {f: getattr(l, f) for f in l.__dataclass_fields__}
    d["game_date"] = l.game_date.isoformat()
    return d


def log_from_dict(d: dict[str, Any]) -> PlayerGameLog:
    d = dict(d)
    d["game_date"] = date.fromisoformat(d["game_date"])
    return PlayerGameLog(**d)


def save_snapshot(snap: Snapshot, path: Path | None = None) -> Path:
    path = path or snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": SCHEMA,
        "fetched_at": _dt(snap.fetched_at),
        "quotes": [quote_to_dict(q) for q in snap.quotes],
        "logs": {k: [log_to_dict(l) for l in v] for k, v in snap.logs.items()},
        "players": snap.players,
        "events": snap.events,
        "providers": snap.providers,
        "issues": snap.issues,
        "messages": snap.messages,
        "logs_fetched_at": {k: _dt(v) for k, v in snap.logs_fetched_at.items()},
        "logs_seasons": snap.logs_seasons,
        "retrieval": snap.retrieval,
        "injuries": snap.injuries,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    if storage.enabled() and path == snapshot_path():
        try:
            storage.get_backend().put("live_snapshot", path.read_text(encoding="utf-8"))
        except Exception:  # il salvataggio remoto e' un extra: non deve rompere l'aggiornamento
            pass
    return path


def sync_from_remote() -> None:
    """Se c'e' un archivio remoto piu' recente del file locale (es. dopo il riavvio dell'hosting), lo scarica come cache locale."""
    if not storage.enabled():
        return
    try:
        got = storage.get_backend().get("live_snapshot")
    except Exception:
        return
    if not got:
        return
    text, updated = got
    path = snapshot_path()
    if path.exists() and path.stat().st_mtime >= updated.timestamp():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_snapshot(path: Path | None = None) -> Snapshot | None:
    path = path or snapshot_path()
    if not path.exists():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != SCHEMA:
        return None
    return Snapshot(
        fetched_at=_parse_dt(body["fetched_at"]),
        quotes=[quote_from_dict(q) for q in body.get("quotes", [])],
        logs={k: [log_from_dict(l) for l in v] for k, v in body.get("logs", {}).items()},
        players=body.get("players", {}),
        events=body.get("events", {}),
        providers=body.get("providers", []),
        issues=body.get("issues", {}),
        messages=body.get("messages", []),
        logs_fetched_at={k: _parse_dt(v) for k, v in body.get("logs_fetched_at", {}).items()},
        logs_seasons={k: [int(x) for x in v] for k, v in body.get("logs_seasons", {}).items()},
        retrieval=body.get("retrieval", {}),
        injuries=body.get("injuries"),
    )


def history_path() -> Path:
    return cache_dir() / "quote_history.jsonl"


def append_history(snap: Snapshot) -> None:
    """Accoda le quote API del prelievo a uno storico locale (BALLDONTLIE non conserva quote storiche)."""
    api = [q for q in snap.quotes if q.source != "csv"]
    if not api:
        return
    record = json.dumps({"fetched_at": _dt(snap.fetched_at), "quotes": [quote_to_dict(q) for q in api]}, ensure_ascii=False)
    if storage.enabled():
        try:
            storage.get_backend().append("quote_history", record)
        except Exception:
            pass
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record + "\n")
