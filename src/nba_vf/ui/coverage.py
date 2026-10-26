"""Copertura dei dati: per ogni giocatore con quote, quanto storico c'e' e perche' una quota non viene prezzata.

Logica pura (nessuna dipendenza da Streamlit). Non stima nulla: classifica soltanto lo storico disponibile e
spiega le esclusioni, usando le stesse soglie del modello (``ModelConfig.min_games`` e ``low_sample_games``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..model.config import ModelConfig
from .board import Board, usable_logs
from .bundle import DataBundle
from .labels import normalize_name

COMPLETE, PARTIAL, INSUFFICIENT, NONE = "Completo", "Parziale", "Insufficiente", "Senza storico"
STATES = (COMPLETE, PARTIAL, INSUFFICIENT, NONE)
STATE_HELP = {
    COMPLETE: "partite giocate ≥ soglia del campione ridotto: stime con incertezza normale",
    PARTIAL: "abbastanza partite per stimare, ma campione ridotto: intervalli più larghi",
    INSUFFICIENT: "meno partite del minimo richiesto dal modello: nessuna probabilità viene calcolata",
    NONE: "nessuna partita disponibile: nessuna probabilità viene calcolata",
}
PLAYER_COLUMNS = ["giocatore", "squadra", "stato", "partite", "ultima_partita", "stagioni", "quote", "prezzate", "nota"]


@dataclass
class Coverage:
    players: pd.DataFrame
    state_counts: dict[str, int]
    n_players: int
    n_quotes: int
    n_evaluable: int  # quote di giocatori con storico sufficiente (>= min_games)
    n_priced: int | None  # None se il tabellone non e' stato calcolato
    reasons: pd.DataFrame  # categoria -> quote e giocatori esclusi
    retrieval: dict = field(default_factory=dict)
    min_games: int = 10
    low_sample: int = 25

    @property
    def n_with_history(self) -> int:
        return self.state_counts[COMPLETE] + self.state_counts[PARTIAL]


def classify(n_games: int, cfg: ModelConfig) -> str:
    if n_games <= 0:
        return NONE
    if n_games < cfg.min_games:
        return INSUFFICIENT
    if n_games < cfg.low_sample_games:
        return PARTIAL
    return COMPLETE


def _season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[2:]}"


def _reason_category(motivo: str) -> str:
    """Raggruppa i motivi di esclusione scritti dal tabellone in poche categorie leggibili."""
    base = (motivo or "").split(" — ")[0].strip()
    if base.startswith("linea non gestita"):
        return "linea non gestita dal modello"
    return base or "non prezzabile"


def build_coverage(bundle: DataBundle, board: Board | None = None, cfg: ModelConfig | None = None) -> Coverage:
    cfg = cfg or ModelConfig()
    first_quote: dict[str, object] = {}
    n_quotes_by_player: dict[str, int] = {}
    for q in bundle.quotes:
        key = normalize_name(q.player_name)
        first_quote.setdefault(key, q)
        n_quotes_by_player[key] = n_quotes_by_player.get(key, 0) + 1

    priced_by_player: dict[str, int] = {}
    if board is not None and len(board.rows):
        priced_by_player = board.rows[board.rows["priced"]].groupby("key").size().to_dict()

    rows = []
    for key, q in first_quote.items():
        logs = usable_logs(bundle, key, q)
        clean = sorted((l for l in logs if l.minutes and l.minutes > 0), key=lambda l: (l.game_date, l.game_id))
        clean = clean[-cfg.max_history_games:]
        n = len(clean)
        state = classify(n, cfg)
        seasons = sorted({l.season for l in clean})
        notes: list[str] = []
        issues = bundle.collection_issues.get(key, [])
        if state == NONE:
            notes.append("; ".join(issues) or "nessuno storico partite disponibile")
        elif state == INSUFFICIENT:
            notes.append(f"{n} partite giocate, ne servono almeno {cfg.min_games}")
            if len(seasons) <= 1:
                notes.append("storico limitato a una sola stagione (stagione appena iniziata o giocatore esordiente)")
        elif state == PARTIAL:
            notes.append(f"campione ridotto ({n} partite, sotto {cfg.low_sample_games})")
        if state != NONE and issues:
            notes.append("; ".join(issues))
        recent_teams = [l.team for l in clean[-10:] if l.team]
        if len(set(recent_teams)) > 1:
            notes.append(f"cambio squadra recente ({recent_teams[0]} → {recent_teams[-1]}): ruolo e minuti da verificare")
        info = bundle.players.get(key)
        rows.append({
            "key": key, "giocatore": bundle.player_name(key), "squadra": info.team if info else None, "stato": state,
            "partite": n, "ultima_partita": clean[-1].game_date if clean else None,
            "stagioni": ", ".join(_season_label(x) for x in seasons) or "—",
            "quote": n_quotes_by_player.get(key, 0), "prezzate": int(priced_by_player.get(key, 0)),
            "nota": " · ".join(notes),
        })
    players = pd.DataFrame(rows, columns=["key"] + PLAYER_COLUMNS)
    if len(players):
        order = {s: i for i, s in enumerate(STATES)}
        players = players.sort_values(["stato", "quote"], key=lambda c: c.map(order) if c.name == "stato" else -c).reset_index(drop=True)

    counts = {s: int((players["stato"] == s).sum()) if len(players) else 0 for s in STATES}
    n_quotes = int(players["quote"].sum()) if len(players) else 0
    evaluable_players = set(players.loc[players["partite"] >= cfg.min_games, "key"]) if len(players) else set()
    n_evaluable = int(players.loc[players["key"].isin(evaluable_players), "quote"].sum()) if len(players) else 0
    n_priced = int(board.rows["priced"].sum()) if board is not None and len(board.rows) else (0 if board is not None else None)

    reasons = pd.DataFrame(columns=["motivo", "quote", "giocatori"])
    if board is not None and len(board.rows):
        un = board.rows[~board.rows["priced"]]
        if len(un):
            cat = un["motivo"].map(_reason_category)
            reasons = (un.assign(cat=cat).groupby("cat").agg(quote=("key", "size"), giocatori=("key", "nunique"))
                       .reset_index().rename(columns={"cat": "motivo"}).sort_values("quote", ascending=False).reset_index(drop=True))
    return Coverage(players=players, state_counts=counts, n_players=len(players), n_quotes=n_quotes, n_evaluable=n_evaluable,
                    n_priced=n_priced, reasons=reasons, retrieval=dict(bundle.retrieval or {}),
                    min_games=cfg.min_games, low_sample=cfg.low_sample_games)
