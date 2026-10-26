"""Generazione delle previsioni walk-forward 'point-in-time' e conversione prior <-> dizionario."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Mapping, Sequence

import numpy as np

from ..model.config import LeaguePrior, ModelConfig
from ..model.engine import GameContext, ValueEngine
from ..model.errors import InsufficientDataError
from ..model.prior import estimate_league_prior
from ..models import Market, PlayerGameLog


class LeakageError(AssertionError):
    """Una previsione ha usato (o potrebbe aver usato) informazioni non disponibili prima della partita."""


@dataclass
class PredRecord:
    candidate: str
    phase: str
    player: str  # chiave del pool
    log: PlayerGameLog  # la partita da prevedere (serve SOLO per l'esito, mai per la previsione)
    history: list[PlayerGameLog]  # esattamente le partite passate al modello
    block_start: date  # blocco del prior di popolazione usato
    pmfs: dict[Market, np.ndarray]
    flags: tuple[str, ...] = ()
    b2b: bool | None = None

    @property
    def hist_last(self) -> date:
        return self.history[-1].game_date

    @property
    def key(self) -> tuple[str, str]:
        return self.player, self.log.game_id


def config_dict(cfg: ModelConfig) -> dict:
    return dataclasses.asdict(cfg)


def config_from_dict(d: dict) -> ModelConfig:
    return ModelConfig(**d)


def stable_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def prior_to_dict(p: LeaguePrior) -> dict:
    return {
        "source": p.source, "n_players": p.n_players, "rho0": p.rho0, "rate_strength_minutes": p.rate_strength_minutes,
        "alpha0": p.alpha0, "p3_prior": list(p.p3_prior) if p.p3_prior else None,
        "resid_corr": None if p.resid_corr is None else np.asarray(p.resid_corr).tolist(),
    }


def prior_from_dict(d: dict) -> LeaguePrior:
    return LeaguePrior(
        source=d["source"], n_players=d["n_players"], rho0=d["rho0"], rate_strength_minutes=d["rate_strength_minutes"],
        alpha0=d["alpha0"], p3_prior=tuple(d["p3_prior"]) if d["p3_prior"] else None,
        resid_corr=None if d["resid_corr"] is None else np.array(d["resid_corr"]),
    )


def history_window(logs_before: Sequence[PlayerGameLog], cfg: ModelConfig) -> list[PlayerGameLog]:
    """Le partite che il motore usa davvero (stesso criterio di ValueEngine._prepare)."""
    clean = [l for l in logs_before if l.minutes and l.minutes > 0]
    return clean[-cfg.max_history_games:]


def assert_no_leak(history: Sequence[PlayerGameLog], game: PlayerGameLog) -> None:
    bad = [l.game_id for l in history if l.game_date >= game.game_date or l.game_id == game.game_id]
    if bad:
        raise LeakageError(f"storico con partite non anteriori a {game.game_date}: {bad[:3]}")


def sort_pool(pool: Mapping[str, Sequence[PlayerGameLog]]) -> dict[str, list[PlayerGameLog]]:
    return {k: sorted(v, key=lambda l: (l.game_date, l.game_id)) for k, v in pool.items()}


def generate_records(
    pool: Mapping[str, Sequence[PlayerGameLog]], cfg: ModelConfig, candidate: str, split_phase, lo_excl: date, hi_incl: date,
    *, block_days: int = 30, min_players: int = 20, use_context: bool = True, markets: Sequence[Market] | None = None,
) -> tuple[list[PredRecord], dict[date, dict]]:
    """Previsioni walk-forward per le partite con ``lo_excl < data <= hi_incl``.

    Ogni previsione riceve SOLO le partite con data strettamente precedente; il prior di popolazione e' ristimato a
    blocchi usando solo partite anteriori all'inizio del blocco. ``split_phase`` mappa una data nella sua fase."""
    cfg = dataclasses.replace(cfg, n_boot=0)
    sorted_pool = sort_pool(pool)
    records: list[PredRecord] = []
    priors: dict[date, dict] = {}
    block_start = lo_excl + timedelta(days=1)
    while block_start <= hi_incl:
        block_end = block_start + timedelta(days=block_days)
        past = {k: [l for l in logs if l.game_date < block_start] for k, logs in sorted_pool.items()}
        n_used = sum(len(v) for v in past.values())
        max_used = max((l.game_date for v in past.values() for l in v), default=None)
        if max_used is not None and max_used >= block_start:  # difesa: non puo' accadere per costruzione
            raise LeakageError("il prior usa partite del blocco da prevedere")
        try:
            prior = estimate_league_prior(past, cfg, min_players)
        except ValueError:
            prior = LeaguePrior.fallback()
        priors[block_start] = {"prior": prior_to_dict(prior), "n_partite": n_used, "ultima_data": max_used.isoformat() if max_used else None}
        engine = ValueEngine(cfg, prior)
        for key, logs in sorted_pool.items():
            for i, game in enumerate(logs):
                if not (block_start <= game.game_date < block_end) or game.game_date > hi_incl:
                    continue
                if not game.minutes or game.minutes <= 0:
                    continue  # non ha giocato: non c'e' esito da prevedere
                hist = history_window([l for l in logs[:i] if l.game_date < game.game_date], cfg)
                if len(hist) < cfg.min_games:
                    continue
                assert_no_leak(hist, game)
                b2b = (game.game_date - hist[-1].game_date).days == 1 if use_context else None
                try:
                    pred = engine.predict(hist, GameContext(back_to_back=b2b), intervals=False)
                except InsufficientDataError:
                    continue
                pmfs = {m: p.pmf for m, p in pred.markets.items() if markets is None or m in markets}
                records.append(PredRecord(candidate, split_phase(game.game_date), key, game, hist, block_start, pmfs, pred.flags, b2b))
        block_start = block_end
    return records, priors
