"""Orchestrazione: dalle partite storiche alle predizioni per mercato, con intervalli di incertezza.

Flusso: storico -> (opzionale) verifica aggiustamenti di contesto -> fit base -> pmf per mercato ->
bootstrap sulle partite (intervalli) -> ``PlayerPrediction``.

Gli intervalli dal bootstrap coprono SOLO l'incertezza di stima dei parametri a modello dato: non coprono
l'errore di specificazione (un modello sbagliato puo' avere un intervallo stretto e restare sbagliato).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..models import Market, PlayerGameLog
from .config import LeaguePrior, ModelConfig
from .context import ContextEffect, estimate_effects
from .errors import InsufficientDataError
from .fit import FittedPlayer, build_inputs, fit_arrays
from .home_away import HomeAwayEffect, estimate_home_away
from .predict import PraDraws, all_pmfs, pra_pmf_independent
from .prediction import MarketPrediction
from .prior import estimate_league_prior

MARKET_KEYS = {
    "points": Market.POINTS,
    "rebounds": Market.REBOUNDS,
    "assists": Market.ASSISTS,
    "threes": Market.THREES,
    "pra": Market.PRA,
}
INTERVAL_FLAG = "intervalli_coprono_solo_incertezza_di_stima"


@dataclass(frozen=True)
class GameContext:
    """Contesto della partita da prezzare. Solo ``back_to_back`` puo' generare un aggiustamento stimato (e solo
    se verificato sullo storico). ``opponent`` e' registrato ma NON entra nel calcolo. ``minutes_override`` e'
    una scelta MANUALE (es. minutaggio atteso con compagni assenti): non e' stimata e viene segnalata."""

    opponent: str | None = None
    back_to_back: bool | None = None
    minutes_override: float | None = None
    is_home: bool | None = None  # usato SOLO se l'effetto casa/trasferta e' stato verificato fuori campione (home_away.py)


@dataclass
class PlayerPrediction:
    player_id: int | None
    player_name: str | None
    fit: FittedPlayer
    markets: dict[Market, MarketPrediction]  # quelle da usare (con aggiustamenti verificati, se ci sono)
    base_markets: dict[Market, MarketPrediction]  # modello base, senza contesto
    adjustments: list[ContextEffect] = field(default_factory=list)
    effects: dict[str, ContextEffect] = field(default_factory=dict)
    context: GameContext = field(default_factory=GameContext)
    diagnostics: dict[str, float] = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    def market(self, m: Market | str) -> MarketPrediction:
        key = Market(m) if not isinstance(m, Market) else m
        if key not in self.markets:
            raise KeyError(f"mercato {key.value} non prezzabile per questo giocatore (dati insufficienti)")
        return self.markets[key]


class ValueEngine:
    def __init__(self, cfg: ModelConfig | None = None, prior: LeaguePrior | None = None):
        self.cfg = cfg or ModelConfig()
        self.prior = prior or LeaguePrior.fallback()
        self.home_away: dict[str, HomeAwayEffect] = {}  # vuoto finche' non si chiama fit_home_away: nessun effetto applicato

    def fit_prior(self, pool: Mapping[int, Sequence[PlayerGameLog]], min_players: int = 20) -> LeaguePrior:
        """Stima il prior di popolazione dal pool e lo adotta; con pool troppo piccolo mantiene il fallback."""
        try:
            self.prior = estimate_league_prior(pool, self.cfg, min_players)
        except ValueError:
            self.prior = LeaguePrior.fallback()
        return self.prior

    def fit_home_away(self, pool: Mapping[object, Sequence[PlayerGameLog]], min_players: int = 20) -> dict[str, HomeAwayEffect]:
        """Stima e verifica fuori campione l'effetto casa/trasferta. Senza verifica superata non cambia nessuna previsione."""
        self.home_away = estimate_home_away(pool, seed=self.cfg.seed, min_players=min_players)
        return self.home_away

    # ------------------------------------------------------------------------------------------
    def _prepare(self, logs: Sequence[PlayerGameLog]) -> list[PlayerGameLog]:
        clean = sorted((l for l in logs if l.minutes and l.minutes > 0), key=lambda l: (l.game_date, l.game_id))
        clean = clean[-self.cfg.max_history_games :]
        if len(clean) < self.cfg.min_games:
            raise InsufficientDataError(f"servono almeno {self.cfg.min_games} partite giocate, ce ne sono {len(clean)}")
        return clean

    def _markets(self, fit, inp, scales, seed, boot: bool) -> dict[Market, MarketPrediction]:
        cfg = self.cfg
        central = all_pmfs(fit, cfg, PraDraws.make(cfg.n_sims_pra, seed))
        reps: dict[str, list[np.ndarray]] = {k: [] for k in central}
        if boot and cfg.n_boot > 0:
            rng = np.random.default_rng(seed + 7)
            draws = PraDraws.make(cfg.n_sims_pra_boot, seed + 1)  # numeri casuali comuni per tutte le repliche
            n = len(inp)
            for _ in range(cfg.n_boot):
                idx = rng.integers(0, n, n)
                try:
                    f = fit_arrays(inp.take(idx), cfg, self.prior, use_bb=fit.use_bb, **scales)
                except InsufficientDataError:
                    continue
                for k, pmf in all_pmfs(f, cfg, draws).items():
                    if pmf is not None:
                        reps[k].append(pmf)
        out: dict[Market, MarketPrediction] = {}
        for k, pmf in central.items():
            if pmf is None:
                continue
            r = np.array(reps[k]) if reps[k] else None
            flags = tuple(fit.flags) + ((INTERVAL_FLAG,) if r is not None else ())
            out[MARKET_KEYS[k]] = MarketPrediction(MARKET_KEYS[k], pmf, r, cfg.ci_level, flags)
        return out

    def predict(
        self,
        logs: Sequence[PlayerGameLog],
        context: GameContext | None = None,
        *,
        intervals: bool = True,
    ) -> PlayerPrediction:
        cfg = self.cfg
        context = context or GameContext()
        clean = self._prepare(logs)
        inp = build_inputs(clean, cfg)
        seed = cfg.seed

        effects: dict[str, ContextEffect] = {}
        applied: list[ContextEffect] = []
        if context.back_to_back:
            effects = estimate_effects(clean, cfg)
            applied = [e for e in effects.values() if e.applied]
        home_scales: dict[str, float] = {}
        if context.is_home is not None:
            for t, he in self.home_away.items():
                if he.applied:
                    f = he.factor_home if context.is_home else he.factor_away
                    home_scales[t] = f
                    applied.append(ContextEffect("casa_trasferta", t, he.n_players, he.n_players, he.ratio, he.ci, True, f, he.reason))

        base_fit = fit_arrays(inp, cfg, self.prior, minutes_override=context.minutes_override)
        base_markets = self._markets(base_fit, inp, {"minutes_override": context.minutes_override}, seed, intervals)

        flags = list(base_fit.flags)
        fit, markets = base_fit, base_markets
        if applied:
            scales: dict = {"minutes_override": context.minutes_override}
            rate_scales: dict[str, float] = {}
            for e in applied:
                if e.target != "minutes":
                    rate_scales[e.target] = rate_scales.get(e.target, 1.0) * e.shrunk_ratio
            if rate_scales:
                scales["rate_scales"] = rate_scales
            for e in applied:
                if e.target == "minutes" and context.minutes_override is None:
                    scales["minutes_scale"] = e.shrunk_ratio
            fit = fit_arrays(inp, cfg, self.prior, use_bb=base_fit.use_bb, **scales)
            markets = self._markets(fit, inp, scales, seed, intervals)
            names = sorted({e.factor for e in applied})
            flags.append("aggiustamento_contesto_applicato:" + "+".join(names))
        if context.opponent:
            flags.append("avversario_registrato_non_usato_nel_calcolo")

        sim_sd = markets[Market.PRA].sd
        ind = pra_pmf_independent(fit, cfg)
        ind_sd = float(np.sqrt(ind @ (np.arange(len(ind)) - ind @ np.arange(len(ind))) ** 2))
        diagnostics = {
            "pra_sd_simulata": sim_sd,
            "pra_sd_indipendente": ind_sd,
            "pra_errore_mc_max_p": float(np.sqrt(0.25 / cfg.n_sims_pra)),
        }
        return PlayerPrediction(
            player_id=clean[-1].player_id,
            player_name=clean[-1].player_name,
            fit=fit,
            markets=markets,
            base_markets=base_markets,
            adjustments=applied,
            effects=effects,
            context=context,
            diagnostics=diagnostics,
            flags=tuple(flags),
        )
