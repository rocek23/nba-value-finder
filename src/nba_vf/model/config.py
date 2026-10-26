"""Iperparametri e prior del modello.

Ogni numero qui sotto e' una SCELTA DI MODELLO dichiarata, non un fatto stimato dai dati reali, salvo dove
indicato. Il backtest walk-forward (backtest.py) serve a confrontarli e a rivederli: finche' non e' stato
eseguito su dati reali, nessun valore e' da considerare "ottimizzato".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Componenti di PRA e chiavi dei tassi per minuto stimati
STATS = ("pts", "reb", "ast")
RATE_KEYS = ("pts", "reb", "ast", "fg3a")


def _default_ymax() -> dict[str, int]:
    # Il supporto delle distribuzioni e' 0..ymax; la massa oltre ymax viene accumulata nell'ultimo bin.
    # I valori sono scelti perche' la massa residua sia trascurabile (verificato nei test).
    return {"points": 90, "rebounds": 45, "assists": 35, "threes": 30, "pra": 140}


@dataclass(frozen=True)
class ModelConfig:
    # --- recenza (mezza-vita in numero di partite; None = nessun decadimento) ---
    half_life_minutes: float | None = 10.0
    half_life_rates: float | None = 20.0
    half_life_pct: float | None = None  # le percentuali al tiro mostrano poca persistenza: peso uniforme

    # --- soglie sui campioni ---
    min_games: int = 10  # sotto questa soglia il modello rifiuta di stimare
    low_sample_games: int = 25  # sotto questa soglia l'output e' marcato "campione ridotto"
    max_history_games: int = 100
    min_minutes_rate: float = 5.0  # partite con meno minuti non entrano nelle stime dei tassi
    min_attempts_3pt: int = 30  # tentativi da tre minimi per prezzare le triple

    # --- shrinkage ---
    # Usato SOLO senza prior di pool: forza (in minuti) con cui il tasso pesato per recenza viene riportato
    # verso il tasso medio non pesato del giocatore. Ipotesi, non stimata.
    fallback_rate_prior_minutes: float = 240.0
    alpha_prior_games: float = 20.0  # peso (in partite) del prior di pool sulla dispersione
    corr_shrink_games: float = 30.0  # forza del ritorno della correlazione residua verso il prior (o 0)
    icc_test_pvalue: float = 0.10  # soglia del test LR per adottare la beta-binomiale
    icc_shrink_games: float = 20.0  # shrinkage dell'ICC stimato verso 0
    icc_max: float = 0.5

    minutes_bandwidth_floor: float = 1.0
    ymax: dict[str, int] = field(default_factory=_default_ymax)

    # --- incertezza e simulazione ---
    n_boot: int = 200
    ci_level: float = 0.90
    n_sims_pra: int = 100_000
    n_sims_pra_boot: int = 20_000
    seed: int = 20261026

    def __post_init__(self) -> None:
        if self.min_games < 5:
            raise ValueError("min_games deve essere >= 5")
        if not 0.5 <= self.ci_level < 1.0:
            raise ValueError("ci_level deve essere in [0.5, 1)")
        if self.n_boot < 0 or self.n_sims_pra < 1 or self.n_sims_pra_boot < 1:
            raise ValueError("n_boot >= 0 e numero di simulazioni >= 1")
        for name in ("half_life_minutes", "half_life_rates", "half_life_pct"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} deve essere > 0 oppure None")


@dataclass(frozen=True, eq=False)
class LeaguePrior:
    """Prior di popolazione per lo shrinkage.

    ``source == "pool"``: stimato dai dati (prior.estimate_league_prior). ``source == "fallback"``: nessun
    dato di pool, quindi nessun numero esterno viene introdotto: il tasso viene riportato verso la media
    del giocatore stesso, le percentuali usano una Beta(0.5, 0.5) non informativa e la correlazione residua
    e' riportata verso zero.
    """

    source: str = "fallback"
    n_players: int = 0
    rho0: dict[str, float] | None = None  # tasso medio per minuto: pts, reb, ast, fg3a
    rate_strength_minutes: dict[str, float] | None = None  # T0: "minuti equivalenti" del prior
    alpha0: dict[str, float] | None = None  # dispersione NB2 tipica
    p3_prior: tuple[float, float] | None = None  # Beta(a0, b0) sulla percentuale da tre
    resid_corr: np.ndarray | None = None  # correlazione residua media (3x3: pts, reb, ast)

    @classmethod
    def fallback(cls) -> "LeaguePrior":
        return cls()

    @property
    def is_pool(self) -> bool:
        return self.source == "pool"
