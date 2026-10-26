"""Divisione TEMPORALE in addestramento / validazione / test (mai casuale).

- addestramento: storico di partenza (serve a stimare il prior di popolazione e la storia dei giocatori);
- validazione:   si sceglie qui la configurazione del modello, la ricalibrazione e la soglia di valore;
- test:          valutato UNA sola volta, con tutto gia' congelato.

Dentro validazione e test vale il walk-forward: ogni previsione usa soltanto partite con data strettamente
precedente. I confini sono date di calendario, quindi nessuna partita puo' stare in due fasi.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

TRAIN, VALIDATION, TEST = "addestramento", "validazione", "test"
PHASES = (TRAIN, VALIDATION, TEST)


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class TemporalSplit:
    first: date
    train_end: date  # ultima data dell'addestramento (inclusa)
    val_end: date  # ultima data della validazione (inclusa)
    last: date

    def __post_init__(self) -> None:
        if not (self.first <= self.train_end < self.val_end < self.last):
            raise SplitError("confini non ordinati: serve first <= train_end < val_end < last")

    def phase(self, d: date) -> str:
        if d <= self.train_end:
            return TRAIN
        return VALIDATION if d <= self.val_end else TEST

    def to_dict(self) -> dict[str, str]:
        return {k: v.isoformat() for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "TemporalSplit":
        return cls(**{k: date.fromisoformat(v) for k, v in d.items()})


def make_split(dates: Iterable[date], train_frac: float = 0.5, val_frac: float = 0.25, min_days: int = 5) -> TemporalSplit:
    """Confini per quantili delle date di gioco distinte (cronologico, nessun rimescolamento)."""
    if not (0.1 <= train_frac < 0.9 and 0.05 <= val_frac < 0.9 and train_frac + val_frac < 0.95):
        raise SplitError("frazioni non valide: servono train >= 0.1, validazione >= 0.05 e un test non vuoto")
    u = sorted(set(dates))
    n = len(u)
    if n < 3 * min_days:
        raise SplitError(f"servono almeno {3 * min_days} giornate di gioco distinte, ce ne sono {n}")
    i_tr = max(int(n * train_frac) - 1, min_days - 1)
    i_val = max(int(n * (train_frac + val_frac)) - 1, i_tr + min_days)
    if n - 1 - i_val < min_days:
        raise SplitError("il periodo di test avrebbe meno di %d giornate" % min_days)
    return TemporalSplit(u[0], u[i_tr], u[i_val], u[-1])
