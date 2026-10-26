"""Quote, probabilita' e valore atteso con gestione ESPLICITA del push (rimborso).

Una scommessa con stake 1 a quota decimale d (che include lo stake) ha tre esiti:
    vince   (p_win)  -> incasso d       (profitto d - 1)
    rimborso(p_push) -> incasso 1       (profitto 0)
    perde   (p_loss) -> incasso 0       (profitto -1)

    ROI atteso            = p_win * d + p_push - 1  = p_win * (d - 1) - p_loss
    quota equa            = (1 - p_push) / p_win          (ROI = 0)
    p_win di pareggio     = (1 - p_push) / d              (soglia oltre cui il ROI e' > 0)
    probabilita' implicita = 1 / d   (grezza: include il margine del book e ignora il push)

La formula "binaria" (p * d - 1, quota equa 1/p) e' il caso particolare p_push = 0. Per questo ``p_push`` e'
un argomento OBBLIGATORIO di tutte le funzioni: non esiste una via implicita alla formula binaria.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import InvalidLineError, ModelError

LINE_TOL = 1e-9
PROB_TOL = 1e-9


@dataclass(frozen=True)
class SideProbs:
    """Probabilita' di vincita / rimborso / perdita di UN lato (Over o Under) di UNA linea."""

    p_win: float
    p_push: float
    p_loss: float

    def __post_init__(self) -> None:
        for name in ("p_win", "p_push", "p_loss"):
            v = getattr(self, name)
            if not (-PROB_TOL <= v <= 1.0 + PROB_TOL):
                raise ModelError(f"{name}={v} fuori da [0, 1]")
        if abs(self.p_win + self.p_push + self.p_loss - 1.0) > 1e-6:
            raise ModelError("p_win + p_push + p_loss deve valere 1")


# --------------------------------------------------------------------------------------------------
# Linee e soglie
# --------------------------------------------------------------------------------------------------
def classify_line(line: float, support_max: int) -> tuple[int, bool]:
    """Valida la linea e restituisce (base, is_integer).

    Sono ammesse solo linee x.0 e x.5. ``base`` e' L per una linea intera, floor(L) per una mezza linea.
    ``support_max`` e' l'ultimo indice della pmf (la cui ultima cella contiene la coda): la linea deve
    stare dentro la griglia, altrimenti la probabilita' dipenderebbe dalla coda accorpata.
    """
    try:
        x = float(line)
    except (TypeError, ValueError) as exc:
        raise InvalidLineError(f"linea non numerica: {line!r}") from exc
    if not math.isfinite(x) or x < 0:
        raise InvalidLineError(f"linea non valida: {line!r}")
    twice = 2.0 * x
    if abs(twice - round(twice)) > LINE_TOL:
        raise InvalidLineError(f"linea {line!r}: sono ammessi solo multipli di 0.5 (x.0 e x.5)")
    is_int = abs(x - round(x)) < LINE_TOL
    base = int(round(x)) if is_int else int(math.floor(x))
    if base + 1 > support_max:
        raise InvalidLineError(f"linea {line!r} fuori dalla griglia del modello (max {support_max - 1})")
    return base, is_int


def split_probs(pmf: np.ndarray, line: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P(Y < L), P(Y = L), P(Y > L)) per ogni pmf (ultimo asse = valori 0..K-1).

    Mezza linea: P(Y = L) = 0 (nessun push). Linea intera: P(Y = L) = pmf[L] (push).
    Le tre probabilita' sommano a 1 per costruzione, quindi Over e Under sono complementari ESATTAMENTE:
    win_over = loss_under, loss_over = win_under, push uguale.
    """
    pmf = np.asarray(pmf, dtype=float)
    if pmf.ndim < 1 or np.any(pmf < -PROB_TOL):
        raise ModelError("pmf non valida")
    if np.any(np.abs(pmf.sum(axis=-1) - 1.0) > 1e-6):
        raise ModelError("la pmf non somma a 1")
    base, is_int = classify_line(line, pmf.shape[-1] - 1)
    p_gt = pmf[..., base + 1 :].sum(axis=-1)
    if is_int:
        p_eq = pmf[..., base]
        p_lt = pmf[..., :base].sum(axis=-1)
    else:
        p_eq = np.zeros_like(p_gt)
        p_lt = pmf[..., : base + 1].sum(axis=-1)
    s = p_lt + p_eq + p_gt
    return p_lt / s, p_eq / s, p_gt / s


def side_probs_arrays(pmf: np.ndarray, line: float, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p_win, p_push, p_loss) per Over ('over') o Under ('under'); accetta pmf singole o a pila."""
    p_lt, p_eq, p_gt = split_probs(pmf, line)
    side = getattr(side, "value", side)
    if side == "over":
        return p_gt, p_eq, p_lt
    if side == "under":
        return p_lt, p_eq, p_gt
    raise ModelError(f"lato sconosciuto: {side!r}")


# --------------------------------------------------------------------------------------------------
# Quote, ROI, EV
# --------------------------------------------------------------------------------------------------
def _check_odds(d):
    d = np.asarray(d, dtype=float)
    if np.any(~np.isfinite(d)) or np.any(d <= 1.0):
        raise ModelError("la quota decimale deve essere finita e > 1.0")
    return d


def implied_probability(decimal_odds):
    """1 / quota: probabilita' implicita GREZZA (include margine del book, ignora l'eventuale push)."""
    return 1.0 / _check_odds(decimal_odds)


def breakeven_win_prob(decimal_odds, p_push):
    """Probabilita' di vincita minima per ROI >= 0, dato il push: (1 - p_push) / d."""
    return (1.0 - np.asarray(p_push, dtype=float)) / _check_odds(decimal_odds)


def expected_roi(p_win, p_push, decimal_odds):
    """ROI atteso per unita' di stake: p_win * d + p_push - 1."""
    d = _check_odds(decimal_odds)
    return np.asarray(p_win, dtype=float) * d + np.asarray(p_push, dtype=float) - 1.0


def fair_odds(p_win, p_push):
    """Quota equa (ROI = 0): (1 - p_push) / p_win. Se p_win = 0 la quota equa e' infinita."""
    p_win = np.asarray(p_win, dtype=float)
    p_push = np.asarray(p_push, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(p_win > 0, (1.0 - p_push) / np.where(p_win > 0, p_win, 1.0), np.inf)
    return out


def expected_value(stake: float, roi):
    """Valore atteso in denaro: stake * ROI."""
    if not math.isfinite(stake) or stake < 0:
        raise ModelError("lo stake deve essere un numero >= 0")
    return stake * np.asarray(roi, dtype=float)
