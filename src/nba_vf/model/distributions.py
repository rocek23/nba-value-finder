"""Primitive distribuzionali del modello. Funzioni pure, vettorializzate e coperte da test.

Convenzioni
-----------
- Tutte le pmf sono su interi 0..ymax; la massa oltre ymax e' accumulata nell'ultimo bin, quindi ogni pmf somma a 1.
- Binomiale negativa "NB2": media mu, varianza mu + alpha * mu^2 (alpha = 1/k). Per alpha -> 0 tende alla Poisson.
- Beta-binomiale parametrizzata da (p, rho) con rho = correlazione intra-classe (ICC); rho -> 0 da la binomiale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln
from scipy.stats import binom, chi2, norm

# Griglia dei minuti giocati: 1..53 (circa 5 minuti oltre i 48 regolamentari, per i supplementari)
MINUTES_GRID = np.arange(1.0, 54.0)

POISSON_ALPHA = 1e-6  # sotto questa dispersione si usa la formula di Poisson (evita cancellazioni numeriche)


# --------------------------------------------------------------------------------------------------
# Pesi e statistiche pesate
# --------------------------------------------------------------------------------------------------
def recency_weights(n: int, half_life: float | None) -> np.ndarray:
    """Pesi per partite ordinate dalla piu' vecchia alla piu' recente: l'ultima vale 1, ogni
    ``half_life`` partite il peso si dimezza. ``None`` = pesi uniformi."""
    if n <= 0:
        return np.zeros(0)
    if half_life is None:
        return np.ones(n)
    age = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (age / half_life)


def effective_n(weights: np.ndarray) -> float:
    """Numerosita' campionaria efficace di Kish: (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return 0.0
    return float(total * total / np.sum(w * w))


def weighted_corr(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Matrice di correlazione pesata di x (n, d). Colonne a varianza nulla -> correlazione 0."""
    w = np.asarray(w, dtype=float)
    mean = (w[:, None] * x).sum(axis=0) / w.sum()
    xc = x - mean
    cov = (w[:, None, None] * xc[:, :, None] * xc[:, None, :]).sum(axis=0) / w.sum()
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 1e-12, cov / denom, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return (corr + corr.T) / 2.0


# --------------------------------------------------------------------------------------------------
# Binomiale negativa
# --------------------------------------------------------------------------------------------------
def nb_pmf_matrix(mu: np.ndarray, alpha: float, ymax: int) -> np.ndarray:
    """pmf NB2 per ogni media in ``mu``: matrice (len(mu), ymax+1), ogni riga somma a 1."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)[:, None]
    y = np.arange(ymax + 1, dtype=float)[None, :]
    if alpha < POISSON_ALPHA:
        logp = y * np.log(mu) - mu - gammaln(y + 1.0)
    else:
        k = 1.0 / alpha
        logp = (
            gammaln(y + k) - gammaln(k) - gammaln(y + 1.0) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu))
        )
    p = np.exp(logp)
    p[:, -1] += np.clip(1.0 - p.sum(axis=1), 0.0, None)  # massa di coda nell'ultimo bin
    return p / p.sum(axis=1, keepdims=True)


def nb_alpha_moment(y: np.ndarray, mu: np.ndarray, w: np.ndarray, cap: float = 5.0) -> float:
    """Stimatore dei momenti della dispersione NB2 condizionata alla media: sum w[(y-mu)^2 - mu] / sum w mu^2,
    troncato in [0, cap]. Un valore <= 0 significa che i dati non mostrano sovradispersione."""
    den = float(np.sum(w * mu * mu))
    if den <= 0:
        return 0.0
    alpha = float(np.sum(w * ((y - mu) ** 2 - mu)) / den)
    return float(min(max(alpha, 0.0), cap))


# --------------------------------------------------------------------------------------------------
# Binomiale / beta-binomiale
# --------------------------------------------------------------------------------------------------
def betabinom_pmf_matrix(amax: int, xmax: int, p: float, rho: float) -> np.ndarray:
    """Matrice P(X = x | a tentativi), righe a = 0..amax, colonne x = 0..xmax. Righe sommano a 1
    (massa oltre xmax nell'ultimo bin; con xmax >= amax non c'e' perdita)."""
    a = np.arange(amax + 1)[:, None]
    x = np.arange(xmax + 1)[None, :]
    if rho < 1e-9:
        mat = binom.pmf(x, a, p)
    else:
        s = (1.0 - rho) / rho
        al, be = p * s, (1.0 - p) * s
        valid = x <= a
        xv = np.where(valid, x, 0)
        av = np.broadcast_to(a, valid.shape)
        logc = gammaln(av + 1.0) - gammaln(xv + 1.0) - gammaln(av - xv + 1.0)
        mat = np.exp(logc + betaln(xv + al, av - xv + be) - betaln(al, be))
        mat = np.where(valid, mat, 0.0)
    mat = np.asarray(mat, dtype=float)
    mat[:, -1] += np.clip(1.0 - mat.sum(axis=1), 0.0, None)
    return mat / mat.sum(axis=1, keepdims=True)


def betabinom_loglik(x: np.ndarray, a: np.ndarray, p: float, rho: float) -> np.ndarray:
    """Log-verosimiglianza per partita (a > 0) della beta-binomiale(a, p, rho)."""
    s = (1.0 - rho) / rho
    al, be = p * s, (1.0 - p) * s
    logc = gammaln(a + 1.0) - gammaln(x + 1.0) - gammaln(a - x + 1.0)
    return logc + betaln(x + al, a - x + be) - betaln(al, be)


@dataclass(frozen=True)
class IccResult:
    rho: float  # ICC stimata (0 se non c'e' evidenza di sovradispersione)
    p_value: float  # test LR binomiale vs beta-binomiale (mistura 0.5*chi2_0 + 0.5*chi2_1)
    lr: float
    ll_bb: float
    ll_bin: float
    n_games: int


def fit_icc(x: np.ndarray, a: np.ndarray, p: float, rho_max: float = 0.5, min_games: int = 5) -> IccResult:
    """Stima l'ICC della beta-binomiale a p fissato e testa H0: rho = 0 (binomiale).

    Il test usa il rapporto di verosimiglianza; poiche' rho = 0 e' sul bordo dello spazio parametrico la
    distribuzione nulla e' 0.5*chi2(0) + 0.5*chi2(1), quindi p = 0.5 * P(chi2_1 > LR).
    """
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    keep = a > 0
    x, a = np.minimum(x[keep], a[keep]), a[keep]
    n = int(len(a))
    if n < min_games or not 0.0 < p < 1.0:
        return IccResult(0.0, 1.0, 0.0, 0.0, 0.0, n)
    ll_bin = float(binom.logpmf(x, a, p).sum())
    res = minimize_scalar(
        lambda r: -float(betabinom_loglik(x, a, p, r).sum()),
        bounds=(1e-6, rho_max),
        method="bounded",
        options={"xatol": 1e-6},
    )
    ll_bb = -float(res.fun)
    lr = max(0.0, 2.0 * (ll_bb - ll_bin))
    if lr <= 0.0:
        return IccResult(0.0, 1.0, 0.0, ll_bb, ll_bin, n)
    return IccResult(float(res.x), float(0.5 * chi2.sf(lr, 1)), lr, ll_bb, ll_bin, n)


# --------------------------------------------------------------------------------------------------
# Distribuzione dei minuti
# --------------------------------------------------------------------------------------------------
def minutes_pmf(
    m: np.ndarray,
    w: np.ndarray,
    target_mean: float,
    extra_var: float,
    grid: np.ndarray = MINUTES_GRID,
    bandwidth_floor: float = 1.0,
) -> np.ndarray:
    """pmf dei minuti sulla griglia: KDE gaussiana pesata dei minuti storici, traslata in modo che la media
    sia ``target_mean`` (la forma, ad es. l'asimmetria da garbage time, e' quella osservata). La larghezza del
    kernel somma la banda di Silverman e ``extra_var`` (incertezza sulla media)."""
    m = np.asarray(m, dtype=float)
    w = np.asarray(w, dtype=float)
    mean_w = float(np.average(m, weights=w))
    sd = float(np.sqrt(np.average((m - mean_w) ** 2, weights=w)))
    n_eff = max(effective_n(w), 1.0)
    h = max(bandwidth_floor, 0.9 * sd * n_eff ** (-0.2))
    s = float(np.sqrt(h * h + max(extra_var, 0.0)))
    x = m + (target_mean - mean_w)
    dens = (w[None, :] * norm.pdf((grid[:, None] - x[None, :]) / s)).sum(axis=1)
    total = dens.sum()
    if total <= 0:  # target lontanissimo dai dati: ricade su una gaussiana centrata sul target
        dens = norm.pdf((grid - target_mean) / s)
        total = dens.sum()
    return dens / total


# --------------------------------------------------------------------------------------------------
# Campionamento da pmf discrete per riga (inversione della CDF, interamente vettorializzata)
# --------------------------------------------------------------------------------------------------
def cdf_table(pmf_rows: np.ndarray) -> tuple[np.ndarray, int]:
    """Tabella piatta e monotona per invertire la CDF di ogni riga: valore = indice_riga + CDF."""
    cdf = np.minimum(np.cumsum(pmf_rows, axis=1), 1.0)
    cdf[:, -1] = 1.0
    flat = (cdf + np.arange(cdf.shape[0])[:, None]).ravel()
    return flat, cdf.shape[1]


def sample_rows(table: np.ndarray, ncols: int, row_idx: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Per ogni campione, il piu' piccolo y con CDF_riga(y) >= u."""
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    pos = np.searchsorted(table, row_idx + u, side="left")
    return pos - row_idx * ncols
