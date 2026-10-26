"""Ricalibrazione logistica e regressione di calibrazione."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .metrics import EPS


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(p, y, ridge: float = 0.0) -> tuple[float, float]:
    """(a, b) di logit(q) = a + b*logit(p) per massima verosimiglianza; ``ridge`` tira verso (0, 1)."""
    z, y = _logit(p), np.asarray(y, float)
    if len(z) < 5 or np.ptp(z) < 1e-9:
        return 0.0, 1.0

    def nll(theta):
        a, b = theta
        q = np.clip(1 / (1 + np.exp(-(a + b * z))), EPS, 1 - EPS)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)) + 0.5 * ridge * (a * a + (b - 1) ** 2)

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    return float(res.x[0]), float(res.x[1])


@dataclass
class Recalibrator:
    """logit(p') = a + b*logit(p), stimato sulla VALIDAZIONE (previsioni fuori campione) con ridge verso l'identita'."""

    a: float = 0.0
    b: float = 1.0
    ridge: float = 20.0
    n_fit: int = 0

    def fit(self, p, y) -> "Recalibrator":
        self.a, self.b = fit_logistic(p, y, self.ridge)
        self.n_fit = int(len(p))
        return self

    def predict(self, p) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a + self.b * _logit(p))))
