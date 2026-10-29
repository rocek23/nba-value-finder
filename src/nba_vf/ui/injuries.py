"""Infortuni dall'elenco BALLDONTLIE (solo informativi: non modificano le probabilita' del modello).

I testi esatti dello stato restituiti dall'API sono **Da verificare** al primo aggiornamento reale: qui si riconosce solo «out»
(assente, anche «Out For Season»); ogni altro stato viene trattato come «dubbio» e mostrato cosi' com'e'.
"""

from __future__ import annotations

OUT, DOUBT, NONE = "out", "dubbio", ""


def injury_level(status: str | None) -> str:
    s = (status or "").strip().lower()
    if not s:
        return NONE
    return OUT if "out" in s.split() or s.startswith("out") else DOUBT


def mask(values, keep) -> "np.ndarray":
    """Maschera booleana sugli stati (sicura anche con zero righe: una lista vuota come indice toglierebbe tutte le colonne)."""
    import numpy as np

    return np.array([keep(injury_level(s)) for s in values], dtype=bool)


def injury_status(bundle, key: str) -> str:
    """Stato segnalato per il giocatore, oppure stringa vuota (anche se l'elenco non e' stato scaricato)."""
    inj = getattr(bundle, "injuries", None)
    if not inj:
        return ""
    return str((inj.get(key) or {}).get("status") or "")


def injury_text(status: str) -> str:
    lvl = injury_level(status)
    return "" if lvl == NONE else f"Segnalato «{status}» nell'elenco infortuni"
