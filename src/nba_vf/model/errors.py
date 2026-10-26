class ModelError(Exception):
    """Errore del motore statistico."""


class InsufficientDataError(ModelError):
    """Campione troppo piccolo per stimare il modello in modo responsabile."""


class InvalidLineError(ModelError, ValueError):
    """Linea non gestita: sono ammesse solo linee intere e mezze (x.0 e x.5) dentro la griglia del modello."""
