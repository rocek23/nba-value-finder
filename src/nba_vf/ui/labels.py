"""Etichette italiane, normalizzazione dei nomi e formattazione."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

from ..models import Market, Side

LOCAL_TZ = ZoneInfo("Europe/Rome")

MARKET_IT = {
    Market.POINTS: "Punti",
    Market.REBOUNDS: "Rimbalzi",
    Market.ASSISTS: "Assist",
    Market.THREES: "Triple",
    Market.PRA: "PRA (P+R+A)",
}
MARKET_FROM_IT = {v: k for k, v in MARKET_IT.items()}
SIDE_IT = {Side.OVER: "Over", Side.UNDER: "Under"}

# Bookmaker italiani citati nel progetto: nessuna API verificata copre le loro player props (vedi docs/API_PROVIDERS.md)
ITALIAN_BOOKS_WITHOUT_API = ("Snai", "Sisal", "Eurobet", "Lottomatica", "Goldbet")

_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str | None) -> str:
    """Chiave stabile per abbinare lo stesso giocatore tra fonti diverse (accenti, punti, suffissi)."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[.'`’\-]", " ", text)
    tokens = [t for t in re.split(r"\s+", text.strip()) if t and t not in _SUFFIX]
    return " ".join(tokens)


def book_label(key: str) -> str:
    key = str(key)
    if key.startswith("demo_"):
        return "Book Demo " + key.split("_", 1)[1].upper()
    return key.replace("_", " ").title()


def to_local(dt: datetime | None) -> datetime | None:
    return dt.astimezone(LOCAL_TZ) if dt else None


def fmt_dt(dt: datetime | None, with_seconds: bool = False) -> str:
    local = to_local(dt)
    if local is None:
        return "—"
    return local.strftime("%d/%m/%Y %H:%M" + (":%S" if with_seconds else ""))


def fmt_age(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    if minutes < 1:
        return "meno di 1 minuto fa"
    if minutes < 90:
        return f"{int(minutes)} min fa"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} h fa"
    return f"{minutes / 1440:.1f} giorni fa"


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or x != x else f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None or x != x else f"{x:.{digits}f}"
