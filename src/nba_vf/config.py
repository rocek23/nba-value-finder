"""Configurazione da variabili d'ambiente. Le chiavi non compaiono mai in repr, log o messaggi."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Mapping


class ConfigError(ValueError):
    pass


def load_env(path: str | os.PathLike | None = None) -> None:
    """Carica .env (se presente) senza sovrascrivere variabili gia' impostate."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=path, override=False)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _number(env: Mapping[str, str], name: str, default, cast, minimum):
    raw = _clean(env.get(name))
    if raw is None:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ConfigError(f"{name} deve essere un numero valido") from None
    if value < minimum:
        raise ConfigError(f"{name} deve essere >= {minimum}")
    return value


_KEY_IN_TEXT = re.compile(r"(?i)(api[_-]?key|apikey|authorization)(\s*[=:]\s*)([^&\s,;]+)")


def redact(text: str, secrets: list[str] | tuple[str, ...] = ()) -> str:
    """Rimuove da un testo i segreti noti e qualunque 'apiKey=...' generico."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return _KEY_IN_TEXT.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


def describe_secret(value: str | None) -> str:
    """Descrizione sicura di una chiave: mai il valore, nemmeno parziale."""
    if not value:
        return "non configurata"
    return f"configurata ({len(value)} caratteri)"


@dataclass(frozen=True)
class Settings:
    balldontlie_api_key: str | None = field(default=None, repr=False)
    odds_api_key: str | None = field(default=None, repr=False)
    balldontlie_requests_per_min: int = 60
    odds_regions: tuple[str, ...] = ("eu", "us")
    nba_api_sleep_seconds: float = 1.5
    http_timeout: float = 15.0
    http_max_retries: int = 3
    max_quote_age_minutes: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        regions_raw = _clean(env.get("ODDS_REGIONS")) or "eu,us"
        regions = tuple(r.strip().lower() for r in regions_raw.split(",") if r.strip())
        if not regions:
            raise ConfigError("ODDS_REGIONS non puo' essere vuoto")
        return cls(
            balldontlie_api_key=_clean(env.get("BALLDONTLIE_API_KEY")),
            odds_api_key=_clean(env.get("THE_ODDS_API_KEY")),
            balldontlie_requests_per_min=_number(env, "BALLDONTLIE_REQUESTS_PER_MIN", 60, int, 1),
            odds_regions=regions,
            nba_api_sleep_seconds=_number(env, "NBA_API_SLEEP_SECONDS", 1.5, float, 0.0),
            http_timeout=_number(env, "HTTP_TIMEOUT_SECONDS", 15.0, float, 1.0),
            http_max_retries=_number(env, "HTTP_MAX_RETRIES", 3, int, 0),
            max_quote_age_minutes=_number(env, "MAX_QUOTE_AGE_MINUTES", 30.0, float, 1.0),
        )

    def secrets(self) -> list[str]:
        return [s for s in (self.balldontlie_api_key, self.odds_api_key) if s]

    def describe(self) -> dict[str, str]:
        """Vista sicura per dashboard e log."""
        return {
            "BALLDONTLIE_API_KEY": describe_secret(self.balldontlie_api_key),
            "THE_ODDS_API_KEY": describe_secret(self.odds_api_key),
            "BALLDONTLIE_REQUESTS_PER_MIN": str(self.balldontlie_requests_per_min),
            "ODDS_REGIONS": ",".join(self.odds_regions),
            "NBA_API_SLEEP_SECONDS": str(self.nba_api_sleep_seconds),
            "HTTP_TIMEOUT_SECONDS": str(self.http_timeout),
            "HTTP_MAX_RETRIES": str(self.http_max_retries),
            "MAX_QUOTE_AGE_MINUTES": str(self.max_quote_age_minutes),
        }
