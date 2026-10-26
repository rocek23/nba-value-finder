"""Client HTTP con timeout, retry, rispetto di Retry-After, rate limit e redazione dei segreti."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import httpx

from .config import redact

RETRY_STATUS = {429, 500, 502, 503, 504}


class ApiError(Exception):
    def __init__(self, message: str, *, status: int | None = None, provider: str | None = None):
        super().__init__(message)
        self.status = status
        self.provider = provider


class RateLimitError(ApiError):
    """429 persistente dopo i retry."""


class TierError(ApiError):
    """401/403: chiave mancante/non valida oppure piano che non include l'endpoint."""


@dataclass
class ApiResponse:
    data: Any
    headers: Mapping[str, str]
    status: int


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        provider: str,
        headers: Mapping[str, str] | None = None,
        default_params: Mapping[str, Any] | None = None,
        secrets: list[str] | tuple[str, ...] = (),
        timeout: float = 15.0,
        max_retries: int = 3,
        min_interval: float = 0.0,
        backoff_base: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self._secrets = [s for s in secrets if s]
        self._default_params = dict(default_params or {})
        self._max_retries = max_retries
        self._min_interval = min_interval
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request: float | None = None
        self.request_count = 0
        self._client = httpx.Client(
            base_url=base_url, headers=dict(headers or {}), timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def _safe(self, text: str) -> str:
        return redact(text, self._secrets)

    def _throttle(self) -> None:
        if self._min_interval <= 0 or self._last_request is None:
            return
        wait = self._min_interval - (self._monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        return self._backoff_base * (2**attempt)

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> ApiResponse:
        merged: dict[str, Any] = {**self._default_params, **(params or {})}
        last_error: ApiError | None = None

        for attempt in range(self._max_retries + 1):
            self._throttle()
            self._last_request = self._monotonic()
            self.request_count += 1
            response: httpx.Response | None = None
            try:
                response = self._client.get(path, params=merged)
            except httpx.TimeoutException as exc:
                last_error = ApiError(
                    self._safe(f"{self.provider}: timeout su {path} ({exc})"), provider=self.provider
                )
            except httpx.TransportError as exc:
                last_error = ApiError(
                    self._safe(f"{self.provider}: errore di rete su {path} ({exc})"),
                    provider=self.provider,
                )
            else:
                status = response.status_code
                if status in RETRY_STATUS:
                    cls = RateLimitError if status == 429 else ApiError
                    last_error = cls(
                        f"{self.provider}: HTTP {status} su {path}", status=status, provider=self.provider
                    )
                elif status in (401, 403):
                    raise TierError(
                        f"{self.provider}: HTTP {status} su {path}. Chiave mancante o non valida, "
                        "oppure il tuo piano non include questo endpoint.",
                        status=status,
                        provider=self.provider,
                    )
                elif status >= 400:
                    raise ApiError(
                        f"{self.provider}: HTTP {status} su {path}", status=status, provider=self.provider
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError:
                        raise ApiError(
                            f"{self.provider}: risposta non JSON su {path}",
                            status=status,
                            provider=self.provider,
                        ) from None
                    return ApiResponse(data=data, headers=response.headers, status=status)

            if attempt < self._max_retries:
                self._sleep(self._retry_delay(response, attempt))

        assert last_error is not None
        raise last_error
