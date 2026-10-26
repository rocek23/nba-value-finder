import httpx
import pytest

from nba_vf.http import ApiClient, ApiError, RateLimitError, TierError


def make_client(handler, **kwargs):
    sleeps: list[float] = []
    client = ApiClient(
        "https://example.test",
        provider="test",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        **kwargs,
    )
    return client, sleeps


def test_success_returns_json_and_headers():
    client, _ = make_client(lambda r: httpx.Response(200, json={"ok": True}, headers={"x-test": "1"}))
    response = client.get("/ping")
    assert response.data == {"ok": True}
    assert response.headers["x-test"] == "1"


def test_retries_on_429_honouring_retry_after():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "3"})
        return httpx.Response(200, json={"ok": 1})

    client, sleeps = make_client(handler, max_retries=2)
    assert client.get("/x").data == {"ok": 1}
    assert calls["n"] == 2
    assert sleeps == [3.0]


def test_persistent_429_raises_rate_limit_error():
    client, _ = make_client(lambda r: httpx.Response(429), max_retries=1)
    with pytest.raises(RateLimitError):
        client.get("/x")


def test_401_is_tier_error_and_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401)

    client, _ = make_client(handler, max_retries=3)
    with pytest.raises(TierError) as info:
        client.get("/stats")
    assert calls["n"] == 1
    assert info.value.status == 401


def test_404_raises_api_error_without_retry():
    client, _ = make_client(lambda r: httpx.Response(404), max_retries=3)
    with pytest.raises(ApiError) as info:
        client.get("/nope")
    assert info.value.status == 404


def test_timeout_message_redacts_api_key():
    def handler(request):
        raise httpx.ConnectTimeout("timed out calling https://example.test/x?apiKey=SECRET123")

    client, _ = make_client(handler, max_retries=1, default_params={"apiKey": "SECRET123"}, secrets=["SECRET123"])
    with pytest.raises(ApiError) as info:
        client.get("/x")
    assert "SECRET123" not in str(info.value)


def test_default_params_are_sent_but_never_in_error_text():
    seen = {}

    def handler(request):
        seen["query"] = str(request.url.query)
        return httpx.Response(500)

    client, _ = make_client(handler, max_retries=0, default_params={"apiKey": "SECRET123"}, secrets=["SECRET123"])
    with pytest.raises(ApiError) as info:
        client.get("/x", {"a": 1})
    assert b"apiKey=SECRET123" in seen["query"].encode()
    assert "SECRET123" not in str(info.value)


def test_non_json_response_is_api_error():
    client, _ = make_client(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(ApiError):
        client.get("/x")


def test_throttle_spaces_requests():
    now = {"t": 0.0}
    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now["t"] += seconds

    client = ApiClient(
        "https://example.test",
        provider="test",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        min_interval=12.0,
        sleep=fake_sleep,
        monotonic=lambda: now["t"],
    )
    client.get("/a")
    client.get("/b")
    assert sleeps == [12.0]
