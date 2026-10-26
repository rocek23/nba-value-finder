import pytest

from nba_vf.config import ConfigError, Settings, describe_secret, redact
from nba_vf.convert import american_to_decimal, parse_decimal_odds


@pytest.mark.parametrize(
    "american, expected",
    [(100, 2.0), (150, 2.5), (-100, 2.0), (-200, 1.5), (-110, 1 + 100 / 110), (250, 3.5)],
)
def test_american_to_decimal(american, expected):
    assert american_to_decimal(american) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0, 50, -50, float("nan"), float("inf")])
def test_american_to_decimal_rejects_invalid(bad):
    with pytest.raises(ValueError):
        american_to_decimal(bad)


def test_parse_decimal_odds_accepts_italian_comma():
    assert parse_decimal_odds("1,85") == pytest.approx(1.85)


@pytest.mark.parametrize("bad", [1.0, 0.5, -2, "abc", None, float("nan")])
def test_parse_decimal_odds_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_decimal_odds(bad)


def test_settings_defaults_and_parsing():
    s = Settings.from_env({"BALLDONTLIE_REQUESTS_PER_MIN": "60", "ODDS_REGIONS": "EU, us "})
    assert s.balldontlie_requests_per_min == 60
    assert s.odds_regions == ("eu", "us")
    assert s.balldontlie_api_key is None


def test_settings_rejects_bad_numbers():
    with pytest.raises(ConfigError):
        Settings.from_env({"HTTP_TIMEOUT_SECONDS": "abc"})
    with pytest.raises(ConfigError):
        Settings.from_env({"BALLDONTLIE_REQUESTS_PER_MIN": "0"})


def test_settings_never_leaks_keys():
    s = Settings.from_env({"BALLDONTLIE_API_KEY": "SECRET-BDL-123", "THE_ODDS_API_KEY": "SECRET-ODDS-456"})
    shown = repr(s) + str(s.describe())
    assert "SECRET-BDL-123" not in shown and "SECRET-ODDS-456" not in shown
    assert describe_secret("abcdef") == "configurata (6 caratteri)"
    assert describe_secret(None) == "non configurata"


def test_redact_known_and_generic_secrets():
    text = "GET https://x/v4/odds?apiKey=ABC123&regions=eu with SECRET-9 inside"
    cleaned = redact(text, ["SECRET-9"])
    assert "ABC123" not in cleaned and "SECRET-9" not in cleaned
    assert "regions=eu" in cleaned
