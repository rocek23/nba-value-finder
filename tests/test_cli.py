"""Aggiornamento automatico da riga di comando: nessuna chiave stampata, registro aggiornato, errori chiari."""

from datetime import timedelta
from pathlib import Path

import pytest

from nba_vf import cli
from nba_vf.model.synthetic import TruePlayer, generate_player
from nba_vf.models import Market, OddsQuote, Side, utc_now
from nba_vf.ui import live, tracker
from nba_vf.ui.snapshot import Snapshot

BDL_KEY = "bdl-secret-key-123456"
ODDS_KEY = "odds-secret-key-654321"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("BALLDONTLIE_API_KEY", BDL_KEY)
    monkeypatch.setenv("THE_ODDS_API_KEY", ODDS_KEY)
    tracker.reset_cache()
    yield
    tracker.reset_cache()


def fake_snapshot():
    now = utc_now()
    logs = generate_player(TruePlayer(name="Giocatore Test"), 45, 3)
    qs = [OddsQuote("the_odds_api", bk, "e1", "Giocatore Test", None, Market.POINTS, 17.5, s, o, now, now, "HOM", "AWY", now + timedelta(hours=6), False)
          for bk in ("book_x", "book_y") for s, o in ((Side.OVER, 2.4), (Side.UNDER, 1.6))]
    return Snapshot(fetched_at=now, quotes=qs, logs={"giocatore test": logs}, players={"giocatore test": {"name": "Giocatore Test", "team": "HOM"}},
                    providers=[{"name": "The Odds API", "status": "ok", "detail": "test", "n_quotes": 4}], injuries={})


def test_refresh_runs_records_picks_and_prints_no_secret(monkeypatch, capsys):
    monkeypatch.setattr(live, "refresh_live", lambda settings, opts, progress=None: fake_snapshot())
    code = cli.main(["refresh"])
    out = capsys.readouterr().out
    assert code == 0 and "Quote: 4" in out and "Giocate nuove registrate" in out
    assert BDL_KEY not in out and ODDS_KEY not in out
    n_logged = len(tracker.load_picks())
    assert f"registrate: {n_logged}" in out and n_logged >= 1


def test_refresh_twice_does_not_duplicate_picks(monkeypatch, capsys):
    monkeypatch.setattr(live, "refresh_live", lambda settings, opts, progress=None: fake_snapshot())
    cli.main(["refresh"])
    first = len(tracker.load_picks())
    tracker.reset_cache()
    cli.main(["refresh"])
    assert len(tracker.load_picks()) == first


def test_no_keys_exits_with_clear_message(monkeypatch, capsys):
    monkeypatch.delenv("BALLDONTLIE_API_KEY")
    monkeypatch.delenv("THE_ODDS_API_KEY")
    assert cli.main(["refresh"]) == 2
    assert "Nessuna chiave API" in capsys.readouterr().out


def test_error_is_reported_without_leaking_keys(monkeypatch, capsys):
    def boom(settings, opts, progress=None):
        raise RuntimeError(f"HTTP 500 apiKey={ODDS_KEY} token {BDL_KEY}")
    monkeypatch.setattr(live, "refresh_live", boom)
    assert cli.main(["refresh"]) == 1
    out = capsys.readouterr().out
    assert "RuntimeError" in out and ODDS_KEY not in out and BDL_KEY not in out


def test_settle_command(monkeypatch, capsys):
    monkeypatch.setattr(live, "settle_now", lambda settings: tracker.SettleReport(settled=3, void=1, waiting=2))
    assert cli.main(["settle"]) == 0
    assert "Chiuse 3 giocate" in capsys.readouterr().out
    monkeypatch.setattr(live, "settle_now", lambda settings: None)
    assert cli.main(["settle"]) == 1


def test_workflow_file_never_contains_secret_values_and_uses_secrets_context():
    text = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "aggiorna-dati.yml").read_text(encoding="utf-8")
    for name in ("BALLDONTLIE_API_KEY", "THE_ODDS_API_KEY", "DATABASE_URL"):
        assert f"${{{{ secrets.{name} }}}}" in text
    assert "permissions:" in text and "contents: read" in text and "cron:" in text
