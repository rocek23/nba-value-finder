"""Archivio remoto opzionale: snapshot e storico quote sopravvivono a un file locale perso (backend in memoria)."""

from datetime import timedelta

import pytest

from nba_vf import storage
from nba_vf.backtest.quotes import load_quote_history
from nba_vf.models import Market, OddsQuote, Side, utc_now
from nba_vf.ui.snapshot import Snapshot, append_history, history_path, load_snapshot, save_snapshot, snapshot_path, sync_from_remote


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    storage.set_backend(None)
    yield
    storage.set_backend(None)


def _snap():
    now = utc_now()
    q = OddsQuote(provider="p", bookmaker="b", event_id="e", player_name="A B", player_id=None, market=Market.POINTS, line=10.5, side=Side.OVER,
                  odds_decimal=1.9, fetched_at=now, quote_time=now, commence_time=now + timedelta(hours=2))
    return Snapshot(fetched_at=now, quotes=[q])


def test_without_database_nothing_changes():
    assert not storage.enabled()
    save_snapshot(_snap())
    append_history(_snap())
    assert snapshot_path().exists() and history_path().exists() and len(load_quote_history(history_path())) == 1


def test_snapshot_and_history_survive_loss_of_local_files():
    storage.set_backend(storage.MemoryBackend())
    save_snapshot(_snap())
    append_history(_snap())
    append_history(_snap())
    snapshot_path().unlink()
    history_path().unlink()  # come dopo il riavvio di un hosting gratuito
    assert load_snapshot() is None
    sync_from_remote()
    assert load_snapshot() is not None and len(load_snapshot().quotes) == 1
    assert len(load_quote_history(history_path())) == 2


def test_remote_failure_does_not_break_saving():
    class Broken(storage.MemoryBackend):
        def put(self, *a): raise RuntimeError("down")
        def append(self, *a): raise RuntimeError("down")

    storage.set_backend(Broken())
    save_snapshot(_snap())
    append_history(_snap())
    assert snapshot_path().exists()
