"""Infortuni: lettura nello snapshot, avviso e esclusione dei giocatori «Out» (senza toccare le probabilita' del modello)."""

from datetime import date

import pandas as pd
import pytest

from nba_vf.models import utc_now
from nba_vf.ui import builder as B
from nba_vf.ui.best import build_picks
from nba_vf.ui.injuries import DOUBT, NONE, OUT, injury_level, injury_status
from nba_vf.ui.live import bundle_from_snapshot
from nba_vf.ui.snapshot import Snapshot, load_snapshot, save_snapshot
from tests.test_builder import make_rows
from tests.test_history_recovery import FakeBdl, NOW, collect, make_logs, spaced


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))


def test_levels():
    assert injury_level("Out") == OUT and injury_level("Out For Season") == OUT and injury_level("out") == OUT
    assert injury_level("Day-To-Day") == DOUBT and injury_level("Questionable") == DOUBT
    assert injury_level("") == NONE and injury_level(None) == NONE


def test_snapshot_round_trip_and_none_means_not_downloaded():
    snap = Snapshot(fetched_at=utc_now(), injuries={"lebron james": {"status": "Out", "description": "x", "return_date": None}})
    save_snapshot(snap)
    assert load_snapshot().injuries["lebron james"]["status"] == "Out"
    save_snapshot(Snapshot(fetched_at=utc_now()))
    assert load_snapshot().injuries is None
    assert bundle_from_snapshot(load_snapshot()).injuries is None


class BdlWithInjuries(FakeBdl):
    def __init__(self, *a, fail=False, **k):
        super().__init__(*a, **k)
        self.fail = fail

    def get_injuries(self):
        if self.fail:
            raise RuntimeError("piano non incluso")
        return [{"player_id": 1, "player_name": "Player One", "status": "Out", "return_date": None, "description": "caviglia"}]


def _bdl(**kw):
    logs = make_logs(1, "Player One", spaced(date(2026, 10, 24), 30), 2026)
    return BdlWithInjuries({1: logs}, {1: "Player One"}, **kw)


def test_collector_stores_injuries_and_survives_errors():
    snap = collect(_bdl())
    assert snap.injuries == {"player one": {"status": "Out", "description": "caviglia", "return_date": None}}
    assert any(p["name"] == "Infortuni" and p["status"] == "ok" for p in snap.providers)
    snap2 = collect(_bdl(fail=True))
    assert snap2.injuries is None and snap2.quotes                                  # l'aggiornamento non si ferma
    assert any(p["name"] == "Infortuni" and p["status"] == "errore" for p in snap2.providers)


def test_injury_status_helper():
    class Bun:
        injuries = {"a b": {"status": "Out"}}
    assert injury_status(Bun, "a b") == "Out" and injury_status(Bun, "x") == ""
    class Empty:
        injuries = None
    assert injury_status(Empty, "a b") == ""


NOW_PD = pd.Timestamp("2029-12-31", tz="UTC").to_pydatetime()


def test_builder_excludes_out_and_labels_doubtful():
    rows = pd.DataFrame([make_rows(prop_id="a", infortunio="Out"), make_rows(prop_id="b", infortunio="Questionable"), make_rows(prop_id="c", infortunio="")])
    legs, funnel = B.eligible_legs(rows, NOW_PD)
    assert sorted(l.prop_id for l in legs) == ["b", "c"]
    assert next(l for l in legs if l.prop_id == "b").infortunio == "Questionable"
    assert dict(funnel)["Giocatore non segnalato «Out»"] == 2


def test_builder_works_without_injury_column():
    legs, _ = B.eligible_legs(pd.DataFrame([make_rows()]), NOW_PD)
    assert len(legs) == 1 and legs[0].infortunio == ""


def _best_rows(**over):
    base = dict(make_rows(), roi=.1, roi_lo=.05, quota_equa=1.9, p_loss=.5, edge=.1, p_pareggio=.45, ev=1.0, mercato="Punti", lato="Over")
    base.update(over)
    return base


def test_best_picks_exclude_out_and_flag_doubtful():
    rows = pd.DataFrame([_best_rows(prop_id="a", key="a", infortunio="Out"), _best_rows(prop_id="b", key="b", infortunio="Day-To-Day"),
                         _best_rows(prop_id="c", key="c", infortunio="")])
    picks = build_picks(rows, 10.0, fresh_only=False)
    assert sorted(picks["prop_id"]) == ["b", "c"]
    assert "segnalato «Day-To-Day»" in picks[picks["prop_id"] == "b"]["nota_dato"].iloc[0]
    assert "segnalato" not in picks[picks["prop_id"] == "c"]["nota_dato"].iloc[0]
