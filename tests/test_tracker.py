"""Registro delle giocate, chiusura con il risultato reale, taratura e giocate per partita. Dati SINTETICI."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from nba_vf import storage
from nba_vf.models import PlayerGameLog
from nba_vf.ui import tracker as K
from nba_vf.ui.best import build_picks, by_game, for_game
from nba_vf.ui.board import BoardParams, annotate, build_board
from nba_vf.ui.live import bundle_from_snapshot, settle_now
from nba_vf.ui.snapshot import save_snapshot

from test_best_picks import rows_for
from test_history_recovery import GAME_START, NOW, FakeBdl, build_world, collect, make_logs

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))
    for k in ("BALLDONTLIE_API_KEY", "THE_ODDS_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)
    storage.set_backend(None)
    K.reset_cache()
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    storage.set_backend(None)
    K.reset_cache()


def rec(i=1, *, side="over", line=20.5, quota=2.0, p=0.6, push=0.0, market="points", start=None, gday="2026-10-26", player_id=7):
    start = start or GAME_START
    return {"id": f"{gday}|p{i}|{market}|{line:g}|{side}", "logged_at": (start - timedelta(hours=3)).isoformat(), "gday": gday,
            "key": f"p{i}", "giocatore": f"Giocatore {i}", "squadra": "LAL", "evento": "BOS @ LAL", "inizio": start.isoformat(),
            "market": market, "mercato": "Punti", "linea": line, "side": side, "lato": "Over" if side == "over" else "Under",
            "bookmaker": "BookA", "quota": quota, "p_win": p, "p_push": push, "roi": p * quota + push - 1, "roi_lo": 0.01,
            "n_partite": 30, "affidabilita": "Alta", "player_id": player_id, "fonte": "balldontlie"}


def glog(pid=7, day=date(2026, 10, 26), pts=25, minutes=30.0, reb=5, ast=5):
    return PlayerGameLog("balldontlie", pid, "X", f"g{day}", day, 2026, "LAL", "BOS", True, minutes, pts, reb, ast, 2, 5, 8, 15, 4, 5)


AFTER = GAME_START + timedelta(hours=6)


# ---- esito --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("side,line,value,expected", [
    ("over", 20.5, 25, "win"), ("over", 20.5, 20, "loss"), ("under", 20.5, 20, "win"), ("under", 20.5, 25, "loss"),
    ("over", 20.0, 20, "push"), ("under", 20.0, 20, "push"), ("over", 20.0, 21, "win"), ("under", 20.0, 21, "loss")])
def test_grade(side, line, value, expected):
    assert K.grade(side, line, value) == expected


def test_settle_from_downloaded_logs_without_network_and_is_saved_once():
    r = rec(1, quota=2.0, p=0.6)
    K._append(K.STREAM_PICKS, r)
    results = K.load_results()
    rep = K.settle_pending(K.load_picks(), results, {"p1": [glog(pts=25)]}, None, AFTER)
    assert rep.settled == 1 and results[r["id"]]["outcome"] == "win" and results[r["id"]]["value"] == 25
    # riletto da disco, non si richiude e non si duplica
    assert len(K.load_results()) == 1
    rep2 = K.settle_pending(K.load_picks(), K.load_results(), {"p1": [glog(pts=10)]}, None, AFTER)
    assert rep2.settled == 0 and len(K.load_results()) == 1


def test_game_not_finished_stays_pending_and_never_fetches():
    K._append(K.STREAM_PICKS, rec(1))
    calls = []
    rep = K.settle_pending(K.load_picks(), {}, {}, lambda pid, d: calls.append(pid) or [], GAME_START + timedelta(hours=1))
    assert rep.settled == 0 and rep.waiting == 1 and not calls


def test_missing_log_is_fetched_once_per_player_and_day():
    K._append(K.STREAM_PICKS, rec(1, market="points"))
    K._append(K.STREAM_PICKS, rec(1, market="rebounds", line=5.5))
    calls = []

    def fetch(pid, day):
        calls.append((pid, day))
        return [glog(pts=30, reb=9)]

    results = K.load_results()
    rep = K.settle_pending(K.load_picks(), results, {}, fetch, AFTER)
    assert rep.settled == 2 and len(calls) == 1 and calls[0] == (7, date(2026, 10, 26))
    assert {v["value"] for v in results.values()} == {30, 9}


def test_player_who_did_not_play_is_void_and_excluded_from_stats():
    K._append(K.STREAM_PICKS, rec(1))
    results = K.load_results()
    K.settle_pending(K.load_picks(), results, {"p1": [glog(pts=0, minutes=0.0)]}, None, AFTER)
    df = K.track_frame(K.load_picks(), results)
    assert df.loc[0, "esito"] == K.VOID and K.summarize(df).n_settled == 0


def test_no_row_is_void_only_after_two_days_and_only_after_a_successful_fetch():
    K._append(K.STREAM_PICKS, rec(1))
    # subito dopo la partita: la statistica potrebbe solo essere in ritardo -> resta in attesa
    r1 = K.settle_pending(K.load_picks(), {}, {}, lambda p, d: [], AFTER)
    assert r1.void == 0 and r1.waiting == 1
    # fetch fallito (errore): mai annullata
    def boom(p, d):
        raise RuntimeError("rete")

    r2 = K.settle_pending(K.load_picks(), {}, {}, boom, GAME_START + timedelta(days=5))
    assert r2.void == 0 and r2.errors == 1
    # risposta valida ma vuota dopo 2 giorni: annullata
    res = {}
    r3 = K.settle_pending(K.load_picks(), res, {}, lambda p, d: [], GAME_START + timedelta(days=5))
    assert r3.void == 1 and list(res.values())[0]["status"] == "void"


def test_fetch_limit_leaves_the_rest_waiting():
    for i in range(5):
        K._append(K.STREAM_PICKS, rec(i, player_id=100 + i))
    rep = K.settle_pending(K.load_picks(), {}, {}, lambda p, d: [glog(pid=p)], AFTER, max_fetch=2)
    assert rep.settled == 2 and rep.waiting == 3


# ---- registro: registrazione ---------------------------------------------------------------------------
def generous_rows():
    rows, bundle = rows_for(lines=17.5)
    rows = rows.copy()
    rows.loc[rows["lato"] == "Over", "quota"] = 3.5
    rows["roi"] = rows["p_win"] * rows["quota"] + rows["p_push"] - 1
    rows["roi_lo"] = rows["roi"] - 0.05
    return rows, bundle


BEFORE = GAME_START - timedelta(hours=2)


def test_record_picks_only_value_future_fresh_live_and_only_once():
    rows, bundle = generous_rows()
    rows = rows.assign(obsoleta=False)
    n = K.record_picks(rows, bundle, BEFORE)
    assert n > 0
    saved = K.load_picks()
    assert len(saved) == n and all(r["roi"] > 0 and r["p_win"] > 0 for r in saved)
    assert len({r["id"] for r in saved}) == n
    assert all(r["player_id"] for r in saved)  # servira' per chiudere la giocata
    assert K.record_picks(rows, bundle, BEFORE + timedelta(minutes=5)) == 0  # niente doppioni
    # anche dopo un riavvio (cache degli ID persa) il file evita i duplicati
    K.reset_cache()
    assert K.record_picks(rows, bundle, BEFORE + timedelta(minutes=10)) == 0 and len(K.load_picks()) == n


def test_first_seen_quote_is_kept_not_rewritten():
    rows, bundle = generous_rows()
    rows = rows.assign(obsoleta=False)
    K.record_picks(rows, bundle, BEFORE)
    before = {r["id"]: r["quota"] for r in K.load_picks()}
    better = rows.assign(quota=rows["quota"] * 1.5, roi=rows["roi"] + 1)  # stesse giocate, quota migliore
    K.record_picks(better, bundle, BEFORE + timedelta(minutes=30))
    after = {r["id"]: r["quota"] for r in K.load_picks()}
    assert all(after[i] == q for i, q in before.items())  # quelle gia' registrate restano con la quota vista per prima


def test_started_games_stale_and_manual_quotes_are_not_recorded():
    rows, bundle = generous_rows()
    rows = rows.assign(obsoleta=False)
    assert K.record_picks(rows, bundle, GAME_START + timedelta(minutes=1)) == 0  # partita gia' iniziata
    assert K.record_picks(rows.assign(obsoleta=True), bundle, BEFORE) == 0
    assert K.record_picks(rows.assign(manuale=True), bundle, BEFORE) == 0


def test_demo_is_never_recorded():
    from nba_vf.ui.demo import build_demo_bundle

    rows, _ = generous_rows()
    assert K.record_picks(rows.assign(obsoleta=False), build_demo_bundle(2026), BEFORE) == 0
    assert K.load_picks() == []


def test_remote_backend_gets_a_copy_and_failures_do_not_break():
    be = storage.MemoryBackend()
    storage.set_backend(be)
    K._append(K.STREAM_PICKS, rec(1))
    assert len(be.read(K.STREAM_PICKS)) == 1
    # il file locale sparisce (riavvio dell'hosting): il registro si ricostruisce dall'archivio remoto
    (Path(K._path(K.STREAM_PICKS))).unlink()
    assert len(K.load_picks()) == 1

    class Broken(storage.MemoryBackend):
        def append(self, *a):
            raise RuntimeError("giu")

        def read(self, *a):
            raise RuntimeError("giu")

    storage.set_backend(Broken())
    K._append(K.STREAM_PICKS, rec(2))  # non solleva
    assert [r["key"] for r in K.load_picks()] == ["p2"]


# ---- statistiche e taratura ---------------------------------------------------------------------------
def make_df(outcomes, p=0.6, quota=1.8):
    records, results = [], {}
    for i, o in enumerate(outcomes):
        r = rec(i, p=p, quota=quota)
        records.append(r)
        if o == "pending":
            continue
        results[r["id"]] = {"id": r["id"], "status": "void" if o == "void" else "final", "value": 1, "outcome": None if o == "void" else o}
    return K.track_frame(records, results, None, AFTER)


def test_summary_counts_and_profit():
    df = make_df(["win", "win", "loss", "push", "void", "pending"], p=0.6, quota=2.0)
    sm = K.summarize(df)
    assert (sm.n_logged, sm.n_settled, sm.n_pending, sm.n_void) == (6, 4, 1, 1)
    assert (sm.wins, sm.losses, sm.pushes) == (2, 1, 1)
    assert sm.profit_units == pytest.approx(1 + 1 - 1 + 0)
    assert sm.expected_units == pytest.approx(4 * (0.6 * 2.0 - 1))
    assert sm.expected_wins == pytest.approx(2.4)


def test_verdict_is_neutral_below_30_even_with_great_results():
    sm = K.summarize(make_df(["win"] * 10))
    level, text = K.verdict(sm)
    assert level == "info" and "Troppo presto" in text


def test_verdict_flags_overoptimistic_model_with_enough_data():
    sm = K.summarize(make_df(["loss"] * 45 + ["win"] * 5, p=0.6, quota=1.8))
    level, text = K.verdict(sm)
    assert level == "warning" and "PEGGIORE" in text


def test_verdict_consistent_when_results_match_prediction():
    sm = K.summarize(make_df(["win"] * 30 + ["loss"] * 20, p=0.6, quota=1.8))
    level, text = K.verdict(sm)
    assert level == "success" and "coerenti" in text
    assert "garanzi" in text.lower()  # non promette nulla


def test_calibration_table_bins_and_interval():
    df = make_df(["win"] * 6 + ["loss"] * 4, p=0.62)
    tab = K.calibration_table(df)
    assert len(tab) == 1 and tab.loc[0, "gruppo"] == "60–70%" and tab.loc[0, "giocate"] == 10
    assert tab.loc[0, "realizzato"] == pytest.approx(60.0) and tab.loc[0, "previsto"] == pytest.approx(62.0)
    assert tab.loc[0, "da"] < 60 < tab.loc[0, "a"]
    assert K.calibration_table(make_df(["pending"])).empty
    assert set(K.calibration_table(df, "affidabilita")["gruppo"]) == {"Alta"}


def test_track_frame_settles_in_memory_from_downloaded_logs_without_saving():
    r = rec(1)
    df = K.track_frame([r], {}, {"p1": [glog(pts=25)]}, AFTER)
    assert df.loc[0, "esito"] == K.WIN and K.load_results() == {}
    assert K.track_frame([r], {}, {"p1": [glog(pts=25)]}, GAME_START - timedelta(hours=1)).loc[0, "esito"] == K.PENDING


def test_empty_frame_is_safe():
    df = K.track_frame([], {})
    sm = K.summarize(df)
    assert sm.n_logged == 0 and K.verdict(sm)[0] == "info" and K.calibration_table(df).empty


# ---- chiusura durante l'aggiornamento live -------------------------------------------------------------
def test_refresh_settles_previous_picks_and_failures_do_not_break_refresh(monkeypatch):
    from nba_vf.ui import live

    K._append(K.STREAM_PICKS, rec(1))
    snap_logs = {"p1": [glog(pts=12)]}

    class Bdl:
        def get_player_game_logs(self, *a, **k):
            raise RuntimeError("giu")

    rep = live._settle(Bdl(), snap_logs, AFTER)
    assert rep is not None and rep.settled == 1 and K.load_results()
    # se anche la lettura del registro esplode, l'aggiornamento prosegue
    monkeypatch.setattr(K, "load_picks", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert live._settle(Bdl(), {}, AFTER) is None


# ---- giocate per partita ------------------------------------------------------------------------------
def test_by_game_and_for_game():
    rows, _ = generous_rows()
    picks = build_picks(rows.assign(obsoleta=False), 10.0, include_low=True, fresh_only=False)
    assert len(picks)
    g = by_game(picks)
    assert g["giocate"].sum() == len(picks) and len(g) == picks.groupby(["partita", "inizio"]).ngroups
    first = g.iloc[0]
    sub = for_game(picks, first["partita"], first["inizio"])
    assert len(sub) == first["giocate"] and sub["guadagno"].is_monotonic_decreasing
    assert first["migliore"].startswith(sub.iloc[0]["giocatore"])
    assert by_game(pd.DataFrame()).empty


# ---- pagine --------------------------------------------------------------------------------------------
def fresh_snapshot():
    from nba_vf.models import utc_now

    logs, names = build_world(n_full=8, n_early=2, n_none=2)
    snap = collect(FakeBdl(logs, names))
    now = utc_now()
    start = now + timedelta(hours=3)
    snap.quotes = [replace(q, odds_decimal=3.6 if q.side.value == "over" else q.odds_decimal, fetched_at=now, quote_time=now,
                           commence_time=start) for q in snap.quotes]
    snap.events = {k: dict(v, commence_time=start.isoformat()) for k, v in snap.events.items()}
    save_snapshot(snap)


def test_tracker_page_is_gentle_in_demo_and_empty_registry():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.switch_page("views/track.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Live" in i.value for i in at.info)  # in demo non si registra nulla


def test_live_page_records_picks_and_registry_page_shows_them():
    fresh_snapshot()
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.switch_page("views/best.py").run()
    assert not at.exception, [e.value for e in at.exception]
    saved = K.load_picks()
    assert saved, "aprendo Migliori giocate in Live le giocate con valore devono finire nel registro"
    assert all(r["giocatore"].startswith("Giocatore") for r in saved)
    tabs = [t.label for t in at.tabs]
    assert "Per partita" in tabs and "Più redditizie" in tabs
    at.switch_page("views/track.py").run()
    assert not at.exception, [e.value for e in at.exception]
    shown = "\n".join(m.value for m in at.markdown)
    assert any("Troppo presto" in i.value for i in at.info) or "Troppo presto" in shown
    assert at.dataframe  # il registro e' visibile
