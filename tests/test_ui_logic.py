"""Logica dell'interfaccia (senza Streamlit): demo, live, tabellone, adattatori finti, snapshot."""

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pytest

from nba_vf.http import ApiError, TierError
from nba_vf.model.synthetic import TruePlayer, generate_player
from nba_vf.models import Market, OddsQuote, Side, utc_now
from nba_vf.ui import charts, reasoning
from nba_vf.ui.board import BoardParams, annotate, build_board
from nba_vf.ui.demo import DEMO_NOTICE, build_demo_bundle
from nba_vf.ui.labels import normalize_name
from nba_vf.ui.live import (EMPTY_LIVE_MESSAGE, CollectOptions, LiveCollector, add_manual_quotes, bundle_from_snapshot,
                            load_live_bundle)
from nba_vf.ui.snapshot import Snapshot, load_snapshot, save_snapshot


@pytest.fixture(scope="module")
def demo():
    return build_demo_bundle(seed=7)


@pytest.fixture(scope="module")
def board(demo):
    return build_board(demo, BoardParams(n_boot=20, n_sims_pra=10_000, n_sims_pra_boot=2_000))


@pytest.fixture(autouse=True)
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))


# ---- demo --------------------------------------------------------------------------------------
def test_demo_is_clearly_synthetic(demo):
    assert demo.is_demo and "SINTETICI" in demo.source_label and demo.messages[0][1] == DEMO_NOTICE
    assert {q.source for q in demo.quotes} == {"demo"} and {q.provider for q in demo.quotes} == {"demo"}
    assert all(l.source == "SINTETICO" for logs in demo.logs.values() for l in logs)
    assert all(n.startswith("Giocatore Demo") for n in (p.name for p in demo.players.values()))
    assert all(b.startswith("demo_") for b in demo.bookmakers)


def test_demo_summary_numbers(demo):
    assert demo.n_games_analyzed > 100 and demo.n_props > 50 and demo.n_bookmakers == 5
    g = demo.games_df()
    assert len(g) == demo.n_games_analyzed and {"data", "squadre", "giocatori"} <= set(g.columns)
    assert demo.updated_at is not None


def test_demo_contains_data_quality_cases(demo):
    sizes = sorted(len(v) for v in demo.logs.values())
    assert sizes[0] < 10 and any(10 <= n < 25 for n in sizes)  # giocatori non stimabili e a campione ridotto


# ---- tabellone ---------------------------------------------------------------------------------
def test_board_rows_and_pricing(board, demo):
    df = board.rows
    assert len(df) == len(demo.quotes) and board.prior_source == "pool"
    p = df[df.priced]
    assert p[["p_win", "p_push", "p_loss"]].apply(lambda c: c.between(0, 1)).all().all()
    assert np.allclose(p.p_win + p.p_push + p.p_loss, 1.0)
    assert np.allclose(p.roi, p.p_win * p.quota + p.p_push - 1)
    assert np.allclose(p.quota_equa, (1 - p.p_push) / p.p_win)
    assert (p.roi_lo <= p.roi_hi).all()
    assert df[~df.priced].motivo.str.len().min() > 0


def test_board_flags_insufficient_data(board):
    assert any("meno partite del minimo" in t for _, t in board.warnings)
    assert any("campione ridotto" in t for _, t in board.warnings)
    assert len(board.skipped) >= 1


def test_annotate_stake_and_staleness(board):
    a = annotate(board.rows, utc_now(), 30, 25.0)
    p = a[a.priced]
    assert np.allclose(p.ev, 25.0 * p.roi)
    assert a.obsoleta.any() and (~a.obsoleta).any()  # una quota demo e' volutamente obsoleta
    assert a[a.obsoleta].avvisi.str.contains("obsoleta").all()
    assert set(a.classe) <= {"positivo_robusto", "positivo_incerto", "non_positivo", "non_prezzata"}
    assert (a[~a.priced].avvisi.str.startswith("Non prezzata")).all()


def test_consensus_groups(board):
    g = board.groups
    assert len(g) > 50 and (g.n_book >= 1).all()
    assert (g.quota_over_max >= g.quota_over_media - 1e-9).all()
    both = g.dropna(subset=["p_over_mercato"])
    assert both.p_over_mercato.between(0, 1).all() and (both.margine_medio > 0).all()


def test_normalize_name():
    assert normalize_name("Nikola Jokić") == normalize_name("nikola jokic") == "nikola jokic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    assert normalize_name("De'Aaron Fox") == "de aaron fox"


# ---- live: nessun dato inventato -----------------------------------------------------------------
def test_live_without_snapshot_is_empty_with_clear_message():
    b = load_live_bundle()
    assert b.mode == "live" and b.status == "vuoto" and b.is_empty
    assert not b.quotes and not b.logs and not b.players and not b.events
    assert b.messages == [("warning", EMPTY_LIVE_MESSAGE)]
    assert "non sostituisce mai dati reali" in EMPTY_LIVE_MESSAGE


def test_live_bundle_never_contains_demo_data(demo):
    b = load_live_bundle()
    assert all(not q.source == "demo" for q in b.quotes)


def test_corrupted_snapshot_reports_error(tmp_path):
    (tmp_path / "live_snapshot.json").write_text("{non json")
    b = load_live_bundle()
    assert b.is_empty and b.messages[0][0] == "error"


def test_snapshot_roundtrip(tmp_path):
    logs = generate_player(TruePlayer(), 15, 1)
    q = OddsQuote("the_odds_api", "book", "e1", "Test Player", None, Market.POINTS, 17.5, Side.OVER, 1.9, utc_now(),
                  utc_now(), "AAA", "BBB", utc_now(), False, "api")
    snap = Snapshot(fetched_at=utc_now(), quotes=[q], logs={"test player": logs}, players={"test player": {"name": "Test Player", "team": "AAA"}},
                    providers=[{"name": "x", "status": "ok", "detail": "", "n_quotes": 1}], logs_fetched_at={"test player": utc_now()})
    save_snapshot(snap)
    back = load_snapshot()
    assert back.quotes == [q] and back.logs["test player"] == logs
    b = bundle_from_snapshot(back)
    assert b.mode == "live" and b.n_props == 1 and b.n_games_analyzed == 15 and b.status == "ok"


# ---- collector con adattatori finti ------------------------------------------------------------------
@dataclass
class _Quota:
    remaining: int | None = 99


@dataclass
class _Parsed:
    quotes: list = field(default_factory=list)
    skipped_markets: set = field(default_factory=set)


@dataclass
class _Res:
    parsed: _Parsed
    quota: _Quota = field(default_factory=_Quota)


class FakeOdds:
    def __init__(self, now, fail=False):
        self.now, self.fail = now, fail

    def list_events(self):
        if self.fail:
            raise ApiError("the_odds_api: HTTP 500 apiKey=SECRETKEY123")
        return [{"id": "ev1", "home_team": "Home", "away_team": "Away", "commence_time": (self.now + timedelta(hours=5)).isoformat()}]

    def get_event_props(self, event_id):
        qs = [OddsQuote("the_odds_api", "bk", event_id, "Test Player", None, Market.POINTS, 17.5, s, o, self.now, self.now,
                        "Home", "Away", self.now + timedelta(hours=5), False) for s, o in ((Side.OVER, 1.9), (Side.UNDER, 1.9))]
        return _Res(_Parsed(qs))


class FakeNba:
    def __init__(self, logs):
        self.logs, self.calls = logs, 0

    def find_player(self, name):
        return [{"id": 7, "full_name": "Test Player"}] if normalize_name(name) == "test player" else []

    def get_player_game_logs(self, player_id, season):
        self.calls += 1
        return list(self.logs)


def test_collector_without_keys_reports_everything_missing():
    snap = LiveCollector(now=utc_now).collect(CollectOptions())
    assert snap.quotes == [] and snap.logs == {}
    st = {p["name"]: p["status"] for p in snap.providers}
    assert st["The Odds API"] == "non_configurato" and st["BALLDONTLIE (props)"] == "non_configurato"
    b = bundle_from_snapshot(snap)
    assert b.is_empty and any("non contiene quote" in t for _, t in b.messages)


def test_collector_with_fake_providers():
    now = utc_now()
    logs = generate_player(TruePlayer(player_id=7, name="Test Player"), 40, 2)
    nba = FakeNba(logs)
    snap = LiveCollector(odds=FakeOdds(now), nba=nba, now=lambda: now).collect(CollectOptions())
    assert len(snap.quotes) == 2 and "test player" in snap.logs and snap.events["ev1"]["home"] == "Home"
    assert {p["name"]: p["status"] for p in snap.providers}["The Odds API"] == "ok"
    # secondo prelievo: lo storico recente viene riusato, non richiesto di nuovo
    calls = nba.calls
    LiveCollector(odds=FakeOdds(now), nba=nba, now=lambda: now).collect(CollectOptions(), previous=snap)
    assert nba.calls == calls


def test_collector_errors_are_reported_and_secrets_redacted():
    now = utc_now()
    snap = LiveCollector(odds=FakeOdds(now, fail=True), secrets=["SECRETKEY123"], now=lambda: now).collect(CollectOptions())
    detail = next(p["detail"] for p in snap.providers if p["name"] == "The Odds API")
    assert next(p["status"] for p in snap.providers if p["name"] == "The Odds API") == "errore"
    assert "SECRETKEY123" not in detail and snap.quotes == []


def test_player_without_history_is_marked_missing_not_invented():
    now = utc_now()
    snap = LiveCollector(odds=FakeOdds(now), nba=FakeNba([]), now=lambda: now).collect(CollectOptions())
    assert snap.logs == {}  # nessuno storico inventato
    assert snap.issues["test player"]
    b = bundle_from_snapshot(snap)
    board = build_board(b, BoardParams(n_boot=0, n_sims_pra=2000))
    assert not board.rows.priced.any() and "storico assente" in board.rows.motivo.iloc[0]


def test_tier_error_on_bdl_stats_is_explained():
    class Bdl:
        def get_games(self, d):
            return []

        def get_player_game_logs(self, *a, **k):
            raise TierError("piano ALL-STAR richiesto")

    now = utc_now()
    snap = LiveCollector(bdl=Bdl(), odds=FakeOdds(now), nba=FakeNba([]), now=lambda: now).collect(CollectOptions(use_balldontlie_props=False))
    assert snap.quotes and not snap.logs


def test_manual_quotes_persist_across_refresh():
    now = utc_now()
    q = OddsQuote("csv", "snai", "csv:1", "Test Player", None, Market.POINTS, 18.5, Side.OVER, 1.85, now, now, source="csv")
    add_manual_quotes([q])
    b = load_live_bundle()
    assert b.n_bookmakers == 1 and b.quotes[0].source == "csv"
    snap = LiveCollector(odds=FakeOdds(now), nba=FakeNba([]), now=lambda: now).collect(CollectOptions(), previous=load_snapshot())
    assert any(x.source == "csv" for x in snap.quotes) and len(snap.quotes) == 3


# ---- grafici e testi ---------------------------------------------------------------------------------
def test_charts_build(demo, board):
    logs = demo.logs[next(iter(demo.logs))]
    for m in Market:
        assert charts.fig_last_games(logs, m, 10.5).data
    assert charts.fig_last_games([], Market.POINTS, None).layout.annotations
    pmf = np.ones(30) / 30
    assert charts.fig_distribution(pmf, 10, "over").data and charts.fig_distribution(pmf, 10.5, "under", np.tile(pmf, (10, 1))).data
    assert charts.fig_ev_scatter(annotate(board.rows, None, 30, 10)).data
    assert charts.fig_ev_hist(board.rows).data and charts.fig_ev_scatter(board.rows.head(0)).layout.annotations


def test_reasoning_wording_and_missing_list(board, demo):
    from nba_vf.model.engine import GameContext, ValueEngine

    key = next(k for k in board.predictions)
    pred = board.predictions[key]
    mp = pred.markets[Market.POINTS]
    v = mp.price(mp.median + 0.5, Side.OVER, 1.9)
    lines = reasoning.explain(pred, Market.POINTS, mp.median + 0.5, "over", v, season_avg=10.0, last10_avg=11.0, hit_all=0.5, hit_20=0.4, market_p_over=0.48)
    text = " ".join(lines).lower()
    assert "garantisce" in text  # il disclaimer c'e' e dice che nulla e' garantito
    assert not any(w in text for w in ("sicura", "sicuro", "certamente"))
    items = reasoning.missing_data(pred, n_logs=40, skipped_reason=None, issues=[], market=Market.POINTS, n_books=2, n_stale=1,
                                   has_opponent=True, mode="demo")
    assert any("Assenze" in t for _, t in items) and any("Quote storiche" in t for _, t in items) and any("sintetici" in t for _, t in items)
    assert reasoning.missing_data(None, n_logs=0, skipped_reason=None, issues=[], market=Market.POINTS, n_books=0, n_stale=0,
                                  has_opponent=False, mode="live")[0][0] == "missing"
