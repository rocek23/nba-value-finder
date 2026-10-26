"""Recupero dello storico per tutti i giocatori delle quote, copertura e assenza di probabilita' inventate.

Scenario reale che ha causato il problema (inizio stagione 2026-27): decine di giocatori con quote, pochissime
partite nella stagione in corso, storico richiesto solo per 30 giocatori e solo per la stagione corrente.
I dati sono SINTETICI e i provider sono finti: qui si verifica la logica, non le API.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from nba_vf.adapters.balldontlie import parse_stat_row
from nba_vf.model.synthetic import TruePlayer, generate_player
from nba_vf.models import Market, OddsQuote, Side
from nba_vf.ui.board import BoardParams, build_board, usable_logs
from nba_vf.ui.coverage import COMPLETE, INSUFFICIENT, NONE, PARTIAL, build_coverage
from nba_vf.ui.live import CollectOptions, LiveCollector, bundle_from_snapshot
from nba_vf.ui.snapshot import load_snapshot, save_snapshot

NOW = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)  # lunedi' 26 ottobre 2026, 14:00 a New York
GAME_START = NOW + timedelta(hours=5)


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))


def make_logs(pid: int, name: str, dates: list[date], season: int, source: str = "balldontlie") -> list:
    base = generate_player(TruePlayer(player_id=pid, name=name), len(dates), seed=pid * 7 + season)
    return [replace(l, game_date=d, game_id=f"{pid}-{d.isoformat()}", season=season, source=source, team="AAA") for l, d in zip(base, dates)]


def spaced(end: date, n: int, step: int = 2) -> list[date]:
    return sorted(end - timedelta(days=step * i) for i in range(n))


class FakeBdl:
    """Imita BALLDONTLIE: props di una partita e storico per ID ufficiale, filtrato per stagioni e start_date."""

    def __init__(self, logs_by_id: dict[int, list], names: dict[int, str], lines: float = 20.5):
        self.logs_by_id, self.names, self.lines = logs_by_id, names, lines
        self.stat_calls: list[tuple] = []
        self.request_count = 0
        self.quote_day = NOW.date()  # giorno in cui la partita risulta in programma

    def get_games(self, day):
        if day != self.quote_day:
            return []
        return [{"id": 1, "datetime": GAME_START.isoformat(), "home_team": {"abbreviation": "AAA"}, "visitor_team": {"abbreviation": "BBB"}}]

    def get_player_props(self, game_id):
        qs = []
        for pid, name in self.names.items():
            for market, line in ((Market.POINTS, self.lines), (Market.REBOUNDS, 5.5)):
                for side, odds in ((Side.OVER, 1.9), (Side.UNDER, 1.9)):
                    qs.append(OddsQuote("balldontlie", "bookA", str(game_id), name, pid, market, line, side, odds, NOW, NOW,
                                        "AAA", "BBB", GAME_START, False))

        class R:
            quotes = qs

        return R()

    def resolve_player_names(self, ids):
        return {i: self.names[i] for i in ids}

    def get_player_game_logs(self, player_id, seasons, season_type="regular", start_date=None):
        seasons = [seasons] if isinstance(seasons, int) else list(seasons)
        self.stat_calls.append((player_id, tuple(seasons), start_date))
        self.request_count += 1
        return [l for l in self.logs_by_id.get(player_id, []) if l.season in seasons and (start_date is None or l.game_date >= start_date)]


def collect(bdl, previous=None, now=NOW, **opts):
    return LiveCollector(bdl=bdl, now=lambda: now).collect(CollectOptions(use_odds_api=False, **opts), previous=previous)


def build_world(n_full=12, n_early=18, n_none=15):
    """Giocatori con 30 partite 2025-26 + 3 del 2026-27, giocatori con sole 4 partite 2026-27, giocatori senza partite."""
    logs, names, pid = {}, {}, 100
    for _ in range(n_full):
        pid += 1
        names[pid] = f"Giocatore Pieno {pid}"
        logs[pid] = make_logs(pid, names[pid], spaced(date(2026, 4, 10), 30), 2025) + make_logs(pid, names[pid], [date(2026, 10, 22), date(2026, 10, 23), date(2026, 10, 25)], 2026)
    for _ in range(n_early):
        pid += 1
        names[pid] = f"Giocatore Nuovo {pid}"
        logs[pid] = make_logs(pid, names[pid], [date(2026, 10, 21), date(2026, 10, 22), date(2026, 10, 23), date(2026, 10, 25)], 2026)
    for _ in range(n_none):
        pid += 1
        names[pid] = f"Giocatore Assente {pid}"
        logs[pid] = []
    return logs, names


# ---- 1. storico per TUTTI i giocatori delle quote (non solo 30) ----------------------------------------
def test_history_requested_for_every_player_in_quotes_not_only_30():
    logs, names = build_world(n_full=20, n_early=25, n_none=10)  # 55 giocatori nelle quote
    bdl = FakeBdl(logs, names)
    snap = collect(bdl)
    assert snap.retrieval["players_in_quotes"] == 55 and snap.retrieval["players_requested"] == 55
    assert {c[0] for c in bdl.stat_calls} == set(names)  # ogni ID ufficiale e' stato richiesto
    assert not any("limite impostato" in m[1] for m in snap.messages)


def test_explicit_cap_is_declared_not_silent():
    logs, names = build_world(n_full=5, n_early=5, n_none=0)
    snap = collect(FakeBdl(logs, names), max_players=6)
    assert snap.retrieval["players_requested"] == 6
    assert any("limite impostato" in m[1] for m in snap.messages)


# ---- 2. due stagioni: a inizio stagione lo storico precedente rende valutabile il giocatore --------------
def test_both_seasons_requested_and_early_season_player_becomes_evaluable():
    logs, names = build_world(n_full=3, n_early=0, n_none=0)
    bdl = FakeBdl(logs, names)
    snap = collect(bdl)
    assert all(c[1] == (2026, 2025) for c in bdl.stat_calls)
    assert snap.retrieval["seasons"] == [2026, 2025]
    for k, v in snap.logs.items():
        assert len(v) == 33 and {l.season for l in v} == {2025, 2026}
    board = build_board(bundle_from_snapshot(snap), BoardParams(n_boot=0, n_sims_pra=2000))
    assert board.rows["priced"].any()  # prima della correzione: 0 quote prezzate


def test_only_current_season_would_not_be_enough_in_october():
    logs, names = build_world(n_full=0, n_early=3, n_none=0)  # 4 partite in totale
    snap = collect(FakeBdl(logs, names))
    board = build_board(bundle_from_snapshot(snap), BoardParams(n_boot=0, n_sims_pra=2000))
    assert not board.rows["priced"].any()
    assert all("storico insufficiente" in m for m in board.rows["motivo"])  # dichiarato, non inventato


# ---- 3. nessuna probabilita' inventata + matematica valida ---------------------------------------------
def test_unpriced_rows_have_no_numbers_and_priced_rows_are_valid():
    logs, names = build_world(n_full=6, n_early=6, n_none=6)
    snap = collect(FakeBdl(logs, names))
    board = build_board(bundle_from_snapshot(snap), BoardParams(n_boot=30, n_sims_pra=3000))
    rows = board.rows
    un, pr = rows[~rows["priced"]], rows[rows["priced"]]
    assert len(un) and len(pr)
    for c in ("p_win", "p_push", "p_loss", "quota_equa", "roi", "n_partite"):
        assert un[c].isna().all(), c  # niente probabilita' per chi non ha dati
    assert (pr["n_partite"] >= 10).all()
    assert np.allclose(pr["p_win"] + pr["p_push"] + pr["p_loss"], 1.0, atol=1e-9)
    assert np.allclose(pr["quota_equa"], (1 - pr["p_push"]) / pr["p_win"])
    assert np.allclose(pr["roi"], pr["p_win"] * pr["quota"] + pr["p_push"] - 1)
    assert ((pr["roi_lo"] <= pr["roi"] + 1e-9) & (pr["roi"] <= pr["roi_hi"] + 1e-9)).all()  # intervallo coerente
    # Over e Under dello stesso mercato sono complementari
    key = pr[pr["mercato"] == pr["mercato"].iloc[0]]
    g = key.groupby(["giocatore", "mercato", "linea"])
    for _, grp in g:
        if len(grp) == 2:
            o, u = grp[grp["lato"] == "Over"].iloc[0], grp[grp["lato"] == "Under"].iloc[0]
            assert o["p_win"] == pytest.approx(u["p_loss"]) and o["p_push"] == pytest.approx(u["p_push"])


# ---- 4. nessuna partita futura o della stessa gara nella previsione ------------------------------------
def test_games_on_or_after_the_quoted_game_day_are_excluded():
    logs, names = build_world(n_full=1, n_early=0, n_none=0)
    pid = next(iter(names))
    same_day = make_logs(pid, names[pid], [date(2026, 10, 26)], 2026)  # la gara quotata stessa
    future = make_logs(pid, names[pid], [date(2026, 10, 28)], 2026)
    logs[pid] = logs[pid] + same_day + future
    bdl = FakeBdl(logs, names)
    snap = collect(bdl)
    assert max(l.game_date for l in snap.logs[next(iter(snap.logs))]) <= NOW.date()  # il futuro non entra nello snapshot
    bundle = bundle_from_snapshot(snap)
    key = next(iter(bundle.logs))
    q = next(q for q in bundle.quotes)
    ul = usable_logs(bundle, key, q)
    assert all(l.game_date < date(2026, 10, 26) for l in ul) and len(ul) == 33


def test_unfinished_and_playoff_rows_are_not_used():
    row = {"player": {"id": 1, "first_name": "A", "last_name": "B"}, "team": {"id": 1, "abbreviation": "AAA"}, "min": "30", "pts": 10,
           "game": {"id": 5, "date": "2026-10-25", "season": 2026, "home_team_id": 1, "visitor_team_id": 2, "status": "Final"}}
    assert parse_stat_row(row) is not None
    live = {**row, "game": {**row["game"], "status": "Q3 4:12"}}
    assert parse_stat_row(live) is None  # partita in corso: statistiche parziali
    assert parse_stat_row({**row, "game": {k: v for k, v in row["game"].items() if k != "status"}}) is not None
    assert parse_stat_row({**row, "game": {**row["game"], "status": "Final/OT"}}) is not None


# ---- 5. cache consapevole delle stagioni + aggiornamento incrementale -----------------------------------
def test_recent_cache_is_reused_without_new_requests():
    logs, names = build_world(n_full=4, n_early=0, n_none=0)
    bdl = FakeBdl(logs, names)
    first = collect(bdl)
    n = len(bdl.stat_calls)
    again = collect(bdl, previous=first, now=NOW + timedelta(hours=1))
    assert len(bdl.stat_calls) == n and again.retrieval["reused_from_cache"] == 4


def test_old_cache_with_wrong_seasons_does_not_block_the_fix():
    """Lo snapshot vecchio ha solo la stagione 2026 e viene considerato 'fresco': non deve impedire il nuovo scarico."""
    logs, names = build_world(n_full=3, n_early=0, n_none=0)
    old = collect(FakeBdl(logs, names), seasons_back=0)  # solo 2026, come il vecchio comportamento
    assert all(len(v) == 3 for v in old.logs.values())
    bdl = FakeBdl(logs, names)
    new = collect(bdl, previous=old, now=NOW + timedelta(hours=1))
    assert len(bdl.stat_calls) == 3 and all(len(v) == 33 for v in new.logs.values())


def test_incremental_download_after_12h_fetches_only_new_games_without_duplicates():
    logs, names = build_world(n_full=2, n_early=0, n_none=0)
    bdl = FakeBdl(logs, names)
    first = collect(bdl)
    pid = next(iter(names))
    logs[pid] = logs[pid] + make_logs(pid, names[pid], [date(2026, 10, 26)], 2026)  # nuova partita conclusa
    bdl.stat_calls.clear()
    bdl.quote_day = NOW.date() + timedelta(days=1)
    later = collect(bdl, previous=first, now=NOW + timedelta(hours=20))  # 27 ottobre
    assert all(c[2] is not None for c in bdl.stat_calls)  # start_date impostata
    key = next(k for k, v in later.players.items() if v["name"] == names[pid])
    ids = [l.game_id for l in later.logs[key]]
    assert len(ids) == len(set(ids)) == 34


def test_snapshot_roundtrip_keeps_seasons_and_retrieval():
    logs, names = build_world(n_full=2, n_early=1, n_none=1)
    snap = collect(FakeBdl(logs, names))
    save_snapshot(snap)
    back = load_snapshot()
    assert back.logs_seasons == snap.logs_seasons and back.retrieval == snap.retrieval
    assert bundle_from_snapshot(back).retrieval["seasons"] == [2026, 2025]


# ---- 6. limite di tempo, errori, ambiguita' -------------------------------------------------------------
def test_time_budget_leaves_players_pending_and_next_run_resumes():
    logs, names = build_world(n_full=4, n_early=0, n_none=0)
    bdl = FakeBdl(logs, names)
    stopped = collect(bdl, time_budget_s=-1)
    assert stopped.retrieval["complete"] is False and len(stopped.retrieval["pending"]) == 4 and not bdl.stat_calls
    assert any("non terminato" in m[1] for m in stopped.messages)
    assert all("in attesa" in " ".join(v) for v in stopped.issues.values())
    resumed = collect(bdl, previous=stopped, now=NOW + timedelta(minutes=5))
    assert resumed.retrieval["complete"] is True and len(resumed.logs) == 4


def test_api_error_keeps_previous_logs_and_does_not_mark_them_fresh():
    from nba_vf.http import ApiError

    logs, names = build_world(n_full=1, n_early=0, n_none=0)
    bdl = FakeBdl(logs, names)
    first = collect(bdl)

    def boom(*a, **k):
        raise ApiError("bdl: HTTP 500")

    bdl.get_player_game_logs = boom
    bdl.quote_day = NOW.date() + timedelta(days=1)
    later = collect(bdl, previous=first, now=NOW + timedelta(hours=20))
    key = next(iter(later.logs))
    assert len(later.logs[key]) == 33 and later.issues[key]
    assert later.logs_fetched_at[key] == first.logs_fetched_at[key]  # resta 'vecchio': sara' ritentato


def test_homonyms_are_not_guessed():
    logs, names = build_world(n_full=1, n_early=0, n_none=0)
    pid = next(iter(names))
    bdl = FakeBdl(logs, {**names, 999: names[pid]})  # due ID ufficiali con lo stesso nome
    snap = collect(bdl)
    assert not bdl.stat_calls
    assert any("nome ambiguo" in " ".join(v) for v in snap.issues.values())


def test_player_without_games_is_reported_not_filled_in():
    logs, names = build_world(n_full=0, n_early=0, n_none=2)
    snap = collect(FakeBdl(logs, names))
    assert snap.logs == {} and all("non ha partite" in " ".join(v) for v in snap.issues.values())


# ---- 7. copertura -------------------------------------------------------------------------------------
def test_coverage_counts_states_evaluable_and_priced():
    logs, names = build_world(n_full=5, n_early=4, n_none=3)
    # un giocatore parziale: 15 partite
    pid = max(names) + 1
    names[pid] = "Giocatore Parziale"
    logs[pid] = make_logs(pid, names[pid], spaced(date(2026, 10, 25), 15), 2026)
    snap = collect(FakeBdl(logs, names))
    bundle = bundle_from_snapshot(snap)
    board = build_board(bundle, BoardParams(n_boot=0, n_sims_pra=2000))
    cov = build_coverage(bundle, board)
    assert cov.state_counts == {COMPLETE: 5, PARTIAL: 1, INSUFFICIENT: 4, NONE: 3}
    assert cov.n_players == 13 and cov.n_quotes == 13 * 4
    assert cov.n_evaluable == 6 * 4 and cov.n_priced <= cov.n_evaluable and cov.n_priced > 0
    assert cov.n_priced == int(board.rows["priced"].sum())
    reasons = dict(zip(cov.reasons["motivo"], cov.reasons["quote"]))
    assert reasons["storico insufficiente"] == 4 * 4 and reasons["storico assente"] == 3 * 4
    notes = dict(zip(cov.players["giocatore"], cov.players["nota"]))
    assert sum("una sola stagione" in n for n in notes.values()) == 4  # i quattro con storico solo del 2026-27
    assert any("in attesa" in n or "non ha partite" in n for n in notes.values())


def test_coverage_flags_recent_team_change():
    logs, names = build_world(n_full=1, n_early=0, n_none=0)
    pid = next(iter(names))
    logs[pid] = [replace(l, team="OLD") if i < 28 else replace(l, team="NEW") for i, l in enumerate(logs[pid])]
    snap = collect(FakeBdl(logs, names))
    cov = build_coverage(bundle_from_snapshot(snap))
    assert "cambio squadra" in cov.players["nota"].iloc[0]


# ---- 8. lo scenario esatto segnalato: 105 giocatori, nessuna quota prezzata ------------------------------
def test_reported_scenario_many_players_early_season_now_prices_quotes():
    logs, names = build_world(n_full=20, n_early=30, n_none=55)  # 105 giocatori nelle quote
    snap = collect(FakeBdl(logs, names))
    assert snap.retrieval["players_in_quotes"] == 105 and snap.retrieval["players_requested"] == 105
    bundle = bundle_from_snapshot(snap)
    cov = build_coverage(bundle, None)
    assert cov.state_counts[COMPLETE] == 20 and cov.state_counts[NONE] == 55 and cov.state_counts[INSUFFICIENT] == 30
    board = build_board(bundle, BoardParams(n_boot=0, n_sims_pra=1500))
    n_priced = int(board.rows["priced"].sum())
    assert n_priced == 20 * 4  # tutte e sole le quote dei giocatori con storico sufficiente
    assert board.rows.loc[~board.rows["priced"], "p_win"].isna().all()
