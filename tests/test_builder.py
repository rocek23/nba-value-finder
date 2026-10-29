"""Bet Builder AI: quota combinata, probabilita' congiunta, EV, vincoli (una selezione per partita, un solo bookmaker) e pagina."""

import itertools
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from nba_vf.model.synthetic import TruePlayer, generate_player
from nba_vf.models import Market, OddsQuote, Side, utc_now
from nba_vf.ui import builder as B
from nba_vf.ui.snapshot import Snapshot, save_snapshot

APP = str(Path(__file__).resolve().parents[1] / "app.py")
T0 = pd.Timestamp("2030-01-01 20:00", tz="UTC")


def leg(i, game, odds, p, *, book="BookA", push=0.0, lo=None, hi=None, line=20.5, lato="Over", key=None):
    lo = p * 0.85 if lo is None else lo
    hi = min(1.0, p * 1.15) if hi is None else hi
    return B.Leg(id=f"L{i}|{book}", key=key or f"pl{i}", giocatore=f"G{i}", squadra="AAA", game=game, evento="AAA @ BBB", inizio=T0, market=Market.POINTS,
                 mercato="Punti", linea=line, lato=lato, book=book, odds=odds, p_win=p, p_push=push, p_lo=lo, p_hi=hi, n_games=40,
                 eta_min=5.0, quote_known=True, prop_id=f"P{i}")


# ---- formule ----------------------------------------------------------------------------------------------
def test_total_odds_is_product():
    legs = [leg(1, "g1", 1.8, .6), leg(2, "g2", 2.5, .45), leg(3, "g3", 1.5, .7)]
    assert B.total_odds(legs) == pytest.approx(1.8 * 2.5 * 1.5)


def test_joint_probability_is_product_of_win_probabilities():
    legs = [leg(1, "g1", 1.8, .6), leg(2, "g2", 2.5, .45)]
    assert B.joint_win_prob(legs) == pytest.approx(.6 * .45)


def test_ev_single_and_multiple_formula():
    a, b = leg(1, "g1", 2.2, .5), leg(2, "g2", 2.2, .5)
    assert B.combo_ev([a]) == pytest.approx(.5 * 2.2 - 1)
    assert B.combo_ev([a, b]) == pytest.approx((.5 * 2.2) ** 2 - 1)  # = 0.21


def test_ev_with_push_removes_the_leg_instead_of_losing():
    a = leg(1, "g1", 2.2, .4, push=.1)
    assert B.combo_ev([a]) == pytest.approx(.4 * 2.2 + .1 - 1)
    b = leg(2, "g2", 1.9, .5)
    assert B.combo_ev([a, b]) == pytest.approx((.4 * 2.2 + .1) * (.5 * 1.9) - 1)


def test_prudent_ev_uses_lower_bounds_and_is_not_higher():
    legs = [leg(1, "g1", 2.2, .5, lo=.40), leg(2, "g2", 2.0, .55, lo=.45)]
    assert B.combo_ev(legs, prudent=True) == pytest.approx((.40 * 2.2) * (.45 * 2.0) - 1)
    assert B.combo_ev(legs, prudent=True) <= B.combo_ev(legs)


def test_uncertainty_label_thresholds():
    assert B.uncertainty_label(1.2) == "Bassa" and B.uncertainty_label(2.0) == "Media" and B.uncertainty_label(4.0) == "Alta"


# ---- ricerca ------------------------------------------------------------------------------------------------
def random_legs(seed, games=6, per=4, book="BookA"):
    rng = np.random.default_rng(seed)
    out, i = [], 0
    for g in range(games):
        for _ in range(per):
            p = rng.uniform(.35, .8)
            out.append(leg(i, f"g{g}", round(1 / p * rng.uniform(.92, 1.12), 2), p, book=book))
            i += 1
    return out


@pytest.mark.parametrize("seed,target", [(1, 5), (2, 10), (3, 5)])
def test_search_best_ev_matches_exhaustive_search(seed, target):
    legs = random_legs(seed)
    res = B.search(legs, target, tolerance=.15, min_sel=2, max_sel=5)
    lo, hi = target * .85, target * 1.15
    by = {}
    for l in legs:
        by.setdefault(l.game, []).append(l)
    best = -9.0
    for n in range(2, 6):
        for gs in itertools.combinations(by, n):
            for combo in itertools.product(*[by[g] for g in gs]):
                if lo <= B.total_odds(combo) <= hi:
                    best = max(best, B.combo_ev(combo))
    if best > 0:
        top = max(res.proposals, key=lambda p: p.ev)
        assert top.ev == pytest.approx(best, rel=1e-9)
    else:
        assert not res.proposals


def test_proposals_respect_all_constraints():
    res = B.search(random_legs(4), 10, tolerance=.15, min_sel=2, max_sel=6)
    assert res.proposals
    for p in res.proposals:
        assert 10 * .85 <= p.odds <= 10 * 1.15
        assert 2 <= p.n <= 6
        assert len({l.game for l in p.legs}) == p.n          # una selezione per partita
        assert len({l.key for l in p.legs}) == p.n           # un giocatore una sola volta
        assert p.ev > 0
        assert p.odds == pytest.approx(math.prod(l.odds for l in p.legs))
        assert p.p_all == pytest.approx(math.prod(l.p_win for l in p.legs))


def test_three_alternatives_have_expected_properties():
    res = B.search(random_legs(5, games=7), 10, tolerance=.2, min_sel=2, max_sel=7)
    ev = next(p for p in res.proposals if B.LABEL_EV in p.labels)
    pr = next(p for p in res.proposals if B.LABEL_PROB in p.labels)
    few = next(p for p in res.proposals if B.LABEL_FEW in p.labels)
    assert ev.ev >= max(p.ev for p in res.proposals) - 1e-12
    assert pr.p_all >= max(p.p_all for p in res.proposals) - 1e-12
    assert few.n <= min(p.n for p in res.proposals)


def test_identical_alternatives_are_merged_with_all_labels():
    legs = [leg(1, "g1", 2.2, .6), leg(2, "g2", 2.2, .6)]
    res = B.search(legs, 4.84, tolerance=.05, min_sel=2, max_sel=2)
    assert len(res.proposals) == 1 and set(res.proposals[0].labels) == {B.LABEL_EV, B.LABEL_PROB, B.LABEL_FEW}


def test_single_bookmaker_by_default_never_mixes_books():
    legs = random_legs(6, games=5, book="BookA") + [dataclass_replace(l, book="BookB", id=l.id + "b") for l in random_legs(7, games=5, book="BookB")]
    res = B.search(legs, 5, tolerance=.2, max_sel=5)
    assert res.proposals
    for p in res.proposals:
        assert len(p.books) == 1 and not p.mixed


def dataclass_replace(l, **kw):
    import dataclasses
    return dataclasses.replace(l, **kw)


def test_mixed_books_only_when_allowed_and_flagged():
    a = [leg(i, f"g{i}", 2.3, .5, book="BookA" if i % 2 == 0 else "BookB") for i in range(4)]
    assert not B.search(a, 5.29, tolerance=.05, min_sel=2, max_sel=2).proposals or all(len(p.books) == 1 for p in B.search(a, 5.29, tolerance=.05).proposals)
    res = B.search(a, 5.29, tolerance=.05, min_sel=2, max_sel=2, mixed_books=True)
    assert res.proposals and any(p.mixed for p in res.proposals)


def test_no_combination_returns_reason_not_invented_picks():
    legs = [leg(1, "g1", 1.5, .6)]
    res = B.search(legs, 100, tolerance=.15, min_sel=2, max_sel=8)
    assert not res.proposals and res.reasons and "solo 1 partite" in res.reasons[0]


def test_unreachable_target_explains_max_reachable_odds():
    legs = [leg(i, f"g{i}", 1.5, .66) for i in range(4)]
    res = B.search(legs, 100, tolerance=.15, min_sel=2, max_sel=4)
    assert not res.proposals and any("quota massima raggiungibile" in r for r in res.reasons)


def test_negative_ev_combinations_are_not_proposed():
    legs = [leg(i, f"g{i}", 2.0, .40) for i in range(6)]          # ogni selezione ha EV -20%
    res = B.search(legs, 8, tolerance=.15, min_sel=2, max_sel=6)
    assert not res.proposals and res.n_window > 0 and res.best_ev_seen < 0
    assert any("nessuna ha un valore atteso positivo" in r for r in res.reasons)


def test_require_prudent_filters_out_fragile_combinations():
    legs = [leg(i, f"g{i}", 2.2, .50, lo=.40) for i in range(4)]    # EV centrale > 0, prudente < 0
    assert B.search(legs, 4.84, tolerance=.05, min_sel=2, max_sel=2).proposals
    assert not B.search(legs, 4.84, tolerance=.05, min_sel=2, max_sel=2, require_prudent=True).proposals


# ---- selezioni ammissibili ---------------------------------------------------------------------------------------
def make_rows(**over):
    base = dict(priced=True, manuale=False, obsoleta=False, inizio=T0, n_partite=40, p_win_lo=.4, p_win_hi=.6, p_win=.5, quota=2.2, lato="Over",
                mercato="Punti", bookmaker="BookA", gday="2030-01-01", evento="AAA @ BBB", prop_id="p1", key="k1", giocatore="G", squadra="AAA",
                market=Market.POINTS, linea=20.5, p_push=0.0, eta_min=3.0, ts_nota=True, book="booka")
    base.update(over)
    return base


def test_eligible_legs_filters_each_reason():
    rows = pd.DataFrame([
        make_rows(prop_id="ok"),
        make_rows(prop_id="unpriced", priced=False), make_rows(prop_id="manual", manuale=True), make_rows(prop_id="stale", obsoleta=True),
        make_rows(prop_id="past", inizio=pd.Timestamp("2020-01-01", tz="UTC")), make_rows(prop_id="short", n_partite=10),
        make_rows(prop_id="noci", p_win_lo=np.nan), make_rows(prop_id="under", lato="Under"), make_rows(prop_id="nodate", inizio=pd.NaT),
    ])
    legs, funnel = B.eligible_legs(rows, pd.Timestamp("2029-12-31", tz="UTC").to_pydatetime())
    assert [l.prop_id for l in legs] == ["ok"]
    assert funnel[0] == ("Quote disponibili", 9)
    assert [l.prop_id for l in B.eligible_legs(rows, pd.Timestamp("2029-12-31", tz="UTC").to_pydatetime(), include_under=True)[0]] == ["ok", "under"]
    assert B.eligible_legs(rows, pd.Timestamp("2029-12-31", tz="UTC").to_pydatetime(), markets=["Rimbalzi"])[0] == []


def test_leg_stats_uses_only_games_before_the_game_day_and_correct_windows():
    logs = generate_player(TruePlayer(name="X"), 30, 1)
    l = leg(1, "g", 2.0, .5, line=10.5)
    l = dataclass_replace(l, inizio=pd.Timestamp(max(x.game_date for x in logs) + timedelta(days=1), tz="UTC"))
    st_ = B.leg_stats(logs, l)
    last10 = sorted(logs, key=lambda x: x.game_date)[-10:]
    assert st_["finestre"][10]["riuscita"] == pytest.approx(sum(x.pts > 10.5 for x in last10) / 10)
    assert st_["finestre"][5]["n"] == 5 and st_["finestre"][20]["n"] == 20
    early = dataclass_replace(l, inizio=pd.Timestamp(sorted(logs, key=lambda x: x.game_date)[10].game_date, tz="UTC"))
    assert B.leg_stats(logs, early)["finestre"][20]["n"] == 10   # la partita del giorno stesso e le successive sono escluse


# ---- pagina ---------------------------------------------------------------------------------------------------------
@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))
    for k in ("BALLDONTLIE_API_KEY", "THE_ODDS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _open(at=None):
    at = at or AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.switch_page("views/builder.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_page_demo_generates_proposals_with_metrics(isolated):
    at = _open()
    assert [b.label for b in at.button if b.key and b.key.startswith("bb_t")] == ["x2", "x5", "x10", "x25", "x50", "x100"]
    _click(at, "bb_t5")
    _click(at, "bb_gen")
    names = [m.label for m in at.metric]
    assert "Probabilità totale" in names and "EV centrale" in names and "EV prudente" in names and "Incertezza" in names
    assert any("non sono ancora sufficientemente validate" in w.value for w in at.warning)


def test_page_high_target_warns_about_low_probability(isolated):
    at = _open()
    _click(at, "bb_t100")
    _click(at, "bb_gen")
    body = " ".join(w.value for w in at.warning) + " ".join(e.value for e in at.error)
    assert "molto bassa" in body or "Nessuna multipla affidabile" in body


def test_page_live_without_data_never_uses_demo_data(isolated):
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.sidebar.radio(key="mode_label").set_value("Live (dati API)").run()
    at.switch_page("views/builder.py").run()
    assert not at.exception, [e.value for e in at.exception]
    body = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning)
    assert "Giocatore Demo" not in body and "MODALITÀ DEMO" not in body
    assert not [b for b in at.button if b.key == "bb_gen"]   # senza quote la pagina si ferma con un messaggio chiaro


def test_page_live_snapshot_uses_only_real_quotes_and_one_selection_per_game(isolated):
    now = utc_now()
    qs, logs, players = [], {}, {}
    for i in range(4):
        name = f"Giocatore Test {i}"
        key = name.lower()
        logs[key] = generate_player(TruePlayer(name=name), 45, i + 1)
        players[key] = {"name": name, "team": f"H{i}"}
        for s, o in ((Side.OVER, 2.2), (Side.UNDER, 1.6)):
            qs.append(OddsQuote("the_odds_api", "book_x", f"e{i}", name, None, Market.POINTS, 5.5, s, o, now, now, f"H{i}", f"A{i}", now + timedelta(hours=6), False))
    save_snapshot(Snapshot(fetched_at=now, quotes=qs, logs=logs, players=players,
                           providers=[{"name": "The Odds API", "status": "ok", "detail": "test", "n_quotes": len(qs)}]))
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.sidebar.radio(key="mode_label").set_value("Live (dati API)").run()
    at.switch_page("views/builder.py").run()
    assert not at.exception, [e.value for e in at.exception]
    _click(at, "bb_t10")
    _click(at, "bb_gen")
    tables = [d.value for d in at.dataframe if "Giocata" in d.value.columns]
    assert tables, "nessuna proposta"
    for t in tables:
        assert set(t["Bookmaker"]) == {"Book X"} and set(t["Quota"]) == {2.2}      # quote reali dello snapshot, un solo bookmaker
        assert t["Partita"].str.split(" · ").str[0].nunique() == len(t)             # una selezione per partita
        assert all(g.startswith("Giocatore Test") for g in t["Giocatore"])
