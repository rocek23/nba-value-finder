"""Multiple vicino a quota 2: probabilita' di mercato (margine tolto, altri bookmaker), selezioni a quota bassa, sezione di Migliori giocate."""

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from nba_vf.model.synthetic import TruePlayer, generate_player
from nba_vf.models import Market, OddsQuote, Side, utc_now
from nba_vf.ui import builder as B
from nba_vf.ui.daily import consensus_table, market_legs
from nba_vf.ui.snapshot import Snapshot, save_snapshot
from tests.test_builder import make_rows

APP = str(Path(__file__).resolve().parents[1] / "app.py")
NOW = pd.Timestamp("2029-12-31", tz="UTC").to_pydatetime()


def rows_for(over_odds, under_odds=2.9, line=5.5, prop="p1", **kw):
    out = []
    for i, (o, u) in enumerate(zip(over_odds, [under_odds] * len(over_odds) if isinstance(under_odds, (int, float)) else under_odds)):
        for side, q in (("Over", o), ("Under", u)):
            out.append(make_rows(prop_id=prop, key=f"k{prop}", lato=side, quota=q, book=f"b{i}", bookmaker=f"Book {i}", linea=line, evento=f"E{prop} @ F{prop}",
                                 gday="2030-01-01", **kw))
    return out


def test_consensus_removes_margin_and_excludes_own_book():
    rows = pd.DataFrame(rows_for([1.50, 1.50, 1.50, 1.50]))
    c = consensus_table(rows)
    expected = (1 / 1.5) / (1 / 1.5 + 1 / 2.9)
    row = c[(c.lato == "Over") & (c.book == "b0")].iloc[0]
    assert row.p_equa == pytest.approx(expected) and row.n_altri == 3
    over_under = c[(c.book == "b0") & (c.prop_id == "p1")].groupby("lato").p_equa.first()
    assert over_under.sum() == pytest.approx(1.0)            # Over + Under = 100% dopo aver tolto il margine


def test_consensus_needs_both_sides():
    rows = pd.DataFrame(rows_for([1.5, 1.5, 1.5]))
    assert consensus_table(rows[rows.lato == "Over"]).empty


def test_market_legs_filters():
    ok = rows_for([1.45, 1.46, 1.47, 1.48])
    legs = market_legs(pd.DataFrame(ok), NOW)
    assert len(legs) == 4 and all(l.lato == "Over" and l.odds <= 1.6 and l.p_win >= .6 and l.p_push == 0 for l in legs)
    assert all(l.p_lo <= l.p_win <= l.p_hi for l in legs)
    assert market_legs(pd.DataFrame(rows_for([1.45, 1.46, 1.47])), NOW) == []                      # servono almeno 3 altri bookmaker
    assert market_legs(pd.DataFrame(rows_for([1.45, 1.46, 1.47, 1.48], line=5.0)), NOW) == []     # linee intere escluse
    assert market_legs(pd.DataFrame(rows_for([2.1, 2.1, 2.1, 2.1], under_odds=1.8)), NOW) == []    # quota troppo alta / probabilita' bassa
    for bad in (dict(manuale=True), dict(obsoleta=True), dict(infortunio="Out")):
        assert market_legs(pd.DataFrame(rows_for([1.45, 1.46, 1.47, 1.48], **bad)), NOW) == []
    assert len(market_legs(pd.DataFrame(rows_for([1.45, 1.46, 1.47, 1.48], infortunio="")), NOW)) == 4
    assert market_legs(pd.DataFrame(rows_for([1.45, 1.46, 1.47, 1.48])), pd.Timestamp("2031-01-01", tz="UTC").to_pydatetime()) == []  # partita iniziata


def test_search_min_ev_allows_slightly_negative_price_but_default_does_not():
    legs = market_legs(pd.DataFrame(rows_for([1.45] * 4, prop="a") + rows_for([1.45] * 4, prop="b")), NOW)
    assert not B.search(legs, 2.1, tolerance=.1, min_sel=2, max_sel=2).proposals                     # EV < 0: mai proposta di default
    res = B.search(legs, 2.1, tolerance=.1, min_sel=2, max_sel=2, min_ev=-.10)
    assert res.proposals
    for p in res.proposals:
        assert -.10 < p.ev < 0 and 1.89 <= p.odds <= 2.31 and len({l.book for l in p.legs}) == 1 and len({l.game for l in p.legs}) == p.n


# ---- pagina: Live con quote di piu' bookmaker ------------------------------------------------------------------------
@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))
    for k in ("BALLDONTLIE_API_KEY", "THE_ODDS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_page_live_doubles_use_only_snapshot_quotes_one_book_one_per_game(isolated):
    now = utc_now()
    qs, logs, players = [], {}, {}
    for i in range(4):
        name = f"Giocatore Test {i}"
        key = name.lower()
        logs[key] = generate_player(TruePlayer(name=name), 45, i + 1)
        players[key] = {"name": name, "team": f"H{i}"}
        for b, o in enumerate((1.42, 1.43, 1.44, 1.45, 1.46)):
            for s, odds in ((Side.OVER, o), (Side.UNDER, 2.9)):
                qs.append(OddsQuote("the_odds_api", f"book_{b}", f"e{i}", name, None, Market.POINTS, 5.5, s, odds, now, now, f"H{i}", f"A{i}",
                                    now + timedelta(hours=6), False))
    save_snapshot(Snapshot(fetched_at=now, quotes=qs, logs=logs, players=players,
                           providers=[{"name": "The Odds API", "status": "ok", "detail": "test", "n_quotes": len(qs)}]))
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.sidebar.radio(key="mode_label").set_value("Live (dati API)").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(s.value == "Multiple vicino a quota 2" for s in at.subheader)
    tables = [d.value for d in at.dataframe if "Giocata" in d.value.columns and "Probabilità" in d.value.columns]
    assert tables, "nessuna multipla proposta"
    for t in tables:
        assert set(t["Quota"]) <= {1.42, 1.43, 1.44, 1.45, 1.46}                      # solo quote reali dello snapshot
        assert len(t) in (2, 3) and t["Partita"].str.split(" · ").str[0].nunique() == len(t)
        assert all(g.startswith("Giocatore Test") for g in t["Giocatore"])
    assert not any("Giocatore Demo" in str(m.value) for m in at.markdown)


def test_page_live_without_data_shows_no_doubles(isolated):
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.sidebar.radio(key="mode_label").set_value("Live (dati API)").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not [s for s in at.subheader if s.value == "Multiple vicino a quota 2"]


def test_empty_after_filters_with_injury_column_does_not_crash():
    """Regressione: se i filtri precedenti svuotano la tabella, il filtro infortuni non deve togliere le colonne (KeyError 'lato')."""
    rows = pd.DataFrame(rows_for([1.45, 1.46, 1.47, 1.48], manuale=True, infortunio=""))
    assert market_legs(rows, NOW) == []
    past = pd.DataFrame(rows_for([1.45, 1.46, 1.47, 1.48], infortunio=""))
    assert market_legs(past, pd.Timestamp("2031-01-01", tz="UTC").to_pydatetime()) == []
    legs, funnel = B.eligible_legs(pd.DataFrame([make_rows(manuale=True, infortunio="")]), NOW)
    assert legs == [] and funnel[-1][1] == 0
    from nba_vf.ui.best import build_picks
    base = dict(make_rows(), roi=-.1, roi_lo=-.2, quota_equa=1.9, ev=-1.0, infortunio="")
    assert build_picks(pd.DataFrame([base]), 10.0, fresh_only=False).empty                     # nessuna giocata con valore: tabella vuota ma valida
    assert build_picks(pd.DataFrame([dict(base, roi=.1, priced=False)]), 10.0, fresh_only=False).empty


def _top_rows(extra=None):
    import pandas as pd
    base = dict(manuale=False, obsoleta=False, priced=True, inizio=pd.Timestamp("2030-01-01", tz="UTC"), n_partite=40, gday="d1", evento="A @ B",
                key="k1", giocatore="Gioc 1", squadra="A", mercato="Punti", market="points", linea=14.5, p_win=0.75, infortunio="", bookmaker="x", ts_nota=1, eta_min=5.0)
    out = []
    for book, o, u in [("b1", 1.30, 3.4), ("b2", 1.40, 3.1), ("b3", 1.32, 3.3), ("b4", 1.31, 3.3)]:
        for lato, q in (("Over", o), ("Under", u)):
            r = dict(base, prop_id="p1", lato=lato, book=book, quota=q, bookmaker=book)
            r.update(extra or {})
            out.append(r)
    return pd.DataFrame(out)


def test_top_of_day_picks_high_market_probability_with_fair_price():
    from datetime import datetime, timezone
    from nba_vf.ui.daily import top_of_day
    now = datetime(2029, 12, 31, tzinfo=timezone.utc)
    t = top_of_day(_top_rows(), now)
    assert len(t) == 1 and t.loc[0, "lato"] == "Over" and t.loc[0, "prob_mercato"] > 70 and t.loc[0, "quota"] == 1.40
    assert top_of_day(_top_rows({"p_win": 0.50}), now).empty        # il modello contraddice il mercato
    assert top_of_day(_top_rows({"linea": 15.0}), now).empty         # linea intera: nessuna probabilita' affidabile
    assert top_of_day(_top_rows({"infortunio": "Out"}), now).empty   # giocatore Out
    assert top_of_day(_top_rows(), now, min_lines={"Punti": 20}).empty  # linea sotto il minimo del bookmaker


def test_top_of_day_funnel_counts_each_step():
    from datetime import datetime, timezone
    from nba_vf.ui.daily import top_of_day
    f = []
    top_of_day(_top_rows(), datetime(2029, 12, 31, tzinfo=timezone.utc), funnel=f)
    labels = [a for a, _ in f]
    assert labels[0].startswith("Giocate") and labels[-1].startswith("Modello") and f[-1][1] == 1
    f2 = []
    top_of_day(_top_rows({"p_win": 0.5}), datetime(2029, 12, 31, tzinfo=timezone.utc), funnel=f2)
    assert f2[-1][1] == 0 and f2[-2][1] == 1
