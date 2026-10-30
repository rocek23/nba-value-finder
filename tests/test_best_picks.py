"""Pagina «Migliori giocate»: classifica leggibile, solo valore positivo, nessun dato inventato. Dati SINTETICI."""

from datetime import timedelta

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from nba_vf.ui.best import HIGH, LOW, MEDIUM, build_picks, most_probable, most_profitable, reliability
from nba_vf.ui.board import BoardParams, annotate, build_board
from nba_vf.ui.labels import game_title, team_label
from nba_vf.ui.live import bundle_from_snapshot
from nba_vf.ui.snapshot import save_snapshot

from test_history_recovery import NOW, FakeBdl, build_world, collect


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))
    for k in ("BALLDONTLIE_API_KEY", "THE_ODDS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    st.cache_resource.clear()


def rows_for(n_full=10, lines=20.5, now_offset_min=1):
    logs, names = build_world(n_full=n_full, n_early=2, n_none=2)
    snap = collect(FakeBdl(logs, names, lines=lines))
    b = bundle_from_snapshot(snap)
    board = build_board(b, BoardParams(n_boot=20, n_sims_pra=2000))
    return annotate(board.rows, NOW + timedelta(minutes=now_offset_min), 30.0, 10.0), b


def test_team_labels_and_titles():
    assert team_label("LAL") == "Lakers (LAL)" and team_label("DMG") == "DMG" and team_label(None) == "—"
    assert game_title("BOS", "LAL") == "Celtics @ Lakers"


def test_reliability_levels():
    assert reliability(10, 0.1, 0.05) == LOW and reliability(30, 0.1, 0.02) == HIGH
    assert reliability(30, 0.1, -0.02) == MEDIUM and reliability(30, 0.1, float("nan")) == MEDIUM


def test_picks_have_value_real_names_and_valid_numbers():
    rows, _ = rows_for(lines=17.5)  # linea bassa: l'Over ha probabilita' alta, serve almeno una quota con valore
    # quote deliberatamente generose su un lato per avere valore positivo
    rows = rows.copy()
    rows.loc[rows["lato"] == "Over", "quota"] = 3.5
    rows["roi"] = rows["p_win"] * rows["quota"] + rows["p_push"] - 1
    rows["roi_lo"] = rows["roi"] - 0.05
    picks = build_picks(rows, 10.0, include_low=True)
    assert len(picks) and (picks["guadagno"] > 0).all()
    assert picks["giocatore"].str.startswith("Giocatore").all()  # nomi veri dei giocatori delle quote
    assert (picks["prob"].between(0, 100)).all() and (picks["vincita"] > 0).all()
    # una sola riga per giocata e lato
    assert not picks.duplicated(["prop_id", "lato"]).any()


def test_unpriced_players_never_appear():
    rows, _ = rows_for()
    picks = build_picks(rows, 10.0, include_low=True, fresh_only=False)
    priced_keys = set(rows.loc[rows["priced"], "key"])
    assert set(picks["key"]) <= priced_keys
    assert not any("Nuovo" in g or "Assente" in g for g in picks["giocatore"])  # senza storico sufficiente: nessuna giocata


def test_low_sample_and_stale_filters():
    rows, _ = rows_for()
    rows = rows.copy()
    rows["quota"] = rows["quota"] * 2.5
    rows["roi"] = rows["p_win"] * rows["quota"] + rows["p_push"] - 1
    all_ = build_picks(rows, 10.0, include_low=True, fresh_only=False)
    no_low = build_picks(rows, 10.0, include_low=False, fresh_only=False)
    assert len(no_low) <= len(all_) and (no_low["affidabilita"] != LOW).all()
    stale = rows.assign(obsoleta=True)
    assert build_picks(stale, 10.0, include_low=True, fresh_only=True).empty
    assert len(build_picks(stale, 10.0, include_low=True, fresh_only=False)) == len(all_)


def test_rankings_sorted():
    rows, _ = rows_for()
    rows = rows.copy()
    rows["quota"] = rows["quota"] * 2.5
    rows["roi"] = rows["p_win"] * rows["quota"] + rows["p_push"] - 1
    picks = build_picks(rows, 10.0, include_low=True, fresh_only=False)
    assert len(picks) > 3
    g = most_profitable(picks, 5)["guadagno"].tolist()
    p = most_probable(picks, 5)["prob"].tolist()
    assert g == sorted(g, reverse=True) and p == sorted(p, reverse=True) and len(g) == 5


def test_empty_rows_give_empty_picks():
    import pandas as pd

    assert build_picks(pd.DataFrame(), 10.0).empty


# ---- pagina e modalita' di default ---------------------------------------------------------------
from pathlib import Path

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_default_mode_is_demo_without_keys_or_snapshot():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    assert at.sidebar.radio(key="mode_label").value == "Demo (dati sintetici)"


def test_default_mode_is_live_when_snapshot_or_key_exists(monkeypatch):
    logs, names = build_world(n_full=3, n_early=0, n_none=0)
    save_snapshot(collect(FakeBdl(logs, names)))
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    assert at.sidebar.radio(key="mode_label").value == "Live (dati API)"
    body = "\n".join([m.value for m in at.markdown] + [m.value for m in at.warning] + [m.value for m in at.info])
    assert "Migliori giocate" in "\n".join(t.value for t in at.title)
    assert "Giocatore Demo" not in body


def test_best_page_renders_real_names_cards_and_tables(monkeypatch):
    from dataclasses import replace

    logs, names = build_world(n_full=8, n_early=2, n_none=2)
    snap = collect(FakeBdl(logs, names))
    snap.quotes = [replace(q, odds_decimal=3.6) if q.side.value == "over" else q for q in snap.quotes]  # quote generose: ci sono giocate con valore
    for q in snap.quotes:  # quote fresche rispetto all'orologio reale
        pass
    snap.quotes = [replace(q, fetched_at=__import__("nba_vf.models", fromlist=["utc_now"]).utc_now(),
                           quote_time=__import__("nba_vf.models", fromlist=["utc_now"]).utc_now()) for q in snap.quotes]
    save_snapshot(snap)
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.switch_page("views/best.py").run()
    assert not at.exception, [e.value for e in at.exception]
    body = "\n".join([m.value for m in at.markdown] + [m.value for m in at.caption] + [m.value for m in at.warning] + [m.value for m in at.info])
    heads = [h.value for h in at.subheader]
    assert "Le 3 giocate migliori" in heads, (heads, body[-600:])  # con quote generose esistono giocate con valore
    assert "Giocatore Demo" not in body
    assert "Giocatore Pieno" in body and "Affidabilità" in body
    assert len(at.dataframe) >= 2


def test_under_picks_can_be_hidden():
    rows, _ = rows_for()
    rows = rows.copy()
    rows["quota"] = rows["quota"] * 2.5
    rows["roi"] = rows["p_win"] * rows["quota"] + rows["p_push"] - 1
    both = build_picks(rows, 10.0, include_low=True, fresh_only=False)
    over = build_picks(rows, 10.0, include_low=True, fresh_only=False, include_under=False)
    assert set(both["lato"]) == {"Over", "Under"} and set(over["lato"]) == {"Over"}
    assert len(over) < len(both)


def test_market_prob_uses_other_books_only_and_is_nan_with_few_books():
    import numpy as np
    import pandas as pd
    from nba_vf.ui.best import market_prob

    def mk(book, over, under, line=24.5):
        return {"prop_id": "p1", "lato": "Over", "book": book, "quota": over, "linea": line}, {"prop_id": "p1", "lato": "Under", "book": book, "quota": under, "linea": line}

    rows = pd.DataFrame([r for b, o, u in [("A", 1.90, 1.90), ("B", 1.80, 2.00), ("C", 1.80, 2.00), ("D", 1.80, 2.00)] for r in mk(b, o, u)])
    pick = rows[(rows["lato"] == "Over") & (rows["book"] == "A")]
    p = market_prob(rows, pick)
    assert 50 < p[0] < 55  # media degli altri tre: 0.5263.. senza margine
    few = rows[rows["book"].isin(["A", "B"])]
    assert np.isnan(market_prob(few, few[(few["lato"] == "Over") & (few["book"] == "A")])[0])
    whole = rows.assign(linea=25.0)
    assert np.isnan(market_prob(whole, whole[(whole["lato"] == "Over") & (whole["book"] == "A")])[0])
