"""Esegue tutte le pagine Streamlit (AppTest): nessuna eccezione, messaggi live chiari, navigazione al dettaglio."""

from datetime import timedelta
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from nba_vf.model.synthetic import TruePlayer, generate_player
from nba_vf.models import Market, OddsQuote, Side, utc_now
from nba_vf.ui.live import EMPTY_LIVE_MESSAGE
from nba_vf.ui.snapshot import Snapshot, save_snapshot

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["best", "overview", "matches", "props", "value", "player", "coverage", "bookmakers", "backtest", "settings"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_VF_CACHE_DIR", str(tmp_path))
    for k in ("BALLDONTLIE_API_KEY", "THE_ODDS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(at) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.warning] + [m.value for m in at.info] + [m.value for m in at.error] + [m.value for m in at.success]
    return "\n".join(parts)


def go(at, page):
    at.switch_page(f"views/{page}.py").run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_runs_in_demo(page):
    at = go(app(), page)
    assert at.title and "MODALITÀ DEMO" in texts(at)


def to_live(at):
    at.sidebar.radio(key="mode_label").set_value("Live (dati API)").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_runs_in_live_without_data(page):
    at = to_live(app())
    go(at, page)
    body = texts(at)
    assert "MODALITÀ LIVE" in body and "Giocatore Demo" not in body and "MODALITÀ DEMO" not in body
    if page not in ("settings", "backtest"):
        assert EMPTY_LIVE_MESSAGE[:40] in body or "Nessuna" in body or "Nessun" in body


def test_live_empty_overview_has_clear_message_and_zero_data():
    at = to_live(app())
    body = texts(at)
    assert EMPTY_LIVE_MESSAGE in body
    assert "Giocatore Demo" not in body and "SINTETIC" not in body.replace("SINTETICI", "")  # nessun dato inventato


def test_live_with_snapshot_shows_real_structure_and_prices():
    now = utc_now()
    logs = generate_player(TruePlayer(name="Giocatore Test"), 45, 3)  # dati di TEST per la fixture, non demo
    qs = [OddsQuote("the_odds_api", bk, "e1", "Giocatore Test", None, Market.POINTS, 17.5, s, o, now, now, "HOM", "AWY", now + timedelta(hours=6), False)
          for bk in ("book_x", "book_y") for s, o in ((Side.OVER, 1.95), (Side.UNDER, 1.85))]
    save_snapshot(Snapshot(fetched_at=now, quotes=qs, logs={"giocatore test": logs}, players={"giocatore test": {"name": "Giocatore Test", "team": "HOM"}},
                           providers=[{"name": "The Odds API", "status": "ok", "detail": "test", "n_quotes": 4}]))
    at = to_live(app())
    assert "MODALITÀ LIVE" in texts(at)
    for page in PAGES:
        go(at, page)
    go(at, "player")
    assert any("Giocatore Test" in str(s.value) for s in at.selectbox)


def test_overview_open_analysis_navigates_to_player():
    at = go(app(), "overview")
    at.button(key="ov_open").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Analisi giocatore"
    assert any("Dati mancanti" in h.value for h in at.subheader)


def test_player_page_shows_required_sections():
    at = go(app(), "player")
    heads = [h.value for h in at.subheader]
    for needed in ("Stima del modello", "Confronto quote tra bookmaker", "Motivazione sintetica della stima", "Dati mancanti e limiti"):
        assert needed in heads, heads
    body = texts(at)
    assert "Media stagione" in body and "Ultime 5" in body and "Ultime 10" in body and "Ultime 20" in body
    assert "storico" in body and "garantisce" in body


def test_player_without_enough_history_explains():
    at = go(app(), "player")
    names = [o for o in at.selectbox[0].options]
    at.selectbox[0].select("Giocatore Demo Matricola").run()
    assert not at.exception
    assert any("non disponibile" in w.value.lower() or "non calcolabile" in m.value.lower()
               for w in at.warning for m in list(at.markdown) + [w])


def test_value_finder_filters_change_results():
    at = go(app(), "value")
    base = [m for m in at.metric][0].value
    at.slider(key="vf_roi").set_value(40).run()
    assert not at.exception
    assert int([m for m in at.metric][0].value) <= int(base)


def test_props_filters_run_and_empty_result_is_handled():
    at = go(app(), "props")
    at.text_input(key="pp_q").set_value("zzz-nessuno").run()
    assert not at.exception


def test_settings_never_shows_key_values(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "TOP-SECRET-VALUE-123456")
    at = go(app(), "settings")
    blob = str(at.get("dataframe")) + texts(at) + str(at.get("json"))
    assert "TOP-SECRET-VALUE-123456" not in blob
    assert "configurata" in str(at.dataframe[0].value)


def test_csv_import_disabled_in_demo():
    at = go(app(), "settings")
    assert any("solo in modalità Live" in m.value for m in at.info)


# ---------------------------------------------------------------- pagina Backtesting
def _small_backtest(at):
    at.slider(key="bt_demo_players").set_value(12)
    at.slider(key="bt_demo_games").set_value(20)
    at.multiselect(key="bt_cands").set_value(["base"])
    at.select_slider(key="bt_sims").set_value(1500)
    return at


def test_backtest_page_demo_runs_and_labels_synthetic(monkeypatch):
    at = go(app(), "backtest")
    assert "SINTETICI" in texts(at) and "Nessun backtest archiviato" in texts(at)
    _small_backtest(at).run()
    at.button(key="bt_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    body = texts(at)
    assert "Dati sintetici: nessuna conclusione sulla NBA" in body
    assert "SUPERATA" not in body and "verifica_superata" not in body  # mai un esito positivo su dati sintetici
    frames = [d.value for d in at.dataframe if hasattr(d.value, "columns")]
    assert any("Brier" in map(str, f.columns) for f in frames)  # tabella con intervalli
    assert any("Yield" in map(str, f.columns) for f in frames)  # ROI calcolato SOLO perche' ci sono quote (sintetiche) con timestamp


def test_backtest_page_verify_button_and_segments_run():
    at = go(app(), "backtest")
    _small_backtest(at).run()
    at.button(key="bt_run").click().run()
    at.button(key="bt_verify").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Tutti i controlli superati" in texts(at)
    for by in ("season", "player", "book"):
        at.radio(key="bt_seg").set_value(by).run()
        assert not at.exception, (by, [e.value for e in at.exception])


def test_backtest_page_live_without_data_never_uses_demo_data():
    at = app()
    to_live(at)
    at = go(at, "backtest")
    body = texts(at)
    assert "Storico insufficiente" in body or "Nessun backtest" in body
    assert "Giocatore Demo" not in body
    assert not at.button or all(b.key != "bt_run" for b in at.button)  # niente backtest senza storico reale
