"""Stato di sessione, cache e accesso ai dati per le pagine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import streamlit as st

from ..config import ConfigError, Settings
from ..model.config import ModelConfig
from ..model.engine import PlayerPrediction, ValueEngine
from ..models import utc_now
from .board import Board, BoardParams, annotate, build_board, context_for, usable_logs
from .labels import normalize_name
from .bundle import DataBundle
from .demo import build_demo_bundle
from .live import load_live_bundle
from .snapshot import snapshot_path, sync_from_remote

MODES = {"Demo (dati sintetici)": "demo", "Live (dati API)": "live"}
DEFAULTS = {"mode_label": "Demo (dati sintetici)", "stake": 10.0, "max_age": 30.0, "n_boot": 40, "ci_level": 0.90}
PERSIST = ("mode_label", "stake", "max_age", "n_boot", "ci_level")


def init_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if "max_age_from_env" not in st.session_state:
        st.session_state["max_age_from_env"] = True
        try:
            st.session_state["max_age"] = float(Settings.from_env().max_quote_age_minutes)
        except ConfigError:
            pass
    if "mode_default_done" not in st.session_state:  # alla prima apertura: Live se ci sono chiavi o dati live, altrimenti Demo
        st.session_state["mode_default_done"] = True
        try:
            st_ = Settings.from_env()
            has_keys = bool(st_.balldontlie_api_key or st_.odds_api_key)
        except ConfigError:
            has_keys = False
        if has_keys or snapshot_path().exists():
            st.session_state["mode_label"] = "Live (dati API)"
    for k in PERSIST:  # mantiene vivi i valori dei widget anche quando la pagina che li mostra non e' attiva
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def settings_or_none() -> tuple[Settings | None, str | None]:
    try:
        return Settings.from_env(), None
    except ConfigError as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def _demo_bundle(seed: int, day: str) -> DataBundle:
    return build_demo_bundle(seed)


@st.cache_resource(show_spinner=False)
def _live_bundle(mtime: float) -> DataBundle:
    return load_live_bundle()


def current_mode() -> str:
    return MODES[st.session_state.get("mode_label", DEFAULTS["mode_label"])]


def get_bundle() -> DataBundle:
    if current_mode() == "demo":
        return _demo_bundle(2026, date.today().isoformat())
    sync_from_remote()
    path = snapshot_path()
    return _live_bundle(path.stat().st_mtime if path.exists() else 0.0)


@st.cache_resource(show_spinner="Calcolo delle stime con il modello…")
def _board(fingerprint: str, params: BoardParams, _bundle: DataBundle) -> Board:
    return build_board(_bundle, params)


def board_params() -> BoardParams:
    return BoardParams(n_boot=int(st.session_state["n_boot"]), ci_level=float(st.session_state["ci_level"]))


def get_board(bundle: DataBundle) -> Board:
    return _board(bundle.fingerprint, board_params(), bundle)


def rows_view(bundle: DataBundle, board: Board) -> pd.DataFrame:
    return annotate(board.rows, utc_now(), float(st.session_state["max_age"]), float(st.session_state["stake"]))


@st.cache_resource(show_spinner="Calcolo dettagliato con intervalli di incertezza…")
def _detail_prediction(fingerprint: str, key: str, ci_level: float, minutes_override: float | None, _bundle: DataBundle) -> PlayerPrediction:
    engine = ValueEngine(ModelConfig(n_boot=200, ci_level=ci_level))
    engine.fit_prior(_bundle.logs)
    first = next((q for q in _bundle.quotes if normalize_name(q.player_name) == key), None)
    return engine.predict(usable_logs(_bundle, key, first), context_for(_bundle, key, first, minutes_override))


def detail_prediction(bundle: DataBundle, key: str, minutes_override: float | None = None) -> PlayerPrediction:
    return _detail_prediction(bundle.fingerprint, key, float(st.session_state["ci_level"]), minutes_override, bundle)


@dataclass
class Ctx:
    bundle: DataBundle
    board: Board | None
    rows: pd.DataFrame
