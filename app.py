"""NBA Value Finder — avvio:  streamlit run app.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from nba_vf.config import load_env  # noqa: E402

load_env(ROOT / ".env")

st.set_page_config(page_title="NBA Value Finder", page_icon="🏀", layout="wide", initial_sidebar_state="expanded")

PAGES = {
    "overview": st.Page("views/overview.py", title="Panoramica", icon=":material/dashboard:", url_path="panoramica", default=True),
    "partite": st.Page("views/matches.py", title="Partite NBA", icon=":material/sports_basketball:", url_path="partite"),
    "props": st.Page("views/props.py", title="Player Props", icon=":material/list:", url_path="props"),
    "value": st.Page("views/value.py", title="Value Finder", icon=":material/search:", url_path="value-finder"),
    "giocatore": st.Page("views/player.py", title="Analisi giocatore", icon=":material/person:", url_path="giocatore"),
    "bookmaker": st.Page("views/bookmakers.py", title="Confronto bookmaker", icon=":material/compare_arrows:", url_path="bookmaker"),
    "backtest": st.Page("views/backtest.py", title="Storico e backtesting", icon=":material/query_stats:", url_path="backtesting"),
    "impostazioni": st.Page("views/settings.py", title="Impostazioni e API", icon=":material/settings:", url_path="impostazioni"),
}
st.session_state["_pages"] = PAGES
navigation = st.navigation({"Analisi": [PAGES[k] for k in ("overview", "partite", "props", "value", "giocatore")],
                            "Strumenti": [PAGES[k] for k in ("bookmaker", "backtest", "impostazioni")]})
navigation.run()
