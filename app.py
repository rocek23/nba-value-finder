"""NBA Value Finder — avvio:  streamlit run app.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402


def _purge_stale_modules() -> None:
    """Dopo un aggiornamento del codice (es. nuovo commit su GitHub) il server puo' tenere in memoria i moduli vecchi:
    li scarta cosi' le pagine nuove non si scontrano con codice vecchio (ImportError) e non serve riavviare l'app a mano."""
    files = list((ROOT / "src" / "nba_vf").rglob("*.py"))
    signature = max((f.stat().st_mtime for f in files), default=0.0)
    previous = getattr(sys, "_nvf_code_signature", None)
    if previous is not None and previous != signature:
        for name in [m for m in sys.modules if m == "nba_vf" or m.startswith("nba_vf.")]:
            del sys.modules[name]
        st.cache_resource.clear()
    sys._nvf_code_signature = signature


_purge_stale_modules()

from nba_vf.config import load_env  # noqa: E402

load_env(ROOT / ".env")

st.set_page_config(page_title="NBA Value Finder", page_icon="🏀", layout="wide", initial_sidebar_state="expanded")

PAGES = {
    "migliori": st.Page("views/best.py", title="Migliori giocate", icon=":material/emoji_events:", url_path="migliori-giocate", default=True),
    "builder": st.Page("views/builder.py", title="Bet Builder AI", icon=":material/layers:", url_path="bet-builder"),
    "overview": st.Page("views/overview.py", title="Panoramica", icon=":material/dashboard:", url_path="panoramica"),
    "partite": st.Page("views/matches.py", title="Partite NBA", icon=":material/sports_basketball:", url_path="partite"),
    "props": st.Page("views/props.py", title="Player Props", icon=":material/list:", url_path="props"),
    "value": st.Page("views/value.py", title="Value Finder", icon=":material/search:", url_path="value-finder"),
    "giocatore": st.Page("views/player.py", title="Analisi giocatore", icon=":material/person:", url_path="giocatore"),
    "registro": st.Page("views/track.py", title="Registro e taratura", icon=":material/fact_check:", url_path="registro"),
    "copertura": st.Page("views/coverage.py", title="Copertura dati", icon=":material/rule:", url_path="copertura"),
    "bookmaker": st.Page("views/bookmakers.py", title="Confronto bookmaker", icon=":material/compare_arrows:", url_path="bookmaker"),
    "backtest": st.Page("views/backtest.py", title="Storico e backtesting", icon=":material/query_stats:", url_path="backtesting"),
    "impostazioni": st.Page("views/settings.py", title="Impostazioni e API", icon=":material/settings:", url_path="impostazioni"),
}
st.session_state["_pages"] = PAGES
navigation = st.navigation({"Analisi": [PAGES[k] for k in ("migliori", "builder", "registro", "overview", "partite", "props", "value", "giocatore")],
                            "Strumenti": [PAGES[k] for k in ("copertura", "bookmaker", "backtest", "impostazioni")]})
navigation.run()
