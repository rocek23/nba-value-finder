"""Pagina Streamlit (lancia il rendering dal pacchetto nba_vf.ui)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_vf.ui.pages.coverage import render  # noqa: E402

render()
