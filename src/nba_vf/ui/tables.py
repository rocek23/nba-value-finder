"""Preparazione delle tabelle da mostrare (nomi italiani, formati, colori) e filtri comuni."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

ESITO = {
    "positivo_robusto": "EV stimato > 0 (anche nel limite inferiore)",
    "positivo_incerto": "EV stimato > 0, incerto",
    "non_positivo": "EV stimato ≤ 0",
    "non_prezzata": "Non prezzata",
}
COLUMN_ORDER = ["giocatore", "squadra", "evento", "mercato", "linea", "lato", "bookmaker", "quota", "p_win", "p_push",
                "quota_equa", "roi", "ev", "roi_lo", "roi_hi", "esito", "n_partite", "eta_min", "avvisi"]


def best_per_prop(df: pd.DataFrame, include_stale: bool = False) -> pd.DataFrame:
    """Per ogni prop e lato tiene la quota piu' alta (le quote obsolete solo se richiesto)."""
    d = df if include_stale else df[~df["obsoleta"]]
    if d.empty:
        return d
    return d.loc[d.groupby(["prop_id", "lato"])["quota"].idxmax()]


def display_frame(df: pd.DataFrame, stake: float, columns: list[str] | None = None) -> pd.DataFrame:
    d = df.copy()
    d["esito"] = d["classe"].map(ESITO)
    for c in ("p_win", "p_push", "roi", "roi_lo", "roi_hi"):
        d[c] = 100 * d[c]
    cols = [c for c in (columns or COLUMN_ORDER) if c in d.columns]
    names = {
        "giocatore": "Giocatore", "squadra": "Sq.", "evento": "Partita", "mercato": "Mercato", "linea": "Linea", "lato": "Lato",
        "bookmaker": "Bookmaker", "quota": "Quota", "p_win": "P vittoria", "p_push": "P rimborso", "quota_equa": "Quota equa",
        "roi": "ROI atteso", "ev": f"EV ({stake:g} €)", "roi_lo": "ROI min (IC)", "roi_hi": "ROI max (IC)", "esito": "Esito",
        "n_partite": "Partite", "eta_min": "Età quota (min)", "avvisi": "Avvisi",
    }
    return d[cols].rename(columns=names)


def _color(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return "color: #2ECC71" if v > 0 else "color: #FF5C6C" if v < 0 else ""


def style_frame(d: pd.DataFrame):
    fmt = {}
    for c in d.columns:
        if c in ("P vittoria", "P rimborso", "ROI atteso", "ROI min (IC)", "ROI max (IC)"):
            fmt[c] = "{:.1f}%"
        elif c in ("Quota", "Quota equa", "Linea"):
            fmt[c] = "{:.2f}" if c != "Linea" else "{:g}"
        elif c.startswith("EV ("):
            fmt[c] = "{:+.2f}"
        elif c == "Età quota (min)":
            fmt[c] = "{:.0f}"
        elif c == "Partite":
            fmt[c] = "{:.0f}"
    sty = d.style.format(fmt, na_rep="—")
    color_cols = [c for c in d.columns if c in ("ROI atteso", "ROI min (IC)", "ROI max (IC)") or c.startswith("EV (")]
    if color_cols:
        sty = sty.map(_color, subset=color_cols)
    return sty


def show_table(df: pd.DataFrame, stake: float, columns: list[str] | None = None, height: int = 460, max_rows: int = 2000) -> None:
    d = display_frame(df.head(max_rows), stake, columns)
    st.dataframe(style_frame(d), hide_index=True, width="stretch", height=height)
    if len(df) > max_rows:
        st.caption(f"Mostrate le prime {max_rows} righe su {len(df)}: restringi i filtri per vedere le altre.")


def csv_download(df: pd.DataFrame, stake: float, name: str, columns: list[str] | None = None) -> None:
    d = display_frame(df, stake, columns)
    st.download_button("Scarica CSV", d.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")
