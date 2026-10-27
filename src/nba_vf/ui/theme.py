"""Tema scuro: palette, CSS e piccoli componenti HTML (i testi inseriti sono sempre escapati)."""

from __future__ import annotations

from html import escape

import streamlit as st

BG = "#0A0F1C"
SURFACE = "#111827"
SURFACE2 = "#172033"
BORDER = "#243049"
TEXT = "#E5E9F2"
MUTED = "#8B97B0"
ACCENT = "#5B8CFF"
GREEN = "#2ECC71"
RED = "#FF5C6C"
AMBER = "#F5B942"
CYAN = "#38BDF8"
PURPLE = "#A78BFA"
GRAY = "#6B7690"
COLORWAY = [ACCENT, CYAN, GREEN, AMBER, PURPLE, RED]

CSS = f"""
<style>
.block-container {{ padding-top: 4rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-size: 1.7rem !important; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
.nvf-kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px; height: 100%; }}
.nvf-kpi .l {{ color: {MUTED}; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; }}
.nvf-kpi .v {{ font-size: 1.85rem; font-weight: 650; line-height: 1.15; margin-top: 2px; }}
.nvf-kpi .s {{ color: {MUTED}; font-size: .78rem; margin-top: 2px; }}
.nvf-badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 600;
  border: 1px solid {BORDER}; background: {SURFACE2}; margin-right: 6px; }}
.nvf-demo {{ background: rgba(245,185,66,.12); border: 1px solid {AMBER}; color: {AMBER}; border-radius: 10px;
  padding: 8px 14px; font-size: .86rem; margin-bottom: 14px; }}
.nvf-live {{ display: inline-block; background: rgba(46,204,113,.10); border: 1px solid {GREEN}55; color: {GREEN}; border-radius: 999px;
  padding: 3px 12px; font-size: .78rem; margin: 0 0 10px; line-height: 1.4; }}
.nvf-note {{ color: {MUTED}; font-size: .8rem; }}
.nvf-reason {{ background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {ACCENT}; border-radius: 8px;
  padding: 12px 16px; font-size: .92rem; line-height: 1.55; }}
.nvf-pos {{ color: {GREEN}; }} .nvf-neg {{ color: {RED}; }} .nvf-warn {{ color: {AMBER}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
button[kind="primary"] {{ font-weight: 600; }}
footer {{ visibility: hidden; }}
</style>
"""


def inject_css() -> None:
    from .widgets import CSS as WIDGET_CSS

    st.markdown(CSS + WIDGET_CSS, unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="nvf-kpi"><div class="l">{escape(label)}</div><div class="v"{style}>{escape(value)}</div>'
            f'<div class="s">{escape(sub)}</div></div>')


def badge(text: str, color: str = MUTED) -> str:
    return f'<span class="nvf-badge" style="color:{color}">{escape(text)}</span>'


def kpi_row(items: list[tuple[str, str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub, color) in zip(cols, items):
        col.markdown(kpi(label, value, sub, color), unsafe_allow_html=True)
