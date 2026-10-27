"""Piccoli componenti HTML dello stile «dashboard»: avatar con iniziali, sigle squadra, pillole e tabelle. Tutto il testo e' escapato.

Niente foto dei giocatori e niente loghi: nessuna delle fonti dati li fornisce con licenza. Al loro posto cerchi colorati con le iniziali.
"""

from __future__ import annotations

import zlib
from html import escape

from . import theme as T
from .labels import TEAM_NAMES

_PALETTE = ["#5B8CFF", "#38BDF8", "#2ECC71", "#F5B942", "#A78BFA", "#FF7A90", "#22D3B7", "#F59E5B"]


def _color(text: str) -> str:
    return _PALETTE[zlib.crc32(str(text).encode()) % len(_PALETTE)]


def initials(name: str) -> str:
    parts = [p for p in str(name).replace(".", " ").split() if p and p[0].isalpha()]
    if not parts:
        return "?"
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


def avatar(name: str, size: int = 28) -> str:
    c = _color(name)
    return (f'<span class="nvf-av" style="width:{size}px;height:{size}px;background:{c}26;border-color:{c};color:{c};'
            f'font-size:{size * 0.38:.0f}px">{escape(initials(name))}</span>')


def team_chip(abbr: str | None, size: int = 26) -> str:
    if not abbr:
        return "—"
    c = _color(abbr)
    name = TEAM_NAMES.get(str(abbr).upper(), "")
    label = escape(name or str(abbr))
    return (f'<span class="nvf-team"><span class="nvf-av" style="width:{size}px;height:{size}px;background:{c}26;border-color:{c};color:{c};'
            f'font-size:{size * 0.34:.0f}px">{escape(str(abbr)[:3].upper())}</span>{label}</span>')


def player_cell(name: str) -> str:
    return f'<span class="nvf-team">{avatar(name)}{escape(str(name))}</span>'


def pill(text: str, color: str = T.GREEN, title: str = "") -> str:
    tip = f' title="{escape(title, quote=True)}"' if title else ""
    return f'<span class="nvf-pill" style="color:{color};background:{color}22"{tip}>{escape(str(text))}</span>'


def reliability_pill(level: str, long: bool = False) -> str:
    from .best import RELIABILITY_HELP

    text = f"Affidabilità {level.lower()}" if long else level
    return pill(text, {"Alta": T.GREEN, "Media": T.AMBER}.get(level, T.GRAY), RELIABILITY_HELP.get(level, ""))


def icon(name: str) -> str:
    return f'<span class="material-symbols-rounded nvf-ic">{escape(name)}</span>'


def card_kpi(icon_name: str, value: str, label: str, sub: str = "") -> str:
    return (f'<div class="nvf-card"><div class="nvf-ico">{icon(icon_name)}</div><div><div class="v">{escape(value)}</div>'
            f'<div class="l">{escape(label)}</div>' + (f'<div class="s">{escape(sub)}</div>' if sub else "") + "</div></div>")


def table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    """Tabella HTML. Le celle sono HTML GIA' costruito con gli helper qui sopra (che escapano): vedi `escape` per testo libero."""
    align = align or ["left"] * len(headers)
    head = "".join(f'<th style="text-align:{a}">{escape(h)}</th>' for h, a in zip(headers, align))
    body = "".join("<tr>" + "".join(f'<td style="text-align:{a}">{c}</td>' for c, a in zip(r, align)) + "</tr>" for r in rows)
    return f'<div class="nvf-tbl-wrap"><table class="nvf-tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');
.nvf-ic {{ font-family: 'Material Symbols Rounded'; font-size: 26px; line-height: 1; font-weight: 400; font-style: normal; }}
.nvf-hello {{ display:flex; justify-content:space-between; align-items:flex-end; margin: 4px 0 16px; }}
.nvf-hello h1 {{ margin:0; padding:0; font-size:1.9rem !important; }}
.nvf-hello .sub {{ color:{T.MUTED}; font-size:.9rem; margin-top:2px; }}
.nvf-hello .upd {{ color:{T.MUTED}; font-size:.78rem; text-align:right; }}
.nvf-cards {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin-bottom:18px; }}
@media (max-width: 900px) {{ .nvf-cards {{ grid-template-columns: repeat(2, 1fr); }} }}
.nvf-card {{ display:flex; gap:14px; align-items:center; background:linear-gradient(160deg,{T.SURFACE2},{T.SURFACE});
  border:1px solid {T.BORDER}; border-radius:14px; padding:16px 18px; }}
.nvf-card .nvf-ico {{ color:{T.ACCENT}; background:{T.ACCENT}1f; border-radius:12px; width:48px; height:48px; display:flex; align-items:center; justify-content:center; }}
.nvf-card .nvf-ic {{ font-size:28px; }}
.nvf-card .v {{ font-size:1.8rem; font-weight:700; line-height:1.1; }}
.nvf-card .l {{ color:{T.MUTED}; font-size:.85rem; }}
.nvf-card .s {{ color:{T.ACCENT}; font-size:.74rem; margin-top:1px; }}
.nvf-panel {{ background:{T.SURFACE}; border:1px solid {T.BORDER}; border-radius:14px; padding:16px 18px 8px; margin-bottom:14px; }}
.nvf-panel h3 {{ margin:0 0 10px; font-size:1.05rem; }}
.nvf-tbl-wrap {{ overflow-x:auto; }}
.nvf-tbl {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
.nvf-tbl th, .nvf-tbl td {{ border-left:none !important; border-right:none !important; border-top:none !important; }}
.nvf-tbl th {{ color:{T.MUTED}; font-weight:500; font-size:.76rem; padding:8px 10px; border-bottom:1px solid {T.BORDER} !important; white-space:nowrap; background:transparent; }}
.nvf-tbl td {{ padding:9px 10px; border-bottom:1px solid {T.BORDER}55 !important; white-space:nowrap; }}
.nvf-tbl tr:last-child td {{ border-bottom:none; }}
.nvf-tbl tbody tr:hover {{ background:{T.SURFACE2}; }}
.nvf-av {{ display:inline-flex; align-items:center; justify-content:center; border-radius:999px; border:1.5px solid; font-weight:700; flex:none; }}
.nvf-team {{ display:inline-flex; align-items:center; gap:9px; }}
.nvf-pill {{ display:inline-block; padding:2px 9px; border-radius:7px; font-weight:600; font-size:.8rem; }}
.nvf-link {{ text-align:right; }}
.nvf-muted {{ color:{T.MUTED}; }}
button[data-baseweb="tab"] {{ font-weight:600; }}
</style>
"""
