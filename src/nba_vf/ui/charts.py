"""Grafici Plotly (funzioni pure che restituiscono Figure, stile scuro coerente)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..model.empirical import stat_value
from ..models import Market, PlayerGameLog
from . import theme as T


def style(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        title=dict(text=title, x=0.0, font=dict(size=15)) if title else None,
        margin=dict(l=10, r=10, t=64 if title else 14, b=10), colorway=T.COLORWAY,
        font=dict(family="Inter, system-ui, sans-serif", color=T.TEXT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=T.SURFACE2, bordercolor=T.BORDER, font=dict(color=T.TEXT)),
    )
    fig.update_xaxes(gridcolor="rgba(139,151,176,.12)", zerolinecolor=T.BORDER, linecolor=T.BORDER)
    fig.update_yaxes(gridcolor="rgba(139,151,176,.12)", zerolinecolor=T.BORDER, linecolor=T.BORDER)
    return fig


def empty_fig(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=T.MUTED, size=14), xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=height)


def _label(l: PlayerGameLog) -> str:
    return f"{l.game_date:%d/%m} {'vs' if l.is_home else '@'} {l.opponent or '?'}" if l.is_home is not None else f"{l.game_date:%d/%m} {l.opponent or ''}".strip()


# ---------------------------------------------------------------------------------------------
def fig_last_games(logs: Sequence[PlayerGameLog], market: Market, line: float | None, n: int = 20, title: str | None = None) -> go.Figure:
    if not logs:
        return empty_fig("Nessuna partita disponibile")
    vals_all = np.array([stat_value(l, market) for l in logs], dtype=float)
    roll5 = pd.Series(vals_all).rolling(5, min_periods=3).mean().to_numpy()
    last = list(logs)[-n:]
    vals = vals_all[-n:]
    labels = [_label(l) for l in last]
    if line is None:
        colors = [T.ACCENT] * len(vals)
    else:
        colors = [T.GREEN if v > line else T.AMBER if v == line else T.RED for v in vals]
    fig = go.Figure()
    fig.add_bar(x=labels, y=vals, marker_color=colors, name="Valore", showlegend=False, hovertemplate="%{x}<br>%{y}<extra></extra>", opacity=0.9)
    fig.add_scatter(x=labels, y=roll5[-n:], mode="lines", name="Media mobile 5", line=dict(color=T.CYAN, width=2))
    if line is not None:
        fig.add_hline(y=line, line_dash="dash", line_color=T.TEXT, annotation_text=f"Linea {line:g}", annotation_position="top left")
    season = [l for l in logs if l.season == logs[-1].season]
    if season:
        fig.add_hline(y=float(np.mean([stat_value(l, market) for l in season])), line_dash="dot", line_color=T.GRAY,
                      annotation_text="Media stagione", annotation_position="bottom right")
    fig.update_xaxes(tickangle=-45)
    return style(fig, title, 380)


def fig_minutes_attempts(logs: Sequence[PlayerGameLog], n: int = 20) -> go.Figure:
    if not logs:
        return empty_fig("Nessuna partita disponibile")
    last = list(logs)[-n:]
    labels = [_label(l) for l in last]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=labels, y=[l.minutes for l in last], name="Minuti", mode="lines+markers", line=dict(color=T.ACCENT), secondary_y=False)
    fig.add_bar(x=labels, y=[l.fg3a for l in last], name="Tentativi da 3", marker_color="rgba(167,139,250,.55)", secondary_y=True)
    fig.update_yaxes(title_text="Minuti", secondary_y=False)
    fig.update_yaxes(title_text="Tentativi da 3", secondary_y=True, showgrid=False)
    fig.update_xaxes(tickangle=-45)
    return style(fig, "Minuti e tentativi da tre", 300)


def fig_hit_rates(hits: dict[str, dict]) -> go.Figure:
    names = {"ultime_5": "Ultime 5", "ultime_10": "Ultime 10", "ultime_20": "Ultime 20", "tutte": "Tutte"}
    x = [names.get(k, k) for k in hits]
    fig = go.Figure()
    for key, color, label in (("freq_over", T.GREEN, "Over"), ("freq_push", T.AMBER, "Rimborso (push)"), ("freq_under", T.RED, "Under")):
        fig.add_bar(x=x, y=[100 * hits[k][key] for k in hits], name=label, marker_color=color,
                    text=[f"{100 * hits[k][key]:.0f}%" for k in hits], textposition="inside")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="% partite", range=[0, 100])
    return style(fig, "Frequenza storica rispetto alla linea", 300)


def fig_distribution(pmf: np.ndarray, line: float, side: str, reps: np.ndarray | None = None, ci: float = 0.9) -> go.Figure:
    s = np.arange(len(pmf))
    top = int(min(len(pmf) - 1, np.searchsorted(np.cumsum(pmf), 0.9995) + 2))
    x, p = s[: top + 1], pmf[: top + 1]
    is_int = abs(line - round(line)) < 1e-9
    if side == "over":
        win = x > line
    else:
        win = x < line
    push = np.isclose(x, line) if is_int else np.zeros_like(x, dtype=bool)
    colors = np.where(push, T.AMBER, np.where(win, T.GREEN, "rgba(139,151,176,.45)"))
    err = None
    if reps is not None and len(reps) > 4:
        lo = np.quantile(reps[:, : top + 1], (1 - ci) / 2, axis=0)
        hi = np.quantile(reps[:, : top + 1], 1 - (1 - ci) / 2, axis=0)
        err = dict(type="data", symmetric=False, array=np.maximum(hi - p, 0), arrayminus=np.maximum(p - lo, 0), color=T.MUTED, thickness=1)
    fig = go.Figure()
    fig.add_bar(x=x, y=p, marker_color=list(colors), error_y=err, hovertemplate="%{x}: %{y:.1%}<extra></extra>")
    fig.add_vline(x=line, line_dash="dash", line_color=T.TEXT, annotation_text=f"Linea {line:g}")
    fig.update_yaxes(tickformat=".0%", title_text="Probabilità")
    fig.update_xaxes(title_text="Valore finale", dtick=1 if top <= 30 else 5)
    fig.update_layout(showlegend=False)
    return style(fig, f"Distribuzione prevista — verde: {'Over' if side == 'over' else 'Under'} vince" + (" · barre d'errore: IC stima" if err else ""), 340)


def fig_odds_by_book(df: pd.DataFrame, fair: float | None, mean: float | None, side_label: str) -> go.Figure:
    """df: colonne bookmaker, quota. Barre orizzontali con la quota equa del modello e la media tra book."""
    if df.empty:
        return empty_fig("Nessuna quota")
    d = df.sort_values("quota")
    fig = go.Figure(go.Bar(x=d["quota"], y=d["bookmaker"], orientation="h", marker_color=T.ACCENT,
                           text=[f"{q:.2f}" for q in d["quota"]], textposition="outside", cliponaxis=False))
    if fair and np.isfinite(fair):
        fig.add_vline(x=fair, line_color=T.GREEN, line_dash="dash", annotation_text=f"Quota equa {fair:.2f}", annotation_position="top")
    if mean and np.isfinite(mean):
        fig.add_vline(x=mean, line_color=T.AMBER, line_dash="dot", annotation_text=f"Media book {mean:.2f}", annotation_position="bottom")
    lo = float(min(d["quota"].min(), fair if fair and np.isfinite(fair) else 99)) * 0.9
    fig.update_xaxes(range=[max(1.0, lo), float(max(d["quota"].max(), fair if fair and np.isfinite(fair) else 0)) * 1.12])
    fig.update_layout(showlegend=False)
    return style(fig, f"Quote {side_label} per bookmaker", max(240, 70 + 42 * len(d)))


def fig_ev_scatter(rows: pd.DataFrame) -> go.Figure:
    d = rows[rows["priced"]]
    if d.empty:
        return empty_fig("Nessuna quota prezzata con i filtri correnti")
    fig = go.Figure()
    for mkt, g in d.groupby("mercato"):
        fig.add_scatter(
            x=g["p_win"], y=100 * g["roi"], mode="markers", name=mkt,
            marker=dict(size=8, opacity=0.8),
            customdata=np.stack([g["giocatore"], g["lato"], g["linea"], g["bookmaker"], g["quota"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]} %{customdata[2]} @ %{customdata[4]} (%{customdata[3]})<br>"
                          "P vittoria %{x:.1%} · ROI atteso %{y:.1f}%<extra></extra>",
        )
    fig.add_hline(y=0, line_color=T.GRAY, line_dash="dash")
    fig.update_xaxes(title_text="Probabilità di vittoria stimata", tickformat=".0%")
    fig.update_yaxes(title_text="ROI atteso (%)")
    return style(fig, "Probabilità stimata vs ROI atteso", 380)


def fig_ev_hist(rows: pd.DataFrame) -> go.Figure:
    d = rows[rows["priced"]]
    if d.empty:
        return empty_fig("Nessuna quota prezzata")
    fig = go.Figure(go.Histogram(x=100 * d["roi"], nbinsx=40, marker_color=T.ACCENT, opacity=0.85))
    fig.add_vline(x=0, line_color=T.AMBER, line_dash="dash")
    fig.update_xaxes(title_text="ROI atteso (%)")
    fig.update_yaxes(title_text="Numero di quote")
    return style(fig, "Distribuzione del ROI atteso", 300)


def fig_games_per_day(games: pd.DataFrame) -> go.Figure:
    if games.empty:
        return empty_fig("Nessuna partita nello storico")
    d = games.groupby("data").size().reset_index(name="partite")
    return style(go.Figure(go.Bar(x=d["data"], y=d["partite"], marker_color=T.ACCENT)).update_yaxes(title_text="Partite"), "Partite per giorno nello storico", 280)


def fig_reliability(rep: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.GRAY, dash="dash"), name="Calibrazione perfetta")
    for mk, r in rep.items():
        t = r["affidabilita"]
        fig.add_scatter(x=t["p_media_predetta"], y=t["freq_osservata"], mode="lines+markers", name=mk,
                        text=[f"n={n}" for n in t["n"]], hovertemplate="predetta %{x:.2f} · osservata %{y:.2f} (%{text})<extra></extra>")
    fig.update_xaxes(title_text="Probabilità predetta (Over)", range=[0, 1])
    fig.update_yaxes(title_text="Frequenza osservata", range=[0, 1])
    return style(fig, "Curva di affidabilità", 380)


def fig_pit(pit: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for mk, g in pit.groupby("mercato"):
        h, _ = np.histogram(g["pit"], bins=10, range=(0, 1))
        fig.add_bar(x=[f"{i / 10:.1f}" for i in range(10)], y=h / max(1, len(g)), name=mk)
    fig.add_hline(y=0.1, line_dash="dash", line_color=T.AMBER, annotation_text="Uniforme se calibrato")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title_text="Quota di partite")
    return style(fig, "Istogramma PIT (uniforme se la distribuzione è calibrata)", 340)


def fig_brier(rep: dict[str, dict]) -> go.Figure:
    names = list(rep)
    fig = go.Figure(go.Bar(x=names, y=[rep[m]["brier"] for m in names], marker_color=T.ACCENT,
                           text=[f"{rep[m]['brier']:.3f}" for m in names], textposition="outside"))
    fig.add_hline(y=0.25, line_dash="dash", line_color=T.AMBER, annotation_text="Riferimento 50/50 = 0.25")
    fig.update_yaxes(title_text="Brier score (più basso = meglio)", range=[0, 0.3])
    return style(fig, "Brier score per mercato", 320)


def fig_book_margin(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_fig("Nessun bookmaker con entrambi i lati quotati")
    d = df.sort_values("margine_medio")
    fig = go.Figure(go.Bar(x=100 * d["margine_medio"], y=d["bookmaker"], orientation="h", marker_color=T.PURPLE,
                           text=[f"{100 * m:.1f}%" for m in d["margine_medio"]], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title_text="Margine medio (overround) %")
    return style(fig, "Margine medio per bookmaker", max(240, 70 + 42 * len(d)))


def fig_pick_calibration(tab: pd.DataFrame) -> go.Figure:
    """Probabilita' prevista contro frequenza realizzata per le giocate consigliate gia' concluse (con intervallo al 90%)."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=T.GRAY, dash="dash"), name="Previsto = realizzato")
    if len(tab):
        fig.add_scatter(
            x=tab["previsto"], y=tab["realizzato"], mode="markers+text", name="Giocate chiuse",
            marker=dict(color=T.ACCENT, size=12), text=[f"n={n}" for n in tab["giocate"]], textposition="top center",
            error_y=dict(type="data", symmetric=False, array=tab["a"] - tab["realizzato"], arrayminus=tab["realizzato"] - tab["da"],
                         color=T.MUTED),
            hovertemplate="previsto %{x:.0f}% · realizzato %{y:.0f}%<extra></extra>",
        )
    fig.update_xaxes(title_text="Probabilità prevista (%)", range=[0, 100])
    fig.update_yaxes(title_text="Realizzata (%)", range=[0, 100])
    return style(fig, "Previsto contro realizzato", 380)
