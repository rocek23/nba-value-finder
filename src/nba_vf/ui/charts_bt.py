"""Grafici del backtest (Plotly, stesso stile del resto dell'interfaccia)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as T
from .charts import empty_fig, style

NICE = {"modello": "Modello", "modello_calibrato": "Modello ricalibrato", "moneta": "Moneta 50/50", "frequenza_ultime20": "Frequenza ultime 20",
        "poisson_ultime10": "Poisson ultime 10", "mercato_consenso": "Consenso di mercato"}
COLORS = {"modello": T.ACCENT, "modello_calibrato": T.CYAN, "moneta": T.GRAY, "frequenza_ultime20": T.AMBER, "poisson_ultime10": T.PURPLE,
          "mercato_consenso": T.GREEN}


def nice(name: str) -> str:
    if name.startswith("valore:"):
        return "Valore · " + NICE.get(name[7:], name[7:])
    if name.startswith("modello/"):
        return "Candidato · " + name.split("/", 1)[1].replace("_", " ")
    return NICE.get(name, name.replace("_", " ").capitalize())


def fig_loss_compare(tab: pd.DataFrame, metric: str = "brier") -> go.Figure:
    if tab.empty:
        return empty_fig("Nessuna osservazione")
    lo, hi = ("brier_lo", "brier_hi") if metric == "brier" else ("ll_lo", "ll_hi")
    v = tab["brier" if metric == "brier" else "log_loss"]
    fig = go.Figure(go.Bar(x=[nice(m) for m in tab["modello"]], y=v, marker_color=[COLORS.get(m, T.ACCENT) for m in tab["modello"]],
                           error_y=dict(type="data", symmetric=False, array=(tab[hi] - v).fillna(0), arrayminus=(v - tab[lo]).fillna(0), color=T.TEXT),
                           text=[f"{x:.3f}" for x in v], textposition="outside", cliponaxis=False))
    ref = 0.25 if metric == "brier" else float(np.log(2))
    fig.add_hline(y=ref, line_dash="dash", line_color=T.AMBER, annotation_text="Moneta 50/50")
    lo_y = max(0.0, float(min(v.min(), ref)) * 0.85)
    fig.update_yaxes(title_text=("Brier" if metric == "brier" else "Log-loss") + " (più basso = meglio)", range=[lo_y, float(max(v.max(), ref)) * 1.06])
    return style(fig, ("Brier score" if metric == "brier" else "Log-loss") + " con intervallo di incertezza", 340)


def fig_delta(tab: pd.DataFrame, metric: str = "log_loss") -> go.Figure:
    """Differenza modello - baseline con IC: a sinistra dello 0 il modello è migliore."""
    if tab.empty:
        return empty_fig("Confronto non disponibile (osservazioni comuni insufficienti)")
    col, lo, hi = ("delta_log_loss", "d_ll_lo", "d_ll_hi") if metric == "log_loss" else ("delta_brier", "d_brier_lo", "d_brier_hi")
    fig = go.Figure()
    fig.add_vline(x=0, line_color=T.GRAY, line_dash="dash")
    for r in tab.itertuples():
        v, l, h = getattr(r, col), getattr(r, lo), getattr(r, hi)
        color = T.GREEN if h < 0 else T.RED if l > 0 else T.AMBER
        fig.add_trace(go.Scatter(x=[v], y=[nice(r.baseline)], mode="markers", marker=dict(color=color, size=11),
                                 error_x=dict(type="data", symmetric=False, array=[h - v], arrayminus=[v - l], color=color), showlegend=False,
                                 hovertemplate=f"Δ {v:+.4f} (IC {l:+.4f} … {h:+.4f}) · n={r.n}<extra></extra>"))
    fig.update_xaxes(title_text=f"Δ {'log-loss' if metric == 'log_loss' else 'Brier'} modello − baseline (negativo = modello migliore)")
    return style(fig, "Il modello batte le baseline? (verde: sì, ambra: non distinguibile, rosso: peggio)", max(260, 110 + 60 * len(tab)))


def fig_reliability_ci(rel: pd.DataFrame, title: str = "Curva di affidabilità") -> go.Figure:
    if rel.empty:
        return empty_fig("Nessuna osservazione")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.GRAY, dash="dash"), name="Calibrazione perfetta")
    fig.add_scatter(x=rel["p_predetta"], y=rel["freq_osservata"], mode="lines+markers", name="Modello", marker=dict(size=8, color=T.ACCENT),
                    error_y=dict(type="data", symmetric=False, array=rel["hi"] - rel["freq_osservata"], arrayminus=rel["freq_osservata"] - rel["lo"], color=T.ACCENT),
                    text=[f"n={n}" for n in rel["n"]], hovertemplate="predetta %{x:.2f} · osservata %{y:.2f} (%{text})<extra></extra>")
    fig.update_xaxes(title_text="Probabilità predetta (Over)", range=[0, 1])
    fig.update_yaxes(title_text="Frequenza osservata (IC Wilson 95%)", range=[0, 1])
    return style(fig, title, 380)


def fig_pit_hist(pit: pd.Series, title: str = "PIT sulla distribuzione intera") -> go.Figure:
    pit = pit.dropna()
    if pit.empty:
        return empty_fig("Nessun dato")
    h, _ = np.histogram(pit, bins=10, range=(0, 1))
    fig = go.Figure(go.Bar(x=[f"{i / 10:.1f}" for i in range(10)], y=h / len(pit), marker_color=T.ACCENT))
    fig.add_hline(y=0.1, line_dash="dash", line_color=T.AMBER, annotation_text="Uniforme se calibrato")
    fig.update_yaxes(tickformat=".0%", title_text="Quota di previsioni")
    return style(fig, title + f" (n={len(pit)})", 340)


def fig_pnl(bets: pd.DataFrame, main: str = "valore:modello") -> go.Figure:
    """Profitto cumulato per strategia (ordine cronologico) e drawdown della strategia principale."""
    if bets.empty:
        return empty_fig("Nessuna scommessa simulata")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    for strat, g in bets.groupby("strategia" if "strategia" in bets else "strategy", sort=False):
        g = g.sort_values(["commence_time", "player", "market", "line"], kind="stable")
        cum = g["profit"].cumsum().to_numpy()
        fig.add_trace(go.Scatter(x=np.arange(1, len(g) + 1), y=cum, mode="lines", name=nice(strat), line=dict(width=3 if strat == main else 1.6)), row=1, col=1)
        if strat == main:
            peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
            fig.add_trace(go.Scatter(x=np.arange(1, len(g) + 1), y=-(peak - cum), mode="lines", fill="tozeroy", name="Drawdown", line=dict(color=T.RED, width=1),
                                     fillcolor="rgba(255,92,108,.25)"), row=2, col=1)
    fig.add_hline(y=0, line_color=T.GRAY, line_dash="dash", row=1, col=1)
    fig.update_yaxes(title_text="Profitto (unità)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_xaxes(title_text="Scommessa n. (ordine cronologico)", row=2, col=1)
    return style(fig, "Profitto cumulato simulato (puntata fissa = 1 unità, quote reali con margine)", 520)


def fig_yield_forest(tab: pd.DataFrame) -> go.Figure:
    if tab.empty:
        return empty_fig("Nessuna scommessa simulata")
    fig = go.Figure()
    fig.add_vline(x=0, line_color=T.GRAY, line_dash="dash")
    for _, r in tab.iterrows():
        y, lo, hi = r["yield"], r["yield_lo"], r["yield_hi"]
        color = T.GREEN if lo > 0 else T.RED if hi < 0 else T.AMBER
        fig.add_trace(go.Scatter(x=[100 * y], y=[nice(r["strategia"])], mode="markers", marker=dict(color=color, size=11), showlegend=False,
                                 error_x=dict(type="data", symmetric=False, array=[100 * (hi - y)], arrayminus=[100 * (y - lo)], color=color),
                                 hovertemplate=f"yield {100 * y:+.1f}% (IC {100 * lo:+.1f}% … {100 * hi:+.1f}%) · n={int(r['n_scommesse'])}<extra></extra>"))
    fig.update_xaxes(title_text="Yield % per unità puntata (con intervallo di incertezza)")
    return style(fig, "Yield per strategia (verde: IC sopra 0 · ambra: non distinguibile da 0 · rosso: sotto 0)", max(280, 120 + 56 * len(tab)))


def fig_breakdown(tab: pd.DataFrame, by: str, value: str, lo: str | None, hi: str | None, title: str, pct: bool = False, ref: float | None = None) -> go.Figure:
    if tab.empty:
        return empty_fig("Nessun dato")
    k = 100 if pct else 1
    v = tab[value] * k
    err = None
    if lo and hi and tab[lo].notna().any():
        err = dict(type="data", symmetric=False, array=((tab[hi] * k) - v).fillna(0), arrayminus=(v - tab[lo] * k).fillna(0), color=T.TEXT)
    fig = go.Figure(go.Bar(x=tab[by].astype(str), y=v, marker_color=[T.ACCENT if c == "ok" else T.GRAY for c in tab["campione"]], error_y=err,
                           text=[f"n={n}" for n in tab.get("n_scommesse", tab.get("n_osservazioni"))], textposition="outside", cliponaxis=False))
    if ref is not None:
        fig.add_hline(y=ref * k, line_dash="dash", line_color=T.AMBER)
    fig.update_yaxes(title_text=value + (" %" if pct else ""))
    return style(fig, title + " — barre grigie: campione insufficiente, senza intervallo", 340)


def fig_margins(m: dict) -> go.Figure:
    if not m:
        return empty_fig("Nessuna coppia Over/Under dello stesso prelievo: margine non misurabile")
    d = pd.DataFrame([{"book": k, **v} for k, v in m.items()]).sort_values("margine_medio")
    fig = go.Figure(go.Bar(x=100 * d["margine_medio"], y=d["book"], orientation="h", marker_color=T.PURPLE,
                           text=[f"{100 * x:.1f}% (n={n})" for x, n in zip(d["margine_medio"], d["coppie"])], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title_text="Margine medio (overround) %")
    return style(fig, "Margine dei bookmaker nelle quote storiche usate", max(240, 80 + 42 * len(d)))
