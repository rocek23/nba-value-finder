"""Analisi giocatore: storico, medie, distribuzione prevista, confronto quote, motivazione e dati mancanti."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ...model.empirical import hit_rates, stat_value
from ...model.errors import InsufficientDataError, ModelError
from ...models import Market, Side
from .. import charts, reasoning, shell
from .. import theme as T
from ..labels import MARKET_FROM_IT, MARKET_IT, fmt_age, normalize_name, pct
from ..session import detail_prediction

PAGE = "Analisi giocatore"


def _mean(vals, n=None):
    v = vals[-n:] if n else vals
    return float(np.mean(v)) if len(v) else None


def _selection(rows, b):
    sel = st.session_state.get("sel_prop") or {}
    key = market = line = None
    if sel.get("prop_id") and sel["prop_id"].count("|") >= 3:
        _, key, mval, line_txt = sel["prop_id"].split("|", 3)
        market, line = Market(mval), float(line_txt)
    elif sel.get("key"):
        key = sel["key"]
    return key, market, line, sel.get("side", "Over")


def render() -> None:
    ctx = shell.begin(PAGE, "Storico, distribuzione prevista, confronto quote e motivazione della stima")
    b, board, rows = ctx.bundle, ctx.board, ctx.rows
    keys = sorted({*b.logs, *b.players, *(rows["key"].unique() if len(rows) else [])}, key=lambda k: b.player_name(k))
    if not keys:
        st.warning("Nessun giocatore disponibile: carica prima i dati (modalità live) o passa alla demo.")
        st.stop()
    n = st.session_state.get("ap_nonce", 0)
    s_key, s_market, s_line, s_side = _selection(rows, b)
    names = {b.player_name(k): k for k in keys}
    name_list = list(names)

    c1, c2, c3, c4 = st.columns([3, 2, 2, 1.4])
    idx = name_list.index(b.player_name(s_key)) if s_key in keys else 0
    key = names[c1.selectbox("Giocatore", name_list, index=idx, key=f"ap_player_{n}")]
    mk_labels = list(MARKET_FROM_IT)
    m_idx = mk_labels.index(MARKET_IT[s_market]) if s_market else 0
    market = MARKET_FROM_IT[c2.selectbox("Mercato", mk_labels, index=m_idx, key=f"ap_market_{n}")]

    mine = rows[(rows["key"] == key) & (rows["market"] == market)] if len(rows) else rows
    quoted = sorted(mine["linea"].unique()) if len(mine) else []
    logs = b.logs.get(key, [])
    vals_all = np.array([stat_value(l, market) for l in logs], dtype=float)
    default_line = float(np.floor(np.median(vals_all[-20:])) + 0.5) if len(vals_all) else 0.5
    use_custom = c3.toggle("Linea personalizzata", value=not quoted, key=f"ap_cust_{n}", disabled=not quoted,
                           help="Se non ci sono quote per questo mercato la linea è sempre manuale.")
    if use_custom or not quoted:
        line = c3.number_input("Linea", min_value=0.5, max_value=150.0, value=default_line, step=0.5, key=f"ap_cline_{n}")
    else:
        li = quoted.index(s_line) if s_line in quoted else 0
        line = float(c3.selectbox("Linea quotata", quoted, index=li, format_func=lambda x: f"{x:g}", key=f"ap_line_{n}"))
    side = c4.radio("Lato", ["Over", "Under"], index=0 if s_side == "Over" else 1, horizontal=True, key=f"ap_side_{n}")
    side_enum = Side.OVER if side == "Over" else Side.UNDER

    info = b.players.get(key)
    st.markdown(T.badge(b.player_name(key), T.TEXT) + T.badge(info.team if info and info.team else "squadra n/d")
                + T.badge(f"{MARKET_IT[market]} {side} {line:g}", T.ACCENT) + T.badge(f"{len(logs)} partite nello storico"), unsafe_allow_html=True)

    with st.expander("Scenario manuale: minuti attesi (es. compagno assente)"):
        mo = st.number_input("Minuti attesi (0 = usa la proiezione del modello)", 0.0, 48.0, 0.0, 1.0, key=f"ap_min_{n}",
                             help="Valore scelto da te, non stimato: il modello non conosce assenze o rotazioni.")
    minutes_override = float(mo) if mo and mo > 0 else None

    if not logs:
        st.error("Nessuno storico partite per questo giocatore: non è possibile calcolare medie né stime.")
        for lvl, txt in reasoning.missing_data(None, n_logs=0, skipped_reason=None, issues=b.collection_issues.get(key, []), market=market,
                                               n_books=0, n_stale=0, has_opponent=False, mode=b.mode):
            st.markdown(f"- {_icon(lvl)} {txt}")
        st.stop()

    # ---- medie e frequenze storiche ----------------------------------------------------------
    season_logs = [l for l in logs if l.season == logs[-1].season]
    sv = np.array([stat_value(l, market) for l in season_logs], dtype=float)
    hits = hit_rates(logs, market, line)
    season_hits = hit_rates(season_logs, market, line)["tutte"]
    key_side = "freq_over" if side == "Over" else "freq_under"
    T.kpi_row([
        (f"Media stagione ({len(season_logs)} partite)", f"{_mean(sv):.1f}", f"{MARKET_IT[market]} · stagione {logs[-1].season}-{(logs[-1].season + 1) % 100:02d}", None),
        ("Ultime 5", f"{_mean(vals_all, 5):.1f}", "media", None),
        ("Ultime 10", f"{_mean(vals_all, 10):.1f}", "media", None),
        ("Ultime 20", f"{_mean(vals_all, 20):.1f}", "media", None),
        (f"{side} {line:g} storico", pct(season_hits[key_side], 0), f"ultime 10: {pct(hits['ultime_10'][key_side], 0)} · ultime 20: {pct(hits['ultime_20'][key_side], 0)}"
         + (f" · rimborso {pct(season_hits['freq_push'], 0)}" if season_hits["freq_push"] > 0 else ""), None),
    ])
    st.write("")
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.fig_last_games(logs, market, line, 20, f"Ultime 20 partite — {MARKET_IT[market]}"), width="stretch")
    right.plotly_chart(charts.fig_hit_rates(hits), width="stretch")

    avg_rows = []
    for label, fn in (("Minuti", lambda l: l.minutes), ("Tiri tentati (FGA)", lambda l: l.fga), ("Tentativi da 3", lambda l: l.fg3a), ("Tiri liberi (FTA)", lambda l: l.fta)):
        arr = np.array([fn(l) for l in logs], dtype=float)
        sarr = np.array([fn(l) for l in season_logs], dtype=float)
        avg_rows.append({"": label, "Stagione": _mean(sarr), "Ultime 5": _mean(arr, 5), "Ultime 10": _mean(arr, 10), "Ultime 20": _mean(arr, 20)})
    a1, a2 = st.columns([2, 3])
    a1.markdown("**Minuti e tentativi medi**")
    a1.dataframe(pd.DataFrame(avg_rows).style.format("{:.1f}", subset=["Stagione", "Ultime 5", "Ultime 10", "Ultime 20"]), hide_index=True, width="stretch")
    a2.plotly_chart(charts.fig_minutes_attempts(logs, 20), width="stretch")

    # ---- modello -------------------------------------------------------------------------------
    st.subheader("Stima del modello")
    pred, skipped = None, None
    try:
        pred = detail_prediction(b, key, minutes_override)
    except InsufficientDataError as exc:
        skipped = str(exc)
    except ModelError as exc:
        skipped = f"errore del modello: {exc}"
    mp = pred.markets.get(market) if pred else None
    if pred is not None and mp is None:
        skipped = "tentativi da tre insufficienti per prezzare le triple" if market is Market.THREES else "mercato non prezzabile"
    book_rows = mine[np.isclose(mine["linea"], line)] if len(mine) else mine
    n_books = book_rows["book"].nunique() if len(book_rows) else 0
    n_stale = int(book_rows["obsoleta"].sum()) if len(book_rows) else 0
    first_q = next((q for q in b.quotes if normalize_name(q.player_name) == key), None)

    if mp is None:
        st.warning(f"Stima non disponibile: {skipped}.")
    else:
        side_rows = book_rows[(book_rows["lato"] == side) & (~book_rows["obsoleta"])] if len(book_rows) else book_rows
        best_odds = float(side_rows["quota"].max()) if len(side_rows) else None
        odds_in = st.number_input("Quota da valutare (default: la migliore non obsoleta)", 1.01, 1000.0, float(best_odds or 1.90), 0.01, key=f"ap_odds_{n}")
        try:
            v = mp.price(line, side_enum, odds_in)
        except ModelError as exc:
            st.error(f"Linea non valutabile: {exc}")
            st.stop()
        d1, d2 = st.columns([3, 2])
        d1.plotly_chart(charts.fig_distribution(mp.pmf, line, side.lower(), mp.reps, mp.ci_level), width="stretch")
        with d2:
            ci = lambda c: f"{pct(c[0])} – {pct(c[1])}" if c else "—"
            st.markdown("**Probabilità e valore atteso**")
            tbl = pd.DataFrame({
                "": ["Vince", "Rimborso", "Perde", "Quota equa", "ROI atteso", f"EV su {st.session_state['stake']:g} €"],
                "Stima": [pct(v.p_win), pct(v.p_push), pct(v.p_loss), f"{v.fair_odds:.2f}", f"{100 * v.roi:+.1f}%", f"{v.ev(float(st.session_state['stake'])):+.2f} €"],
                f"IC {v.ci_level:.0%}": [ci(v.p_win_ci), "—", "—", f"{v.fair_odds_ci[0]:.2f} – {v.fair_odds_ci[1]:.2f}" if v.fair_odds_ci else "—",
                                         f"{100 * v.roi_ci[0]:+.1f}% – {100 * v.roi_ci[1]:+.1f}%" if v.roi_ci else "—", "—"],
            })
            st.dataframe(tbl, hide_index=True, width="stretch")
            st.markdown(f"**{v.verdict}.**  \n<span class='nvf-note'>Prob. implicita grezza 1/quota = {pct(v.implied_prob_raw)} · "
                        f"P vittoria di pareggio = {pct(v.breakeven_p_win)} · media prevista {mp.mean:.1f} (mediana {mp.median}).</span>", unsafe_allow_html=True)
        with st.expander("Probabilità per ogni soglia"):
            lt = mp.line_table().rename(columns={"linea": "Linea", "p_over": "P Over", "p_push": "P rimborso", "p_under": "P Under",
                                                 "quota_equa_over": "Quota equa Over", "quota_equa_under": "Quota equa Under",
                                                 "p_over_lo": "P Over min", "p_over_hi": "P Over max"})
            pcols = [c for c in lt.columns if c.startswith("P ")]
            st.dataframe(lt.style.format({**{c: "{:.1%}" for c in pcols}, "Quota equa Over": "{:.2f}", "Quota equa Under": "{:.2f}", "Linea": "{:g}"}),
                         hide_index=True, width="stretch", height=300)

    # ---- confronto bookmaker per questa prop -----------------------------------------------
    st.subheader("Confronto quote tra bookmaker")
    grp = None
    if len(book_rows):
        piv = book_rows.pivot_table(index=["book", "bookmaker"], columns="lato", values="quota", aggfunc="max").reset_index()
        for c in ("Over", "Under"):
            if c not in piv:
                piv[c] = np.nan
        ages = book_rows.groupby("book")["eta_min"].min()
        piv["Età (min)"] = piv["book"].map(ages)
        both = piv["Over"].notna() & piv["Under"].notna()
        piv["Margine"] = np.where(both, 100 * (1 / piv["Over"] + 1 / piv["Under"] - 1), np.nan)
        piv["P Over senza margine"] = np.where(both, 100 * (1 / piv["Over"]) / (1 / piv["Over"] + 1 / piv["Under"]), np.nan)
        cc1, cc2 = st.columns([3, 2])
        cc1.dataframe(piv.drop(columns="book").rename(columns={"bookmaker": "Bookmaker"}).style.format(
            {"Over": "{:.2f}", "Under": "{:.2f}", "Età (min)": "{:.0f}", "Margine": "{:.1f}%", "P Over senza margine": "{:.1f}%"}, na_rep="—"),
            hide_index=True, width="stretch")
        col = side
        g = piv[["bookmaker", col]].dropna().rename(columns={col: "quota"})
        cc2.plotly_chart(charts.fig_odds_by_book(g, mp and (v.fair_odds if mp else None), float(g["quota"].mean()) if len(g) else None, side), width="stretch")
        gm = board.groups[(board.groups["giocatore"] == b.player_name(key)) & (board.groups["mercato"] == MARKET_IT[market]) & np.isclose(board.groups["linea"], line)] if board is not None else None
        if gm is not None and len(gm):
            grp = gm.iloc[0]
            st.caption(f"Media generale tra {int(grp['n_book'])} bookmaker — Over {grp['quota_over_media']:.2f}, Under {grp['quota_under_media']:.2f}; "
                       f"migliori: Over {grp['quota_over_max']:.2f} ({grp['book_over_max']}), Under {grp['quota_under_max']:.2f} ({grp['book_under_max']}). "
                       f"Probabilità Over di mercato senza margine: {pct(grp['p_over_mercato'])}.")
    else:
        st.info("Nessuna quota disponibile per questa linea: il confronto tra bookmaker non è possibile. "
                "Puoi importare quote da CSV in «Impostazioni e API».")

    # ---- motivazione e dati mancanti -----------------------------------------------------------
    st.subheader("Motivazione sintetica della stima")
    if mp is not None:
        lines = reasoning.explain(pred, market, line, side.lower(), v, season_avg=_mean(sv), last10_avg=_mean(vals_all, 10),
                                  hit_all=hits["tutte"]["freq_over" if side == "Over" else "freq_under"], hit_20=hits["ultime_20"]["freq_over" if side == "Over" else "freq_under"],
                                  market_p_over=(None if grp is None or pd.isna(grp["p_over_mercato"]) else float(grp["p_over_mercato"])))
        st.markdown("<div class='nvf-reason'>" + "<br>".join("• " + l.replace("<", "&lt;") for l in lines) + "</div>", unsafe_allow_html=True)
    else:
        st.info("Nessuna motivazione: la stima non è disponibile (vedi dati mancanti qui sotto).")
    st.subheader("Dati mancanti e limiti")
    for lvl, txt in reasoning.missing_data(pred, n_logs=len(logs), skipped_reason=skipped, issues=b.collection_issues.get(key, []), market=market,
                                            n_books=n_books, n_stale=n_stale, has_opponent=bool(pred and pred.context.opponent), mode=b.mode):
        st.markdown(f"- {_icon(lvl)} {txt}")
    if pred is not None:
        with st.expander("Parametri stimati del modello"):
            st.json({k: (v_ if not isinstance(v_, float) else round(v_, 4)) for k, v_ in pred.fit.summary().items()})


def _icon(level: str) -> str:
    return {"ok": "✅", "warn": "⚠️", "missing": "❌"}.get(level, "•")
