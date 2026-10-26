"""Backtesting: valutazione fuori campione del modello, senza look-ahead, con baseline, quote storiche reali e archivio verificabile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ...backtest import BacktestConfig, run_backtest
from ...backtest import report as R
from ...backtest.demo_data import DEMO_BT_NOTICE, build_backtest_demo
from ...backtest.evaluate import MARKET_BASELINE
from ...backtest.quotes import load_quote_history
from ...backtest.runner import make_run_id
from ...backtest.splits import SplitError
from ...backtest.store import BacktestStore, unpack_pmf
from ...models import Market
from .. import charts, charts_bt as CB, shell
from .. import theme as T
from ..snapshot import history_path

MARKETS = {m.value: m for m in Market}
MARKET_LABEL = {"points": "Punti", "rebounds": "Rimbalzi", "assists": "Assist", "threes": "Triple", "pra": "PRA"}
CAND_LABEL = {"base": "Base (recenza standard)", "recenza_corta": "Recenza corta", "senza_recenza": "Nessuna recenza"}
SOURCE_LABEL = {"sintetica": "Linee sintetiche (qualità predittiva)", "quota_reale": "Quote storiche reali"}
FOOTER = ("Il backtest non inserisce scommesse e non garantisce alcun risultato futuro. Un risultato positivo nel passato, anche fuori campione, "
          "non dimostra un vantaggio stabile.")


@st.cache_resource(show_spinner="Genero lo storico sintetico (qualche secondo)…")
def _demo_inputs(n_players: int, games: int):
    return build_backtest_demo(n_players=n_players, games_per_season=games)


@st.cache_resource(show_spinner="Calcolo gli intervalli di incertezza…")
def _tables(run_id: str, phase: str, source: str, level: float, B: int, _run):
    ev = _run.evals
    pred = R.predictive_table(ev, phase, source, level=level, B=B)
    base = ["moneta", "frequenza_ultime20", "poisson_ultime10"]
    vs = R.paired_vs_baselines(ev, phase, source, "modello", base, level=level, B=B)
    vs_m = R.paired_vs_baselines(ev, phase, source, "modello", [MARKET_BASELINE], level=level, B=B) if source == "quota_reale" else pd.DataFrame()
    rel = R.reliability_view(ev, phase, source, "modello")
    cal = R.calibration_summary(ev, phase, source, "modello", level=level)
    return pred, vs, vs_m, rel, cal


@st.cache_resource(show_spinner="Calcolo le metriche delle scommesse simulate…")
def _bet_tables(run_id: str, level: float, B: int, bankroll: float, _run):
    return {ph: R.bet_table(_run.bets, ph, bankroll=bankroll, level=level, B=B) for ph in ("validazione", "test")}


def _fmt_pct(x, d=1):
    return "n/d" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}%"


def _verdict(v: dict) -> None:
    box = {"success": st.success, "error": st.error, "warning": st.warning}.get(v.get("level"), st.info)
    box(f"**{v['title']}**\n\n{v['text']}")
    if v.get("checks"):
        rows = [{"Condizione": c[0], "Esito": "✅" if c[1] else "❌", "Dettaglio": c[2]} for c in v["checks"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("«Verifica superata» richiede TUTTE le condizioni, sul periodo di TEST (mai usato per scegliere nulla) e con quote storiche reali pre-partita.")


def _pred_view(run, run_id, cfg, phase_label_key: str) -> None:
    ev = run.evals
    c1, c2 = st.columns(2)
    phase = c1.radio("Periodo", ["test", "validazione"], horizontal=True, key=f"bt_phase_{phase_label_key}", format_func=lambda p: "Test (fuori campione)" if p == "test" else "Validazione")
    avail = [s for s in ("sintetica", "quota_reale") if ((ev["phase"] == phase) & (ev["line_source"] == s)).any()]
    source = c2.radio("Linee valutate", avail, horizontal=True, key=f"bt_src_{phase_label_key}", format_func=SOURCE_LABEL.get)
    if source == "sintetica":
        st.caption("Mezze linee generate dallo storico del giocatore (non dal modello), uguali per modello e baseline. Misurano la **qualità delle probabilità**; non hanno quote, quindi non producono ROI.")
    else:
        st.caption("Linee reali dei bookmaker, catturate prima dell'inizio. Si usano solo le mezze linee e il lato Over per le metriche di probabilità (l'Under è il complemento).")
    level, B = float(cfg["ci_level"]), int(cfg["n_boot"])
    pred, vs, vs_m, rel, cal = _tables(run_id, phase, source, level, B, run)
    if pred.empty:
        st.warning("Nessuna osservazione per questa combinazione.")
        return
    n = int(pred.iloc[0]["n_osservazioni"])
    st.markdown(f"**Campione:** {n:,} osservazioni su {int(pred.iloc[0]['giornate'])} giornate di gioco · intervalli al {level:.0%} (bootstrap per giornata)".replace(",", "."))
    show = pred.assign(modello=pred["modello"].map(CB.nice))[["modello", "n_osservazioni", "brier", "brier_lo", "brier_hi", "log_loss", "ll_lo", "ll_hi", "brier_skill_vs_moneta", "p_media", "freq_osservata"]]
    show.columns = ["Modello", "N", "Brier", "IC inf.", "IC sup.", "Log-loss", "IC inf. ", "IC sup. ", "Skill vs moneta", "P media", "Freq. osservata"]
    st.dataframe(show.style.format({"Brier": "{:.4f}", "IC inf.": "{:.4f}", "IC sup.": "{:.4f}", "Log-loss": "{:.4f}", "IC inf. ": "{:.4f}", "IC sup. ": "{:.4f}",
                                    "Skill vs moneta": "{:+.1%}", "P media": "{:.1%}", "Freq. osservata": "{:.1%}"}), hide_index=True, width="stretch")
    g1, g2 = st.columns(2)
    g1.plotly_chart(CB.fig_loss_compare(pred, "brier"), width="stretch")
    g2.plotly_chart(CB.fig_loss_compare(pred, "log_loss"), width="stretch")
    allvs = pd.concat([vs, vs_m], ignore_index=True) if not vs_m.empty else vs
    st.plotly_chart(CB.fig_delta(allvs, "log_loss"), width="stretch")
    if source == "quota_reale" and vs_m.empty:
        st.info("Il confronto con il consenso di mercato richiede coppie Over/Under dello stesso prelievo: non disponibili.")
    g3, g4 = st.columns(2)
    g3.plotly_chart(CB.fig_reliability_ci(rel, "Curva di affidabilità del modello"), width="stretch")
    pit = run.pmf[(run.pmf["phase"] == phase) & (run.pmf["model"] == "modello")]["pit"]
    g4.plotly_chart(CB.fig_pit_hist(pit), width="stretch")
    if cal.get("pendenza") is not None:
        st.markdown(f"**Calibrazione (regressione logistica):** pendenza {cal['pendenza']:.2f} (IC {cal['pendenza_lo']:.2f}…{cal['pendenza_hi']:.2f}), intercetta {cal['intercetta']:+.2f}, "
                    f"ECE {100 * cal['ece']:.1f}%. Pendenza < 1: modello troppo sicuro di sé; > 1: troppo prudente.")
    sc = run.pmf[(run.pmf["phase"] == phase)]
    if not sc.empty:
        t = sc[sc["model"].isin(["modello", "poisson_ultime10"])].groupby("model")[["log_score", "crps"]].mean()
        t.index = [CB.nice(i) for i in t.index]
        st.markdown("**Distribuzione intera (indipendente dalle linee)** — log score negativo e CRPS medi, più basso = meglio")
        st.dataframe(t.rename(columns={"log_score": "Log score", "crps": "CRPS"}).style.format("{:.4f}"), width="stretch")


def _bets_view(run, run_id, cfg) -> None:
    ev = run.evals
    has = ((ev["phase"] == "test") & (ev["line_source"] == "quota_reale")).any()
    if not has:
        st.error("**ROI non calcolabile — nessuna quota storica reale nel periodo di test.**  \n"
                 "Non si inventano quote: senza quote con timestamp precedente all'inizio della partita si può valutare solo la qualità predittiva. "
                 "Ogni «Aggiorna dati live» salva una copia delle quote in `data/cache/quote_history.jsonl`; i CSV con timestamp e `commence_time` si importano da «Impostazioni e API».")
        return
    level, B = float(cfg["ci_level"]), int(cfg["n_boot"])
    tabs = _bet_tables(run_id, level, B, float(cfg["bankroll"]), run)
    thr = run.meta["threshold"]
    st.caption(f"Regola: per ogni giocatore/partita/mercato si sceglie la riga (linea, lato, libro) con EV massimo e si punta 1 unità se EV ≥ {thr:.0%} "
               "(soglia scelta sulla validazione). EV con rimborso incluso; quota REALE del libro, quindi il margine è già pagato. "
               "Yield = profitto / puntato · ROI capitale = profitto / capitale iniziale · hit rate = vinte / (vinte + perse), rimborsi esclusi.")
    for ph, label in (("test", "Test (fuori campione) — risultato che conta"), ("validazione", "Validazione (usata per scegliere la soglia: ottimistica)")):
        t = tabs[ph]
        st.markdown(f"**{label}**")
        if t.empty:
            st.info("Nessuna scommessa simulata.")
            continue
        show = t.assign(strategia=t["strategia"].map(CB.nice))[["strategia", "n_scommesse", "giornate", "yield", "yield_lo", "yield_hi", "roi_capitale", "hit_rate", "hit_lo", "hit_hi", "quota_media",
                                                                 "profitto", "max_drawdown", "max_drawdown_lo", "max_drawdown_hi"]]
        show.columns = ["Strategia", "Scommesse", "Giornate", "Yield", "IC inf.", "IC sup.", "ROI capitale", "Hit rate", "IC inf. ", "IC sup. ", "Quota media", "Profitto (u.)",
                        "Max drawdown", "DD inf.", "DD sup."]
        st.dataframe(show.style.format({"Yield": "{:+.1%}", "IC inf.": "{:+.1%}", "IC sup.": "{:+.1%}", "ROI capitale": "{:+.1%}", "Hit rate": "{:.1%}", "IC inf. ": "{:.1%}",
                                        "IC sup. ": "{:.1%}", "Quota media": "{:.2f}", "Profitto (u.)": "{:+.1f}", "Max drawdown": "{:.1f}", "DD inf.": "{:.1f}", "DD sup.": "{:.1f}"}),
                     hide_index=True, width="stretch")
    st.caption(f"Capitale iniziale {cfg['bankroll']:.0f} unità, puntata fissa {cfg['stake']:.0f}. Il drawdown è in unità; l'IC è un bootstrap sulle giornate (ordine casuale), quindi indicativo.")
    tb = run.bets[run.bets["phase"] == "test"]
    st.plotly_chart(CB.fig_yield_forest(tabs["test"]), width="stretch")
    st.plotly_chart(CB.fig_pnl(tb), width="stretch")
    st.caption("Le strategie «Sempre Over/Under» non usano il modello: mostrano quanto costa il margine a chi punta senza vantaggio.")


def _segments_view(run, run_id, cfg) -> None:
    ev = run.evals
    level = float(cfg["ci_level"])
    st.caption("Sottogruppi con meno di 30 osservazioni sono mostrati senza intervallo (barre grigie): non se ne può concludere nulla. Più sottogruppi si guardano, più è facile trovarne uno «bello» per caso.")
    phase = "test"
    by = st.radio("Segmenta per", ["market", "season", "player", "book"], horizontal=True, key="bt_seg",
                  format_func=lambda x: {"market": "Mercato", "season": "Stagione", "player": "Giocatore", "book": "Bookmaker"}[x])
    syn = ev[(ev["phase"] == phase) & (ev["line_source"] == "sintetica")]
    has_q = ((ev["phase"] == phase) & (ev["line_source"] == "quota_reale")).any()
    if by != "book":
        src = "sintetica" if syn.shape[0] else "quota_reale"
        st.markdown("**Qualità predittiva del modello** (Brier, linee sintetiche, test)")
        t = R.predictive_breakdown(ev, phase, src, by, level=level)
        if by == "player":
            t = t.sort_values("n_osservazioni", ascending=False)
        if t.empty:
            st.info("Nessun dato.")
        else:
            if by == "market":
                t[by] = t[by].map(MARKET_LABEL)
            st.plotly_chart(CB.fig_breakdown(t if by != "player" else t.head(25), by, "brier", "brier_lo", "brier_hi", "Brier per " + by, ref=0.25), width="stretch")
            st.dataframe(t, hide_index=True, width="stretch")
    st.markdown("**Rendimento simulato** (strategia di valore del modello, test)")
    if not has_q:
        st.info("ROI non calcolabile: nessuna quota storica reale nel test.")
        return
    if by == "book":
        t = R.bets_by_book(ev, "modello", run.meta["threshold"], phase, float(cfg["stake"]), level=level)
        st.caption("Ogni libro è valutato da solo (solo le sue quote): le scommesse si sovrappongono tra libri, le righe non si sommano.")
    else:
        t = R.bet_breakdown(run.bets, "valore:modello", phase, by, level=level)
        if by == "market" and not t.empty:
            t[by] = t[by].map(MARKET_LABEL)
    if t.empty:
        st.info("Nessuna scommessa.")
        return
    st.plotly_chart(CB.fig_breakdown(t if by != "player" else t.sort_values("n_scommesse", ascending=False).head(25), by, "yield", "yield_lo", "yield_hi", "Yield per " + by, pct=True, ref=0.0), width="stretch")
    st.dataframe(t.style.format({"yield": "{:+.1%}", "yield_lo": "{:+.1%}", "yield_hi": "{:+.1%}", "hit_rate": "{:.1%}", "quota_media": "{:.2f}", "profitto": "{:+.1f}"}), hide_index=True, width="stretch")


def _quotes_view(run) -> None:
    qi = run.meta["quote_info"]
    c = st.columns(4)
    for col, (l, v) in zip(c, [("Quote ricevute", qi["quote_ricevute"]), ("Pre-partita selezionate", qi["quote_pre_partita_selezionate"]),
                               ("Abbinate a una partita", qi["quote_abbinate_a_partite"]), ("Libri", len(qi["libri"]))]):
        col.markdown(T.kpi(l, f"{v:,}".replace(",", "."), ""), unsafe_allow_html=True)
    if qi["quote_ricevute"] == 0:
        st.warning("Nessuna quota storica disponibile: ROI non calcolabile. Premi «Aggiorna dati live» (modalità live) per iniziare a raccogliere lo storico, oppure importa un CSV con `timestamp` e `commence_time`.")
    from ...backtest.quotes import REASONS

    sc = qi.get("scartate", {})
    if sc:
        st.markdown("**Quote scartate** (regole anti-leakage)")
        st.dataframe(pd.DataFrame([{"Motivo": REASONS.get(k, k), "Quote": v} for k, v in sorted(sc.items(), key=lambda kv: -kv[1])]), hide_index=True, width="stretch")
        if sc.get("dopo_inizio"):
            st.caption("Le quote catturate a partita iniziata non vengono mai usate: conoscerebbero parte dell'esito.")
    st.plotly_chart(CB.fig_margins(qi.get("margini_per_libro", {})), width="stretch")
    st.caption("Il margine si misura solo dalle coppie Over/Under dello stesso prelievo. Il ROI simulato usa le quote reali, quindi lo paga per intero.")


def _store_view(run, run_id, store: BacktestStore) -> None:
    st.markdown("**Verifica indipendente dell'archivio**")
    st.caption("Ricontrolla fasi, storico, prior e orari delle quote, e RICALCOLA un campione di previsioni dai soli dati salvati.")
    if st.button("Verifica questo backtest", key="bt_verify"):
        with st.spinner("Verifica in corso…"):
            res = store.verify(run_id, sample=20)
        st.dataframe(pd.DataFrame([{"Controllo": n, "Esito": "✅" if ok else "❌", "Dettaglio": d} for n, ok, d in res]), hide_index=True, width="stretch")
        if all(ok for _, ok, _ in res):
            st.success("Tutti i controlli superati.")
        else:
            st.error("Alcuni controlli NON sono superati: non fidarti di questi risultati.")
    st.markdown("**Previsioni salvate**")
    q = store.query("SELECT p.candidate AS candidato, p.phase AS fase, p.player AS giocatore, p.game_id AS partita, p.game_date AS data, p.market AS mercato, p.mean AS media_prevista, "
                    "p.y AS esito, i.history_n AS partite_storico, i.history_last_date AS ultimo_storico, p.prior_block_start AS blocco_prior, p.flags AS avvisi "
                    "FROM predictions p JOIN inputs i ON i.run_id = p.run_id AND i.player = p.player AND i.game_id = p.game_id WHERE p.run_id = ? ORDER BY p.game_date, p.player", (run_id,))
    c1, c2, c3 = st.columns(3)
    fase = c1.selectbox("Fase", ["(tutte)"] + sorted(q["fase"].unique()), key="bt_ex_phase")
    merc = c2.selectbox("Mercato", ["(tutti)"] + sorted(q["mercato"].unique()), key="bt_ex_mk")
    gio = c3.selectbox("Giocatore", ["(tutti)"] + sorted(q["giocatore"].unique()), key="bt_ex_pl")
    if fase != "(tutte)":
        q = q[q["fase"] == fase]
    if merc != "(tutti)":
        q = q[q["mercato"] == merc]
    if gio != "(tutti)":
        q = q[q["giocatore"] == gio]
    st.dataframe(q.head(2000), hide_index=True, width="stretch")
    st.caption(f"{len(q):,} previsioni (mostrate le prime 2000). Ogni riga conserva la distribuzione completa, gli id delle partite passate al modello e il prior usato.".replace(",", "."))
    if len(q):
        pick = st.selectbox("Distribuzione salvata", list(range(min(len(q), 200))), key="bt_ex_pick",
                            format_func=lambda i: f"{q.iloc[i]['data']} · {q.iloc[i]['giocatore']} · {q.iloc[i]['mercato']} · {q.iloc[i]['candidato']}")
        r = q.iloc[pick]
        blob = store.query("SELECT pmf, support FROM predictions WHERE run_id=? AND candidate=? AND player=? AND game_id=? AND market=?",
                           (run_id, r["candidato"], r["giocatore"], r["partita"], r["mercato"])).iloc[0]
        pmf = unpack_pmf(blob["pmf"], int(blob["support"]))
        line = float(np.floor(r["media_prevista"]) + 0.5)
        st.plotly_chart(charts.fig_distribution(pmf, line, "over"), width="stretch")
        ids = store.query("SELECT history_ids FROM inputs WHERE run_id=? AND player=? AND game_id=?", (run_id, r["giocatore"], r["partita"])).iloc[0]["history_ids"]
        st.caption(f"Esito reale: {r['esito']:.0f} · partite passate al modello: {r['partite_storico']} (ultima {r['ultimo_storico']}) · id: {ids[:160]}…")
    d1, d2 = st.columns(2)
    d1.download_button("Scarica scommesse simulate (CSV)", run.bets.to_csv(index=False).encode(), file_name=f"scommesse_{run_id}.csv", mime="text/csv")
    d2.download_button("Scarica valutazioni per linea (CSV)", run.evals.drop(columns=["obs_id"]).to_csv(index=False).encode(), file_name=f"valutazioni_{run_id}.csv", mime="text/csv")


def _method_view(run) -> None:
    sp, sel = run.split, run.meta["selection"]
    st.markdown(f"""
**Periodi (cronologici, mai rimescolati)**
- Addestramento: {sp.first:%d/%m/%Y} → {sp.train_end:%d/%m/%Y} (storico di partenza e prior di popolazione)
- Validazione: fino al {sp.val_end:%d/%m/%Y} — qui si sceglie il candidato, la ricalibrazione e la soglia di EV
- Test: fino al {sp.last:%d/%m/%Y} — valutato **una sola volta**, con tutto già congelato

**Regole applicate**
1. Ogni previsione usa solo partite con data strettamente precedente; il prior di popolazione è ristimato a blocchi di {run.cfg['block_days']} giorni solo su partite anteriori al blocco.
2. Walk-forward: la storia cresce partita dopo partita; nessun parametro è stimato sull'intero periodo.
3. Quote: solo quelle catturate prima dell'inizio (ultima disponibile), mai quote attuali applicate al passato; quote senza orario d'inizio sono scartate.
4. Le linee sintetiche derivano dallo storico del giocatore, non dal modello; baseline e modello sono valutati sulle stesse osservazioni.
5. Senza quote storiche reali non si calcola alcun ROI.
6. Ogni previsione, i dati usati e le quote sono archiviati in `data/cache/backtest.sqlite` e verificabili.

**Limiti noti (Da verificare con dati reali)**: assenze/DNP non hanno un esito e sono escluse (le regole di rimborso dei libri variano); la data della partita è quella ET dell'inizio;
gli intervalli coprono l'errore di campionamento, non i cambi di regime; il test conta davvero solo se non lo si guarda più volte.
""")
    st.markdown("**Scelta del candidato (solo validazione)**")
    st.dataframe(pd.DataFrame(sel["tabella"]).assign(scelto=lambda d: d["candidato"] == sel["scelto"]).rename(columns={"candidato": "Candidato", "log_loss_validazione": "Log-loss validazione", "n": "N"}),
                 hide_index=True, width="stretch")
    for n in run.meta["notes"]:
        st.info(n)


def render() -> None:
    ctx = shell.begin("Backtesting", "Verifica fuori campione, senza look-ahead, con baseline, quote storiche reali e archivio verificabile")
    b = ctx.bundle
    store = BacktestStore()
    source = "demo" if b.is_demo else "live"
    if b.is_demo:
        st.warning(DEMO_BT_NOTICE)
    st.caption(FOOTER)

    with st.expander("Parametri del backtest", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        train = c1.slider("Quota addestramento", 0.3, 0.7, 0.5, 0.05, key="bt_train")
        val = c2.slider("Quota validazione", 0.1, 0.3, 0.25, 0.05, key="bt_val")
        lead = c3.number_input("Anticipo minimo quote (min)", 0, 600, 0, 5, key="bt_lead", help="Scarta le quote catturate negli ultimi N minuti prima dell'inizio.")
        cands = c4.multiselect("Candidati del modello", list(CAND_LABEL), default=["base", "recenza_corta", "senza_recenza"], format_func=CAND_LABEL.get, key="bt_cands")
        c5, c6, c7, c8 = st.columns(4)
        min_bets = c5.number_input("Scommesse minime per il verdetto", 50, 5000, 200, 50, key="bt_minbets")
        level = c6.select_slider("Livello degli intervalli", [0.80, 0.90, 0.95], value=0.90, key="bt_level")
        sims = c7.select_slider("Simulazioni PRA", [1500, 4000, 10000], value=4000, key="bt_sims")
        if b.is_demo:
            n_players = c8.slider("Giocatori demo", 10, 30, 22, key="bt_demo_players")
            games = st.slider("Partite per stagione (demo)", 20, 40, 30, key="bt_demo_games")

    if b.is_demo:
        pool, quotes = _demo_inputs(int(n_players), int(games))
    else:
        pool = {k: v for k, v in b.logs.items() if len(v) >= 15}
        quotes = load_quote_history(history_path())
        if not pool:
            st.warning("Storico insufficiente per il backtest: servono giocatori con almeno 15 partite. Premi «Aggiorna dati live» per scaricare lo storico.")
            pool = {}
    cfg = BacktestConfig(train_frac=float(train), val_frac=float(val), lead_minutes=float(lead), candidates=tuple(cands or ["base"]), min_bets_verdict=int(min_bets),
                         ci_level=float(level), n_sims_pra=int(sims), min_players=min(20, max(5, len(pool) // 2)))

    if pool:
        rid, fp = make_run_id(pool, quotes, cfg, source)
        exists = store.has_run(rid)
        n_prev = store.n_test_configs(fp, exclude_run=rid)
        if n_prev:
            st.warning(f"Su questi stessi dati sono già state valutate {n_prev} altre configurazioni: più configurazioni si guardano sul test, più cresce il rischio di sovradattamento al test.")
        label = "Riesegui backtest" if exists else "Esegui backtest"
        if st.button(label, type="primary", key="bt_run", help="Walk-forward su tutto lo storico: può richiedere qualche minuto."):
            bar = st.progress(0.0, text="Avvio…")
            try:
                res = run_backtest(pool, quotes, cfg, source=source, label=("Demo sintetico" if b.is_demo else "Live"), n_test_runs=n_prev + 1,
                                   progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
                store.save(res)
                st.session_state["bt_run_id"] = res.run_id
            except SplitError as exc:
                st.error(f"Storico troppo corto per dividere in addestramento/validazione/test: {exc}")
            except ValueError as exc:
                st.error(f"Backtest non eseguibile: {exc}")
            finally:
                bar.empty()
        elif exists and "bt_run_id" not in st.session_state:
            st.session_state["bt_run_id"] = rid

    runs = store.runs()
    runs = runs[runs["source"] == source] if len(runs) else runs
    if runs.empty:
        st.info("Nessun backtest archiviato per questa modalità. Premi «Esegui backtest»." if pool else "Nessun backtest disponibile.")
        _history_note(b, quotes)
        return
    ids = list(runs["run_id"])
    cur = st.session_state.get("bt_run_id")
    default = ids.index(cur) if cur in ids else 0
    run_id = st.selectbox("Backtest archiviato", ids, index=default, key="bt_pick",
                          format_func=lambda i: f"{runs[runs['run_id'] == i].iloc[0]['label']} · {runs[runs['run_id'] == i].iloc[0]['created_at'][:16].replace('T', ' ')} UTC · {i}")
    run = _load(store.path, run_id)
    rcfg = run.cfg

    _verdict(run.verdict)
    sp = run.split
    ev = run.evals
    n_pred = int(run.pmf[run.pmf["model"].isin(["modello"])].shape[0]) if len(run.pmf) else 0
    nq = int(((ev["phase"] == "test") & (ev["line_source"] == "quota_reale") & (ev["model"] == "modello")).sum())
    nb = int(((run.bets["phase"] == "test") & (run.bets["strategy"] == "valore:modello")).sum()) if len(run.bets) else 0
    T.kpi_row([
        ("Previsioni nel test", f"{n_pred:,}".replace(",", "."), f"partite-mercato dal {sp.val_end + pd.Timedelta(days=1):%d/%m/%Y}", None),
        ("Quote reali nel test", f"{nq:,}".replace(",", "."), "righe lato/linea/libro", T.GREEN if nq else T.AMBER),
        ("Scommesse simulate (test)", f"{nb:,}".replace(",", "."), f"soglia EV {run.meta['threshold']:.0%}", None),
        ("Candidato scelto", CAND_LABEL.get(run.meta["selection"]["scelto"], run.meta["selection"]["scelto"]), "sulla validazione", None),
        ("Archivio", run_id, "id del backtest", None),
    ])
    t1, t2, t3, t4, t5, t6 = st.tabs(["Qualità predittiva", "Rendimento simulato", "Segmentazioni", "Quote storiche", "Previsioni salvate e verifica", "Metodo e limiti"])
    with t1:
        _pred_view(run, run_id, rcfg, "p")
    with t2:
        _bets_view(run, run_id, rcfg)
    with t3:
        _segments_view(run, run_id, rcfg)
    with t4:
        _quotes_view(run)
        _history_note(b, quotes)
    with t5:
        _store_view(run, run_id, store)
    with t6:
        _method_view(run)


@st.cache_resource(show_spinner="Carico il backtest archiviato…")
def _load(path, run_id: str):
    return BacktestStore(path).load(run_id)


def _history_note(b, quotes) -> None:
    if b.is_demo:
        st.caption("In demo le quote storiche sono sintetiche e prezzate da un mercato inventato.")
    elif not quotes:
        st.caption("Storico quote locale vuoto: BALLDONTLIE non conserva quote storiche. Ogni «Aggiorna dati live» ne salva una copia in data/cache/quote_history.jsonl.")
