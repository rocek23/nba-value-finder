"""Calcolo del tabellone: per ogni quota, probabilita' del modello, quota equa, ROI/EV e avvisi.

Logica pura (nessuna dipendenza da Streamlit), cosi' e' testabile. Le grandezze che dipendono dal tempo (eta' della
quota) o dallo stake sono applicate da ``annotate`` al momento della visualizzazione.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..model.config import ModelConfig
from ..model.engine import GameContext, PlayerPrediction, ValueEngine
from ..model.errors import InsufficientDataError, InvalidLineError, ModelError
from ..models import Market, Side, utc_now
from .bundle import DataBundle
from .labels import MARKET_IT, SIDE_IT, book_label, fmt_age, normalize_name

NY = ZoneInfo("America/New_York")
PRICED_COLUMNS = ["p_win", "p_push", "p_loss", "quota_equa", "roi", "roi_lo", "roi_hi", "p_win_lo", "p_win_hi", "edge", "p_pareggio"]


@dataclass(frozen=True)
class BoardParams:
    n_boot: int = 40
    n_sims_pra: int = 40_000
    n_sims_pra_boot: int = 4_000
    ci_level: float = 0.90
    min_pool_players: int = 20


@dataclass
class Board:
    rows: pd.DataFrame
    groups: pd.DataFrame
    predictions: dict[str, PlayerPrediction]
    skipped: dict[str, str]  # chiave giocatore -> motivo per cui non e' stimabile
    prior_source: str
    prior_players: int
    params: BoardParams
    engine: ValueEngine | None = None
    warnings: list[tuple[str, str]] = field(default_factory=list)


def game_day(q) -> str:
    """Giorno (ora della costa Est USA) della partita: unisce le quote dello stesso incontro tra provider diversi."""
    if q.commence_time:
        return q.commence_time.astimezone(NY).date().isoformat()
    return str(q.event_id)


def prop_id(day: str, key: str, market: Market, line: float) -> str:
    return f"{day}|{key}|{market.value}|{line:g}"


def context_for(bundle: DataBundle, key: str, q, minutes_override: float | None = None) -> GameContext:
    """Contesto della partita: avversario (solo registrato) e back-to-back dedotto dallo storico del giocatore."""
    info = bundle.players.get(key)
    opponent = None
    if info and info.team and q is not None and q.home_team and q.away_team:
        opponent = q.away_team if info.team == q.home_team else q.home_team if info.team == q.away_team else None
    b2b = None
    logs = bundle.logs.get(key)
    if q is not None and q.commence_time and logs:
        b2b = (q.commence_time.astimezone(NY).date() - max(l.game_date for l in logs)).days == 1
    return GameContext(opponent=opponent, back_to_back=b2b, minutes_override=minutes_override)


def build_board(bundle: DataBundle, params: BoardParams = BoardParams()) -> Board:
    cfg = ModelConfig(n_boot=params.n_boot, n_sims_pra=params.n_sims_pra, n_sims_pra_boot=params.n_sims_pra_boot,
                      ci_level=params.ci_level)
    engine = ValueEngine(cfg)
    engine.fit_prior(bundle.logs, min_players=params.min_pool_players)

    # una predizione per giocatore (il contesto usa la sua prima partita in programma)
    first_quote: dict[str, object] = {}
    for q in bundle.quotes:
        first_quote.setdefault(normalize_name(q.player_name), q)

    predictions: dict[str, PlayerPrediction] = {}
    skipped: dict[str, str] = {}
    for key, q in first_quote.items():
        logs = bundle.logs.get(key)
        if not logs:
            reason = "; ".join(bundle.collection_issues.get(key, [])) or "nessuno storico partite disponibile"
            skipped[key] = f"storico assente — {reason}"
            continue
        ctx = context_for(bundle, key, q)
        try:
            predictions[key] = engine.predict(logs, ctx)
        except InsufficientDataError as exc:
            skipped[key] = f"storico insufficiente — {exc}"
        except ModelError as exc:  # errore del modello: va mostrato, non nascosto
            skipped[key] = f"errore del modello — {exc}"

    rows = []
    for q in bundle.quotes:
        key = normalize_name(q.player_name)
        day = game_day(q)
        info = bundle.players.get(key)
        ev = bundle.events.get(q.event_id)
        title = f"{q.away_team or (ev.away if ev else '?')} @ {q.home_team or (ev.home if ev else '?')}"
        row = {
            "prop_id": prop_id(day, key, q.market, q.line), "gday": day, "key": key, "evento": title,
            "giocatore": bundle.player_name(key), "squadra": info.team if info else None,
            "market": q.market, "mercato": MARKET_IT[q.market], "linea": q.line, "side": q.side, "lato": SIDE_IT[q.side],
            "book": q.bookmaker, "bookmaker": book_label(q.bookmaker), "quota": q.odds_decimal,
            "ts": q.reference_time, "inizio": q.commence_time, "manuale": q.source == "csv", "fonte": q.provider,
            "priced": False, "motivo": "", "n_partite": np.nan, "avvisi_modello": [],
        }
        row.update({c: np.nan for c in PRICED_COLUMNS})
        pred = predictions.get(key)
        if pred is None:
            row["motivo"] = skipped.get(key, "giocatore non stimabile")
        else:
            mp = pred.markets.get(q.market)
            if mp is None:
                row["motivo"] = ("tentativi da tre insufficienti per prezzare le triple" if q.market is Market.THREES
                                 else "mercato non prezzabile")
            else:
                try:
                    v = mp.price(q.line, q.side, q.odds_decimal)
                except InvalidLineError as exc:
                    row["motivo"] = f"linea non gestita ({exc})"
                else:
                    row.update(
                        priced=True, p_win=v.p_win, p_push=v.p_push, p_loss=v.p_loss, quota_equa=v.fair_odds, roi=v.roi,
                        roi_lo=v.roi_ci[0] if v.roi_ci else np.nan, roi_hi=v.roi_ci[1] if v.roi_ci else np.nan,
                        p_win_lo=v.p_win_ci[0] if v.p_win_ci else np.nan, p_win_hi=v.p_win_ci[1] if v.p_win_ci else np.nan,
                        edge=v.edge, p_pareggio=v.breakeven_p_win, n_partite=pred.fit.n_games,
                    )
                    avv = []
                    if "campione_ridotto" in pred.fit.flags:
                        avv.append(f"Campione ridotto ({pred.fit.n_games} partite)")
                    if "prior_fallback_nessun_pool" in pred.fit.flags:
                        avv.append("Prior di popolazione non disponibile (pool troppo piccolo)")
                    if pred.adjustments:
                        avv.append("Aggiustamento back-to-back applicato (verificato sullo storico)")
                    if v.p_push > 0:
                        avv.append("Linea intera: possibile rimborso")
                    row["avvisi_modello"] = avv
        rows.append(row)

    df = pd.DataFrame(rows)
    groups = _consensus(df)
    board = Board(df, groups, predictions, skipped, engine.prior.source, engine.prior.n_players, params, engine)
    board.warnings = _warnings(bundle, board)
    return board


def _consensus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = ["prop_id", "evento", "giocatore", "squadra", "mercato", "linea"]
    pair = (df.pivot_table(index=keys + ["book"], columns="lato", values="quota", aggfunc="max").reset_index())
    for col in ("Over", "Under"):
        if col not in pair:
            pair[col] = np.nan
    both = pair["Over"].notna() & pair["Under"].notna()
    pair["margine"] = np.where(both, 1 / pair["Over"] + 1 / pair["Under"] - 1, np.nan)
    pair["novig_over"] = np.where(both, (1 / pair["Over"]) / (1 / pair["Over"] + 1 / pair["Under"]), np.nan)
    pair["imp_over"] = 1 / pair["Over"]
    out = []
    model = df[df["priced"] & (df["lato"] == "Over")].groupby("prop_id")[["p_win", "p_push", "quota_equa"]].first()
    model_u = df[df["priced"] & (df["lato"] == "Under")].groupby("prop_id")[["quota_equa"]].first()
    for pid, g in pair.groupby("prop_id", sort=False):
        first = g.iloc[0]
        ov, un = g["Over"].dropna(), g["Under"].dropna()
        rec = {
            **{k: first[k] for k in keys}, "n_book": g["book"].nunique(),
            "quota_over_media": ov.mean() if len(ov) else np.nan, "quota_under_media": un.mean() if len(un) else np.nan,
            "quota_over_max": ov.max() if len(ov) else np.nan, "book_over_max": book_label(g.loc[ov.idxmax(), "book"]) if len(ov) else "—",
            "quota_under_max": un.max() if len(un) else np.nan, "book_under_max": book_label(g.loc[un.idxmax(), "book"]) if len(un) else "—",
            "prob_impl_over_media": g["imp_over"].mean(), "p_over_mercato": g["novig_over"].mean(),
            "margine_medio": g["margine"].mean(), "n_book_due_lati": int(both[g.index].sum()),
        }
        if pid in model.index:
            rec["p_over_modello"] = model.loc[pid, "p_win"]
            rec["p_push_modello"] = model.loc[pid, "p_push"]
            rec["quota_equa_over"] = model.loc[pid, "quota_equa"]
            rec["quota_equa_under"] = model_u.loc[pid, "quota_equa"] if pid in model_u.index else np.nan
        else:
            rec.update(p_over_modello=np.nan, p_push_modello=np.nan, quota_equa_over=np.nan, quota_equa_under=np.nan)
        rec["delta_modello_mercato"] = rec["p_over_modello"] - rec["p_over_mercato"]
        out.append(rec)
    return pd.DataFrame(out)


def _warnings(bundle: DataBundle, board: Board) -> list[tuple[str, str]]:
    w: list[tuple[str, str]] = []
    df = board.rows
    if df.empty:
        return w
    no_hist = [bundle.player_name(k) for k, r in board.skipped.items() if r.startswith("storico assente")]
    few = [bundle.player_name(k) for k, r in board.skipped.items() if r.startswith("storico insufficiente")]
    err = [bundle.player_name(k) for k, r in board.skipped.items() if r.startswith("errore")]
    if no_hist:
        w.append(("warning", f"{len(no_hist)} {_pl(len(no_hist))} hanno quote ma nessuno storico: non vengono stimati ({_names(no_hist)})."))
    if few:
        w.append(("warning", f"{len(few)} {_pl(len(few))} {'ha' if len(few)==1 else 'hanno'} meno partite del minimo richiesto: non {'viene stimato' if len(few)==1 else 'vengono stimati'} ({_names(few)})."))
    if err:
        w.append(("error", f"Errore del modello per {len(err)} {_pl(len(err))} ({_names(err)})."))
    small = [bundle.player_name(k) for k, p in board.predictions.items() if "campione_ridotto" in p.fit.flags]
    if small:
        w.append(("warning", f"{len(small)} {_pl(len(small))} {'ha' if len(small)==1 else 'hanno'} un campione ridotto: stime più incerte, gli intervalli sono più larghi ({_names(small)})."))
    unpriced = df[~df["priced"]]
    if len(unpriced):
        reasons = unpriced.groupby("motivo").size().sort_values(ascending=False)
        w.append(("info", f"{len(unpriced)} quote non sono prezzabili ({'; '.join(f'{n} × {r.split(' — ')[0]}' for r, n in reasons.head(3).items())})."))
    if board.prior_source != "pool":
        w.append(("warning", "Prior di popolazione non stimabile (pochi giocatori nel pool): lo shrinkage usa solo il tasso del singolo giocatore."))
    if df["manuale"].any():
        w.append(("info", f"{int(df['manuale'].sum())} quote provengono da CSV inserito manualmente: verifica l'orario di rilevazione."))
    return w


def _pl(n: int) -> str:
    return "giocatore" if n == 1 else "giocatori"


def _names(names: list[str], k: int = 4) -> str:
    shown = ", ".join(sorted(set(names))[:k])
    return shown + ("…" if len(set(names)) > k else "")


def annotate(df: pd.DataFrame, now: datetime | None, max_age_min: float, stake: float) -> pd.DataFrame:
    """Aggiunge i campi che dipendono da tempo e stake: eta' quota, obsolescenza, EV in denaro, avvisi completi."""
    if df.empty:
        return df.assign(eta_min=[], obsoleta=[], ev=[], avvisi=[], n_avvisi=[])
    now = now or utc_now()
    out = df.copy()
    out["eta_min"] = [(now - t).total_seconds() / 60 for t in out["ts"]]
    out["obsoleta"] = out["eta_min"] > max_age_min
    out["ev"] = stake * out["roi"]
    avvisi = []
    for r in out.itertuples():
        a = list(r.avvisi_modello)
        if r.obsoleta:
            a.insert(0, f"Quota obsoleta ({fmt_age(r.eta_min)})")
        if r.manuale:
            a.append("Quota manuale (CSV)")
        if not r.priced:
            a.insert(0, f"Non prezzata: {r.motivo}")
        avvisi.append(" · ".join(a))
    out["avvisi"] = avvisi
    out["n_avvisi"] = [len(x.split(" · ")) if x else 0 for x in avvisi]
    out["classe"] = np.where(~out["priced"], "non_prezzata",
                    np.where(out["roi_lo"] > 0, "positivo_robusto",
                    np.where(out["roi"] > 0, "positivo_incerto", "non_positivo")))
    return out
