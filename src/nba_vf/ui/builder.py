"""Bet Builder AI: motore delle multiple (logica pura, testabile).

Principi:
* usa solo quote e probabilita' gia' calcolate dal tabellone (nessun valore inventato);
* UNA sola selezione per partita (e quindi per giocatore): tra partite diverse l'indipendenza e' un'approssimazione ragionevole
  (**Da verificare** con dati reali); le selezioni nella stessa partita NON sono ammesse perche' correlate;
* per default una multipla usa un solo bookmaker; le quote di bookmaker diversi si combinano solo se l'utente lo consente;
* formule: quota totale = prodotto delle quote; probabilita' che vincano tutte = prodotto delle probabilita' di vincita;
  EV = prodotto di (p_vittoria * quota + p_rimborso) - 1 (un rimborso toglie la selezione dalla multipla);
  EV prudente = stessa formula con ogni probabilita' al limite basso del proprio intervallo di incertezza.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from .injuries import OUT, injury_level

TARGETS = (2, 5, 10, 25, 50, 100)
MIN_GAMES = 25            # sotto questo storico una stima non e' considerata abbastanza solida
STEP = 0.005              # risoluzione (in logaritmo della quota) della ricerca
HIGH_TARGET = 25          # da questa quota la probabilita' di vincita e' molto bassa
LABEL_EV, LABEL_PROB, LABEL_FEW = "Miglior EV complessivo stimato", "Maggiore probabilità di successo", "Minor numero di selezioni"


@dataclass(frozen=True)
class Leg:
    id: str
    key: str
    giocatore: str
    squadra: str
    game: str            # identificativo della partita (giorno + incontro)
    evento: str
    inizio: pd.Timestamp
    market: object       # Market
    mercato: str
    linea: float
    lato: str
    book: str
    odds: float
    p_win: float
    p_push: float
    p_lo: float
    p_hi: float
    n_games: int
    eta_min: float
    quote_known: bool
    prop_id: str
    infortunio: str = ""     # stato segnalato nell'elenco infortuni (solo informativo)

    @property
    def descr(self) -> str:
        return f"{self.mercato} {self.lato} {self.linea:g}"

    @property
    def f(self) -> float:           # fattore di ritorno atteso
        return self.p_win * self.odds + self.p_push

    @property
    def f_lo(self) -> float:
        return self.p_lo * self.odds + self.p_push


# ---- formule ------------------------------------------------------------------------------------------
def total_odds(legs: Sequence[Leg]) -> float:
    return float(np.prod([l.odds for l in legs])) if legs else 1.0


def joint_win_prob(legs: Sequence[Leg]) -> float:
    """Probabilita' che tutte le selezioni vincano (indipendenza tra partite diverse, una selezione per partita)."""
    return float(np.prod([l.p_win for l in legs])) if legs else 1.0


def combo_ev(legs: Sequence[Leg], *, prudent: bool = False) -> float:
    """Guadagno atteso per 1 unita' puntata. Un rimborso (push) toglie la selezione: contribuisce con la sua probabilita' a quota 1."""
    if not legs:
        return 0.0
    return float(np.prod([(l.f_lo if prudent else l.f) for l in legs])) - 1.0


def uncertainty_label(ratio: float) -> str:
    """Convenzione semplice (non statistica): rapporto tra probabilita' congiunta alta e bassa."""
    return "Bassa" if ratio <= 1.5 else "Media" if ratio <= 2.5 else "Alta"


@dataclass
class Proposal:
    labels: list[str]
    legs: list[Leg]
    book: str
    mixed: bool
    target: float

    @property
    def odds(self) -> float: return total_odds(self.legs)
    @property
    def p_all(self) -> float: return joint_win_prob(self.legs)
    @property
    def p_lo(self) -> float: return float(np.prod([l.p_lo for l in self.legs]))
    @property
    def p_hi(self) -> float: return float(np.prod([l.p_hi for l in self.legs]))
    @property
    def ev(self) -> float: return combo_ev(self.legs)
    @property
    def ev_prudent(self) -> float: return combo_ev(self.legs, prudent=True)
    @property
    def n(self) -> int: return len(self.legs)
    @property
    def uncertainty(self) -> str: return uncertainty_label(self.p_hi / self.p_lo if self.p_lo > 0 else math.inf)
    @property
    def has_push_risk(self) -> bool: return any(l.p_push > 0 for l in self.legs)
    @property
    def ids(self) -> frozenset: return frozenset(l.id for l in self.legs)
    @property
    def books(self) -> list[str]: return sorted({l.book for l in self.legs})


@dataclass
class BuildResult:
    proposals: list[Proposal] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    funnel: list[tuple[str, int]] = field(default_factory=list)
    n_legs: int = 0
    n_games: int = 0
    n_window: int = 0               # combinazioni trovate nella finestra di quota (con o senza EV positivo)
    best_ev_seen: float | None = None
    books: dict[str, int] = field(default_factory=dict)   # bookmaker -> partite con almeno una selezione valida


# ---- selezioni ammissibili -------------------------------------------------------------------------------
def eligible_legs(rows: pd.DataFrame, now: datetime, *, markets: Sequence[str] | None = None, books: Sequence[str] | None = None,
                  include_under: bool = False, min_games: int = MIN_GAMES) -> tuple[list[Leg], list[tuple[str, int]]]:
    """Selezioni utilizzabili e imbuto dei filtri (quante quote restano a ogni passo). Una riga = una quota reale di un bookmaker."""
    funnel: list[tuple[str, int]] = []
    if rows is None or rows.empty:
        return [], [("Quote disponibili", 0)]
    d = rows
    funnel.append(("Quote disponibili", len(d)))
    d = d[d["priced"]]
    funnel.append(("Con probabilità stimata dal modello", len(d)))
    d = d[~d["manuale"] & ~d["obsoleta"]]
    funnel.append(("Quota recente e non inserita a mano", len(d)))
    d = d[d["inizio"].notna()]
    d = d[[pd.Timestamp(t).to_pydatetime() > now for t in d["inizio"]]] if len(d) else d
    funnel.append(("Partita non ancora iniziata", len(d)))
    d = d[d["n_partite"] >= min_games]
    funnel.append((f"Storico di almeno {min_games} partite", len(d)))
    d = d[d["p_win_lo"].notna() & d["p_win_hi"].notna() & (d["p_win"] > 0) & (d["quota"] > 1)]
    funnel.append(("Con intervallo di incertezza", len(d)))
    if "infortunio" in d.columns:
        d = d[[injury_level(s) != OUT for s in d["infortunio"].fillna("")]]
    funnel.append(("Giocatore non segnalato «Out»", len(d)))
    if not include_under:
        d = d[d["lato"] == "Over"]
    funnel.append(("Lato ammesso (Over" + (" e Under)" if include_under else ")"), len(d)))
    if markets:
        d = d[d["mercato"].isin(markets)]
    funnel.append(("Mercati scelti", len(d)))
    if books:
        d = d[d["bookmaker"].isin(books)]
    funnel.append(("Bookmaker scelti", len(d)))
    legs = []
    for r in d.itertuples():
        lo = min(float(r.p_win_lo), float(r.p_win))
        hi = max(float(r.p_win_hi), float(r.p_win))
        game = f"{r.gday}|{r.evento}"
        legs.append(Leg(
            id=f"{r.prop_id}|{r.lato}|{r.book}", key=r.key, giocatore=r.giocatore, squadra=str(r.squadra or ""), game=game, evento=r.evento,
            inizio=pd.Timestamp(r.inizio), market=r.market, mercato=r.mercato, linea=float(r.linea), lato=r.lato, book=r.bookmaker,
            odds=float(r.quota), p_win=float(r.p_win), p_push=float(r.p_push) if r.p_push == r.p_push else 0.0, p_lo=lo, p_hi=hi,
            n_games=int(r.n_partite), eta_min=float(r.eta_min) if r.eta_min == r.eta_min else float("nan"),
            quote_known=bool(r.ts_nota), prop_id=r.prop_id,
            infortunio=(getattr(r, "infortunio", "") if isinstance(getattr(r, "infortunio", ""), str) else "")))
    return legs, funnel


def best_across_books(legs: Sequence[Leg]) -> list[Leg]:
    """Per ogni giocata (prop + lato) tiene la quota piu' alta fra i bookmaker (solo per la modalita' 'bookmaker diversi')."""
    best: dict[tuple, Leg] = {}
    for l in legs:
        k = (l.prop_id, l.lato)
        if k not in best or l.odds > best[k].odds:
            best[k] = l
    return list(best.values())


# ---- ricerca ------------------------------------------------------------------------------------------------
def _dp(legs: Sequence[Leg], weights: Sequence[float], max_sel: int, max_bucket: int):
    """Programmazione dinamica: una selezione al massimo per partita; massimizza la somma dei pesi per ogni (n. selezioni, quota)."""
    games: dict[str, list[int]] = {}
    for i, l in enumerate(legs):
        games.setdefault(l.game, []).append(i)
    buckets = [max(1, int(round(math.log(l.odds) / STEP))) for l in legs]
    val = np.full((max_sel + 1, max_bucket + 1), -np.inf)
    val[0, 0] = 0.0
    stages = []
    for idxs in games.values():
        new = val.copy()
        choice = np.full(val.shape, -1, dtype=np.int32)
        for j in idxs:
            b = buckets[j]
            if b > max_bucket:
                continue
            cand = val[:-1, : max_bucket + 1 - b] + weights[j]
            cur = new[1:, b:]
            mask = cand > cur
            cur[mask] = cand[mask]
            choice[1:, b:][mask] = j
        stages.append(choice)
        val = new
    return val, stages, buckets


def _backtrack(stages, buckets, n: int, b: int) -> list[int]:
    out = []
    for choice in reversed(stages):
        j = int(choice[n, b])
        if j >= 0:
            out.append(j)
            n -= 1
            b -= buckets[j]
    return out


def _pool(legs: Sequence[Leg], lo: float, hi: float, min_sel: int, max_sel: int) -> list[list[Leg]]:
    """Combinazioni complete con quota totale in [lo, hi], trovate ottimizzando sia il valore atteso sia la probabilita'."""
    if not legs or len({l.game for l in legs}) < min_sel:
        return []
    max_bucket = int(math.ceil(math.log(hi) / STEP)) + int(max_sel) + 4
    blo = max(0, int(math.floor(math.log(lo) / STEP)) - max_sel)
    bhi = max_bucket
    found: dict[frozenset, list[Leg]] = {}
    for weights in ([math.log(l.f) if l.f > 0 else -50.0 for l in legs], [math.log(l.p_win) for l in legs]):
        val, stages, buckets = _dp(legs, weights, max_sel, max_bucket)
        for n in range(max(1, min_sel), max_sel + 1):
            for b in range(blo, bhi + 1):
                if not np.isfinite(val[n, b]):
                    continue
                combo = [legs[j] for j in _backtrack(stages, buckets, n, b)]
                if len(combo) != n:
                    continue
                o = total_odds(combo)
                if lo <= o <= hi:
                    found.setdefault(frozenset(l.id for l in combo), combo)
    return list(found.values())


def _choose(cands: list[list[Leg]], key) -> list[Leg] | None:
    return max(cands, key=key) if cands else None


def search(legs: Sequence[Leg], target: float, *, tolerance: float = 0.15, min_sel: int = 2, max_sel: int = 8,
           mixed_books: bool = False, require_prudent: bool = False, books_filter: Sequence[str] | None = None,
           min_ev: float = 0.0) -> BuildResult:
    """Cerca le combinazioni complete e restituisce fino a tre proposte (migliore EV, piu' probabile, meno selezioni)."""
    res = BuildResult()
    lo, hi = target * (1 - tolerance), target * (1 + tolerance)
    legs = [l for l in legs if (not books_filter or l.book in books_filter)]
    res.n_legs = len(legs)
    res.n_games = len({l.game for l in legs})
    by_book: dict[str, list[Leg]] = {}
    for l in legs:
        by_book.setdefault(l.book, []).append(l)
    res.books = {b: len({l.game for l in ls}) for b, ls in by_book.items()}
    if not legs:
        res.reasons.append("Nessuna selezione valida con questi filtri: guarda i passaggi nel controllo dati.")
        return res

    runs = [("Vari", best_across_books(legs), True)] if mixed_books else [(b, ls, False) for b, ls in by_book.items()]
    admissible: list[tuple[list[Leg], str, bool]] = []
    for name, ls, mixed in runs:
        for combo in _pool(ls, lo, hi, min_sel, max_sel):
            res.n_window += 1
            ev = combo_ev(combo)
            res.best_ev_seen = ev if res.best_ev_seen is None else max(res.best_ev_seen, ev)
            if ev > min_ev and (not require_prudent or combo_ev(combo, prudent=True) > 0):
                admissible.append((combo, name, mixed))
    if not admissible:
        res.reasons = _diagnose(res, runs, target, lo, hi, min_sel, max_sel, mixed_books, require_prudent)
        return res

    def mk(item, labels):
        combo, name, mixed = item
        return Proposal(labels, sorted(combo, key=lambda l: l.inizio), name, mixed or len({l.book for l in combo}) > 1, target)

    ev_of = lambda it: (combo_ev(it[0]), joint_win_prob(it[0]))
    best_ev = max(admissible, key=ev_of)
    best_p = max(admissible, key=lambda it: (joint_win_prob(it[0]), combo_ev(it[0])))
    n_min = min(len(it[0]) for it in admissible)
    few = max((it for it in admissible if len(it[0]) == n_min), key=ev_of)
    merged: dict[frozenset, tuple] = {}
    for label, it in ((LABEL_EV, best_ev), (LABEL_PROB, best_p), (LABEL_FEW, few)):
        k = frozenset(l.id for l in it[0])
        merged.setdefault(k, (it, []))[1].append(label)
    res.proposals = [mk(it, labels) for it, labels in merged.values()]
    return res


def _diagnose(res: BuildResult, runs, target, lo, hi, min_sel, max_sel, mixed, require_prudent) -> list[str]:
    out: list[str] = []
    n_games = res.n_games
    if n_games < min_sel:
        out.append(f"Ci sono solo {n_games} partite con selezioni valide, ma servono almeno {min_sel} selezioni (una per partita). "
                   "Con una sola selezione per partita non si può costruire questa multipla.")
        return out
    # quota massima raggiungibile con le partite disponibili
    best_per_game: dict[str, float] = {}
    for _, ls, _m in runs:
        for g in {l.game for l in ls}:
            best_per_game[g] = max(best_per_game.get(g, 1.0), max(l.odds for l in ls if l.game == g))
    top = sorted(best_per_game.values(), reverse=True)[:max_sel]
    reach = float(np.prod(top)) if top else 1.0
    if not mixed and len(runs) > 1:
        pass
    if reach < lo:
        out.append(f"Con le selezioni disponibili la quota massima raggiungibile è circa {reach:.1f}, sotto l'obiettivo {target:g} "
                   f"(accettato da {lo:.1f} a {hi:.1f}). Aumenta il numero massimo di selezioni o la tolleranza, oppure scegli una quota più bassa.")
    if res.n_window == 0 and reach >= lo:
        out.append("Esistono selezioni a sufficienza, ma nessuna combinazione cade nella finestra di quota richiesta"
                   + ("" if mixed else " usando un solo bookmaker") + ". Prova ad aumentare la tolleranza"
                   + ("" if mixed else " oppure a consentire bookmaker diversi nei filtri avanzati") + ".")
    elif res.n_window > 0:
        ev = res.best_ev_seen
        out.append(f"Ho trovato {res.n_window} combinazioni vicine alla quota {target:g}, ma nessuna ha un valore atteso positivo "
                   f"(la migliore: {100 * ev:+.1f}%)." if not require_prudent or (ev is not None and ev <= 0) else
                   f"Ho trovato {res.n_window} combinazioni con valore atteso positivo, ma nessuna lo resta nello scenario prudente.")
        out.append("È un risultato normale: le quote dei bookmaker includono un margine che, moltiplicato su più selezioni, "
                   "pesa di più. Non costruisco multiple con valore negativo solo per raggiungere la quota.")
    if not mixed and len(res.books) > 1 and res.n_window == 0:
        small = [b for b, n in res.books.items() if n < min_sel]
        if small:
            out.append("Bookmaker con troppo poche partite coperte: " + ", ".join(f"{b} ({res.books[b]})" for b in small) + ".")
    return out or ["Nessuna combinazione affidabile trovata con questi filtri."]


# ---- dettagli di una selezione (solo dati reali dello storico caricato) ---------------------------------------
def leg_stats(logs: Sequence, leg: Leg) -> dict:
    """Ultime 5/10/20 partite: frequenza con cui la giocata sarebbe riuscita, media e minuti; storico contro l'avversario."""
    from ..model.empirical import stat_value
    from ..models import Market

    market = leg.market if isinstance(leg.market, Market) else Market(getattr(leg.market, "value", leg.market))
    day = leg.inizio.date().isoformat()
    past = sorted([l for l in logs if l.game_date.isoformat() < day], key=lambda l: l.game_date)

    def win(v: float) -> bool | None:
        return v > leg.linea if leg.lato == "Over" else v < leg.linea if leg.lato == "Under" else None

    windows = {}
    for w in (5, 10, 20):
        part = past[-w:]
        vals = [stat_value(l, market) for l in part]
        windows[w] = {
            "n": len(part), "riuscita": sum(1 for v in vals if win(v)) / len(vals) if vals else None,
            "media": float(np.mean(vals)) if vals else None, "minuti": float(np.mean([l.minutes for l in part])) if part else None,
        }
    opp = None
    if " @ " in leg.evento:
        away, home = leg.evento.split(" @ ", 1)
        mine = (leg.squadra or "").strip().lower()
        opp = home if mine and mine == away.strip().lower() else away if mine and mine == home.strip().lower() else None
    vs = None
    if opp:
        same = [l for l in past if l.opponent and str(l.opponent).strip().lower() == opp.strip().lower()]
        vals = [stat_value(l, market) for l in same]
        vs = {"avversario": opp, "n": len(same), "media": float(np.mean(vals)) if vals else None,
              "riuscita": sum(1 for v in vals if win(v)) / len(vals) if vals else None}
    return {"finestre": windows, "contro_avversario": vs}


def simple_reason(leg: Leg, stats: dict | None) -> str:
    """Spiegazione breve e fattuale del perche' la selezione e' stata scelta."""
    fair = (1 - leg.p_push) / leg.p_win if leg.p_win > 0 else float("nan")
    edge = 100 * (leg.f - 1)
    txt = f"Il modello stima {100 * leg.p_win:.0f}% di probabilità: la quota equa sarebbe {fair:.2f} e {leg.book} offre {leg.odds:.2f}. "
    txt += (f"Il prezzo è più alto del dovuto (guadagno atteso della singola {edge:+.1f}%)." if edge > 0 else
            f"Da sola ha guadagno atteso {edge:+.1f}%: è inclusa perché, nella combinazione, avvicina la quota totale all'obiettivo con la probabilità più alta possibile.")
    if stats:
        w10 = stats["finestre"].get(10)
        if w10 and w10["n"]:
            txt += f" Nelle ultime {w10['n']} partite la giocata sarebbe riuscita {round(100 * w10['riuscita'])}% delle volte (media {w10['media']:.1f})."
    txt += f" Storico usato dal modello: {leg.n_games} partite."
    return txt
