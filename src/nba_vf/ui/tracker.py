"""Registro delle giocate consigliate e dei loro risultati reali, piu' la taratura delle probabilita'.

Logica senza Streamlit. Il registro e' l'unico modo onesto per sapere se le percentuali del modello sono credibili:
ogni giocata con valore viene salvata PRIMA della partita (solo la prima volta che compare, con la quota di quel
momento) e, a partita finita, viene chiusa con la statistica reale presa dalle API. Niente viene modificato a posteriori.

- Solo modalita' Live: i dati Demo non entrano mai nel registro.
- Una giocata e' identificata da partita + giocatore + mercato + linea + lato. Se riappare con un'altra quota
  non viene registrata di nuovo (resta quella vista per prima: nessun «ritocco» a posteriori).
- Se il giocatore non ha giocato (statistica assente o 0 minuti) la giocata e' «Annullata» e non conta nelle statistiche.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd

from .. import storage
from ..model.empirical import stat_value
from ..models import Market, PlayerGameLog, parse_iso
from .best import reliability
from .board import NY
from .labels import game_title, team_label, to_local
from .snapshot import cache_dir
from .tables import best_per_prop

STREAM_PICKS, STREAM_RESULTS = "pick_log", "pick_results"
GAME_DURATION = timedelta(hours=4)  # dopo inizio + 4 ore la partita si considera conclusa
VOID_AFTER = timedelta(days=2)  # senza statistica dopo 2 giorni la giocata e' «Annullata o senza dato»
MIN_FOR_VERDICT = 30

WIN, LOSS, PUSH, PENDING, VOID = "Vinta", "Persa", "Rimborso", "In attesa", "Annullata"
OUTCOME_LABEL = {"win": WIN, "loss": LOSS, "push": PUSH}
TRACK_COLUMNS = ["registrata", "giocatore", "squadra", "partita", "inizio", "giocata", "bookmaker", "quota", "prob",
                 "affidabilita", "esito", "risultato", "profitto"]
_known_ids: set[str] | None = None


# ---- persistenza -------------------------------------------------------------------------------
def _path(stream: str):
    return cache_dir() / f"{stream}.jsonl"


def _read(stream: str) -> list[dict]:
    out: dict[str, dict] = {}
    texts: list[str] = []
    p = _path(stream)
    if p.exists():
        texts += [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if storage.enabled():
        try:
            texts += storage.get_backend().read(stream)
        except Exception:  # l'archivio remoto e' un extra: se non risponde si usa il file locale
            pass
    for t in texts:
        try:
            rec = json.loads(t)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("id"):
            out.setdefault(rec["id"], rec)  # vale la prima registrazione: nessuna riscrittura a posteriori
    return list(out.values())


def _append(stream: str, rec: dict) -> None:
    line = json.dumps(rec, ensure_ascii=False)
    p = _path(stream)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if storage.enabled():
        try:
            storage.get_backend().append(stream, line)
        except Exception:
            pass


def load_picks() -> list[dict]:
    return _read(STREAM_PICKS)


def load_results() -> dict[str, dict]:
    return {r["id"]: r for r in _read(STREAM_RESULTS)}


def reset_cache() -> None:
    """Dimentica gli ID gia' noti (usato dai test e dopo cambi di cartella dati)."""
    global _known_ids
    _known_ids = None


# ---- registrazione ------------------------------------------------------------------------------
def pick_id(gday: str, key: str, market, line: float, side) -> str:
    return f"{gday}|{key}|{getattr(market, 'value', market)}|{line:g}|{getattr(side, 'value', side)}"


def _bdl_ids(bundle) -> dict[str, int]:
    ids: dict[str, int] = {}
    for q in bundle.quotes:
        if q.provider == "balldontlie" and q.player_id is not None:
            from .labels import normalize_name

            ids.setdefault(normalize_name(q.player_name), int(q.player_id))
    return ids


def candidate_records(rows: pd.DataFrame, bundle, now: datetime) -> list[dict]:
    """Giocate con valore positivo, quota aggiornata, partita non ancora iniziata: la migliore quota per ogni giocata."""
    if rows is None or rows.empty or bundle.is_demo:
        return []
    d = rows[rows["priced"] & (rows["roi"] > 0) & ~rows["manuale"] & ~rows["obsoleta"]]
    d = d[d["inizio"].notna()]
    if d.empty:
        return []
    d = d[[pd.Timestamp(t).to_pydatetime() > now for t in d["inizio"]]]
    if d.empty:
        return []
    d = best_per_prop(d, include_stale=True)
    ids = _bdl_ids(bundle)
    out = []
    for r in d.itertuples():
        start = pd.Timestamp(r.inizio).to_pydatetime()
        out.append({
            "id": pick_id(r.gday, r.key, r.market, r.linea, r.side), "logged_at": now.isoformat(), "gday": r.gday, "key": r.key,
            "giocatore": r.giocatore, "squadra": r.squadra, "evento": r.evento, "inizio": start.isoformat(),
            "market": getattr(r.market, "value", r.market), "mercato": r.mercato, "linea": float(r.linea),
            "side": getattr(r.side, "value", r.side), "lato": r.lato,
            "bookmaker": r.bookmaker, "quota": float(r.quota), "p_win": float(r.p_win), "p_push": float(r.p_push),
            "roi": float(r.roi), "roi_lo": None if r.roi_lo != r.roi_lo else float(r.roi_lo),
            "n_partite": int(r.n_partite), "affidabilita": reliability(r.n_partite, r.roi, r.roi_lo),
            "player_id": ids.get(r.key), "fonte": r.fonte,
        })
    return out


def record_picks(rows: pd.DataFrame, bundle, now: datetime) -> int:
    """Salva le giocate nuove. Restituisce quante ne ha aggiunte. Non solleva mai errori verso la pagina."""
    global _known_ids
    try:
        if _known_ids is None:
            _known_ids = {r["id"] for r in load_picks()}
        new = [c for c in candidate_records(rows, bundle, now) if c["id"] not in _known_ids]
        for c in new:
            _append(STREAM_PICKS, c)
            _known_ids.add(c["id"])
        return len(new)
    except Exception:
        return 0


# ---- chiusura con il risultato reale -----------------------------------------------------------------
def grade(side: str, line: float, value: float) -> str:
    if value == line:
        return "push"
    over = value > line
    return "win" if over == (side == "over") else "loss"


def _find_log(logs, gday: str) -> PlayerGameLog | None:
    day = date.fromisoformat(gday)
    for l in logs or []:
        if l.game_date == day:
            return l
    return None


def _finished(rec: dict, now: datetime) -> bool:
    start = parse_iso(rec.get("inizio"))
    if start is not None:
        return now >= start + GAME_DURATION
    return date.fromisoformat(rec["gday"]) < now.astimezone(NY).date()


def result_from_log(rec: dict, log: PlayerGameLog, now: datetime) -> dict:
    if not log.minutes or log.minutes <= 0:
        return {"id": rec["id"], "settled_at": now.isoformat(), "status": "void", "value": None, "outcome": None,
                "note": "non ha giocato"}
    value = stat_value(log, Market(rec["market"]))
    return {"id": rec["id"], "settled_at": now.isoformat(), "status": "final", "value": value,
            "outcome": grade(rec["side"], rec["linea"], value), "note": ""}


@dataclass
class SettleReport:
    settled: int = 0
    void: int = 0
    fetched: int = 0
    errors: int = 0
    waiting: int = 0


def settle_pending(records: list[dict], results: dict[str, dict], logs_by_key: dict[str, list[PlayerGameLog]],
                   fetch: Callable[[int, date], list[PlayerGameLog]] | None, now: datetime, *, max_fetch: int = 40) -> SettleReport:
    """Chiude le giocate concluse usando prima lo storico gia' scaricato, poi (se serve) una richiesta per giocatore."""
    rep = SettleReport()
    memo: dict[tuple[int, str], list[PlayerGameLog] | None] = {}
    for rec in records:
        if rec["id"] in results:
            continue
        if not _finished(rec, now):
            rep.waiting += 1
            continue
        log = _find_log(logs_by_key.get(rec["key"]), rec["gday"])
        fetched_ok = False
        pid = rec.get("player_id")
        if log is None and fetch is not None and pid:
            mk = (int(pid), rec["gday"])
            if mk not in memo:
                if rep.fetched >= max_fetch:
                    rep.waiting += 1
                    continue
                rep.fetched += 1
                try:
                    memo[mk] = fetch(int(pid), date.fromisoformat(rec["gday"]))
                except Exception:
                    memo[mk] = None
                    rep.errors += 1
            if memo[mk] is not None:
                fetched_ok = True
                log = _find_log(memo[mk], rec["gday"])
        if log is not None:
            res = result_from_log(rec, log, now)
        elif fetched_ok and (parse_iso(rec.get("inizio")) or now) + VOID_AFTER < now:
            res = {"id": rec["id"], "settled_at": now.isoformat(), "status": "void", "value": None, "outcome": None,
                   "note": "nessuna statistica trovata (non ha giocato o dato assente)"}
        else:
            rep.waiting += 1
            continue
        _append(STREAM_RESULTS, res)
        results[rec["id"]] = res
        rep.void += res["status"] == "void"
        rep.settled += res["status"] == "final"
    return rep


# ---- tabelle e statistiche ---------------------------------------------------------------------------
def track_frame(records: list[dict], results: dict[str, dict], logs_by_key: dict[str, list[PlayerGameLog]] | None = None,
                now: datetime | None = None) -> pd.DataFrame:
    """Una riga per giocata registrata, con l'esito. Se il risultato non e' ancora salvato ma lo storico gia' scaricato
    contiene la partita, l'esito viene calcolato al volo (senza salvarlo)."""
    now = now or datetime.now(timezone.utc)
    rows = []
    for rec in records:
        res = results.get(rec["id"])
        if res is None and logs_by_key is not None and _finished(rec, now):
            log = _find_log(logs_by_key.get(rec["key"]), rec["gday"])
            if log is not None:
                res = result_from_log(rec, log, now)
        d = rec["quota"]
        if res is None:
            esito, value, profit, outcome = PENDING, None, None, None
        elif res["status"] == "void":
            esito, value, profit, outcome = VOID, None, None, None
        else:
            outcome, value = res["outcome"], res["value"]
            esito = OUTCOME_LABEL[outcome]
            profit = d - 1 if outcome == "win" else (0.0 if outcome == "push" else -1.0)
        away, _, home = str(rec["evento"]).partition(" @ ")
        start = parse_iso(rec.get("inizio"))
        rows.append({
            "id": rec["id"], "registrata": to_local(parse_iso(rec["logged_at"])), "giocatore": rec["giocatore"],
            "squadra": team_label(rec.get("squadra")), "partita": game_title(away, home) if home else rec["evento"],
            "inizio": to_local(start), "giocata": f"{rec['mercato']} {rec['lato']} {rec['linea']:g}",
            "mercato": rec["mercato"], "bookmaker": rec["bookmaker"], "quota": d, "prob": 100 * rec["p_win"],
            "p_win": rec["p_win"], "p_push": rec["p_push"], "roi": rec["roi"], "affidabilita": rec["affidabilita"],
            "esito": esito, "outcome": outcome, "risultato": value, "profitto": profit,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["id"] + TRACK_COLUMNS + ["mercato", "p_win", "p_push", "roi", "outcome"])
    return df.sort_values("registrata", ascending=False).reset_index(drop=True)


def export_csv(df: pd.DataFrame) -> bytes:
    """Registro completo in CSV (date in formato leggibile, probabilita' e quote come numeri) per analisi esterne."""
    out = df.copy()
    for c in ("registrata", "inizio"):
        if c in out.columns:
            out[c] = [x.isoformat() if hasattr(x, "isoformat") else "" for x in out[c]]
    return out.to_csv(index=False).encode("utf-8")


def settled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["outcome"].notna()] if len(df) else df


@dataclass
class Summary:
    n_logged: int
    n_settled: int
    n_pending: int
    n_void: int
    wins: int
    losses: int
    pushes: int
    expected_wins: float
    z_wins: float | None
    profit_units: float
    expected_units: float
    z_profit: float | None


def _z(obs: float, exp: float, var: float) -> float | None:
    return (obs - exp) / math.sqrt(var) if var > 1e-12 else None


def summarize(df: pd.DataFrame) -> Summary:
    s = settled(df)
    n = len(s)
    if n:
        p = s["p_win"].to_numpy(float)
        q = s["p_push"].to_numpy(float)
        d = s["quota"].to_numpy(float)
        roi = s["roi"].to_numpy(float)
        wins = int((s["outcome"] == "win").sum())
        profit = float(s["profitto"].sum())
        var_w = float((p * (1 - p)).sum())
        var_p = float((p * (d - 1) ** 2 + np.clip(1 - p - q, 0, 1) - roi ** 2).sum())
        exp_w, exp_p = float(p.sum()), float(roi.sum())
        z_w, z_p = _z(wins, exp_w, var_w), _z(profit, exp_p, var_p)
    else:
        wins, profit, exp_w, exp_p, z_w, z_p = 0, 0.0, 0.0, 0.0, None, None
    return Summary(
        n_logged=len(df), n_settled=n, n_pending=int((df["esito"] == PENDING).sum()) if len(df) else 0,
        n_void=int((df["esito"] == VOID).sum()) if len(df) else 0, wins=wins,
        losses=int((s["outcome"] == "loss").sum()) if n else 0, pushes=int((s["outcome"] == "push").sum()) if n else 0,
        expected_wins=exp_w, z_wins=z_w, profit_units=profit, expected_units=exp_p, z_profit=z_p,
    )


def verdict(sm: Summary) -> tuple[str, str]:
    """(livello, testo) in linguaggio semplice. Sotto MIN_FOR_VERDICT giocate chiuse non si conclude nulla."""
    if sm.n_settled < MIN_FOR_VERDICT:
        return "info", (f"Troppo presto per giudicare: {sm.n_settled} giocate chiuse su {MIN_FOR_VERDICT} minime. "
                        "Con pochi casi la fortuna pesa più del modello: un risultato brillante o pessimo adesso non dice nulla.")
    zs = [z for z in (sm.z_wins, sm.z_profit) if z is not None]
    if not zs:
        return "info", "Non ci sono abbastanza differenze tra i casi per confrontare previsto e realizzato."
    worst, best = min(zs), max(zs)
    if worst < -2:
        return "warning", ("Finora il risultato reale è PEGGIORE di quanto il modello prevedeva, oltre la normale variabilità. "
                           "Le percentuali mostrate sono probabilmente troppo ottimistiche: usale con molta cautela.")
    if best > 2:
        return "success", ("Finora il risultato reale è MIGLIORE di quanto il modello prevedeva, oltre la normale variabilità. "
                           "Può essere fortuna o prudenza del modello: servono più giocate per distinguerle.")
    return "success", ("Finora previsto e realizzato sono coerenti, entro la normale variabilità. "
                       "Non è una garanzia sul futuro: con il tempo il confronto diventa più affidabile.")


BINS = [0.0, 0.40, 0.50, 0.60, 0.70, 1.01]
BIN_LABELS = ["sotto 40%", "40–50%", "50–60%", "60–70%", "70% e oltre"]


def _wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration_table(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Per fascia di probabilita' prevista (o per `by`): quante giocate, quanto previsto, quanto realizzato (intervallo 90%)."""
    s = settled(df)
    cols = ["gruppo", "giocate", "previsto", "realizzato", "da", "a"]
    if s.empty:
        return pd.DataFrame(columns=cols)
    s = s.assign(gruppo=pd.cut(s["p_win"], BINS, labels=BIN_LABELS, right=False) if by is None else s[by])
    out = []
    for g, part in s.groupby("gruppo", observed=True):
        n = len(part)
        k = int((part["outcome"] == "win").sum())
        lo, hi = _wilson(k, n)
        out.append({"gruppo": str(g), "giocate": n, "previsto": 100 * float(part["p_win"].mean()), "realizzato": 100 * k / n,
                    "da": 100 * lo, "a": 100 * hi})
    res = pd.DataFrame(out, columns=cols)
    if by is None:
        res["_o"] = res["gruppo"].map({l: i for i, l in enumerate(BIN_LABELS)})
        res = res.sort_values("_o").drop(columns="_o")
    return res.reset_index(drop=True)
