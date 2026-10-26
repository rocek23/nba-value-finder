"""Quote storiche utilizzabili in un backtest: solo quelle REALMENTE osservate prima dell'inizio, con timestamp.

Regole (tutte verificate nei test):
  1. serve l'orario d'inizio (``commence_time``): senza, non si puo' dimostrare che la quota fosse pre-partita -> scartata;
  2. l'istante di cattura e' ``fetched_at`` (quando noi l'abbiamo vista), mai un orario dichiarato dalla fonte che
     sia piu' tardi di essa; una quota catturata a/dopo l'inizio (meno ``lead_minutes``) e' scartata;
  3. per ogni (partita, giocatore, mercato, linea, libro, lato) si usa l'ULTIMA quota catturata prima del limite;
  4. nessuna quota "attuale" e' mai applicata a partite passate: ogni quota vale solo per la partita del suo evento;
  5. il margine del libro e' misurato solo da coppie Over/Under dello STESSO prelievo (stesso ``fetched_at``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from ..models import OddsQuote, PlayerGameLog, Side
from ..ui.labels import normalize_name

NY = ZoneInfo("America/New_York")
CLOCK_TOLERANCE = timedelta(minutes=1)

REASONS = {
    "senza_giocatore": "quota senza nome del giocatore",
    "senza_orario_inizio": "orario d'inizio ignoto: impossibile provare che sia pre-partita",
    "dopo_inizio": "catturata a partita iniziata (o dopo)",
    "troppo_vicina_all_inizio": "catturata dopo il limite scelto prima dell'inizio",
    "timestamp_incoerente": "quote_time successivo al momento di cattura",
    "quota_non_valida": "quota decimale <= 1 o non finita",
    "linea_non_supportata": "linea non multipla di 0.5",
    "partita_non_trovata": "nessuna partita del giocatore nella data dell'evento (assente/non ha giocato?)",
    "linea_fuori_griglia": "linea fuori dalla griglia del modello",
}


def utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def capture_of(q: OddsQuote) -> datetime:
    """Istante in cui la quota e' stata realmente osservata: ``fetched_at`` per le API; per i CSV, che si importano DOPO,
    il ``timestamp`` dichiarato dall'utente (verificabile solo sulla sua parola: Da verificare)."""
    if q.source == "csv" and q.quote_time is not None:
        return utc(q.quote_time)
    return utc(q.fetched_at)


def game_date_of(commence: datetime) -> date:
    """Data di calendario della partita = data ET dell'inizio (come nei box score). Da verificare sul provider reale."""
    return utc(commence).astimezone(NY).date()


@dataclass(frozen=True)
class SelectedQuote:
    quote: OddsQuote
    player_key: str  # nome normalizzato
    game_date: date
    capture_time: datetime
    novig_p_over: float | None  # probabilita' Over senza margine (solo se c'e' la coppia nello stesso prelievo)
    overround: float | None  # 1/do + 1/du - 1

    @property
    def minutes_before_start(self) -> float:
        return (utc(self.quote.commence_time) - self.capture_time).total_seconds() / 60.0


def reject_reason(q: OddsQuote, lead_minutes: float = 0.0) -> str | None:
    if not q.player_name or not normalize_name(q.player_name):
        return "senza_giocatore"
    if q.commence_time is None:
        return "senza_orario_inizio"
    if not (q.odds_decimal and q.odds_decimal > 1.0 and q.odds_decimal < 1e6):
        return "quota_non_valida"
    if abs(2 * q.line - round(2 * q.line)) > 1e-9 or q.line < 0:
        return "linea_non_supportata"
    cap, start = capture_of(q), utc(q.commence_time)
    if q.source != "csv" and q.quote_time is not None and utc(q.quote_time) > cap + CLOCK_TOLERANCE:
        return "timestamp_incoerente"
    if cap >= start:
        return "dopo_inizio"
    if cap > start - timedelta(minutes=lead_minutes):
        return "troppo_vicina_all_inizio"
    return None


def select_pre_game_quotes(quotes: Iterable[OddsQuote], lead_minutes: float = 0.0) -> tuple[list[SelectedQuote], Counter]:
    """Filtra e sceglie l'ultima quota pre-partita per ogni (partita, giocatore, mercato, linea, libro, lato)."""
    rej: Counter = Counter()
    groups: dict[tuple, OddsQuote] = {}
    snaps: dict[tuple, dict[datetime, dict[Side, OddsQuote]]] = defaultdict(lambda: defaultdict(dict))
    n_in = 0
    for q in quotes:
        n_in += 1
        reason = reject_reason(q, lead_minutes)
        if reason:
            rej[reason] += 1
            continue
        pk = normalize_name(q.player_name)
        gdate = game_date_of(q.commence_time)
        base = (q.event_id, pk, q.market, float(q.line), q.bookmaker)
        key = base + (q.side,)
        cur = groups.get(key)
        if cur is None or capture_of(q) >= capture_of(cur):
            groups[key] = q
        snaps[base][capture_of(q)][q.side] = q
    # ultima coppia Over/Under dello stesso prelievo, per libro e linea
    pairs: dict[tuple, tuple[datetime, float, float]] = {}
    for base, by_time in snaps.items():
        both = [t for t, s in by_time.items() if Side.OVER in s and Side.UNDER in s]
        if both:
            t = max(both)
            do, du = by_time[t][Side.OVER].odds_decimal, by_time[t][Side.UNDER].odds_decimal
            io, iu = 1 / do, 1 / du
            pairs[base] = (t, io / (io + iu), io + iu - 1.0)
    out: list[SelectedQuote] = []
    for key, q in groups.items():
        event_id, pk, market, line, book, _side = key
        pair = pairs.get((event_id, pk, market, line, book))
        nv = ovr = None
        if pair and pair[0] == capture_of(q):
            nv, ovr = pair[1], pair[2]
        out.append(SelectedQuote(q, pk, game_date_of(q.commence_time), capture_of(q), nv, ovr))
    out.sort(key=lambda s: (utc(s.quote.commence_time), s.player_key, s.quote.market.value, s.quote.line, s.quote.bookmaker, s.quote.side.value))
    return out, rej


def match_to_games(selected: Sequence[SelectedQuote], pool: Mapping[str, Sequence[PlayerGameLog]]) -> tuple[list[tuple[SelectedQuote, str, PlayerGameLog]], Counter]:
    """Abbina ogni quota alla partita del giocatore NELLA DATA DELL'EVENTO. Restituisce (quota, chiave pool, riga partita)."""
    by_name: dict[str, str] = {}
    index: dict[tuple[str, date], PlayerGameLog] = {}
    for key, logs in pool.items():
        for l in logs:
            nk = normalize_name(l.player_name)
            if nk:
                by_name[nk] = key
                if l.minutes and l.minutes > 0:
                    index[(nk, l.game_date)] = l
    rej: Counter = Counter()
    out = []
    for s in selected:
        log = index.get((s.player_key, s.game_date))
        if log is None:
            rej["partita_non_trovata"] += 1
            continue
        out.append((s, by_name[s.player_key], log))
    return out, rej


def margins_by_book(selected: Sequence[SelectedQuote]) -> dict[str, dict[str, float]]:
    """Margine (overround) medio e numero di coppie per libro, dalle sole coppie Over/Under dello stesso prelievo."""
    acc: dict[str, list[float]] = defaultdict(list)
    for s in selected:
        if s.overround is not None and s.quote.side is Side.OVER:
            acc[s.quote.bookmaker].append(s.overround)
    return {b: {"margine_medio": sum(v) / len(v), "coppie": len(v)} for b, v in acc.items()}


def load_quote_history(path: str | Path) -> list[OddsQuote]:
    """Legge lo storico locale ``quote_history.jsonl`` scritto da ``append_history`` a ogni «Aggiorna dati live»."""
    import json

    from ..ui.snapshot import quote_from_dict

    from .. import storage

    p = Path(path)
    out: list[OddsQuote] = []
    lines: list[str] = []
    if storage.enabled():  # su hosting con archivio remoto la fonte e' il database
        try:
            lines = storage.get_backend().read("quote_history")
        except Exception:
            lines = []
    if not lines and p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            for d in json.loads(line)["quotes"]:
                out.append(quote_from_dict(d))
        except (ValueError, KeyError):
            continue
    return out
