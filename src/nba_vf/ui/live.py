"""Modalita' LIVE: prelievo dalle API reali (adapter di fase 1) e caricamento dell'ultimo prelievo da disco.

Regole:
- Le pagine NON chiamano le API: leggono l'ultimo snapshot. Il prelievo parte solo dal pulsante "Aggiorna dati
  live" (le API costano crediti e hanno limiti di richieste).
- Se qualcosa manca (chiave assente, piano insufficiente, errore di rete, giocatore senza storico) lo si registra e
  lo si mostra. Non viene MAI generato nessun dato sostitutivo: nessun fallback sul generatore demo.
- Le quote inserite da CSV sono dati reali dell'utente e sono marcate come manuali.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any, Callable

from ..adapters.balldontlie import BalldontlieAdapter
from ..adapters.nba_api_adapter import NbaApiAdapter, current_season_start_year
from ..adapters.the_odds_api import TheOddsApiAdapter
from ..config import Settings, redact
from ..http import ApiError, TierError
from ..models import OddsQuote, PlayerGameLog, parse_iso, utc_now
from .bundle import DataBundle, EventInfo, PlayerInfo, ProviderStatus
from .labels import normalize_name
from .snapshot import Snapshot, append_history, load_snapshot, save_snapshot

EMPTY_LIVE_MESSAGE = (
    "Nessun dato live disponibile. Configura le chiavi nel file .env (vedi «Impostazioni e API») e premi "
    "«Aggiorna dati live», oppure importa un CSV di quote. L'interfaccia non sostituisce mai dati reali "
    "mancanti con dati inventati: per vedere un esempio passa alla modalità Demo."
)
LOGS_MAX_AGE = timedelta(hours=12)
MAX_PLAYERS_HARD_CAP = 400  # protezione: oltre questo numero il tempo di recupero non e' piu' ragionevole


@dataclass(frozen=True)
class CollectOptions:
    days_ahead: int = 1
    max_events: int = 8
    max_players: int | None = None  # None = tutti i giocatori che compaiono nelle quote
    seasons_back: int = 1  # stagioni precedenti da scaricare oltre a quella in corso (1 = 2026-27 e 2025-26)
    time_budget_s: float = 600.0  # oltre questo tempo il recupero si ferma e prosegue al prossimo aggiornamento
    min_games_current: int = 25
    use_odds_api: bool = True
    use_balldontlie_props: bool = True


def build_adapters(settings: Settings):
    """Crea gli adapter per le sole chiavi configurate. Restituisce (bdl, odds, nba) con None se assenti."""
    bdl = (
        BalldontlieAdapter(settings.balldontlie_api_key, requests_per_min=settings.balldontlie_requests_per_min,
                           timeout=settings.http_timeout, max_retries=settings.http_max_retries)
        if settings.balldontlie_api_key else None
    )
    odds = (
        TheOddsApiAdapter(settings.odds_api_key, regions=settings.odds_regions, timeout=settings.http_timeout,
                          max_retries=settings.http_max_retries)
        if settings.odds_api_key else None
    )
    return bdl, odds, NbaApiAdapter(sleep_seconds=settings.nba_api_sleep_seconds, timeout=max(settings.http_timeout, 20.0))


class LiveCollector:
    def __init__(self, *, bdl=None, odds=None, nba=None, secrets: list[str] | None = None,
                 now: Callable[[], datetime] = utc_now, progress: Callable[[float, str], None] | None = None) -> None:
        self.bdl, self.odds, self.nba = bdl, odds, nba
        self.secrets = secrets or []
        self.now = now
        self.progress = progress or (lambda frac, msg: None)

    def _err(self, exc: Exception) -> str:
        return redact(f"{type(exc).__name__}: {exc}", self.secrets)[:300]

    # ------------------------------------------------------------------------------------------
    def collect(self, opts: CollectOptions = CollectOptions(), previous: Snapshot | None = None) -> Snapshot:
        now = self.now()
        snap = Snapshot(fetched_at=now)
        if previous:  # le quote manuali (CSV) dell'utente sopravvivono all'aggiornamento
            snap.quotes = [q for q in previous.quotes if q.source == "csv"]
        providers: list[dict[str, Any]] = []
        events: dict[str, dict[str, Any]] = {}
        names_by_key: dict[str, str] = {}
        bdl_ids: dict[str, int] = {}
        api_quotes: list[OddsQuote] = []

        # ---- The Odds API: partite e quote ----------------------------------------------------
        if opts.use_odds_api:
            if self.odds is None:
                providers.append(_prov("The Odds API", "non_configurato", "THE_ODDS_API_KEY assente"))
            else:
                self.progress(0.05, "The Odds API: elenco partite")
                try:
                    evs = [e for e in self.odds.list_events() if self._within(e.get("commence_time"), now, opts.days_ahead)]
                    n_q, remaining, skipped_markets = 0, None, set()
                    for i, ev in enumerate(evs[: opts.max_events]):
                        self.progress(0.05 + 0.2 * i / max(1, len(evs)), f"The Odds API: props {ev.get('home_team')}")
                        res = self.odds.get_event_props(str(ev["id"]))
                        api_quotes.extend(res.parsed.quotes)
                        n_q += len(res.parsed.quotes)
                        remaining = res.quota.remaining
                        skipped_markets |= res.parsed.skipped_markets
                        events[str(ev["id"])] = {"home": ev.get("home_team"), "away": ev.get("away_team"),
                                                 "commence_time": ev.get("commence_time"), "source": "the_odds_api"}
                    detail = f"{len(evs)} partite, {n_q} quote" + (f"; crediti residui {remaining}" if remaining is not None else "")
                    if n_q == 0 and evs:
                        detail += " — nessuna player prop restituita (mercati non disponibili per queste partite o per il piano)"
                    providers.append(_prov("The Odds API", "ok", detail, n_q))
                except ApiError as exc:
                    providers.append(_prov("The Odds API", "errore", self._err(exc)))

        # ---- BALLDONTLIE: props (piano GOAT) ----------------------------------------------------
        if opts.use_balldontlie_props:
            if self.bdl is None:
                providers.append(_prov("BALLDONTLIE (props)", "non_configurato", "BALLDONTLIE_API_KEY assente"))
            else:
                n_q, status, detail = 0, "ok", ""
                try:
                    for d in range(opts.days_ahead + 1):
                        for g in self.bdl.get_games(now.date() + timedelta(days=d))[: opts.max_events]:
                            res = self.bdl.get_player_props(int(g["id"]))
                            home = (g.get("home_team") or {}).get("abbreviation")
                            away = (g.get("visitor_team") or {}).get("abbreviation")
                            events[str(g["id"])] = {"home": home, "away": away, "commence_time": g.get("datetime") or g.get("date"),
                                                    "source": "balldontlie"}
                            for q in res.quotes:
                                api_quotes.append(replace(q, home_team=home, away_team=away,
                                                          commence_time=parse_iso(g.get("datetime") or g.get("date"))))
                            n_q += len(res.quotes)
                    detail = f"{n_q} quote"
                except TierError as exc:
                    status, detail = "non_disponibile", f"piano insufficiente per le props: {self._err(exc)}"
                except ApiError as exc:
                    status, detail = "errore", self._err(exc)
                providers.append(_prov("BALLDONTLIE (props)", status, detail, n_q))
                missing_names = sorted({q.player_id for q in api_quotes if q.player_name is None and q.player_id is not None})
                if missing_names:
                    try:
                        id_to_name = self.bdl.resolve_player_names(missing_names)
                        api_quotes = [replace(q, player_name=id_to_name.get(q.player_id)) if q.player_name is None else q for q in api_quotes]
                    except ApiError as exc:
                        snap.messages.append(["warning", f"Nomi giocatori BALLDONTLIE non risolti: {self._err(exc)}"])

        # quote senza nome non sono utilizzabili: si scartano e lo si dichiara
        usable = [q for q in api_quotes if q.player_name]
        dropped = len(api_quotes) - len(usable)
        if dropped:
            snap.messages.append(["warning", f"{dropped} quote scartate perché il nome del giocatore non è risolvibile."])
        snap.quotes = [q for q in snap.quotes if q.source == "csv"] + usable
        csv_n = sum(1 for q in snap.quotes if q.source == "csv")
        providers.append(_prov("Quote manuali (CSV)", "manuale" if csv_n else "non_disponibile",
                               f"{csv_n} quote importate" if csv_n else "nessun CSV importato", csv_n))

        # ---- storico giocatori -----------------------------------------------------------------
        counts: Counter[str] = Counter()
        ids_seen: dict[str, Counter[int]] = {}
        for q in snap.quotes:
            key = normalize_name(q.player_name)
            counts[key] += 1
            names_by_key.setdefault(key, q.player_name or key)
            if q.player_id is not None and q.provider == "balldontlie":
                ids_seen.setdefault(key, Counter())[int(q.player_id)] += 1
        bdl_ids = {k: c.most_common(1)[0][0] for k, c in ids_seen.items()}  # ID ufficiali BALLDONTLIE
        homonyms = {k for k, c in ids_seen.items() if len(c) > 1}  # stesso nome, ID diversi: non si indovina
        cap = opts.max_players if opts.max_players else MAX_PLAYERS_HARD_CAP
        ranked = [k for k, _ in counts.most_common()]
        targets = ranked[:cap]
        if len(ranked) > len(targets):
            snap.messages.append(["warning", f"Storico richiesto per {len(targets)} giocatori su {len(ranked)} (limite impostato: {cap}). "
                                             "I restanti restano non valutati."])
        season = current_season_start_year(now.date())
        seasons = [season - i for i in range(max(0, opts.seasons_back) + 1)]
        today = now.date()
        prev_logs_all = previous.logs if previous else {}
        prev_seasons_all = previous.logs_seasons if previous else {}
        prev_at_all = previous.logs_fetched_at if previous else {}

        def reusable(key: str) -> bool:
            at = prev_at_all.get(key)
            return bool(prev_logs_all.get(key)) and set(seasons) <= set(prev_seasons_all.get(key, [])) and at is not None and now - at < LOGS_MAX_AGE

        order = sorted(targets, key=lambda k: (reusable(k), -counts[k]))  # prima chi ha davvero bisogno di dati
        stats_ok = {"bdl": 0, "nba": 0}
        n_reused = n_incremental = n_pending = 0
        tier_blocked = nba_disabled = False
        nba_errors = 0
        pending: list[str] = []
        t0 = time.monotonic()
        for i, key in enumerate(order):
            self.progress(0.3 + 0.7 * i / max(1, len(order)), f"Storico {i + 1}/{len(order)}: {names_by_key[key]}")
            prev_logs = prev_logs_all.get(key, [])
            if reusable(key):
                snap.logs[key] = prev_logs
                snap.logs_fetched_at[key] = prev_at_all[key]
                snap.logs_seasons[key] = prev_seasons_all[key]
                n_reused += 1
                continue
            if time.monotonic() - t0 > opts.time_budget_s:  # limite di tempo: si prosegue al prossimo aggiornamento
                pending.append(key)
                n_pending += 1
                if prev_logs:  # si conserva lo storico vecchio, dichiarato come non aggiornato
                    snap.logs[key] = prev_logs
                    snap.logs_fetched_at[key] = prev_at_all.get(key, now)
                    snap.logs_seasons[key] = prev_seasons_all.get(key, [])
                snap.issues[key] = ["in attesa: limite di tempo raggiunto, premi di nuovo «Aggiorna dati live» per proseguire"]
                continue
            logs: list[PlayerGameLog] = []
            issues: list[str] = []
            keep_old_stamp = False
            fetched_ok = False  # una risposta valida del provider (anche vuota) e' un risultato, non un errore
            if key in homonyms:
                issues.append("nome ambiguo: più ID BALLDONTLIE per lo stesso nome, giocatore non attribuibile con certezza")
            elif self.bdl is not None and key in bdl_ids and not tier_blocked:
                # incrementale: solo se lo storico precedente viene tutto da BALLDONTLIE per le stesse stagioni
                same_seasons = set(seasons) <= set(prev_seasons_all.get(key, []))
                incremental = bool(prev_logs) and same_seasons and all(l.source == "balldontlie" for l in prev_logs)
                start = max(l.game_date for l in prev_logs) if incremental else None
                try:
                    new = self.bdl.get_player_game_logs(bdl_ids[key], seasons, start_date=start)
                    merged = {l.game_id: l for l in (prev_logs if incremental else [])}
                    merged.update({l.game_id: l for l in new})
                    logs = [l for l in merged.values() if l.game_date <= today]  # nessuna partita futura
                    fetched_ok = True
                    n_incremental += int(incremental)
                    stats_ok["bdl"] += bool(logs)
                    if not logs:
                        issues.append(f"BALLDONTLIE non ha partite per questo giocatore nelle stagioni {', '.join(f'{x}-{str(x + 1)[2:]}' for x in sorted(seasons))} "
                                      "(esordiente o mai sceso in campo)")
                except TierError as exc:
                    tier_blocked = True
                    issues.append(f"BALLDONTLIE statistiche: piano insufficiente ({self._err(exc)})")
                except ApiError as exc:
                    issues.append(f"BALLDONTLIE statistiche: {self._err(exc)}")
                    if prev_logs:  # l'errore non cancella quello che si aveva (e non lo fa passare per aggiornato)
                        logs, keep_old_stamp = prev_logs, True
            elif self.bdl is not None and key not in bdl_ids:
                issues.append("nessun ID BALLDONTLIE per questo giocatore (quote da altri provider)")
            if not fetched_ok and not logs and self.nba is not None and key not in homonyms and not nba_disabled:
                try:
                    matches = [m for m in self.nba.find_player(names_by_key[key]) if normalize_name(m.get("full_name")) == key]
                    if not matches:
                        issues.append("nba_api: giocatore non trovato per nome")
                    else:
                        pid = int(matches[0]["id"])
                        logs = self.nba.get_player_game_logs(pid, season)
                        if len(logs) < opts.min_games_current:
                            logs = self.nba.get_player_game_logs(pid, season - 1) + logs
                        logs = [l for l in logs if l.game_date <= today]
                        fetched_ok = True
                        stats_ok["nba"] += bool(logs)
                    nba_errors = 0
                except Exception as exc:  # rete, blocco IP, formato: si registra e si prosegue
                    issues.append(f"nba_api: {self._err(exc)}")
                    nba_errors += 1
                    if nba_errors >= 3:  # dopo tre errori di fila non si perde altro tempo con nba_api
                        nba_disabled = True
                        snap.messages.append(["warning", "nba_api non risponde (probabile blocco dell'IP del server): disattivata per questo aggiornamento."])
            if not logs and self.bdl is None and self.nba is None:
                issues.append("nessun provider di statistiche disponibile")
            snap.logs[key] = sorted(logs, key=lambda l: (l.game_date, l.game_id))
            snap.logs_fetched_at[key] = prev_at_all.get(key, now) if keep_old_stamp else now
            if keep_old_stamp:
                snap.logs_seasons[key] = prev_seasons_all.get(key, [])
            elif fetched_ok:
                snap.logs_seasons[key] = seasons
            if issues or not logs:
                snap.issues[key] = issues or ["nessuna partita restituita"]
        providers.append(_prov("Statistiche giocatori", "ok" if any(stats_ok.values()) or n_reused else ("errore" if targets else "non_disponibile"),
                               f"BALLDONTLIE: {stats_ok['bdl']} giocatori scaricati ({n_incremental} in modo incrementale), {n_reused} dalla cache; "
                               f"nba_api: {stats_ok['nba']}; in attesa: {n_pending}", sum(stats_ok.values()) + n_reused))
        if pending:
            snap.messages.append(["warning", f"Recupero storico non terminato: {n_pending} giocatori in attesa (limite di tempo). Premi di nuovo «Aggiorna dati live»: riparte da dove si era fermato."])
        snap.logs = {k: v for k, v in snap.logs.items() if v}
        snap.players = {k: {"name": names_by_key[k], "team": (snap.logs[k][-1].team if k in snap.logs else None)} for k in targets}
        snap.events = events
        snap.providers = providers
        snap.retrieval = {
            "seasons": seasons, "players_in_quotes": len(ranked), "players_requested": len(targets), "with_logs": len(snap.logs),
            "reused_from_cache": n_reused, "incremental": n_incremental, "pending": pending, "tier_blocked": tier_blocked,
            "nba_api_disabled": nba_disabled, "elapsed_s": round(time.monotonic() - t0, 1),
            "requests_bdl": getattr(self.bdl, "request_count", None), "finished_at": now.isoformat(), "complete": not pending,
        }
        self.progress(1.0, "Completato")
        return snap

    @staticmethod
    def _within(value: str | None, now: datetime, days: int) -> bool:
        dt = parse_iso(value)
        return dt is not None and now - timedelta(hours=4) <= dt <= now + timedelta(days=days + 1)


def _prov(name: str, status: str, detail: str, n: int = 0) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "n_quotes": n}


# --------------------------------------------------------------------------------------------------
def bundle_from_snapshot(snap: Snapshot | None) -> DataBundle:
    if snap is None:
        return DataBundle(
            mode="live", source_label="Dati LIVE (API)", status="vuoto", updated_at=None,
            messages=[("warning", EMPTY_LIVE_MESSAGE)], fingerprint="live:vuoto",
        )
    messages = [(m[0], m[1]) for m in snap.messages]
    players = {k: PlayerInfo(k, v["name"], v.get("team")) for k, v in snap.players.items()}
    for q in snap.quotes:  # anche i giocatori delle quote senza storico devono avere un nome
        key = normalize_name(q.player_name)
        players.setdefault(key, PlayerInfo(key, q.player_name or key, None))
    events = {
        eid: EventInfo(eid, e.get("home"), e.get("away"), parse_iso(e.get("commence_time")), e.get("source", ""))
        for eid, e in snap.events.items()
    }
    providers = [ProviderStatus(p["name"], p["status"], p["detail"], p.get("n_quotes", 0)) for p in snap.providers]
    status = "ok" if snap.quotes and snap.logs else "parziale"
    if not snap.quotes:
        messages.append(("warning", "L'ultimo aggiornamento non contiene quote: controlla lo stato dei provider in «Impostazioni e API»."))
    if not snap.logs:
        messages.append(("warning", "L'ultimo aggiornamento non contiene storico partite: nessuna stima può essere calcolata."))
    return DataBundle(
        mode="live", source_label="Dati LIVE (API)", status=status, updated_at=snap.fetched_at, quotes=snap.quotes,
        logs=snap.logs, players=players, events=events, providers=providers, messages=messages,
        collection_issues=snap.issues, retrieval=snap.retrieval,
        fingerprint=f"live:{snap.fetched_at.isoformat()}:{len(snap.quotes)}:{sum(len(v) for v in snap.logs.values())}",
    )


def load_live_bundle() -> DataBundle:
    try:
        return bundle_from_snapshot(load_snapshot())
    except Exception as exc:  # file danneggiato: lo si dice, non lo si ignora
        b = bundle_from_snapshot(None)
        b.messages = [("error", f"Lo snapshot live salvato non è leggibile ({type(exc).__name__}). Premi «Aggiorna dati live» per rigenerarlo.")]
        return b


def refresh_live(settings: Settings, opts: CollectOptions, progress=None) -> Snapshot:
    bdl, odds, nba = build_adapters(settings)
    try:
        snap = LiveCollector(bdl=bdl, odds=odds, nba=nba, secrets=settings.secrets(), progress=progress).collect(opts, load_snapshot())
    finally:
        for a in (bdl, odds):
            if a is not None:
                a.close()
    save_snapshot(snap)
    append_history(snap)
    return snap


def add_manual_quotes(quotes: list[OddsQuote]) -> Snapshot:
    """Aggiunge quote CSV allo snapshot (le sostituisce se gia' presenti). Non richiede la rete."""
    snap = load_snapshot() or Snapshot(fetched_at=utc_now())
    keep = [q for q in snap.quotes if q.source != "csv"]
    snap.quotes = keep + quotes
    snap.providers = [p for p in snap.providers if p["name"] != "Quote manuali (CSV)"] + [
        _prov("Quote manuali (CSV)", "manuale", f"{len(quotes)} quote importate", len(quotes))
    ]
    snap.fetched_at = utc_now()
    save_snapshot(snap)
    return snap
