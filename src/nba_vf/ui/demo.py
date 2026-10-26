"""Modalita' DEMO: lega, partite, giocatori e quote interamente SINTETICI.

Niente qui e' reale: nomi, squadre, statistiche e quote sono inventati (seed fisso) per mostrare il funzionamento
dell'interfaccia. Le righe hanno ``source = "SINTETICO"`` / ``"demo"`` e la UI mostra un'etichetta permanente.
Questo modulo non viene mai usato per riempire la modalita' live.
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np

from ..models import Market, OddsQuote, PlayerGameLog, Side, utc_now
from ..model.empirical import stat_value
from ..model.synthetic import SOURCE, TruePlayer, generate_player
from .bundle import DataBundle, EventInfo, PlayerInfo, ProviderStatus
from .labels import normalize_name

TEAMS = [
    ("DMA", "Demo Atlas"), ("DMB", "Demo Borea"), ("DMC", "Demo Cirrus"), ("DMD", "Demo Delta"),
    ("DME", "Demo Eclipse"), ("DMF", "Demo Fenice"), ("DMG", "Demo Galassia"), ("DMH", "Demo Horizon"),
]
N_ROUNDS = 62
DEMO_BOOKS = ["demo_a", "demo_b", "demo_c", "demo_d", "demo_e"]
MARGIN = 0.06
DEMO_NOTICE = (
    "MODALITÀ DEMO — tutti i dati (squadre, giocatori, statistiche, quote) sono SINTETICI e inventati. "
    "Non rappresentano la NBA reale e non vanno usati per decisioni."
)


def _pairings(n_teams: int, round_idx: int) -> list[tuple[int, int]]:
    """Metodo del cerchio: abbinamenti (casa, trasferta) per il turno."""
    teams = list(range(n_teams))
    rest = teams[1:]
    shift = round_idx % len(rest)
    rest = rest[shift:] + rest[:shift]
    order = [teams[0]] + rest
    pairs = []
    for i in range(n_teams // 2):
        a, b = order[i], order[n_teams - 1 - i]
        pairs.append((a, b) if (round_idx + i) % 2 == 0 else (b, a))
    return pairs


def _with_shooting(l: PlayerGameLog, rng: np.random.Generator) -> PlayerGameLog:
    """Completa FGM/FGA/FTM/FTA in modo coerente con i punti (sintetici), solo per la visualizzazione."""
    ftm = int(min(l.pts // 3, rng.binomial(max(l.pts, 0), 0.14)))
    fta = int(ftm + rng.binomial(max(ftm, 0) // 2 + 1, 0.4))
    fg2m = max(0, (l.pts - 3 * l.fg3m - ftm) // 2)
    fga = l.fg3a + int(round(fg2m / 0.52))
    return dataclasses.replace(l, ftm=ftm, fta=max(fta, ftm), fgm=l.fg3m + fg2m, fga=max(fga, l.fg3m + fg2m))


def build_demo_bundle(seed: int = 2026, now: datetime | None = None) -> DataBundle:
    now = now or utc_now()
    rng = np.random.default_rng(seed)
    today = now.date()
    season = today.year if today.month >= 10 else today.year - 1

    # ---- calendario storico condiviso: ogni turno e' una data, 4 partite ----------------------
    dates = [today - timedelta(days=2 * (N_ROUNDS - r)) for r in range(N_ROUNDS)]
    sched: dict[int, list[dict]] = {t: [] for t in range(len(TEAMS))}
    for r in range(N_ROUNDS):
        for home, away in _pairings(len(TEAMS), r):
            gid = f"DEMO-{r:03d}-{TEAMS[home][0]}-{TEAMS[away][0]}"
            sched[home].append({"gid": gid, "date": dates[r], "opp": TEAMS[away][0], "home": True})
            sched[away].append({"gid": gid, "date": dates[r], "opp": TEAMS[home][0], "home": False})

    # ---- giocatori ---------------------------------------------------------------------------
    def gam(mean: float, cv: float) -> float:
        shape = 1.0 / cv**2
        return float(rng.gamma(shape, mean / shape))

    specs: list[tuple[int, str, TruePlayer, int]] = []  # (squadra, nome, verita', n_partite)
    pid = 0
    for t in range(len(TEAMS)):
        for _ in range(3):
            pid += 1
            truth = TruePlayer(
                player_id=pid, name=f"Giocatore Demo {pid:02d}", minutes_mean=float(rng.uniform(22, 36)),
                rho={"pts": gam(0.45, 0.3), "reb": gam(0.17, 0.5), "ast": gam(0.12, 0.6), "fg3a": gam(0.12, 0.5)},
                p3=float(rng.beta(0.36 * 300, 0.64 * 300)),
            )
            specs.append((t, truth.name, truth, N_ROUNDS))
    base = {"pts": 0.40, "reb": 0.15, "ast": 0.10}
    specials = [
        ("Giocatore Demo Matricola", TruePlayer(player_id=101, minutes_mean=14, rho={**base, "fg3a": 0.10}), 8),
        ("Giocatore Demo Panchina", TruePlayer(player_id=102, minutes_mean=17, rho={**base, "fg3a": 0.012}), 18),
        ("Giocatore Demo Rientro", TruePlayer(player_id=103, minutes_mean=26, rho={"pts": 0.5, "reb": 0.18, "ast": 0.13, "fg3a": 0.13}), 14),
    ]
    for i, (name, truth, n) in enumerate(specials):
        specs.append((i, name, dataclasses.replace(truth, name=name), n))

    logs: dict[str, list[PlayerGameLog]] = {}
    players: dict[str, PlayerInfo] = {}
    team_of: dict[str, int] = {}
    for team_idx, name, truth, n_games in specs:
        raw = generate_player(truth, n_games, seed=int(rng.integers(1, 2**31 - 1)), season=season)
        games = sched[team_idx][-n_games:]
        rows = [
            _with_shooting(dataclasses.replace(l, game_id=g["gid"], game_date=g["date"], team=TEAMS[team_idx][0],
                                               opponent=g["opp"], is_home=g["home"], player_name=name), rng)
            for l, g in zip(raw, games)
        ]
        key = normalize_name(name)
        logs[key] = rows
        players[key] = PlayerInfo(key, name, TEAMS[team_idx][0])
        team_of[key] = team_idx

    # ---- partite in programma (domani) e quote sintetiche -------------------------------------
    tomorrow = now + timedelta(days=1)
    events: dict[str, EventInfo] = {}
    pairs = _pairings(len(TEAMS), N_ROUNDS)
    for i, (home, away) in enumerate(pairs):
        eid = f"DEMO-EVT-{i + 1}"
        start = tomorrow.replace(hour=0, minute=30, second=0, microsecond=0) + timedelta(hours=i)
        events[eid] = EventInfo(eid, TEAMS[home][0], TEAMS[away][0], start.astimezone(timezone.utc), "demo")

    quotes: list[OddsQuote] = []
    stale_done = False
    for eid, ev in events.items():
        team_codes = {ev.home, ev.away}
        for key, info in players.items():
            if info.team not in team_codes:
                continue
            for market in Market:
                vals = np.array([stat_value(l, market) for l in logs[key]], dtype=float)
                if len(vals) < 5:
                    continue
                med = float(np.median(vals[-30:]))
                if market is Market.THREES and vals.sum() < 8:
                    continue  # nessun libro quota le triple di chi non tira da tre
                line = float(round(med)) if rng.random() < 0.25 else float(np.floor(med) + 0.5)
                line = max(line, 0.5)
                p_over = ((vals > line).sum() + 0.5 * (vals == line).sum()) / len(vals)
                books = [b for b in DEMO_BOOKS if rng.random() < 0.85]
                if len(books) < 2:
                    books = DEMO_BOOKS[:2]
                for book in books:
                    pb = float(np.clip(p_over + rng.normal(0, 0.035), 0.05, 0.95))
                    age = float(rng.uniform(1, 20))
                    if not stale_done:
                        age, stale_done = 190.0, True  # una quota volutamente obsoleta, per mostrare l'avviso
                    for side, p in ((Side.OVER, pb), (Side.UNDER, 1 - pb)):
                        odds = round(1.0 / (p + MARGIN / 2), 2)
                        quotes.append(OddsQuote(
                            provider="demo", bookmaker=book, event_id=eid, player_name=info.name, player_id=None,
                            market=market, line=line, side=side, odds_decimal=max(odds, 1.01), fetched_at=now,
                            quote_time=now - timedelta(minutes=age), home_team=ev.home, away_team=ev.away,
                            commence_time=ev.commence_time, is_alternate=False, source="demo",
                        ))

    fingerprint = hashlib.sha1(f"demo:{seed}:{today}".encode()).hexdigest()[:12]
    return DataBundle(
        mode="demo", source_label="Dati SINTETICI (demo)", status="ok", updated_at=now, quotes=quotes, logs=logs,
        players=players, events=events,
        providers=[ProviderStatus("Generatore demo", "ok", "dati sintetici, nessuna chiamata di rete", len(quotes))],
        messages=[("warning", DEMO_NOTICE)], fingerprint=fingerprint,
    )
