"""Archivio SQLite di ogni backtest: dati usati, previsioni, quote, valutazioni e scommesse, per poterli verificare.

Per ogni previsione si salvano: la pmf completa, le partite passate al modello (id), il blocco di prior usato
(con la sua data massima), la configurazione del modello e l'esito. ``verify_run`` ricontrolla l'assenza di
leakage e RICALCOLA un campione di previsioni dai soli dati salvati. Il file sta in data/cache/ (escluso da Git).
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..model.empirical import stat_value
from ..model.engine import GameContext, ValueEngine
from ..model.odds_math import split_probs
from ..models import Market, PlayerGameLog
from ..ui.snapshot import cache_dir, log_from_dict, log_to_dict
from .records import config_from_dict, history_window, prior_from_dict
from .runner import BacktestConfig, BacktestResult, candidate_configs
from .splits import TemporalSplit

SCHEMA = 1

DDL = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_at TEXT, label TEXT, source TEXT, data_fingerprint TEXT, config_hash TEXT,
  config_json TEXT, split_json TEXT, selection_json TEXT, verdict_json TEXT, threshold REAL, calibrator_json TEXT, quote_info_json TEXT,
  notes_json TEXT, schema_version INTEGER);
CREATE TABLE IF NOT EXISTS game_logs (run_id TEXT, player TEXT, game_id TEXT, game_date TEXT, season INTEGER, data_json TEXT);
CREATE INDEX IF NOT EXISTS ix_gl ON game_logs(run_id, player, game_id);
CREATE TABLE IF NOT EXISTS priors (run_id TEXT, candidate TEXT, block_start TEXT, n_partite INTEGER, ultima_data TEXT, prior_json TEXT);
CREATE TABLE IF NOT EXISTS inputs (run_id TEXT, player TEXT, game_id TEXT, history_ids TEXT, history_n INTEGER, history_last_date TEXT);
CREATE INDEX IF NOT EXISTS ix_in ON inputs(run_id, player, game_id);
CREATE TABLE IF NOT EXISTS predictions (run_id TEXT, candidate TEXT, phase TEXT, player TEXT, game_id TEXT, game_date TEXT, market TEXT,
  pmf BLOB, support INTEGER, mean REAL, y REAL, prior_block_start TEXT, b2b INTEGER, flags TEXT);
CREATE INDEX IF NOT EXISTS ix_pr ON predictions(run_id, candidate, phase);
CREATE TABLE IF NOT EXISTS quotes_used (run_id TEXT, player TEXT, game_id TEXT, event_id TEXT, market TEXT, line REAL, side TEXT, book TEXT,
  odds REAL, fetched_at TEXT, quote_time TEXT, commence_time TEXT, novig_p_over REAL, overround REAL);
"""


def default_path() -> Path:
    return cache_dir() / "backtest.sqlite"


def pack_pmf(pmf: np.ndarray) -> tuple[bytes, int]:
    nz = np.flatnonzero(np.asarray(pmf) > 1e-14)
    keep = int(nz[-1]) + 1 if len(nz) else 1
    return zlib.compress(np.asarray(pmf[:keep], dtype="<f8").tobytes(), 6), len(pmf)


def unpack_pmf(blob: bytes, support: int) -> np.ndarray:
    a = np.frombuffer(zlib.decompress(blob), dtype="<f8")
    out = np.zeros(support)
    out[: len(a)] = a
    return out


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return str(o)


def _dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


def _s(v) -> str | None:
    return None if v is None else (v.isoformat() if hasattr(v, "isoformat") else str(v))


class BacktestStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c:
            c.executescript(DDL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    # ------------------------------------------------------------------ scrittura
    def save(self, res: BacktestResult, matched_quotes=None) -> None:
        rid = res.run_id
        matched_quotes = res.matched if matched_quotes is None else matched_quotes
        with closing(self._conn()) as c:
            for t in ("runs", "game_logs", "priors", "inputs", "predictions", "quotes_used", "evals", "bets", "pmf_scores"):
                try:
                    c.execute(f"DELETE FROM {t} WHERE run_id = ?", (rid,))
                except sqlite3.OperationalError:
                    pass  # tabella non ancora creata da pandas
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                rid, datetime.now(timezone.utc).isoformat(), res.label, res.source, res.data_fingerprint, res.cfg.hash(),
                json.dumps(asdict(res.cfg)), json.dumps(res.split.to_dict()), _dumps(res.selection),
                _dumps(res.verdict), res.threshold,
                json.dumps({"a": res.recalibrator.a, "b": res.recalibrator.b, "n": res.recalibrator.n_fit}),
                _dumps(res.quote_info), json.dumps(res.notes), SCHEMA))
            c.executemany("INSERT INTO game_logs VALUES (?,?,?,?,?,?)", [
                (rid, k, l.game_id, l.game_date.isoformat(), l.season, json.dumps(log_to_dict(l))) for k, v in res.pool.items() for l in v])
            c.executemany("INSERT INTO priors VALUES (?,?,?,?,?,?)", [
                (rid, cand, d.isoformat(), v["n_partite"], v["ultima_data"], json.dumps(v["prior"])) for (cand, d), v in res.priors.items()])
            seen: set = set()
            inputs, preds = [], []
            for r in res.records:
                if (r.player, r.log.game_id) not in seen:
                    seen.add((r.player, r.log.game_id))
                    inputs.append((rid, r.player, r.log.game_id, json.dumps([l.game_id for l in r.history]), len(r.history), r.hist_last.isoformat()))
                for m, pmf in r.pmfs.items():
                    blob, support = pack_pmf(pmf)
                    preds.append((rid, r.candidate, r.phase, r.player, r.log.game_id, r.log.game_date.isoformat(), m.value, blob, support,
                                  float(pmf @ np.arange(len(pmf))), float(stat_value(r.log, m)), r.block_start.isoformat(),
                                  None if r.b2b is None else int(r.b2b), ",".join(r.flags)))
            c.executemany("INSERT INTO inputs VALUES (?,?,?,?,?,?)", inputs)
            c.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", preds)
            c.executemany("INSERT INTO quotes_used VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (rid, k, l.game_id, s.quote.event_id, s.quote.market.value, s.quote.line, s.quote.side.value, s.quote.bookmaker,
                 s.quote.odds_decimal, _s(s.capture_time), _s(s.quote.quote_time), _s(s.quote.commence_time), s.novig_p_over, s.overround)
                for s, k, l in matched_quotes])
            for name, df in (("evals", res.evals), ("bets", res.bets), ("pmf_scores", res.pmf_scores)):
                d = df.copy()
                d.insert(0, "run_id", rid)
                if "date" in d:
                    d["date"] = d["date"].astype(str)
                if name == "evals":  # orari ripetuti solo sulle righe del modello (le altre hanno lo stesso obs_id): archivio piu' leggero
                    d.loc[d["model"] != "modello", ["capture_time", "commence_time"]] = ""
                d.to_sql(name, c, if_exists="append", index=False)
            c.commit()

    # ------------------------------------------------------------------ lettura
    def runs(self) -> pd.DataFrame:
        with closing(self._conn()) as c:
            return pd.read_sql("SELECT run_id, created_at, label, source, data_fingerprint, config_hash, threshold FROM runs ORDER BY created_at DESC", c)

    def has_run(self, run_id: str) -> bool:
        with closing(self._conn()) as c:
            return c.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is not None

    def meta(self, run_id: str) -> dict:
        with closing(self._conn()) as c:
            row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            cols = [d[0] for d in c.execute("SELECT * FROM runs LIMIT 0").description]
        if row is None:
            raise KeyError(run_id)
        m = dict(zip(cols, row))
        for k in ("config_json", "split_json", "selection_json", "verdict_json", "calibrator_json", "quote_info_json", "notes_json"):
            m[k[:-5]] = json.loads(m.pop(k))
        return m

    def frame(self, table: str, run_id: str, where: str = "", params: tuple = ()) -> pd.DataFrame:
        with closing(self._conn()) as c:
            return pd.read_sql(f"SELECT * FROM {table} WHERE run_id = ? {where}", c, params=(run_id, *params))

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with closing(self._conn()) as c:
            return pd.read_sql(sql, c, params=params)

    def n_test_configs(self, data_fingerprint: str, exclude_run: str | None = None) -> int:
        """Quante configurazioni distinte sono gia' state valutate sul test di questi stessi dati (rischio di sovradattamento al test)."""
        with closing(self._conn()) as c:
            return int(c.execute("SELECT COUNT(DISTINCT config_hash) FROM runs WHERE data_fingerprint = ? AND run_id != ?",
                                 (data_fingerprint, exclude_run or "")).fetchone()[0])

    def load(self, run_id: str) -> "LoadedRun":
        meta = self.meta(run_id)
        chosen = f"modello/{meta['selection']['scelto']}"
        ev, pmf = self.frame("evals", run_id), self.frame("pmf_scores", run_id)
        for df in (ev, pmf):  # il candidato scelto si chiama «modello» anche in validazione
            df["model"] = df["model"].replace({chosen: "modello"})
        return LoadedRun(meta=meta, split=TemporalSplit.from_dict(meta["split"]), evals=ev, bets=self.frame("bets", run_id), pmf=pmf)

    def delete(self, run_id: str) -> None:
        with closing(self._conn()) as c:
            for t in ("runs", "game_logs", "priors", "inputs", "predictions", "quotes_used", "evals", "bets", "pmf_scores"):
                c.execute(f"DELETE FROM {t} WHERE run_id = ?", (run_id,))
            c.commit()

    # ------------------------------------------------------------------ verifica
    def verify(self, run_id: str, sample: int = 30, seed: int = 0) -> list[tuple[str, bool, str]]:
        return verify_run(self, run_id, sample, seed)


@dataclass
class LoadedRun:
    meta: dict
    split: TemporalSplit
    evals: pd.DataFrame
    bets: pd.DataFrame
    pmf: pd.DataFrame

    @property
    def run_id(self) -> str:
        return self.meta["run_id"]

    @property
    def cfg(self) -> dict:
        return self.meta["config"]

    @property
    def verdict(self) -> dict:
        return self.meta["verdict"]


def verify_run(store: BacktestStore, run_id: str, sample: int = 30, seed: int = 0) -> list[tuple[str, bool, str]]:
    """Controlli indipendenti sul contenuto salvato. Restituisce (controllo, esito, dettaglio)."""
    out: list[tuple[str, bool, str]] = []
    meta = store.meta(run_id)
    split = TemporalSplit.from_dict(meta["split"])
    cfg = BacktestConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
    preds = store.frame("predictions", run_id)
    inputs = store.frame("inputs", run_id)
    logs = store.frame("game_logs", run_id)
    priors = store.frame("priors", run_id)
    evals = store.frame("evals", run_id)

    # 1. fasi coerenti con le date di confine
    ph_ok = True
    for ph, g in preds.groupby("phase"):
        d = pd.to_datetime(g["game_date"]).dt.date
        ph_ok &= all(split.phase(x) == ph for x in d.unique())
    out.append(("Le previsioni stanno nella fase corretta (confini cronologici)", ph_ok, f"{len(preds)} previsioni"))

    # 2. lo storico passato al modello e' interamente anteriore alla partita
    date_of = {(r.player, r.game_id): r.game_date for r in logs.itertuples()}
    bad = 0
    gdate = {(r.player, r.game_id): r.game_date for r in preds.drop_duplicates(["player", "game_id"]).itertuples()}
    for r in inputs.itertuples():
        gd = gdate.get((r.player, r.game_id))
        if gd is None:
            continue
        ids = json.loads(r.history_ids)
        if any(date_of.get((r.player, i), "9999") >= gd for i in ids) or r.history_last_date >= gd:
            bad += 1
    out.append(("Nessuna previsione usa partite non anteriori (storico)", bad == 0, f"{len(inputs)} input controllati, {bad} violazioni"))

    # 3. prior stimato solo su partite anteriori al blocco; il blocco non e' successivo alla partita
    pb = priors.assign(ok=lambda d: d["ultima_data"].isna() | (d["ultima_data"] < d["block_start"]))
    prior_ok = bool(pb["ok"].all()) and bool((preds["prior_block_start"] <= preds["game_date"]).all())
    out.append(("Il prior di popolazione usa solo partite anteriori al suo blocco", prior_ok, f"{len(priors)} blocchi"))

    # 4. quote: catturate prima dell'inizio
    q = evals[(evals["line_source"] == "quota_reale") & (evals["capture_time"] != "")]
    qbad = 0
    if len(q):
        cap = pd.to_datetime(q["capture_time"], utc=True)
        com = pd.to_datetime(q["commence_time"], utc=True)
        qbad = int((cap >= com).sum())
    out.append(("Tutte le quote usate sono state catturate prima dell'inizio", qbad == 0, f"{len(q)} righe con quota, {qbad} violazioni"))

    # 5. il test e' valutato con un solo candidato (quello scelto in validazione)
    test_c = set(preds[preds["phase"] == "test"]["candidate"])
    out.append(("Il test usa solo il candidato scelto in validazione", test_c <= {meta["selection"]["scelto"]}, f"candidati nel test: {sorted(test_c)}"))

    # 6. esiti coerenti con i dati salvati
    stat = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "fg3m"}
    parsed = {(r.player, r.game_id): json.loads(r.data_json) for r in logs.itertuples()}
    mis = 0
    for r in preds.sample(min(len(preds), 400), random_state=seed).itertuples():
        d = parsed[(r.player, r.game_id)]
        y = d["pts"] + d["reb"] + d["ast"] if r.market == "pra" else d[stat[r.market]]
        mis += int(abs(y - r.y) > 1e-9)
    out.append(("Gli esiti salvati coincidono con i box score salvati", mis == 0, f"campione di {min(len(preds), 400)}, {mis} differenze"))

    # 7. ricalcolo di un campione di previsioni dai soli dati salvati
    cand_cfg = candidate_configs(cfg)
    pri = {(r.candidate, r.block_start): prior_from_dict(json.loads(r.prior_json)) for r in priors.itertuples()}
    pool: dict[str, list[PlayerGameLog]] = {}
    for r in logs.itertuples():
        pool.setdefault(r.player, []).append(log_from_dict(json.loads(r.data_json)))
    by_id = {(k, l.game_id): l for k, v in pool.items() for l in v}
    in_ids = {(r.player, r.game_id): json.loads(r.history_ids) for r in inputs.itertuples()}
    keys = preds.drop_duplicates(["candidate", "player", "game_id"])
    keys = keys.sample(min(sample, len(keys)), random_state=seed)
    worst = 0.0
    for r in keys.itertuples():
        history = [by_id[(r.player, i)] for i in in_ids[(r.player, r.game_id)]]
        engine = ValueEngine(cand_cfg[r.candidate], pri[(r.candidate, r.prior_block_start)])
        pred = engine.predict(history, GameContext(back_to_back=None if pd.isna(r.b2b) else bool(r.b2b)), intervals=False)
        for row in preds[(preds["candidate"] == r.candidate) & (preds["player"] == r.player) & (preds["game_id"] == r.game_id)].itertuples():
            stored = unpack_pmf(row.pmf, row.support)
            worst = max(worst, float(np.abs(stored - pred.markets[Market(row.market)].pmf).max()))
    out.append(("Ricalcolo di un campione di previsioni dai dati salvati: pmf identica", worst < 1e-9, f"{len(keys)} previsioni ricalcolate, scarto massimo {worst:.1e}"))

    # 8. probabilita' delle linee ricalcolabili dalla pmf salvata
    syn = evals[(evals["line_source"] == "sintetica") & (evals["model"].isin(["modello"] + [f"modello/{c}" for c in cand_cfg]))]
    worst_p, n_chk = 0.0, 0
    if len(syn):
        pm = {(r.candidate, r.player, r.game_id, r.market): r for r in preds.itertuples()}
        for r in syn.sample(min(len(syn), 300), random_state=seed).itertuples():
            cand = r.model.split("/")[1] if "/" in r.model else meta["selection"]["scelto"]
            row = pm.get((cand, r.player, r.game_id, r.market))
            if row is None:
                continue
            _, _, pg = split_probs(unpack_pmf(row.pmf, row.support), r.line)
            worst_p = max(worst_p, abs(float(pg) - r.p_win))
            n_chk += 1
    out.append(("Le probabilità delle linee coincidono con la pmf salvata", worst_p < 1e-9, f"{n_chk} righe controllate, scarto massimo {worst_p:.1e}"))
    return out
