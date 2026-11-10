"""实验记录库：自动优化过程中的每一次试验都完整留痕（SQLite）。"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from ..utils import NumpyEncoder

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, stage TEXT, study TEXT, trial INTEGER, family TEXT, tasks TEXT,
    status TEXT, value REAL, params TEXT, metrics TEXT, duration REAL, created INTEGER
);
CREATE INDEX IF NOT EXISTS idx_trials_run ON trials(run_id, stage);
"""


class ExperimentDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=60)
        c.row_factory = sqlite3.Row
        return c

    def log(self, run_id: str, stage: str, study: str, trial: int, family: str, tasks, status: str, value,
            params: dict, metrics: dict, duration: float) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO trials(run_id,stage,study,trial,family,tasks,status,value,params,metrics,duration,created)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (run_id, stage, study, trial, family, json.dumps(tasks), status,
                       None if value is None else float(value), json.dumps(params, cls=NumpyEncoder),
                       json.dumps(metrics, cls=NumpyEncoder), float(duration), int(time.time() * 1000)))

    def query(self, run_id: str | None = None, stage: str | None = None, limit: int = 2000) -> list[dict]:
        q, args = "SELECT * FROM trials WHERE 1=1", []
        if run_id:
            q += " AND run_id=?"; args.append(run_id)
        if stage:
            q += " AND stage=?"; args.append(stage)
        q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q, args)]
        for r in rows:
            for k in ("tasks", "params", "metrics"):
                r[k] = json.loads(r[k]) if r[k] else None
        return rows

    def count(self, run_id: str) -> dict:
        with self._conn() as c:
            return {r["stage"] + ":" + r["status"]: r["n"] for r in
                    c.execute("SELECT stage, status, COUNT(*) n FROM trials WHERE run_id=? GROUP BY stage, status", (run_id,))}
