"""实时预测记录库（SQLite）：保存每一轮、每一项预测（包括未开仓的），以及赔率与真实订单。"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL, symbol TEXT NOT NULL, horizon INTEGER NOT NULL, version TEXT,
    p_up REAL, p_down REAL, p_flat REAL, confidence REAL,
    signal TEXT,              -- 模型+策略层面的方向（风控之前）
    decision TEXT,            -- 最终输出：HIGHER / LOWER / NO_TRADE
    executable INTEGER DEFAULT 0,   -- 是否为可执行开仓提示（预测质量+赔率+期望+风控全部通过）
    skip_reason TEXT, risk_reasons TEXT,
    payout REAL, ev REAL, stake REAL,
    entry_price REAL, expiry_ts INTEGER, settle_price REAL,
    actual TEXT,              -- 到期实际方向 UP / DOWN / TIE
    correct INTEGER,          -- 方向预测是否正确（p_up>0.5 视为看涨），平局为 NULL
    pnl REAL,                 -- 可执行信号按固定/录入赔率模拟的盈亏
    latency_ms REAL, created INTEGER,
    UNIQUE(ts, symbol, horizon)
);
CREATE INDEX IF NOT EXISTS idx_pred_expiry ON predictions(expiry_ts, actual);
CREATE TABLE IF NOT EXISTS odds (
    symbol TEXT, horizon INTEGER, direction TEXT, payout REAL, updated INTEGER,
    PRIMARY KEY(symbol, horizon, direction)
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER, ts INTEGER, symbol TEXT, horizon INTEGER, direction TEXT, stake REAL, payout REAL,
    entry_price REAL, expiry_ts INTEGER, result TEXT, pnl REAL, result_source TEXT, note TEXT, created INTEGER
);
CREATE TABLE IF NOT EXISTS heartbeat (k TEXT PRIMARY KEY, v TEXT, updated INTEGER);
"""


class LiveStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ---- 预测 ----
    def insert_predictions(self, rows: list[dict]) -> None:
        cols = ["ts", "symbol", "horizon", "version", "p_up", "p_down", "p_flat", "confidence", "signal", "decision", "executable",
                "skip_reason", "risk_reasons", "payout", "ev", "stake", "entry_price", "expiry_ts", "latency_ms", "created"]
        with self.conn() as c:
            c.executemany(f"INSERT OR REPLACE INTO predictions({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                          [[r.get(k) for k in cols] for r in rows])

    def unsettled(self, now_ts: int) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM predictions WHERE actual IS NULL AND expiry_ts<=? ORDER BY expiry_ts", (now_ts,))]

    def settle(self, pred_id: int, settle_price: float, actual: str, correct: int | None, pnl: float | None) -> None:
        with self.conn() as c:
            c.execute("UPDATE predictions SET settle_price=?, actual=?, correct=?, pnl=? WHERE id=?", (settle_price, actual, correct, pnl, pred_id))

    def expire_stale(self, before_ts: int) -> None:
        """超过可结算窗口仍无价格的记录标记为 UNKNOWN，避免无限堆积。"""
        with self.conn() as c:
            c.execute("UPDATE predictions SET actual='UNKNOWN' WHERE actual IS NULL AND expiry_ts<?", (before_ts,))

    def latest(self) -> list[dict]:
        with self.conn() as c:
            r = c.execute("SELECT MAX(ts) m FROM predictions").fetchone()
            if not r or r["m"] is None:
                return []
            return [dict(x) for x in c.execute("SELECT * FROM predictions WHERE ts=? ORDER BY symbol, horizon", (r["m"],))]

    def history(self, limit: int = 200, offset: int = 0, symbol: str | None = None, horizon: int | None = None,
                only_signals: bool = False) -> list[dict]:
        q, a = "SELECT * FROM predictions WHERE 1=1", []
        if symbol:
            q += " AND symbol=?"; a.append(symbol)
        if horizon:
            q += " AND horizon=?"; a.append(horizon)
        if only_signals:
            q += " AND decision!='NO_TRADE'"
        q += " ORDER BY ts DESC, symbol, horizon LIMIT ? OFFSET ?"; a += [limit, offset]
        with self.conn() as c:
            return [dict(r) for r in c.execute(q, a)]

    def open_executable(self, now_ts: int) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM predictions WHERE executable=1 AND actual IS NULL AND expiry_ts>? ORDER BY ts", (now_ts,))]

    def stats(self, since_ts: int = 0) -> list[dict]:
        """分任务统计：区分方向准确率（全部预测）与有效信号胜率（仅开仓信号）。"""
        q = """SELECT symbol, horizon,
                 COUNT(*) n_pred,
                 SUM(CASE WHEN actual IN ('UP','DOWN') THEN 1 ELSE 0 END) n_settled,
                 SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) n_correct,
                 SUM(CASE WHEN decision!='NO_TRADE' THEN 1 ELSE 0 END) n_signal,
                 SUM(CASE WHEN decision!='NO_TRADE' AND actual IN ('UP','DOWN') THEN 1 ELSE 0 END) n_signal_settled,
                 SUM(CASE WHEN decision!='NO_TRADE' AND correct=1 THEN 1 ELSE 0 END) n_signal_win,
                 SUM(CASE WHEN executable=1 THEN 1 ELSE 0 END) n_exec,
                 SUM(CASE WHEN executable=1 AND pnl>0 THEN 1 ELSE 0 END) n_exec_win,
                 SUM(CASE WHEN executable=1 AND pnl<0 THEN 1 ELSE 0 END) n_exec_loss,
                 SUM(CASE WHEN executable=1 THEN COALESCE(pnl,0) ELSE 0 END) sim_pnl
               FROM predictions WHERE ts>=? GROUP BY symbol, horizon ORDER BY symbol, horizon"""
        with self.conn() as c:
            return [dict(r) for r in c.execute(q, (since_ts,))]

    def skip_reason_counts(self, since_ts: int = 0) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT COALESCE(NULLIF(risk_reasons,''), skip_reason) reason, COUNT(*) n FROM predictions "
                "WHERE ts>=? AND executable=0 GROUP BY 1 ORDER BY n DESC LIMIT 30", (since_ts,))]

    def daily_pnl(self, days: int = 30) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT expiry_ts/86400000 day, SUM(pnl) pnl, COUNT(*) n FROM predictions WHERE executable=1 AND pnl IS NOT NULL "
                "GROUP BY 1 ORDER BY 1 DESC LIMIT ?", (days,))]

    # ---- 赔率（手工录入模式）----
    def set_odds(self, symbol: str, horizon: int, direction: str, payout: float) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO odds VALUES(?,?,?,?,?)", (symbol, horizon, direction, payout, int(time.time() * 1000)))

    def get_odds(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM odds ORDER BY symbol, horizon, direction")]

    # ---- 真实订单（用户确认）----
    def add_order(self, o: dict) -> int:
        cols = ["prediction_id", "ts", "symbol", "horizon", "direction", "stake", "payout", "entry_price", "expiry_ts", "note", "created"]
        o = dict(o, created=int(time.time() * 1000))
        with self.conn() as c:
            return int(c.execute(f"INSERT INTO orders({','.join(cols)}) VALUES({','.join('?' * len(cols))})", [o.get(k) for k in cols]).lastrowid)

    def set_order_result(self, order_id: int, result: str, pnl: float, source: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE orders SET result=?, pnl=?, result_source=? WHERE id=?", (result, pnl, source, order_id))

    def orders(self, limit: int = 200, open_only: bool = False) -> list[dict]:
        q = "SELECT * FROM orders" + (" WHERE result IS NULL" if open_only else "") + " ORDER BY id DESC LIMIT ?"
        with self.conn() as c:
            return [dict(r) for r in c.execute(q, (limit,))]

    def order_stats(self) -> dict:
        with self.conn() as c:
            r = c.execute("SELECT COUNT(*) n, SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) win, SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) loss,"
                          " SUM(COALESCE(pnl,0)) pnl, SUM(stake) staked,"
                          " SUM(CASE WHEN result_source='user' THEN 1 ELSE 0 END) n_user_confirmed FROM orders").fetchone()
            return dict(r)

    # ---- 心跳 ----
    def beat(self, key: str, value: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO heartbeat VALUES(?,?,?)", (key, json.dumps(value, ensure_ascii=False), int(time.time() * 1000)))

    def heartbeat(self, key: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM heartbeat WHERE k=?", (key,)).fetchone()
        return None if not r else {"value": json.loads(r["v"]), "updated": r["updated"]}
