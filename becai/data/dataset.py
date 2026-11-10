"""把各数据源对齐成按"决策时刻"索引的 1 分钟面板。

时间约定（全系统统一，防泄漏的基础）：
  * 一根 open_time=T 的 1 分钟K线覆盖 [T, T+60s)，在 T+60s 收盘。
  * 面板索引 ts = T + 60s = 决策时刻。ts 行只包含 ts 时刻已经收盘、已经发布的数据。
  * 离线训练与实时预测都走 assemble_panel 这同一个函数，保证口径一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import MINUTE_MS, date_to_ms
from .store import load_raw

PREFIX = {"index": "ix", "spot": "sp", "futures": "fu", "mark": "mk", "premium": "pm"}
SHORT = {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v", "quote_volume": "qv",
         "count": "n", "taker_buy_volume": "tbv", "taker_buy_quote_volume": "tbqv"}
METRIC_FIELDS = ["oi", "oi_value", "ls_top_acct", "ls_top_pos", "ls_acct", "taker_ls"]


def _asof(grid: np.ndarray, times: np.ndarray, values: np.ndarray, max_age_ms: int | None) -> np.ndarray:
    """对每个 grid 时刻取 times<=grid 的最后一个值；超过 max_age 视为缺失。"""
    out = np.full(len(grid), np.nan)
    if len(times) == 0:
        return out
    pos = np.searchsorted(times, grid, side="right") - 1
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    if max_age_ms is not None:
        age = grid - times[np.clip(pos, 0, None)]
        out[ok & (age > max_age_ms)] = np.nan
    return out


def assemble_panel(frames: dict[str, pd.DataFrame], start_ms: int, end_ms: int, metrics_lag_minutes: int = 6) -> pd.DataFrame:
    """frames: {dataset: 规范化原始DataFrame}。返回以 ts(决策时刻, ms) 为索引的面板。

    start_ms / end_ms 是决策时刻范围 [start_ms, end_ms]（含端点，对齐到整分钟）。
    """
    grid = np.arange(start_ms, end_ms + 1, MINUTE_MS, dtype=np.int64)
    panel = pd.DataFrame(index=pd.Index(grid, name="ts"))
    for name, pre in PREFIX.items():
        df = frames.get(name)
        if df is None or df.empty:
            continue
        ts = df["open_time"].to_numpy(dtype=np.int64) + MINUTE_MS      # 收盘时刻
        sub = df.drop(columns=["open_time"]).copy()
        sub.index = ts
        sub = sub[~sub.index.duplicated(keep="last")].reindex(grid)
        cols = ["close"] if name == "mark" else list(sub.columns)
        for c in cols:
            panel[f"{pre}_{SHORT[c]}"] = sub[c].to_numpy(dtype=np.float64)
    fr = frames.get("funding")
    if fr is not None and not fr.empty:
        t = fr["time"].to_numpy(dtype=np.int64)
        panel["fr"] = _asof(grid, t, fr["rate"].to_numpy(dtype=np.float64), None)
        last_t = _asof(grid, t, t.astype(np.float64), None)
        panel["fr_age"] = (grid - last_t) / MINUTE_MS
    mt = frames.get("metrics")
    if mt is not None and not mt.empty:
        # 指标在 create_time 之后才发布：加上发布延迟保护后才允许使用
        t = mt["time"].to_numpy(dtype=np.int64) + metrics_lag_minutes * MINUTE_MS
        for f in METRIC_FIELDS:
            if f in mt:
                panel[f"mt_{f}"] = _asof(grid, t, mt[f].to_numpy(dtype=np.float64), 30 * MINUTE_MS)
    return panel


def load_panel(cfg, symbol: str, start: str | int | None = None, end: str | int | None = None) -> pd.DataFrame:
    """从本地存储构建某币种的面板。"""
    s = date_to_ms(start or cfg.data.start) if not isinstance(start, (int, np.integer)) else int(start)
    frames = {}
    for name, on in cfg.data.datasets.items():
        if on:
            frames[name] = load_raw(cfg, name, symbol, s - MINUTE_MS, None)
    idx = frames.get("index")
    if idx is None or idx.empty:
        raise RuntimeError(f"{symbol} 缺少指数价格数据，请先运行数据同步")
    last = int(idx["open_time"].max()) + MINUTE_MS
    if end is not None:
        e = date_to_ms(end) if not isinstance(end, (int, np.integer)) else int(end)
        last = min(last, e)
    return assemble_panel(frames, s, last, int(cfg.data.metrics_lag_minutes))


def load_panels(cfg, start=None, end=None) -> dict[str, pd.DataFrame]:
    """所有币种的面板，裁剪到共同时间网格（同一行号 = 同一时刻）。"""
    panels = {s: load_panel(cfg, s, start, end) for s in cfg.symbols}
    lo = max(p.index[0] for p in panels.values())
    hi = min(p.index[-1] for p in panels.values())
    return {s: p.loc[lo:hi] for s, p in panels.items()}
