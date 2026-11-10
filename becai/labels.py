"""事件合约标签（指数价格）与"分钟级标签近似误差"评估。

币安规则（官方FAQ，2026-10 更新）：
  * 入场价 = 下单后下一秒的指数价格；结算价 = 到期时刻的指数价格；
  * 结算价 > 入场价 -> HIGHER 胜；< 入场价 -> LOWER 胜；相等 -> 退还本金；
  * 不能提前平仓。

本系统的标签约定（用户指定）：
  入场价 = 决策分钟的收盘指数价（= 下一分钟开盘价），到期价 = h 分钟后的收盘指数价。
  y = 1 上涨 / 0 下跌 / -1 平局(退款，不参与训练) / -2 无效(缺数据)
实际下单发生在分钟内的某一秒，分钟标签是对秒级结算的近似，approximation_report 评估该误差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UP, DOWN, TIE, INVALID = 1, 0, -1, -2


def make_labels(ix_close: np.ndarray, horizon: int, entry_delay: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """返回 (y, move)。move = 到期价/入场价 - 1。行 i 的决策时刻为 ts[i]。"""
    c = np.asarray(ix_close, dtype=np.float64)
    n = len(c)
    y = np.full(n, INVALID, dtype=np.int8)
    move = np.full(n, np.nan)
    last = n - horizon - entry_delay
    if last <= 0:
        return y, move
    entry = c[entry_delay:entry_delay + last]
    expiry = c[entry_delay + horizon:entry_delay + horizon + last]
    ok = np.isfinite(entry) & np.isfinite(expiry) & (entry > 0)
    mv = np.where(ok, expiry / np.where(ok, entry, 1.0) - 1.0, np.nan)
    lab = np.where(expiry > entry, UP, np.where(expiry < entry, DOWN, TIE)).astype(np.int8)
    y[:last] = np.where(ok, lab, INVALID)
    move[:last] = mv
    return y, move


def approximation_report(panel: pd.DataFrame, horizons: list[int], spot_1s: pd.DataFrame | None = None) -> dict:
    """评估分钟标签对真实秒级结算的近似误差。

    1) 平局比例；
    2) 脆弱标签比例：|涨跌幅| 小于"1 分钟典型波动的一小部分"，入场时刻差几秒就可能翻转；
    3) 入场延迟 1 分钟的标签一致率（分钟数据能给出的保守下界）；
    4) 指数价 vs 现货价 的方向一致率（指数由多家交易所现货加权，现货只是代理）；
    5) 若提供现货 1 秒K线：入场延迟 δ 秒时标签的一致率（最接近真实下单场景的估计）。
    """
    c = panel["ix_c"].to_numpy(dtype=np.float64)
    r1 = np.abs(np.diff(np.log(np.where(c > 0, c, np.nan))))
    typical_1m = float(np.nanmedian(r1))
    out: dict = {"typical_1m_abs_move_bp": typical_1m * 1e4, "horizons": {}}
    for h in horizons:
        y0, mv0 = make_labels(c, h, 0)
        y1, _ = make_labels(c, h, 1)
        valid = (y0 >= 0) & (y1 >= 0)
        decided = y0 != INVALID
        rep = {
            "n": int(decided.sum()),
            "tie_rate": float((y0[decided] == TIE).mean()) if decided.any() else None,
            "up_rate": float((y0[y0 >= 0] == UP).mean()) if (y0 >= 0).any() else None,
            "median_abs_move_bp": float(np.nanmedian(np.abs(mv0)) * 1e4),
            # 入场价偏差约为 1 秒~数秒的波动；用 1 分钟典型波动的 25% 作为"脆弱带"
            "fragile_rate": float((np.abs(mv0[y0 >= 0]) < 0.25 * typical_1m).mean()),
            "agree_delay_1m": float((y0[valid] == y1[valid]).mean()) if valid.any() else None,
        }
        if "sp_c" in panel:
            ys, _ = make_labels(panel["sp_c"].to_numpy(dtype=np.float64), h, 0)
            v2 = (y0 >= 0) & (ys >= 0)
            rep["agree_index_vs_spot"] = float((y0[v2] == ys[v2]).mean()) if v2.any() else None
        out["horizons"][str(h)] = rep
    if spot_1s is not None and len(spot_1s) > 3600:
        out["seconds"] = _second_level(spot_1s, horizons)
    return out


def _second_level(spot_1s: pd.DataFrame, horizons: list[int], delays=(1, 3, 5, 10, 20, 30)) -> dict:
    """用现货 1 秒收盘价作为指数代理，测量入场延迟 δ 秒造成的标签翻转率。"""
    s = spot_1s.drop_duplicates("open_time").set_index("open_time")["close"]
    t0, t1 = int(s.index.min()) // 1000, int(s.index.max()) // 1000
    grid = np.arange(t0, t1 + 1, dtype=np.int64) * 1000
    px = s.reindex(grid).ffill().to_numpy(dtype=np.float64)      # 第 k 秒结束时的价格
    sec = grid // 1000
    minute_idx = np.where((sec + 1) % 60 == 0)[0]                # 每分钟最后一秒 = 分钟收盘
    res: dict = {}
    for h in horizons:
        span = h * 60
        base = minute_idx[minute_idx + span + max(delays) < len(px)]
        ref = np.sign(px[base + span] - px[base])
        row = {}
        for d in delays:
            alt = np.sign(px[base + d + span] - px[base + d])
            ok = (ref != 0) & (alt != 0)
            row[f"agree_delay_{d}s"] = float((ref[ok] == alt[ok]).mean())
        row["n_minutes"] = int(len(base))
        res[str(h)] = row
    return res
