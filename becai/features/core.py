"""自动特征生成：数据源 × 特征族 × 时间尺度。

设计原则（防未来数据泄漏）：
  1. 所有特征只由"截至决策时刻 ts 已收盘"的 1 分钟数据通过**有限长度的回看窗口**计算；
     不使用 EWM / expanding / 全局归一化，因此"用最近 N 分钟缓冲计算"与"用全历史计算"结果一致
     （实时与离线同口径，见 tests/test_features.py 的截断不变性测试）。
  2. 高周期信息有两种来源，都严格因果：
       - 多尺度滚动窗口（窗口终点 = 决策时刻）；
       - 对齐高周期K线：只用已收盘K线 + 当前K线"截至此刻"的部分。
  3. 不预设哪个周期/数据源有效：每个特征都带 (source, family, scale) 元数据，
     由后续的分组重要性、搜索和消融实验决定取舍。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils import MINUTE_MS

EPS = 1e-12


@dataclass
class FeatureSet:
    X: np.ndarray                 # float32 [n, F]
    columns: list[str]
    meta: dict[str, dict]         # col -> {source, family, scale}
    ts: np.ndarray                # int64 决策时刻
    channels: np.ndarray          # float32 [n, C] 供序列模型使用的逐分钟通道
    channel_names: list[str]
    regime: pd.DataFrame          # 市场状态标签（仅用于过滤与分层统计）

    def groups(self) -> dict[str, list[int]]:
        g: dict[str, list[int]] = {}
        for j, c in enumerate(self.columns):
            m = self.meta[c]
            g.setdefault(f"{m['source']}|{m['family']}|{m['scale']}", []).append(j)
        return g


class _Builder:
    def __init__(self, n: int):
        self.n = n
        self.cols: dict[str, np.ndarray] = {}
        self.meta: dict[str, dict] = {}

    def add(self, name: str, values, source: str, family: str, scale: int = 0) -> None:
        v = np.asarray(values, dtype=np.float64)
        v = np.where(np.isfinite(v), v, np.nan).astype(np.float32)
        self.cols[name] = v
        self.meta[name] = {"source": source, "family": family, "scale": int(scale)}


def _mp(w: int) -> int:
    return max(2, int(0.9 * w))


def _rsum(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=_mp(w)).sum()


def _rmean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=_mp(w)).mean()


def _rstd(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=_mp(w)).std()


def _safe_log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))


def _gather_back(arr: np.ndarray, back: np.ndarray) -> np.ndarray:
    """取 arr[i - back[i]]，越界为 NaN。"""
    idx = np.arange(len(arr)) - back
    out = np.full(len(arr), np.nan)
    ok = idx >= 0
    out[ok] = arr[idx[ok]]
    return out


def _trade_features(b: _Builder, p: pd.DataFrame, pre: str, source: str, lix: pd.Series, scales: list[int]) -> None:
    """现货 / 永续 的成交行为特征。"""
    if f"{pre}_c" not in p:
        return
    c, v, qv, n, tbv = (p[f"{pre}_{k}"] for k in ("c", "v", "qv", "n", "tbv"))
    lc = _safe_log(c)
    basis = lc - lix
    b.add(f"{pre}_basis", basis * 1e4, source, "basis", 0)
    v_day, n_day = _rmean(v, 1440), _rmean(n, 1440)
    signed = 2 * tbv - v
    b.add(f"{pre}_tbi_1", signed / (v + EPS), source, "flow", 1)
    b.add(f"{pre}_relvol_1", np.log1p(v / (v_day + EPS)), source, "volume", 1)
    avg_sz_day = _rsum(qv, 1440) / (_rsum(n, 1440) + EPS)
    for w in scales:
        sv = _rsum(v, w)
        b.add(f"{pre}_tbi_{w}", _rsum(signed, w) / (sv + EPS), source, "flow", w)
        b.add(f"{pre}_basis_chg_{w}", (basis - basis.shift(w)) * 1e4, source, "basis", w)
        b.add(f"{pre}_vwapdev_{w}", (lc - np.log((_rsum(qv, w) + EPS) / (sv + EPS))) * 1e4, source, "vwap", w)
        if w < 1440:
            b.add(f"{pre}_relvol_{w}", np.log1p(sv / (v_day * w + EPS)), source, "volume", w)
            b.add(f"{pre}_ntr_{w}", np.log1p(_rsum(n, w) / (n_day * w + EPS)), source, "intensity", w)
            b.add(f"{pre}_avgsz_{w}", np.log((_rsum(qv, w) / (_rsum(n, w) + EPS) + EPS) / (avg_sz_day + EPS)), source, "tradesize", w)


def build_features(panel: pd.DataFrame, others: dict[str, pd.DataFrame] | None, cfg, out_path=None) -> FeatureSet:
    """panel: 本币种面板；others: 其它币种面板（同一时间网格），用于跨品种特征。

    out_path: 给定时特征矩阵直接写入该 .npy 文件（内存映射），返回只读映射；用于长历史时控制内存峰值。
    """
    fc = cfg.features
    scales = [int(w) for w in fc.scales]
    n = len(panel)
    ts = panel.index.to_numpy(dtype=np.int64)
    b = _Builder(n)

    # ---------- 指数价格（标签同源）----------
    c, h, l, o = panel["ix_c"], panel["ix_h"], panel["ix_l"], panel["ix_o"]
    lix = _safe_log(c)
    r1 = lix.diff()
    absr = r1.abs()
    vol60 = _rstd(r1, 60)
    vol_day = _rstd(r1, 1440)
    for k in range(int(fc.micro_lags)):
        b.add(f"ix_r1_lag{k}", r1.shift(k) / (vol60 + EPS), "index", "micro", 1)
    rng1 = (h - l) / (c + EPS)
    b.add("ix_body", (c - o) / (h - l + EPS), "index", "micro", 1)
    b.add("ix_wick_up", (h - np.maximum(c, o)) / (h - l + EPS), "index", "micro", 1)
    b.add("ix_wick_dn", (np.minimum(c, o) - l) / (h - l + EPS), "index", "micro", 1)
    b.add("ix_rng1", rng1 / (vol60 + EPS), "index", "micro", 1)
    up = (r1 > 0).astype(float).where(r1.notna())
    for w in scales:
        ret = lix - lix.shift(w)
        b.add(f"ix_ret_{w}", ret * 1e4, "index", "return", w)
        vol = _rstd(r1, w) if w >= 5 else _rmean(absr, w)
        b.add(f"ix_vol_{w}", vol * 1e4, "index", "volatility", w)
        b.add(f"ix_z_{w}", ret / (vol60 * np.sqrt(w) + EPS), "index", "momentum", w)
        hh, ll = h.rolling(w, min_periods=_mp(w)).max(), l.rolling(w, min_periods=_mp(w)).min()
        b.add(f"ix_pos_{w}", (c - ll) / (hh - ll + EPS), "index", "range_pos", w)
        b.add(f"ix_rng_{w}", np.log(hh / ll) / (vol60 * np.sqrt(w) + EPS), "index", "range", w)
        b.add(f"ix_er_{w}", ret.abs() / (_rsum(absr, w) + EPS), "index", "efficiency", w)
        b.add(f"ix_up_{w}", _rmean(up, w), "index", "up_ratio", w)
        b.add(f"ix_acc_{w}", (ret - ret.shift(w)) / (vol60 * np.sqrt(w) + EPS), "index", "acceleration", w)
        if w < 1440:
            b.add(f"ix_volratio_{w}", vol / (vol_day + EPS), "index", "vol_ratio", w)

    # ---------- 对齐高周期K线（严格因果）----------
    minute_of = (ts // MINUTE_MS).astype(np.int64)
    lix_np = lix.to_numpy()
    for bar in [int(x) for x in fc.aligned_bars]:
        e = minute_of % bar
        e = np.where(e == 0, bar, e)                 # 恰好收盘时：当前K线=刚收盘的完整K线
        open_px = _gather_back(lix_np, e)
        prev_open = _gather_back(lix_np, e + bar)
        scale = vol60.to_numpy() * np.sqrt(bar) + EPS
        b.add(f"al_ret_{bar}", (lix_np - open_px) / scale, "index", "aligned_bar", bar)
        b.add(f"al_prev_{bar}", (open_px - prev_open) / scale, "index", "aligned_bar", bar)
        b.add(f"al_frac_{bar}", e / bar, "time", "aligned_phase", bar)

    # ---------- 现货 / 永续 / 标记 / 溢价 ----------
    _trade_features(b, panel, "sp", "spot", lix, scales)
    _trade_features(b, panel, "fu", "futures", lix, scales)
    if "fu_qv" in panel and "sp_qv" in panel:
        for w in scales:
            b.add(f"fu_sp_qv_{w}", np.log((_rsum(panel["fu_qv"], w) + EPS) / (_rsum(panel["sp_qv"], w) + EPS)), "futures", "venue_share", w)
    if "mk_c" in panel:
        b.add("mk_basis", (_safe_log(panel["mk_c"]) - lix) * 1e4, "mark", "basis", 0)
    if "pm_c" in panel:
        pm = panel["pm_c"] * 1e4
        b.add("pm_c", pm, "premium", "level", 0)
        b.add("pm_rng1", (panel["pm_h"] - panel["pm_l"]) * 1e4, "premium", "level", 1)
        for w in scales:
            m = _rmean(pm, w)
            b.add(f"pm_mean_{w}", m, "premium", "mean", w)
            b.add(f"pm_chg_{w}", pm - pm.shift(w), "premium", "change", w)
            if w >= 5:
                b.add(f"pm_z_{w}", (pm - m) / (_rstd(pm, w) + 1e-6), "premium", "zscore", w)

    # ---------- 资金费率 ----------
    if "fr" in panel:
        b.add("fr", panel["fr"] * 1e4, "funding", "level", 0)
        b.add("fr_age", panel["fr_age"] / 480.0, "funding", "phase", 0)

    # ---------- 持仓量 / 多空比（5分钟数据，带发布延迟保护）----------
    if "mt_oi" in panel:
        loi = _safe_log(panel["mt_oi"])
        for w in [w for w in scales if w >= 5]:
            b.add(f"oi_chg_{w}", (loi - loi.shift(w)) * 1e3, "metrics", "oi_change", w)
        for f in ("ls_top_acct", "ls_top_pos", "ls_acct", "taker_ls"):
            if f"mt_{f}" not in panel:
                continue
            lv = _safe_log(panel[f"mt_{f}"])
            b.add(f"{f}", lv, "metrics", f, 0)
            for w in (30, 240, 1440):
                b.add(f"{f}_chg_{w}", lv - lv.shift(w), "metrics", f, w)

    # ---------- 跨品种 ----------
    for osym, op in (others or {}).items():
        tag = osym.replace("USDT", "").lower()
        olix = _safe_log(op["ix_c"].reindex(panel.index))
        or1 = olix.diff()
        ovol60 = _rstd(or1, 60)
        for k in range(3):
            b.add(f"x_{tag}_r1_lag{k}", or1.shift(k) / (ovol60 + EPS), "cross", "micro", 1)
        for w in scales:
            oret = olix - olix.shift(w)
            b.add(f"x_{tag}_z_{w}", oret / (ovol60 * np.sqrt(w) + EPS), "cross", "momentum", w)
            b.add(f"x_{tag}_rel_{w}", ((lix - lix.shift(w)) - oret) * 1e4, "cross", "relative", w)
            if w >= 30:
                b.add(f"x_{tag}_corr_{w}", r1.rolling(w, min_periods=_mp(w)).corr(or1), "cross", "correlation", w)
        b.add(f"x_{tag}_volratio", vol60 / (ovol60 + EPS), "cross", "vol_ratio", 60)

    # ---------- 时间 ----------
    mins = minute_of
    for name, period in (("hour", 60), ("day", 1440), ("week", 10080), ("fund", 480)):
        ang = 2 * np.pi * (mins % period) / period
        b.add(f"t_{name}_sin", np.sin(ang), "time", "cyclic", period)
        b.add(f"t_{name}_cos", np.cos(ang), "time", "cyclic", period)
    b.add("t_to_funding", (480 - mins % 480) / 480.0, "time", "funding_phase", 480)

    # ---------- 市场状态（因果分位）----------
    rw = int(fc.regime_window)
    vol_pct = vol60.rolling(rw, min_periods=_mp(rw)).rank(pct=True)
    z240 = (lix - lix.shift(240)) / (vol60 * np.sqrt(240) + EPS)
    b.add("rg_volpct", vol_pct, "index", "regime", rw)
    hour = (mins % 1440) // 60
    regime = pd.DataFrame({
        "vol": np.select([vol_pct < 1 / 3, vol_pct < 2 / 3], ["低波动", "中波动"], "高波动"),
        "trend": np.select([z240 > 1, z240 < -1], ["上行趋势", "下行趋势"], "震荡"),
        "session": np.select([hour < 8, hour < 16], ["亚洲时段", "欧洲时段"], "美洲时段"),
        "weekend": np.where(((mins // 1440 + 3) % 7) >= 5, "周末", "工作日"),   # 1970-01-01 是周四(=3)
    }, index=panel.index)
    regime.loc[vol_pct.isna().to_numpy(), "vol"] = "未知"

    # ---------- 序列模型通道 ----------
    ch: dict[str, np.ndarray] = {"r_ix": (r1 / (vol_day + EPS)).to_numpy(), "rng_ix": (rng1 / (vol_day + EPS)).to_numpy()}
    for pre in ("sp", "fu"):
        if f"{pre}_c" in panel:
            v = panel[f"{pre}_v"]
            ch[f"r_{pre}"] = (_safe_log(panel[f"{pre}_c"]).diff() / (vol_day + EPS)).to_numpy()
            ch[f"lv_{pre}"] = np.log1p(v / (_rmean(v, 1440) + EPS)).to_numpy()
            ch[f"tbi_{pre}"] = ((2 * panel[f"{pre}_tbv"] - v) / (v + EPS)).to_numpy()
            ch[f"basis_{pre}"] = ((_safe_log(panel[f"{pre}_c"]) - lix) * 1e3).to_numpy()
    if "pm_c" in panel:
        ch["pm"] = (panel["pm_c"] * 1e3).to_numpy()
    for osym, op in (others or {}).items():
        ch[f"r_x_{osym[:3].lower()}"] = (_safe_log(op["ix_c"].reindex(panel.index)).diff() / (vol_day + EPS)).to_numpy()
    channels = np.column_stack([np.where(np.isfinite(v), v, 0.0) for v in ch.values()]).astype(np.float32)
    np.clip(channels, -20, 20, out=channels)

    columns = list(b.cols)
    if out_path is not None:
        # 按行分块顺序写入磁盘（逐列写入行优先文件会对整个文件反复随机写，极慢）
        X = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, len(columns)))
        step = 200_000
        for lo in range(0, n, step):
            hi = min(n, lo + step)
            X[lo:hi] = np.column_stack([b.cols[c][lo:hi] for c in columns])
        X.flush()
        del X
        b.cols.clear()
        X = np.load(out_path, mmap_mode="r")
    else:
        X = np.empty((n, len(columns)), dtype=np.float32)
        for j, name in enumerate(columns):
            X[:, j] = b.cols[name]
            b.cols[name] = None  # 及时释放
    return FeatureSet(X, columns, b.meta, ts, channels, list(ch), regime)


def max_lookback(cfg) -> int:
    """特征依赖的最大回看分钟数（实时缓冲必须不小于它）。"""
    fc = cfg.features
    return int(max(2 * 1440, max(fc.scales) + 1440, 2 * max(fc.aligned_bars), int(fc.regime_window) + 60)) + 5
