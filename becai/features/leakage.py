"""特征数据泄漏检测。

1) 截断不变性：把面板在时刻 k 截断后重新计算特征，k 及之前的特征值必须与全量计算一致。
   任何使用未来数据的写法（shift(-1)、居中窗口、bfill、全局归一化…）都会被检出。
2) 尾部一致性：只用最近 L 分钟缓冲计算的最后一行，必须与全量计算一致（实时/离线同口径）。
3) 单特征可疑预测力：单个特征对标签的 AUC 过高几乎一定是泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import build_features, max_lookback


def _compare(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    """返回每列是否不一致。NaN 位置必须相同。"""
    nan_mismatch = (np.isnan(a) != np.isnan(b)).any(axis=0)
    diff = np.abs(np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0))
    tol = atol + rtol * np.maximum(np.abs(np.nan_to_num(a, nan=0.0)), np.abs(np.nan_to_num(b, nan=0.0)))
    return nan_mismatch | (diff > tol).any(axis=0)


def truncation_check(panels: dict[str, pd.DataFrame], cfg, n_cuts: int = 3, tail: int = 300,
                     rtol: float = 2e-3, atol: float = 2e-3, seed: int = 0) -> dict:
    """对每个币种在若干随机截断点检查特征是否依赖未来数据。"""
    rng = np.random.default_rng(seed)
    bad: dict[str, list[str]] = {}
    checked = 0
    for sym, p in panels.items():
        others = {o: q for o, q in panels.items() if o != sym}
        full = build_features(p, others, cfg)
        n = len(p)
        lo = min(n - 2, max_lookback(cfg) + tail)
        if lo >= n - 1:
            continue
        for cut in sorted(rng.integers(lo, n - 1, size=n_cuts)):
            sub = build_features(p.iloc[: cut + 1], {o: q.iloc[: cut + 1] for o, q in others.items()}, cfg)
            sl = slice(cut + 1 - tail, cut + 1)
            mism = _compare(full.X[sl], sub.X[sl], rtol, atol)
            mism_c = _compare(full.channels[sl], sub.channels[sl], rtol, atol)
            checked += 1
            names = [full.columns[j] for j in np.where(mism)[0]] + [f"channel:{full.channel_names[j]}" for j in np.where(mism_c)[0]]
            if names:
                bad.setdefault(sym, []).extend(names)
    return {"passed": not bad, "cuts_checked": checked, "leaky_features": {k: sorted(set(v)) for k, v in bad.items()}}


def tail_parity_check(panels: dict[str, pd.DataFrame], cfg, buffer_minutes: int | None = None,
                      rtol: float = 2e-3, atol: float = 2e-3) -> dict:
    """实时缓冲（只含最近 N 分钟）计算出的最后一行特征是否与全量一致。"""
    buf = int(buffer_minutes or cfg.live.buffer_minutes)
    bad: dict[str, list[str]] = {}
    for sym, p in panels.items():
        if len(p) <= buf:
            continue
        others = {o: q for o, q in panels.items() if o != sym}
        full = build_features(p, others, cfg)
        sub = build_features(p.iloc[-buf:], {o: q.iloc[-buf:] for o, q in others.items()}, cfg)
        mism = _compare(full.X[-1:], sub.X[-1:], rtol, atol)
        if mism.any():
            bad[sym] = [full.columns[j] for j in np.where(mism)[0]]
    return {"passed": not bad, "buffer_minutes": buf, "required_minutes": max_lookback(cfg), "mismatch": bad}


def suspicious_auc(X: np.ndarray, y: np.ndarray, columns: list[str], threshold: float = 0.60, max_rows: int = 200_000) -> dict:
    """单特征 AUC 扫描。10/30 分钟方向预测中，单特征 AUC 超过 0.60 基本可判定为泄漏。"""
    from scipy.stats import rankdata

    idx = np.where(y >= 0)[0]
    if len(idx) > max_rows:
        idx = idx[:: len(idx) // max_rows + 1]
    yy = y[idx].astype(bool)
    n1, n0 = int(yy.sum()), int((~yy).sum())
    flagged, aucs = [], {}
    if n1 == 0 or n0 == 0:
        return {"passed": True, "flagged": [], "max_auc": None}
    for j, name in enumerate(columns):
        x = np.asarray(X[idx, j], dtype=np.float64)
        ok = np.isfinite(x)
        if ok.sum() < 1000:
            continue
        r = rankdata(x[ok])
        a1 = int(yy[ok].sum())
        a0 = int(ok.sum()) - a1
        if a1 == 0 or a0 == 0:
            continue
        auc = (r[yy[ok]].sum() - a1 * (a1 + 1) / 2) / (a1 * a0)
        aucs[name] = float(max(auc, 1 - auc))
        if aucs[name] > threshold:
            flagged.append({"feature": name, "auc": aucs[name]})
    return {"passed": not flagged, "flagged": flagged, "max_auc": max(aucs.values()) if aucs else None,
            "top": sorted(aucs.items(), key=lambda kv: -kv[1])[:10]}
