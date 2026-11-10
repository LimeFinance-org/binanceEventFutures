"""统计工具：概率质量指标、置信区间、重叠样本处理、过拟合检测。"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy.stats import norm


# ---------------- 概率质量 ----------------
def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """期望校准误差（等宽分箱）。"""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
             "pred": float(p[idx == b].mean()), "actual": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def prob_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    return {"n": int(len(y)), "logloss": logloss(y, p), "brier": brier(y, p), "auc": auc(y, p), "ece": ece(y, p),
            "accuracy": float(((p > 0.5) == (y > 0.5)).mean()), "base_rate": float(y.mean())}


# ---------------- 置信区间 ----------------
def wilson_ci(k: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(0.5 + conf / 2)
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(mid - half), float(mid + half))


def block_bootstrap_mean(values: np.ndarray, ts_ms: np.ndarray, block_minutes: int, n_boot: int = 2000,
                         conf: float = 0.95, seed: int = 0) -> dict:
    """按时间块重采样的均值置信区间。

    每分钟滚动的 10/30 分钟合约彼此高度重叠，样本不独立；
    把同一时间块内的样本整体重采样，可保留块内相关性，得到诚实（更宽）的区间。
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_blocks": 0}
    block = (np.asarray(ts_ms) // (block_minutes * 60_000)).astype(np.int64)
    _, inv = np.unique(block, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    cnts = np.bincount(inv).astype(np.float64)
    nb = len(sums)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, nb, size=(n_boot, nb))
    means = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    a = (1 - conf) / 2
    return {"mean": float(values.mean()), "lo": float(np.quantile(means, a)), "hi": float(np.quantile(means, 1 - a)),
            "n_blocks": int(nb), "se": float(means.std())}


def effective_sample_size(values: np.ndarray, ts_ms: np.ndarray, max_lag_minutes: int = 60) -> float:
    """按分钟自相关估计有效样本量 N_eff = N / (1 + 2Σρ_k)。"""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 10:
        return float(n)
    t = (np.asarray(ts_ms) // 60_000).astype(np.int64)
    span = int(t.max() - t.min()) + 1
    series = np.full(span, np.nan)
    series[t - t.min()] = values
    mu = values.mean()
    var = values.var()
    if var <= 0:
        return float(n)
    s = 0.0
    d = series - mu
    for lag in range(1, max_lag_minutes + 1):
        prod = d[lag:] * d[:-lag]
        ok = np.isfinite(prod)
        if ok.sum() < 20:
            break
        rho = prod[ok].mean() / var
        if rho <= 0.02:
            break
        s += rho
    return float(n / (1 + 2 * s))


def non_overlapping(ts_ms: np.ndarray, horizon_minutes: int) -> np.ndarray:
    """贪心挑选互不重叠（间隔 >= 期限）的样本下标，用于"去重叠胜率"。"""
    keep, last = [], -10**18
    gap = horizon_minutes * 60_000
    for i, t in enumerate(np.asarray(ts_ms)):
        if t - last >= gap:
            keep.append(i)
            last = t
    return np.asarray(keep, dtype=np.int64)


def win_rate_summary(win: np.ndarray, ts_ms: np.ndarray, horizon: int, block_minutes: int, n_boot: int = 2000,
                     conf: float = 0.95, seed: int = 0) -> dict:
    """信号胜率的完整统计：原始、Wilson、区块自助法、有效样本量、去重叠子样本。"""
    win = np.asarray(win, dtype=np.float64)
    n = len(win)
    if n == 0:
        return {"n": 0, "win_rate": None}
    wl, wh = wilson_ci(win.sum(), n, conf)
    bb = block_bootstrap_mean(win, ts_ms, max(block_minutes, horizon), n_boot, conf, seed)
    neff = effective_sample_size(win, ts_ms)
    el, eh = wilson_ci(win.mean() * neff, neff, conf)
    idx = non_overlapping(ts_ms, horizon)
    nl, nh = wilson_ci(win[idx].sum(), len(idx), conf)
    return {
        "n": int(n), "win_rate": float(win.mean()),
        "wilson_ci": [wl, wh],                       # 假设独立（偏乐观，仅作参考）
        "block_bootstrap_ci": [bb["lo"], bb["hi"]],  # 推荐使用
        "n_effective": float(neff), "effective_ci": [el, eh],
        "non_overlap": {"n": int(len(idx)), "win_rate": float(win[idx].mean()), "ci": [nl, nh]},
    }


# ---------------- 过拟合检测 ----------------
def pbo_cscv(perf: np.ndarray, max_combos: int = 2000, seed: int = 0) -> dict:
    """回测过拟合概率 PBO（Bailey 等的 CSCV 方法）。

    perf: [T 个时间段, N 个候选配置]，数值越大越好。
    把时间段对半划分为"样本内/样本外"，统计样本内最优配置在样本外排名低于中位数的概率。
    """
    perf = np.asarray(perf, dtype=np.float64)
    T, N = perf.shape
    if T < 4 or N < 2:
        return {"pbo": None, "n_combos": 0, "note": "时间段或候选数不足"}
    half = T // 2
    combos = list(combinations(range(T), half))
    if len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    logits = []
    for is_idx in combos:
        oos_idx = [t for t in range(T) if t not in is_idx]
        best = int(np.argmax(perf[list(is_idx)].mean(axis=0)))
        oos = perf[oos_idx].mean(axis=0)
        rank = (oos < oos[best]).sum() + 0.5 * (oos == oos[best]).sum()
        w = np.clip(rank / N, 1e-6, 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    logits = np.asarray(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combos": len(combos), "total_combos": comb(T, half),
            "median_logit": float(np.median(logits))}


def multiple_testing_threshold(n_trials: int, alpha: float = 0.05) -> float:
    """Šidák 校正后的单侧 z 阈值：尝试了 n_trials 个配置后，"显著"需要达到的 z 值。"""
    a = 1 - (1 - alpha) ** (1 / max(1, n_trials))
    return float(norm.ppf(1 - a))


def edge_significance(win_rate: float, se: float, breakeven: float, n_trials: int) -> dict:
    """胜率是否显著高于盈亏平衡点（考虑搜索过的配置数）。"""
    if not se or se <= 0 or win_rate is None:
        return {"z": None, "significant": False}
    z = (win_rate - breakeven) / se
    zc = multiple_testing_threshold(n_trials)
    return {"z": float(z), "z_required": zc, "p_value_single": float(1 - norm.cdf(z)), "significant": bool(z > zc),
            "n_trials_considered": int(n_trials)}


def paired_block_bootstrap(diff: np.ndarray, ts_ms: np.ndarray, block_minutes: int, n_boot: int = 1000,
                           conf: float = 0.95, seed: int = 0) -> dict:
    """成对差值（如两个模型的逐样本对数损失之差）的区块自助法检验。"""
    r = block_bootstrap_mean(diff, ts_ms, block_minutes, n_boot, 2 * conf - 1, seed)   # 单侧置信下界
    return {"mean": r["mean"], "lower": r["lo"], "upper": r["hi"], "significant": bool(r["lo"] > 0)}
