"""开仓信号策略：置信度阈值 + 市场状态过滤，全部在验证数据（样本外预测）上自动选择。

决策：HIGHER / LOWER / NO_TRADE。
选择规则（不人为固定"最佳参数"）：
  1. 可行性约束：覆盖率、信号数、去重叠有效样本数、跨折稳定性；
  2. 若存在"胜率置信下界 >= 研究目标(70%)"的阈值，取其中覆盖率最高者；
  3. 否则取"按胜率置信下界计算的期望收益 × 覆盖率"最大且为正的阈值（正期望但未达 70%）；
  4. 都不满足：该任务不发出开仓信号（如实报告未发现可交易优势）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..validation.stats import block_bootstrap_mean, non_overlapping

HIGHER, LOWER, NO_TRADE = "HIGHER", "LOWER", "NO_TRADE"
REGIME_DIMS = ("vol", "trend", "session", "weekend")


def breakeven_win_rate(payout: float) -> float:
    """赔率 r（赢得 r 倍本金，输掉全部本金）下的盈亏平衡胜率。"""
    return 1.0 / (1.0 + payout)


@dataclass
class Policy:
    task: list
    threshold: float | None                      # None = 不开仓
    excluded: dict = field(default_factory=dict)  # {状态维度: [被过滤的取值]}
    status: str = "no_edge"                      # target_met / positive_ev / no_edge
    payout: float = 0.8
    validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Policy":
        return Policy(**{k: d[k] for k in ("task", "threshold", "excluded", "status", "payout", "validation") if k in d})

    def regime_mask(self, regime: pd.DataFrame) -> np.ndarray:
        ok = np.ones(len(regime), dtype=bool)
        for dim, vals in (self.excluded or {}).items():
            if dim in regime:
                ok &= ~regime[dim].isin(vals).to_numpy()
        return ok

    def decide(self, p_up: np.ndarray, regime: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """返回 (direction[+1/-1/0], reason[str])。"""
        p = np.asarray(p_up, dtype=np.float64)
        conf = np.maximum(p, 1 - p)
        direction = np.zeros(len(p), dtype=np.int8)
        reason = np.full(len(p), "", dtype=object)
        bad = ~np.isfinite(p)
        reason[bad] = "数据不可用或特征无效"
        if self.threshold is None:
            reason[~bad] = "该任务在验证期未发现可交易优势"
            return direction, reason
        low = ~bad & (conf < self.threshold)
        reason[low] = [f"置信度不足 ({c:.3f} < 阈值 {self.threshold:.3f})" for c in conf[low]]
        rm = self.regime_mask(regime)
        blocked = ~bad & ~low & ~rm
        if blocked.any():
            desc = "、".join(f"{'/'.join(v)}" for v in self.excluded.values())
            reason[blocked] = f"市场状态被过滤 ({desc})"
        go = ~bad & ~low & rm
        direction[go] = np.where(p[go] >= 0.5, 1, -1)
        return direction, reason


def _signal_stats(p: np.ndarray, y: np.ndarray, ts: np.ndarray, fold: np.ndarray, mask: np.ndarray, tau: float,
                  horizon: int, payout: float, block: int, n_boot: int, n_total: int, conf: float = 0.95) -> dict | None:
    c = np.maximum(p, 1 - p)
    sel = mask & (c >= tau) & (y >= 0)
    n = int(sel.sum())
    if n < 30:
        return None
    win = ((p[sel] >= 0.5) == (y[sel] == 1)).astype(np.float64)
    bb = block_bootstrap_mean(win, ts[sel], max(block, horizon), n_boot, conf, seed=0)
    folds = np.unique(fold[sel])
    be = breakeven_win_rate(payout)
    fold_wr = [float(win[fold[sel] == k].mean()) for k in folds if (fold[sel] == k).sum() >= 20]
    wr, lcb = float(win.mean()), float(bb["lo"])
    return {
        "threshold": float(tau), "n": n, "coverage": n / max(1, n_total), "win_rate": wr, "lcb": lcb, "ucb": float(bb["hi"]),
        "n_nonoverlap": int(len(non_overlapping(ts[sel], horizon))),
        "fold_pass_ratio": float(np.mean([w > be for w in fold_wr])) if fold_wr else 0.0,
        "fold_win_rates": fold_wr, "ev": wr * payout - (1 - wr), "ev_lcb": lcb * payout - (1 - lcb),
    }


def optimize_policy(task, p_cal: np.ndarray, y: np.ndarray, ts: np.ndarray, fold: np.ndarray, regime: pd.DataFrame, cfg,
                    payout: float) -> tuple[Policy, dict]:
    """在校准后的样本外预测上选择阈值与市场状态过滤。返回 (策略, 诊断信息)。"""
    sc = cfg.signal
    horizon = int(task[1])
    block, n_boot = int(cfg.validation.bootstrap.block_minutes), min(500, int(cfg.validation.bootstrap.n_boot))
    valid = np.isfinite(p_cal) & (y >= -1)
    n_total = int(valid.sum())
    lo, hi, step = sc.threshold_grid
    grid = np.round(np.arange(lo, hi + 1e-9, step), 4)

    filters: list[dict] = [{}]
    if sc.regime_filters:
        for dim in REGIME_DIMS:
            for val in sorted(set(regime[dim].unique()) - {"未知"}):
                filters.append({dim: [val]})

    def feasible(s: dict) -> bool:
        return (s["coverage"] >= sc.min_coverage and s["n"] >= sc.min_signals and s["n_nonoverlap"] >= sc.min_effective_n
                and s["fold_pass_ratio"] >= sc.min_fold_pass_ratio)

    def best_for(excluded: dict):
        mask = valid.copy()
        for dim, vals in excluded.items():
            mask &= ~regime[dim].isin(vals).to_numpy()
        curve = [s for tau in grid if (s := _signal_stats(p_cal, y, ts, fold, mask, tau, horizon, payout, block, n_boot, n_total))]
        feas = [s for s in curve if feasible(s)]
        hit = [s for s in feas if s["lcb"] >= sc.target_win_rate]
        if hit:
            return max(hit, key=lambda s: s["coverage"]), "target_met", curve
        pos = [s for s in feas if s["ev_lcb"] > 0]
        if pos:
            return max(pos, key=lambda s: s["ev_lcb"] * s["coverage"]), "positive_ev", curve
        return None, "no_edge", curve

    rank = {"target_met": 2, "positive_ev": 1, "no_edge": 0}
    base_pick, base_status, base_curve = best_for({})
    best = (base_pick, base_status, {}, base_curve)

    def score(pick, status):
        if pick is None:
            return (0, 0.0)
        return (rank[status], pick["coverage"] if status == "target_met" else pick["ev_lcb"] * pick["coverage"])

    tried = [{"excluded": {}, "status": base_status, "pick": base_pick, "adopted": True}]
    for ex in filters[1:]:
        pick, status, curve = best_for(ex)
        entry = {"excluded": ex, "status": status, "pick": pick, "adopted": False}
        tried.append(entry)
        if pick is None:
            continue
        # 过滤器要同时满足两点才采用（一共会尝试十几个过滤器，必须防止偶然性）：
        #   a) 目标值比当前最优提升 10% 以上；
        #   b) 被排除状态下的信号胜率"显著"低于保留部分（排除部分的置信上界 < 保留部分的置信下界）。
        removed = valid.copy()
        hit = np.zeros(len(valid), dtype=bool)
        for dim, vals in ex.items():
            hit |= regime[dim].isin(vals).to_numpy()
        removed &= hit
        ex_stats = _signal_stats(p_cal, y, ts, fold, removed, pick["threshold"], horizon, payout, block, n_boot, n_total)
        entry["excluded_stats"] = None if ex_stats is None else {k: ex_stats[k] for k in ("n", "win_rate", "lcb", "ucb")}
        significant = ex_stats is not None and ex_stats["ucb"] < pick["lcb"]
        s_new, s_old = score(pick, status), score(best[0], best[1])
        better = s_new[0] > s_old[0] or (s_new[0] == s_old[0] and s_new[1] > 1.1 * s_old[1])
        if significant and better:
            best = (pick, status, ex, curve)
            entry["adopted"] = True
    pick, status, excluded, curve = best
    policy = Policy(list(task), None if pick is None else pick["threshold"], excluded if pick else {}, status, payout,
                    pick or {})
    diag = {"curve": base_curve, "chosen_curve": curve, "filters_tried": tried, "n_validation_minutes": n_total,
            "breakeven": breakeven_win_rate(payout), "target": float(sc.target_win_rate),
            "best_effort": max(base_curve, key=lambda s: s["lcb"]) if base_curve else None}
    return policy, diag
