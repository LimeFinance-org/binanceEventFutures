"""样本外评估：区分 方向准确率 / 有效信号胜率 / 固定赔率下的期望 / 模拟收益。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest.simulator import simulate, stress_test
from .bundle import DataBundle
from .labels import make_labels
from .signals.policy import Policy, breakeven_win_rate
from .validation.stats import (block_bootstrap_mean, edge_significance, prob_metrics, reliability_table, wilson_ci,
                               win_rate_summary)


def evaluate_task(bundle: DataBundle, task: tuple[str, int], rows: np.ndarray, p_cal: np.ndarray, policy: Policy, cfg,
                  n_trials: int) -> tuple[dict, list[dict]]:
    """返回 (报告, 信号列表)。信号列表供组合回测使用。"""
    sym, h = task
    sd = bundle.symbols[sym]
    y = sd.labels[h][rows]
    ts = bundle.ts[rows]
    regime = sd.regime.iloc[rows]
    payout = float(cfg.payout.fixed)
    be = breakeven_win_rate(payout)
    bs = cfg.validation.bootstrap
    rep: dict = {"task": [sym, h], "n_minutes": int(len(rows)), "payout": payout, "breakeven_win_rate": be,
                 "target_win_rate": float(cfg.signal.target_win_rate)}
    decided = (y >= 0) & np.isfinite(p_cal)
    rep["tie_rate"] = float((y == -1).mean())
    # 1) 全部分钟的方向预测质量（不筛选）
    rep["direction_all"] = prob_metrics(y[decided].astype(float), p_cal[decided]) if decided.sum() > 100 else {}
    rep["reliability"] = reliability_table(y[decided].astype(float), p_cal[decided], 10) if decided.sum() > 100 else []
    # 2) 有效开仓信号
    direction, reason = policy.decide(p_cal, regime)
    sig = (direction != 0) & (y >= -1)
    n_sig = int(sig.sum())
    rep["policy"] = policy.to_dict()
    rep["signals"] = {"n": n_sig, "coverage": n_sig / max(1, len(rows)), "per_day": n_sig / max(1e-9, len(rows) / 1440),
                      "higher": int((direction[sig] == 1).sum()), "lower": int((direction[sig] == -1).sum()),
                      "ties": int((y[sig] == -1).sum())}
    skip = pd.Series(reason[direction == 0]).str.replace(r"\(.*\)", "", regex=True).str.strip().value_counts()
    rep["skip_reasons"] = {k: int(v) for k, v in skip.items()}
    signals: list[dict] = []
    res = sig & (y >= 0)
    if res.sum() >= 1:
        win = ((direction[res] == 1) == (y[res] == 1)).astype(float)
        ts_s = ts[res]
        summ = win_rate_summary(win, ts_s, h, int(bs.block_minutes), int(bs.n_boot))
        rep["signals"].update(summ)
        pnl = np.where(win > 0, payout, -1.0)
        ev = block_bootstrap_mean(pnl, ts_s, max(int(bs.block_minutes), h), int(bs.n_boot))
        rep["expected_value_per_unit"] = {"mean": ev["mean"], "ci": [ev["lo"], ev["hi"]], "note": f"固定赔率 {payout:.0%} 假设下每 1 USDT 本金的期望盈亏"}
        se = (summ["block_bootstrap_ci"][1] - summ["block_bootstrap_ci"][0]) / (2 * 1.96)
        rep["edge_vs_breakeven"] = edge_significance(summ["win_rate"], se, be, n_trials)
        lcb = summ["block_bootstrap_ci"][0]
        tgt = float(cfg.signal.target_win_rate)
        rep["target_assessment"] = ("达标（置信下界 ≥ 目标）" if lcb >= tgt else
                                    "点估计达标，但统计上不显著（置信下界低于目标）" if summ["win_rate"] >= tgt else "未达标")
        # 3) 分市场状态 / 分月稳定性
        by = {}
        reg_s = regime[res]
        for dim in ("vol", "trend", "session", "weekend"):
            rows_ = []
            for val, idx in reg_s.groupby(dim).indices.items():
                lo, hi = wilson_ci(win[idx].sum(), len(idx))
                rows_.append({"value": val, "n": int(len(idx)), "win_rate": float(win[idx].mean()), "ci": [lo, hi]})
            by[dim] = rows_
        month = pd.to_datetime(ts_s, unit="ms").strftime("%Y-%m")
        by["month"] = [{"value": m, "n": int((month == m).sum()), "win_rate": float(win[month == m].mean())} for m in sorted(set(month))]
        week = pd.to_datetime(ts_s, unit="ms").strftime("%G-W%V")
        wk = pd.Series(win).groupby(np.asarray(week)).agg(["mean", "size"])
        wk = wk[wk["size"] >= 20]
        by["weekly"] = {"n_weeks": int(len(wk)), "min": float(wk["mean"].min()) if len(wk) else None,
                        "median": float(wk["mean"].median()) if len(wk) else None,
                        "share_above_breakeven": float((wk["mean"] > be).mean()) if len(wk) else None}
        rep["stability"] = by
        # 4) 置信度分桶：置信度越高胜率是否越高（单调性诊断，不用于选参）
        conf = np.maximum(p_cal[res], 1 - p_cal[res])
        edges = np.quantile(conf, [0, 0.25, 0.5, 0.75, 1.0])
        rep["confidence_buckets"] = [{"range": [float(edges[i]), float(edges[i + 1])],
                                      "n": int(((conf >= edges[i]) & (conf <= edges[i + 1])).sum()),
                                      "win_rate": float(win[(conf >= edges[i]) & (conf <= edges[i + 1])].mean())}
                                     for i in range(4) if ((conf >= edges[i]) & (conf <= edges[i + 1])).any()]
        # 5) 入场延迟 1 分钟的敏感性（实际下单晚于决策时刻时的保守估计）
        y1, _ = make_labels(sd.ix_close, h, 1)
        y1 = y1[rows][res]
        ok1 = y1 >= 0
        if ok1.sum() > 20:
            w1 = ((direction[res][ok1] == 1) == (y1[ok1] == 1)).astype(float)
            rep["entry_delay_1m"] = {"n": int(ok1.sum()), "win_rate": float(w1.mean()), "ci": list(wilson_ci(w1.sum(), len(w1)))}
    idx_sig = np.where(sig)[0]
    for i in idx_sig:
        d = int(direction[i])
        signals.append({"ts": int(ts[i]), "symbol": sym, "horizon": h, "direction": d,
                        "p_win": float(p_cal[i] if d == 1 else 1 - p_cal[i]),
                        "outcome": 0 if y[i] == -1 else (1 if y[i] == 1 else -1)})
    return rep, signals


def rolling_correlation(bundle: DataBundle, rows: np.ndarray) -> float | None:
    """评估区间之前的 BTC/ETH 分钟收益相关系数（只用评估起点之前的数据，供风控使用）。"""
    syms = list(bundle.symbols)
    if len(syms) < 2 or len(rows) == 0:
        return None
    end = int(rows.min())
    lo = max(0, end - 30 * 1440)
    a = np.diff(np.log(bundle.symbols[syms[0]].ix_close[lo:end]))
    b = np.diff(np.log(bundle.symbols[syms[1]].ix_close[lo:end]))
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 1000 else None


def portfolio_backtest(signals: list[dict], cfg, correlation: float | None) -> dict:
    """组合层面：四个任务的信号一起经过全局风控。主情景 + 赔率敏感性 + 压力测试。"""
    main = simulate(signals, cfg.risk, float(cfg.payout.fixed), correlation)
    trades = main.pop("trades")
    out = {"disclaimer": "固定赔率模拟，不是真实订单结算收益；历史上没有可用的真实赔率数据。",
           "correlation_used": correlation, "main": main, "scenarios": {}}
    for r in cfg.backtest.payout_scenarios:
        s = simulate(signals, cfg.risk, float(r), correlation)
        s.pop("trades"); s.pop("equity_curve")
        out["scenarios"][f"{float(r):.2f}"] = s
    out["stress"] = stress_test(trades, cfg.risk, float(cfg.payout.fixed), list(cfg.backtest.stress.win_rate_haircuts),
                                int(cfg.backtest.stress.n_paths))
    out["unconstrained"] = {"note": "不经过风控、每个信号各投 1 单位的理论值（仅用于对照风控的影响）",
                            "n": len(signals),
                            "win_rate": float(np.mean([s["direction"] == s["outcome"] for s in signals if s["outcome"] != 0])) if signals else None}
    return out
