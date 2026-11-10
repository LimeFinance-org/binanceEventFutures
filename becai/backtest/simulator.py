"""逐分钟事件驱动回测：信号 -> 全局风控审批 -> 开仓 -> 到期按指数价结算。

重要声明：没有历史真实赔率。这里使用配置的固定赔率（默认 80%）做模拟，
结果是"固定赔率模拟收益"，不是"真实订单结算收益"。
"""
from __future__ import annotations

import numpy as np

from ..risk.manager import Proposal, RiskManager

MIN_MS = 60_000


def simulate(signals: list[dict], cfg_risk, payout: float, correlation: float | None = None) -> dict:
    """signals: 每条 {ts, symbol, horizon, direction(+1/-1), p_win, outcome(+1/-1/0)}，outcome 为到期实际方向。

    按时间顺序处理；同一分钟内先结算到期合约，再审批新信号（与实时流程一致）。
    """
    rm = RiskManager(cfg_risk, correlation)
    sigs = sorted(signals, key=lambda s: (s["ts"], s["symbol"], s["horizon"]))
    outcome_of: dict[int, int] = {}
    trades, rejected = [], {}
    equity_curve: list[tuple[int, float]] = []

    def settle_until(ts: int) -> None:
        for pos in rm.due(ts):
            pnl = rm.settle(pos, outcome_of.pop(pos.id))
            trades[pos.id - 1]["pnl"] = pnl
            equity_curve.append((pos.expiry_ts, rm.equity))

    for s in sigs:
        settle_until(s["ts"])
        pr = Proposal(s["symbol"], int(s["horizon"]), int(s["direction"]), float(s["p_win"]), payout, int(s["ts"]))
        dec = rm.evaluate(pr)
        if not dec.approved:
            for r in dec.reasons:
                key = r.split("（")[0].split(" (")[0].split("将达")[0]
                rejected[key] = rejected.get(key, 0) + 1
            continue
        pos = rm.open_position(pr, dec.stake)
        outcome_of[pos.id] = int(s["outcome"])
        trades.append({"id": pos.id, "ts": s["ts"], "symbol": s["symbol"], "horizon": s["horizon"], "direction": s["direction"],
                       "stake": dec.stake, "outcome": int(s["outcome"]), "pnl": None})
    settle_until(10**18)

    pnl = np.asarray([t["pnl"] for t in trades], dtype=np.float64)
    n = len(trades)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    days: dict[int, float] = {}
    for t in trades:
        d = (t["ts"] + t["horizon"] * MIN_MS) // 86_400_000
        days[d] = days.get(d, 0.0) + t["pnl"]
    daily = np.asarray(list(days.values())) if days else np.zeros(0)
    staked = float(sum(t["stake"] for t in trades))
    return {
        "payout": payout, "n_signals": len(sigs), "n_trades": n, "n_rejected": len(sigs) - n, "rejected_reasons": rejected,
        "wins": wins, "losses": losses, "ties": int(n - wins - losses),
        "win_rate": wins / max(1, wins + losses), "total_pnl": float(pnl.sum()), "total_staked": staked,
        "roi_on_staked": float(pnl.sum() / staked) if staked else 0.0, "max_drawdown": float(rm.max_drawdown),
        "n_days": len(days), "worst_day": float(daily.min()) if len(daily) else 0.0, "best_day": float(daily.max()) if len(daily) else 0.0,
        "positive_days_ratio": float((daily > 0).mean()) if len(daily) else 0.0,
        "daily_sharpe": float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(365)) if len(daily) > 5 else None,
        "equity_curve": _thin(equity_curve, 400), "trades": trades,
    }


def _thin(curve: list[tuple[int, float]], n: int) -> list[list]:
    if len(curve) <= n:
        return [[int(t), float(v)] for t, v in curve]
    idx = np.linspace(0, len(curve) - 1, n).astype(int)
    return [[int(curve[i][0]), float(curve[i][1])] for i in idx]


def stress_test(trades: list[dict], cfg_risk, payout: float, haircuts: list[float], n_paths: int = 1000, seed: int = 0) -> dict:
    """压力测试：对已执行交易按"天"做区块重采样（保留亏损聚集），并叠加胜率折减。

    胜率折减 h：把 h 比例的盈利单随机翻转为亏损单，模拟实盘胜率低于回测（延迟、滑点、模型衰减）。
    输出期末盈亏分布、最大回撤分布、触及每日亏损上限的概率。
    """
    if not trades:
        return {"note": "无交易"}
    rng = np.random.default_rng(seed)
    day = np.asarray([t["ts"] // 86_400_000 for t in trades])
    stake = np.asarray([t["stake"] for t in trades], dtype=np.float64)
    res = np.sign(np.asarray([t["pnl"] for t in trades], dtype=np.float64))
    uniq = np.unique(day)
    by_day = [np.where(day == d)[0] for d in uniq]
    limit = float(cfg_risk.daily_loss_limit_usdt)
    out = {}
    for h in haircuts:
        finals, dds, breaches = [], [], []
        for _ in range(n_paths):
            pick = rng.integers(0, len(uniq), len(uniq))
            seq = np.concatenate([by_day[di] for di in pick])
            r = res[seq].copy()
            if h > 0:
                r[(r > 0) & (rng.random(len(r)) < h)] = -1
            pnl = np.where(r > 0, stake[seq] * payout, np.where(r < 0, -stake[seq], 0.0))
            cum = np.cumsum(pnl)
            peak = np.maximum.accumulate(np.maximum(cum, 0.0))
            starts = np.cumsum([0] + [len(by_day[di]) for di in pick[:-1]])
            day_pnl = np.add.reduceat(pnl, starts)
            finals.append(float(cum[-1])); dds.append(float((peak - cum).max()))
            breaches.append(float((day_pnl <= -limit).mean()))
        f, d = np.asarray(finals), np.asarray(dds)
        out[f"haircut_{h:.2f}"] = {
            "win_rate_haircut": h, "pnl_p05": float(np.quantile(f, 0.05)), "pnl_p50": float(np.median(f)),
            "pnl_p95": float(np.quantile(f, 0.95)), "prob_loss": float((f < 0).mean()),
            "max_drawdown_p50": float(np.median(d)), "max_drawdown_p95": float(np.quantile(d, 0.95)),
            "daily_limit_breach_rate": float(np.mean(breaches)),
        }
    return out
