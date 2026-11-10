"""全局风险管理器 —— 独立于预测模型。

它只接收"开仓提议"（币种、期限、方向、胜率估计、赔率），不接触模型内部；
所有额度都是配置给定的上限，回测/优化流程不能通过放大额度来改善收益。

统一考虑：全部未到期合约、BTC/ETH 相关性、不同期限的持仓重叠、连续同向开仓的集中度、
单笔上限、总风险预算、每日亏损上限（含未到期本金，UTC 00:00 重置）、币安平台限制。
事件合约特性：本金即最大亏损，不能提前平仓。
"""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_MS = 60_000
DAY_MS = 86_400_000


@dataclass
class Position:
    id: int
    symbol: str
    horizon: int
    direction: int            # +1 HIGHER / -1 LOWER
    stake: float
    payout: float
    open_ts: int
    expiry_ts: int
    entry_price: float | None = None
    p_win: float | None = None


@dataclass
class Proposal:
    symbol: str
    horizon: int
    direction: int
    p_win: float              # 校准后的方向胜率估计
    payout: float | None      # 赔率；None = 无可靠赔率
    ts: int
    payout_age_seconds: float = 0.0


@dataclass
class Decision:
    approved: bool
    stake: float = 0.0
    reasons: list[str] = field(default_factory=list)
    ev: float | None = None


class RiskManager:
    def __init__(self, cfg_risk, correlation: float | None = None):
        self.c = cfg_risk
        self.corr = float(correlation if correlation is not None else cfg_risk.btc_eth_correlation)
        self.open: list[Position] = []
        self.daily_realized: dict[int, float] = {}     # UTC 日 -> 已实现盈亏
        self.consecutive_losses = 0
        self.cooldown_until = 0
        self.history: list[tuple[int, str, int]] = []   # (ts, symbol, direction) 最近开仓，用于连续同向检测
        self._next_id = 1
        self.equity = 0.0
        self.peak = 0.0
        self.max_drawdown = 0.0

    # ---------- 状态 ----------
    @staticmethod
    def day_of(ts: int) -> int:
        return ts // DAY_MS

    def open_stake(self) -> float:
        return sum(p.stake for p in self.open)

    def day_loss_exposure(self, ts: int) -> float:
        """当日已实现亏损 + 未到期本金（与币安平台口径一致：未到期订单的潜在亏损计入）。"""
        realized = self.daily_realized.get(self.day_of(ts), 0.0)
        return max(0.0, -realized) + self.open_stake()

    def directional_exposure(self, symbol: str, ts: int) -> float:
        """相关性加权的净方向敞口（USDT）。同币种权重 1，跨币种权重 = 相关系数。"""
        tot = 0.0
        for p in self.open:
            w = 1.0 if p.symbol == symbol else self.corr
            tot += p.direction * p.stake * w
        return tot

    def snapshot(self, ts: int) -> dict:
        return {
            "open_positions": len(self.open), "open_stake": self.open_stake(),
            "daily_realized": self.daily_realized.get(self.day_of(ts), 0.0),
            "daily_loss_exposure": self.day_loss_exposure(ts), "daily_loss_limit": float(self.c.daily_loss_limit_usdt),
            "consecutive_losses": self.consecutive_losses, "cooldown_until": self.cooldown_until,
            "equity": self.equity, "max_drawdown": self.max_drawdown,
            "positions": [p.__dict__ for p in self.open],
        }

    # ---------- 审批 ----------
    def evaluate(self, pr: Proposal) -> Decision:
        c = self.c
        reasons: list[str] = []
        stake = float(c.stake_usdt)
        if pr.payout is None:
            return Decision(False, 0.0, ["无可靠赔率：仅输出研究预测"])
        if pr.payout_age_seconds > float(c.max_odds_age_seconds):
            return Decision(False, 0.0, [f"赔率已过期 ({pr.payout_age_seconds:.0f}s)"])
        ev = pr.p_win * pr.payout - (1.0 - pr.p_win)
        if ev < float(c.min_edge):
            reasons.append(f"期望收益不足 ({ev:+.3f} < {float(c.min_edge):.3f})")
        if stake < float(c.min_order_usdt) or stake > float(c.max_order_usdt):
            reasons.append(f"单笔金额 {stake} 超出平台允许范围 [{c.min_order_usdt}, {c.max_order_usdt}]")
        if pr.ts < self.cooldown_until:
            reasons.append(f"连续亏损冷静期中（剩余 {(self.cooldown_until - pr.ts) // MIN_MS} 分钟）")
        if len(self.open) >= int(c.max_open_positions):
            reasons.append(f"未到期持仓数已达上限 ({c.max_open_positions})")
        if self.open_stake() + stake > float(c.max_total_open_usdt) + 1e-9:
            reasons.append(f"总未到期本金将超过预算 ({c.max_total_open_usdt} USDT)")
        sym_open = sum(p.stake for p in self.open if p.symbol == pr.symbol)
        if sym_open + stake > float(c.max_symbol_open_usdt) + 1e-9:
            reasons.append(f"{pr.symbol} 未到期本金将超过上限 ({c.max_symbol_open_usdt} USDT)")
        d_exp = self.directional_exposure(pr.symbol, pr.ts) + pr.direction * stake
        if abs(d_exp) > float(c.max_directional_usdt) + 1e-9:
            reasons.append(f"相关性加权同向敞口将达 {abs(d_exp):.1f} USDT，超过上限 {c.max_directional_usdt}")
        last_same = [p.open_ts for p in self.open if p.symbol == pr.symbol and p.horizon == pr.horizon]
        if last_same and pr.ts - max(last_same) < int(c.min_spacing_minutes) * MIN_MS:
            reasons.append(f"同一任务开仓间隔不足 {c.min_spacing_minutes} 分钟（持仓重叠过高）")
        win0 = pr.ts - int(c.streak_window_minutes) * MIN_MS
        recent = [d for (t, s, d) in self.history if t >= win0 and s == pr.symbol]
        streak = 0
        for d in reversed(recent):
            if d != pr.direction:
                break
            streak += 1
        if streak >= int(c.max_same_direction_streak):
            reasons.append(f"{pr.symbol} 已连续同向开仓 {streak} 次，风险过度集中")
        exposure = self.day_loss_exposure(pr.ts) + stake
        if exposure > float(c.daily_loss_limit_usdt) + 1e-9:
            reasons.append(f"将触及每日亏损上限 ({c.daily_loss_limit_usdt} USDT)")
        if exposure > float(c.platform_daily_loss_cap_usdt) + 1e-9:
            reasons.append("将触及平台每日亏损上限")
        return Decision(not reasons, stake if not reasons else 0.0, reasons, ev)

    def open_position(self, pr: Proposal, stake: float, entry_price: float | None = None) -> Position:
        pos = Position(self._next_id, pr.symbol, pr.horizon, pr.direction, stake, float(pr.payout), pr.ts,
                       pr.ts + pr.horizon * MIN_MS, entry_price, pr.p_win)
        self._next_id += 1
        self.open.append(pos)
        self.history.append((pr.ts, pr.symbol, pr.direction))
        if len(self.history) > 500:
            self.history = self.history[-250:]
        return pos

    # ---------- 结算 ----------
    def settle(self, pos: Position, outcome: int) -> float:
        """outcome: +1 结算价高于入场价 / -1 低于 / 0 平局（退本金）。返回盈亏。"""
        if outcome == 0:
            pnl = 0.0
        elif outcome == pos.direction:
            pnl = pos.stake * pos.payout
        else:
            pnl = -pos.stake
        self.open = [p for p in self.open if p.id != pos.id]
        day = self.day_of(pos.expiry_ts)
        self.daily_realized[day] = self.daily_realized.get(day, 0.0) + pnl
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= int(self.c.max_consecutive_losses):
                self.cooldown_until = pos.expiry_ts + int(self.c.cooldown_minutes) * MIN_MS
                self.consecutive_losses = 0
        elif pnl > 0:
            self.consecutive_losses = 0
        self.equity += pnl
        self.peak = max(self.peak, self.equity)
        self.max_drawdown = max(self.max_drawdown, self.peak - self.equity)
        return pnl

    def due(self, ts: int) -> list[Position]:
        return sorted([p for p in self.open if p.expiry_ts <= ts], key=lambda p: p.expiry_ts)
