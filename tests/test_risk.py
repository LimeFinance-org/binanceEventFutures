import pytest

from becai.backtest.simulator import simulate, stress_test
from becai.config import load_config
from becai.risk.manager import MIN_MS, Proposal, RiskManager
from becai.signals.policy import breakeven_win_rate

DAY = 86_400_000


@pytest.fixture()
def rc():
    return load_config().risk


def prop(ts=0, symbol="BTCUSDT", horizon=10, direction=1, p=0.7, payout=0.8):
    return Proposal(symbol, horizon, direction, p, payout, ts)


def test_payout_rule_100_wins_80_or_loses_100(rc):
    assert breakeven_win_rate(0.8) == pytest.approx(1 / 1.8)
    rc = load_config(overrides={"risk": {"stake_usdt": 100, "max_total_open_usdt": 1000, "max_symbol_open_usdt": 1000,
                                         "max_directional_usdt": 1000, "daily_loss_limit_usdt": 1000}}).risk
    rm = RiskManager(rc)
    a = rm.open_position(prop(), 100)
    assert rm.settle(a, +1) == pytest.approx(80.0)            # 猜对：100 -> 180，净赚 80
    b = rm.open_position(prop(ts=MIN_MS * 20), 100)
    assert rm.settle(b, -1) == pytest.approx(-100.0)          # 猜错：本金归零
    c = rm.open_position(prop(ts=MIN_MS * 40), 100)
    assert rm.settle(c, 0) == 0.0                             # 平局：退还本金
    assert rm.equity == pytest.approx(-20.0)


def test_rejects_without_reliable_odds_or_edge(rc):
    rm = RiskManager(rc)
    assert not rm.evaluate(prop(payout=None)).approved
    stale = Proposal("BTCUSDT", 10, 1, 0.7, 0.8, 0, payout_age_seconds=float(rc.max_odds_age_seconds) + 1)
    assert not rm.evaluate(stale).approved
    d = rm.evaluate(prop(p=0.55))                             # 0.55*0.8-0.45 < 0
    assert not d.approved and "期望收益不足" in d.reasons[0]
    assert rm.evaluate(prop(p=0.70)).approved


def test_position_count_budget_and_spacing_limits(rc):
    rm = RiskManager(rc)
    n = 0
    for i in range(20):
        pr = prop(ts=i * int(rc.min_spacing_minutes) * MIN_MS, symbol="BTCUSDT" if i % 2 else "ETHUSDT", horizon=30,
                  direction=1 if i % 4 < 2 else -1)
        d = rm.evaluate(pr)
        if d.approved:
            rm.open_position(pr, d.stake); n += 1
    assert n <= int(rc.max_open_positions)
    assert rm.open_stake() <= float(rc.max_total_open_usdt)
    assert any("间隔不足" in r for r in rm.evaluate(prop(ts=rm.open[-1].open_ts + MIN_MS, symbol=rm.open[-1].symbol, horizon=30,
                                                        direction=rm.open[-1].direction)).reasons)


def test_correlated_same_direction_exposure_is_capped():
    rc = load_config(overrides={"risk": {"stake_usdt": 10, "max_directional_usdt": 25, "max_total_open_usdt": 500,
                                         "max_symbol_open_usdt": 500, "max_open_positions": 50, "min_spacing_minutes": 0,
                                         "max_same_direction_streak": 99, "btc_eth_correlation": 0.9}}).risk
    rm = RiskManager(rc)
    for i, sym in enumerate(["BTCUSDT", "ETHUSDT"]):
        pr = prop(ts=i * MIN_MS, symbol=sym, horizon=30)
        assert rm.evaluate(pr).approved
        rm.open_position(pr, 10)
    d = rm.evaluate(prop(ts=3 * MIN_MS, symbol="BTCUSDT", horizon=10))     # 10 + 0.9*10 + 10 = 29 > 25
    assert not d.approved and any("同向敞口" in r for r in d.reasons)
    assert rm.evaluate(prop(ts=3 * MIN_MS, symbol="BTCUSDT", horizon=10, direction=-1)).approved   # 反向对冲允许


def test_daily_loss_limit_counts_open_premium_and_resets_at_utc_midnight():
    rc = load_config(overrides={"risk": {"stake_usdt": 10, "daily_loss_limit_usdt": 25, "max_consecutive_losses": 99,
                                         "min_spacing_minutes": 0, "max_same_direction_streak": 99}}).risk
    rm = RiskManager(rc)
    for i in range(2):
        pos = rm.open_position(prop(ts=i * 15 * MIN_MS), 10)
        rm.settle(pos, -1)
    assert rm.evaluate(prop(ts=40 * MIN_MS)).approved is False          # 已亏 20 + 新本金 10 > 25
    assert rm.evaluate(prop(ts=DAY + MIN_MS)).approved                  # 次日重置


def test_consecutive_losses_trigger_cooldown():
    rc = load_config(overrides={"risk": {"max_consecutive_losses": 2, "cooldown_minutes": 60, "daily_loss_limit_usdt": 1000,
                                         "min_spacing_minutes": 0, "max_same_direction_streak": 99}}).risk
    rm = RiskManager(rc)
    for i in range(2):
        rm.settle(rm.open_position(prop(ts=i * 15 * MIN_MS), 10), -1)
    blocked = rm.evaluate(prop(ts=40 * MIN_MS))
    assert not blocked.approved and "冷静期" in blocked.reasons[0]
    assert rm.evaluate(prop(ts=(15 + 10 + 61) * MIN_MS)).approved


def test_same_direction_streak_limit(rc):
    rc = load_config(overrides={"risk": {"max_same_direction_streak": 2, "min_spacing_minutes": 0, "max_total_open_usdt": 500,
                                         "max_symbol_open_usdt": 500, "max_directional_usdt": 500, "max_open_positions": 50}}).risk
    rm = RiskManager(rc)
    for i in range(2):
        rm.open_position(prop(ts=i * MIN_MS), 10)
    assert any("连续同向" in r for r in rm.evaluate(prop(ts=3 * MIN_MS)).reasons)


def test_simulation_respects_risk_and_is_not_improved_by_more_signals(rc):
    sigs = [{"ts": i * MIN_MS, "symbol": "BTCUSDT", "horizon": 10, "direction": 1, "p_win": 0.7, "outcome": 1 if i % 3 else -1}
            for i in range(600)]
    res = simulate(sigs, rc, 0.8)
    assert 0 < res["n_trades"] < len(sigs) and res["n_rejected"] == len(sigs) - res["n_trades"]
    assert res["wins"] + res["losses"] + res["ties"] == res["n_trades"]
    expected = sum(t["stake"] * 0.8 if t["pnl"] > 0 else -t["stake"] for t in res["trades"])
    assert res["total_pnl"] == pytest.approx(expected)
    st = stress_test(res["trades"], rc, 0.8, [0.0, 0.1], n_paths=50)
    assert st["haircut_0.10"]["pnl_p50"] < st["haircut_0.00"]["pnl_p50"]      # 胜率折减后收益必然更差
