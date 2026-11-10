import numpy as np
import pandas as pd

from becai.signals.policy import Policy, breakeven_win_rate, optimize_policy


def _data(n, edge, seed=0):
    rng = np.random.default_rng(seed)
    p = np.clip(0.5 + edge * rng.standard_normal(n), 0.02, 0.98)       # 校准良好的概率
    y = (rng.uniform(size=n) < (p if edge > 0 else 0.5)).astype(np.int8)
    ts = np.arange(n, dtype=np.int64) * 60_000
    fold = (np.arange(n) * 6 // n).astype(np.int16)
    regime = pd.DataFrame({"vol": rng.choice(["低波动", "中波动", "高波动"], n), "trend": "震荡", "session": "亚洲时段", "weekend": "工作日"})
    return p, y, ts, fold, regime


def test_finds_high_win_rate_threshold_when_edge_exists(cfg):
    p, y, ts, fold, regime = _data(120_000, 0.15)
    pol, diag = optimize_policy(("BTCUSDT", 10), p, y, ts, fold, regime, cfg, 0.8)
    assert pol.status == "target_met" and pol.threshold is not None and pol.threshold > 0.55
    assert pol.validation["lcb"] >= 0.70 and pol.validation["coverage"] >= cfg.signal.min_coverage
    d, reason = pol.decide(np.array([0.9, 0.1, 0.52, np.nan]), regime.iloc[:4])
    assert list(d[:2]) == [1, -1] and d[2] == 0 and "置信度不足" in reason[2] and "数据不可用" in reason[3]


def test_reports_no_edge_on_pure_noise(cfg):
    p, y, ts, fold, regime = _data(120_000, 0.0)
    p = np.clip(0.5 + 0.1 * np.random.default_rng(5).standard_normal(len(y)), 0.02, 0.98)   # 自信但毫无信息量的预测
    pol, diag = optimize_policy(("BTCUSDT", 10), p, y, ts, fold, regime, cfg, 0.8)
    assert pol.status == "no_edge" and pol.threshold is None          # 不伪造结果：没有优势就不开仓
    d, reason = pol.decide(np.array([0.95]), regime.iloc[:1])
    assert d[0] == 0 and "未发现可交易优势" in reason[0]
    assert diag["breakeven"] == breakeven_win_rate(0.8)


def test_modest_edge_is_positive_ev_but_not_target(cfg):
    p, y, ts, fold, regime = _data(150_000, 0.04, seed=3)
    pol, _ = optimize_policy(("BTCUSDT", 10), p, y, ts, fold, regime, cfg, 0.8)
    assert pol.status in ("positive_ev", "no_edge")
    if pol.status == "positive_ev":
        assert breakeven_win_rate(0.8) < pol.validation["lcb"] < 0.70


def test_regime_filter_blocks_signals():
    regime = pd.DataFrame({"vol": ["高波动", "低波动"], "trend": ["震荡"] * 2, "session": ["亚洲时段"] * 2, "weekend": ["工作日"] * 2})
    pol = Policy(["BTCUSDT", 10], 0.6, {"vol": ["高波动"]}, "positive_ev", 0.8, {})
    d, reason = pol.decide(np.array([0.8, 0.8]), regime)
    assert list(d) == [0, 1] and "市场状态被过滤" in reason[0]
    assert Policy.from_dict(pol.to_dict()).excluded == {"vol": ["高波动"]}


def test_regime_filter_adopted_only_when_bucket_is_significantly_worse(cfg):
    p, y, ts, fold, regime = _data(150_000, 0.06, seed=7)
    pol, diag = optimize_policy(("BTCUSDT", 10), p, y, ts, fold, regime, cfg, 0.8)
    assert pol.excluded == {}                                   # 各状态表现相同：不应采用任何过滤器
    assert sum(t["adopted"] for t in diag["filters_tried"]) == 1
    rng = np.random.default_rng(11)
    bad = (regime["vol"] == "高波动").to_numpy()
    y2 = y.copy()
    y2[bad] = (rng.uniform(size=bad.sum()) < 0.5).astype(np.int8)   # 高波动状态下预测完全失效
    pol2, _ = optimize_policy(("BTCUSDT", 10), p, y2, ts, fold, regime, cfg, 0.8)
    assert pol2.excluded == {"vol": ["高波动"]}
