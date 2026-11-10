import numpy as np
import pytest

from becai.validation.holdout import HoldoutLedger, HoldoutLockError
from becai.validation.splits import check_no_overlap, final_fit_split, make_timeline, walk_forward
from becai.validation.stats import (block_bootstrap_mean, effective_sample_size, multiple_testing_threshold, non_overlapping,
                                    pbo_cscv, prob_metrics, wilson_ci, win_rate_summary)


def test_timeline_keeps_holdout_isolated():
    tl = make_timeline(100 * 1440, holdout_days=10, purge=90)
    assert tl.holdout_start == 90 * 1440 and tl.dev_end == tl.holdout_start - 90


def test_walk_forward_is_purged_and_ordered():
    purge = 90
    tl = make_timeline(120 * 1440, 10, purge)
    folds = walk_forward(tl.dev_end, n_folds=4, val_days=10, train_days=40, purge=purge, es_fraction=0.1, stride=3)
    assert len(folds) == 4
    for f in folds:
        assert check_no_overlap(f, purge)
        assert f.train.max() + purge <= f.es.min() and f.es.max() + purge <= f.val.min()
        assert f.val.max() < tl.dev_end
    assert all(a.val.max() < b.val.min() for a, b in zip(folds, folds[1:]))         # 验证窗口互不重叠且按时间前滚
    assert len(np.intersect1d(folds[0].val, folds[1].train)) > 0                    # 后面的折可以用前面的验证期做训练
    ff = final_fit_split(tl.dev_end, 40, purge, 0.1, 2)
    assert ff.es.max() < tl.dev_end <= tl.holdout_start and check_no_overlap(ff, purge)


def test_holdout_ledger_blocks_repeated_peeking(tmp_path):
    led = HoldoutLedger(tmp_path / "ledger.json")
    w = (1000, 2000)
    assert led.acquire(w, "run_a", "cfg", 1)["sequence"] == 1
    assert led.acquire(w, "run_a", "cfg", 1).get("reused")          # 同一运行断点恢复不重复计数
    with pytest.raises(HoldoutLockError):
        led.acquire(w, "run_b", "cfg", 1)
    forced = led.acquire(w, "run_b", "cfg", 1, force=True)
    assert forced["forced"] and forced["sequence"] == 2 and led.count(w) == 2
    assert led.acquire((2000, 3000), "run_c", "cfg", 1)["sequence"] == 1   # 新的测试窗口可以正常评估


def test_wilson_and_metrics():
    lo, hi = wilson_ci(70, 100)
    assert 0.60 < lo < 0.70 < hi < 0.79
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 20000)
    y = (rng.uniform(size=20000) < p).astype(float)
    m = prob_metrics(y, p)
    assert m["logloss"] < np.log(2) and m["auc"] > 0.6 and m["ece"] < 0.03


def test_block_bootstrap_is_wider_for_overlapping_outcomes():
    rng = np.random.default_rng(1)
    base = (rng.uniform(size=400) < 0.6).astype(float)
    win = np.repeat(base, 30)                       # 每个结果重复 30 分钟：模拟高度重叠的合约
    ts = np.arange(len(win)) * 60_000
    iid = wilson_ci(win.sum(), len(win))
    bb = block_bootstrap_mean(win, ts, block_minutes=120, n_boot=500)
    assert (bb["hi"] - bb["lo"]) > 3 * (iid[1] - iid[0])
    assert effective_sample_size(win, ts) < len(win) / 10
    s = win_rate_summary(win, ts, horizon=30, block_minutes=120, n_boot=300)
    assert s["non_overlap"]["n"] == 400 and s["n"] == len(win)


def test_non_overlapping_spacing():
    ts = np.array([0, 5, 10, 11, 25, 40]) * 60_000
    assert list(non_overlapping(ts, 10)) == [0, 2, 4, 5]


def test_pbo_detects_noise_vs_real_edge():
    rng = np.random.default_rng(2)
    noise = rng.standard_normal((12, 30))
    assert 0.25 < pbo_cscv(noise)["pbo"] < 0.75                  # 纯噪声：样本内最优在样本外约一半概率低于中位数
    real = noise * 0.2
    real[:, 0] += 2.0                                            # 一个真实更优的配置
    assert pbo_cscv(real)["pbo"] < 0.05
    assert multiple_testing_threshold(1000) > multiple_testing_threshold(1) > 1.6


def test_lookback_trim_and_early_stopping_cap():
    from becai.validation.splits import trim_lookback
    purge = 90
    tl = make_timeline(400 * 1440, 10, purge)
    f = walk_forward(tl.dev_end, 2, 10, None, purge, 0.1, 1, es_max_days=5)[-1]
    assert len(f.es) == 5 * 1440 and check_no_overlap(f, purge)          # 扩展窗口下早停集被限制为 5 天
    short = trim_lookback(f, 30)
    assert len(short.train) == 30 * 1440 and short.train.max() == f.train.max()
    assert (short.es == f.es).all() and (short.val == f.val).all() and check_no_overlap(short, purge)
    assert trim_lookback(f, 0) is f and trim_lookback(f, None) is f
