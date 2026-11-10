import numpy as np

import becai.features.core as core
from becai.features.core import build_features, max_lookback
from becai.features.leakage import suspicious_auc, tail_parity_check, truncation_check


def _tail(panels, n=9000):
    return {s: p.iloc[-n:] for s, p in panels.items()}


def test_features_have_metadata_and_no_infinite_values(cfg, panels):
    fs = build_features(panels["BTCUSDT"].iloc[-6000:], {"ETHUSDT": panels["ETHUSDT"].iloc[-6000:]}, cfg)
    assert fs.X.dtype == np.float32 and fs.X.shape[1] == len(fs.columns) > 200
    assert not np.isinf(fs.X).any()
    assert all(set(fs.meta[c]) == {"source", "family", "scale"} for c in fs.columns)
    assert {"index", "spot", "futures", "premium", "metrics", "cross", "time"} <= {m["source"] for m in fs.meta.values()}
    assert np.isfinite(fs.X[-1]).mean() > 0.98           # 预热之后几乎没有缺失


def test_truncation_invariance_no_future_data(cfg, panels):
    rep = truncation_check(_tail(panels), cfg, n_cuts=2, tail=200)
    assert rep["passed"], rep["leaky_features"]


def test_leak_detector_catches_a_planted_future_feature(cfg, panels, monkeypatch):
    orig = core._trade_features

    def leaky(b, p, pre, source, lix, scales):
        orig(b, p, pre, source, lix, scales)
        if pre == "sp":
            b.add("leak_future_ret", lix.shift(-5) - lix, "index", "leak", 5)     # 故意使用未来 5 分钟价格

    monkeypatch.setattr(core, "_trade_features", leaky)
    rep = truncation_check(_tail(panels), cfg, n_cuts=1, tail=200)
    assert not rep["passed"]
    assert "leak_future_ret" in rep["leaky_features"]["BTCUSDT"]


def test_live_buffer_matches_full_history(cfg, panels):
    assert int(cfg.live.buffer_minutes) >= max_lookback(cfg)
    rep = tail_parity_check({s: p.iloc[-12000:] for s, p in panels.items()}, cfg)
    assert rep["passed"], rep["mismatch"]


def test_suspicious_auc_flags_label_leak(bundle):
    sd = bundle.symbols["BTCUSDT"]
    y = sd.labels[10]
    rows = np.where(sd.feat_ok & (y >= 0))[0][:20000]
    X = np.column_stack([np.asarray(sd.X[rows, :5]), sd.moves[10][rows]])    # 把未来涨跌幅当特征
    rep = suspicious_auc(X, y[rows], bundle.columns[:5] + ["future_move"])
    assert not rep["passed"] and rep["flagged"][0]["feature"] == "future_move"
    clean = suspicious_auc(np.asarray(sd.X[rows, :50]), y[rows], bundle.columns[:50], threshold=0.75)
    assert clean["passed"]
