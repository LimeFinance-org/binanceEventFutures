import numpy as np
import pytest

from becai.calibration import Calibrator, forward_chain_calibrate
from becai.models.base import Candidate
from becai.models.zoo import make_candidate, select_features
from becai.search.engine import task_eval
from becai.validation.splits import make_timeline, walk_forward

TORCH = {"max_epochs": 2, "steps_per_epoch": 25, "batch_size": 256, "patience": 2, "amp": False, "es_max_samples": 4000,
         "d_model": 16, "n_layers": 1, "head_hidden": 32, "tab_hidden": 32, "seq_len": 16}


def _fold(bundle):
    tl = make_timeline(len(bundle.ts), 7, 60)
    return walk_forward(tl.dev_end, 1, 5, 20, 60, 0.1, 4)[0]


def _spec(bundle, family, **params):
    feats = [c for c in bundle.columns if bundle.meta[c]["scale"] <= 15 and bundle.meta[c]["source"] in ("index", "spot")][:40]
    tasks = [["BTCUSDT", 10]] if family in ("logreg", "lightgbm", "xgboost") else [["BTCUSDT", 10], ["BTCUSDT", 30], ["ETHUSDT", 10], ["ETHUSDT", 30]]
    return {"family": family, "tasks": tasks, "features": feats, "device": "cpu", "threads": 2, "seed": 1, "params": params}


@pytest.mark.parametrize("family,params", [
    ("logreg", {"C": 0.1}),
    ("lightgbm", {"n_estimators": 80, "num_leaves": 15, "min_data_in_leaf": 100, "learning_rate": 0.1}),
    ("xgboost", {"n_estimators": 80, "max_depth": 4, "eta": 0.1}),
    ("mlp", dict(TORCH, arch="mlp")),
    ("gru", dict(TORCH, arch="gru", scales=[1])),
    ("tcn", dict(TORCH, arch="tcn", scales=[1])),
    ("transformer", dict(TORCH, arch="transformer", scales=[1], patch=4, heads=2)),
    ("multiscale", dict(TORCH, arch="gru", scales=[1, 5])),
])
def test_model_learns_planted_signal_and_roundtrips(bundle, tmp_path, family, params):
    f = _fold(bundle)
    cand = make_candidate(_spec(bundle, family, **params))
    cand.fit(bundle, f.train, f.es)
    preds = cand.predict(bundle, f.val)
    assert set(preds) == set(cand.tasks)
    for t, p in preds.items():
        assert len(p) == len(f.val) and np.nanmin(p) >= 0 and np.nanmax(p) <= 1
    m = task_eval(bundle, ("BTCUSDT", 10), f.val, preds[("BTCUSDT", 10)])
    assert m["logloss"] < np.log(2) and m["accuracy"] > 0.53, m          # 合成数据里有真实信号，必须学得到
    cand.save(tmp_path / "m")
    again = Candidate.load(tmp_path / "m").predict(bundle, f.val[:500])
    np.testing.assert_allclose(again[("BTCUSDT", 10)], preds[("BTCUSDT", 10)][:500], rtol=1e-4, atol=1e-5, equal_nan=True)


def test_prediction_is_nan_during_warmup(bundle):
    f = _fold(bundle)
    cand = make_candidate(_spec(bundle, "logreg", C=0.1))
    cand.fit(bundle, f.train, f.es)
    p = cand.predict(bundle, np.arange(0, 50))[("BTCUSDT", 10)]
    assert np.isnan(p).all()


def test_feature_subset_selection(bundle):
    ranking = sorted(bundle.groups())
    all_cols = select_features(bundle.columns, bundle.meta, ranking, {})
    no_spot = select_features(bundle.columns, bundle.meta, ranking, {"use_spot": False})
    short = select_features(bundle.columns, bundle.meta, ranking, {"feat_max_scale": 15})
    assert len(all_cols) == len(bundle.columns)
    assert not any(bundle.meta[c]["source"] == "spot" for c in no_spot) and len(no_spot) < len(all_cols)
    assert all(bundle.meta[c]["scale"] <= 15 or bundle.meta[c]["source"] == "time" or bundle.meta[c]["family"] == "regime" for c in short)
    assert len(select_features(bundle.columns, bundle.meta, ranking, {"feat_top_groups": 3})) < 20


def test_calibration_fixes_overconfidence_without_peeking():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.35, 0.65, 40000)
    y = (rng.uniform(size=40000) < true_p).astype(np.int8)
    z = np.log(true_p / (1 - true_p)) * 3                      # 过度自信的模型
    p_raw = 1 / (1 + np.exp(-z))
    fold = np.repeat(np.arange(8), 5000)
    res = forward_chain_calibrate(p_raw, y, fold)
    assert res["best"] in ("platt", "isotonic") and res["scores"][res["best"]] < res["scores"]["none"] - 0.01
    assert np.isnan(res["p_cal"][fold == 0]).all()             # 第一折没有更早的数据可用于校准
    cal = Calibrator.from_dict(res["calibrator"].to_dict())
    np.testing.assert_allclose(cal.transform(p_raw[:100]), res["calibrator"].transform(p_raw[:100]))
    assert abs(cal.transform(np.array([0.5]))[0] - 0.5) < 0.03
