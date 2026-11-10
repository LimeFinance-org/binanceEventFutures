"""测试夹具：生成带"已知可预测信号"的合成行情，全部离线，不访问网络。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from becai.config import load_config
from becai.utils import MINUTE_MS, date_to_ms, dump_json

START = "2024-01-01"
DAYS = 45


def synth_frames(seed: int, n: int, start_ms: int, beta: float = 2e-4, price0: float = 40000.0) -> dict[str, pd.DataFrame]:
    """指数价 = 随机游走 + beta × 滞后的主动买入失衡 x（AR(1)）。x 通过现货 taker_buy 比例可被观测到。"""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + 0.436 * eps[i]
    r = 5e-4 * rng.standard_normal(n)
    r[1:] += beta * x[:-1]
    close = price0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[price0], close[:-1]])
    spread = np.abs(5e-4 * rng.standard_normal(n)) * close * 0.5
    high, low = np.maximum(open_, close) + spread, np.minimum(open_, close) - spread
    t = start_ms + np.arange(n, dtype=np.int64) * MINUTE_MS
    px = pd.DataFrame({"open_time": t, "open": open_, "high": high, "low": low, "close": close})

    def trades(noise: float, scale: float) -> pd.DataFrame:
        v = scale * np.exp(0.5 * rng.standard_normal(n))
        buy = np.clip(0.5 + 0.2 * np.tanh(x + noise * rng.standard_normal(n)), 0.02, 0.98)
        c = close * (1 + 1e-5 * rng.standard_normal(n))
        d = px.copy()
        d["close"] = c
        d["high"], d["low"] = np.maximum(d["high"], c), np.minimum(d["low"], c)
        d["volume"], d["quote_volume"] = v, v * c
        d["count"] = np.round(50 * v / scale + 5)
        d["taker_buy_volume"], d["taker_buy_quote_volume"] = v * buy, v * buy * c
        return d

    prem = 1e-4 * rng.standard_normal(n)
    premium = pd.DataFrame({"open_time": t, "open": prem, "high": prem + 5e-5, "low": prem - 5e-5, "close": prem})
    ft = np.arange(start_ms, start_ms + n * MINUTE_MS, 480 * MINUTE_MS, dtype=np.int64)
    funding = pd.DataFrame({"time": ft, "rate": 1e-4 * (1 + 0.1 * rng.standard_normal(len(ft))), "interval_hours": 8.0})
    mt = np.arange(start_ms, start_ms + n * MINUTE_MS, 5 * MINUTE_MS, dtype=np.int64)
    oi = 1e5 * np.exp(np.cumsum(1e-3 * rng.standard_normal(len(mt))))
    metrics = pd.DataFrame({"time": mt, "oi": oi, "oi_value": oi * price0,
                            **{k: 1 + 0.1 * np.abs(rng.standard_normal(len(mt))) for k in ("ls_top_acct", "ls_top_pos", "ls_acct", "taker_ls")}})
    return {"index": px, "spot": trades(0.3, 10.0), "futures": trades(0.6, 100.0), "mark": px.copy(), "premium": premium,
            "funding": funding, "metrics": metrics}


def write_raw(root, symbol: str, frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        tcol = "open_time" if "open_time" in df else "time"
        d = root / "raw" / name / symbol
        d.mkdir(parents=True, exist_ok=True)
        month = pd.to_datetime(df[tcol], unit="ms").dt.strftime("%Y-%m")
        for m, part in df.groupby(month.to_numpy()):
            part.to_parquet(d / f"{m}.parquet", index=False)
            dump_json({"complete": True, "days": [], "source": "synthetic", "rows": len(part)}, d / f"{m}.json")


TEST_OVERRIDES = {
    "data": {"start": START, "spot_1s_sample_days": 0},
    "features": {"leak_auc_threshold": 0.85},      # 合成数据里故意埋了很强的信号
    "validation": {"holdout_days": 7, "embargo_minutes": 30,
                   "search": {"n_folds": 2, "val_days": 5, "train_days": 20, "train_stride": 3},
                   "final": {"n_folds": 4, "val_days": 5, "train_days": 20, "train_stride": 3},
                   "bootstrap": {"n_boot": 200, "block_minutes": 120}},
    "search": {"time_budget_hours": 0.2, "cpu_threads": 2, "top_k_per_task": 3, "parallel_workers": 1,
               "screen": {"stride": 4, "perm_repeats": 1},
               "families": {"logreg": {"trials": 4, "device": "cpu"}, "lightgbm": {"trials": 8, "device": "cpu"},
                            "xgboost": {"trials": 0}, "mlp": {"trials": 0}, "gru": {"trials": 1, "device": "cpu"},
                            "tcn": {"trials": 0}, "transformer": {"trials": 0}, "multiscale": {"trials": 0}},
               "torch_scopes": ["shared"],
               "torch": {"max_epochs": 2, "steps_per_epoch": 30, "batch_size": 256, "patience": 2, "amp": False, "es_max_samples": 5000,
                         "d_model_choices": [16], "n_layers_max": 1, "seq_len_choices": [16], "seq_len_multiscale_choices": [16]}},
    "signal": {"min_signals": 100, "min_effective_n": 30, "min_coverage": 0.005},
    "backtest": {"stress": {"n_paths": 50}},
    "live": {"buffer_minutes": 4500, "max_latency_seconds": 60},
}


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("becai")
    n = DAYS * 1440
    start_ms = date_to_ms(START)
    write_raw(root / "data", "BTCUSDT", synth_frames(1, n, start_ms, price0=40000.0))
    write_raw(root / "data", "ETHUSDT", synth_frames(2, n, start_ms, price0=2500.0))
    return root


@pytest.fixture(scope="session")
def cfg(workdir):
    ov = dict(TEST_OVERRIDES)
    ov["paths"] = {"data_dir": str(workdir / "data"), "runs_dir": str(workdir / "runs"), "registry_dir": str(workdir / "registry"),
                   "db_path": str(workdir / "data" / "becai.sqlite")}
    return load_config(overrides=ov)


@pytest.fixture(scope="session")
def panels(cfg):
    from becai.data.dataset import load_panels
    return load_panels(cfg)


@pytest.fixture(scope="session")
def bundle(cfg, panels):
    from becai.bundle import bundle_from_panels
    return bundle_from_panels(cfg, panels)
