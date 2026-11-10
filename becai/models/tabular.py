"""表格模型：逻辑回归（简单基线）、LightGBM、XGBoost（GPU）。"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..bundle import DataBundle
from .base import Candidate, Task, take_rows


class _Scaler:
    """稳健标准化：训练集均值/标准差，截尾到 ±5σ，缺失值置 0。"""

    def fit(self, X: np.ndarray) -> "_Scaler":
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0).astype(np.float32)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-8), self.std, 1.0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        np.clip(Z, -5, 5, out=Z)
        return np.nan_to_num(Z, nan=0.0, copy=False)


class TabularCandidate(Candidate):
    max_train_rows: int | None = None

    def __init__(self, spec: dict):
        super().__init__(spec)
        assert len(self.tasks) == 1, "表格模型为单任务"
        self.symbol, self.horizon = self.tasks[0]
        self.model = None
        self.info: dict = {}

    def _xy(self, bundle: DataBundle, rows: np.ndarray, cap: int | None = None):
        rows = bundle.usable(self.symbol, self.horizon, rows, train=True)
        if cap and len(rows) > cap:
            rows = rows[:: len(rows) // cap + 1]
        sd = bundle.symbols[self.symbol]
        return take_rows(sd.X, rows, self.feature_idx(bundle)), sd.labels[self.horizon][rows].astype(np.float32)

    def fit(self, bundle: DataBundle, train_rows: np.ndarray, es_rows: np.ndarray) -> dict:
        Xt, yt = self._xy(bundle, train_rows, self.max_train_rows)
        Xe, ye = self._xy(bundle, es_rows, 300_000)
        self.info = self._fit(Xt, yt, Xe, ye) or {}
        self.info.update({"n_train": int(len(yt)), "n_es": int(len(ye)), "n_features": int(Xt.shape[1])})
        return self.info

    def predict(self, bundle: DataBundle, rows: np.ndarray) -> dict[Task, np.ndarray]:
        sd = bundle.symbols[self.symbol]
        out = np.full(len(rows), np.nan)
        ok = sd.feat_ok[rows]
        if ok.any():
            out[ok] = self._predict(take_rows(sd.X, rows[ok], self.feature_idx(bundle)))
        return {(self.symbol, self.horizon): out}

    def _fit(self, Xt, yt, Xe, ye) -> dict:
        raise NotImplementedError

    def _predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def _save_state(self, d: Path) -> None:
        with open(d / "model.pkl", "wb") as f:
            pickle.dump({"model": self.model, "extra": getattr(self, "scaler", None), "info": self.info}, f)

    def _load_state(self, d: Path) -> None:
        with open(d / "model.pkl", "rb") as f:
            st = pickle.load(f)
        self.model, self.info = st["model"], st.get("info", {})
        if st.get("extra") is not None:
            self.scaler = st["extra"]


class LogRegCandidate(TabularCandidate):
    family = "logreg"
    max_train_rows = 400_000

    def _fit(self, Xt, yt, Xe, ye) -> dict:
        from sklearn.linear_model import LogisticRegression
        self.scaler = _Scaler().fit(Xt)
        self.model = LogisticRegression(C=float(self.params.get("C", 0.1)), max_iter=200)
        self.model.fit(self.scaler.transform(Xt), yt)
        return {}

    def _predict(self, X) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


class LightGBMCandidate(TabularCandidate):
    family = "lightgbm"

    def _fit(self, Xt, yt, Xe, ye) -> dict:
        import lightgbm as lgb
        p = self.params
        params = {
            "objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": self.seed,
            "num_threads": self.threads, "learning_rate": p.get("learning_rate", 0.03),
            "num_leaves": int(p.get("num_leaves", 31)), "min_data_in_leaf": int(p.get("min_data_in_leaf", 500)),
            "feature_fraction": p.get("feature_fraction", 0.7), "bagging_fraction": p.get("bagging_fraction", 0.7),
            "bagging_freq": 1, "lambda_l1": p.get("lambda_l1", 0.0), "lambda_l2": p.get("lambda_l2", 1.0),
            "max_bin": int(p.get("max_bin", 255)), "min_gain_to_split": p.get("min_gain_to_split", 0.0),
            "feature_pre_filter": False,
        }
        if self.device == "cuda" and p.get("lgb_gpu", False):
            params["device_type"] = "gpu"
        dt = lgb.Dataset(Xt, yt, free_raw_data=True)
        de = lgb.Dataset(Xe, ye, reference=dt)
        self.model = lgb.train(params, dt, num_boost_round=int(p.get("n_estimators", 1500)), valid_sets=[de],
                               callbacks=[lgb.early_stopping(int(p.get("early_stopping", 50)), verbose=False)])
        return {"best_iteration": int(self.model.best_iteration or 0), "es_logloss": float(self.model.best_score["valid_0"]["binary_logloss"])}

    def _predict(self, X) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.model.best_iteration or None, num_threads=self.threads)

    def gain_importance(self) -> np.ndarray:
        return self.model.feature_importance(importance_type="gain")

    def _save_state(self, d: Path) -> None:
        self.model.save_model(str(d / "model.txt"), num_iteration=self.model.best_iteration or None)

    def _load_state(self, d: Path) -> None:
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=str(d / "model.txt"))


class XGBoostCandidate(TabularCandidate):
    family = "xgboost"

    def _fit(self, Xt, yt, Xe, ye) -> dict:
        import xgboost as xgb
        p = self.params
        params = {
            "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "device": "cuda" if self.device == "cuda" else "cpu", "nthread": self.threads, "seed": self.seed,
            "eta": p.get("eta", 0.03), "max_depth": int(p.get("max_depth", 6)),
            "min_child_weight": p.get("min_child_weight", 50.0), "subsample": p.get("subsample", 0.7),
            "colsample_bytree": p.get("colsample_bytree", 0.7), "lambda": p.get("lambda", 1.0),
            "alpha": p.get("alpha", 0.0), "gamma": p.get("gamma", 0.0), "max_bin": int(p.get("max_bin", 256)),
        }
        dt = xgb.QuantileDMatrix(Xt, yt, max_bin=params["max_bin"])
        de = xgb.QuantileDMatrix(Xe, ye, ref=dt)
        self.model = xgb.train(params, dt, num_boost_round=int(p.get("n_estimators", 1500)), evals=[(de, "es")],
                               early_stopping_rounds=int(p.get("early_stopping", 50)), verbose_eval=False)
        return {"best_iteration": int(self.model.best_iteration), "es_logloss": float(self.model.best_score)}

    def _predict(self, X) -> np.ndarray:
        import xgboost as xgb
        self.model.set_param({"device": "cuda" if self.device == "cuda" else "cpu", "nthread": self.threads})
        return self.model.predict(xgb.DMatrix(X), iteration_range=(0, self.model.best_iteration + 1))

    def _save_state(self, d: Path) -> None:
        self.model.save_model(str(d / "model.ubj"))
        (d / "best_iteration.txt").write_text(str(self.model.best_iteration))

    def _load_state(self, d: Path) -> None:
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(str(d / "model.ubj"))
        self.model.best_iteration = int((d / "best_iteration.txt").read_text())
