"""训练/评估引擎：单折拟合预测、样本外(OOF)预测、任务指标。"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from ..bundle import DataBundle
from ..models.base import Task
from ..models.zoo import make_candidate
from ..validation.splits import Fold
from ..validation.stats import prob_metrics

LN2 = float(np.log(2))


def spec_id(spec: dict) -> str:
    key = {k: spec.get(k) for k in ("family", "params", "tasks", "features", "members", "weights")}
    return spec["family"] + "_" + hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def task_eval(bundle: DataBundle, task: Task, rows: np.ndarray, p: np.ndarray) -> dict:
    """方向预测的概率指标（剔除平局与无效行）。skill = ln2 - logloss，>0 才优于抛硬币。"""
    y = bundle.symbols[task[0]].labels[task[1]][rows]
    ok = (y >= 0) & np.isfinite(p)
    if ok.sum() < 50:
        return {"n": int(ok.sum()), "logloss": float("nan")}
    m = prob_metrics(y[ok].astype(float), p[ok])
    m["skill_1e4"] = (LN2 - m["logloss"]) * 1e4
    return m


def fit_predict(spec: dict, bundle: DataBundle, fold: Fold, predict_rows: np.ndarray | None = None):
    """在一折上训练并预测。返回 (模型, {task: p}, 训练信息)。"""
    cand = make_candidate(spec)
    t0 = time.time()
    info = cand.fit(bundle, fold.train, fold.es)
    rows = fold.val if predict_rows is None else predict_rows
    preds = cand.predict(bundle, rows) if len(rows) else {}
    info = dict(info or {}) | {"fit_seconds": time.time() - t0}
    return cand, preds, info


def oof_predict(spec: dict, bundle: DataBundle, folds: list[Fold], on_fold=None) -> dict:
    """Walk-forward 样本外预测：每一折的预测都只来自在其之前的数据上训练的模型。"""
    tasks = [(s, int(h)) for s, h in spec["tasks"]]
    rows_all, fold_ids = [], []
    ps: dict[Task, list] = {t: [] for t in tasks}
    fold_metrics, seconds = [], 0.0
    for f in folds:
        _, preds, info = fit_predict(spec, bundle, f)
        seconds += info["fit_seconds"]
        rows_all.append(f.val)
        fold_ids.append(np.full(len(f.val), f.k, dtype=np.int16))
        fm = {"fold": f.k}
        for t in tasks:
            ps[t].append(preds[t])
            fm[f"{t[0]}_{t[1]}m"] = task_eval(bundle, t, f.val, preds[t])
        fold_metrics.append(fm)
        if on_fold:
            on_fold(f.k, fm)
    return {"rows": np.concatenate(rows_all), "fold": np.concatenate(fold_ids),
            "p": {t: np.concatenate(v) for t, v in ps.items()}, "fold_metrics": fold_metrics, "fit_seconds": seconds}


def save_oof(path, oof: dict) -> None:
    arrays = {"rows": oof["rows"], "fold": oof["fold"]}
    for (s, h), p in oof["p"].items():
        arrays[f"p__{s}__{h}"] = p.astype(np.float32)
    np.savez_compressed(path, **arrays)


def load_oof(path) -> dict:
    z = np.load(path)
    p = {}
    for k in z.files:
        if k.startswith("p__"):
            _, s, h = k.split("__")
            p[(s, int(h))] = z[k].astype(np.float64)
    return {"rows": z["rows"], "fold": z["fold"], "p": p}
