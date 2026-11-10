"""特征研究：分组重要性排序 + 数据源 / 周期 / 特征族的置换重要性。

回答"哪些数据源、哪些K线周期、哪些特征族有稳定预测价值"：
  * 细粒度分组 (source|family|scale) 用 LightGBM 增益在多折上的均值排序，供后续搜索做子集选择；
  * 数据源、周期、特征族三个维度用验证折上的"联合置换重要性"（打乱后对数损失的恶化量）评估，
    并统计跨折稳定性（多少折上为正）。
"""
from __future__ import annotations

import numpy as np

from ..bundle import DataBundle
from ..models.base import take_rows
from ..models.tabular import LightGBMCandidate
from ..validation.splits import Fold
from ..validation.stats import logloss


def screen_task(bundle: DataBundle, task: tuple[str, int], folds: list[Fold], threads: int, seed: int = 42,
                stride: int = 5, perm_repeats: int = 2, max_val: int = 60_000) -> dict:
    sym, h = task
    sd = bundle.symbols[sym]
    cols = bundle.columns
    groups = bundle.groups()
    dims = {"source": {}, "scale": {}, "family": {}}
    for j, c in enumerate(cols):
        m = bundle.meta[c]
        dims["source"].setdefault(m["source"], []).append(j)
        dims["family"].setdefault(m["family"], []).append(j)
        if m["source"] != "time":
            dims["scale"].setdefault(str(m["scale"]), []).append(j)
    gain = np.zeros((len(folds), len(cols)))
    perm: dict[str, dict[str, list[float]]] = {d: {k: [] for k in v} for d, v in dims.items()}
    base_ll = []
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(cols))
    for fi, f in enumerate(folds):
        n_tr = len(f.train[::stride])
        spec = {"family": "lightgbm", "tasks": [[sym, h]], "features": cols, "threads": threads, "seed": seed,
                "params": {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": max(20, min(800, n_tr // 60)), "feature_fraction": 0.5,
                           "bagging_fraction": 0.7, "lambda_l2": 10.0, "n_estimators": 600, "early_stopping": 40}}
        cand = LightGBMCandidate(spec)
        cand.fit(bundle, f.train[::stride], f.es[::stride])
        g = cand.gain_importance().astype(np.float64)
        gain[fi] = g / (g.sum() + 1e-12)
        rows = bundle.usable(sym, h, f.val, train=True)
        if len(rows) > max_val:
            rows = rows[:: len(rows) // max_val + 1]
        Xv = take_rows(sd.X, rows, all_idx)
        yv = sd.labels[h][rows].astype(np.float64)
        ll0 = logloss(yv, cand._predict(Xv))
        base_ll.append(ll0)
        for d, units in dims.items():
            for name, idx in units.items():
                deltas = []
                for _ in range(perm_repeats):
                    Xp = Xv.copy()
                    Xp[:, idx] = Xv[rng.permutation(len(Xv))][:, idx]      # 组内列联合置换，保留组内相关性
                    deltas.append(logloss(yv, cand._predict(Xp)) - ll0)
                perm[d][name].append(float(np.mean(deltas)))
    mean_gain = gain.mean(axis=0)
    group_rows = []
    for gname, idx in groups.items():
        per_fold = gain[:, idx].sum(axis=1)
        group_rows.append({"group": gname, "gain": float(per_fold.mean()), "stability": float((per_fold > 0).mean()),
                           "n_features": len(idx)})
    group_rows.sort(key=lambda r: -r["gain"])
    feat_rank = [cols[j] for j in np.argsort(-mean_gain)]

    def table(d: str) -> list[dict]:
        rows_ = [{"name": k, "delta_logloss_1e4": float(np.mean(v)) * 1e4, "positive_folds": float(np.mean(np.asarray(v) > 0)),
                  "n_features": len(dims[d][k])} for k, v in perm[d].items()]
        return sorted(rows_, key=lambda r: -r["delta_logloss_1e4"])

    return {
        "task": [sym, h], "baseline_logloss": base_ll, "baseline_skill_1e4": [(np.log(2) - x) * 1e4 for x in base_ll],
        "group_ranking": [r["group"] for r in group_rows], "groups": group_rows[:80], "feature_ranking": feat_rank,
        "top_features": [{"feature": cols[j], "gain": float(mean_gain[j])} for j in np.argsort(-mean_gain)[:40]],
        "by_source": table("source"), "by_scale": table("scale"), "by_family": table("family"),
    }
