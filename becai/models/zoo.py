"""模型库：工厂函数 + 各模型族的超参数搜索空间 + 特征子集搜索空间。

搜索空间只规定"合理的边界"，具体取值由 Optuna(TPE) 在样本外对数损失上自动选择。
"""
from __future__ import annotations

from .base import Candidate

TABULAR = ("logreg", "lightgbm", "xgboost")
NEURAL = ("mlp", "gru", "tcn", "transformer", "multiscale")
OPTIONAL_SOURCES = ("spot", "futures", "premium", "metrics", "cross")


def make_candidate(spec: dict) -> Candidate:
    fam = spec["family"]
    if fam in TABULAR:
        from .tabular import LightGBMCandidate, LogRegCandidate, XGBoostCandidate
        return {"logreg": LogRegCandidate, "lightgbm": LightGBMCandidate, "xgboost": XGBoostCandidate}[fam](spec)
    if fam in NEURAL:
        from .torch_models import TorchCandidate
        cand = TorchCandidate(spec)
        cand.family = fam
        return cand
    raise ValueError(f"未知模型族: {fam}")


# ---------------- 特征子集搜索 ----------------
def suggest_feature_params(trial, scales: list[int], n_groups: int) -> dict:
    """特征选择本身就是搜索的一部分：用多少个分组、最长用到哪个周期、启用哪些数据源。"""
    opts = sorted({min(n_groups, k) for k in (8, 16, 32, 64, 128, n_groups)})
    p = {"feat_top_groups": trial.suggest_categorical("feat_top_groups", opts),
         "feat_max_scale": trial.suggest_categorical("feat_max_scale", sorted({s for s in scales if s >= 15} | {max(scales)}))}
    for src in OPTIONAL_SOURCES:
        p[f"use_{src}"] = trial.suggest_categorical(f"use_{src}", [True, False])
    return p


def select_features(columns: list[str], meta: dict, group_ranking: list[str], fp: dict) -> list[str]:
    """根据特征子集参数从排序后的分组中选出特征列。指数价格与时间特征始终保留为候选。"""
    top = set(group_ranking[: int(fp.get("feat_top_groups", len(group_ranking)))])
    max_scale = int(fp.get("feat_max_scale", 10**9))
    out = []
    for c in columns:
        m = meta[c]
        if f"{m['source']}|{m['family']}|{m['scale']}" not in top:
            continue
        if m["source"] in OPTIONAL_SOURCES and not fp.get(f"use_{m['source']}", True):
            continue
        if m["source"] != "time" and m["family"] != "regime" and m["scale"] > max_scale:
            continue
        out.append(c)
    if not out:        # 兜底：至少保留排名第一的分组
        g0 = group_ranking[0]
        out = [c for c in columns if f"{meta[c]['source']}|{meta[c]['family']}|{meta[c]['scale']}" == g0]
    return out


# ---------------- 各模型族超参数空间 ----------------
def suggest_params(trial, family: str, torch_cfg: dict | None = None) -> dict:
    t = trial
    if family == "logreg":
        return {"C": t.suggest_float("C", 1e-4, 1.0, log=True)}
    if family == "lightgbm":
        return {
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": t.suggest_int("num_leaves", 7, 255, log=True),
            "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 100, 5000, log=True),
            "feature_fraction": t.suggest_float("feature_fraction", 0.3, 1.0),
            "bagging_fraction": t.suggest_float("bagging_fraction", 0.4, 1.0),
            "lambda_l1": t.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": t.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
            "min_gain_to_split": t.suggest_float("min_gain_to_split", 0.0, 1.0),
        }
    if family == "xgboost":
        return {
            "eta": t.suggest_float("eta", 0.01, 0.1, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 10),
            "min_child_weight": t.suggest_float("min_child_weight", 10.0, 2000.0, log=True),
            "subsample": t.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
            "lambda": t.suggest_float("lambda", 1e-2, 100.0, log=True),
            "alpha": t.suggest_float("alpha", 1e-3, 10.0, log=True),
            "gamma": t.suggest_float("gamma", 0.0, 2.0),
        }
    tc = dict(torch_cfg or {})
    base = {
        "lr": t.suggest_float("lr", 2e-4, 5e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.4),
        "head_hidden": t.suggest_categorical("head_hidden", [64, 128, 256]),
        "n_tab": t.suggest_categorical("n_tab", [0, 32, 64, 128]),
        "tab_hidden": t.suggest_categorical("tab_hidden", [64, 128, 256]),
        "max_epochs": int(tc.get("max_epochs", 30)), "steps_per_epoch": int(tc.get("steps_per_epoch", 400)),
        "batch_size": int(tc.get("batch_size", 4096)), "patience": int(tc.get("patience", 4)), "amp": bool(tc.get("amp", True)),
    }
    if family == "mlp":
        base["arch"] = "mlp"
        base["n_tab"] = t.suggest_categorical("n_tab_mlp", [32, 64, 128, 256])
        base["n_layers"] = 1
        return base
    base["d_model"] = t.suggest_categorical("d_model", [32, 64, 128, 256])
    base["n_layers"] = t.suggest_int("n_layers", 1, 4)
    if family == "multiscale":
        # 多时间尺度融合：融合哪些尺度、各尺度用什么编码器，都由搜索决定
        base["arch"] = t.suggest_categorical("arch", ["gru", "tcn", "transformer"])
        base["scales"] = list(t.suggest_categorical("scales", ["1,5", "1,15", "1,5,30", "1,5,15", "1,15,30", "5,30"]).split(","))
        base["scales"] = [int(s) for s in base["scales"]]
        base["seq_len"] = t.suggest_categorical("seq_len_ms", [32, 64])
    else:
        base["arch"] = family
        base["scales"] = [t.suggest_categorical("scale", [1, 3, 5])]
        base["seq_len"] = t.suggest_categorical("seq_len", [64, 128, 256])
    if base["arch"] == "transformer":
        base["patch"] = t.suggest_categorical("patch", [2, 4, 8])
        base["heads"] = t.suggest_categorical("heads", [2, 4, 8])
    return base
