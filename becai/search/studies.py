"""Optuna 超参数 + 特征子集 + 结构搜索。

  * 目标函数：轻量 walk-forward 各折样本外对数损失的均值（不是训练集准确率）；
  * 逐折上报并剪枝（MedianPruner），不做无意义的穷举；
  * JournalStorage 落盘：可断点续跑，可多进程并行；
  * 每个试验的配置与逐折指标写入实验记录库。
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import optuna

from ..bundle import DataBundle
from ..models.zoo import NEURAL, TABULAR, select_features, suggest_feature_params, suggest_params
from ..utils import get_logger, resolve_device
from ..validation.splits import Fold
from .engine import fit_predict, task_eval
from .experiment_db import ExperimentDB

log = get_logger("becai.search")
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def make_storage(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from optuna.storages.journal import JournalFileBackend
        return optuna.storages.JournalStorage(JournalFileBackend(str(path)))
    except ImportError:  # optuna < 4
        return optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(path)))


def study_defs(cfg) -> list[dict]:
    """根据配置展开所有 study：表格模型按任务；神经网络按作用域（共享多任务 / 单币种多期限）。"""
    defs = []
    horizons = [int(h) for h in cfg.horizons]
    for fam, fc in cfg.search.families.items():
        n = int(fc.get("trials", 0))
        if n <= 0:
            continue
        if fam in TABULAR:
            per = max(1, n // (len(cfg.symbols) * len(horizons)))
            for s in cfg.symbols:
                for h in horizons:
                    defs.append({"name": f"{fam}__{s}_{h}m", "family": fam, "tasks": [[s, h]], "trials": per, "device": fc.get("device", "cpu")})
        elif fam in NEURAL:
            scopes = list(cfg.search.torch_scopes)
            n_studies = (1 if "shared" in scopes and len(cfg.symbols) > 1 else 0) + (len(cfg.symbols) if "per_symbol" in scopes else 0)
            per = max(1, n // max(1, n_studies))
            if "shared" in scopes and len(cfg.symbols) > 1:
                defs.append({"name": f"{fam}__shared", "family": fam, "tasks": [[s, h] for s in cfg.symbols for h in horizons],
                             "trials": per, "device": fc.get("device", "auto")})
            if "per_symbol" in scopes:
                for s in cfg.symbols:
                    defs.append({"name": f"{fam}__{s}", "family": fam, "tasks": [[s, h] for h in horizons], "trials": per,
                                 "device": fc.get("device", "auto")})
    return defs


def build_spec(trial, sdef: dict, bundle: DataBundle, screens: dict, cfg) -> dict:
    fam = sdef["family"]
    spec = {"family": fam, "tasks": sdef["tasks"], "device": resolve_device(sdef.get("device", "auto")),
            "threads": int(cfg.search.cpu_threads), "seed": int(cfg.search.seed)}
    choices = [int(x) for x in (cfg.search.get("lookback_days_choices") or [])]
    if choices:     # 用多长的历史训练（0 = 全部可用历史）也由搜索决定
        spec["train_lookback_days"] = trial.suggest_categorical("train_lookback_days", choices)
    if fam in TABULAR:
        sc = screens[f"{sdef['tasks'][0][0]}_{sdef['tasks'][0][1]}m"]
        fp = suggest_feature_params(trial, [int(s) for s in cfg.features.scales], len(sc["group_ranking"]))
        spec["params"] = suggest_params(trial, fam)
        spec["features"] = select_features(bundle.columns, bundle.meta, sc["group_ranking"], fp)
        spec["feature_params"] = fp
    else:
        spec["params"] = suggest_params(trial, fam, dict(cfg.search.torch))
        # 多任务模型的表格输入：各任务特征排名的轮转合并（保证每个任务的头部特征都在）
        ranks = [screens[f"{s}_{h}m"]["feature_ranking"] for s, h in sdef["tasks"]]
        merged, seen = [], set()
        for i in range(max(len(r) for r in ranks)):
            for r in ranks:
                if i < len(r) and r[i] not in seen:
                    seen.add(r[i]); merged.append(r[i])
        spec["features"] = merged[: int(spec["params"]["n_tab"])]
    return spec


def make_objective(sdef: dict, bundle: DataBundle, folds: list[Fold], screens: dict, cfg, exp: ExperimentDB, run_id: str):
    tasks = [(s, int(h)) for s, h in sdef["tasks"]]

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        spec = build_spec(trial, sdef, bundle, screens, cfg)
        fold_scores, per_task = [], {f"{s}_{h}m": [] for s, h in tasks}
        status, value = "COMPLETE", None
        try:
            for i, f in enumerate(folds):
                _, preds, _ = fit_predict(spec, bundle, f)
                lls = []
                for t in tasks:
                    m = task_eval(bundle, t, f.val, preds[t])
                    per_task[f"{t[0]}_{t[1]}m"].append(m["logloss"])
                    lls.append(m["logloss"])
                fold_scores.append(float(np.mean(lls)))
                trial.report(float(np.mean(fold_scores)), i)
                if trial.should_prune():
                    status = "PRUNED"
                    raise optuna.TrialPruned()
            value = float(np.mean(fold_scores))
            if not np.isfinite(value):
                raise ValueError("目标值非有限数")
            trial.set_user_attr("spec", spec)
            trial.set_user_attr("task_logloss", {k: float(np.mean(v)) for k, v in per_task.items()})
            trial.set_user_attr("fold_scores", fold_scores)
            return value
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # noqa: BLE001  单个试验失败（如显存不足）不应中断整个搜索
            status = f"FAIL: {type(exc).__name__}: {exc}"[:300]
            log.warning("试验失败 %s #%d: %s", sdef["name"], trial.number, exc)
            raise optuna.TrialPruned()
        finally:
            exp.log(run_id, "search", sdef["name"], trial.number, sdef["family"], sdef["tasks"], status, value,
                    {"params": spec.get("params"), "feature_params": spec.get("feature_params"), "n_features": len(spec.get("features", [])),
                     "train_lookback_days": spec.get("train_lookback_days")},
                    {"fold_scores": fold_scores, "task_logloss": per_task}, time.time() - t0)

    return objective


def run_study(sdef: dict, bundle: DataBundle, folds: list[Fold], screens: dict, cfg, run_dir: Path, run_id: str,
              timeout: float | None, n_trials: int | None = None, on_trial=None) -> optuna.Study:
    storage = make_storage(run_dir / "optuna" / f"{sdef['name']}.log")
    sampler = optuna.samplers.TPESampler(seed=int(cfg.search.seed), multivariate=True, group=True, n_startup_trials=8)
    study = optuna.create_study(study_name=sdef["name"], storage=storage, load_if_exists=True, direction="minimize",
                                sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0))
    done = len([t for t in study.trials if t.state.is_finished()])
    target = int(n_trials if n_trials is not None else sdef["trials"])
    remaining = max(0, target - done)
    if remaining == 0:
        return study
    exp = ExperimentDB(run_dir / "experiments.sqlite")
    # MaxTrialsCallback 统计所有进程的试验总数：多进程并行时也不会超出预算
    cbs = [optuna.study.MaxTrialsCallback(target, states=None)]
    if on_trial:
        cbs.append(lambda st, tr: on_trial(sdef, st, tr))
    study.optimize(make_objective(sdef, bundle, folds, screens, cfg, exp, run_id), timeout=timeout,
                   callbacks=cbs, gc_after_trial=True, catch=())
    return study


def load_study(sdef: dict, run_dir: Path) -> optuna.Study | None:
    try:
        return optuna.load_study(study_name=sdef["name"], storage=make_storage(run_dir / "optuna" / f"{sdef['name']}.log"))
    except Exception:  # noqa: BLE001
        return None


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and "spec" in t.user_attrs]
