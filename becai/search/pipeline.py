"""全自动训练流水线（可断点恢复）。

阶段：
  prepare   数据包构建 + 数据质量 + 泄漏检测 + 标签近似误差评估
  screen    特征分组/数据源/周期重要性研究
  search    各模型族的超参数 + 特征子集 + 结构搜索（受时间预算约束）
  finalize  候选模型完整 walk-forward 样本外预测，生成排行榜
  select    集成、"简单优先"模型选择、概率校准、信号策略优化、PBO
  ablation  数据源与周期消融实验
  holdout   在独立测试集上一次性评估（登记台账）+ 组合回测 + 压力测试
  register  登记模型版本，按规则决定是否上线

holdout 之前的所有阶段只接触开发区间 [0, dev_end) 的数据。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..bundle import DataBundle, build_bundle
from ..calibration import forward_chain_calibrate
from ..config import Config, task_name, tasks as cfg_tasks
from ..evaluation import evaluate_task, portfolio_backtest, rolling_correlation
from ..models.base import COMPLEXITY
from ..models.zoo import NEURAL, TABULAR
from ..registry import Registry, blend_logit
from ..signals.policy import optimize_policy
from ..utils import dump_json, get_logger, load_json, ms_to_iso, resolve_device
from ..validation.holdout import HoldoutLedger
from ..validation.splits import check_no_overlap, final_fit_split, make_timeline, purge_minutes, walk_forward
from ..validation.stats import logloss, paired_block_bootstrap, pbo_cscv, prob_metrics
from . import studies as st
from .engine import LN2, fit_predict, load_oof, oof_predict, save_oof, spec_id, task_eval
from .screen import screen_task

log = get_logger("becai.pipeline")
STAGES = ["prepare", "screen", "search", "finalize", "select", "ablation", "holdout", "register"]
STAGE_CN = {"prepare": "数据准备与泄漏检测", "screen": "特征研究", "search": "模型搜索", "finalize": "候选模型样本外验证",
            "select": "模型选择/校准/信号优化", "ablation": "消融实验", "holdout": "独立测试集评估", "register": "版本登记"}


class Pipeline:
    def __init__(self, cfg: Config, run_id: str | None = None, force_holdout: bool = False):
        self.cfg = cfg
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())
        self.dir = Path(cfg.path("runs_dir")) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = load_json(self.dir / "state.json", {"run_id": self.run_id, "stages": {}, "created": int(time.time() * 1000)})
        digest = cfg.digest("symbols", "horizons", "data", "features", "labels", "validation", "signal", "payout")
        if "config" not in self.state:
            self.state["config"], self.state["config_digest"] = cfg.to_dict(), digest
            self._save_state()
        elif self.state.get("config_digest") not in (None, digest):
            log.warning("运行 %s 创建后配置发生了变化（数据/特征/验证/信号相关）。已完成的阶段不会重算，"
                        "如需按新配置完整重跑，请使用新的运行ID。", self.run_id)
        self.force_holdout = force_holdout
        self._bundle: DataBundle | None = None
        self.tasks = cfg_tasks(cfg)
        self.purge = purge_minutes(cfg)

    # ---------- 基础设施 ----------
    def _save_state(self) -> None:
        dump_json(self.state, self.dir / "state.json")

    def progress(self, **kw) -> None:
        p = load_json(self.dir / "progress.json", {})
        p.update(kw)
        p["updated"] = int(time.time() * 1000)
        dump_json(p, self.dir / "progress.json")

    @property
    def bundle(self) -> DataBundle:
        if self._bundle is None:
            self._bundle = build_bundle(self.cfg, end=self.state.get("data_end_ms"))
            if "data_end_ms" not in self.state:       # 固定数据终点：断点恢复时不会因新数据改变切分
                self.state["data_end_ms"] = int(self._bundle.ts[-1])
                self._save_state()
        return self._bundle

    @property
    def timeline(self):
        return make_timeline(len(self.bundle.ts), int(self.cfg.validation.holdout_days), self.purge)

    def folds(self, kind: str):
        v = self.cfg.validation[kind]
        fs = walk_forward(self.timeline.dev_end, int(v.n_folds), int(v.val_days), v.train_days, self.purge,
                          float(self.cfg.validation.es_fraction), int(v.train_stride))
        assert fs, "开发区间数据不足，无法构建 walk-forward 折，请增加历史数据或减少折数/验证天数"
        assert all(check_no_overlap(f, self.purge) for f in fs)
        assert max(f.val.max() for f in fs) < self.timeline.dev_end <= self.timeline.holdout_start
        return fs

    def run(self, until: str | None = None) -> dict:
        for i, stage in enumerate(STAGES):
            if self.state["stages"].get(stage, {}).get("done"):
                continue
            log.info("===== 阶段 %d/%d: %s (%s) =====", i + 1, len(STAGES), stage, STAGE_CN[stage])
            self.progress(stage=stage, stage_cn=STAGE_CN[stage], stage_index=i + 1, n_stages=len(STAGES), detail="")
            t0 = time.time()
            getattr(self, f"stage_{stage}")()
            self.state["stages"][stage] = {"done": True, "seconds": time.time() - t0, "finished": int(time.time() * 1000)}
            self._save_state()
            if stage == until:
                break
        if all(self.state["stages"].get(s, {}).get("done") for s in STAGES):
            self.progress(stage="done", stage_cn="全部完成", detail="")
        return self.state

    # ---------- prepare ----------
    def stage_prepare(self) -> None:
        from ..data.dataset import load_panels
        from ..data.store import load_raw, quality_report
        from ..features.leakage import suspicious_auc, tail_parity_check, truncation_check
        from ..labels import approximation_report
        b, tl = self.bundle, self.timeline
        rep = {
            "rows": len(b.ts), "start": ms_to_iso(b.ts[0]), "end": ms_to_iso(b.ts[-1]),
            "dev_end": ms_to_iso(b.ts[tl.dev_end - 1]), "holdout_start": ms_to_iso(b.ts[tl.holdout_start]),
            "n_features": len(b.columns), "n_groups": len(b.groups()), "channels": b.channel_names, "purge_minutes": self.purge,
            "search_folds": [f.describe(b.ts) for f in self.folds("search")],
            "final_folds": [f.describe(b.ts) for f in self.folds("final")],
        }
        self.progress(detail="特征泄漏检测")
        tail = load_panels(self.cfg, start=int(b.ts[max(0, len(b.ts) - 14000)]), end=int(b.ts[-1]))
        rep["leakage"] = {"truncation": truncation_check(tail, self.cfg, n_cuts=3), "tail_parity": tail_parity_check(tail, self.cfg)}
        dev = np.arange(tl.dev_end)
        auc = {}
        for sym, h in self.tasks:
            auc[task_name(sym, h)] = suspicious_auc(np.asarray(b.symbols[sym].X[max(0, tl.dev_end - 400_000):tl.dev_end]),
                                                    b.symbols[sym].labels[h][dev][-400_000:], b.columns,
                                                    float(self.cfg.features.leak_auc_threshold))
        rep["leakage"]["single_feature_auc"] = auc
        rep["leakage"]["passed"] = bool(rep["leakage"]["truncation"]["passed"] and rep["leakage"]["tail_parity"]["passed"]
                                        and all(a["passed"] for a in auc.values()))
        if not rep["leakage"]["passed"]:
            dump_json(rep, self.dir / "prepare.json")
            raise RuntimeError("特征泄漏检测未通过，训练已中止。详见 prepare.json")
        self.progress(detail="标签近似误差评估")
        lo = int(b.ts[max(0, tl.dev_end - 180 * 1440)])
        dev_panels = load_panels(self.cfg, start=lo, end=int(b.ts[tl.dev_end - 1]))
        rep["label_approximation"] = {}
        for sym, p in dev_panels.items():
            s1 = load_raw(self.cfg, "spot_1s", sym) if int(self.cfg.data.get("spot_1s_sample_days", 0) or 0) > 0 else None
            rep["label_approximation"][sym] = approximation_report(p, [int(h) for h in self.cfg.horizons], s1 if s1 is not None and len(s1) else None)
        rep["data_quality"] = quality_report(self.cfg)
        dump_json(rep, self.dir / "prepare.json")

    # ---------- screen ----------
    def screens(self) -> dict:
        return {task_name(s, h): load_json(self.dir / "screen" / f"{task_name(s, h)}.json") for s, h in self.tasks}

    def stage_screen(self) -> None:
        folds = self.folds("search")
        sc = self.cfg.search.screen
        for i, (sym, h) in enumerate(self.tasks):
            out = self.dir / "screen" / f"{task_name(sym, h)}.json"
            if out.exists():
                continue
            self.progress(detail=f"特征研究 {task_name(sym, h)} ({i + 1}/{len(self.tasks)})")
            res = screen_task(self.bundle, (sym, h), folds, int(self.cfg.search.cpu_threads), int(self.cfg.search.seed),
                              int(sc.stride), int(sc.perm_repeats))
            dump_json(res, out)

    # ---------- search ----------
    def stage_search(self) -> None:
        defs = st.study_defs(self.cfg)
        dump_json(defs, self.dir / "studies.json")
        budget = float(self.cfg.search.time_budget_hours) * 3600
        folds, screens = self.folds("search"), self.screens()
        workers = int(self.cfg.search.parallel_workers)
        status = load_json(self.dir / "search_status.json", {"elapsed": 0.0, "studies": {}})

        def on_trial(sdef, study, trial):
            done = len([t for t in study.trials if t.state.is_finished()])
            try:
                best = study.best_value
            except ValueError:
                best = None
            status["studies"][sdef["name"]] = {"done": done, "total": sdef["trials"], "best": best,
                                               "best_skill_1e4": None if best is None else (LN2 - best) * 1e4, "family": sdef["family"]}
            dump_json(status, self.dir / "search_status.json")
            self.progress(detail=f"{sdef['name']}: {done}/{sdef['trials']} 次试验")

        for i, sdef in enumerate(defs):
            done_info = status["studies"].get(sdef["name"], {})
            if done_info.get("finished"):
                continue
            remaining_w = sum(d["trials"] for d in defs[i:] if not status["studies"].get(d["name"], {}).get("finished"))
            timeout = max(60.0, (budget - status["elapsed"]) * sdef["trials"] / max(1, remaining_w))
            if budget - status["elapsed"] <= 0:
                log.warning("搜索时间预算已用完，跳过 %s", sdef["name"])
                continue
            t0 = time.time()
            self.progress(detail=f"{sdef['name']} 开始（预算 {timeout / 60:.0f} 分钟）")
            # 只有跑在 GPU 上的 study 才多进程并行；CPU 模型本身已用满 cpu_threads，再并行只会互相争抢
            if workers > 1 and resolve_device(sdef.get("device", "auto")) == "cuda":
                procs = [subprocess.Popen([sys.executable, "-m", "becai.cli", "worker", "--run-id", self.run_id, "--study", sdef["name"],
                                           "--timeout", str(timeout)], env=os.environ.copy()) for _ in range(workers)]
                for p in procs:
                    p.wait()
                study = st.load_study(sdef, self.dir)
                if study is not None:
                    on_trial(sdef, study, None)
            else:
                st.run_study(sdef, self.bundle, folds, screens, self.cfg, self.dir, self.run_id, timeout, on_trial=on_trial)
            status["elapsed"] += time.time() - t0
            status["studies"].setdefault(sdef["name"], {"done": 0, "total": sdef["trials"], "family": sdef["family"]})["finished"] = True
            dump_json(status, self.dir / "search_status.json")

    def worker(self, study_name: str, timeout: float) -> None:
        """并行搜索的子进程入口。"""
        sdef = next(d for d in load_json(self.dir / "studies.json") if d["name"] == study_name)
        st.run_study(sdef, self.bundle, self.folds("search"), self.screens(), self.cfg, self.dir, self.run_id, timeout)

    # ---------- finalize ----------
    def _pick_candidates(self) -> dict:
        saved = load_json(self.dir / "candidates.json")
        if saved:
            return saved
        defs = load_json(self.dir / "studies.json", [])
        per_task: dict[str, list] = {task_name(s, h): [] for s, h in self.tasks}
        for sdef in defs:
            study = st.load_study(sdef, self.dir)
            if study is None:
                continue
            trials = st.completed_trials(study)
            for s, h in sdef["tasks"]:
                tn = task_name(s, h)
                scored = [(t.user_attrs["task_logloss"][tn], t) for t in trials if np.isfinite(t.user_attrs["task_logloss"].get(tn, np.nan))]
                if scored:
                    score, tr = min(scored, key=lambda x: x[0])
                    per_task[tn].append({"score": score, "study": sdef["name"], "family": sdef["family"], "trial": tr.number,
                                         "spec": tr.user_attrs["spec"], "n_trials": len(study.trials)})
        cands: dict = {}
        k = int(self.cfg.search.top_k_per_task)
        for tn, lst in per_task.items():
            lst.sort(key=lambda c: c["score"])
            chosen = lst[:k]
            simple = [c for c in lst if c["family"] == "logreg"]
            if simple and simple[0] not in chosen:
                chosen.append(simple[0])          # 简单基线必须参与完整评估，用于"复杂模型需证明更优"
            for c in chosen:
                cid = spec_id(c["spec"])
                cands.setdefault(cid, {"id": cid, "spec": c["spec"], "family": c["family"], "study": c["study"], "trial": c["trial"],
                                       "search_score": {}, "for_tasks": []})
                cands[cid]["search_score"][tn] = c["score"]
                cands[cid]["for_tasks"].append(tn)
        if not cands:
            raise RuntimeError("搜索阶段没有产生任何完成的试验，请检查时间预算与日志")
        dump_json(cands, self.dir / "candidates.json")
        return cands

    def stage_finalize(self) -> None:
        cands, folds = self._pick_candidates(), self.folds("final")
        (self.dir / "oof").mkdir(exist_ok=True)
        for i, (cid, c) in enumerate(cands.items()):
            if (self.dir / "oof" / f"{cid}.json").exists():
                continue
            self.progress(detail=f"样本外验证 {cid} ({i + 1}/{len(cands)})")
            oof = oof_predict(c["spec"], self.bundle, folds,
                              on_fold=lambda k, fm, cid=cid, i=i: self.progress(detail=f"样本外验证 {cid} ({i + 1}/{len(cands)}) 第 {k + 1}/{len(folds)} 折"))
            save_oof(self.dir / "oof" / f"{cid}.npz", oof)
            dump_json({"id": cid, "fold_metrics": oof["fold_metrics"], "fit_seconds": oof["fit_seconds"]}, self.dir / "oof" / f"{cid}.json")
        dump_json(self._leaderboard(cands), self.dir / "leaderboard.json")

    def _leaderboard(self, cands: dict) -> dict:
        b = self.bundle
        board: dict[str, list] = {}
        for sym, h in self.tasks:
            tn, rows_ = task_name(sym, h), []
            for cid, c in cands.items():
                if [sym, h] not in c["spec"]["tasks"]:
                    continue
                oof, info = load_oof(self.dir / "oof" / f"{cid}.npz"), load_json(self.dir / "oof" / f"{cid}.json")
                m = task_eval(b, (sym, h), oof["rows"], oof["p"][(sym, h)])
                fl = [fm[tn]["logloss"] for fm in info["fold_metrics"]]
                rows_.append({
                    "id": cid, "family": c["family"], "study": c["study"], "complexity": COMPLEXITY.get(c["family"], 9),
                    "n_tasks": len(c["spec"]["tasks"]), "n_features": len(c["spec"].get("features", [])),
                    "oof_logloss": m.get("logloss"), "oof_skill_1e4": m.get("skill_1e4"), "oof_auc": m.get("auc"),
                    "oof_accuracy": m.get("accuracy"), "oof_ece": m.get("ece"), "oof_brier": m.get("brier"),
                    "fold_skill_1e4": [(LN2 - x) * 1e4 for x in fl], "fold_skill_std": float(np.std([(LN2 - x) * 1e4 for x in fl])),
                    "folds_beating_coin": float(np.mean([x < LN2 for x in fl])),
                    "search_logloss": c["search_score"].get(tn), "fit_seconds": info["fit_seconds"],
                    "params": c["spec"].get("params"), "feature_params": c["spec"].get("feature_params"),
                })
            rows_.sort(key=lambda r: (r["oof_logloss"] is None, r["oof_logloss"]))
            board[tn] = rows_
        return board

    # ---------- select ----------
    @staticmethod
    def _caruana(P: np.ndarray, y: np.ndarray, max_members: int, iters: int = 20) -> np.ndarray:
        """贪心集成选择（有放回），在 logit 空间平均。返回权重。"""
        n = P.shape[1]
        Z = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
        counts, total, best = np.zeros(n), np.zeros(len(y)), float("inf")
        for it in range(iters):
            scores = []
            for j in range(n):
                if counts[j] == 0 and (counts > 0).sum() >= max_members:
                    scores.append(float("inf")); continue
                z = (total + Z[:, j]) / (it + 1)
                scores.append(logloss(y, 1 / (1 + np.exp(-z))))
            j = int(np.argmin(scores))
            if scores[j] >= best - 1e-7:
                break
            best = scores[j]; counts[j] += 1; total += Z[:, j]
        return counts / counts.sum() if counts.sum() else np.eye(n)[0]

    def stage_select(self) -> None:
        b, cands = self.bundle, self._pick_candidates()
        payout = float(self.cfg.payout.fixed)
        block = int(self.cfg.validation.bootstrap.block_minutes)
        (self.dir / "selection").mkdir(exist_ok=True)
        for sym, h in self.tasks:
            tn = task_name(sym, h)
            self.progress(detail=f"模型选择 {tn}")
            ids = [cid for cid, c in cands.items() if [sym, h] in c["spec"]["tasks"]]
            oofs = {cid: load_oof(self.dir / "oof" / f"{cid}.npz") for cid in ids}
            rows, fold = oofs[ids[0]]["rows"], oofs[ids[0]]["fold"]
            P = np.column_stack([oofs[cid]["p"][(sym, h)] for cid in ids])
            y = b.symbols[sym].labels[h][rows]
            ts = b.ts[rows]
            ok = (y >= 0) & np.isfinite(P).all(axis=1)
            fold_ids = np.unique(fold)
            half = fold_ids[: max(1, len(fold_ids) // 2)]
            A, B = ok & np.isin(fold, half), ok & ~np.isin(fold, half)
            if B.sum() < 1000:
                B = ok
            yf = y.astype(np.float64)
            # 集成：权重在前半折上确定，在后半折上与单模型同台比较
            w = self._caruana(P[A], yf[A], int(self.cfg.search.ensemble.max_members)) if A.sum() > 1000 and len(ids) > 1 else np.eye(len(ids))[0]
            entries = [{"name": cid, "family": cands[cid]["family"], "complexity": COMPLEXITY.get(cands[cid]["family"], 9),
                        "members": [cid], "weights": [1.0], "p": P[:, j]} for j, cid in enumerate(ids)]
            if (w > 0).sum() > 1:
                mem = [ids[j] for j in np.where(w > 0)[0]]
                pe = np.full(len(rows), np.nan)
                pe[ok] = blend_logit([P[ok][:, j] for j in np.where(w > 0)[0]], list(w[w > 0]))
                entries.append({"name": "ensemble", "family": "ensemble", "complexity": COMPLEXITY["ensemble"], "members": mem,
                                "weights": [float(x) for x in w[w > 0]], "p": pe})

            def ll_i(p):
                q = np.clip(p[B], 1e-6, 1 - 1e-6)
                return -(yf[B] * np.log(q) + (1 - yf[B]) * np.log(1 - q))

            for e in entries:
                e["select_logloss"] = float(ll_i(e["p"]).mean())
            # 简单优先：按复杂度从低到高，更复杂者必须显著更优才替换
            levels: dict[int, dict] = {}
            for e in entries:
                if e["complexity"] not in levels or e["select_logloss"] < levels[e["complexity"]]["select_logloss"]:
                    levels[e["complexity"]] = e
            order = [levels[k] for k in sorted(levels)]
            chosen, comparisons = order[0], []
            for cand in order[1:]:
                test = paired_block_bootstrap(ll_i(chosen["p"]) - ll_i(cand["p"]), ts[B], max(block, h), 1000,
                                              float(self.cfg.search.simplicity.confidence))
                comparisons.append({"simple": chosen["name"], "complex": cand["name"], "improvement_1e4": test["mean"] * 1e4,
                                    "lower_bound_1e4": test["lower"] * 1e4, "adopt_complex": test["significant"]})
                if test["significant"]:
                    chosen = cand
            fc = forward_chain_calibrate(chosen["p"], y, fold)
            policy, diag = optimize_policy((sym, h), fc["p_cal"], y, ts, fold, b.symbols[sym].regime.iloc[rows], self.cfg, payout)
            ev = ok & np.isfinite(fc["p_cal"])
            perf = np.array([[-logloss(yf[ok & (fold == k)], P[ok & (fold == k), j]) for j in range(len(ids))] for k in fold_ids])
            sel = {
                "task": [sym, h],
                "chosen": {k: chosen[k] for k in ("name", "family", "complexity", "members", "weights", "select_logloss")},
                "entries": [{k: e[k] for k in ("name", "family", "complexity", "members", "weights", "select_logloss")} for e in entries],
                "simplicity_comparisons": comparisons,
                "coin_flip_logloss": LN2,
                "calibration": {"method": fc["best"], "scores": fc["scores"], "calibrator": fc["calibrator"].to_dict()},
                "oof_raw": prob_metrics(yf[ok], chosen["p"][ok]), "oof_calibrated": prob_metrics(yf[ev], fc["p_cal"][ev]) if ev.sum() > 100 else {},
                "policy": policy.to_dict(), "policy_diagnostics": diag,
                "pbo": pbo_cscv(perf) if len(ids) > 1 else {"pbo": None, "note": "只有一个候选"},
            }
            dump_json(sel, self.dir / "selection" / f"{tn}.json")

    # ---------- ablation ----------
    def stage_ablation(self) -> None:
        b, cands, folds = self.bundle, self._pick_candidates(), self.folds("search")
        board = load_json(self.dir / "leaderboard.json")
        (self.dir / "ablation").mkdir(exist_ok=True)
        for sym, h in self.tasks:
            tn = task_name(sym, h)
            out = self.dir / "ablation" / f"{tn}.json"
            if out.exists():
                continue
            gbdt = [r for r in board[tn] if r["family"] in ("lightgbm", "xgboost")]
            if not gbdt:
                dump_json({"task": [sym, h], "note": "没有树模型候选，跳过消融"}, out)
                continue
            base_spec = cands[gbdt[0]["id"]]["spec"]
            feats = base_spec["features"]
            meta = b.meta

            def score(features: list[str]) -> float | None:
                if len(features) < 2:
                    return None
                spec = dict(base_spec, features=features)
                return float(np.mean([task_eval(b, (sym, h), f.val, fit_predict(spec, b, f)[1][(sym, h)])["logloss"] for f in folds]))

            self.progress(detail=f"消融实验 {tn}")
            base = score(feats)
            rows_ = []
            for src in sorted({meta[c]["source"] for c in feats}):
                keep = [c for c in feats if meta[c]["source"] != src]
                s = score(keep)
                if s is not None:
                    rows_.append({"type": "去掉数据源", "name": src, "n_features": len(keep), "delta_logloss_1e4": (s - base) * 1e4})
            scales = sorted({meta[c]["scale"] for c in feats if meta[c]["source"] != "time"})
            for m in [x for x in (5, 15, 60, 240) if x < max(scales, default=0)]:
                keep = [c for c in feats if meta[c]["source"] == "time" or meta[c]["scale"] <= m]
                s = score(keep)
                if s is not None:
                    rows_.append({"type": "只用不超过该周期", "name": f"<={m}分钟", "n_features": len(keep), "delta_logloss_1e4": (s - base) * 1e4})
            for m in [x for x in (15, 60, 240) if x <= max(scales, default=0)]:
                keep = [c for c in feats if meta[c]["source"] == "time" or meta[c]["scale"] >= m]
                s = score(keep)
                if s is not None:
                    rows_.append({"type": "只用不低于该周期", "name": f">={m}分钟", "n_features": len(keep), "delta_logloss_1e4": (s - base) * 1e4})
            dump_json({"task": [sym, h], "base_model": gbdt[0]["id"], "base_logloss": base, "base_skill_1e4": (LN2 - base) * 1e4,
                       "note": "delta>0 表示去掉后变差（该部分有贡献）；delta<=0 表示去掉无损甚至更好（可删除）", "rows": rows_}, out)

    # ---------- holdout ----------
    def stage_holdout(self) -> None:
        from ..report import write_report
        b, tl, cands = self.bundle, self.timeline, self._pick_candidates()
        window = (int(b.ts[tl.holdout_start]), int(b.ts[-1]))
        ledger = HoldoutLedger(Path(self.cfg.path("runs_dir")) / "holdout_ledger.json")
        entry = ledger.acquire(window, self.run_id, self.cfg.digest("features", "labels", "validation", "search", "signal"),
                               int(self.cfg.validation.max_holdout_evaluations), self.force_holdout)
        v = self.cfg.validation.final
        fold = final_fit_split(tl.dev_end, v.train_days, self.purge, float(self.cfg.validation.es_fraction), int(v.train_stride))
        hold = np.arange(tl.holdout_start, len(b.ts), dtype=np.int64)
        sels = {task_name(s, h): load_json(self.dir / "selection" / f"{task_name(s, h)}.json") for s, h in self.tasks}
        board = load_json(self.dir / "leaderboard.json")
        need = {m for s in sels.values() for m in s["chosen"]["members"]}
        baseline_ids = {}
        for tn, rows_ in board.items():
            simple = [r for r in rows_ if r["family"] == "logreg"]
            if simple:
                baseline_ids[tn] = simple[0]["id"]
        vdir = self.dir / "version"
        (self.dir / "holdout").mkdir(exist_ok=True)
        preds: dict[str, dict] = {}
        for i, cid in enumerate(sorted(need | set(baseline_ids.values()))):
            cache = self.dir / "holdout" / f"{cid}.npz"
            if cache.exists() and (cid not in need or (vdir / "models" / cid / "spec.json").exists()):
                preds[cid] = load_oof(cache)["p"]
                continue
            self.progress(detail=f"训练最终模型 {cid} ({i + 1})")
            cand, p, _ = fit_predict(cands[cid]["spec"], b, fold, predict_rows=hold)
            if cid in need:
                cand.save(vdir / "models" / cid)
            save_oof(cache, {"rows": hold, "fold": np.zeros(len(hold), dtype=np.int16), "p": p})
            preds[cid] = p
        report = {"run_id": self.run_id, "window": [ms_to_iso(window[0]), ms_to_iso(window[1])], "ledger": entry,
                  "train_range": [ms_to_iso(b.ts[fold.train[0]]), ms_to_iso(b.ts[fold.es[-1]])], "tasks": {}}
        all_signals: list[dict] = []
        from ..calibration import Calibrator
        from ..signals.policy import Policy
        for sym, h in self.tasks:
            tn, sel = task_name(sym, h), sels[task_name(sym, h)]
            raw = blend_logit([preds[m][(sym, h)] for m in sel["chosen"]["members"]], sel["chosen"]["weights"])
            p_cal = Calibrator.from_dict(sel["calibration"]["calibrator"]).transform(raw)
            policy = Policy.from_dict(sel["policy"])
            ss = load_json(self.dir / "search_status.json", {}).get("studies", {})
            n_trials = sum(int(sd.get("done", 0)) for name, sd in ss.items()
                           if name.endswith(f"__{tn}") or name.endswith("__shared") or name.endswith(f"__{sym}"))
            rep, sigs = evaluate_task(b, (sym, h), hold, p_cal, policy, self.cfg, max(1, n_trials))
            rep["model"] = sel["chosen"]
            rep["validation_expectation"] = sel["policy"].get("validation", {})
            if tn in baseline_ids:
                rep["simple_baseline"] = task_eval(b, (sym, h), hold, preds[baseline_ids[tn]][(sym, h)]) | {"id": baseline_ids[tn]}
            rep["coin_flip_logloss"] = LN2
            report["tasks"][tn] = rep
            all_signals.extend(sigs)
            dump_json({"task": [sym, h], "members": sel["chosen"]["members"], "weights": sel["chosen"]["weights"],
                       "calibrator": sel["calibration"]["calibrator"], "policy": sel["policy"]}, vdir / "tasks" / f"{tn}.json")
        self.progress(detail="组合回测与压力测试")
        report["portfolio"] = portfolio_backtest(all_signals, self.cfg, rolling_correlation(b, hold))
        dump_json(report, self.dir / "holdout_report.json")
        write_report(self.dir)

    # ---------- register ----------
    def stage_register(self) -> None:
        rep = load_json(self.dir / "holdout_report.json")
        reg = Registry(self.cfg.path("registry_dir"))
        summary = {tn: {"logloss": r.get("direction_all", {}).get("logloss"), "accuracy": r.get("direction_all", {}).get("accuracy"),
                        "signals": r["signals"].get("n"), "win_rate": r["signals"].get("win_rate"),
                        "win_rate_ci": r["signals"].get("block_bootstrap_ci"), "status": r["policy"]["status"],
                        "model": r["model"]["name"], "family": r["model"]["family"]} for tn, r in rep["tasks"].items()}
        b = self.bundle
        meta = {"version": "v_" + self.run_id.replace("run_", ""), "run_id": self.run_id, "holdout_window": rep["window"],
                "train_range": rep["train_range"], "data_end_ms": int(b.ts[-1]), "summary": summary, "columns": b.columns,
                "tie_rate": {tn: r.get("tie_rate", 0.0) for tn, r in rep["tasks"].items()},     # 实时输出"持平概率"用
                "portfolio": {k: rep["portfolio"]["main"].get(k) for k in ("n_trades", "win_rate", "total_pnl", "max_drawdown")},
                "config": self.cfg.to_dict()}
        version = reg.add(self.dir / "version", meta)
        champ = reg.champion()
        decision = {"version": version, "previous": champ}
        new_ll = float(np.nanmean([s["logloss"] or np.nan for s in summary.values()]))
        if champ is None:
            reg.promote(version, "首个版本")
            decision |= {"promoted": True, "reason": "首个版本"}
        elif not self.cfg.scheduler.promote_requires_improvement:
            reg.promote(version, "配置为始终上线新版本")
            decision |= {"promoted": True, "reason": "配置为始终上线新版本"}
        else:
            # 冠军/挑战者：旧版本在新测试窗口上同样是样本外，直接同窗口比较对数损失
            try:
                from ..registry import Predictor
                tl = self.timeline
                hold = np.arange(tl.holdout_start, len(b.ts), dtype=np.int64)
                old = Predictor(reg.path(champ), "cpu")
                if int(old.meta.get("data_end_ms", 0)) > int(b.ts[tl.holdout_start]):
                    raise RuntimeError("旧版本的训练数据与新测试窗口重叠，无法公平比较")
                op = old.predict(b, hold)
                old_ll = float(np.mean([task_eval(b, t, hold, op[t]["p_up"])["logloss"] for t in op]))
                better = new_ll < old_ll
                decision |= {"promoted": bool(better), "new_logloss": new_ll, "champion_logloss": old_ll,
                             "reason": "新版本在同一测试窗口上对数损失更低" if better else "新版本未优于当前线上版本，保持不变"}
                if better:
                    reg.promote(version, decision["reason"])
            except Exception as exc:  # noqa: BLE001
                decision |= {"promoted": False, "reason": f"无法与当前线上版本比较（{exc}），保持不变，可在控制台手动上线"}
        dump_json(decision, self.dir / "register.json")
        log.info("版本登记: %s", decision)
