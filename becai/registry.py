"""模型版本库与统一预测器。

版本目录结构:
  registry/versions/{version}/
      meta.json                 版本元数据（来源运行、数据区间、指标摘要、配置快照）
      models/{spec_id}/         每个已训练的基础模型（多个任务可共享同一模型）
      tasks/{task}.json         该任务的集成成员与权重、校准器、信号策略
  registry/champion.json        当前线上版本指针 + 历史（用于回退）

Predictor 是训练期测试集评估与实时预测共用的同一段代码："测什么就上线什么"。
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from .bundle import DataBundle
from .calibration import Calibrator
from .models.base import Candidate
from .signals.policy import Policy
from .utils import dump_json, get_logger, load_json

log = get_logger("becai.registry")


def blend_logit(ps: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """按权重在 logit 空间平均。任一成员为 NaN 则结果为 NaN。"""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    z = np.zeros(len(ps[0]))
    for p, wi in zip(ps, w):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        z += wi * np.log(q / (1 - q))
    return 1 / (1 + np.exp(-z))


class Predictor:
    def __init__(self, version_dir: str | Path, device: str = "cpu"):
        self.dir = Path(version_dir)
        self.meta = load_json(self.dir / "meta.json", {})
        self.version = self.meta.get("version", self.dir.name)
        self.task_cfg: dict[tuple[str, int], dict] = {}
        for f in sorted((self.dir / "tasks").glob("*.json")):
            d = load_json(f)
            self.task_cfg[(d["task"][0], int(d["task"][1]))] = d
        ids = sorted({m for d in self.task_cfg.values() for m in d["members"]})
        self.models: dict[str, Candidate] = {i: Candidate.load(self.dir / "models" / i, device=device) for i in ids}
        self.calibrators = {t: Calibrator.from_dict(d["calibrator"]) for t, d in self.task_cfg.items()}
        self.policies = {t: Policy.from_dict(d["policy"]) for t, d in self.task_cfg.items()}

    def predict(self, bundle: DataBundle, rows: np.ndarray) -> dict[tuple[str, int], dict]:
        """返回每个任务的 {raw, p_up}（p_up 为校准后的上涨概率）。"""
        raw = {i: m.predict(bundle, rows) for i, m in self.models.items()}
        out = {}
        for t, d in self.task_cfg.items():
            p = blend_logit([raw[i][t] for i in d["members"]], d["weights"])
            out[t] = {"raw": p, "p_up": self.calibrators[t].transform(p)}
        return out


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)
        self.pointer = self.root / "champion.json"

    def versions(self) -> list[dict]:
        champ = self.champion()
        out = []
        for d in sorted((self.root / "versions").iterdir(), reverse=True):
            meta = load_json(d / "meta.json")
            if meta:
                out.append(meta | {"is_champion": meta.get("version") == champ})
        return out

    def champion(self) -> str | None:
        return (load_json(self.pointer, {}) or {}).get("version")

    def path(self, version: str) -> Path:
        return self.root / "versions" / version

    def add(self, src_dir: str | Path, meta: dict) -> str:
        version = meta.get("version") or time.strftime("v%Y%m%d_%H%M%S", time.gmtime())
        dst = self.path(version)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir, dst)
        dump_json(meta | {"version": version, "created": int(time.time() * 1000)}, dst / "meta.json")
        log.info("已登记模型版本 %s", version)
        return version

    def promote(self, version: str, reason: str = "") -> None:
        if not (self.path(version) / "meta.json").exists():
            raise FileNotFoundError(f"版本不存在: {version}")
        cur = load_json(self.pointer, {"version": None, "history": []})
        hist = cur.get("history", [])
        if cur.get("version") and cur["version"] != version:
            hist.append({"version": cur["version"], "until": int(time.time() * 1000)})
        dump_json({"version": version, "since": int(time.time() * 1000), "reason": reason, "history": hist[-50:]}, self.pointer)
        log.info("线上版本切换为 %s (%s)", version, reason)

    def rollback(self) -> str | None:
        """回退到上一个线上版本。"""
        cur = load_json(self.pointer, {})
        hist = cur.get("history", [])
        while hist:
            prev = hist.pop()
            if (self.path(prev["version"]) / "meta.json").exists():
                dump_json({"version": prev["version"], "since": int(time.time() * 1000),
                           "reason": f"从 {cur.get('version')} 回退", "history": hist}, self.pointer)
                return prev["version"]
        return None

    def load_champion(self, device: str = "cpu") -> Predictor | None:
        v = self.champion()
        return Predictor(self.path(v), device) if v else None
