"""候选模型统一接口。

一个 Candidate 负责一个或多个任务 (币种, 期限)：
  * 表格模型（logreg / lightgbm / xgboost）：单任务；
  * 神经网络（mlp / gru / tcn / transformer / multiscale）：多任务（多期限输出头，可跨币种共享）。
predict 返回 {(symbol, horizon): P(上涨)}，与传入的 rows 一一对应，特征无效的行为 NaN。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..bundle import DataBundle
from ..utils import dump_json, load_json

Task = tuple[str, int]

# 复杂度等级：用于"复杂模型必须证明优于简单模型"的选择规则
COMPLEXITY = {"logreg": 0, "lightgbm": 1, "xgboost": 1, "mlp": 2, "gru": 3, "tcn": 3, "transformer": 3,
              "multiscale": 4, "ensemble": 5}


class Candidate:
    family: str = "base"

    def __init__(self, spec: dict):
        """spec: {family, params, tasks:[[sym,h],..], features:[列名], device, threads, seed}"""
        self.spec = spec
        self.params: dict = dict(spec.get("params", {}))
        self.tasks: list[Task] = [(s, int(h)) for s, h in spec["tasks"]]
        self.features: list[str] = list(spec.get("features", []))
        self.device: str = spec.get("device", "cpu")
        self.threads: int = int(spec.get("threads", 4))
        self.seed: int = int(spec.get("seed", 42))

    @property
    def complexity(self) -> int:
        return COMPLEXITY.get(self.family, 9)

    def feature_idx(self, bundle: DataBundle) -> np.ndarray:
        pos = {c: j for j, c in enumerate(bundle.columns)}
        missing = [c for c in self.features if c not in pos]
        if missing:
            raise KeyError(f"数据中缺少模型需要的特征: {missing[:5]}…")
        return np.asarray([pos[c] for c in self.features], dtype=np.int64)

    def fit(self, bundle: DataBundle, train_rows: np.ndarray, es_rows: np.ndarray) -> dict:
        raise NotImplementedError

    def predict(self, bundle: DataBundle, rows: np.ndarray) -> dict[Task, np.ndarray]:
        raise NotImplementedError

    # ---- 持久化 ----
    def _save_state(self, d: Path) -> None:
        raise NotImplementedError

    def _load_state(self, d: Path) -> None:
        raise NotImplementedError

    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        dump_json(self.spec | {"family": self.family}, d / "spec.json")
        self._save_state(d)

    @staticmethod
    def load(d: str | Path, device: str | None = None) -> "Candidate":
        from .zoo import make_candidate
        d = Path(d)
        spec = load_json(d / "spec.json")
        if device:
            spec["device"] = device
        cand = make_candidate(spec)
        cand._load_state(d)
        return cand


def take_rows(X: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """从（可能是内存映射的）特征矩阵取子块，返回 float32 连续数组。"""
    if len(rows) == 0:
        return np.empty((0, len(cols)), dtype=np.float32)
    lo, hi = int(rows.min()), int(rows.max()) + 1
    if (hi - lo) <= 8 * len(rows):
        out = np.asarray(X[lo:hi])[rows - lo][:, cols]     # 密集区间：整块读入再抽取
    else:
        out = np.asarray(X[rows])[:, cols]
    return np.ascontiguousarray(out, dtype=np.float32)
