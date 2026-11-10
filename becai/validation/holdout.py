"""最终测试集访问台账：防止反复查看测试集来挑选参数。"""
from __future__ import annotations

import time
from pathlib import Path

from ..utils import dump_json, load_json, ms_to_iso


class HoldoutLockError(RuntimeError):
    pass


class HoldoutLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        return load_json(self.path, [])

    def count(self, window: tuple[int, int]) -> int:
        return sum(1 for e in self.entries() if tuple(e["window"]) == tuple(window))

    def acquire(self, window: tuple[int, int], run_id: str, config_digest: str, max_evaluations: int, force: bool = False) -> dict:
        """登记一次测试集评估。同一时间窗口超过次数上限时拒绝（force 可越过但会被永久记录）。"""
        entries = self.entries()
        prior = [e for e in entries if tuple(e["window"]) == tuple(window)]
        same_run = [e for e in prior if e["run_id"] == run_id]
        if same_run:
            return same_run[0] | {"reused": True}          # 同一次运行的断点恢复不重复计数
        if len(prior) >= max_evaluations and not force:
            raise HoldoutLockError(
                f"测试集窗口 {ms_to_iso(window[0])} ~ {ms_to_iso(window[1])} 已被评估 {len(prior)} 次（上限 {max_evaluations}）。"
                "继续在同一测试集上比较模型会造成选择偏差。请等待新数据形成新的测试窗口，或使用 --force（将记入台账并在报告中标注）。")
        entry = {"window": list(window), "run_id": run_id, "config": config_digest, "time": int(time.time() * 1000),
                 "sequence": len(prior) + 1, "forced": bool(force and len(prior) >= max_evaluations)}
        entries.append(entry)
        dump_json(entries, self.path)
        return entry
