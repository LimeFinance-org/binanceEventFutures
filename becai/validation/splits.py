"""金融时间序列验证切分：独立测试集 + Purged Walk-Forward + Embargo。

面板是规则的 1 分钟网格，所以"行号差 = 分钟差"，purge/embargo 直接用行数表达。
标签 y[t] 使用了 t+h 时刻的价格，因此训练样本必须满足 t + h + embargo <= 验证起点。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAY = 1440


@dataclass
class Fold:
    k: int
    train: np.ndarray
    es: np.ndarray          # 早停集（训练窗口尾部，与训练段、验证段之间都有 purge）
    val: np.ndarray

    def describe(self, ts: np.ndarray) -> dict:
        from ..utils import ms_to_iso
        f = lambda r: [ms_to_iso(ts[r[0]]), ms_to_iso(ts[r[-1]])] if len(r) else None  # noqa: E731
        return {"fold": self.k, "train": f(self.train), "es": f(self.es), "val": f(self.val),
                "n_train": len(self.train), "n_es": len(self.es), "n_val": len(self.val)}


@dataclass
class Timeline:
    n: int                  # 总行数
    dev_end: int            # 开发区间 [0, dev_end)
    holdout_start: int      # 最终测试区间 [holdout_start, n)
    purge: int


def make_timeline(n_rows: int, holdout_days: int, purge: int) -> Timeline:
    holdout_start = max(0, n_rows - int(holdout_days) * DAY)
    return Timeline(n_rows, max(0, holdout_start - purge), holdout_start, purge)


def purge_minutes(cfg, horizon: int | None = None) -> int:
    h = int(horizon or max(cfg.horizons)) + int(cfg.labels.entry_delay_minutes)
    return h + int(cfg.validation.embargo_minutes)


def walk_forward(dev_end: int, n_folds: int, val_days: int, train_days: int | None, purge: int,
                 es_fraction: float = 0.1, stride: int = 1, start: int = 0) -> list[Fold]:
    """验证窗口取开发区间最后 n_folds*val_days 天，依次前滚；训练窗口严格在验证窗口之前。"""
    val_len = int(val_days) * DAY
    folds = []
    for k in range(n_folds):
        ve = dev_end - (n_folds - 1 - k) * val_len
        vs = ve - val_len
        t_end = vs - purge
        t_start = start if not train_days else max(start, t_end - int(train_days) * DAY)
        if t_end - t_start < 5 * DAY or vs < 0:
            continue
        es_len = max(DAY, int((t_end - t_start) * es_fraction))
        es_start = t_end - es_len
        tr_end = es_start - purge
        if tr_end - t_start < 3 * DAY:
            continue
        folds.append(Fold(k, np.arange(t_start, tr_end, max(1, stride), dtype=np.int64),
                          np.arange(es_start, t_end, max(1, stride), dtype=np.int64),
                          np.arange(vs, ve, dtype=np.int64)))
    return folds


def final_fit_split(dev_end: int, train_days: int | None, purge: int, es_fraction: float, stride: int, start: int = 0) -> Fold:
    """最终模型：用开发区间全部数据训练（尾部留早停集），之后在测试集上一次性评估。"""
    t_start = start if not train_days else max(start, dev_end - int(train_days) * DAY)
    es_len = max(DAY, int((dev_end - t_start) * es_fraction))
    es_start = dev_end - es_len
    return Fold(-1, np.arange(t_start, es_start - purge, max(1, stride), dtype=np.int64),
                np.arange(es_start, dev_end, max(1, stride), dtype=np.int64), np.empty(0, dtype=np.int64))


def check_no_overlap(fold: Fold, purge: int) -> bool:
    """断言式检查：训练/早停样本的标签窗口不触及验证集。"""
    ok = True
    if len(fold.val):
        if len(fold.train):
            ok &= fold.train.max() + purge <= fold.val.min()
        if len(fold.es):
            ok &= fold.es.max() + purge <= fold.val.min()
    if len(fold.train) and len(fold.es):
        ok &= fold.train.max() + purge <= fold.es.min()
    return bool(ok)
