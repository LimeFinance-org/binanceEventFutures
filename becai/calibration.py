"""概率校准：Platt / 等渗回归 / 不校准，按"前向链式"样本外对数损失自动选择。"""
from __future__ import annotations

import numpy as np

from .validation.stats import logloss


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """method: none / platt / isotonic"""

    def __init__(self, method: str = "none"):
        self.method = method
        self.a, self.b = 1.0, 0.0
        self.xs: np.ndarray | None = None
        self.ys: np.ndarray | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e6, max_iter=200).fit(_logit(p).reshape(-1, 1), y)
            self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
            self.xs, self.ys = iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if self.method == "platt":
            out = 1 / (1 + np.exp(-(self.a * _logit(p) + self.b)))
        elif self.method == "isotonic" and self.xs is not None:
            out = np.interp(p, self.xs, self.ys)
        else:
            out = p.copy()
        return np.where(np.isfinite(p), np.clip(out, 1e-4, 1 - 1e-4), np.nan)

    def to_dict(self) -> dict:
        return {"method": self.method, "a": self.a, "b": self.b,
                "xs": None if self.xs is None else self.xs.tolist(), "ys": None if self.ys is None else self.ys.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "Calibrator":
        c = Calibrator(d.get("method", "none"))
        c.a, c.b = float(d.get("a", 1.0)), float(d.get("b", 0.0))
        if d.get("xs") is not None:
            c.xs, c.ys = np.asarray(d["xs"], dtype=float), np.asarray(d["ys"], dtype=float)
        return c


def forward_chain_calibrate(p: np.ndarray, y: np.ndarray, fold_id: np.ndarray, methods=("none", "platt", "isotonic")) -> dict:
    """按折的时间顺序做前向链式校准：第 k 折的校准器只用 <k 折的样本外预测拟合。

    返回：
      best: 选中的方法；scores: 各方法在后续折上的对数损失；
      p_cal: 用选中方法得到的"校准后的样本外概率"（第一个折无法校准，为 NaN）；
      calibrator: 在全部样本外预测上拟合的最终校准器（用于测试集与实时预测）。
    """
    folds = np.unique(fold_id)
    ok = np.isfinite(p) & (y >= 0)
    scores, cal_preds = {}, {}
    for m in methods:
        out = np.full(len(p), np.nan)
        for k in folds[1:]:
            tr = ok & (fold_id < k)
            te = (fold_id == k) & np.isfinite(p)
            if tr.sum() < 500 or not te.any():
                continue
            out[te] = Calibrator(m).fit(p[tr], y[tr]).transform(p[te])
        ev = ok & np.isfinite(out)
        scores[m] = logloss(y[ev], out[ev]) if ev.any() else float("inf")
        cal_preds[m] = out
    best = min(scores, key=scores.get)
    final = Calibrator(best).fit(p[ok], y[ok]) if ok.sum() >= 500 else Calibrator("none")
    return {"best": best, "scores": scores, "p_cal": cal_preds[best], "calibrator": final}
