"""原始数据读取、覆盖情况与质量检查。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import MINUTE_MS, load_json, ms_to_iso
from .archive import SPECS, _time_col


def raw_dir(cfg) -> Path:
    return cfg.path("data_dir") / "raw"


def load_raw(cfg, dataset: str, symbol: str, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    """读取 [start_ms, end_ms) 范围内的规范化原始数据。"""
    spec = SPECS[dataset]
    tcol = _time_col(spec)
    d = raw_dir(cfg) / dataset / symbol
    frames = []
    for f in sorted(d.glob("*.parquet")) if d.exists() else []:
        y, m = int(f.stem[:4]), int(f.stem[5:7])
        m0 = pd.Timestamp(year=y, month=m, day=1, tz="UTC")
        lo, hi = int(m0.timestamp() * 1000), int((m0 + pd.offsets.MonthBegin(1)).timestamp() * 1000)
        if (end_ms is not None and lo >= end_ms) or (start_ms is not None and hi <= start_ms):
            continue
        frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame(columns=[tcol])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(tcol).sort_values(tcol)
    if start_ms is not None:
        df = df[df[tcol] >= start_ms]
    if end_ms is not None:
        df = df[df[tcol] < end_ms]
    return df.reset_index(drop=True)


def coverage(cfg) -> list[dict]:
    """每个数据集/币种的时间覆盖与缺口统计（供Web界面展示）。"""
    rows = []
    base = raw_dir(cfg)
    for name, spec in SPECS.items():
        for sym in cfg.symbols:
            d = base / name / sym
            files = sorted(d.glob("*.parquet")) if d.exists() else []
            if not files:
                rows.append({"dataset": name, "symbol": sym, "files": 0})
                continue
            tcol = _time_col(spec)
            first = pd.read_parquet(files[0], columns=[tcol])[tcol].min()
            last = pd.read_parquet(files[-1], columns=[tcol])[tcol].max()
            n_rows = sum(int(load_json(f.with_suffix(".json"), {}).get("rows", 0)) for f in files)
            incomplete = [f.stem for f in files if not load_json(f.with_suffix(".json"), {}).get("complete")]
            rows.append({
                "dataset": name, "symbol": sym, "files": len(files), "rows": n_rows,
                "start": ms_to_iso(first), "end": ms_to_iso(last), "incomplete_months": incomplete[-6:],
                "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
            })
    return rows


def gap_report(df: pd.DataFrame, tcol: str = "open_time", step_ms: int = MINUTE_MS) -> dict:
    """缺失K线检查：数量、最长连续缺口。"""
    if len(df) < 2:
        return {"rows": int(len(df)), "missing": 0, "max_gap_minutes": 0, "gaps": []}
    t = df[tcol].to_numpy()
    diff = np.diff(t) // step_ms
    idx = np.where(diff > 1)[0]
    gaps = [{"after": ms_to_iso(int(t[i])), "missing": int(diff[i] - 1)} for i in idx]
    gaps.sort(key=lambda g: -g["missing"])
    return {
        "rows": int(len(df)),
        "missing": int((diff[idx] - 1).sum()),
        "max_gap_minutes": int((diff.max() - 1)) if len(diff) else 0,
        "gaps": gaps[:20],
        "duplicates": int(len(t) - len(np.unique(t))),
    }


def quality_report(cfg) -> dict:
    out = {}
    for name, on in cfg.data.datasets.items():
        spec = SPECS[name]
        if not on or spec.kind != "kline":
            continue
        for sym in cfg.symbols:
            df = load_raw(cfg, name, sym)
            rep = gap_report(df)
            if len(df) and {"high", "low", "close"} <= set(df.columns) and name != "premium":
                bad = (df["high"] < df["low"]) | (df["close"] <= 0)
                rep["invalid_bars"] = int(bad.sum())
            out[f"{name}/{sym}"] = rep
    return out
