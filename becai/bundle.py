"""训练数据包：所有币种在同一时间网格上的特征、序列通道、标签与市场状态。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data.dataset import load_panels
from .features.core import FeatureSet, build_features, max_lookback
from .labels import make_labels
from .utils import dump_json, get_logger, load_json, timer

log = get_logger("becai.bundle")


@dataclass
class SymbolData:
    X: np.ndarray                    # float32 [n, F]（可能是只读内存映射）
    channels: np.ndarray             # float32 [n, C]
    ix_close: np.ndarray
    regime: pd.DataFrame
    labels: dict[int, np.ndarray] = field(default_factory=dict)   # horizon -> y(int8)
    moves: dict[int, np.ndarray] = field(default_factory=dict)
    feat_ok: np.ndarray | None = None


@dataclass
class DataBundle:
    ts: np.ndarray                   # int64 决策时刻（所有币种共用）
    columns: list[str]
    meta: dict[str, dict]
    channel_names: list[str]
    symbols: dict[str, SymbolData]
    horizons: list[int]

    def groups(self) -> dict[str, list[int]]:
        g: dict[str, list[int]] = {}
        for j, c in enumerate(self.columns):
            m = self.meta[c]
            g.setdefault(f"{m['source']}|{m['family']}|{m['scale']}", []).append(j)
        return g

    def usable(self, symbol: str, horizon: int, rows: np.ndarray, train: bool = True) -> np.ndarray:
        """过滤出特征有效、标签有效的行。train=True 时剔除平局。"""
        sd = self.symbols[symbol]
        y = sd.labels[horizon][rows]
        ok = sd.feat_ok[rows] & ((y >= 0) if train else (y >= -1))
        return rows[ok]

    def rows_between(self, start_ms: int, end_ms: int) -> np.ndarray:
        lo, hi = np.searchsorted(self.ts, [start_ms, end_ms])
        return np.arange(lo, hi, dtype=np.int64)


def attach_labels(sd: SymbolData, horizons: list[int], entry_delay: int, warmup: int) -> None:
    for h in horizons:
        sd.labels[h], sd.moves[h] = make_labels(sd.ix_close, h, entry_delay)
    ok = np.isfinite(sd.ix_close)
    ok[:warmup] = False
    sd.feat_ok = ok


def bundle_from_panels(cfg, panels: dict[str, pd.DataFrame]) -> DataBundle:
    """由面板直接构建（实时预测、测试与小数据场景用，不落盘）。"""
    warmup = max(int(cfg.features.warmup_minutes), max_lookback(cfg))
    symbols: dict[str, SymbolData] = {}
    first: FeatureSet | None = None
    for sym, p in panels.items():
        fs = build_features(p, {o: q for o, q in panels.items() if o != sym}, cfg)
        first = first or fs
        sd = SymbolData(fs.X, fs.channels, p["ix_c"].to_numpy(dtype=np.float64), fs.regime)
        attach_labels(sd, [int(h) for h in cfg.horizons], int(cfg.labels.entry_delay_minutes), warmup)
        symbols[sym] = sd
    assert first is not None
    return DataBundle(first.ts, first.columns, first.meta, first.channel_names, symbols, [int(h) for h in cfg.horizons])


def build_bundle(cfg, start=None, end=None, cache: bool = True) -> DataBundle:
    """从本地原始数据构建数据包；特征矩阵缓存为 .npy 并以只读内存映射加载（多进程共享）。"""
    panels = load_panels(cfg, start, end)
    any_p = next(iter(panels.values()))
    key = f"{cfg.digest('data', 'features', 'labels', 'symbols', 'horizons')}_{int(any_p.index[0])}_{int(any_p.index[-1])}"
    cdir = Path(cfg.path("data_dir")) / "cache" / key
    info = load_json(cdir / "info.json")
    if cache and info:
        symbols = {}
        for sym, p in panels.items():
            sd = SymbolData(np.load(cdir / f"{sym}_X.npy", mmap_mode="r"), np.load(cdir / f"{sym}_C.npy", mmap_mode="r"),
                            p["ix_c"].to_numpy(dtype=np.float64), pd.read_parquet(cdir / f"{sym}_regime.parquet"))
            attach_labels(sd, [int(h) for h in cfg.horizons], int(cfg.labels.entry_delay_minutes), int(info["warmup"]))
            symbols[sym] = sd
        log.info("加载特征缓存 %s: %d 行 × %d 特征", key, len(any_p), len(info["columns"]))
        return DataBundle(any_p.index.to_numpy(dtype=np.int64), info["columns"], info["meta"], info["channel_names"],
                          symbols, [int(h) for h in cfg.horizons])
    if not cache:
        with timer(f"构建特征 ({len(any_p)} 行)", log):
            return bundle_from_panels(cfg, panels)
    # 逐币种构建并直接写入磁盘（内存映射），长历史时内存峰值只取决于单个币种
    cdir.mkdir(parents=True, exist_ok=True)
    warmup = max(int(cfg.features.warmup_minutes), max_lookback(cfg))
    symbols, first = {}, None
    with timer(f"构建特征 ({len(any_p)} 行)", log):
        for sym, p in panels.items():
            fs = build_features(p, {o: q for o, q in panels.items() if o != sym}, cfg, out_path=cdir / f"{sym}_X.npy")
            np.save(cdir / f"{sym}_C.npy", fs.channels)
            fs.regime.to_parquet(cdir / f"{sym}_regime.parquet")
            sd = SymbolData(fs.X, np.load(cdir / f"{sym}_C.npy", mmap_mode="r"), p["ix_c"].to_numpy(dtype=np.float64), fs.regime)
            attach_labels(sd, [int(h) for h in cfg.horizons], int(cfg.labels.entry_delay_minutes), warmup)
            symbols[sym] = sd
            first = first or fs
            del fs
    dump_json({"columns": first.columns, "meta": first.meta, "channel_names": first.channel_names, "warmup": warmup}, cdir / "info.json")
    return DataBundle(first.ts, first.columns, first.meta, first.channel_names, symbols, [int(h) for h in cfg.horizons])
