"""配置加载：default.yaml <- local.yaml <- BECAI_CONFIG 指定文件 <- 代码内覆盖。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "default.yaml"


class Config(dict):
    """支持属性访问的嵌套字典。"""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)

    def digest(self, *sections: str) -> str:
        """指定配置段的稳定哈希，用于缓存键和测试集台账。"""
        d = self.to_dict()
        payload = {s: d.get(s) for s in sections} if sections else d
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    with open(DEFAULT_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    candidates = [ROOT / "config" / "local.yaml"]
    env = os.environ.get("BECAI_CONFIG")
    if env:
        candidates.append(Path(env))
    if path:
        candidates.append(Path(path))
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return Config.wrap(cfg)


def tasks(cfg: Config) -> list[tuple[str, int]]:
    """四项预测任务：(币种, 期限分钟)。"""
    return [(s, int(h)) for s in cfg.symbols for h in cfg.horizons]


def task_name(symbol: str, horizon: int) -> str:
    return f"{symbol}_{horizon}m"
