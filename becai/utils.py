from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MINUTE_MS = 60_000


def get_logger(name: str = "becai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("becai").handlers:
        root = logging.getLogger("becai")
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        root.addHandler(h)
        root.setLevel(os.environ.get("BECAI_LOG", "INFO"))
        root.propagate = False
    return logger


def utcnow_ms() -> int:
    return int(time.time() * 1000)


def ms_to_iso(ms: int | float | None) -> str | None:
    if ms is None or (isinstance(ms, float) and np.isnan(ms)):
        return None
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def date_to_ms(s: str | datetime) -> int:
    if isinstance(s, datetime):
        dt = s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(s)).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if np.isnan(v) or np.isinf(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def _clean(o: Any) -> Any:
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def dump_json(obj: Any, path: str | os.PathLike | None = None) -> str:
    text = json.dumps(_clean(json.loads(json.dumps(obj, cls=NumpyEncoder))), ensure_ascii=False, indent=2)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)  # 原子写，避免异常中断留下半个文件
    return text


def load_json(path: str | os.PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    t0 = time.time()
    yield
    (logger or get_logger()).info("%s 用时 %.1fs", label, time.time() - t0)


def resolve_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
