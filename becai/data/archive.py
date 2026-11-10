"""币安公开历史数据归档 (data.binance.vision) 下载与规范化。

只使用官方公开、可回溯的数据；所有时间戳统一为 UTC 毫秒。
存储布局: data/raw/{dataset}/{symbol}/{YYYY-MM}.parquet + 同名 .json 清单。
"""
from __future__ import annotations

import calendar
import hashlib
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from ..utils import dump_json, get_logger, load_json

log = get_logger("becai.data")

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
PRICE_KEEP = ["open_time", "open", "high", "low", "close"]
TRADE_KEEP = PRICE_KEEP + ["volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str               # 归档路径模板中的类别段
    kind: str               # kline / funding / metrics
    monthly: bool
    daily: bool
    keep: tuple[str, ...] = ()
    interval: str = "1m"


SPECS: dict[str, DatasetSpec] = {
    "index": DatasetSpec("index", "futures/um/{p}/indexPriceKlines", "kline", True, True, tuple(PRICE_KEEP)),
    "spot": DatasetSpec("spot", "spot/{p}/klines", "kline", True, True, tuple(TRADE_KEEP)),
    "futures": DatasetSpec("futures", "futures/um/{p}/klines", "kline", True, True, tuple(TRADE_KEEP)),
    "mark": DatasetSpec("mark", "futures/um/{p}/markPriceKlines", "kline", True, True, tuple(PRICE_KEEP)),
    "premium": DatasetSpec("premium", "futures/um/{p}/premiumIndexKlines", "kline", True, True, tuple(PRICE_KEEP)),
    "funding": DatasetSpec("funding", "futures/um/{p}/fundingRate", "funding", True, False),
    "metrics": DatasetSpec("metrics", "futures/um/{p}/metrics", "metrics", False, True),
    "spot_1s": DatasetSpec("spot_1s", "spot/{p}/klines", "kline", False, True, tuple(PRICE_KEEP), "1s"),
}


def archive_url(base: str, spec: DatasetSpec, symbol: str, period: str, stamp: str) -> str:
    root = f"{base}/{spec.path.format(p=period)}/{symbol}"
    if spec.kind == "kline":
        return f"{root}/{spec.interval}/{symbol}-{spec.interval}-{stamp}.zip"
    if spec.kind == "funding":
        return f"{root}/{symbol}-fundingRate-{stamp}.zip"
    return f"{root}/{symbol}-metrics-{stamp}.zip"


def normalize_time_ms(t: pd.Series) -> pd.Series:
    """币安现货归档自2025年起使用微秒时间戳，这里统一到毫秒。"""
    t = t.astype("int64")
    return t.where(t < 10**14, t // 1000)


def parse_kline_csv(raw: bytes, keep: tuple[str, ...]) -> pd.DataFrame:
    head = raw[:64].split(b",", 1)[0]
    has_header = not head.strip().lstrip(b"-").replace(b".", b"").isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=KLINE_COLS if not has_header else None)
    df.columns = KLINE_COLS[: len(df.columns)]
    df["open_time"] = normalize_time_ms(df["open_time"])
    out = df[list(keep)].copy()
    for c in keep[1:]:
        out[c] = out[c].astype("float64")
    return out


def parse_funding_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df = df.rename(columns={"calc_time": "time", "last_funding_rate": "rate", "funding_interval_hours": "interval_hours"})
    df["time"] = normalize_time_ms(df["time"])
    return df[["time", "rate", "interval_hours"]].astype({"rate": "float64", "interval_hours": "float64"})


METRIC_COLS = {
    "sum_open_interest": "oi",
    "sum_open_interest_value": "oi_value",
    "count_toptrader_long_short_ratio": "ls_top_acct",
    "sum_toptrader_long_short_ratio": "ls_top_pos",
    "count_long_short_ratio": "ls_acct",
    "sum_taker_long_short_vol_ratio": "taker_ls",
}


def parse_metrics_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    t = pd.to_datetime(df["create_time"], utc=True)
    ms = (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    out = pd.DataFrame({"time": ms.astype("int64")})
    for src, dst in METRIC_COLS.items():
        out[dst] = pd.to_numeric(df.get(src), errors="coerce") if src in df else np.nan
    return out


def parse_archive(spec: DatasetSpec, raw_zip: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        raw = zf.read(zf.namelist()[0])
    if spec.kind == "kline":
        return parse_kline_csv(raw, spec.keep)
    if spec.kind == "funding":
        return parse_funding_csv(raw)
    return parse_metrics_csv(raw)


class ArchiveClient:
    def __init__(self, base: str, verify_checksum: bool = True, session: requests.Session | None = None):
        self.base = base.rstrip("/")
        self.verify_checksum = verify_checksum
        self.session = session or requests.Session()

    def fetch(self, url: str, retries: int = 4) -> bytes | None:
        """返回 zip 内容；404 返回 None；其它错误重试后抛出。"""
        last: Exception | None = None
        for i in range(retries):
            try:
                r = self.session.get(url, timeout=60)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                data = r.content
                if self.verify_checksum:
                    c = self.session.get(url + ".CHECKSUM", timeout=30)
                    if c.status_code == 200:
                        expected = c.text.split()[0].strip()
                        if hashlib.sha256(data).hexdigest() != expected:
                            raise IOError(f"校验和不一致: {url}")
                return data
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2 ** i)
        raise IOError(f"下载失败 {url}: {last}")


def month_iter(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _time_col(spec: DatasetSpec) -> str:
    return "open_time" if spec.kind == "kline" else "time"


def sync_month(client: ArchiveClient, raw_dir: Path, spec: DatasetSpec, symbol: str, y: int, m: int,
               first_day: date, last_day: date) -> dict:
    """同步某数据集某月数据。幂等：已完整的月份直接跳过。"""
    out_dir = raw_dir / spec.name / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{y:04d}-{m:02d}"
    pq, mf = out_dir / f"{stamp}.parquet", out_dir / f"{stamp}.json"
    manifest = load_json(mf, {"complete": False, "days": [], "source": None})
    if manifest.get("complete") and pq.exists():
        return {"month": stamp, "status": "cached"}
    days_in_month = calendar.monthrange(y, m)[1]
    month_end = date(y, m, days_in_month)
    tcol = _time_col(spec)

    if spec.monthly and month_end <= last_day:
        raw = client.fetch(archive_url(client.base, spec, symbol, "monthly", stamp))
        if raw is not None:
            df = parse_archive(spec, raw).drop_duplicates(tcol).sort_values(tcol)
            repaired = []
            if spec.kind == "kline" and spec.interval == "1m":
                # 月度归档偶尔缺失某些天（实测存在），用日度归档补齐
                per_day = (pd.to_datetime(df[tcol], unit="ms").dt.day).value_counts()
                for day in range(1, days_in_month + 1):
                    if per_day.get(day, 0) < 1440:
                        ds = date(y, m, day).isoformat()
                        extra = client.fetch(archive_url(client.base, spec, symbol, "daily", ds))
                        if extra is not None:
                            df = pd.concat([df, parse_archive(spec, extra)], ignore_index=True)
                            repaired.append(ds)
                df = df.drop_duplicates(tcol).sort_values(tcol)
            df.to_parquet(pq, index=False)
            dump_json({"complete": True, "days": [], "source": "monthly", "rows": len(df), "repaired_days": repaired}, mf)
            return {"month": stamp, "status": "monthly", "rows": len(df), "repaired": len(repaired)}
    if not spec.daily:
        return {"month": stamp, "status": "pending"}

    have = set(manifest.get("days", []))
    frames = [pd.read_parquet(pq)] if pq.exists() and have else []
    d0 = max(date(y, m, 1), first_day)
    d1 = min(month_end, last_day)
    new_days = []
    d = d0
    while d <= d1:
        ds = d.isoformat()
        if ds not in have:
            raw = client.fetch(archive_url(client.base, spec, symbol, "daily", ds))
            if raw is not None:
                frames.append(parse_archive(spec, raw))
                new_days.append(ds)
        d += timedelta(days=1)
    if not frames:
        return {"month": stamp, "status": "missing"}
    df = pd.concat(frames, ignore_index=True).drop_duplicates(tcol).sort_values(tcol)
    df.to_parquet(pq, index=False)
    days = sorted(have | set(new_days))
    # 只有日度来源的数据集（如持仓量指标）在整月日文件齐全后即视为完成
    complete = (not spec.monthly) and len(days) == days_in_month
    dump_json({"complete": complete, "days": days, "source": "daily", "rows": len(df)}, mf)
    return {"month": stamp, "status": "daily", "rows": len(df), "new_days": len(new_days)}


def sync_all(cfg, datasets: list[str] | None = None, progress: Callable[[dict], None] | None = None) -> list[dict]:
    """按配置同步全部数据集。可重复运行，只下载缺失部分。"""
    raw_dir = cfg.path("data_dir") / "raw"
    client = ArchiveClient(cfg.data.archive_base, cfg.data.verify_checksum)
    start = datetime.fromisoformat(str(cfg.data.start)).date()
    today = datetime.now(timezone.utc).date()
    end = datetime.fromisoformat(str(cfg.data.end)).date() if cfg.data.end else today - timedelta(days=1)
    names = datasets or [n for n, on in cfg.data.datasets.items() if on]
    jobs = []
    for name in names:
        spec = SPECS[name]
        for sym in cfg.symbols:
            for y, m in month_iter(start, end):
                jobs.append((spec, sym, y, m, start, end))
    n1s = int(cfg.data.get("spot_1s_sample_days", 0) or 0)
    if n1s > 0 and datasets is None:
        s1 = max(start, end - timedelta(days=n1s - 1))
        for sym in cfg.symbols:
            for y, m in month_iter(s1, end):
                jobs.append((SPECS["spot_1s"], sym, y, m, s1, end))
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=int(cfg.data.download_workers)) as ex:
        futs = {ex.submit(sync_month, client, raw_dir, *j): j for j in jobs}
        for fut in as_completed(futs):
            spec, sym, y, m, *_ = futs[fut]
            done += 1
            try:
                res = {"dataset": spec.name, "symbol": sym, **fut.result()}
            except Exception as exc:  # noqa: BLE001
                res = {"dataset": spec.name, "symbol": sym, "month": f"{y:04d}-{m:02d}", "status": "error", "error": str(exc)}
                log.warning("同步失败 %s %s %04d-%02d: %s", spec.name, sym, y, m, exc)
            results.append(res)
            if progress:
                progress({"done": done, "total": len(jobs), "last": res})
    log.info("数据同步完成: %d 个文件任务, 错误 %d", len(results), sum(r["status"] == "error" for r in results))
    return results
