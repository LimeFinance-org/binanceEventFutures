"""实时公开行情（REST）。事件合约本身没有 API，这里只获取用于预测的公开行情数据。

返回的 DataFrame 与归档解析后的格式完全一致，因此实时与离线共用同一套面板/特征代码。
只返回"已经收盘"的K线。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from ..utils import MINUTE_MS, get_logger, utcnow_ms
from .archive import PRICE_KEEP, TRADE_KEEP

log = get_logger("becai.rest")


class MarketDataError(RuntimeError):
    pass


class RestClient:
    def __init__(self, cfg, session: requests.Session | None = None):
        r = cfg.data.rest
        self.spot_bases = [r.spot_base, r.spot_fallback]
        self.fut_base = r.futures_base
        self.timeout = float(r.timeout_seconds)
        self.session = session or requests.Session()

    def _get(self, bases: list[str], path: str, params: dict):
        last: Exception | None = None
        for base in bases:
            try:
                resp = self.session.get(base + path, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                last = MarketDataError(f"HTTP {resp.status_code} {base}{path}")
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise MarketDataError(str(last))

    def _klines(self, bases, path, params, keep, start_ms: int, now_ms: int, limit: int) -> pd.DataFrame:
        rows, cursor = [], start_ms
        while cursor < now_ms:
            data = self._get(bases, path, {**params, "interval": "1m", "startTime": cursor, "limit": limit})
            if not data:
                break
            rows.extend(data)
            nxt = int(data[-1][0]) + MINUTE_MS
            if nxt <= cursor or len(data) < limit:
                break
            cursor = nxt
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
                "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        df = df[df["close_time"].astype("int64") < now_ms] if len(df) else df      # 仅保留已收盘K线
        out = df[list(keep)].copy()
        out["open_time"] = out["open_time"].astype("int64")
        for c in keep[1:]:
            out[c] = out[c].astype("float64")
        return out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    def fetch(self, dataset: str, symbol: str, start_ms: int, now_ms: int | None = None) -> pd.DataFrame:
        now_ms = now_ms or utcnow_ms()
        fb = [self.fut_base]
        if dataset == "spot":
            return self._klines(self.spot_bases, "/api/v3/klines", {"symbol": symbol}, TRADE_KEEP, start_ms, now_ms, 1000)
        if dataset == "futures":
            return self._klines(fb, "/fapi/v1/klines", {"symbol": symbol}, TRADE_KEEP, start_ms, now_ms, 1500)
        if dataset == "index":
            return self._klines(fb, "/fapi/v1/indexPriceKlines", {"pair": symbol}, PRICE_KEEP, start_ms, now_ms, 1500)
        if dataset == "mark":
            return self._klines(fb, "/fapi/v1/markPriceKlines", {"symbol": symbol}, PRICE_KEEP, start_ms, now_ms, 1500)
        if dataset == "premium":
            return self._klines(fb, "/fapi/v1/premiumIndexKlines", {"symbol": symbol}, PRICE_KEEP, start_ms, now_ms, 1500)
        if dataset == "funding":
            data = self._get(fb, "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start_ms - 3 * 24 * 60 * MINUTE_MS, "limit": 1000})
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame(columns=["time", "rate", "interval_hours"])
            return pd.DataFrame({"time": df["fundingTime"].astype("int64"), "rate": df["fundingRate"].astype("float64"),
                                 "interval_hours": np.nan})
        if dataset == "metrics":
            return self._metrics(symbol, start_ms)
        raise ValueError(dataset)

    def _metrics(self, symbol: str, start_ms: int) -> pd.DataFrame:
        p = {"symbol": symbol, "period": "5m", "limit": 500}
        parts = {}
        oi = pd.DataFrame(self._get([self.fut_base], "/futures/data/openInterestHist", p))
        if oi.empty:
            return pd.DataFrame(columns=["time"])
        base = pd.DataFrame({"time": oi["timestamp"].astype("int64"), "oi": oi["sumOpenInterest"].astype("float64"),
                             "oi_value": oi["sumOpenInterestValue"].astype("float64")})
        endpoints = {
            "ls_top_acct": ("/futures/data/topLongShortAccountRatio", "longShortRatio"),
            "ls_top_pos": ("/futures/data/topLongShortPositionRatio", "longShortRatio"),
            "ls_acct": ("/futures/data/globalLongShortAccountRatio", "longShortRatio"),
            "taker_ls": ("/futures/data/takerlongshortRatio", "buySellRatio"),
        }
        for name, (path, field) in endpoints.items():
            try:
                d = pd.DataFrame(self._get([self.fut_base], path, p))
                parts[name] = pd.Series(d[field].astype("float64").to_numpy(), index=d["timestamp"].astype("int64"))
            except Exception as exc:  # noqa: BLE001  单个指标失败不影响整体
                log.warning("指标 %s 获取失败: %s", name, exc)
        for name, s in parts.items():
            base[name] = base["time"].map(s)
        return base[base["time"] >= start_ms - 60 * MINUTE_MS].reset_index(drop=True)


class LiveBuffer:
    """维护每个币种最近 N 分钟的原始数据，增量更新。"""

    def __init__(self, cfg, client: RestClient | None = None):
        self.cfg = cfg
        self.client = client or RestClient(cfg)
        self.minutes = int(cfg.live.buffer_minutes)
        self.frames: dict[str, dict[str, pd.DataFrame]] = {s: {} for s in cfg.symbols}
        self.errors: dict[str, str] = {}
        self._slow_ts: dict[tuple[str, str], float] = {}

    def update(self, now_ms: int | None = None) -> None:
        now_ms = now_ms or utcnow_ms()
        floor = (now_ms // MINUTE_MS) * MINUTE_MS - self.minutes * MINUTE_MS
        self.errors = {}
        for sym in self.cfg.symbols:
            for name, on in self.cfg.data.datasets.items():
                if not on:
                    continue
                cur = self.frames[sym].get(name)
                tcol = "time" if name in ("funding", "metrics") else "open_time"
                if name in ("funding", "metrics"):
                    # 低频数据：同一分钟内不重复刷新（用行情时钟而非墙上时钟，回放时同样成立）
                    if cur is not None and now_ms - self._slow_ts.get((sym, name), 0) < 50_000:
                        continue
                    start = floor
                else:
                    start = floor if cur is None or cur.empty else int(cur[tcol].max()) + MINUTE_MS
                try:
                    new = self.client.fetch(name, sym, start, now_ms)
                    self._slow_ts[(sym, name)] = now_ms
                except Exception as exc:  # noqa: BLE001
                    self.errors[f"{name}/{sym}"] = str(exc)
                    continue
                if cur is not None and not cur.empty and name not in ("funding", "metrics"):
                    new = pd.concat([cur, new], ignore_index=True).drop_duplicates(tcol, keep="last").sort_values(tcol)
                    new = new[new[tcol] >= floor - MINUTE_MS].reset_index(drop=True)
                self.frames[sym][name] = new

    def latest_decision_ts(self, symbol: str) -> int | None:
        ix = self.frames[symbol].get("index")
        if ix is None or ix.empty:
            return None
        return int(ix["open_time"].max()) + MINUTE_MS
