"""实时行情接口与实时服务的异常路径。"""
import pandas as pd
import pytest

from becai.config import Config
from becai.data.rest import LiveBuffer, MarketDataError, RestClient
from becai.live.service import LiveService
from becai.live.store import LiveStore
from becai.utils import MINUTE_MS


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, now_ms, fail_hosts=()):
        self.now, self.fail, self.calls = now_ms, fail_hosts, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if any(h in url for h in self.fail):
            return FakeResp(451, {})
        start = params["startTime"]
        rows = []
        t = start
        while t <= self.now and len(rows) < params["limit"]:       # 最后一根是尚未收盘的K线
            rows.append([t, "100", "101", "99", "100.5", "5", t + MINUTE_MS - 1, "500", 10, "2", "200", "0"])
            t += MINUTE_MS
        return FakeResp(200, rows)


def test_rest_returns_only_closed_bars_and_falls_back(cfg):
    now = 1_700_000_000_000 // MINUTE_MS * MINUTE_MS + 20_000            # 当前分钟过去了 20 秒
    sess = FakeSession(now, fail_hosts=("api.binance.com",))
    df = RestClient(cfg, sess).fetch("spot", "BTCUSDT", now - 5 * MINUTE_MS - 20_000, now)
    assert any("data-api.binance.vision" in u for u in sess.calls)       # 主地址被拒后自动切换备用地址
    assert len(df) == 5 and (df["open_time"] + MINUTE_MS <= now).all()   # 未收盘的那根被丢弃
    assert df["close"].dtype == float and list(df.columns)[:5] == ["open_time", "open", "high", "low", "close"]
    with pytest.raises(MarketDataError):
        RestClient(cfg, FakeSession(now, fail_hosts=("fapi.binance.com",))).fetch("index", "BTCUSDT", now - 5 * MINUTE_MS, now)


class DeadClient:
    def fetch(self, dataset, symbol, start_ms, now_ms):
        raise MarketDataError("HTTP 451")


def test_live_service_degrades_to_no_trade_when_market_data_unavailable(cfg, tmp_path):
    c = Config.wrap(cfg.to_dict())
    c["paths"]["registry_dir"] = str(tmp_path / "empty_registry")
    store = LiveStore(tmp_path / "live.sqlite")
    svc = LiveService(c, LiveBuffer(c, DeadClient()), store)
    now = 1_700_000_000_000
    recs = svc.run_once(now)
    assert len(recs) == 4                                                # 四项预测照常落库
    assert all(r["decision"] == "NO_TRADE" and r["executable"] == 0 and r["skip_reason"] for r in recs)
    saved = store.latest()
    assert len(saved) == 4 and {(r["symbol"], r["horizon"]) for r in saved} == {(s, h) for s in c.symbols for h in c.horizons}
    hb = store.heartbeat("live")["value"]
    assert hb["errors"] and hb["version"] is None
    svc.run_once(now + MINUTE_MS)                                        # 连续失败不会让服务崩溃
    assert len(store.history(limit=100)) == 8


def test_store_settlement_and_stats(tmp_path):
    st = LiveStore(tmp_path / "s.sqlite")
    base = {"ts": 0, "symbol": "BTCUSDT", "version": "v", "p_up": 0.6, "p_down": 0.4, "p_flat": 0.0, "confidence": 0.6,
            "signal": "HIGHER", "skip_reason": "", "risk_reasons": "", "payout": 0.8, "ev": 0.08, "entry_price": 100.0,
            "latency_ms": 5.0, "created": 0}
    st.insert_predictions([base | {"horizon": 10, "decision": "HIGHER", "executable": 1, "stake": 100.0, "expiry_ts": 600_000},
                           base | {"horizon": 30, "decision": "NO_TRADE", "executable": 0, "stake": None, "expiry_ts": 1_800_000,
                                   "skip_reason": "置信度不足 (0.600 < 阈值 0.650)"}])
    due = st.unsettled(600_000)
    assert len(due) == 1
    st.settle(due[0]["id"], 101.0, "UP", 1, 80.0)
    s = {(r["symbol"], r["horizon"]): r for r in st.stats()}
    assert s[("BTCUSDT", 10)]["n_exec_win"] == 1 and s[("BTCUSDT", 10)]["sim_pnl"] == 80.0
    assert s[("BTCUSDT", 30)]["n_signal"] == 0 and st.skip_reason_counts()[0]["n"] == 1
    assert pd.DataFrame(st.history()).shape[0] == 2
