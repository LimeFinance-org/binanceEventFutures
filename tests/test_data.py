import numpy as np
import pandas as pd

from becai.data.archive import archive_url, month_iter, parse_kline_csv, SPECS, PRICE_KEEP, TRADE_KEEP
from becai.data.dataset import assemble_panel
from becai.data.store import gap_report
from becai.utils import MINUTE_MS
from datetime import date


def test_parse_kline_with_and_without_header():
    body = "1700000000000,1,2,0.5,1.5,10,1700000059999,15,7,4,6,0\n1700000060000,1.5,2,1,1.2,5,1700000119999,6,3,2,2.4,0\n"
    header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
    a = parse_kline_csv(body.encode(), tuple(TRADE_KEEP))
    b = parse_kline_csv((header + body).encode(), tuple(TRADE_KEEP))
    pd.testing.assert_frame_equal(a, b)
    assert list(a["open_time"]) == [1700000000000, 1700000060000] and a["close"].iloc[1] == 1.2


def test_microsecond_timestamps_normalised():
    body = "1794096000000000,1,2,0.5,1.5,10,1794096059999999,15,7,4,6,0\n"
    df = parse_kline_csv(body.encode(), tuple(PRICE_KEEP))
    assert df["open_time"].iloc[0] == 1794096000000


def test_archive_urls_and_months():
    u = archive_url("https://x/data", SPECS["index"], "BTCUSDT", "monthly", "2024-01")
    assert u == "https://x/data/futures/um/monthly/indexPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    assert archive_url("b", SPECS["metrics"], "ETHUSDT", "daily", "2024-01-02").endswith("metrics/ETHUSDT/ETHUSDT-metrics-2024-01-02.zip")
    assert list(month_iter(date(2023, 11, 15), date(2024, 2, 1))) == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_gap_report():
    t = np.array([0, 1, 2, 5, 6]) * MINUTE_MS
    rep = gap_report(pd.DataFrame({"open_time": t}))
    assert rep["missing"] == 2 and rep["max_gap_minutes"] == 2


def test_panel_is_indexed_by_close_time_and_never_sees_unpublished_data():
    t0 = 1_700_000_040_000 // MINUTE_MS * MINUTE_MS
    idx = pd.DataFrame({"open_time": t0 + np.arange(10) * MINUTE_MS, "open": 1.0, "high": 2.0, "low": 0.5, "close": np.arange(10.0) + 1})
    metrics = pd.DataFrame({"time": [t0 + 5 * MINUTE_MS], "oi": [123.0]})
    funding = pd.DataFrame({"time": [t0 + 3 * MINUTE_MS], "rate": [0.0001], "interval_hours": [8.0]})
    p = assemble_panel({"index": idx, "metrics": metrics, "funding": funding}, t0, t0 + 20 * MINUTE_MS, metrics_lag_minutes=6)
    # 开盘时间为 t0 的K线，在 t0+1min（收盘时刻）才出现在面板里
    assert np.isnan(p.loc[t0, "ix_c"]) and p.loc[t0 + MINUTE_MS, "ix_c"] == 1.0
    # 资金费率在结算时刻之前不可见
    assert np.isnan(p.loc[t0 + 2 * MINUTE_MS, "fr"]) and p.loc[t0 + 3 * MINUTE_MS, "fr"] == 0.0001
    # 持仓量指标需要等发布延迟（6 分钟）之后才可见
    assert np.isnan(p.loc[t0 + 10 * MINUTE_MS, "mt_oi"]) and p.loc[t0 + 11 * MINUTE_MS, "mt_oi"] == 123.0
