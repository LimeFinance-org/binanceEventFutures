import numpy as np

from becai.labels import DOWN, INVALID, TIE, UP, approximation_report, make_labels


def test_labels_use_entry_at_decision_close_and_expiry_h_minutes_later():
    c = np.array([100, 101, 100, 100, 99, 105, np.nan, 100.0])
    y, mv = make_labels(c, 2)
    assert list(y[:4]) == [TIE, DOWN, DOWN, UP]          # 100->100 平局, 101->100 跌, 100->99 跌, 100->105 涨
    assert y[4] == INVALID and y[5] == DOWN              # 到期价缺失 -> 无效
    assert list(y[-2:]) == [INVALID, INVALID]            # 末尾没有未来价格
    assert abs(mv[3] - 0.05) < 1e-12


def test_entry_delay_shifts_the_contract_window():
    c = np.array([100, 90, 95, 99, 80.0, 81])
    y0, _ = make_labels(c, 2, 0)
    y1, _ = make_labels(c, 2, 1)
    assert y0[0] == DOWN and y1[0] == UP                  # 延迟后入场价 90，到期 99


def test_label_never_uses_data_beyond_expiry():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(1e-3 * rng.standard_normal(500)))
    y_full, _ = make_labels(c, 10)
    c2 = c.copy()
    c2[300:] *= 5                                          # 修改 300 之后的价格
    y_mod, _ = make_labels(c2, 10)
    assert (y_full[:290] == y_mod[:290]).all()             # t+10 < 300 的标签不受影响


def test_approximation_report(panels):
    rep = approximation_report(panels["BTCUSDT"].iloc[:20000], [10, 30])
    h10 = rep["horizons"]["10"]
    assert 0.4 < h10["up_rate"] < 0.6 and 0.7 < h10["agree_delay_1m"] <= 1.0 and h10["agree_index_vs_spot"] > 0.9
    assert rep["horizons"]["30"]["agree_delay_1m"] > h10["agree_delay_1m"]   # 期限越长，对入场时刻越不敏感
