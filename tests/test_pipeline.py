"""端到端测试：合成数据上跑完整流水线 -> 版本库 -> 实时回放 -> Web API。"""
import numpy as np
import pytest

from becai.config import task_name
from becai.registry import Predictor, Registry
from becai.search.pipeline import STAGES, Pipeline
from becai.utils import load_json
from becai.validation.holdout import HoldoutLockError


@pytest.fixture(scope="module")
def run(cfg):
    pipe = Pipeline(cfg, "test_run")
    pipe.run()
    return pipe


def test_all_stages_complete_and_artifacts_exist(run):
    assert all(run.state["stages"][s]["done"] for s in STAGES)
    for f in ("prepare.json", "leaderboard.json", "candidates.json", "holdout_report.json", "report.md", "register.json", "experiments.sqlite"):
        assert (run.dir / f).exists(), f
    prep = load_json(run.dir / "prepare.json")
    assert prep["leakage"]["passed"] and prep["holdout_start"] > prep["dev_end"]


def test_search_never_touched_holdout(run):
    tl, b = run.timeline, run.bundle
    for kind in ("search", "final"):
        for f in run.folds(kind):
            assert max(f.train.max(), f.es.max(), f.val.max()) < tl.dev_end < tl.holdout_start
    for cid in load_json(run.dir / "candidates.json"):
        z = np.load(run.dir / "oof" / f"{cid}.npz")
        assert z["rows"].max() < tl.dev_end
    assert len(b.ts) - tl.holdout_start == 7 * 1440


def test_feature_research_finds_the_planted_source(run):
    sc = load_json(run.dir / "screen" / "BTCUSDT_10m.json")
    top_sources = [r["name"] for r in sc["by_source"][:2]]
    assert "spot" in top_sources or "futures" in top_sources      # 合成信号藏在主动买入失衡里
    assert any("flow" in g for g in sc["group_ranking"][:5])
    assert min(sc["baseline_skill_1e4"]) > 0


def test_leaderboard_selection_and_simplicity_rule(run):
    board = load_json(run.dir / "leaderboard.json")
    for sym, h in run.tasks:
        tn = task_name(sym, h)
        fams = {r["family"] for r in board[tn]}
        assert "logreg" in fams and len(board[tn]) >= 2
        assert board[tn][0]["oof_logloss"] < np.log(2)
        sel = load_json(run.dir / "selection" / f"{tn}.json")
        assert sel["chosen"]["members"] and abs(sum(sel["chosen"]["weights"]) - 1) < 1e-6
        assert sel["calibration"]["method"] in ("none", "platt", "isotonic")
        for cmp_ in sel["simplicity_comparisons"]:
            if cmp_["adopt_complex"]:
                assert cmp_["lower_bound_1e4"] > 0        # 只有显著更优才会采用更复杂的模型
        assert (run.dir / "ablation" / f"{tn}.json").exists()


def test_holdout_report_is_honest_and_complete(run, cfg):
    rep = load_json(run.dir / "holdout_report.json")
    assert rep["ledger"]["sequence"] == 1 and not rep["ledger"]["forced"]
    for tn, r in rep["tasks"].items():
        assert r["breakeven_win_rate"] == pytest.approx(1 / 1.8)
        assert r["direction_all"]["accuracy"] > 0.52                 # 合成信号在测试集上依然有效
        assert r["direction_all"]["logloss"] < r["coin_flip_logloss"]
        s = r["signals"]
        if r["policy"]["threshold"] is None:
            assert s["n"] == 0
        if s["n"] > 50:
            lo, hi = s["block_bootstrap_ci"]
            assert lo <= s["win_rate"] <= hi and (hi - lo) >= (s["wilson_ci"][1] - s["wilson_ci"][0]) * 0.8
            assert "target_assessment" in r and "stability" in r and "entry_delay_1m" in r
    pf = rep["portfolio"]
    assert "不是真实订单结算收益" in pf["disclaimer"]
    assert pf["main"]["n_trades"] <= pf["main"]["n_signals"] and set(pf["scenarios"]) == {"0.70", "0.75", "0.80", "0.85"}
    assert pf["scenarios"]["0.85"]["total_pnl"] >= pf["scenarios"]["0.70"]["total_pnl"]


def test_holdout_cannot_be_reused_by_another_run(cfg, run):
    other = Pipeline(cfg, "test_run_b")
    other.state["data_end_ms"] = run.state["data_end_ms"]
    with pytest.raises(HoldoutLockError):
        other.stage_holdout_guard_only() if hasattr(other, "stage_holdout_guard_only") else _acquire(other)


def _acquire(pipe):
    from pathlib import Path
    from becai.validation.holdout import HoldoutLedger
    b, tl = pipe.bundle, pipe.timeline
    HoldoutLedger(Path(pipe.cfg.path("runs_dir")) / "holdout_ledger.json").acquire(
        (int(b.ts[tl.holdout_start]), int(b.ts[-1])), pipe.run_id, "x", int(pipe.cfg.validation.max_holdout_evaluations))


def test_resume_skips_finished_stages(cfg, run):
    again = Pipeline(cfg, "test_run")
    before = dict(again.state["stages"])
    again.run()
    assert again.state["stages"] == before


def test_registry_predictor_matches_holdout_predictions(cfg, run):
    reg = Registry(cfg.path("registry_dir"))
    assert reg.champion() == "v_test_run"
    pred = Predictor(reg.path(reg.champion()))
    tl = run.timeline
    rows = np.arange(tl.holdout_start, tl.holdout_start + 300)
    out = pred.predict(run.bundle, rows)
    assert set(out) == set(run.tasks)
    for t, o in out.items():
        assert np.isfinite(o["p_up"]).all() and (o["p_up"] > 0).all() and (o["p_up"] < 1).all()
    v2 = reg.add(reg.path("v_test_run"), {"version": "v_copy"})
    reg.promote(v2, "测试")
    assert reg.champion() == "v_copy" and reg.rollback() == "v_test_run" and reg.champion() == "v_test_run"


def test_live_replay_saves_every_prediction_and_matches_offline(cfg, run):
    from becai.live.service import replay
    res = replay(cfg, minutes=45)
    assert res["predictions"] == res["expected"] == 46 * 4          # 每分钟四项预测全部落库（包括未开仓）
    assert res["with_probability"] == res["predictions"]
    assert res["settled"] >= (46 - 30) * 2 + (46 - 10) * 2 - 4      # 到期的预测被自动结算
    assert res["latency_ms_p95"] < 20_000
    # 实时路径的概率与离线批量预测一致（同一份特征代码、同一模型）
    from becai.live.store import LiveStore
    hist = {(h["ts"], h["symbol"], h["horizon"]): h for h in LiveStore(res["db"]).history(limit=1000)}
    pred = Predictor(Registry(cfg.path("registry_dir")).path("v_test_run"))
    b = run.bundle
    rows = np.arange(len(b.ts) - 20, len(b.ts))
    off = pred.predict(b, rows)
    for (sym, h), o in off.items():
        live = np.array([hist[(int(b.ts[r]), sym, h)]["p_up"] + 0 for r in rows])
        np.testing.assert_allclose(live, o["p_up"], atol=2e-3)
    for h in hist.values():
        assert h["decision"] in ("HIGHER", "LOWER", "NO_TRADE")
        if h["decision"] == "NO_TRADE":
            assert h["skip_reason"]                                   # 未开仓必须有原因
        if h["executable"]:
            assert h["payout"] == 0.8 and h["stake"] > 0 and h["ev"] >= cfg.risk.min_edge


def test_web_api(cfg, run):
    from fastapi.testclient import TestClient
    from becai.web.app import create_app
    c = TestClient(create_app(cfg))
    assert "控制台" in c.get("/").text
    ov = c.get("/api/overview").json()
    assert ov["champion"] == "v_test_run" and ov["payout"]["breakeven"] == pytest.approx(1 / 1.8)
    runs = c.get("/api/runs").json()
    assert any(r["run_id"] == "test_run" and r["has_report"] for r in runs)
    assert "BTCUSDT_10m" in c.get("/api/runs/test_run/leaderboard").json()
    assert c.get("/api/runs/test_run/selection", params={"task": "ETHUSDT_30m"}).json()["chosen"]
    assert c.get("/api/runs/test_run/screen", params={"task": "BTCUSDT_10m"}).json()["by_source"]
    assert c.get("/api/runs/test_run/holdout").json()["portfolio"]
    assert len(c.get("/api/runs/test_run/trials").json()) > 5
    assert "训练运行报告" in c.get("/api/runs/test_run/report").text
    assert c.get("/api/runs/..%2F..%2Fetc/state").status_code in (404, 422)
    assert len(c.get("/api/data/status").json()) >= 14
    assert c.get("/api/models").json()["champion"] == "v_test_run"
    assert "cpu_percent" in c.get("/api/gpu").json()
    assert c.get("/api/ledger").json()[0]["run_id"] == "test_run"
    oid = c.post("/api/orders", json={"symbol": "BTCUSDT", "horizon": 10, "direction": "HIGHER", "stake": 100, "payout": 0.8, "entry_price": 40000}).json()["id"]
    assert c.post(f"/api/orders/{oid}/result", json={"result": "WIN"}).json()["pnl"] == pytest.approx(80.0)
    assert c.post("/api/odds", json={"symbol": "BTCUSDT", "horizon": 10, "direction": "HIGHER", "payout": 9}).status_code == 400
    assert c.post("/api/settings", json={"risk": {"not_a_key": 1}}).status_code == 400
    assert c.get("/api/risk").json()["order_stats"]["win"] == 1


def test_web_auth_token_protects_writes(cfg):
    from fastapi.testclient import TestClient
    from becai.config import Config
    from becai.web.app import create_app
    secured = Config.wrap(cfg.to_dict() | {"web": {"host": "127.0.0.1", "port": 1, "auth_token": "s3cret"}})
    c = TestClient(create_app(secured))
    assert c.post("/api/models/rollback").status_code == 401
    assert c.get("/api/overview").status_code == 200
