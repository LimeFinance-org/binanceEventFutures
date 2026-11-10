"""中文 Web 控制台后端（FastAPI）。前端为 static/ 下的单页应用，不依赖任何外部 CDN。"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..config import ROOT, deep_merge, load_config
from ..jobs import JobManager
from ..live.store import LiveStore
from ..registry import Registry
from ..search.experiment_db import ExperimentDB
from ..signals.policy import breakeven_win_rate
from ..utils import load_json, utcnow_ms

STATIC = Path(__file__).parent / "static"
EDITABLE = {  # 允许在控制台修改的配置项（风控上限与赔率由人设定）
    "risk": {"stake_usdt", "max_order_usdt", "max_open_positions", "max_total_open_usdt", "max_symbol_open_usdt",
             "max_directional_usdt", "min_spacing_minutes", "max_same_direction_streak", "daily_loss_limit_usdt",
             "max_consecutive_losses", "cooldown_minutes", "min_edge", "track_mode"},
    "payout": {"mode", "fixed"},
}


def gpu_status() -> dict:
    out = {"available": False, "gpus": [], "cpu_percent": psutil.cpu_percent(interval=0.1), "cpu_count": psutil.cpu_count(),
           "ram_used_gb": psutil.virtual_memory().used / 1e9, "ram_total_gb": psutil.virtual_memory().total / 1e9,
           "load_avg": list(os.getloadavg())}
    if not shutil.which("nvidia-smi"):
        return out
    try:
        q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        txt = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        for line in txt.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            out["gpus"].append({"index": int(p[0]), "name": p[1], "util": float(p[2]), "mem_used_mb": float(p[3]),
                                "mem_total_mb": float(p[4]), "temp": float(p[5]), "power_w": float(p[6]) if p[6].replace(".", "").isdigit() else None})
        out["available"] = bool(out["gpus"])
        procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout
        out["processes"] = [dict(zip(("pid", "name", "mem_mb"), [x.strip() for x in ln.split(",")])) for ln in procs.strip().splitlines() if ln.strip()]
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def create_app(cfg=None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="币安事件合约 AI 控制台", docs_url="/api/docs", redoc_url=None)
    runs_dir = Path(cfg.path("runs_dir"))
    store = LiveStore(cfg.path("db_path"))
    registry = Registry(cfg.path("registry_dir"))
    settings_path = Path(os.environ.get("BECAI_CONFIG") or ROOT / "config" / "local.yaml")
    jobs = JobManager(cfg, os.environ.get("BECAI_CONFIG"))

    def current_cfg():
        return load_config()

    def auth(x_auth_token: str | None = Header(default=None)):
        token = cfg.web.auth_token
        if token and x_auth_token != token:
            raise HTTPException(401, "需要有效的 X-Auth-Token")

    def run_path(run_id: str) -> Path:
        p = (runs_dir / run_id).resolve()
        if runs_dir.resolve() not in p.parents or not p.is_dir():
            raise HTTPException(404, "运行不存在")
        return p

    # ---------- 总览 ----------
    @app.get("/api/overview")
    def overview():
        c = current_cfg()
        hb = store.heartbeat("live")
        js = jobs.reconcile()
        day0 = (utcnow_ms() // 86_400_000) * 86_400_000
        champ = registry.champion()
        return {
            "now": utcnow_ms(), "champion": champ,
            "champion_meta": next(({k: v.get(k) for k in ("version", "holdout_window", "summary", "created")} for v in registry.versions() if v["version"] == champ), None),
            "live": hb, "live_alive": bool(hb and utcnow_ms() - hb["updated"] < 150_000),
            "latest": store.latest(), "today": store.stats(day0),
            "running_jobs": [j for j in js if j["status"] == "running"],
            "payout": {"mode": c.payout.mode, "fixed": c.payout.fixed, "breakeven": breakeven_win_rate(float(c.payout.fixed))},
            "target_win_rate": c.signal.target_win_rate,
        }

    # ---------- 实时预测 / 历史 ----------
    @app.get("/api/live/latest")
    def live_latest():
        return {"predictions": store.latest(), "heartbeat": store.heartbeat("live")}

    @app.get("/api/live/history")
    def live_history(limit: int = 200, offset: int = 0, symbol: str | None = None, horizon: int | None = None, only_signals: bool = False):
        return store.history(min(limit, 2000), offset, symbol, horizon, only_signals)

    @app.get("/api/live/stats")
    def live_stats(days: int = 0):
        since = 0 if days <= 0 else utcnow_ms() - days * 86_400_000
        return {"by_task": store.stats(since), "skip_reasons": store.skip_reason_counts(since), "daily_pnl": store.daily_pnl(60),
                "note": "方向准确率=全部预测；信号胜率=仅开仓信号；模拟盈亏按固定/录入赔率计算，不是真实订单结算收益。"}

    # ---------- 赔率 / 订单 / 风控 ----------
    @app.get("/api/odds")
    def get_odds():
        c = current_cfg()
        return {"mode": c.payout.mode, "fixed": c.payout.fixed, "manual": store.get_odds(), "max_age_seconds": c.risk.max_odds_age_seconds}

    @app.post("/api/odds", dependencies=[Depends(auth)])
    def set_odds(body: dict):
        payout = float(body["payout"])
        if not 0 < payout < 5:
            raise HTTPException(400, "赔率应为 0~5 之间的小数，例如 0.8 表示赢得本金的 80%")
        store.set_odds(str(body["symbol"]), int(body["horizon"]), str(body["direction"]), payout)
        return {"ok": True}

    @app.get("/api/risk")
    def risk():
        c = current_cfg()
        hb = store.heartbeat("live")
        return {"config": c.risk, "state": (hb or {}).get("value", {}).get("risk"), "orders": store.orders(100), "order_stats": store.order_stats(),
                "note": "真实订单的结算结果默认按分钟指数价近似；可手工更正为 App 中的实际结果。"}

    @app.post("/api/orders", dependencies=[Depends(auth)])
    def add_order(body: dict):
        """用户在币安 App 手动下单后，在此登记真实订单。"""
        h = int(body["horizon"])
        ts = int(body.get("ts") or utcnow_ms())
        oid = store.add_order({"prediction_id": body.get("prediction_id"), "ts": ts, "symbol": body["symbol"], "horizon": h,
                               "direction": body["direction"], "stake": float(body["stake"]), "payout": float(body["payout"]),
                               "entry_price": body.get("entry_price"), "expiry_ts": ts + h * 60_000, "note": body.get("note", "")})
        return {"id": oid}

    @app.post("/api/orders/{order_id}/result", dependencies=[Depends(auth)])
    def order_result(order_id: int, body: dict):
        res = body["result"]
        if res not in ("WIN", "LOSS", "TIE"):
            raise HTTPException(400, "result 必须是 WIN / LOSS / TIE")
        o = next((x for x in store.orders(5000) if x["id"] == order_id), None)
        if not o:
            raise HTTPException(404, "订单不存在")
        pnl = 0.0 if res == "TIE" else (o["stake"] * o["payout"] if res == "WIN" else -o["stake"])
        store.set_order_result(order_id, res, pnl, "user")
        return {"ok": True, "pnl": pnl}

    # ---------- 数据 ----------
    @app.get("/api/data/status")
    def data_status():
        from ..data.store import coverage
        return coverage(current_cfg())

    # ---------- 任务 ----------
    @app.get("/api/jobs")
    def list_jobs():
        return jobs.reconcile()[:50]

    @app.post("/api/jobs", dependencies=[Depends(auth)])
    def start_job(body: dict):
        t = body.get("type")
        if t not in ("train", "data-sync"):
            raise HTTPException(400, "type 必须是 train 或 data-sync")
        args = {k: body[k] for k in ("run_id", "until", "force_holdout") if body.get(k)}
        try:
            return {"id": jobs.start(t, args)}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/jobs/{job_id}/stop", dependencies=[Depends(auth)])
    def stop_job(job_id: int):
        return {"ok": jobs.stop(job_id)}

    @app.post("/api/jobs/{job_id}/resume", dependencies=[Depends(auth)])
    def resume_job(job_id: int):
        try:
            return {"id": jobs.resume(job_id)}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: int, lines: int = 200):
        return jobs.tail(job_id, min(lines, 2000))

    @app.get("/api/gpu")
    def gpu():
        return gpu_status()

    # ---------- 训练运行 ----------
    @app.get("/api/runs")
    def runs():
        out = []
        for d in sorted(runs_dir.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            s = load_json(d, {})
            out.append({"run_id": d.parent.name, "created": s.get("created"), "stages": s.get("stages", {}),
                        "progress": load_json(d.parent / "progress.json", {}), "has_report": (d.parent / "holdout_report.json").exists()})
        return out

    @app.get("/api/runs/{run_id}/{item}")
    def run_item(run_id: str, item: str, task: str | None = None):
        p = run_path(run_id)
        simple = {"prepare": "prepare.json", "progress": "progress.json", "search": "search_status.json", "leaderboard": "leaderboard.json",
                  "holdout": "holdout_report.json", "register": "register.json", "candidates": "candidates.json", "state": "state.json"}
        if item in simple:
            return load_json(p / simple[item], {})
        if item in ("screen", "selection", "ablation"):
            if task:
                return load_json(p / item / f"{Path(task).name}.json", {})
            return {f.stem: load_json(f) for f in sorted((p / item).glob("*.json"))} if (p / item).exists() else {}
        if item == "trials":
            db = p / "experiments.sqlite"
            return ExperimentDB(db).query(run_id, limit=1000) if db.exists() else []
        if item == "report":
            f = p / "report.md"
            return PlainTextResponse(f.read_text(encoding="utf-8") if f.exists() else "报告尚未生成")
        raise HTTPException(404, "未知条目")

    @app.get("/api/ledger")
    def ledger():
        return load_json(runs_dir / "holdout_ledger.json", [])

    # ---------- 模型版本 ----------
    @app.get("/api/models")
    def models():
        return {"champion": registry.champion(), "pointer": load_json(registry.pointer, {}),
                "versions": [{k: v.get(k) for k in ("version", "run_id", "created", "holdout_window", "train_range", "summary", "portfolio", "is_champion")}
                             for v in registry.versions()]}

    @app.post("/api/models/{version}/promote", dependencies=[Depends(auth)])
    def promote(version: str):
        try:
            registry.promote(Path(version).name, "控制台手动上线")
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/models/rollback", dependencies=[Depends(auth)])
    def rollback():
        v = registry.rollback()
        if v is None:
            raise HTTPException(409, "没有可回退的历史版本")
        return {"version": v}

    # ---------- 配置 ----------
    @app.get("/api/config")
    def get_config():
        c = current_cfg().to_dict()
        c["web"]["auth_token"] = "***" if c["web"].get("auth_token") else ""
        return c

    @app.post("/api/settings", dependencies=[Depends(auth)])
    def update_settings(body: dict):
        patch: dict = {}
        for section, keys in EDITABLE.items():
            for k, v in (body.get(section) or {}).items():
                if k not in keys:
                    raise HTTPException(400, f"不允许修改 {section}.{k}")
                if k in ("track_mode", "mode"):
                    if v not in ("signals", "confirmed", "fixed", "manual"):
                        raise HTTPException(400, f"{k} 取值无效")
                else:
                    v = float(v)
                    if v < 0:
                        raise HTTPException(400, f"{k} 不能为负")
                patch.setdefault(section, {})[k] = v
        cur = yaml.safe_load(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(yaml.safe_dump(deep_merge(cur or {}, patch), allow_unicode=True, sort_keys=False), encoding="utf-8")
        return {"ok": True, "note": "已保存，实时服务将在下一轮自动加载", "time": int(time.time() * 1000)}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
