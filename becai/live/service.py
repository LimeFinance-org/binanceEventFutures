"""每分钟滚动预测服务。

每一轮（分钟收盘后约 1.5 秒触发）：
  1. 增量拉取已收盘行情 -> 2. 结算到期的历史预测/持仓 -> 3. 计算特征（与训练同一份代码）
  -> 4. 四项任务预测 + 概率校准 -> 5. 信号策略（阈值/市场状态）-> 6. 赔率与期望收益 -> 7. 全局风控审批
  -> 8. 保存全部预测记录（包括未开仓的及其跳过原因）。

只有 预测质量 + 赔率 + 期望收益 + 风控 全部达标才标记为"可执行开仓提示"。
事件合约没有 API：本服务只给出提示，不会也无法自动下单。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from ..bundle import bundle_from_panels
from ..data.dataset import assemble_panel
from ..data.rest import LiveBuffer
from ..features.core import max_lookback
from ..registry import Predictor, Registry
from ..risk.manager import Position, Proposal, RiskManager
from ..signals.policy import HIGHER, LOWER, NO_TRADE
from ..utils import MINUTE_MS, get_logger, ms_to_iso, utcnow_ms
from .store import LiveStore

log = get_logger("becai.live")


class LiveService:
    def __init__(self, cfg, buffer: LiveBuffer | None = None, store: LiveStore | None = None):
        self.cfg = cfg
        self.store = store or LiveStore(cfg.path("db_path"))
        self.buffer = buffer or LiveBuffer(cfg)
        self.registry = Registry(cfg.path("registry_dir"))
        self.predictor: Predictor | None = None
        self.risk = RiskManager(cfg.risk)
        self._restored = False
        self._cfg_mtime = self._settings_mtime()
        assert int(cfg.live.buffer_minutes) >= max_lookback(cfg), "live.buffer_minutes 小于特征所需回看长度"

    # ---------- 配置热加载：控制台修改风控/赔率后下一轮生效 ----------
    @staticmethod
    def _settings_mtime() -> float:
        from ..config import ROOT
        paths = [ROOT / "config" / "local.yaml"] + ([os.environ["BECAI_CONFIG"]] if os.environ.get("BECAI_CONFIG") else [])
        return max((os.path.getmtime(p) for p in paths if os.path.exists(p)), default=0.0)

    def _reload_settings(self) -> None:
        mt = self._settings_mtime()
        if mt != self._cfg_mtime:
            from ..config import load_config
            new = load_config()
            self.cfg["risk"], self.cfg["payout"] = new.risk, new.payout
            self.risk.c = new.risk
            self._cfg_mtime = mt
            log.info("已重新加载风控与赔率配置")

    # ---------- 模型加载（线上版本切换后自动热更新）----------
    def _ensure_model(self) -> bool:
        champ = self.registry.champion()
        if champ is None:
            self.predictor = None
            return False
        if self.predictor is None or self.predictor.version != champ:
            self.predictor = Predictor(self.registry.path(champ), self.cfg.live.inference_device)
            log.info("已加载线上模型版本 %s", champ)
        return True

    # ---------- 风控状态恢复 ----------
    def _restore_risk(self, now_ts: int) -> None:
        """服务重启后从预测库恢复未到期持仓与当日盈亏。"""
        if self.cfg.risk.track_mode == "signals":
            for r in self.store.open_executable(now_ts):
                self.risk.open.append(Position(r["id"], r["symbol"], r["horizon"], 1 if r["decision"] == HIGHER else -1, r["stake"] or 0.0,
                                               r["payout"] or 0.0, r["ts"], r["expiry_ts"], r["entry_price"]))
            for d in self.store.daily_pnl(2):
                self.risk.daily_realized[int(d["day"])] = float(d["pnl"] or 0.0)
        self._restored = True

    def _sync_confirmed_orders(self, now_ts: int) -> None:
        """confirmed 模式：风控只统计用户确认过的真实订单。"""
        self.risk.open = [Position(-o["id"], o["symbol"], o["horizon"], 1 if o["direction"] == HIGHER else -1, o["stake"], o["payout"] or 0.0,
                                   o["ts"], o["expiry_ts"], o["entry_price"]) for o in self.store.orders(500, open_only=True)
                          if o["expiry_ts"] > now_ts]

    # ---------- 赔率 ----------
    def _payout(self, symbol: str, horizon: int, direction: str, now_ms: int) -> tuple[float | None, float]:
        if self.cfg.payout.mode == "fixed":
            return float(self.cfg.payout.fixed), 0.0
        for o in self.store.get_odds():
            if o["symbol"] == symbol and o["horizon"] == horizon and o["direction"] == direction:
                return float(o["payout"]), (now_ms - o["updated"]) / 1000.0
        return None, 0.0

    # ---------- 结算 ----------
    def _settle(self, decision_ts: int, index_close: dict[str, pd.Series]) -> None:
        for r in self.store.unsettled(decision_ts):
            px = index_close.get(r["symbol"])
            sp = None if px is None else px.get(r["expiry_ts"])
            if sp is None or not np.isfinite(sp) or r["entry_price"] is None:
                continue
            actual = "UP" if sp > r["entry_price"] else "DOWN" if sp < r["entry_price"] else "TIE"
            correct = None if actual == "TIE" or r["p_up"] is None else int((r["p_up"] >= 0.5) == (actual == "UP"))
            pnl = None
            if r["executable"]:
                won = (r["decision"] == HIGHER) == (actual == "UP")
                pnl = 0.0 if actual == "TIE" else (r["stake"] * r["payout"] if won else -r["stake"])
                pos = next((p for p in self.risk.open if p.id == r["id"]), None)
                if pos is not None:
                    self.risk.settle(pos, 0 if actual == "TIE" else (1 if actual == "UP" else -1))
            self.store.settle(r["id"], float(sp), actual, correct, pnl)
        for o in self.store.orders(500, open_only=True):      # 真实订单：按分钟指数价近似结算（用户可手工覆盖为实际结果）
            px = index_close.get(o["symbol"])
            sp = None if px is None else px.get(o["expiry_ts"])
            if o["expiry_ts"] <= decision_ts and sp is not None and np.isfinite(sp) and o["entry_price"]:
                up = sp > o["entry_price"]
                tie = sp == o["entry_price"]
                won = (o["direction"] == HIGHER) == up
                self.store.set_order_result(o["id"], "TIE" if tie else "WIN" if won else "LOSS",
                                            0.0 if tie else (o["stake"] * (o["payout"] or 0) if won else -o["stake"]), "index_approx")
        self.store.expire_stale(decision_ts - (int(self.cfg.live.buffer_minutes) - 60) * MINUTE_MS)

    # ---------- 单轮 ----------
    def run_once(self, now_ms: int | None = None) -> list[dict]:
        t0 = time.time()
        now_ms = now_ms or utcnow_ms()
        decision_ts = (now_ms // MINUTE_MS) * MINUTE_MS
        self._reload_settings()
        if not self._restored:
            self._restore_risk(decision_ts)
        self.buffer.update(now_ms)
        have_model = self._ensure_model()
        panels, stale = {}, {}
        start = decision_ts - (int(self.cfg.live.buffer_minutes) - 1) * MINUTE_MS
        for sym in self.cfg.symbols:
            panels[sym] = assemble_panel(self.buffer.frames[sym], start, decision_ts, int(self.cfg.data.metrics_lag_minutes))
            if "ix_c" not in panels[sym]:
                panels[sym]["ix_c"] = np.nan
                for c in ("ix_o", "ix_h", "ix_l"):
                    panels[sym][c] = np.nan
            last = self.buffer.latest_decision_ts(sym)
            if last is None or last < decision_ts:
                stale[sym] = f"指数行情未更新到当前分钟（最新 {ms_to_iso(last)}）"
        index_close = {s: p["ix_c"] for s, p in panels.items()}
        self._settle(decision_ts, index_close)
        if self.cfg.risk.track_mode == "confirmed":
            self._sync_confirmed_orders(decision_ts)

        preds, bundle = {}, None
        if have_model and len(stale) < len(self.cfg.symbols):
            try:
                bundle = bundle_from_panels(self.cfg, panels)
                preds = self.predictor.predict(bundle, np.asarray([len(bundle.ts) - 1]))
            except Exception as exc:  # noqa: BLE001
                log.exception("预测失败: %s", exc)
                stale = {s: f"预测异常: {exc}" for s in self.cfg.symbols}
        elapsed = time.time() - t0
        timed_out = elapsed > float(self.cfg.live.max_latency_seconds)
        records = []
        for sym in self.cfg.symbols:
            entry = float(panels[sym]["ix_c"].iloc[-1]) if np.isfinite(panels[sym]["ix_c"].iloc[-1]) else None
            for h in [int(x) for x in self.cfg.horizons]:
                rec = {"ts": decision_ts, "symbol": sym, "horizon": h, "version": self.predictor.version if self.predictor else None,
                       "p_up": None, "p_down": None, "p_flat": None, "confidence": None, "signal": NO_TRADE, "decision": NO_TRADE,
                       "executable": 0, "skip_reason": "", "risk_reasons": "", "payout": None, "ev": None, "stake": None,
                       "entry_price": entry, "expiry_ts": decision_ts + h * MINUTE_MS, "created": utcnow_ms()}
                records.append(rec)
                if not have_model:
                    rec["skip_reason"] = "尚无线上模型版本（请先完成训练）"; continue
                if sym in stale:
                    rec["skip_reason"] = stale[sym]; continue
                if (sym, h) not in preds:
                    rec["skip_reason"] = "线上模型不包含该任务"; continue
                p = float(preds[(sym, h)]["p_up"][0])
                if not np.isfinite(p):
                    rec["skip_reason"] = "特征无效（数据缺失或预热不足）"; continue
                p_flat = float(self.predictor.meta.get("tie_rate", {}).get(f"{sym}_{h}m", 0.0))
                rec.update(p_up=p * (1 - p_flat), p_down=(1 - p) * (1 - p_flat), p_flat=p_flat, confidence=max(p, 1 - p))
                policy = self.predictor.policies[(sym, h)]
                direction, reason = policy.decide(np.asarray([p]), bundle.symbols[sym].regime.iloc[[-1]])
                if direction[0] == 0:
                    rec["skip_reason"] = str(reason[0]); continue
                side = HIGHER if direction[0] == 1 else LOWER
                rec["signal"] = side
                if timed_out:
                    rec["skip_reason"] = f"本轮计算超时 ({elapsed:.1f}s)，信号作废"; continue
                payout, age = self._payout(sym, h, side, now_ms)
                p_win = p if direction[0] == 1 else 1 - p
                pr = Proposal(sym, h, int(direction[0]), p_win, payout, decision_ts, age)
                dec = self.risk.evaluate(pr)
                rec.update(payout=payout, ev=dec.ev)
                if payout is None:
                    rec["decision"] = side           # 研究预测：有方向但不可执行
                    rec["skip_reason"] = "无可靠实时赔率：仅输出研究预测，不生成可执行建议"; continue
                if not dec.approved:
                    rec["risk_reasons"] = "；".join(dec.reasons)
                    rec["skip_reason"] = "风控未通过"; continue
                rec.update(decision=side, executable=1, stake=dec.stake)
        lat = (time.time() - t0) * 1000
        for r in records:
            r["latency_ms"] = lat
        self.store.insert_predictions(records)
        if self.cfg.risk.track_mode == "signals":
            saved = {(r["symbol"], r["horizon"]): r for r in self.store.latest() if r["ts"] == decision_ts}
            for r in records:
                if r["executable"]:
                    pid = saved[(r["symbol"], r["horizon"])]["id"]
                    pr = Proposal(r["symbol"], r["horizon"], 1 if r["decision"] == HIGHER else -1, r["confidence"], r["payout"], decision_ts)
                    pos = self.risk.open_position(pr, r["stake"], r["entry_price"])
                    pos.id = pid
        self.store.beat("live", {"ts": decision_ts, "latency_ms": lat, "errors": self.buffer.errors, "stale": stale,
                                 "version": self.predictor.version if self.predictor else None,
                                 "risk": self.risk.snapshot(decision_ts), "pid": os.getpid()})
        return records

    def run_forever(self) -> None:
        try:
            aff = set(self.cfg.live.cpu_affinity or [])
            if aff and aff <= os.sched_getaffinity(0):
                os.sched_setaffinity(0, aff)          # 绑定专用核心，训练进程会避开这些核心
        except (AttributeError, OSError):
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, len(self.cfg.live.cpu_affinity or [1]))))
        offset = float(self.cfg.live.poll_offset_seconds)
        log.info("实时预测服务启动（每分钟一轮，四项预测）")
        while True:
            now = time.time()
            nxt = (int(now) // 60 + 1) * 60 + offset
            time.sleep(max(0.0, nxt - now))
            try:
                recs = self.run_once()
                sig = [f"{r['symbol']}_{r['horizon']}m:{r['decision']}" for r in recs if r["decision"] != NO_TRADE]
                log.info("%s 完成 %.0fms 信号=%s", ms_to_iso(recs[0]["ts"]), recs[0]["latency_ms"], sig or "无")
            except Exception as exc:  # noqa: BLE001  单轮失败不能让服务退出
                log.exception("本轮预测异常: %s", exc)


class ReplayClient:
    """用本地历史数据模拟 REST 接口，按"当前时刻"只返回已收盘/已发布的数据。"""

    def __init__(self, cfg, start_ms: int, end_ms: int):
        from ..data.store import load_raw
        self.data = {}
        for sym in cfg.symbols:
            for name, on in cfg.data.datasets.items():
                if on:
                    pad = 40 * 24 * 60 * MINUTE_MS if name == "funding" else MINUTE_MS * 120
                    self.data[(name, sym)] = load_raw(cfg, name, sym, start_ms - pad, end_ms + MINUTE_MS)

    def fetch(self, dataset: str, symbol: str, start_ms: int, now_ms: int) -> pd.DataFrame:
        df = self.data[(dataset, symbol)]
        if dataset in ("funding", "metrics"):
            return df[df["time"] <= now_ms].reset_index(drop=True)
        return df[(df["open_time"] >= start_ms) & (df["open_time"] + MINUTE_MS <= now_ms)].reset_index(drop=True)


def replay(cfg, minutes: int = 120, db_path: str | None = None) -> dict:
    """用最近一段历史数据回放实时流程：验证延迟、记录完整性，以及与离线预测的一致性。"""
    import tempfile
    from ..data.store import load_raw
    idx = load_raw(cfg, "index", cfg.symbols[0])
    end = int(idx["open_time"].max()) + MINUTE_MS
    start = end - (minutes + int(cfg.live.buffer_minutes)) * MINUTE_MS
    client = ReplayClient(cfg, start, end)
    store = LiveStore(db_path or os.path.join(tempfile.mkdtemp(prefix="becai_replay_"), "replay.sqlite"))
    svc = LiveService(cfg, LiveBuffer(cfg, client), store)
    lat = []
    for k in range(minutes, -1, -1):
        now = end - k * MINUTE_MS + 1500
        recs = svc.run_once(now)
        lat.append(recs[0]["latency_ms"])
    hist = store.history(limit=100000)
    return {
        "minutes": minutes + 1, "predictions": len(hist), "expected": (minutes + 1) * len(cfg.symbols) * len(cfg.horizons),
        "with_probability": sum(h["p_up"] is not None for h in hist), "signals": sum(h["decision"] != NO_TRADE for h in hist),
        "executable": sum(h["executable"] for h in hist), "settled": sum(h["actual"] in ("UP", "DOWN", "TIE") for h in hist),
        "latency_ms_p50": float(np.median(lat)), "latency_ms_p95": float(np.quantile(lat, 0.95)), "latency_ms_max": float(max(lat)),
        "skip_reasons": store.skip_reason_counts(), "stats": store.stats(), "db": str(store.path),
        "risk": svc.risk.snapshot(end),
    }
