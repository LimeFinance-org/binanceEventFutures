"""命令行入口: python -m becai.cli <命令>"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .utils import get_logger

log = get_logger("becai.cli")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="becai", description="币安事件合约 AI 训练与预测系统")
    ap.add_argument("--config", help="额外的配置文件 (覆盖 default.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data-sync", help="下载/增量更新历史数据")
    sub.add_parser("data-status", help="查看数据覆盖情况")
    t = sub.add_parser("train", help="运行全自动训练流水线（可断点恢复）")
    t.add_argument("--run-id"); t.add_argument("--until"); t.add_argument("--force-holdout", action="store_true")
    t.add_argument("--job-id", type=int)
    w = sub.add_parser("worker", help="并行搜索子进程（内部使用）")
    w.add_argument("--run-id", required=True); w.add_argument("--study", required=True); w.add_argument("--timeout", type=float, required=True)
    lv = sub.add_parser("live", help="启动每分钟实时预测服务")
    lv.add_argument("--once", action="store_true", help="只运行一轮")
    rp = sub.add_parser("replay", help="用历史数据回放实时流程（验证/演示）")
    rp.add_argument("--minutes", type=int, default=120)
    web = sub.add_parser("web", help="启动中文 Web 控制台")
    web.add_argument("--host"); web.add_argument("--port", type=int)
    web.add_argument("--with-scheduler", action="store_true", help="在同一进程内同时运行调度器（Docker 单容器部署用）")
    sub.add_parser("scheduler", help="启动自动更新调度器（数据更新 + 定期再训练）")
    pm = sub.add_parser("promote", help="把某个模型版本设为线上版本"); pm.add_argument("version")
    sub.add_parser("rollback", help="回退到上一个线上版本")
    sub.add_parser("versions", help="列出模型版本")
    args = ap.parse_args(argv)
    if args.config:
        import os
        os.environ["BECAI_CONFIG"] = args.config      # 让并行搜索子进程、任务子进程继承同一份配置
    cfg = load_config()

    if args.cmd == "data-sync":
        from .data.archive import sync_all
        res = sync_all(cfg)
        errs = [r for r in res if r["status"] == "error"]
        print(f"完成 {len(res)} 项, 错误 {len(errs)}")
        return 1 if errs else 0
    if args.cmd == "data-status":
        from .data.store import coverage
        for r in coverage(cfg):
            print(r)
        return 0
    if args.cmd == "train":
        from .jobs import JobStore
        from .search.pipeline import Pipeline
        pipe = Pipeline(cfg, args.run_id, args.force_holdout)
        if args.job_id:
            JobStore(cfg.path("db_path")).update(args.job_id, run_id=pipe.run_id)
        state = pipe.run(args.until)
        print(json.dumps({"run_id": pipe.run_id, "stages": list(state["stages"])}, ensure_ascii=False))
        return 0
    if args.cmd == "worker":
        from .search.pipeline import Pipeline
        Pipeline(cfg, args.run_id).worker(args.study, args.timeout)
        return 0
    if args.cmd == "live":
        from .live.service import LiveService
        svc = LiveService(cfg)
        svc.run_once() if args.once else svc.run_forever()
        return 0
    if args.cmd == "replay":
        from .live.service import replay
        print(json.dumps(replay(cfg, args.minutes), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "web":
        import uvicorn
        from .web.app import create_app
        if args.with_scheduler:
            import threading
            from .jobs import run_scheduler
            threading.Thread(target=run_scheduler, args=(cfg,), daemon=True).start()
        uvicorn.run(create_app(cfg), host=args.host or cfg.web.host, port=args.port or int(cfg.web.port), log_level="warning")
        return 0
    if args.cmd == "scheduler":
        from .jobs import run_scheduler
        run_scheduler(cfg)
        return 0
    from .registry import Registry
    reg = Registry(cfg.path("registry_dir"))
    if args.cmd == "promote":
        reg.promote(args.version, "命令行手动上线")
    elif args.cmd == "rollback":
        print("已回退到:", reg.rollback())
    elif args.cmd == "versions":
        for v in reg.versions():
            print(("* " if v["is_champion"] else "  ") + v["version"], v.get("holdout_window"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
