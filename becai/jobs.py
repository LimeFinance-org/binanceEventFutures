"""任务管理：训练/数据任务以独立子进程运行，与实时预测服务隔离；支持停止、异常恢复与定时调度。

  * 每个任务一个子进程（python -m becai.jobs <job_id>），日志写入 data/logs/job_<id>.log；
  * 训练进程会避开实时服务绑定的 CPU 核心并降低优先级，GPU 默认只给训练用（实时推理走 CPU）；
  * 进程异常退出（断电、OOM、被杀）后状态标记为"中断"，流水线各阶段有检查点，可从断点恢复。
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .utils import get_logger

log = get_logger("becai.jobs")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT, status TEXT, pid INTEGER, run_id TEXT, args TEXT, log_path TEXT, message TEXT,
    attempts INTEGER DEFAULT 0, created INTEGER, started INTEGER, finished INTEGER
);
"""
STATUS_CN = {"queued": "排队中", "running": "运行中", "done": "已完成", "failed": "失败", "stopped": "已停止", "interrupted": "异常中断"}


class JobStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def create(self, type_: str, args: dict, run_id: str | None = None) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO jobs(type,status,args,run_id,created) VALUES(?,?,?,?,?)",
                            (type_, "queued", json.dumps(args), run_id, int(time.time() * 1000)))
            return int(cur.lastrowid)

    def update(self, job_id: int, **kw) -> None:
        if not kw:
            return
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {', '.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), job_id))

    def get(self, job_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            return [self._row(r) for r in c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["args"] = json.loads(d["args"] or "{}")
        d["status_cn"] = STATUS_CN.get(d["status"], d["status"])
        return d


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:        # 排除僵尸进程
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ")[-1][0] != "Z"
    except OSError:
        return True


class JobManager:
    def __init__(self, cfg, config_path: str | None = None):
        self.cfg = cfg
        self.config_path = config_path
        self.store = JobStore(cfg.path("db_path"))
        self.log_dir = Path(cfg.path("data_dir")) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start(self, type_: str, args: dict | None = None) -> int:
        args = dict(args or {})
        if type_ == "train" and any(j["type"] == "train" and j["status"] in ("queued", "running") for j in self.reconcile()):
            raise RuntimeError("已有训练任务在运行，请等待完成或先停止")
        job_id = self.store.create(type_, args, args.get("run_id"))
        log_path = self.log_dir / f"job_{job_id}.log"
        cmd = [sys.executable, "-m", "becai.jobs", str(job_id)]
        env = os.environ.copy()
        if self.config_path:
            env["BECAI_CONFIG"] = str(self.config_path)
        with open(log_path, "ab") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                    cwd=str(Path(__file__).resolve().parent.parent))
        self.store.update(job_id, pid=proc.pid, log_path=str(log_path), status="running", started=int(time.time() * 1000))
        return job_id

    def stop(self, job_id: int) -> bool:
        job = self.store.get(job_id)
        if not job or job["status"] != "running":
            return False
        try:
            os.killpg(job["pid"], signal.SIGTERM)       # 连同并行搜索子进程一起终止
        except ProcessLookupError:
            pass
        self.store.update(job_id, status="stopped", finished=int(time.time() * 1000), message="用户手动停止（可恢复）")
        return True

    def reconcile(self) -> list[dict]:
        """把"进程已不存在但状态仍为运行中"的任务标记为异常中断。"""
        jobs = self.store.list(200)
        for j in jobs:
            if j["status"] == "running" and not _alive(j["pid"]):
                self.store.update(j["id"], status="interrupted", finished=int(time.time() * 1000), message="进程异常退出，可从断点恢复")
                j["status"], j["status_cn"], j["message"] = "interrupted", STATUS_CN["interrupted"], "进程异常退出，可从断点恢复"
        return jobs

    def resume(self, job_id: int) -> int:
        """从断点恢复：复用同一个 run_id，已完成的阶段与试验不会重跑。"""
        job = self.store.get(job_id)
        if not job or job["type"] != "train" or not job.get("run_id"):
            raise RuntimeError("只有带 run_id 的训练任务可以恢复")
        new_id = self.start("train", job["args"] | {"run_id": job["run_id"]})
        self.store.update(new_id, attempts=int(job.get("attempts") or 0) + 1)
        return new_id

    def tail(self, job_id: int, lines: int = 200) -> str:
        job = self.store.get(job_id)
        if not job or not job.get("log_path") or not Path(job["log_path"]).exists():
            return ""
        with open(job["log_path"], "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 60_000))
            return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-lines:])


def _isolate_from_live(cfg) -> None:
    """训练进程避开实时服务的 CPU 核心，并降低调度优先级。"""
    try:
        live = set(cfg.live.cpu_affinity or [])
        allc = os.sched_getaffinity(0)
        rest = allc - live
        if live and len(rest) >= 2:
            os.sched_setaffinity(0, rest)
        os.nice(5)
    except (AttributeError, OSError):
        pass


def run_job(job_id: int) -> int:
    from .cli import main as cli_main
    from .config import load_config
    cfg = load_config()
    store = JobStore(cfg.path("db_path"))
    job = store.get(job_id)
    if job is None:
        return 2
    a = job["args"]
    try:
        if job["type"] == "train":
            _isolate_from_live(cfg)
            argv = ["train", "--job-id", str(job_id)]
            if a.get("run_id"):
                argv += ["--run-id", a["run_id"]]
            if a.get("until"):
                argv += ["--until", a["until"]]
            if a.get("force_holdout"):
                argv += ["--force-holdout"]
        elif job["type"] == "data-sync":
            argv = ["data-sync"]
        else:
            raise ValueError(f"未知任务类型 {job['type']}")
        rc = cli_main(argv)
        store.update(job_id, status="done" if rc == 0 else "failed", finished=int(time.time() * 1000),
                     message="" if rc == 0 else f"退出码 {rc}")
        return rc
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        store.update(job_id, status="failed", finished=int(time.time() * 1000), message=f"{type(exc).__name__}: {exc}"[:500])
        return 1


def run_scheduler(cfg, poll_seconds: int = 60, max_auto_resume: int = 3) -> None:
    """长期运行的调度器：每日数据更新、定期再训练、训练异常自动恢复。"""
    mgr = JobManager(cfg, os.environ.get("BECAI_CONFIG"))
    log.info("调度器启动：每日 %02d:00 UTC 更新数据，每 %s 天再训练", int(cfg.scheduler.data_update_cron_hour_utc), cfg.scheduler.retrain_every_days)
    while True:
        try:
            jobs = mgr.reconcile()
            now = time.time()
            gm = time.gmtime(now)
            day_start = (int(now) // 86400) * 86400 * 1000
            running = [j for j in jobs if j["status"] in ("running", "queued")]
            # 1) 异常中断的训练自动恢复
            for j in jobs:
                if j["type"] == "train" and j["status"] == "interrupted" and int(j.get("attempts") or 0) < max_auto_resume \
                        and not any(r["type"] == "train" for r in running):
                    later = [x for x in jobs if x["id"] > j["id"] and x["type"] == "train" and x.get("run_id") == j.get("run_id")]
                    if not later and j.get("run_id"):
                        log.info("自动恢复训练任务 #%d (run %s)", j["id"], j["run_id"])
                        mgr.resume(j["id"])
                        running.append(j)
            # 2) 每日数据更新
            synced_today = any(j["type"] == "data-sync" and (j["created"] or 0) >= day_start for j in jobs)
            if gm.tm_hour >= int(cfg.scheduler.data_update_cron_hour_utc) and not synced_today and not any(r["type"] == "data-sync" for r in running):
                mgr.start("data-sync")
            # 3) 定期再训练（在当日数据更新完成之后）
            every = int(cfg.scheduler.retrain_every_days or 0)
            if every > 0 and not any(r["type"] == "train" for r in running):
                trains = [j for j in jobs if j["type"] == "train" and j["status"] == "done"]
                last = max((j["finished"] or 0 for j in trains), default=0)
                sync_done = any(j["type"] == "data-sync" and j["status"] == "done" and (j["created"] or 0) >= day_start for j in jobs)
                pending = any(j["type"] == "train" and j["status"] in ("interrupted", "stopped", "failed") and (j["created"] or 0) > last for j in jobs)
                if sync_done and not pending and now * 1000 - last >= every * 86400 * 1000:
                    mgr.start("train", {})
        except Exception as exc:  # noqa: BLE001
            log.warning("调度循环异常: %s", exc)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    sys.exit(run_job(int(sys.argv[1])))
