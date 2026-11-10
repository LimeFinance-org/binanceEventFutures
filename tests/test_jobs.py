"""任务管理：子进程运行、异常中断识别、从断点恢复。"""
import os
import signal
import time

import yaml

from becai.config import load_config
from becai.jobs import JobManager
from becai.utils import load_json


def _wait(mgr, job_id, statuses, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = next(j for j in mgr.reconcile() if j["id"] == job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.5)
    raise AssertionError(f"任务 {job_id} 超时，日志:\n{mgr.tail(job_id, 30)}")


def test_train_job_killed_then_resumed_from_checkpoint(cfg, workdir, monkeypatch):
    conf = workdir / "jobs_cfg.yaml"
    d = cfg.to_dict()
    d["paths"]["db_path"] = str(workdir / "data" / "jobs.sqlite")
    d["paths"]["runs_dir"] = str(workdir / "runs_jobs")
    conf.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("BECAI_CONFIG", str(conf))
    jcfg = load_config()
    mgr = JobManager(jcfg, str(conf))

    jid = mgr.start("train", {"run_id": "job_run", "until": "screen"})
    try:
        mgr.start("train", {})
        raise AssertionError("不应允许同时运行两个训练任务")
    except RuntimeError:
        pass
    run_dir = jcfg.path("runs_dir") / "job_run"
    t0 = time.time()
    while not (run_dir / "prepare.json").exists() and time.time() - t0 < 180:      # 等第一个阶段完成
        time.sleep(0.3)
    assert (run_dir / "prepare.json").exists(), mgr.tail(jid, 30)
    os.killpg(mgr.store.get(jid)["pid"], signal.SIGKILL)                           # 模拟进程被杀/断电
    job = _wait(mgr, jid, {"interrupted", "done"})
    if job["status"] == "interrupted":
        assert "断点恢复" in job["message"]
        new = mgr.resume(jid)
        assert mgr.store.get(new)["attempts"] == 1
        done = _wait(mgr, new, {"done", "failed"})
        assert done["status"] == "done", mgr.tail(new, 40)
        assert "阶段 1/8" not in mgr.tail(new, 400)                                # 已完成的阶段没有重跑
    state = load_json(run_dir / "state.json")
    assert state["stages"]["prepare"]["done"] and state["stages"]["screen"]["done"] and "search" not in state["stages"]
    assert mgr.stop(jid) is False                                                  # 已结束的任务不能再停止
