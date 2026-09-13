"""摄取任务状态机（`jobs` + `worker`）：提交 / 阶段耗时 / 取消 / 重试 / 中断恢复。

为什么必须有：异步化最危险的失败模式不是"报错"，而是**永远停在 running**
（用户一直转圈、没有重试可点）。这里把每条终态路径都钉住，包括两条真实踩过的坑：
① `os.replace` 在 Windows 上因"目标被读"失败 → job 卡在 queued；
② `ThreadPoolExecutor` **静默吞掉**未捕获异常 → job 卡在 running。
"""
from __future__ import annotations

import threading
import time

import pytest

from paperpilot import jobs, worker


def _wait(job_id: str, limit: float = 20.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < limit:
        cur = jobs.load(job_id) or {}
        if cur.get("status") in (jobs.STATUS_READY, jobs.STATUS_FAILED,
                                 jobs.STATUS_CANCELLED):
            return cur
        time.sleep(0.05)
    return jobs.load(job_id) or {}


# ───────────────────────── jobs：落盘与查询 ─────────────────────────


def test_new_job_and_roundtrip(tmp_assets):
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    got = jobs.load(job["job_id"])
    assert got and got["pdf"] == "a.pdf" and got["status"] == jobs.STATUS_QUEUED
    assert got["stem"] == "a"
    assert jobs.job_path(job["job_id"]).exists()


def test_load_missing_returns_none(tmp_assets):
    assert jobs.load("j-19700101-ffffff") is None


def test_update_persists(tmp_assets):
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    jobs.update(job["job_id"], stage="mineru")
    assert jobs.load(job["job_id"])["stage"] == "mineru"


def test_latest_and_find_active(tmp_assets):
    j1 = jobs.new_job("a.pdf")
    j1["created_at"] = "2026-01-01T00:00:00"
    j1["status"] = jobs.STATUS_FAILED          # 旧 job 已结束 → 不该被当成"进行中"
    jobs.save(j1)
    j2 = jobs.new_job("a.pdf")
    j2["created_at"] = "2026-01-02T00:00:00"
    j2["status"] = jobs.STATUS_READY
    jobs.save(j2)
    assert jobs.latest("a.pdf")["job_id"] == j2["job_id"]
    assert jobs.find_active("a.pdf") is None          # ready 不算"进行中"
    j2["status"] = jobs.STATUS_RUNNING
    jobs.save(j2)
    assert jobs.find_active("a.pdf")["job_id"] == j2["job_id"]


def test_stage_label_never_empty():
    assert jobs.stage_label("mineru")
    assert jobs.stage_label("report:claims")
    assert jobs.stage_label("不认识的阶段")


# ───────────────────────── 取消：标记 + 阶段边界生效 ─────────────────────────


def test_request_cancel_sets_flag_and_event(tmp_assets):
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    jobs.request_cancel(job["job_id"])
    assert jobs.load(job["job_id"])["cancel_requested"] is True
    assert jobs.cancel_event(job["job_id"]).is_set() is True


def test_check_cancel_raises():
    ev = threading.Event()
    worker._check_cancel(ev)                 # 未置位 → 不抛
    ev.set()
    with pytest.raises(worker.JobCancelled):
        worker._check_cancel(ev)


# ───────────────────────── worker：终态与阶段耗时 ─────────────────────────


@pytest.fixture
def stubbed_lanes(monkeypatch):
    """把三条 lane 换成"只打点"的假实现（不碰 PDF / MinerU / LLM）。"""
    def _mineru(job, ev):
        worker._check_cancel(ev)
        worker._mark(job, "mineru", status="running")
        time.sleep(0.01)
        worker._mark(job, "mineru", status="ok", seconds=0.01)
        return "ok"

    def _report(job, ev):
        worker._check_cancel(ev)
        worker._mark(job, "report", status="running")
        worker._mark(job, "report:claims", status="ok", seconds=0.01)
        worker._mark(job, "report", status="ok", seconds=0.02)

    def _index(job, ev):
        worker._check_cancel(ev)
        worker._mark(job, "index", status="ok", seconds=0.01)

    monkeypatch.setattr(worker, "_lane_mineru", _mineru)
    monkeypatch.setattr(worker, "_lane_report", _report)
    monkeypatch.setattr(worker, "_lane_index", _index)


def test_run_job_ready_with_stage_timings(tmp_assets, stubbed_lanes):
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert got["status"] == jobs.STATUS_READY
    assert got["elapsed"] >= 0
    assert got["stages"]["mineru"]["status"] == "ok"
    assert got["stages"]["index"]["status"] == "ok"
    assert got["stages"]["report:claims"]["seconds"] >= 0
    assert got["started_at"] and got["finished_at"]


def test_run_job_cancelled_before_start(tmp_assets, stubbed_lanes):
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    jobs.request_cancel(job["job_id"])
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert got["status"] == jobs.STATUS_CANCELLED
    assert "取消" in got["error"]


def test_run_job_failed_on_lane_error(tmp_assets, monkeypatch):
    def _boom(job, ev):
        raise RuntimeError("模拟 MinerU 爆炸")

    monkeypatch.setattr(worker, "_lane_mineru", _boom)
    monkeypatch.setattr(worker, "_lane_report", lambda job, ev: None)
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert got["status"] == jobs.STATUS_FAILED
    assert "模拟 MinerU 爆炸" in got["error"]


def test_guard_marks_failed_on_silent_exception(tmp_assets):
    """**真实坑的护栏**：executor 会静默吞掉 `_run_job` 的异常 → 必须落成 failed。

    没有这层，任何意外都会让前端永远转圈（无错误、无重试）。
    """
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    fut = worker._EXEC.submit(lambda: 1 / 0)
    worker._guard(fut, job)
    got = jobs.load(job["job_id"])
    assert got["status"] == jobs.STATUS_FAILED
    assert "ZeroDivisionError" in got["error"]


def test_guard_keeps_terminal_status(tmp_assets, stubbed_lanes):
    """已到终态的 job 不该被 guard 覆盖。"""
    job = jobs.new_job("a.pdf")
    job["status"] = jobs.STATUS_READY
    jobs.save(job)
    fut = worker._EXEC.submit(lambda: 1 / 0)
    worker._guard(fut, job)
    assert jobs.load(job["job_id"])["status"] == jobs.STATUS_READY


# ───────────────────────── 中断恢复 / 重试 ─────────────────────────


def test_recover_interrupted(monkeypatch, tmp_assets):
    monkeypatch.setattr(jobs, "_recovered", False)      # 一个进程只扫一次 → 测试里重置
    stuck = jobs.new_job("a.pdf")
    stuck["status"] = jobs.STATUS_RUNNING
    jobs.save(stuck)
    done = jobs.new_job("b.pdf")
    done["status"] = jobs.STATUS_READY
    jobs.save(done)

    n = jobs.recover_interrupted()
    assert n == 1
    assert jobs.load(stuck["job_id"])["status"] == jobs.STATUS_FAILED
    assert "重启" in jobs.load(stuck["job_id"])["error"]
    assert jobs.load(done["job_id"])["status"] == jobs.STATUS_READY


def test_retry_creates_new_job(tmp_assets, stubbed_lanes):
    old = jobs.new_job("a.pdf")
    old["status"] = jobs.STATUS_FAILED
    jobs.save(old)
    new = worker.retry(old["job_id"])
    assert new and new["job_id"] != old["job_id"]
    got = _wait(new["job_id"])
    assert got["status"] == jobs.STATUS_READY


def test_retry_while_running_is_noop(tmp_assets):
    running = jobs.new_job("a.pdf")
    running["status"] = jobs.STATUS_RUNNING
    jobs.save(running)
    same = worker.retry(running["job_id"])
    assert same["job_id"] == running["job_id"]


def test_submit_dedupes_active_job(tmp_assets, stubbed_lanes, monkeypatch):
    """同一篇已有未结束 job → 不重复排队（避免两篇抢 GPU）。"""
    first = worker.submit("a.pdf")
    again = worker.submit("a.pdf")
    assert again["job_id"] == first["job_id"]
    _wait(first["job_id"])
