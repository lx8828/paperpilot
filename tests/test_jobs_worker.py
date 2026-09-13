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


def test_concurrent_read_never_misses(tmp_assets):
    """**bug 回归（Windows 读写竞态）**：一边写 job、一边读，读**绝不能**读到"没有"。

    旧实现把读异常一律吞成 `None` → 前端轮询偶发 404「任务不存在」；
    `all_jobs()` 读失败就 `continue` → 正在跑的 job 从列表里消失，
    `find_active()`（问答门控）会误判"没有进行中的解析"。
    探针实测：修复前 15230 次并发读里有 **11 次**读到 None。
    """
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    jid = job["job_id"]

    stop = threading.Event()
    misses: list[str] = []
    lock = threading.Lock()

    def _reader():
        while not stop.is_set():
            if jobs.load(jid) is None:
                with lock:
                    misses.append("load→None")
            if not jobs.all_jobs():                 # 目录里明明有 job
                with lock:
                    misses.append("all_jobs 漏掉了它")
            time.sleep(0.001)

    readers = [threading.Thread(target=_reader, daemon=True) for _ in range(3)]
    for t in readers:
        t.start()
    try:
        for i in range(60):
            job["stages"]["mineru"] = {"status": "running", "seconds": float(i)}
            jobs.save(job)
    finally:
        stop.set()
        for t in readers:
            t.join(timeout=5)

    assert not misses, f"并发读丢失 {len(misses)} 次（前端会当成 404）"
    assert jobs.load(jid)["status"] == jobs.STATUS_QUEUED


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


def test_cancel_during_index_is_honoured(tmp_assets, monkeypatch):
    """**bug 回归**：编码期间取消 → 终态必须 `cancelled`，不能是 `ready`。

    `idx.vectors()` 内部的 `encode_texts()` 是**一次不可中断**的调用（几秒~几十秒），
    而它正是索引 lane 里"边界之后"的唯一工作。只在 lane **开头**检查取消，就会出现
    "用户点了取消、任务照样 ready"。这里让假 `ChunkIndex.vectors()` 在编码期间置起
    取消信号 —— 与真实时序一致（不是测试后知后觉地手动置位）。
    """
    from paperpilot.agents import embedder

    box: dict = {}

    class _SlowIndex:
        """假的索引：`vectors()` 模拟"编码跑到一半用户点了取消"。"""

        def __init__(self, pdf: str) -> None:
            box["pdf"] = pdf

        def vectors(self):
            jobs.request_cancel(box["job_id"])
            return None

    monkeypatch.setattr(worker, "_lane_mineru", lambda job, ev: "ok")
    monkeypatch.setattr(worker, "_lane_report",
                        lambda job, ev: worker._mark(job, "report", status="ok"))
    monkeypatch.setattr(embedder, "ChunkIndex", _SlowIndex)     # 走**真实** `_lane_index`

    job = jobs.new_job("a.pdf")
    jobs.save(job)
    box["job_id"] = job["job_id"]

    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert got["status"] == jobs.STATUS_CANCELLED              # 关键：不是 ready
    assert got["stages"]["index"]["status"] == "cancelled"
    assert got["stages"]["index"]["seconds"] >= 0               # 记下"编码跑了多久"


def test_cancel_arriving_after_last_stage_still_cancels(tmp_assets, monkeypatch):
    """**写终态前的最后一道门**：所有阶段都跑完、取消信号才到 → 仍必须 `cancelled`。

    信号刻意放在 `_lane_index` 的**最后一步之后**，覆盖两类真实场景：
    ① 取消晚到一步（编码刚结束、或"跳过索引"那条分支根本没有检查点）；
    ② MinerU 失败 → 不建索引 → 直接 `_finish(ready)` 的路径。
    """
    monkeypatch.setattr(worker, "_lane_mineru", lambda job, ev: "ok")
    monkeypatch.setattr(worker, "_lane_report", lambda job, ev: None)

    def _index_then_cancel(job, ev):
        worker._mark(job, "index", status="ok", seconds=0.01)
        jobs.request_cancel(job["job_id"])                      # 恰好在最后一步之后

    monkeypatch.setattr(worker, "_lane_index", _index_then_cancel)

    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert got["status"] == jobs.STATUS_CANCELLED
    assert "取消" in got["error"]


def test_mineru_failed_then_late_cancel_is_cancelled(tmp_assets, monkeypatch):
    """MinerU 失败 → 跳过索引 → 此时才收到取消 → 终态也必须 `cancelled`。

    这条路径上**没有索引 lane 可当检查点**（`_lane_index` 根本不会被调用），
    只有"写终态前那道门"能拦住它。
    """
    seen = {"index": 0}

    def _fail_then_cancel(job, ev):
        worker._mark(job, "mineru", status="failed")
        jobs.request_cancel(job["job_id"])          # 跳过索引的瞬间用户点了取消
        return "failed"

    def _index(job, ev):                            # 不该被调用
        seen["index"] += 1

    monkeypatch.setattr(worker, "_lane_mineru", _fail_then_cancel)
    monkeypatch.setattr(worker, "_lane_report", lambda job, ev: None)
    monkeypatch.setattr(worker, "_lane_index", _index)

    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert seen["index"] == 0                       # MinerU 失败 → 索引确实没跑
    assert got["status"] == jobs.STATUS_CANCELLED   # 取消没被吞掉



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


# ───────────── MinerU lane 的**真状态**必须被采纳（2026-09-14 审查）─────────────
#
# 并行 lane 里最容易写错的一处：**凭"future 没抛异常"推断 MinerU 成功**。
# `_lane_mineru()` 的失败路径是**正常返回** `failed`（不抛异常），所以
# `"ok" if not f_m.cancelled() else "cancelled"` 这种写法会把失败读成 ok：
# 既白跑一次索引，又把任务状态说错（问答那边倒是被 meta 闸门拦住了，所以不易发现）。


@pytest.fixture
def lane_recorder(monkeypatch):
    """假三条 lane：MinerU 返回**指定状态**，并记录索引 lane 被调了几次。"""
    calls = {"index": 0}

    def _make(mineru_status: str) -> dict:
        def _mineru(job, ev):
            worker._check_cancel(ev)
            worker._mark(job, "mineru", status=mineru_status, seconds=0.01)
            return mineru_status

        def _report(job, ev):
            worker._check_cancel(ev)
            worker._mark(job, "report", status="ok", seconds=0.01)

        def _index(job, ev):
            calls["index"] += 1
            worker._mark(job, "index", status="ok", seconds=0.01)

        monkeypatch.setattr(worker, "_lane_mineru", _mineru)
        monkeypatch.setattr(worker, "_lane_report", _report)
        monkeypatch.setattr(worker, "_lane_index", _index)
        return calls

    return _make


def test_mineru_failed_skips_index(tmp_assets, lane_recorder):
    """**bug 回归**：MinerU 失败（lane 正常返回 `failed`）→ 不建索引 + 状态如实记录。

    报告仍然可用（`ready`），问答由 `qa_blocked_reason()` 明确拦住 —— 这是设计，
    但"索引白跑"和"index 标成 ok"是错的。
    """
    calls = lane_recorder("failed")
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert calls["index"] == 0                                   # 不再白跑索引
    assert got["stages"]["index"]["status"] == "skipped"
    assert "failed" in got["stages"]["index"]["note"]             # 跳过原因可查
    assert got["status"] == jobs.STATUS_READY                     # 报告仍可用


def test_mineru_skipped_still_builds_index(tmp_assets, lane_recorder):
    """`skipped`（QASPER / `PAPERPILOT_MINERU=0` 演示模式）**不是失败** → 索引照建。

    检索视图只是退化为纯 pymupdf，问答照常可用；若把 skipped 也当失败，演示模式
    下问答会失去索引（退化到首次提问现建）。
    """
    calls = lane_recorder("skipped")
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert calls["index"] == 1
    assert got["stages"]["index"]["status"] == "ok"
    assert got["status"] == jobs.STATUS_READY


def test_mineru_ok_builds_index(tmp_assets, lane_recorder):
    calls = lane_recorder("ok")
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert calls["index"] == 1
    assert got["status"] == jobs.STATUS_READY


def test_mineru_cancelled_ends_cancelled(tmp_assets, lane_recorder):
    """lane 报 cancelled（报告链**恰好已先跑完**）→ 终态必须仍是 cancelled。

    否则用户点了取消、前端却等到一个 `ready` ——"取消看起来失效"。
    """
    calls = lane_recorder("cancelled")
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert got["status"] == jobs.STATUS_CANCELLED
    assert calls["index"] == 0


def test_serial_mode_uses_lane_status(tmp_assets, lane_recorder, monkeypatch):
    """`PAPERPILOT_USE_MINERU=1`（同源模式）走串行：状态同样必须如实采纳。"""
    monkeypatch.setenv("PAPERPILOT_USE_MINERU", "1")
    calls = lane_recorder("failed")
    job = jobs.new_job("a.pdf")
    jobs.save(job)
    worker._run_job(job)
    got = jobs.load(job["job_id"])
    assert calls["index"] == 0
    assert got["stages"]["index"]["status"] == "skipped"


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
