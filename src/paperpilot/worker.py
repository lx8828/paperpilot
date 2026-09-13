"""摄取 worker：**后台跑** MinerU ∥ 报告链 → 索引，job 状态落盘（2026-09-13）。

为什么要并行（真实依赖，已核实）：
- **默认模式**（`PAPERPILOT_USE_MINERU` 未设）：报告链读 pymupdf，MinerU 产物只经
  `retrieval_chunks` 注入**检索视图** → 两条路**互不依赖**，可并行；
- **`PAPERPILOT_USE_MINERU=1`**：报告链（claims/chunks/figures）也从 MinerU
  `content_list` 取 → 必须**先 MinerU 后报告**，本模块自动退回串行；
- **索引必须在 MinerU 之后**（表块要先注入检索视图才能 encode）→ 两条 lane 都完成后收口。

并发模型（刻意保守）：
- **job 之间串行**：全局 `ThreadPoolExecutor(max_workers=1)` —— 只有一块 GPU，
  两篇同时跑 MinerU 会抢显存；
- **job 内部并行**：局部 `max_workers=2`（MinerU 吃 GPU、报告链吃 API 往返，互补）。

取消 = **标记 + 边界生效**：MinerU 真 `terminate` 子进程；报告链的 LLM 阶段
在**阶段边界**检查信号（单个 LLM 调用中途打断不了，可接受）。

重试 = 再排一次同一篇：链路幂等 + 分阶段缓存，已完成的阶段秒过。
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from paperpilot import jobs

# job 之间串行（GPU 独占）；job 内部再开 2 个 lane
_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest-job")
# job 状态读改写锁（两个 lane 会并发更新同一份 job）
_JOB_LOCK = threading.RLock()


class JobCancelled(Exception):
    """阶段边界检测到取消信号 → 中止 job（已完成的阶段产物保留）。"""


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# ───────────── job 状态读写（所有变更都过锁 + 立即落盘）─────────────


def _mark(job: dict[str, Any], stage: str, *, status: str,
          seconds: float | None = None, error: str = "") -> None:
    with _JOB_LOCK:
        st = job.setdefault("stages", {}).setdefault(stage, {})
        st["status"] = status
        if seconds is not None:
            st["seconds"] = round(float(seconds), 1)
        st["at"] = _now_iso()
        job["stage"] = stage
        if error:
            job["error"] = error
        jobs.save(job)


def _finish(job: dict[str, Any], status: str, *, error: str = "") -> dict[str, Any]:
    with _JOB_LOCK:
        job["status"] = status
        job["stage"] = "done" if status == jobs.STATUS_READY else job.get("stage", "")
        job["finished_at"] = _now_iso()
        job["error"] = error
        t0 = job.get("_t0")
        if t0:
            job["elapsed"] = round(time.time() - float(t0), 1)
        job.pop("_t0", None)
        jobs.save(job)
        jobs.clear_event(job["job_id"])
        return job


def _check_cancel(ev: threading.Event) -> None:
    if ev.is_set():
        raise JobCancelled()


# ───────────── 三条 lane ─────────────


def _lane_mineru(job: dict[str, Any], ev: threading.Event) -> str:
    """MinerU 版面解析（GPU）。返回 status（ok/failed/skipped/cancelled）。"""
    from paperpilot.ingest import mineru_version, read_meta, run_mineru
    from paperpilot.agents.document_cache import is_qasper

    pdf = job["pdf"]
    if is_qasper(pdf):
        _mark(job, "mineru", status="skipped")
        return "skipped"

    _check_cancel(ev)
    _mark(job, "mineru", status="running")
    t0 = time.time()
    m = run_mineru(pdf, force=bool(job.get("force")), cancel=ev)
    st = str(m.get("status") or "failed")
    # 版本 / 命令落 meta（与旧 ingest 行为一致，便于排查与复用判定）
    try:
        meta = read_meta(Path(pdf).stem) or {}
        m["version"] = mineru_version()
        m["cmd"] = m.get("cmd") or ""
        meta["mineru"] = m
        meta["updated_at"] = _now_iso()
        from paperpilot.ingest import _write_meta
        _write_meta(Path(pdf).stem, meta)
    except Exception:  # noqa: BLE001  落 meta 失败不影响主流程
        pass
    _mark(job, "mineru", status=("ok" if st == "ok" else st),
          seconds=time.time() - t0,
          error="" if st == "ok" else str(m.get("reason") or ""))
    return st


def _lane_report(job: dict[str, Any], ev: threading.Event) -> None:
    """报告链（pymupdf + LLM）。

    子阶段耗时由 `on_stage` 回调在**阶段边界**打点：进入新阶段时把**上一阶段**收尾成
    `ok` 并记耗时（否则所有子阶段会永远停在 `running`），全部结束后再收尾最后一个。
    """
    from paperpilot.pipeline import process_pdf

    pdf = job["pdf"]
    _check_cancel(ev)
    _mark(job, "report", status="running")
    t0 = time.time()
    state: dict[str, Any] = {"prev": "", "t": t0}

    def _tick(name: str) -> None:
        _check_cancel(ev)                     # 阶段边界协作式取消
        now = time.time()
        prev = state["prev"]
        if prev:
            _mark(job, f"report:{prev}", status="ok", seconds=now - state["t"])
        state["prev"] = name
        state["t"] = now
        _mark(job, f"report:{name}", status="running")

    process_pdf(pdf, force=bool(job.get("force")), verbose=False, on_stage=_tick)
    if state["prev"]:
        _mark(job, f"report:{state['prev']}", status="ok", seconds=time.time() - state["t"])
    _mark(job, "report", status="ok", seconds=time.time() - t0)


def _lane_index(job: dict[str, Any], ev: threading.Event) -> None:
    """向量索引收口（问答首次提问不再现建）。必须在 MinerU 之后。"""
    from paperpilot.agents.embedder import ChunkIndex

    _check_cancel(ev)
    _mark(job, "index", status="running")
    t0 = time.time()
    idx = ChunkIndex(job["pdf"])
    idx.vectors()                             # 触发 encode + 落 cvec 缓存
    _mark(job, "index", status="ok", seconds=time.time() - t0)


# ───────────── job 主流程 ─────────────


def _run_job(job: dict[str, Any]) -> None:
    ev = jobs.cancel_event(job["job_id"])
    pdf = job["pdf"]
    # ⚠️ **整段都在 try 里**：任何异常（含最初的落盘）都必须变成 job 的 `failed`
    # 终态。否则 job 会永远停在 `queued/running`，而前端只会一直转圈（2026-09-13
    # 首轮自测就踩到过：executor 会静默吞掉 `_run_job` 的异常）。
    try:
        with _JOB_LOCK:
            job["status"] = jobs.STATUS_RUNNING
            job["started_at"] = _now_iso()
            job["_t0"] = time.time()          # 内部字段，_finish 时移除
            jobs.save(job)

        serial = os.environ.get("PAPERPILOT_USE_MINERU") == "1"   # 同源模式：必须先 MinerU
        mineru_status = "skipped"
        if serial:
            mineru_status = _lane_mineru(job, ev)
            _lane_report(job, ev)
        else:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"lane-{pdf[:8]}") as ex:
                f_m = ex.submit(_lane_mineru, job, ev)
                f_r = ex.submit(_lane_report, job, ev)
                errs: list[BaseException] = []
                for f in (f_m, f_r):
                    try:
                        f.result()
                    except BaseException as e:  # noqa: BLE001  收集后统一处理
                        errs.append(e)
                if errs:
                    raise errs[0]
                mineru_status = "ok" if not f_m.cancelled() else "cancelled"

        # 索引：MinerU 失败也无妨（检索视图退化为纯 pymupdf），但问答会被闸门拦，
        # 此时**不建索引**（省 3~10s，避免给一份不可问答的论文做无用功）。
        if mineru_status in ("ok", "skipped"):
            _lane_index(job, ev)
        else:
            _mark(job, "index", status="skipped")

        _finish(job, jobs.STATUS_READY)

    except JobCancelled:
        _mark(job, job.get("stage") or "", status="cancelled")
        _finish(job, jobs.STATUS_CANCELLED, error="已取消（已完成的阶段产物保留）")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _finish(job, jobs.STATUS_FAILED, error=f"{type(e).__name__}: {e}")


def _guard(fut: Any, job: dict[str, Any]) -> None:
    """兜底：executor 会**静默吞掉** `_run_job` 的异常 → 这里把它落成 job 的 `failed`。

    没有这层，任何意外（连 job 落盘都可能失败）都会让前端**永远转圈**——
    这是异步化最危险的失败模式（用户不知道发生了什么，也没有"重试"可点）。
    """
    try:
        exc = fut.exception()
    except Exception:  # noqa: BLE001  future 被取消等
        exc = None
    if exc is None:
        return
    cur = jobs.load(job.get("job_id", "")) or job
    if cur.get("status") in jobs.ACTIVE_STATUSES:
        _finish(cur, jobs.STATUS_FAILED,
                error=f"worker 异常：{type(exc).__name__}: {exc}")


def submit(pdf: str, *, force: bool = False, note: str = "") -> dict[str, Any]:
    """提交摄取 job（**立即返回**）；同一篇已有未结束 job → 直接返回它（不重复排）。"""
    jobs.recover_interrupted()
    with _JOB_LOCK:
        active = jobs.find_active(pdf)
        if active is not None:
            return active
        job = jobs.new_job(pdf, force=force)
        job["note"] = note
        jobs.save(job)
        fut = _EXEC.submit(_run_job, job)
        fut.add_done_callback(lambda f: _guard(f, job))
        return job


def retry(job_id: str) -> dict[str, Any] | None:
    """重试：重新排一个新 job（**已完成的阶段自动复用**，所以重试很便宜）。"""
    old = jobs.load(job_id)
    if old is None:
        return None
    if old.get("status") in jobs.ACTIVE_STATUSES:
        return old                       # 还在跑：不重复排
    jobs.clear_event(job_id)
    return submit(str(old.get("pdf") or ""), force=bool(old.get("force")),
                  note=str(old.get("note") or ""))


__all__ = ["JobCancelled", "retry", "submit"]
