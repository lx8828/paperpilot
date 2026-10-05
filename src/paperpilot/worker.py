"""摄取 worker：**后台跑** MinerU ∥ 报告链 → 索引，job 状态落盘（2026-09-13）。

为什么要并行（真实依赖，已核实）：
- **旧默认**（`PAPERPILOT_USE_MINERU=0`，现为容灾档）：报告链读 pymupdf，MinerU 产物只经
  `retrieval_chunks` 注入**检索视图** → 两条路**互不依赖**，可并行；
- **默认（2026-09-22 起）**：报告链（claims/chunks/figures）也从 MinerU
  `content_list` 取 → 必须**先 MinerU 后报告**，本模块自动退回串行
  （代价：单篇墙钟 ≈100s → ≈157s，MinerU 的 58s 不再被报告链盖住）；
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
          seconds: float | None = None, error: str = "", note: str = "") -> None:
    """打点一个阶段。

    `note` 是**阶段自己的说明**（如"index 为何被跳过"），与 `error` 不同：
    `error` 会写进 job 顶层 `error`（只有失败/取消才该有），而 `note` 只挂在该
    阶段上、不污染终态语义（ready 的 job 也能带 note）。
    """
    with _JOB_LOCK:
        st = job.setdefault("stages", {}).setdefault(stage, {})
        st["status"] = status
        if seconds is not None:
            st["seconds"] = round(float(seconds), 1)
        if note:
            st["note"] = note
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
    from paperpilot import sources
    from paperpilot.ingest import mineru_version, read_meta, run_mineru

    pdf = job["pdf"]
    if not sources.has_file(pdf):
        _mark(job, "mineru", status="skipped")
        return "skipped"     # 无磁盘文件的源：MinerU 不适用（它读的是 PDF 版面产物）

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
    """向量索引收口（问答首次提问不再现建）。必须在 MinerU 之后。

    ⚠️ **编码前后各查一次取消**（2026-09-14 审查）：`idx.vectors()` 内部的
    `encode_texts()` 是**一次不可中断**的调用（可能几秒~几十秒），它是本 lane 里
    "边界之后"的唯一工作。只查开头 → 用户在编码期间点取消会被吞掉：
    编码跑完照样标 `index: ok`、job 照样变 `ready`。
    中途打断做不到（要改 embedder），所以这里是**在编码结束后立刻再查一次** ——
    取消的代价是"等这次编码跑完"，但**终态必须是 cancelled**。
    """
    from paperpilot.agents.nodes.read_full import index_needed

    _check_cancel(ev)
    # 默认 `fullctx` 读全文只调 `_doc_chunks()`（读**切块**产物）→ **一个向量都不读**
    # （2026-10-03）。eager 建纯属白花（实测 5 篇 21.8s）；真需要向量的路径会**惰性现建**。
    if not index_needed():
        _mark(job, "index", status="skipped",
              note="当前读取器(fullctx)只需要切块；真需要时会惰性现建")
        return

    from paperpilot.agents.embedder import ChunkIndex

    _mark(job, "index", status="running")
    t0 = time.time()
    try:
        idx = ChunkIndex(job["pdf"])
        idx.vectors()                         # 触发 encode + 落 cvec 缓存
        _check_cancel(ev)                     # ← 编码可能耗时几十秒，这里必须再查
    except JobCancelled:
        # 记下"编码跑了多久才被取消"（`_run_job` 的处理器只标状态、不带秒数，
        # 而 `_mark` 不会覆盖已写入的 seconds）——排查"取消为何要等这么久"靠它。
        _mark(job, "index", status="cancelled", seconds=time.time() - t0)
        raise
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

        from paperpilot.agents.document_cache import mineru_skeleton_enabled
        serial = mineru_skeleton_enabled()   # MinerU 骨架（默认）：报告链必须等 MinerU 完成
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
                # ⚠️ **必须取 lane 的返回值**，不能凭"future 没抛异常"推断 ok
                # （2026-09-14 审查）：MinerU 失败是**正常返回** `failed`（不抛异常），
                # 旧写法 `"ok" if not f_m.cancelled() else "cancelled"` 会把它误标 ok
                # → ①白跑一次索引 ②任务状态说错。
                # 另外 `f.cancelled()` 在这里恒为 False（已启动的 future 不会变 cancelled），
                # 所以旧写法连 cancelled 也基本测不到。
                mineru_status = str(f_m.result() or "failed")

        # MinerU lane 报了 cancelled（用户取消，且报告链**恰好已先跑完**）→ 终态也必须是
        # cancelled；否则取消看起来"失效"，job 会变成 ready。
        if mineru_status == "cancelled":
            raise JobCancelled()

        # 索引：**由 `read_full.index_needed()` 决定建不建**（2026-10-03）。
        #   · 默认 `fullctx` → **跳过**：读全文只读切块，向量一个都不读；
        #   · `retrieval` / `set` → 建（块级检索要用）；`PAPERPILOT_EAGER_INDEX=1` 可强制建。
        # 历史上这里还有一层"MinerU ok/skipped/failed 都建"的判断 —— 那条逻辑仍然成立
        # （凡是**决定要建**的情形都一样：解析失败也建，因为问答会降级到 pymupdf 容灾），
        # 只是"要不要建"这件事现在由**读取器**先回答；不建时向量会在需要时惰性现建。
        # cancelled 已在上面 raise，到不了这里。
        _lane_index(job, ev)

        # **写终态前的最后一道门**（2026-09-14 审查）：各 lane 只在各自的阶段边界
        # 检查取消，而这些"边界"到写终态之间仍有工作在跑（索引编码、跳过索引的分支、
        # 最后一个 `_mark` 落盘）。少了这一道，取消信号只要晚到一步就会被吞掉、
        # 任务变成 `ready`（用户点了取消却看到成功）。
        _check_cancel(ev)
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


# ───────────── 「方向 → k 篇」批任务（2026-10-02）─────────────
#
# 为什么复用同一套 `jobs.py` 存储：落盘 / 重读 / 取消 / 中断恢复 / `/api/job/{id}`
# 都是现成且踩过坑的（Windows 读写竞态、executor 吞异常、终态must-be-written…）。
# 方向任务只多了一个 `kind` 字段与 `papers` 列表，没必要再写一套状态机。
#
# 与单篇 job 的两点差别：
#   · **独立 executor**：单篇的 `_EXEC` 只有 1 路（GPU 独占）。方向任务本身在阶段内
#     已串行，但**不该和用户手动上传的摄取抢那一路** —— 否则"点了检索并解析"会把
#     用户自己上传的任务堵在后面。
#   · **`pdf` 键是合成的**（`dir:<sha8>`）：让 `find_active` 能对**同一方向**去重，
#     又不与真实 PDF 名相撞。

_DIR_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="direction-job")


def _direction_key(query: str, phase: str = "all",
                   arxiv_ids: list[str] | None = None) -> str:
    """方向 → 合成 pdf 键（`find_active` / `latest` 用的正是 `pdf` 字段）。

    ★ 必须**带上 `phase`（以及所选 id）**：否则同一个方向上的"只检索"与
    "处理所选"两个 job 会互相去重 —— 第二个提交会被直接吞掉（返回第一个 job）。
    """
    import hashlib
    raw = f"{phase}|{str(query).strip()}|{','.join(sorted(arxiv_ids or []))}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"dir-{phase}:{h}"


def _paper_view(pr: Any) -> dict[str, Any]:
    """`direction.PaperRun` → 可 JSON 落盘的纯 dict（前端每张卡片的数据源）。"""
    return {
        "arxiv_id": pr.arxiv_id,
        "title": pr.title,
        "abstract": (pr.abstract or "")[:400],
        "score": round(float(pr.score or 0.0), 4),
        "rank": int(pr.rank or 0),
        "pdf_name": pr.pdf_name,
        "pdf_url": pr.pdf_url,
        "ok": bool(pr.ok),
        "reason": pr.reason,
        "stages": {k: {"status": v.status, "seconds": round(v.seconds, 1), "note": v.note}
                   for k, v in (pr.stages or {}).items()},
    }


def _run_by_phase(job: dict[str, Any], on_event: Any):
    """按 `job["phase"]` 选入口（2026-10-02 拆两段后新增）。

    · `search`  → `search_papers`：**只跑阶段①**，把 10 篇候选交出去让用户挑
    · `process` → `run_selected`：处理**用户挑的那几篇**（②③④⑤）
    · `all`     → `run_direction`：一条龙（CLI / 旧行为 / 不给用户挑的场景）
    """
    from paperpilot import direction as _dir

    query = str(job.get("query") or "")
    phase = str(job.get("phase") or "all")
    if phase == "search":
        return _dir.search_papers(query, k=int(job.get("k") or _dir.SEARCH_K),
                                  on_event=on_event, verbose=False)
    if phase == "process":
        return _dir.run_selected(query, list(job.get("arxiv_ids") or []),
                                 k=int(job.get("k") or _dir.SEARCH_K),
                                 force=bool(job.get("force")),
                                 on_event=on_event, verbose=False)
    return _dir.run_direction(query, k=int(job.get("k") or _dir.DEFAULT_K),
                              force=bool(job.get("force")),
                              on_event=on_event, verbose=False)


def _run_direction_job(job: dict[str, Any]) -> None:
    """执行「方向」任务：把 `direction.*` 的事件流落成 job 状态。

    ⚠️ **不能在这里 `from paperpilot import direction`** —— 见 `str(job.get("phase"))`
    的分派已挪到 `_run_by_phase`（保持本函数只关心"状态怎么落盘"）。
    """

    ev = jobs.cancel_event(job["job_id"])

    def _upsert(pr: Any) -> None:
        """把一篇的最新状态写进 `job["papers"]`（**调用方须已持有 `_JOB_LOCK`**）。"""
        papers = job.setdefault("papers", [])
        idx = next((i for i, p in enumerate(papers)
                    if p.get("arxiv_id") == pr.arxiv_id), None)
        if idx is None:
            papers.append(_paper_view(pr))
        else:
            papers[idx] = _paper_view(pr)

    def on_event(stage: str, done: int, total: int, note: str = "",
                 paper: Any = None) -> None:
        _check_cancel(ev)                      # 阶段边界协作式取消
        with _JOB_LOCK:
            job["stage"] = stage
            job["progress"] = {"stage": stage, "done": int(done),
                               "total": int(total), "note": str(note or "")}
            st = job.setdefault("stages", {}).setdefault(stage, {})
            # 只把**还没落终态**的阶段标 running（别把已 ok 的阶段倒回 running）
            if st.get("status") not in ("ok", "failed", "skipped"):
                st["status"] = "running"
                st["at"] = _now_iso()
            if paper is not None:
                _upsert(paper)
            jobs.save(job)

    try:
        with _JOB_LOCK:
            job["status"] = jobs.STATUS_RUNNING
            job["started_at"] = _now_iso()
            job["_t0"] = time.time()          # 内部字段，_finish 时移除
            jobs.save(job)

        r = _run_by_phase(job, on_event)

        with _JOB_LOCK:
            for s in r.steps:                 # 5 个阶段的终态与耗时
                st = job.setdefault("stages", {}).setdefault(s.name, {})
                st["status"] = s.status
                st["seconds"] = round(s.seconds, 1)
                st["at"] = _now_iso()
                if s.note:
                    st["note"] = s.note
            for pr in r.papers:
                _upsert(pr)
            job["reason"] = r.reason
            job["n_usable"] = len(r.usable)

        if not r.ok:
            _finish(job, jobs.STATUS_FAILED, error=r.reason or "方向流程失败")
            return
        _check_cancel(ev)                     # 写终态前的最后一道门（同单篇）
        _finish(job, jobs.STATUS_READY)

    except JobCancelled:
        _mark(job, job.get("stage") or "", status="cancelled")
        _finish(job, jobs.STATUS_CANCELLED, error="已取消（已完成的阶段产物保留）")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _finish(job, jobs.STATUS_FAILED, error=f"{type(e).__name__}: {e}")


def submit_direction(query: str, *, k: int = 5, force: bool = False,
                     phase: str = "all",
                     arxiv_ids: list[str] | None = None) -> dict[str, Any]:
    """提交「方向」任务（**立即返回 job**）。

    Args:
        query: 研究方向。
        k: 候选数（`phase="search"`/`"all"` 时是阶段①取回几篇）。
        force: MinerU 与报告链全链路重跑。
        phase: `"search"`（只跑阶段①检索，出候选清单）/ `"process"`（处理
            `arxiv_ids` 指定的那几篇）/ `"all"`（一条龙）。
        arxiv_ids: 仅 `phase="process"` 用 —— 用户在候选里挑中的 arXiv id。

    同一 `(方向, phase, 所选 id)` 已有未结束的 job → 直接返回它（不重复排队、
    也不重复花钱）。
    """
    ids = [str(x).strip() for x in (arxiv_ids or []) if str(x).strip()]
    jobs.recover_interrupted()
    key = _direction_key(query, phase=phase, arxiv_ids=ids)
    with _JOB_LOCK:
        active = jobs.find_active(key)
        if active is not None:
            return active
        job = jobs.new_job(key, force=force)
        job.update({
            "kind": "direction",
            "phase": str(phase or "all"),
            "query": str(query).strip(),
            "k": int(k),
            "arxiv_ids": ids,
            "papers": [],                     # 每篇的状态（前端每张卡片的数据源）
            "progress": {"stage": "", "done": 0, "total": 0, "note": ""},
            "note": f"方向：{str(query).strip()[:60]}",
        })
        jobs.save(job)
        fut = _DIR_EXEC.submit(_run_direction_job, job)
        fut.add_done_callback(lambda f: _guard(f, job))
        return job


# ───────────── 多答案开放域检索（2026-10-05）：**与方向流程完全分开的一条线** ─────────────
# 为什么必须走 job（不能同步请求）：实测建索引编码 **50 篇 ≈ 156 秒**（bge-m3），
# 加上逐篇判定 ~20s → 一次 20 秒~3 分钟。同步 HTTP 会超时。
_MA_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="multianswer-job")

# 阶段键（前端据此渲染进度）。★ `chunk` 只在"语料是**刚取料下来的 PDF**"时出现
# （2026-10-05 接通 ②→③ 之后），"用现成切块"那条路仍然只有 index / run 两步。
MA_STAGES = ("chunk", "index", "run")


def _ma_key(question: str, *, n: int, corpus: str, pdfs: tuple[str, ...] = ()) -> str:
    """job 去重键。同问题 + 同语料 + 同 N 且未结束 → 直接复用（不重复花钱）。

    ★ `pdfs` **必须进 key**：语料是"刚下好的那批 PDF"时，换了 PDF 就是换了语料 ——
      漏进 key 会让"换了篇集"命中旧 job，静默用错语料。
    """
    import hashlib
    raw = f"{question}|{corpus}|{'|'.join(pdfs)}|{n}"
    return f"ma:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}:{n}"


def _run_multianswer_job(job: dict[str, Any]) -> None:
    """执行「多答案开放域检索」：`multianswer.answer()`（薄适配生产 `set_judge`）。

    ⚠️ `answer()` 在**判官未配置**时**抛 RuntimeError**（不静默返回空结果）——
       这里必须让它落成 job 的 `failed`，否则前端会一直转圈（`_guard` 只兜底未捕获异常）。
    """
    ev = jobs.cancel_event(job["job_id"])
    try:
        with _JOB_LOCK:
            job["status"] = jobs.STATUS_RUNNING
            job["started_at"] = _now_iso()
            job["_t0"] = time.time()
            jobs.save(job)

        question = str(job.get("question") or "")
        n = int(job.get("n") or 0)
        corpus = str(job.get("corpus") or "")
        b = int(job.get("b") or 12)
        workers = int(job.get("workers") or 8)

        # ⓪ ★ 语料来源二：**刚取料下来的 PDF**（② 的产物）→ 先**生产切块**成 parquet。
        #   走这条时 `corpus` 是**切块的输出路径**（调用方给，worker 不硬编码仓库路径）。
        #   ★ 切块器与既有簇**同一套**（`ordered_chunks`，见 `multianswer.pipeline`
        #     docstring）—— 否则"换了语料"会同时换掉口径，数字**不可比**。
        pdfs = [str(x) for x in (job.get("pdfs") or []) if str(x).strip()]
        if pdfs:
            _check_cancel(ev)
            _mark(job, "chunk", status="running")
            t = time.time()
            from pathlib import Path as _Path
            from paperpilot.multianswer import pipeline as _P
            pairs = [(_Path(p).stem, _Path(p).name) for p in pdfs]
            _P.chunk_papers(pairs, corpus, verbose=False)
            _mark(job, "chunk", status="ok", seconds=time.time() - t)

        # ① 建索引（★ 耗时 ∝ 语料规模；命中缓存则秒级）
        _check_cancel(ev)
        _mark(job, "index", status="running")
        t0 = time.time()
        from paperpilot.multianswer import CorpusIndex
        idx = CorpusIndex.from_parquet(corpus, n=(n or None), verbose=False)
        _mark(job, "index", status="ok", seconds=time.time() - t0)
        with _JOB_LOCK:
            job["corpus_papers"] = len(idx.pdfs)
            jobs.save(job)

        # ② 检索 + 逐篇判定（生产 set_judge）
        _check_cancel(ev)
        _mark(job, "run", status="running")
        t1 = time.time()
        from paperpilot.multianswer import answer, render
        res = answer(question, idx, n=None, b=b, workers=workers)   # 判全部候选 → 前端可切 N
        _mark(job, "run", status="ok", seconds=time.time() - t1)

        # ③ 落结果（前端只需 `papers[]`：篇 + 理由 + 证据片段）
        cand = [str(c.get("pdf")) for c in (res.get("candidates") or [])]
        order = {p: i for i, p in enumerate(cand)}
        papers = []
        for p in (res.get("papers") or []):
            papers.append({
                "pdf": str(p.get("pdf") or ""),
                "rank": order.get(str(p.get("pdf")), 0) + 1,
                "score": float(p.get("score") or 0.0),
                "why": str(p.get("why") or ""),
                "evidence": list(p.get("evidence") or []),
                "snippet": str(p.get("snippet") or ""),
                "page": int(p.get("page") or 0),
                "extract": str(p.get("extract") or ""),
            })
        papers.sort(key=lambda x: x["rank"] or 999)

        # ★★ 「回答」：把逐篇判定的结果**塞进 LLM 汇总**（2026-10-06 新增）——
        #   这是「多答案展示页」的**主输出**（在此之前只有 `render()` 拼的清单）。
        #   ⚠️ **必须放在 `_JOB_LOCK` 之外**：LLM 是秒级调用，持锁会把其它 job 卡住。
        #   ★ `synth.synthesize()` 内部兜底、**不抛异常**；失败则回退成「依据」清单。
        from paperpilot.multianswer import synth as _synth
        sy = _synth.synthesize(question, res)
        # ★ 「依据」：逐篇判定清单（原有 `render()` 的产物，**保留**，供页面上折叠展开）
        evidence = render(question, res)

        with _JOB_LOCK:
            job["papers"] = papers
            job["n_yes"] = int(res.get("n_yes") or 0)
            job["n_candidates"] = int(res.get("n_papers") or 0)
            job["labels"] = {str(k): str(v) for k, v in (res.get("labels") or {}).items()}
            job["candidates"] = [{"pdf": str(c.get("pdf")), "score": float(c.get("score") or 0.0)}
                                 for c in (res.get("candidates") or [])]
            job["warning"] = str(res.get("warning") or "")     # ★ 全 unclear 的提示，不吞掉
            job["answer"] = str(sy.get("answer") or "")        # ★ 回答（LLM 汇总）
            job["answer_ok"] = bool(sy.get("ok"))
            job["answer_error"] = str(sy.get("error") or "")
            job["evidence_text"] = evidence                    # ★ 依据（可折叠）
            # 兼容旧字段：汇总成功用它，失败回退清单（宁可显示清单，也不白屏）
            job["answer_text"] = job["answer"] if job["answer_ok"] else evidence
            job["retr"] = str(res.get("retr") or "")
            job["win_chars"] = int(res.get("win_chars") or 0)
            jobs.save(job)

        _check_cancel(ev)
        _finish(job, jobs.STATUS_READY)

    except JobCancelled:
        _mark(job, job.get("stage") or "", status="cancelled")
        _finish(job, jobs.STATUS_CANCELLED, error="已取消")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _finish(job, jobs.STATUS_FAILED, error=f"{type(e).__name__}: {e}")


def submit_multianswer(question: str, *, n: int, corpus: str,
                       pdfs: list[str] | None = None,
                       b: int = 12, workers: int = 8) -> dict[str, Any]:
    """提交「多答案开放域检索」任务（**立即返回 job**）。

    Args:
        question: 开放域问题（如"哪些论文报告了显著性检验"）。
        n: **语料规模**（用户说的"选 N"）—— 从多少篇里找。`0` = 不限（用整个语料）。
        corpus: 切块 parquet 路径（需含 `docid` / `text`）。★ 由调用方给，
                worker 不硬编码仓库路径。
                ★★ **`pdfs` 非空时，`corpus` 的含义变成「切块的输出路径」**
                （那批 PDF 会先被生产切块器切好写到这里，再当语料用）。
        pdfs: **语料来源二**：② 取料下载下来的 PDF 文件名列表（`assets/papers/` 下）。
              ★ 给了它 → 走「先切块 → 再检索判定」（多一个 `chunk` 阶段）。
              不给 → 走"现成切块 parquet"（只有 `index` / `run` 两阶段）。
        b / workers: 每篇给判官的块数 / 判官并发（生产默认 12 / 8）。

    同一 `(问题, 语料/篇集, N)` 已有未结束的 job → 直接返回它。
    """
    ids = tuple(str(x).strip() for x in (pdfs or []) if str(x).strip())
    jobs.recover_interrupted()
    key = _ma_key(question, n=int(n or 0), corpus=str(corpus), pdfs=ids)
    with _JOB_LOCK:
        active = jobs.find_active(key)
        if active is not None:
            return active
        job = jobs.new_job(key)
        job.update({
            "kind": "multianswer",
            "question": str(question).strip(),
            "n": int(n or 0),
            "corpus": str(corpus),
            "pdfs": list(ids),
            "b": int(b),
            "workers": int(workers),
            "papers": [],
            "progress": {"stage": "", "done": 0, "total": 0, "note": ""},
            "note": (f"多答案（语料={len(ids)} 篇刚取料的 PDF）：{str(question).strip()[:50]}"
                     if ids else f"多答案：{str(question).strip()[:60]}"),
        })
        jobs.save(job)
        fut = _MA_EXEC.submit(_run_multianswer_job, job)
        fut.add_done_callback(lambda f: _guard(f, job))
        return job


# ─────────── ② 论文取料下载（2026-10-05）：① rag1 拿到 id 之后的下一步 ───────────
# ★ 为什么也必须走 job（不能同步）：取料 = **下载 PDF + MinerU 解析**，而 MinerU
#   实测 **每篇 50~220 秒**（50 篇 61.9 分钟）→ 同步 HTTP 必超时。
_FETCH_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ma-fetch-job")

# 阶段键（前端据此渲染进度）。★ 只有两个：下载 + 解析（MinerU 在 `fetch_papers` 内部做）。
MA_FETCH_STAGES = ("download", "ingest")


def _run_ma_fetch_job(job: dict[str, Any]) -> None:
    """执行「② 论文取料下载」：arxiv-id → `assets/papers/<id>.pdf`（+ MinerU 解析）。

    直接调编排里的 `multianswer.pipeline.fetch_papers`（= `arxiv_fetch.fetch_arxiv`
    + `ingest.ingest`，与 `cli/run_fetch.py --ingest` **同源**）—— **不重写取料逻辑**。

    ⚠️ `fetch_papers` 对失败篇**不抛异常**（返回 `ok=false` + `reason`）→ 这里**原样落进
       `job["papers"]`**，前端逐条看得到「哪篇没下到、为什么」，而不是只给一个"成功 N 篇"。
    """
    ev = jobs.cancel_event(job["job_id"])
    try:
        with _JOB_LOCK:
            job["status"] = jobs.STATUS_RUNNING
            job["started_at"] = _now_iso()
            job["_t0"] = time.time()
            jobs.save(job)

        refs = [{"arxiv_id": str(x)} for x in (job.get("arxiv_ids") or [])]

        _check_cancel(ev)
        _mark(job, "download", status="running")
        t0 = time.time()
        from paperpilot.multianswer import pipeline as P
        rows = P.fetch_papers(refs, force=bool(job.get("force")),
                              ingest=bool(job.get("ingest", True)), verbose=False)
        _mark(job, "download", status="ok", seconds=time.time() - t0)
        _mark(job, "ingest", status="ok")     # MinerU 在 fetch_papers 内逐篇做，这里只归档

        papers = [{"arxiv_id": str(r.get("arxiv_id") or ""),
                   "pdf": str(r.get("pdf") or ""),
                   "docid": str(r.get("docid") or ""),
                   "ok": bool(r.get("ok")),
                   "cached": bool(r.get("cached")),
                   "reason": str(r.get("reason") or "")} for r in rows]
        with _JOB_LOCK:
            job["papers"] = papers
            job["n_ok"] = sum(1 for p in papers if p["ok"])
            jobs.save(job)

        _check_cancel(ev)
        _finish(job, jobs.STATUS_READY)
    except JobCancelled:
        _mark(job, job.get("stage") or "", status="cancelled")
        _finish(job, jobs.STATUS_CANCELLED, error="已取消")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _finish(job, jobs.STATUS_FAILED, error=f"{type(e).__name__}: {e}")


def submit_multianswer_fetch(arxiv_ids: list[str], *, force: bool = False,
                             ingest: bool = True) -> dict[str, Any]:
    """提交「② 论文取料下载」（**立即返回 job**）：arxiv-id 列表 → `assets/papers/`。

    ★ 与 `submit_multianswer`（③ 多答案检索）**刻意分成两个 job**：
      ①→②→③ 一步一个 job，前端能**逐步看到**进度，出问题也能定位在**哪一步**
      （而不是一个长 job 里黑箱）。
    """
    import hashlib

    ids = [str(x).strip() for x in (arxiv_ids or []) if str(x).strip()]
    if not ids:
        raise ValueError("arxiv_ids 不能为空（先跑 ① 检索候选）")
    jobs.recover_interrupted()
    key = f"mafetch:{hashlib.sha1('|'.join(ids).encode('utf-8')).hexdigest()[:8]}:{len(ids)}"
    with _JOB_LOCK:
        active = jobs.find_active(key)
        if active is not None:
            return active
        job = jobs.new_job(key)
        job.update({
            "kind": "ma_fetch",
            "arxiv_ids": ids,
            "force": bool(force),
            "ingest": bool(ingest),
            "papers": [],
            "progress": {"stage": "", "done": 0, "total": len(ids), "note": ""},
            "note": f"② 取料下载：{len(ids)} 篇",
        })
        jobs.save(job)
        fut = _FETCH_EXEC.submit(_run_ma_fetch_job, job)
        fut.add_done_callback(lambda f: _guard(f, job))
        return job


__all__ = ["JobCancelled", "retry", "submit", "submit_direction", "submit_multianswer",
           "submit_multianswer_fetch"]
