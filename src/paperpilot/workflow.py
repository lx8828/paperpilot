"""Workflow 编排：**问题 → ①检索 → ②抓取 → ③精读 → 报告**。

## 定位

**固定顺序，没有 agent 决策** —— 这就是 workflow 阶段（`design.md` 的"当前 MVP 范围"）。
将来做 agent 时，本模块的 `step_*` 会被包装成 tool，由 LLM 决定调用顺序与参数；
**现在不引入那层**，先把链路跑通。

与 `pipeline.py` 的分工：

    pipeline.process_pdf   = ③ 精读（PDF → 报告），**单篇**，不关心论文从哪来
    workflow.run_query     = ①②③ 串联（问题 → 报告），**整条链**

## 复用 web 的三条 lane（顺序化）

`web/worker.py` 跑的是 **MinerU ∥ process_pdf → 索引**（前两条互不依赖，所以并行）。
这里为了 CLI 可读性**串行**跑，产物完全等价：

    ① search        问题 → top-k 论文（`tools/corpus_search.py`）
    ② fetch         第 N 篇 → assets/papers/*.pdf（`tools/arxiv_fetch.py`）
    ③ mineru        版面解析（GPU）→ out_mineru/<stem>/   ← 失败不致命
    ④ read          报告链（claims/view/skeleton/figures/report）→ *.report.md
    ⑤ index         问答向量索引（可选）→ 之后可在 web 上对这篇提问

## 失败语义

- **②③⑤ 失败不致命**：MinerU 挂掉报告照常生成（只是问答会被闸门拦），索引挂掉同理；
- **①④ 失败致命**：检索空 → 没东西可读；报告链报错 → 没有交付物；
- `on_step` 抛异常 → **直接中止并向上抛**（协作式取消契约，与 `pipeline.process_pdf`
  的 `on_stage` 一致）—— 取消必须能真的停下来。

用法：
    from paperpilot.workflow import run_query
    r = run_query("如何用对比学习做跨语言摘要生成")
    print(r.report_md)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from paperpilot.tools.arxiv_fetch import fetch_arxiv

ROOT = Path(__file__).resolve().parents[2]      # src/paperpilot/workflow.py → 项目根

# 定稿检索参数（改动必须同步更新评测报告）
POOL = 200

OnStep = Callable[[str], None]

_ENV_LOADED = False


def _ensure_env() -> None:
    """加载项目 `.env`（幂等）。

    ⚠️ **必须有，不能依赖调用方记得加载**（2026-09-21 实测踩到）：
    `run_query` 之前有个**隐藏依赖** —— 检索那一步 `import corpus_search` 时
    **顺带**调了 `load_env()`。于是走 `--paper`（跳过检索）时 `.env` 从未被读，
    报告链直接报 `LLM 未配置`（MinerU 白跑 68s）。
    本模块是工作流入口，配置就该在这里保证。
    """
    global _ENV_LOADED
    if not _ENV_LOADED:
        from paperpilot.tools import llm
        llm._load_dotenv(str(ROOT))
        _ENV_LOADED = True


# ── 返回值 ───────────────────────────────────────────────────────────────────

@dataclass
class Step:
    name: str
    status: str                  # ok / skipped / failed
    seconds: float = 0.0
    note: str = ""


@dataclass
class QueryResult:
    """一次「问题 → 报告」的完整结果。失败时 `ok=False`，看 `reason`。"""
    ok: bool
    query: str
    reason: str = ""
    hits: list[Any] = field(default_factory=list)     # list[tools.corpus_search.Paper]
    picked: Any | None = None                          # 实际精读的那篇 Paper
    pdf_name: str = ""
    pdf_path: str = ""
    fetch_cached: bool = False
    mineru: dict[str, Any] = field(default_factory=dict)
    report: Any | None = None                          # pipeline.PaperReport
    report_md: Path | None = None
    steps: list[Step] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    def step(self, name: str) -> Step | None:
        return next((s for s in self.steps if s.name == name), None)


# ── 内部：带计时的步骤 ───────────────────────────────────────────────────────

class _Run:
    """收集各步骤的耗时与状态（每步一个 `Step`）。"""

    def __init__(self, on_step: OnStep | None) -> None:
        self.on_step = on_step
        self.steps: list[Step] = []

    def start(self, name: str) -> float:
        if self.on_step:                     # 抛异常 = 协作式取消，直接向上传播
            self.on_step(name)
        return time.time()

    def done(self, name: str, t0: float, status: str = "ok", note: str = "") -> Step:
        s = Step(name=name, status=status, seconds=time.time() - t0, note=note)
        self.steps.append(s)
        return s


# ── 主入口 ───────────────────────────────────────────────────────────────────

def run_query(query: str, *, k: int = 5, pick: int = 1,
              paper: str | None = None,
              force: bool = False, skip_llm: bool = False, workers: int = 4,
              use_llm: bool = True, pool: int = POOL,
              since: str | None = None, until: str | None = None,
              build_index: bool = True,
              on_step: OnStep | None = None,
              verbose: bool = True) -> QueryResult:
    """跑完整条链：**问题 → top-k → 第 `pick` 篇 → 精读报告**。

    Args:
        query: 自然语言查询。
        k: 检索返回的候选篇数（给用户挑的清单）。
        pick: 精读第几篇（1-based，按检索位次）。越界直接失败，不静默取别的。
        paper: 直接指定 arXiv id（**跳过检索**）—— 便于调试与"我知道我要哪篇"。
        force: True 时 MinerU 与报告链**全链路重跑**（否则复用缓存）。
        skip_llm: True 时报告链只装配已有产物、不调 LLM（缺产物即报错）。
        workers: claims 提取并发数。
        use_llm: 检索的 LLM 精排开关（省 ~0.8s 与费用）。
        pool: 检索进 CE 的候选数。
        since: 只要该日期后**提交**的论文（`2026` / `2026-04` / `2026-04-01`）。
        until: 只要该日期前提交的论文。
        build_index: 是否建问答向量索引（建了才能在 web 上对这篇提问）。
        on_step: `fn(阶段名)`，在**每个阶段开始前**调用；抛异常即中止（取消契约）。
        verbose: 打印各阶段进度。

    Returns:
        `QueryResult`。**已知失败不抛异常**（`ok=False` + `reason`），
        只有 `on_step` 主动抛出的取消异常会穿透。
    """
    _ensure_env()                                 # 见上：不能依赖调用方加载 .env
    log = (lambda m: print(m)) if verbose else (lambda m: None)
    run = _Run(on_step)
    r = QueryResult(ok=False, query=query)

    # ── ① 定位论文：检索 or 直接指定 ─────────────────────────────────
    if paper:
        r.picked = _DirectHit(paper)
        r.hits = [r.picked]
        run.steps.append(Step("search", "skipped", 0.0, f"直接指定 {paper}"))
        log(f"  ① 检索：跳过（直接指定 {paper}）")
    else:
        t0 = run.start("search")
        log(f"  ① 检索（pool={pool}, LLM精排={'开' if use_llm else '关'}）…")
        from paperpilot.tools.corpus_search import search
        sr = search(query, k=max(k, pick), use_llm=use_llm, pool=pool,
                    since=since, until=until)
        r.hits = sr.papers
        note = f"{len(sr.papers)} 篇候选 / {sr.total_ms:.0f}ms"
        # ⚠️ 首次调用会**顺带把索引与两个模型加载进常驻单例**（~25s），
        # 这段时间被算进本步墙钟但不属于"检索"。不说明的话，
        # "search 用了 30 秒"会被误读成检索慢（实测检索只 4 秒）。
        load_s = (time.time() - t0) - sr.total_ms / 1000
        if load_s > 1.5:
            note += f"（含首次加载索引+模型 {load_s:.0f}s）"
        if sr.degraded:
            note += f" | {sr.degraded}"
        run.done("search", t0, note=note)
        log(f"  ① 检索完成：{len(sr.papers)} 篇候选（{sr.total_ms:.0f}ms）")
        if not sr.papers:
            r.reason = "检索没有返回任何结果（检查语料索引是否建好）"
            r.steps = run.steps
            return r
        if not (1 <= pick <= len(sr.papers)):
            r.reason = f"--pick={pick} 越界：只有 {len(sr.papers)} 篇候选"
            r.steps = run.steps
            return r
        r.picked = sr.papers[pick - 1]

    hit = r.picked
    log(f"     选中 #{pick} {hit.arxiv_id} · {hit.title[:70]}")

    # ── ② 抓取全文 ───────────────────────────────────────────────────
    t0 = run.start("fetch")
    log(f"  ② 抓取 {hit.arxiv_id} …")
    f = fetch_arxiv(hit.arxiv_id)
    r.fetch_cached = bool(f.get("cached"))
    if not f.get("ok"):
        run.done("fetch", t0, "failed", str(f.get("reason") or ""))
        r.reason = f"抓取失败：{f.get('reason')}"
        r.steps = run.steps
        return r
    r.pdf_name = str(f["pdf_name"])
    r.pdf_path = str(f["path"])
    run.done("fetch", t0, note=f"{f['bytes'] / 1024:.0f}KB"
                                f"{'（缓存）' if f['cached'] else '（新下载）'}")
    log(f"  ② 抓取完成：{r.pdf_name}（{f['bytes'] / 1024:.0f}KB，"
        f"{'缓存' if f['cached'] else '新下载'}）")

    # ── ③ MinerU 版面解析（失败不致命）────────────────────────────────
    from paperpilot.tools.mock_llm import mineru_enabled
    t0 = run.start("mineru")
    if not mineru_enabled():
        run.done("mineru", t0, "skipped", "PAPERPILOT_MINERU=0")
        r.mineru = {"status": "skipped", "reason": "PAPERPILOT_MINERU=0"}
        log("  ③ MinerU：跳过（PAPERPILOT_MINERU=0）")
    else:
        log("  ③ MinerU 版面解析（GPU，可能要几分钟）…")
        from paperpilot.ingest import run_mineru
        m = run_mineru(r.pdf_name, force=force)
        r.mineru = m
        st = str(m.get("status") or "failed")
        run.done("mineru", t0, "ok" if st == "ok" else st,
                 str(m.get("reason") or ""))
        log(f"  ③ MinerU：{st}（{float(m.get('seconds') or 0):.0f}s）"
            + (f" — {m.get('reason')}" if st != "ok" else ""))

    # ── ④ 报告链（致命）──────────────────────────────────────────────
    t0 = run.start("read")
    log("  ④ 报告链（claims → view → skeleton → figures → report）…")

    def _tick(name: str) -> None:            # 透传子阶段给上层（web worker 同款用法）
        if on_step:
            on_step(f"read:{name}")

    from paperpilot.pipeline import VIEW_DIR, process_pdf
    try:
        r.report = process_pdf(r.pdf_name, force=force, skip_llm=skip_llm,
                               workers=workers, verbose=verbose, on_stage=_tick)
    except Exception as e:  # noqa: BLE001
        run.done("read", t0, "failed", f"{type(e).__name__}: {e}")
        r.reason = f"报告链失败：{type(e).__name__}: {e}"
        r.steps = run.steps
        return r
    md = VIEW_DIR / f"{Path(r.pdf_name).stem}.report.md"
    r.report_md = md if md.exists() else None
    run.done("read", t0, note=md.name if r.report_md else "report.md 未落盘")
    log(f"  ④ 报告链完成（{run.steps[-1].seconds:.0f}s）")

    # ── ⑤ 问答索引（失败不致命）──────────────────────────────────────
    if build_index:
        t0 = run.start("index")
        try:
            log("  ⑤ 建问答索引…")
            from paperpilot.agents.embedder import ChunkIndex
            ChunkIndex(r.pdf_name).vectors()
            run.done("index", t0)
            log(f"  ⑤ 索引完成（{run.steps[-1].seconds:.0f}s）")
        except Exception as e:  # noqa: BLE001
            run.done("index", t0, "failed", f"{type(e).__name__}: {e}")
            log(f"  ⑤ 索引失败（不影响报告）：{type(e).__name__}: {e}")

    r.ok = True
    r.steps = run.steps
    return r


class _DirectHit:
    """`paper=` 直传时的占位命中对象（与 `corpus_search.Paper` 同形）。"""

    def __init__(self, arxiv_id: str) -> None:
        self.arxiv_id = arxiv_id
        self.docid = -1
        self.title = ""
        self.abstract = ""
        self.score = 0.0
        self.rank = 1
        self.stage = "direct"


__all__ = ["POOL", "QueryResult", "Step", "run_query"]
