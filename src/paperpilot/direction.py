"""方向编排：**一个研究方向 → k 篇论文 → 报告 → 索引**（2026-10-02）。

## 三个入口（web 走前两个，一步到位的 `run_direction` 留给 CLI/对照）

    search_papers(query, k=10)     阶段① 检索 → **候选清单**（交给用户挑）
    run_selected(query, arxiv_ids) 用户挑完 → ②③④⑤ 只处理**他选的那几篇**
    run_direction(query, k=5)      ① 检索 k 篇 + ②③④⑤ 一条龙（CLI / 旧行为）

**为什么拆成两段**（2026-10-02 改）：全文直读的上下文预算是有限的（`read_full.py`
的 `S/C ≤ 1`；实测我们这批 **5 篇 = 230,691 字符 ≈ 61.5k token**），所以"最终处理几篇"
该由**用户**按需决定 —— 系统先给 `SEARCH_K` 篇候选，用户挑 ≤ `SELECT_MAX` 篇再往下走
（2026-10-05 起两者一致：**候选 10 篇、最多选 10 篇**；依据见 `SELECT_MAX` 的注释）。

## 为什么"检索 10 篇"零成本

`corpus_search` 的 LLM listwise **本来就只输出前 `N_OUT = 10` 名**（注释原文：
"交付 top-5 只需要这些"）—— 即 **5 是交付选择，10 才是 LLM 的输出预算**。
所以 `search(k=10)` 拿到的正好是那 10 篇 LLM 精排结果，**质量与 5 篇同源**，
不是"前 5 精排 + 后 5 凑数"。

## 与 `workflow.run_query` 的分工

    workflow.run_query      = 问题 → ①检索 → **单篇** ②③④⑤   （CLI / 单篇调试）
    direction.*             = 方向 → ①检索 → **N 篇** ②③④⑤   （web「检索并解析」）

两者共用同一批底层件（`corpus_search.search` / `fetch_arxiv` / `run_mineru` /
`process_pdf` / `ChunkIndex`），只是**推进粒度不同**（见下）。

## 为什么是「按阶段横推」而不是「按篇纵推」

前端过渡页是 **5 个顺序阶段**（检索/取料/解析/报告/索引），每阶段一条总进度。
若按篇纵推（P1 全跑完再 P2），阶段会在 ②③④⑤ 之间来回跳、"第 ③ 步 60%"这种
语义直接崩坏。所以这里**先把 N 篇取完 → 再把 N 篇解析完 → … → 最后建 N 个索引**。

## 并发

**阶段内串行**，沿用 `worker.py` 的保守模型：只有一块 GPU，两篇同时跑 MinerU 会抢显存；
报告链又打 LLM，并发会互相抢限流。取料串行还顺带满足 arXiv 的礼貌间隔
（`fetch_arxiv` 自带 `min_interval`）。

## 失败语义（逐篇独立，沿用 `run_query`）

- **检索失败（或无结果）→ 致命**：整体 `ok=False`，没有可读的东西。
- **单篇失败不致命**：某篇取料/解析/报告失败 → 该篇标 `failed` 并记原因，其余篇继续。
  · 解析失败（MinerU 挂）**照样往下走报告链** —— 报告链会降级到 pymupdf 视图，
    「解析失败」不等于「这篇废了」（与 `worker.py` 的 `_lane_index` 判断同源）。
  · 报告失败 → 该篇没有交付物 → 跳过它的索引。
- 只要有 **≥1 篇报告成功**，整体 `ok=True`（前端据此进工作台）。
- `on_event` 抛异常 → **直接中止并向上抛**（协作式取消契约，与 `pipeline.process_pdf`
  的 `on_stage` 一致）。

用法：
    from paperpilot.direction import search_papers, run_selected
    r = search_papers("如何用对比学习做跨语言摘要生成")      # 10 篇候选
    r2 = run_selected(r.query, [p.arxiv_id for p in r.papers[:3]])
    for p in r2.papers:
        print(p.arxiv_id, p.ok, p.reason)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paperpilot.tools.arxiv_fetch import fetch_arxiv
from paperpilot.workflow import POOL, _ensure_env

DEFAULT_K = 5

# 阶段①给用户挑的候选数。**取 10 是有依据的**：`corpus_search.N_OUT = 10`
# ——LLM listwise 本来就只输出前 10 名，所以 k=10 拿到的是**全部 LLM 精排结果**，
# 一份钱没多花、一篇质量没降。k>10 才会开始混入 CE 顺序兜上来的（质量更低）。
SEARCH_K = 10

# 用户能从候选里选来**实际处理**的最大篇数（2026-10-05：5 → 10）。
# 两条依据：
#   · **不大于候选数**：阶段① 只给 `SEARCH_K`（=10）篇；要选更多就得去取更靠后的名次，
#     而那些名次**不是 LLM 精排的**（见 `corpus_search.N_OUT = 10`）→ 质量掉一档。
#     所以「候选 10 / 最多选 10」是上限的自然形状；
#   · **上下文预算内留足余量**：问答是全文直读，受 `fullctx` 的预算约束。
#     按实测均值 **46,138 字符/篇**：10 篇 ≈ 461k 字符 ≈ **预算(1.2M) 的 38%**；
#     即使篇篇都是实测最长（60,606 字符）也只 ≈ 606k（51%）→ 稳在预算内。
#
# ⚠️ 别把这条上限当成"窗口够大就能随便加"：模型窗口 1M token
#   （`fullctx.CTX_WINDOW_TOKENS`），10 篇只占 **~6%** —— 真正的约束是**质量**
#   （长上下文会退化），不是窗口。要再往上加，应该**先做质量实测**，而不是看窗口余量。
SELECT_MAX = 10

# 过渡页的 5 个阶段：`(键, 文案)`。**顺序即前端展示顺序**，键同时用作
# job 里 `stages` 的字段名与前端的 `data-s`。
STAGES: list[tuple[str, str]] = [
    ("search", "论文检索"),
    ("fetch", "论文取料"),
    ("mineru", "解析切块"),
    ("report", "生成报告"),
    ("index", "建索引"),
]
STAGE_LABEL: dict[str, str] = dict(STAGES)
STAGE_KEYS: list[str] = [k for k, _ in STAGES]


@dataclass
class Step:
    name: str
    status: str = "ok"          # ok / failed / skipped
    seconds: float = 0.0
    note: str = ""


@dataclass
class PaperRun:
    """一篇论文在本轮里的全部状态（前端每张卡片的数据源）。"""
    arxiv_id: str
    title: str = ""
    abstract: str = ""
    score: float = 0.0
    rank: int = 0
    pdf_name: str = ""
    pdf_url: str = ""
    fetch_cached: bool = False
    mineru: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Step] = field(default_factory=dict)
    reason: str = ""            # 失败原因（空 = 目前没问题）

    @property
    def ok(self) -> bool:
        """这篇能不能用（= 报告产出成功，工作台能读）。"""
        return self.stages.get("report", Step("report", "failed")).status == "ok"

    @property
    def stem(self) -> str:
        return Path(self.pdf_name).stem if self.pdf_name else self.arxiv_id


@dataclass
class DirectionResult:
    ok: bool
    query: str = ""
    reason: str = ""
    phase: str = "all"          # search（只检索）/ process（处理所选）/ all（一条龙）
    papers: list[PaperRun] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)      # 各阶段全局耗时

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    @property
    def usable(self) -> list[PaperRun]:
        return [p for p in self.papers if p.ok]


# `on_event(阶段键, 已完成数, 总数, note=…, paper=…)` —— 抛异常 = 取消/中止
OnEvent = Any


def _mk(stage: str, status: str, note: str = "", seconds: float = 0.0) -> Step:
    return Step(name=stage, status=status, seconds=seconds, note=note)


def _emitter(on_event: OnEvent | None):
    def emit(stage: str, done: int, total: int, note: str = "",
             paper: PaperRun | None = None) -> None:
        if on_event:
            on_event(stage, done, total, note=note, paper=paper)
    return emit


def _finisher(r: DirectionResult):
    def finish(name: str, t0: float, status: str = "ok", note: str = "") -> Step:
        s = _mk(name, status, note, time.time() - t0)
        r.steps.append(s)
        return s
    return finish


# ── 阶段① 检索（致命）─────────────────────────────────────────────────────────

def _search_stage(r: DirectionResult, *, query: str, k: int, use_llm: bool,
                  pool: int, emit: Any, finish: Any, log: Any) -> bool:
    """跑阶段①，把候选写进 `r.papers`。返回 False = 致命失败（`r.reason` 已写好）。"""
    t0 = time.time()
    emit("search", 0, 1, note="检索中…")
    log(f"① 检索（k={k}, pool={pool}, LLM精排={'开' if use_llm else '关'}）…")
    try:
        from paperpilot.tools.corpus_search import search
        sr = search(query, k=k, use_llm=use_llm, pool=pool)
    except Exception as e:                          # noqa: BLE001
        r.reason = f"检索失败：{type(e).__name__}: {e}"
        finish("search", t0, "failed", r.reason)
        emit("search", 0, 1, note=r.reason)
        return False

    r.papers = [PaperRun(arxiv_id=str(p.arxiv_id), title=str(p.title or ""),
                         abstract=str(p.abstract or ""), score=float(p.score or 0.0),
                         rank=int(p.rank or 0), pdf_url=str(getattr(p, "pdf_url", "") or ""))
                for p in sr.papers]
    if not r.papers:
        r.reason = ("检索没有返回任何结果（检查语料索引 `retrieval/data/arxiv` "
                    "是否已建好）")
        finish("search", t0, "failed", r.reason)
        emit("search", 0, 1, note=r.reason)
        return False

    n = len(r.papers)
    # ⚠️ 首次调用会顺带把索引与两个模型加载进常驻单例（~60s），这段时间被算进
    #    本步墙钟但不属于"检索"。不说明的话，"检索用了 60 秒"会被误读（实测检索 6s）。
    load_s = (time.time() - t0) - float(sr.total_ms) / 1000.0
    note = f"{n} 篇候选 / {sr.total_ms:.0f}ms"
    if load_s > 1.5:
        note += f"（含首次加载索引+模型 {load_s:.0f}s）"
    if getattr(sr, "degraded", ""):
        note += f" | {sr.degraded}"
    finish("search", t0, note=note)
    emit("search", 1, 1, note=note)
    log(f"① 检索完成：{n} 篇候选")
    return True


# ── 阶段②③④⑤ 处理指定的那几篇 ───────────────────────────────────────────────

def _process_stage(r: DirectionResult, papers: list[PaperRun], *, force: bool,
                   skip_llm: bool, workers: int, build_index: bool,
                   emit: Any, finish: Any, log: Any) -> None:
    """对 `papers` 依次跑 ②取料 → ③解析 → ④报告 → ⑤索引，就地写 `r.ok`/`r.reason`。"""
    n = len(papers)

    # ── ② 取料（单篇失败不致命）────────────────────────────────────
    t0 = time.time()
    log(f"② 取料 {n} 篇 …")
    for i, pr in enumerate(papers):
        emit("fetch", i, n, note=f"{pr.arxiv_id} 下载中…", paper=pr)
        try:
            f = fetch_arxiv(pr.arxiv_id)
        except Exception as e:                      # noqa: BLE001
            f = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        pr.fetch_cached = bool(f.get("cached"))
        if not f.get("ok"):
            pr.reason = f"取料失败：{f.get('reason')}"
            pr.stages["fetch"] = _mk("fetch", "failed", pr.reason)
        else:
            pr.pdf_name = str(f["pdf_name"])
            pr.stages["fetch"] = _mk("fetch", "ok",
                                     f"{f['bytes'] / 1024:.0f}KB"
                                     f"{'（缓存）' if f['cached'] else '（新下载）'}")
        emit("fetch", i + 1, n, note=pr.reason or pr.pdf_name, paper=pr)
    got = [p for p in papers if p.pdf_name]
    finish("fetch", t0, "ok" if got else "failed", f"{len(got)}/{n} 篇拿到 PDF")
    log(f"② 取料完成：{len(got)}/{n}")

    # ── ③ 解析切块（MinerU，失败不致命）────────────────────────────
    t0 = time.time()
    from paperpilot.tools.mock_llm import mineru_enabled
    if not got:
        finish("mineru", t0, "skipped", "没有可解析的 PDF")
        emit("mineru", 0, n, note="跳过（上一步没有拿到 PDF）")
    elif not mineru_enabled():
        for pr in got:
            pr.stages["mineru"] = _mk("mineru", "skipped", "PAPERPILOT_MINERU=0")
        finish("mineru", t0, "skipped", "PAPERPILOT_MINERU=0")
        emit("mineru", len(got), len(got), note="跳过（PAPERPILOT_MINERU=0）")
        log("③ MinerU：整体跳过（PAPERPILOT_MINERU=0）")
    else:
        log(f"③ MinerU 版面解析 {len(got)} 篇（GPU，可能要几分钟）…")
        from paperpilot.ingest import run_mineru
        for i, pr in enumerate(got):
            emit("mineru", i, len(got), note=f"{pr.arxiv_id} 解析中…", paper=pr)
            try:
                m = run_mineru(pr.pdf_name, force=force)
            except Exception as e:                  # noqa: BLE001
                m = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
            pr.mineru = m
            st = str(m.get("status") or "failed")
            pr.stages["mineru"] = _mk("mineru", "ok" if st == "ok" else st,
                                      str(m.get("reason") or "")[:200])
            emit("mineru", i + 1, len(got), note=f"{pr.arxiv_id}: {st}", paper=pr)
        n_ok = sum(1 for p in got if p.stages["mineru"].status == "ok")
        finish("mineru", t0, note=f"{n_ok}/{len(got)} 篇解析成功（失败不致命，报告链会降级）")
        log(f"③ 解析完成：{n_ok}/{len(got)}")

    # ── ④ 报告链（每篇独立；致命性只到"这一篇"）────────────────────
    t0 = time.time()
    from paperpilot.pipeline import process_pdf
    log(f"④ 报告链 {len(got)} 篇（claims → view → skeleton → figures → report）…")
    for i, pr in enumerate(got):
        emit("report", i, len(got), note=f"{pr.arxiv_id} 生成中…", paper=pr)
        sub = {"i": i}

        def _tick(name: str, _pr: PaperRun = pr) -> None:
            # 子阶段透传给上层：前端第 ④ 步据此显示"正在抽主张 / 建骨架 …"
            emit("report", sub["i"], len(got),
                 note=f"{_pr.arxiv_id} · {_SUBSTAGE.get(name, name)}", paper=_pr)

        tt = time.time()
        try:
            process_pdf(pr.pdf_name, force=force, skip_llm=skip_llm,
                        workers=workers, verbose=False, on_stage=_tick)
            pr.stages["report"] = _mk("report", "ok", seconds=time.time() - tt)
        except Exception as e:                      # noqa: BLE001
            pr.stages["report"] = _mk("report", "failed",
                                      f"{type(e).__name__}: {e}"[:200],
                                      seconds=time.time() - tt)
            if not pr.reason:
                pr.reason = f"报告链失败：{type(e).__name__}: {e}"
        emit("report", i + 1, len(got), note=pr.reason or pr.arxiv_id, paper=pr)
    n_rep = sum(1 for p in got if p.ok)
    finish("report", t0, "ok" if n_rep else "failed", f"{n_rep}/{len(got)} 篇报告成功")
    log(f"④ 报告完成：{n_rep}/{len(got)}")

    # ── ⑤ 建索引（失败不致命；只给报告成功的那几篇建）───────────────
    r.ok = n_rep > 0
    if not r.ok:
        r.reason = ("全部论文的报告链都失败了（看每篇的 reason；常见原因："
                    "LLM 未配置 / MinerU 与 pymupdf 都读不出内容）")
        finish("index", time.time(), "skipped", "没有报告成功的篇，无需建索引")
        emit("index", 0, 1, note="跳过（没有可问答的篇）")
        return

    # ⑤ 建索引 —— **只给真需要向量的读取器建**（2026-10-03）。默认 `fullctx` 读全文走
    # `ChunkIndex._doc_chunks()`（读**切块**产物），**一个向量都不会被读到** → eager 建是白花
    # （实测 5 篇 21.8s）。真需要它的路径（`search_l3` / L0 的 `_audit_absence` / `set_judge`）
    # 调 `ChunkIndex(pdf).vectors()` 时会**惰性现建** —— 所以跳过**不会**造成功能缺失。
    t0 = time.time()
    todo = r.usable
    from paperpilot.agents.nodes.read_full import index_needed
    reason = "" if build_index else "build_index=False"
    if not reason and not index_needed():
        reason = "当前读取器(fullctx)只需要切块、不需要向量；真需要时会惰性现建"
    if reason:
        for pr in todo:
            pr.stages["index"] = _mk("index", "skipped", reason)
        finish("index", t0, "skipped", reason)
        emit("index", len(todo), len(todo), note=f"跳过（{reason}）")
        log(f"⑤ 索引：跳过（{reason}）")
    else:
        log(f"⑤ 建问答索引 {len(todo)} 篇 …")
        from paperpilot.agents.embedder import ChunkIndex
        for i, pr in enumerate(todo):
            emit("index", i, len(todo), note=f"{pr.arxiv_id} 编码中…", paper=pr)
            tt = time.time()
            try:
                ChunkIndex(pr.pdf_name).vectors()   # 触发 encode + 落 cvec 缓存
                pr.stages["index"] = _mk("index", "ok", seconds=time.time() - tt)
            except Exception as e:                  # noqa: BLE001
                # 与 worker 一致：索引失败**不影响交付**（问答首次提问时会现建/降级）
                pr.stages["index"] = _mk("index", "failed",
                                         f"{type(e).__name__}: {e}"[:200],
                                         seconds=time.time() - tt)
            emit("index", i + 1, len(todo), note=f"{pr.arxiv_id}: "
                 f"{pr.stages['index'].status}", paper=pr)
        n_idx = sum(1 for p in todo if p.stages["index"].status == "ok")
        finish("index", t0, note=f"{n_idx}/{len(todo)} 篇索引成功")
        log(f"⑤ 索引完成：{n_idx}/{len(todo)}")


# ── 公共入口 ─────────────────────────────────────────────────────────────────

def _log(verbose: bool):
    return (lambda m: print(m)) if verbose else (lambda m: None)


def search_papers(query: str, *, k: int = SEARCH_K, use_llm: bool = True,
                  pool: int = POOL, on_event: OnEvent | None = None,
                  verbose: bool = True) -> DirectionResult:
    """**只做阶段①检索**：把候选清单交出去让用户挑，不再往下处理。

    配 `run_selected` 用，就是「先给 10 篇 → 用户挑 ≤`SELECT_MAX` 篇 → 再处理」。
    """
    _ensure_env()
    r = DirectionResult(ok=False, query=query, phase="search")
    if not _search_stage(r, query=query, k=k, use_llm=use_llm, pool=pool,
                         emit=_emitter(on_event), finish=_finisher(r),
                         log=_log(verbose)):
        return r
    r.ok = True
    return r


def run_selected(query: str, arxiv_ids: list[str], *, k: int = SEARCH_K,
                 force: bool = False, skip_llm: bool = False, workers: int = 4,
                 build_index: bool = True, use_llm: bool = True, pool: int = POOL,
                 on_event: OnEvent | None = None,
                 verbose: bool = True) -> DirectionResult:
    """**跳过"让用户挑"这一步**：按 `arxiv_ids` 只处理指定的那几篇（②③④⑤）。

    仍会**重跑一次阶段①检索**（热路径 ~5s）—— 为了让 title/abstract/score 来自
    **唯一真源**（检索结果），而不是信调用方传来的字符串。

    ⚠️ 用户选的 id 若这次没出现在前 `k` 名里（LLM 精排理论上可能抖动），**不静默丢弃**：
    仍按 id 往下处理，只是标题会是空的，并在日志里点名。
    """
    ids = [str(x).strip() for x in (arxiv_ids or []) if str(x).strip()]
    _ensure_env()
    log = _log(verbose)
    r = DirectionResult(ok=False, query=query, phase="process")
    emit, finish = _emitter(on_event), _finisher(r)
    if not ids:
        r.reason = "没有指定要处理的论文（arxiv_ids 为空）"
        return r

    # 检索取 max(k, len(ids))：用户挑的篇数理论上可能多于 k，取够才能全部找到
    ask_k = max(int(k), len(ids))
    if not _search_stage(r, query=query, k=ask_k, use_llm=use_llm, pool=pool,
                         emit=emit, finish=finish, log=log):
        return r

    by_id = {p.arxiv_id: p for p in r.papers}
    picked = [by_id.get(i) or PaperRun(arxiv_id=i) for i in ids]
    miss = [i for i in ids if i not in by_id]
    if miss:
        log(f"     ⚠ {len(miss)} 篇不在本次检索前 {ask_k} 名里，仍按 id 处理：{miss}")
    r.papers = picked
    log(f"     处理用户所选 {len(picked)} 篇：{', '.join(ids)}")

    _process_stage(r, picked, force=force, skip_llm=skip_llm, workers=workers,
                   build_index=build_index, emit=emit, finish=finish, log=log)
    return r


def run_direction(query: str, *, k: int = DEFAULT_K, force: bool = False,
                  skip_llm: bool = False, workers: int = 4,
                  build_index: bool = True, use_llm: bool = True, pool: int = POOL,
                  on_event: OnEvent | None = None,
                  verbose: bool = True) -> DirectionResult:
    """一条龙：① 检索 **k** 篇 + ②③④⑤ 全处理（CLI / 旧行为 / 对照用）。

    ⚠️ Web 前端**不走这条** —— 它用 `search_papers` → 用户挑 → `run_selected`。
    保留它是为了 CLI、脚本与"不给用户挑"的场景。

    Args:
        query: 自然语言研究方向。
        k: 检索返回并逐篇处理的论文数。
        force: True 时 MinerU 与报告链**全链路重跑**（否则复用缓存，通常快很多）。
        skip_llm: 报告链只装配已有产物、不调 LLM。
        workers: 单篇报告链内部 claims 的并发数。
        build_index: 是否建问答向量索引（**问答的第一步会用到**）。
        use_llm: 检索的 LLM 精排开关。
        pool: 检索进 CE 的候选数。
        on_event: 见 `OnEvent`。**抛异常即中止**。
        verbose: 打印各阶段进度。

    Returns:
        `DirectionResult`。**已知失败不抛异常**（`ok=False` + `reason`）；
        只有 `on_event` 主动抛出的异常会穿透。
    """
    _ensure_env()
    log = _log(verbose)
    r = DirectionResult(ok=False, query=query, phase="all")
    emit, finish = _emitter(on_event), _finisher(r)
    if not _search_stage(r, query=query, k=k, use_llm=use_llm, pool=pool,
                         emit=emit, finish=finish, log=log):
        return r
    _process_stage(r, r.papers, force=force, skip_llm=skip_llm, workers=workers,
                   build_index=build_index, emit=emit, finish=finish, log=log)
    return r


# 报告链子阶段 → 中文（前端第 ④ 步的细粒度提示；`pipeline.process_pdf` 的 `on_stage` 名）
_SUBSTAGE: dict[str, str] = {
    "claims": "抽取主张",
    "view": "去重 / 打标 / 打分",
    "skeleton": "构建论证骨架",
    "figures": "识别图表",
    "report_text": "生成概述 / 导读",
    "assemble": "装配报告",
}


__all__ = ["DEFAULT_K", "SEARCH_K", "SELECT_MAX", "STAGE_KEYS", "STAGE_LABEL", "STAGES",
           "DirectionResult", "PaperRun", "Step",
           "run_direction", "run_selected", "search_papers"]
