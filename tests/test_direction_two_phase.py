"""「方向」两阶段（**search → 用户挑 → process**）的契约与去重键回归（2026-10-02）。

守两件最容易**静默**出错的事：

1. **job 去重键必须按 `phase`（与所选 id）隔离** —— 否则"只检索"与"处理所选"两个
   job 会互相吞掉：第二个提交被 `find_active` 判成"已有"、直接返回第一个 job，
   于是用户点了"开始解析"却什么都没发生，**而且没有任何报错**。
2. **选篇上限 `SELECT_MAX` 是一个契约值**（= 候选数 = 10；依据见其注释），
   前端经 `/api/meta` 取它 —— 不许前后端各写一个数字。
"""
from __future__ import annotations

import inspect

from paperpilot import direction as D
from paperpilot import worker


# ── 去重键隔离 ──────────────────────────────────────────────────────────────


def test_job_key_isolated_by_phase():
    """同一方向：all / search / process 三个键必须互不相同。"""
    q = "如何用对比学习做跨语言摘要生成"
    keys = {worker._direction_key(q),
            worker._direction_key(q, "search"),
            worker._direction_key(q, "process", ["2408.09273"])}
    assert len(keys) == 3


def test_job_key_isolated_by_selection():
    """同方向 + 同 phase，但**所选不同** → 也必须不同键。

    否则"选 3 篇"与"选 5 篇"会互相吞并：用户换个选择点解析，会被当成重复提交。
    """
    a = worker._direction_key("q", "process", ["a", "b"])
    b = worker._direction_key("q", "process", ["a", "c"])
    assert a != b


def test_job_key_stable_for_same_input():
    """同输入（含**顺序无关**）→ 同键 —— 这才是"重复提交去重"能生效的前提。"""
    assert (worker._direction_key("q", "process", ["a", "b"])
            == worker._direction_key("q", "process", ["b", "a"]))


def test_job_key_is_a_usable_pdf_field():
    """键会当 `pdf` 字段用（`jobs.new_job` 里 `Path(key).stem`）→ 不能带路径分隔符。"""
    for k in (worker._direction_key("q"), worker._direction_key("q", "search"),
              worker._direction_key("q", "process", ["2408.09273"])):
        assert "/" not in k and "\\" not in k
        assert ":" in k                       # 形状：`dir-<phase>:<sha8>`
        assert k.startswith("dir-")


# ── 契约值 ──────────────────────────────────────────────────────────────────


def test_contract_constants():
    """上限 / 候选数 / 阶段键 —— 都是对外契约，改动必须同步前端与文档。"""
    assert D.SELECT_MAX == 10, "上限 = 候选数（再靠后的名次不是 LLM 精排的）"
    assert D.SEARCH_K == 10, "10 = corpus_search.N_OUT（LLM 精排的全部名次）"
    assert D.STAGE_KEYS == ["search", "fetch", "mineru", "report", "index"]


def test_select_max_fits_in_context_budget():
    """★ 不变式：**把上限选满也必须在上下文预算内**。

    这条是 `SELECT_MAX` 与 `fullctx` 预算之间的真正契约 —— 改任何一个都可能踩它，
    踩了的表现是"用户勾满 → 问答被预算闸拒"，而不是任何一处报错。
    """
    from paperpilot.components import fullctx as F

    need = D.SELECT_MAX * F.CTX_PER_PAPER_CHARS
    assert need < F.DEFAULT_CTX_BUDGET_CHARS, (
        f"选满 {D.SELECT_MAX} 篇 ≈ {need:,} 字符，已超预算 "
        f"{F.DEFAULT_CTX_BUDGET_CHARS:,} —— 要么调小 SELECT_MAX，要么调大 "
        f"PAPERPILOT_CTX_BUDGET_CHARS（见 fullctx 的注释）")


def test_三个入口都在():
    """两阶段用前两个；`run_direction` 保留给 CLI / 不给用户挑的场景。"""
    assert callable(D.search_papers)
    assert callable(D.run_selected)
    assert callable(D.run_direction)
    # `run_selected` 的第二个位置参数必须是 arxiv_ids（前端按这个契约调）
    assert list(inspect.signature(D.run_selected).parameters)[1] == "arxiv_ids"


def test_worker_dispatches_by_phase():
    """`submit_direction` 必须接受 phase / arxiv_ids 并落到 job 上。"""
    sig = inspect.signature(worker.submit_direction)
    assert "phase" in sig.parameters and "arxiv_ids" in sig.parameters
