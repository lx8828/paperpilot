"""问答读取器开关（2026-09-27）：默认 **`fullctx`（全文直读）**，可一键切回 RAG-2。

守住三件事（都不需要 key / GPU / 网络）：

1. **默认必须是 `fullctx`** —— 否则"下线 RAG-2 的生产路径"这件事根本没生效；
2. 切回 `PAPERPILOT_QA_READER=retrieval` 时，图与路由**回到原样**（RAG-2 未删、可展示）；
3. 直读产物在 `generate_answer` 里被**原样返回**（不再走"挑块 + 组上下文"，
   因此**不触碰 embedder / LLM** —— 这也是它能在 CI 里被断言的原因）。

背景与判据见 `agents/nodes/read_full.py` 的模块 docstring（S/C 不等式）。
"""
from __future__ import annotations

import pytest

from paperpilot.agents.nodes import answer as A
from paperpilot.agents.nodes.read_full import ENV_READER, reader
from paperpilot.graph import qa_graph_v3

CANNED = {"answer": "直读答案：本文用图结构组织证据 [P1·§ABSTRACT·¶c1]",
          "cites": [{"pdf": "a.pdf", "chunk_id": "c1", "page": 1,
                     "section": "ABSTRACT", "evidence": "sample evidence"}],
          "ctx_chars": 1234, "n_papers": 1, "style": "base", "seconds": 1.2}


# ───────────────────────── ① 开关与别名 ─────────────────────────


def test_reader_default_is_fullctx():
    assert reader() == "fullctx"


@pytest.mark.parametrize("val, expect", [
    ("fullctx", "fullctx"), ("", "fullctx"), ("FULLCTX", "fullctx"),
    ("retrieval", "retrieval"), ("RAG2", "retrieval"), ("l3", "retrieval"),
], ids=["fullctx", "空", "大写", "retrieval", "rag2 别名", "l3 别名"])
def test_reader_env_aliases(monkeypatch, val, expect):
    monkeypatch.setenv(ENV_READER, val)
    assert reader() == expect


# ───────────────────────── ② 路由：三态 ─────────────────────────


def test_route_by_reader(monkeypatch):
    # 报告层够 → 一律 L0 直答（与读取器无关）
    assert qa_graph_v3.route_after_judge0_pure({"verdict": {"enough": True}}) == "answer"
    # 不够 → 默认读全文
    assert qa_graph_v3.route_after_judge0_pure({"verdict": {"enough": False}}) == "read_full"
    # 切回 RAG-2 → 回到原来的检索节点
    monkeypatch.setenv(ENV_READER, "retrieval")
    assert qa_graph_v3.route_after_judge0_pure({"verdict": {"enough": False}}) == "search_l3"


# ───────────────────────── ③ 图：两个读者的节点都在 ─────────────────────────


def test_graphs_keep_both_readers():
    """RAG-2 的 `search_l3` **必须还在图里**（面试展示 / env 切回都靠它）。"""
    g1 = qa_graph_v3.build_qa_graph_v3_nol3j()
    n1 = set(g1.get_graph().nodes)
    assert {"router", "read_full", "search_l3", "answer"} <= n1
    g2 = qa_graph_v3.build_qa_graph_v3()          # 带 judge_l3 的变体
    n2 = set(g2.get_graph().nodes)
    assert {"read_full", "search_l3", "judge_l3"} <= n2


# ───────────────────────── ④ read_full 节点的状态契约 ─────────────────────────


def test_read_full_writes_state(monkeypatch):
    from paperpilot.agents.nodes import answer_fullctx as RF
    from paperpilot.components import fullctx as FC

    monkeypatch.setattr(FC, "answer", lambda *_a, **_k: dict(CANNED))
    out = RF({"question": "q", "pdfs": ["a.pdf"],
              "debug": {"l0": {"n": 1}}, "route": ["L0"]})
    assert out["fullctx"]["answer"] == CANNED["answer"]
    assert "FULLCTX" in out["route"] and "L0" in out["route"]      # 履历追加、不覆盖
    assert out["debug"]["reader"]["path"] == "fullctx"             # 读取器标记
    assert out["debug"]["l0"] == {"n": 1}                          # 不覆盖已有 debug


def test_read_full_without_corpus_does_not_call_llm(monkeypatch):
    """无语料：不该炸、也不该调用 LLM（`fullctx.answer` 绝不能被调到）。"""
    from paperpilot.agents.nodes import answer_fullctx as RF
    from paperpilot.components import fullctx as FC

    def _boom(*_a, **_k):
        raise AssertionError("没有任何语料时不应调用 fullctx.answer")

    monkeypatch.setattr(FC, "answer", _boom)
    out = RF({"question": "q", "pdfs": []})
    assert out["fullctx"]["n_papers"] == 0
    assert out["debug"]["reader"]["skipped"] == "no_corpus"


# ───────────────────────── ⑤ generate_answer 直读分支：原样返回 ─────────────────────────


def test_generate_answer_returns_fullctx_verbatim():
    state = {"question": "q", "pdfs": ["a.pdf"], "fullctx": dict(CANNED),
             "route": ["L0", "FULLCTX"], "debug": {"l0": {"n": 1}}}
    out = A.generate_answer(state)
    assert out["answer"] == CANNED["answer"]        # 一字不改
    assert out["cites"] == CANNED["cites"]          # cites 原样透传（前端适配是后续改动）
    assert out["debug"]["answer"]["level"] == "FULLCTX"
    assert out["debug"]["answer"]["n_entries"] == 1  # 闸门"引用越界"基准
    assert out["debug"]["l0"] == {"n": 1}            # 不覆盖已有 debug
    assert out["route"] == ["L0", "FULLCTX", "answer_fullctx"]


def test_retrieval_reader_path_unchanged_without_fullctx(fake_llm):
    """没有 `fullctx` 字段时，`generate_answer` 必须**仍走原来的分档逻辑**（不能改坏 RAG-2）。

    用 `fake_llm` 走离线：L0 档（无 `l3_chunks`）→ 生成答案 → level 仍是 `L0`。
    """
    state = {"question": "这篇论文的核心方法是什么？", "pdfs": [], "title": "t",
             "overview": "概述", "core_points": [], "limitations": []}
    out = A.generate_answer(state)
    assert out["debug"]["answer"]["level"] == "L0"   # 无 l3_chunks → L0 档（不是 FULLCTX）
    assert out["route"] == ["answer_L0"]
    assert "fullctx" not in out
