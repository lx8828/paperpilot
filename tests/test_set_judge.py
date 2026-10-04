"""集合问答（`set` reader）的离线回归测试。

覆盖：分组取块 / 判定解析 / 聚合只留 yes / 渲染不做完备性宣称 /
节点输出契约（`set_papers` + `cites`）/ `generate_answer` 透传 / 图接线。
全部替身，**无 key、无 GPU、无网络**（见 conftest 的 isolate）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperpilot.agents.nodes import answer_set  # noqa: E402
from paperpilot.agents.nodes.answer import generate_answer  # noqa: E402
from paperpilot.agents.nodes.read_full import reader  # noqa: E402
from paperpilot.components import set_judge  # noqa: E402


# ───────────────────── 假索引 / 假 LLM ─────────────────────
class _FakeIdx:
    """最小 MultiChunkIndex 替身：只实现被测代码用到的两个成员。"""

    def __init__(self, pdfs=None):
        self.pdfs = list(pdfs or ["a.pdf", "b.pdf", "c.pdf"])
        self._hits = [
            {"pdf": "a.pdf", "chunk_id": "a1", "page": 1, "section": "Intro",
             "text": "we ablate each component", "score": 0.9},
            {"pdf": "b.pdf", "chunk_id": "b1", "page": 2, "section": "Method",
             "text": "we remove one module at a time", "score": 0.8},
            {"pdf": "c.pdf", "chunk_id": "c1", "page": 3, "section": "Results",
             "text": "nothing relevant here", "score": 0.5},
            {"pdf": "a.pdf", "chunk_id": "a2", "page": 1, "section": "Intro",
             "text": "ablation table 2", "score": 0.4},
        ]

    def _doc_chunks(self):
        return list(range(len(self._hits)))

    def search_hybrid(self, query, top_k=8):
        return self._hits[:top_k]


def _fake_chat_json(answers_map):
    """按 user 里出现的片段文本决定判定（确定性，不依赖网络）。"""
    def _f(system, user, **kw):
        assert system == set_judge.SYS_SET
        if "we ablate each component" in user:
            return {"label": "yes", "evidence": [1], "why": "本文做了消融"}
        if "we remove one module" in user:
            return {"label": "no", "evidence": [], "why": "只是提及"}
        return {"label": "unclear", "evidence": [], "why": "证据不足"}
    return _f


@pytest.fixture
def fake_set_llm(monkeypatch):
    monkeypatch.setattr(set_judge.llm, "chat_json",
                        _fake_chat_json(None), raising=True)


@pytest.fixture
def fake_idx(monkeypatch):
    monkeypatch.setattr("paperpilot.agents.embedder.MultiChunkIndex", _FakeIdx, raising=True)
    return _FakeIdx


# ───────────────────── 组件 ─────────────────────
def test_per_paper_hits_按篇分组且每篇不超过b(fake_idx):
    idx = _FakeIdx()
    per = set_judge.per_paper_hits(idx, "哪几篇做了消融？", b=1)
    assert set(per) == {"a.pdf", "b.pdf", "c.pdf"}          # c 也有命中（分低）
    assert all(len(v) <= 1 for v in per.values())
    assert per["a.pdf"][0]["chunk_id"] == "a1"               # 取的是该篇最高分块


def test_judge_one_解析标签与证据(fake_set_llm):
    hits = _FakeIdx()._hits[:1]
    got = set_judge.judge_one("该论文做了消融实验", hits)
    assert got["label"] == "yes" and got["evidence"] == [1]


def test_judge_one_失败降级为unclear(monkeypatch):
    def boom(*_a, **_k):
        raise set_judge.llm.LLMError("模拟失败")
    monkeypatch.setattr(set_judge.llm, "chat_json", boom, raising=True)
    got = set_judge.judge_one("q", _FakeIdx()._hits[:1])
    assert got["label"] == "unclear" and "失败" in got["why"]


def test_run_只留yes且按分数降序(fake_idx, fake_set_llm):
    res = set_judge.run("哪几篇做了消融？", _FakeIdx(), workers=2)
    assert [p["pdf"] for p in res["papers"]] == ["a.pdf"]    # 只有 a 判 yes
    assert res["n_papers"] == 3 and res["n_yes"] == 1
    assert res["papers"][0]["evidence"] == [1]
    assert res["papers"][0]["snippet"]                       # 带片段原文（供 cites）


def test_render_不做完备性宣称(fake_idx, fake_set_llm):
    res = set_judge.run("哪几篇做了消融？", _FakeIdx(), workers=2)
    txt = set_judge.render("哪几篇做了消融？", res)
    assert "1 篇" in txt and "a.pdf" in txt
    for word in ("都看过了", "全部覆盖", "已穷尽"):
        assert word not in txt
    assert "证据不足" in txt                                  # unclear 要如实报出


# ───────────────────── 节点 / 图 ─────────────────────
def test_answer_set_节点输出契约(fake_idx, fake_set_llm):
    out = answer_set({"question": "哪几篇做了消融？", "pdfs": ["a.pdf", "b.pdf", "c.pdf"]})
    sp = out["set_papers"]
    assert sp["n_papers"] == 3 and sp["n_yes"] == 1 and sp["b"] >= 1
    assert "answer" in sp and out["cites"][0]["pdf"] == "a.pdf"
    assert out["cites"][0]["evidence"]                        # 闸门靠 cites[].evidence 重建
    assert "SET" in out["route"]
    assert out["debug"]["reader"]["path"] == "set"


def test_answer_set_单篇时跳过(monkeypatch):
    out = answer_set({"question": "q", "pdfs": ["a.pdf"]})
    assert out["set_papers"]["n_papers"] == 1
    assert out["debug"]["reader"]["skipped"] == "need>=2_papers"


def test_generate_answer_透传set(fake_idx, fake_set_llm):
    st = answer_set({"question": "哪几篇做了消融？", "pdfs": ["a.pdf", "b.pdf", "c.pdf"]})
    st = {"question": "哪几篇做了消融？", "pdfs": ["a.pdf", "b.pdf", "c.pdf"],
          "route": st["route"], "debug": st["debug"],
          "set_papers": st["set_papers"], "cites": st["cites"]}
    got = generate_answer(st)
    assert got["debug"]["answer"]["level"] == "SET"
    assert got["answer"] == st["set_papers"]["answer"]
    assert len(got["cites"]) == 1


def test_reader_开关支持set(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_QA_READER", "set")
    assert reader() == "set"
    monkeypatch.setenv("PAPERPILOT_QA_READER", "sets")
    assert reader() == "set"
    monkeypatch.setenv("PAPERPILOT_QA_READER", "retrieval")
    assert reader() == "retrieval"


def test_图里接了answer_set节点():
    from paperpilot.graph.qa_graph_v3 import build_qa_graph_v3
    g = build_qa_graph_v3()
    assert "answer_set" in (getattr(g, "nodes", {}) or {})


def test_状态键已登记():
    from paperpilot.agents.state import QAState
    assert "set_papers" in QAState.__annotations__


def test_提示词四纪律未丢():
    s = set_judge.SYS_SET
    assert "引用的他人工作" in s          # 本文 vs 引用他人
    assert "列出/提及 ≠ 做了" in s
    assert "不要用先验知识补充" in s
    assert "yes|no|unclear" in s
