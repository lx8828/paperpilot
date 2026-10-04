"""修复链的**读取器分流**回归（2026-10-03）。

两条路径的前提不同，修法必须不同：
  · fullctx（默认）：提示词里**已有全部语料** → 修复 = 「同一份上下文 + 初稿 + 意见」
    再答一次（不重检索 → 覆盖不丢、`[P…]` 锚点不换；`sys+ctx` 稳定前缀 → 命中缓存）。
  · retrieval（RAG-2）：上下文**只有 top-k 块** → 缺口证据不在里面 → **必须**重检索（CRAG）。

实测反例（新路径要避免的）：fullctx 下走 CRAG，`search_multi_hybrid(top_k=16)` 是**全局**
取块、无篇配额 → 跨篇问题被修成"只讲第 1 篇"，且答案改写成 `[n]` → `[P…]` 锚点整体丢失。

本文件只用**桩**验证"分流决策"，不调 LLM、不碰语料。
"""
from __future__ import annotations

import pytest

from paperpilot.components import repairer

_HIGH = [{"sev": "high", "type": "number", "sentence": "提升 4.2", "detail": "无据"}]


@pytest.fixture
def stubs(monkeypatch):
    """把三条修复路径都换成记录调用的桩。"""
    called: list[str] = []

    def _mk(name):
        def _f(*_a, **_k):
            called.append(name)
            return ("修好的答案", [{"n": 1, "chunk_id": "c1", "pdf": "A.pdf",
                                    "evidence": "x", "page": 1}])
        return _f

    monkeypatch.setattr(repairer, "_try_fullctx_reask", _mk("reask"))
    monkeypatch.setattr(repairer, "_try_refine", _mk("refine"))
    monkeypatch.setattr(repairer, "_try_crag", _mk("crag"))
    return called


def test_fullctx_routes_to_reask_only(monkeypatch, stubs):
    """fullctx → **只**走"同上下文再答"，绝不碰 CRAG（碰了就丢覆盖 + 换锚点）。"""
    monkeypatch.setattr(repairer, "_is_fullctx", lambda: True)
    got = repairer.repair("q", ["A.pdf", "B.pdf"], "初稿", _HIGH, [])
    assert got and got[0] == "修好的答案"
    assert stubs == ["reask"]


def test_retrieval_keeps_old_path(monkeypatch, stubs):
    """retrieval → 老路（证据型问题 → CRAG；不能没）。"""
    monkeypatch.setattr(repairer, "_is_fullctx", lambda: False)
    repairer.repair("q", ["A.pdf"], "初稿", _HIGH, [])
    assert stubs == ["crag"]


def test_retrieval_generation_issue_goes_refine(monkeypatch, stubs):
    """生成型问题（偏题/含糊）在 retrieval 下仍先 Self-Refine。"""
    monkeypatch.setattr(repairer, "_is_fullctx", lambda: False)
    iss = [{"sev": "high", "type": "off_topic", "sentence": "x", "detail": "偏题"}]
    repairer.repair("q", ["A.pdf"], "初稿", iss, [])
    assert stubs == ["refine"]


def test_no_target_issue_returns_none(monkeypatch, stubs):
    """没有 high/mid 目标问题 → 不修（不白花调用）。"""
    monkeypatch.setattr(repairer, "_is_fullctx", lambda: True)
    assert repairer.repair("q", ["A.pdf"], "初稿", [], []) is None
    assert stubs == []


def test_is_fullctx_reads_env(monkeypatch):
    """`_is_fullctx()` 跟随 `PAPERPILOT_QA_READER`（`retrieval` 时必须是 False）。"""
    monkeypatch.setenv("PAPERPILOT_QA_READER", "fullctx")
    assert repairer._is_fullctx() is True
    monkeypatch.setenv("PAPERPILOT_QA_READER", "retrieval")
    assert repairer._is_fullctx() is False


def test_is_fullctx_never_raises(monkeypatch):
    """读不出读取器时**不能抛**（修复链上抛断 = 整个回答挂掉），退成 False 即可。"""
    import paperpilot.agents.nodes.read_full as rf

    def _boom():
        raise RuntimeError("模拟读取器模块异常")

    monkeypatch.setattr(rf, "reader", _boom)
    assert repairer._is_fullctx() is False
