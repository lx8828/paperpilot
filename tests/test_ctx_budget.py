"""上下文预算窗口（`fullctx`）回归（2026-10-03）。

要守住的三件事：
1. **单位是字符**、判据 fail-safe（非法配置只能收紧，不许静默放开）；
2. **超预算时绝不能去调 API** —— `build_context` 与 `llm.chat_text` 都没有截断保护，
   真发出去就是一次难懂的 400；必须在**调用前**换成可操作的话术；
3. 闸门要在 **validator 之前**（否则那句"话术"会被当成无据答案去修复/拒答），
   且**只对 fullctx 生效**（`retrieval`/`set` 的上下文只有 top-k 块，拿语料总大小拦它们是错的）。

全部用桩：不碰语料、不联网。
"""
from __future__ import annotations

import pytest

from paperpilot.components import fullctx
from paperpilot.graph import ask as graph_ask


class _Chunk:
    def __init__(self, cid: str, text: str, page: int = 1) -> None:
        self.chunk_id, self.text = cid, text
        self.page_span = (page, page)
        self.title_path = ["1", "intro"]


class _Idx:
    """固定 2 块 × 100 字符 → `estimate_ctx_chars` 应得 120 + 148*2 = 416。"""

    def __init__(self, pdf: str) -> None:
        self.pdf = str(pdf)

    def _doc_chunks(self):
        return [_Chunk("c1", "x" * 100), _Chunk("c2", "y" * 100)]


@pytest.fixture(autouse=True)
def _stub_all(monkeypatch):
    monkeypatch.setattr(fullctx, "ChunkIndex", _Idx)
    monkeypatch.delenv(fullctx.ENV_CTX_BUDGET, raising=False)
    monkeypatch.setenv("PAPERPILOT_QA_READER", "fullctx")


# ── 判据本身 ────────────────────────────────────────────────────────────────


def test_default_budget():
    assert fullctx.ctx_budget_chars() == fullctx.DEFAULT_CTX_BUDGET_CHARS
    assert fullctx.DEFAULT_CTX_BUDGET_CHARS == 1_200_000


def test_env_override(monkeypatch):
    monkeypatch.setenv(fullctx.ENV_CTX_BUDGET, "500000")
    assert fullctx.ctx_budget_chars() == 500_000


def test_zero_means_unlimited(monkeypatch):
    """`0` = 不限制（**显式**写 0 才算，与 `_max_pdf_bytes` 同语义）。"""
    monkeypatch.setenv(fullctx.ENV_CTX_BUDGET, "0")
    assert fullctx.ctx_budget_chars() == 0


@pytest.mark.parametrize("bad", ["-1", "abc", "1.5", "  "])
def test_bad_config_falls_back_to_default(monkeypatch, bad):
    """非法 → 回退默认（**不许**静默放大权限：负数若当成 0 就等于关掉了闸）。"""
    monkeypatch.setenv(fullctx.ENV_CTX_BUDGET, bad)
    assert fullctx.ctx_budget_chars() == fullctx.DEFAULT_CTX_BUDGET_CHARS


def test_estimate_is_cheap_and_sane():
    assert fullctx.estimate_ctx_chars(["a.pdf"]) == 416


def test_estimate_returns_zero_when_unreadable(monkeypatch):
    """读不出产物 → 返回 0 = **不拦**（交给下游自己的错误处理，别在这里误拦）。"""
    class _Boom:
        def __init__(self, _pdf):
            raise FileNotFoundError("没有产物")

    monkeypatch.setattr(fullctx, "ChunkIndex", _Boom)
    assert fullctx.estimate_ctx_chars(["a.pdf"]) == 0


# ── answer()：超预算必须**在调用前**拦下 ─────────────────────────────────────


def test_over_budget_does_not_call_llm(monkeypatch):
    """★ 核心：超预算 → 直接给话术，**一次 LLM 都不许调**。"""
    calls: list[int] = []
    monkeypatch.setattr(fullctx.llm, "chat_text",
                        lambda *a, **k: calls.append(1) or "不该被调到")
    monkeypatch.setenv(fullctx.ENV_CTX_BUDGET, "100")     # ctx ≈ 250 字符 > 100
    out = fullctx.answer(["a.pdf"], "问题")
    assert out.get("over_budget") is True
    assert calls == []
    assert "预算" in out["answer"] and "减少篇数" in out["answer"]
    assert out["cites"] == []


def test_within_budget_answers_normally(monkeypatch):
    monkeypatch.setattr(fullctx.llm, "chat_text",
                        lambda *a, **k: "结论见 [P1·§intro·¶c1]。")
    monkeypatch.setenv(fullctx.ENV_CTX_BUDGET, "100000")
    out = fullctx.answer(["a.pdf"], "问题")
    assert "over_budget" not in out
    assert out["answer"].startswith("结论")
    assert [c["chunk_id"] for c in out["cites"]] == ["c1"]


def test_zero_budget_never_blocks(monkeypatch):
    monkeypatch.setattr(fullctx.llm, "chat_text", lambda *a, **k: "答案")
    monkeypatch.setenv(fullctx.ENV_CTX_BUDGET, "0")
    assert "over_budget" not in fullctx.answer(["a.pdf"], "问题")


# ── graph.ask：闸门要在 validator 之前，且只对 fullctx 生效 ────────────────────


@pytest.fixture
def graph_stubs(monkeypatch):
    """桩掉真正的图 + 桩掉"几篇就超预算"，并记录图有没有被跑到。"""
    ran: list[int] = []
    monkeypatch.setattr(
        "paperpilot.graph.qa_graph_v3.ask",
        lambda *a, **k: ran.append(1) or {"answer": "图跑了", "cites": [],
                                          "route": ["stub"]})
    monkeypatch.setattr("paperpilot.ingest.qa_blocked_reason", lambda _p: "")
    monkeypatch.setattr(fullctx, "estimate_ctx_chars", lambda _pdfs: 9_999_999)
    return ran


def test_graph_blocks_over_budget_before_graph(monkeypatch, graph_stubs):
    """★ 超预算 → 早返回 `ctx_over_budget` + `validator.action=blocked`，**不跑图**。"""
    out = graph_ask("问题", ["a.pdf", "b.pdf"])
    assert out["route"] == ["ctx_over_budget"]
    assert out["validator"]["action"] == "blocked"
    assert out["cites"] == []
    assert "预算" in out["answer"]
    assert graph_stubs == []                    # 关键：图根本没跑（也不会去调 API）


def test_graph_does_not_apply_budget_for_retrieval_reader(monkeypatch, graph_stubs):
    """`retrieval` 读取器的上下文只有 top-k 块 → 拿语料总大小拦它是**错的**。"""
    monkeypatch.setenv("PAPERPILOT_QA_READER", "retrieval")
    out = graph_ask("问题", ["a.pdf"])
    assert out["route"] != ["ctx_over_budget"]
    assert graph_stubs == [1]                   # 图跑了（说明没被预算拦）


def test_graph_within_budget_runs_graph(monkeypatch, graph_stubs):
    monkeypatch.setattr(fullctx, "estimate_ctx_chars", lambda _pdfs: 1_000)
    out = graph_ask("问题", ["a.pdf"])
    assert out["answer"] == "图跑了"
    assert graph_stubs == [1]
