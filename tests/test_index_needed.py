"""摄取期「要不要预建向量索引」的回归（2026-10-03）。

背景：默认读取器 `fullctx` 读全文走的是 `ChunkIndex._doc_chunks()`（读**切块产物**），
**一个向量都不会被读到**。而摄取期不管什么读取器都 eager 建 `<stem>.cvec.npy` ——
实测 5 篇 **21.8s**，全白花。

改法：判据收敛到 `read_full.index_needed()`（**一个地方**），`worker._lane_index` 与
`direction` 的阶段⑤ 都据此跳过。**跳过不等于用不了**：索引是惰性的，真需要的路径
（`search_l3` / L0 的 `_audit_absence` / `set_judge` / CRAG）会现建现用。

本文件只用桩，不碰语料、不编码、不落盘。
"""
from __future__ import annotations

import threading

import pytest

from paperpilot.agents.nodes import read_full
from paperpilot.agents import embedder
from paperpilot import worker


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PAPERPILOT_EAGER_INDEX", raising=False)
    monkeypatch.setenv("PAPERPILOT_QA_READER", "fullctx")


# ── 判据本身 ────────────────────────────────────────────────────────────────


def test_fullctx_does_not_need_index():
    """★ 默认 fullctx → **不建**（这就是本次要省掉的那 21.8s）。"""
    assert read_full.index_needed() is False


@pytest.mark.parametrize("r", ["retrieval", "set"])
def test_block_readers_need_index(monkeypatch, r):
    """块级检索的读取器（RAG-2 / 集合问答）→ 必须建。"""
    monkeypatch.setenv("PAPERPILOT_QA_READER", r)
    assert read_full.index_needed() is True


def test_eager_override(monkeypatch):
    """逃生开关：`PAPERPILOT_EAGER_INDEX=1` → 无论读取器都预建（退回旧行为）。"""
    monkeypatch.setenv("PAPERPILOT_EAGER_INDEX", "1")
    assert read_full.index_needed() is True
    monkeypatch.setenv("PAPERPILOT_QA_READER", "retrieval")
    assert read_full.index_needed() is True


def test_eager_override_off_values(monkeypatch):
    """`0` / `false` / 空 → 不算开（沿用仓库既有的开关语义）。"""
    for v in ("0", "false", "", "  "):
        monkeypatch.setenv("PAPERPILOT_EAGER_INDEX", v)
        assert read_full.index_needed() is False, v


# ── 摄取 lane 真的按它跳过 ──────────────────────────────────────────────────


class _FakeIdx:
    """记录"有没有真的去建索引"。"""
    calls: list[tuple[str, str]] = []

    def __init__(self, pdf: str) -> None:
        self.pdf = str(pdf)
        _FakeIdx.calls.append(("init", str(pdf)))

    def vectors(self):
        _FakeIdx.calls.append(("vectors", self.pdf))
        return None


@pytest.fixture
def lane_env(monkeypatch):
    _FakeIdx.calls = []
    monkeypatch.setattr(embedder, "ChunkIndex", _FakeIdx)
    monkeypatch.setattr(worker.jobs, "save", lambda _j: None)     # 测试里不落盘
    return _FakeIdx.calls


def _job() -> dict:
    return {"job_id": "j-test-index", "pdf": "x.pdf", "stages": {}, "force": False}


def test_lane_index_skips_without_touching_encoder(monkeypatch, lane_env):
    """★ fullctx → 标 `skipped`，且**一次都没碰编码器**（省的就是这笔）。"""
    job = _job()
    worker._lane_index(job, threading.Event())
    assert job["stages"]["index"]["status"] == "skipped"
    assert lane_env == []


def test_lane_index_builds_when_needed(monkeypatch, lane_env):
    """需要向量时（`=retrieval`）→ 照常建（不能把功能一起关掉）。"""
    monkeypatch.setenv("PAPERPILOT_QA_READER", "retrieval")
    job = _job()
    worker._lane_index(job, threading.Event())
    assert job["stages"]["index"]["status"] == "ok"
    assert ("vectors", "x.pdf") in lane_env


def test_lane_index_respects_eager_override(monkeypatch, lane_env):
    """`PAPERPILOT_EAGER_INDEX=1` → fullctx 下也建。"""
    monkeypatch.setenv("PAPERPILOT_EAGER_INDEX", "1")
    job = _job()
    worker._lane_index(job, threading.Event())
    assert job["stages"]["index"]["status"] == "ok"
    assert ("vectors", "x.pdf") in lane_env
