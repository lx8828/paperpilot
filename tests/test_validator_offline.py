"""输出闸门的**纯机器**判据（迁移自 `qa/recall/_selftest_validator_20260913.py`）。

覆盖审查项「无引用、无依据的答案会被静默放行」的修复，并守住一条硬约束：
**`gate()` 的动作不变**（仍只对 HIGH 拦）——否则一次"顺手加固"就会把好答案换成兜底话术。

LLM 体检部分（真实裁判模型）标记为 `local`，默认跳过（CI 不需要 key）。
"""
from __future__ import annotations

import pytest

from paperpilot.components import validator as V

FAB = "本文提出了一种双塔检索结构，并用对比学习在多个数据集上验证了有效性。"
REFUSE = "抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。"
SHORT = "文中未给出该数值。"


# ───────────────────────── 机器判据：什么算"实质断言" ─────────────────────────


@pytest.mark.parametrize("text, expect", [
    (FAB + "提升 87.3%", True),
    (FAB, True),
    (REFUSE, False),
    (SHORT, False),
    ("", False),
    ("见原文。", False),
], ids=["含数值", "纯陈述", "拒答话术", "过短", "空", "极短"])
def test_substantive_claim(text, expect):
    assert V._substantive_claim(text)[0] is expect


# ───────────────────────── 零引用：独立类型与分级 ─────────────────────────


def test_zero_citation_is_its_own_type_and_mid():
    iss, _, _ = V._machine_checks("这篇论文的核心方法是什么？", FAB, [], n_entries=12)
    nc = [i for i in iss if i["type"] == "no_citation"]
    assert len(nc) == 1
    assert nc[0]["sev"] == V.MID
    assert nc[0].get("substantive") is True


def test_zero_citation_refusal_is_low():
    """拒答话术本身不算"该引用没引用" → LOW（避免误标）。"""
    iss, _, _ = V._machine_checks("这篇论文的核心方法是什么？", REFUSE, [], n_entries=12)
    nc = [i for i in iss if i["type"] == "no_citation"]
    assert nc and nc[0]["sev"] == V.LOW


# ───────────────────────── 硬约束：gate 动作不变 ─────────────────────────


def test_gate_zero_citation_still_passes():
    """零引用**不拦**（只标注）：这是本档的硬约束，改了就回归失败。"""
    g = V.gate("这篇论文的核心方法是什么？", FAB, [], use_llm=False, n_entries=12)
    assert g["action"] == "pass"
    assert any(i["type"] == "no_citation" for i in g["issues"])


@pytest.mark.parametrize("answer, n_entries", [
    (FAB + " [9]", 3),          # 引用越界
    (FAB + " [1]", 0),          # 有引用但上下文为空
], ids=["越界", "无上下文"])
def test_gate_out_of_range_still_high(answer, n_entries):
    """越界引用仍走 HIGH 通路（修复/兜底）——原通路没被加固破坏。"""
    g = V.gate("问题", answer, [], use_llm=False, n_entries=n_entries)
    assert g["action"] == "fallback" or any(i["sev"] == V.HIGH for i in g["issues"])


def test_gate_llm_guard_when_unconfigured(monkeypatch):
    """**无 key 时闸门必须安全降级**（不能因为调不了裁判模型就崩或放行错答案）。"""
    from paperpilot.tools import llm

    monkeypatch.delenv("PAPERPILOT_JUDGE_API_KEY", raising=False)
    assert llm.judge_configured() is False
    g = V.gate("问题", FAB + " [1]", [{"n": 1, "evidence": FAB, "page": 1}], n_entries=1)
    assert g["action"] in {"pass", "repaired", "fallback"}


# ───────────────────────── 本地：真实裁判模型（默认跳过）─────────────────────────


@pytest.mark.local
def test_llm_uncited_with_real_model():
    """`uv run pytest -m local` 时才跑：需要真实 key（CI 不跑）。"""
    from paperpilot.tools import llm

    if not llm.judge_configured():
        pytest.skip("未配置裁判模型（PAPERPILOT_JUDGE_*）")
    got = V._llm_uncited("这篇论文的核心方法是什么？", FAB)
    assert len(got) == 1 and got[0]["type"] == "no_citation"
    assert V._llm_uncited("问题", REFUSE) == []      # 自带守卫
