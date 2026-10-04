"""输出闸门的**纯机器**判据（迁移自 `qa/recall/_selftest_validator_20260913.py`，
该脚本**已不在仓库**）。

覆盖审查项「无引用、无依据的答案会被静默放行」的修复，并守住一条硬约束：
**`gate()` 的动作不变**（仍只对 HIGH 拦）——否则一次"顺手加固"就会把好答案换成兜底话术。

LLM 体检部分（真实裁判模型）标记为 `local`，默认跳过（CI 不需要 key）。
"""
from __future__ import annotations

import pytest

from paperpilot.components import repairer
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


# ─────────────── 兜底前改走「LLM 补充说明」（2026-10-02）───────────────
# 硬约束：**不接 salvage 时行为必须与旧版逐位一致**（仍是拒答），
# 否则一次"顺手加功能"就会把旧链路的行为漂掉。

_CITES1 = [{"evidence": FAB, "page": 1, "ref": "c1"}]
_HIGH_ANS = FAB + " [9]"          # [9] 越界 → HIGH（见上面 test_gate_out_of_range_still_high）


def test_gate_salvage_default_off_keeps_refusal():
    """不传 `salvage`（= 旧接线）→ 仍是拒答。这是本档的行为不变约束。"""
    g = V.gate("问题", _HIGH_ANS, [], use_llm=False, n_entries=3)
    assert g["action"] == "fallback"
    assert g["answer"] == V.FALLBACK_MSG


def test_gate_uses_salvage_after_repair_fails():
    """repair 修不动 → 不再拒答，改走 salvage（`action="supplemented"`）。"""
    g = V.gate("问题", _HIGH_ANS, _CITES1, repair=lambda *a: None,
               salvage=lambda *a: (V.SUPPLEMENT_MARK + "\n能从证据确证的部分…", _CITES1),
               use_llm=False, n_entries=3)
    assert g["action"] == "supplemented"
    assert g["answer"].startswith(V.SUPPLEMENT_MARK)
    assert g["cites"] == _CITES1


def test_gate_repair_wins_over_salvage():
    """repair 成功时**不**再叫 salvage（不多花一次调用）。"""
    called: list[int] = []
    g = V.gate("问题", _HIGH_ANS, _CITES1, repair=lambda *a: ("修好的答案 [1]", _CITES1),
               salvage=lambda *a: called.append(1), use_llm=False, n_entries=3)
    assert g["action"] == "repaired"
    assert not called


def test_gate_salvage_exception_falls_back():
    """salvage 抛错 → 绝不外抛，退回避答。"""
    def _boom(*_a):
        raise RuntimeError("模拟补充调用爆炸")

    g = V.gate("问题", _HIGH_ANS, _CITES1, repair=lambda *a: None, salvage=_boom,
               use_llm=False, n_entries=3)
    assert g["action"] == "fallback"
    assert g["answer"] == V.FALLBACK_MSG


def test_gate_salvage_empty_result_falls_back():
    """salvage 返回空 → 视为没补成，仍拒答。"""
    g = V.gate("问题", _HIGH_ANS, _CITES1, repair=lambda *a: None,
               salvage=lambda *a: ("   ", _CITES1), use_llm=False, n_entries=3)
    assert g["action"] == "fallback"


def test_salvage_forces_banner(monkeypatch):
    """★ 模型忘写标注也必须补上 —— "不得冒充系统作答"不能依赖模型自觉。"""
    monkeypatch.setattr(repairer.llm, "chat_text",
                        lambda *a, **k: "能从证据确证：本文提出双塔检索结构 [1]。")
    got = repairer.salvage("问题", _HIGH_ANS,
                           [{"sev": V.HIGH, "type": "citation", "detail": "引用越界"}], _CITES1)
    assert got is not None
    text, cites = got
    assert text.startswith(V.SUPPLEMENT_MARK)
    assert cites == _CITES1                 # 依据同一批被引证据 → cites 原样带过


def test_salvage_needs_cites():
    """没有被引原文 → 没什么可补充的 → None（交给拒答）。"""
    assert repairer.salvage("问题", _HIGH_ANS, [], []) is None


def test_gate_marks_any_salvage_output():
    """★ 标注由 **gate 这一层**强制 —— 换任何 salvage 实现都不会漏（不依赖实现自觉）。

    这是"用户必须能分清系统作答与补充"的硬保证。
    """
    g = V.gate("问题", _HIGH_ANS, _CITES1, repair=lambda *a: None,
               salvage=lambda *a: ("这里是一段没有任何标注的补充文本。", _CITES1),
               use_llm=False, n_entries=3)
    assert g["action"] == "supplemented"
    assert g["answer"].startswith(V.SUPPLEMENT_MARK)


# ───────────────────────── 本地：真实裁判模型（默认跳过）─────────────────────────


@pytest.mark.local
def test_llm_uncited_with_real_model(real_env):
    """`uv run pytest -m local` 时才跑：需要真实 key（CI 不跑）。

    `real_env`（conftest）负责把本机 `.env` 灌进来 —— 否则 autouse 的 `isolate`
    已经把 key 清空，这条会**永远 skip**（等于没有）。
    """
    from paperpilot.tools import llm

    if not llm.judge_configured():
        pytest.skip("未配置裁判模型（PAPERPILOT_JUDGE_*）")
    got = V._llm_uncited("这篇论文的核心方法是什么？", FAB)
    assert len(got) == 1 and got[0]["type"] == "no_citation"
    assert V._llm_uncited("问题", REFUSE) == []      # 自带守卫
