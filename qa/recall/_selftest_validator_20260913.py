"""Validator 自测（2026-09-13，审查修复 A 档）。

覆盖审查项「无引用、无依据的答案会被放行」的修复：
  ① 零引用独立成 `no_citation` 类型；
  ② 含实质断言 → MID、纯拒答/过短 → LOW（机器判据 `_substantive_claim`）；
  ③ 零引用时**不再跳过检查** → 跑格式体检 `_llm_uncited`（LLM，仅必要时触发）；
  ④ **gate 动作不变**（仍只对 HIGH 拦）——这是本档的硬约束，必须回归验证。

用法：
    uv run python qa/recall/_selftest_validator_20260913.py          # 纯机器（0 LLM 调用）
    PP_LLM=1 uv run python qa/recall/_selftest_validator_20260913.py  # 额外跑一次真实格式体检
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FAILS: list[str] = []


def ck(name: str, cond: bool) -> None:
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


FAB = "本文提出了一种双塔检索结构，并用对比学习在多个数据集上验证了有效性。"
REFUSE = "抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。"
SHORT = "文中未给出该数值。"


def main() -> int:
    from paperpilot.components import validator as V

    # ① + ② 机器判据
    ck("实质断言被识别（含数值）", V._substantive_claim(FAB + "提升 87.3%")[0] is True)
    ck("实质断言被识别（纯陈述）", V._substantive_claim(FAB)[0] is True)
    ck("拒答话术不算实质断言", V._substantive_claim(REFUSE)[0] is False)
    ck("过短概括不算实质断言", V._substantive_claim(SHORT)[0] is False)
    ck("空答案不算实质断言", V._substantive_claim("")[0] is False)

    # 类型独立 + 分级
    iss, _, _ = V._machine_checks("这篇论文的核心方法是什么？", FAB, [], n_entries=12)
    nc = [i for i in iss if i["type"] == "no_citation"]
    ck("零引用 → 独立类型 no_citation", len(nc) == 1)
    ck("实质断言 → MID", nc and nc[0]["sev"] == V.MID)
    ck("带上 substantive 标记（可统计）", nc and nc[0].get("substantive") is True)
    iss2, _, _ = V._machine_checks("这篇论文的核心方法是什么？", REFUSE, [], n_entries=12)
    nc2 = [i for i in iss2 if i["type"] == "no_citation"]
    ck("拒答式零引用 → LOW（不误标）", nc2 and nc2[0]["sev"] == V.LOW)

    # ④ 硬约束：动作不变（零引用仍 pass，只是被标注）
    g = V.gate("这篇论文的核心方法是什么？", FAB, [], use_llm=False, n_entries=12)
    ck("gate 动作不变：零引用仍 pass", g["action"] == "pass")
    ck("gate 返回里带 no_citation 标注（供前端/记录）",
       any(i["type"] == "no_citation" for i in g["issues"]))

    # 越界引用仍拦（回归：HIGH 通路没被破坏）
    g2 = V.gate("问题", FAB + " [9]", [], use_llm=False, n_entries=3)
    ck("越界引用仍 HIGH → fallback（原通路未破）",
       g2["action"] == "fallback" or any(i["sev"] == V.HIGH for i in g2["issues"]))

    # ③ 格式体检（需要 LLM；可选）
    if os.environ.get("PP_LLM") == "1":
        from paperpilot.tools import llm
        llm._load_dotenv(str(ROOT))
        got = V._llm_uncited("这篇论文的核心方法是什么？", FAB)
        ck("格式体检可用（返回 1 条 MID no_citation）",
           len(got) == 1 and got[0]["type"] == "no_citation" and got[0]["sev"] == V.MID)
        print("       体检输出:", str(got[0].get("detail", ""))[:110] if got else "(空)")
        got2 = V._llm_uncited("这篇论文的核心方法是什么？", REFUSE)
        ck("格式体检自带守卫：拒答话术直接返回空（不花调用、不误报）", got2 == [])
    else:
        print("  （跳过 LLM 格式体检：设 PP_LLM=1 可测）")

    print()
    print("SELFTEST-VALIDATOR:", "ALL PASS" if not FAILS else f"FAILED ({len(FAILS)})")
    for f in FAILS:
        print("   FAIL:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
