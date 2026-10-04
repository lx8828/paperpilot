r"""演示：**闸门修复失败 → LLM 补充说明（标注「非系统作答」）**。

用**桩 LLM**（零网络、零花费）把流水线跑通，只为看**用户最终看到的样子**。
真实链路里的接线在 `graph/__init__.py`（`salvage=_salv`，env `PAPERPILOT_VALIDATOR_SALVAGE` 默认开）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from paperpilot.components import repairer  # noqa: E402
from paperpilot.components import validator as V  # noqa: E402

FAB = "本文提出了一种双塔检索结构，并用对比学习在多个数据集上验证了有效性。"
CITES = [{"evidence": FAB, "page": 3, "ref": "c1"}]
# 故意造一个 HIGH：正文说「12 个数据集 / 提升 87.3%」，但引用 [9] 越界（上下文只有 3 条）
ANS = FAB + " 并在 12 个数据集上平均提升 87.3% [9]"

# —— 桩掉 LLM：模拟一次真实的「补充说明」输出 ——
repairer.llm.chat_text = lambda *a, **k: (          # type: ignore[assignment]
    "**能从证据确证的部分**\n"
    "- 本文提出了一种双塔检索结构 [1]。\n"
    "- 该结构用对比学习在多个数据集上做了验证，但**未给出数据集数量** [1]。\n\n"
    "**无法确证的部分**\n"
    "- 原答案称「在 12 个数据集上平均提升 87.3%」："
    "在【可引用原文】中**未找到支撑**，且其引用编号 [9] 越界。\n"
    "- 数据集数量与提升幅度**给定文本中未提及**，无法确认。"
)


def show(title: str, g: dict) -> None:
    print("=" * 88)
    print(title)
    print(f"  action = {g['action']} ｜ cites = {len(g['cites'])} 条")
    print("-" * 88)
    print(g["answer"])
    print()


def main() -> int:
    print("素材：正文＝双塔检索结构；断言＝「12 个数据集 / +87.3%」+ 越界引用 [9]（HIGH）\n")

    show("① 旧行为：不接 salvage → 直接拒答",
         V.gate("本文的方法是什么？效果如何？", ANS, CITES,
                repair=lambda *a: None, use_llm=False, n_entries=3))

    show("② 新行为：repair 失败 → LLM 补充说明（标注「非系统作答」）",
         V.gate("本文的方法是什么？效果如何？", ANS, CITES,
                repair=lambda *a: None, salvage=repairer.salvage,
                use_llm=False, n_entries=3))

    show("③ 安全网：模型忘写标注 → 代码强制补上",
         V.gate("本文的方法是什么？效果如何？", ANS, CITES,
                repair=lambda *a: None,
                salvage=lambda *a: ("能从证据确证：双塔检索结构 [1]。", CITES),
                use_llm=False, n_entries=3))

    show("④ 反例：连被引原文都没有 → 没什么可补充的 → 仍拒答",
         V.gate("本文的方法是什么？效果如何？", ANS, [],
                repair=lambda *a: None, salvage=repairer.salvage,
                use_llm=False, n_entries=3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
