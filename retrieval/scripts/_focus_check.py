"""指代解析单元自检（临时，秒级、无 LLM）。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from paperpilot.components.focus import _names, resolve_focus  # noqa: E402

CORPUS = ["2609.02094v1.pdf", "2605.09341.pdf", "2606.18837.pdf",
          "2604.20087.pdf", "2606.25389.pdf"]

CASES = [
    ("篇 1 用了什么数据集？",                 ["2609.02094v1.pdf"]),
    ("第 2 篇的实验设置是什么？",              ["2605.09341.pdf"]),
    ("paper 3 的方法是什么？",                ["2606.18837.pdf"]),
    ("篇A 讲了什么？",                        ["2609.02094v1.pdf"]),
    ("篇 2 和篇 4 分别用了什么数据集？",        ["2605.09341.pdf", "2604.20087.pdf"]),
    ("在 MASkills 中，技能库怎么演化？",       ["2609.02094v1.pdf"]),
    ("这 5 篇分别用了什么数据集？",            []),
    ("这篇论文研究什么问题？",                 []),
    ("实验结果如何？",                        []),
    ("CrossSum 用了哪些语言对？",             []),   # 不在语料里 → 不该命中
]


def main() -> int:
    print("[短名候选]（篇名指代靠它）")
    for p, ns in _names(tuple(CORPUS)).items():
        print(f"  {p:<20} {list(ns)}")

    print("\n[解析]")
    bad = 0
    for q, want in CASES:
        got = resolve_focus(q, CORPUS)
        ok = got == want
        bad += not ok
        print(f"  {'✓' if ok else '✗'} {q:<34} -> "
              f"{[x.replace('.pdf', '') for x in got]}"
              + ("" if ok else f"   **期望 {[x.replace('.pdf','') for x in want]}**"))
    print(f"\n  {len(CASES) - bad}/{len(CASES)} 通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
