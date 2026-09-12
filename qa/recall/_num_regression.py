"""数字归一化回归集：真实误杀案例 + 边界写法（零 LLM）。

**为什么有这个文件**：2026-09-11 闸门诊断（`GATE_NUM_DIAG_20260911.md`）发现
QASPER 233 的闸门兜底 10 道里，18 个被报"无证据支撑"的数字 **18/18 是误杀**，
根因集中在四处：单位吸附缺词界、LaTeX 残留、年份剥离过宽、豁免②不含乘除。

设计原则（用户提出、已固化）：**封闭语法用规则一次写死；开放语义不裁决**。
本集把那批**真实案例**固化成回归用例 —— 新写法**进测试，不进生产事故**。

用法：
    uv run python qa/recall/_num_regression.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from paperpilot.components.validator import (_canonical_nums, _is_derived,  # noqa: E402
                                             _strip_year_context)

# (说明, 文本, 必须包含的值, 必须不含的值)
NUM_CASES: list[tuple[str, str, set[float], set[float]]] = [
    # ── ① 单位吸附：右词界（原文实测，QASPER 233 闸门误杀） ──
    ("右词界·most", "all features apart from the top 15,000 most frequent ones", {15000}, {1.5e10}),
    ("右词界·kernels", "We used PySpark in Jupyter notebooks with Python 2.7 kernels", {2.7}, {2700}),
    ("右词界·between", "We found a Pearson correlation of 0.81 between the two", {0.81}, {8.1e8}),
    ("右词界·words", "the essay has 500 words in total", {500}, {5e6}),
    # ── ② 单位吸附：左词界（`F1 百分点` 不得造出 100） ──
    ("左词界·F1百分点", "某些类别（support）的差距可超过 50 个 F1 百分点 [5]", None, {100}),
    ("左词界·百分整词", "提升了 50 个百分点", None, {5000}),
    ("左词界·v前缀仍提取", "We used PySpark v2.3 in Jupyter", {2.3}, None),
    # ── ③ LaTeX 残留（QASPER 文本源，封闭语法） ──
    ("LaTeX·星号幂", "this yields a search space of $7.30 * 10^{115}$ models", {7.3e115}, None),
    ("LaTeX·times幂", "Google N-grams : a corpus of $8.5 \\times 10^{11}$ tokens", {8.5e11}, None),
    ("LaTeX·转义百分号", "around 54$\\%$ tweets are neutral, 29$\\%$ positive", {54.0, 29.0}, None),
    # ── ④ 年份：只在明确语境剥离（`Switchboard-2000` 必须保留） ──
    ("年份·数据集名保留", "Switchboard-2000 contains 2000 more hours of speech", {2000}, None),
    ("年份·中文日期剥离", "该论文发表于 2019年，作者来自", None, {2019}),
    ("年份·英文语境剥离", "published in 2019 at the ACL conference", None, {2019}),
    # ── 常规写法不许回归 ──
    ("千分位", "there are 1,000,000 samples", {1e6}, None),
    ("欧式小数", "accuracy rose to 7,5 points", {7.5}, None),
    ("英文单位", "trained on 36 million tokens with 1.5b parameters", {3.6e7, 1.5e9}, None),
    ("中文网络单位", "大约 3.6w 用户", {3.6e4}, None),
    ("中文数词", "三万六千人参与", {36000.0}, None),
    ("日常百分号", "the score improved by 12.5% overall", {12.5}, None),
]

# 豁免②：四则运算 + 四舍五入容差（真实案例）
DERIVED_CASES: list[tuple[str, float, list[float], bool]] = [
    ("差量 2000−300=1700", 1700.0, [2000.0, 300.0], True),
    ("倍数 2000÷300≈6.67", 6.67, [2000.0, 300.0], True),
    ("除法+四舍五入 293÷4262≈0.069", 0.069, [293.0, 4262.0], True),
    ("舍入 0.0687→0.07", 0.07, [293.0, 4262.0], True),
    ("无关数字不该豁免", 999.0, [2000.0, 300.0], False),
]


def main() -> int:
    L: list[str] = ["# 数字归一化回归集结果（零 LLM）", "",
                    "> 真实误杀案例（QASPER 233 闸门诊断）+ 边界写法；改 `numbers.py`/`validator.py` 后必跑", ""]
    n_fail = 0

    L.append("## A. 数值归一化（validator 实际用的 `_canonical_nums`）")
    for note, text, must, must_not in NUM_CASES:
        got = _canonical_nums(text)
        bad = [v for v in (must or ())
               if not any(abs(g - v) <= max(1e-6, abs(v) * 1e-3) for g in got)]
        bad += [v for v in (must_not or ())
                if any(abs(g - v) <= max(1e-6, abs(v) * 1e-3) for g in got)]
        ok = not bad
        n_fail += 0 if ok else 1
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {note} → `{sorted(got)}`")
        if not ok:
            L.append(f"  - 期望含 `{sorted(must or ())}`／不含 `{sorted(must_not or ())}` → **违反 `{bad}`**")

    L += ["", "## B. 豁免② 派生数（四则运算 + 四舍五入）"]
    for note, v, ops, want in DERIVED_CASES:
        got = _is_derived(v, ops)
        ok = got == want
        n_fail += 0 if ok else 1
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {note} → got={got} want={want}")

    L += ["", "## C. 年份剥离（上下文感知）"]
    for note, text, keep in (("数据集名保留", "Switchboard-2000 hours", "2000"),
                             ("数值保留", "2000 more hours", "2000"),
                             ("中文日期剥离", "2019年发表", None),
                             ("英文语境剥离", "in 2019 the model", None)):
        out = _strip_year_context(text)
        ok = bool(keep in out if keep else "20" not in out)
        n_fail += 0 if ok else 1
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {note}: `{text}` → `{out}`")

    L += ["", "---", "",
          f"**结果：{'全部通过 ✅' if n_fail == 0 else f'{n_fail} 项失败 ❌'}**"]
    txt = "\n".join(L)
    (ROOT / "qa/recall/NUM_REGRESSION_20260911.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
