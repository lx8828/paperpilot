"""离线判定：闸门报的"无证据支撑数字"到底在不在论文里（零 LLM）。

读 `_gate_probe_qasper.py` 产出的诊断报告，对每道题、每个被报数字判为：

    A 归一化后就能在论文里匹配到        → 正常（不该报）
    B 归一化后匹配不到，但**原文含该数字的字符串**（含千分位/整数形式）
      → **误杀：归一化把原文的数字搞坏了**（本轮回查的主力根因）
    C 原文没有，但可由草稿里其他数字四则运算推得 → **误杀**（豁免② 只覆盖 差/和）
    D 都不满足                            → **真拦**（疑似编数，闸门判对了）

用法：
    uv run python qa/recall/_gate_num_origin.py --md qa/recall/GATE_PROBE_QASPER_20260911.md
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import retrieval_chunks  # noqa: E402
from paperpilot.components.validator import _canonical_nums  # noqa: E402

_BLOCK = re.compile(r"## \d+\. (\S+?)｜(.+)")
_FLAG = re.compile(r"疑似无证据支撑的数字: (.+)")
_DRAFT_NUMS = re.compile(r"草稿里 validator 提取到的数字: `\[(.*?)\]`")


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-6)


def _raw_forms(v: float) -> list[str]:
    """该数值在原文里可能出现的字符串形式（千分位/纯整数/小数）。"""
    forms = [f"{v:g}"]
    if abs(v - round(v)) < 1e-9:
        n = int(round(v))
        forms += [str(n), f"{n:,}"]
    return sorted(set(forms))


# LaTeX 科学计数法（QASPER 文本源里公式以源码残留，如 `$8.5 \times 10^{11}$`、
# `$7.30 * 10^{115}$`）——scan_numbers 会拆成 8.5 / 10 / 11 三个数，合成不了。
_SCI_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\\times|\*|×)\s*10\s*\^?\s*\{?\s*(\d+)")


def _sci_set(text: str) -> set[float]:
    out: set[float] = set()
    for mant, exp in _SCI_RE.findall(text or ""):
        try:
            out.add(float(mant) * (10 ** int(exp)))
        except (ValueError, OverflowError):
            continue
    return out


def _derived(v: float, ops: list[float]) -> str:
    """返回推导式（空串=推不出）。四则运算：差/和/积/商（豁免②只认差/和）。"""
    def near(x: float, y: float) -> bool:
        # 容忍答案的四舍五入（0.0687 写成 0.069 / 0.07）
        return (math.isclose(x, y, rel_tol=1e-3, abs_tol=1e-6)
                or math.isclose(x, y, rel_tol=5e-2, abs_tol=1e-6))

    for i, a in enumerate(ops):
        for b in ops[i + 1:]:
            if near(abs(a - b), v):
                return f"{a:g} − {b:g}"
            if near(a + b, v):
                return f"{a:g} + {b:g}"
            if b and near(a * b, v):
                return f"{a:g} × {b:g}"
            if b and near(a / b, v):
                return f"{a:g} ÷ {b:g} ≈"
            if a and near(b / a, v):
                return f"{b:g} ÷ {a:g} ≈"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="qa/recall/GATE_PROBE_QASPER_20260911.md")
    args = ap.parse_args()

    md = (ROOT / args.md).read_text(encoding="utf-8")
    blocks = [b for b in md.split("=" * 96) if "疑似无证据支撑的数字" in b]

    L: list[str] = ["# 闸门「无证据支撑数字」溯源判定（离线，零 LLM）", "",
                    f"> 来源 `{args.md}`｜对每个被报数字查：**在论文 chunk 里吗？能由其他数字推得吗？**", ""]
    tally = {"A 归一化即匹配": 0, "B 原文有·归一化坏": 0, "C 可推导": 0, "D 真找不到": 0}
    cache: dict[str, list[str]] = {}

    for b in blocks:
        m = _BLOCK.search(b)
        fm = _FLAG.search(b)
        if not (m and fm):
            continue
        pid, q = m.group(1), m.group(2).strip()
        flagged = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e\+?\d+)?", fm.group(1))]
        dm = _DRAFT_NUMS.search(b)
        draft_nums = ([float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", dm.group(1))]
                      if dm else [])

        pdf = f"qasper_{pid}.qpdf"
        if pdf not in cache:
            try:
                cache[pdf] = [str(getattr(c, "text", "") or "")
                              for c in retrieval_chunks(pdf)]
            except Exception as e:  # noqa: BLE001
                cache[pdf] = []
                L.append(f"> ⚠️ {pdf} 取 chunk 失败: {type(e).__name__}: {e}")
        chunks = cache[pdf]
        chunk_nums = [_canonical_nums(t) for t in chunks]
        full = "\n".join(chunks)

        L.append(f"## {pid}｜{q}")
        L.append(f"- 论文 chunk 数: **{len(chunks)}**｜共 {len(full):,} 字符")
        for v in flagged:
            hit_i = [i for i, s in enumerate(chunk_nums) if any(_close(v, x) for x in s)]
            if hit_i:
                tally["A 归一化即匹配"] += 1
                L.append(f"- `{v:g}` → **A**（chunk #{hit_i[:4]}）→ 归一化后本该匹配，不该报")
                continue
            if any(_close(v, x) for x in _sci_set(full)):
                tally["B 原文有·归一化坏"] += 1
                L.append(f"- `{v:g}` → **B 原文以 LaTeX 科学计数法存在**"
                         f"（`$… \\times 10^{{…}}$`）→ scan 拆成三个数合成不了 → **误杀**")
                continue
            raw = [f for f in _raw_forms(v) if f in full]
            if raw:
                tally["B 原文有·归一化坏"] += 1
                pos = full.find(raw[0])
                ctx = full[max(0, pos - 70): pos + 40].replace("\n", " ")
                L.append(f"- `{v:g}` → **B 原文确有 `{raw[0]}`，归一化后丢了**")
                L.append(f"  - 原文：`{ctx}` → **误杀（解析把数字弄坏）**")
                continue
            expr = _derived(v, [x for x in draft_nums if not _close(x, v)])
            if expr:
                tally["C 可推导"] += 1
                L.append(f"- `{v:g}` → **C 可由草稿数字推得**（`{expr}`）→ **误杀**（豁免② 不覆盖）")
            else:
                tally["D 真找不到"] += 1
                L.append(f"- `{v:g}` → **D 论文里找不到、也推不出** → **真拦（疑似编数）**")
        L.append("")

    mis = tally["B 原文有·归一化坏"] + tally["C 可推导"]
    L += ["## 汇总", "", f"- {tally}", "",
          f"> 误杀（B 归一化损坏 + C 豁免不覆盖）= **{mis}**｜真拦（D）= **{tally['D 真找不到']}**｜"
          f"A（理论上不该报）= {tally['A 归一化即匹配']}"]
    txt = "\n".join(L)
    out = ROOT / "qa/recall/GATE_NUM_ORIGIN_20260911.md"
    out.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\n已写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
