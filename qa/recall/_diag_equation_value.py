"""公式块值不值得注入？三条实测（零 LLM）。

① 被引用的外部块里，有多少其实是公式（而非表格）
② 公式块相对 QASPER 正文段落的**冗余度**（内容词命中率）
③ 有多少题的 gold 含数学标记（可能存在「只在公式里」的信息），其通过率如何

用法：uv run python qa/recall/_diag_equation_value.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools.mineru_bridge import _find_content_list, element_text  # noqa: E402

TOK = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
EQ_SIG = re.compile(r"\$\$|\\\\(frac|sum|alpha|beta|theta|mathbb|mathrm)|[_^]\{")
MATH_GOLD = re.compile(r"[\\$]|[_^]\{|\\\\?(frac|sum|alpha|beta|theta)")


def main() -> int:
    papers = load_papers()

    print("=== ① 翻绿题引用的外部块里，有多少其实是公式？===")
    d = json.loads(Path("qa/recall/qasper_tbl_cites_20260911.json").read_text(encoding="utf-8"))
    tot = eq = 0
    for x in d:
        for h in x["xtbl_refs"]:
            tot += 1
            if EQ_SIG.search(h["ev"]):
                eq += 1
    print(f"  引用到的外部块 {tot} 条，其中公式 {eq} 条（{eq/max(tot,1):.0%}）")

    print("\n=== ② 公式块的冗余度（内容词有多少已在正文段落里）===")
    rates: list[float] = []
    n_eq = 0
    for dd in sorted(Path("assets/artifacts/out_mineru").iterdir()):
        if not dd.is_dir():
            continue
        pid = dd.name[:-2] if dd.name.endswith("v1") else dd.name
        paper = papers.get(pid)
        if not paper:
            continue
        full = " ".join(str(pp) for s in (paper.get("full_text") or [])
                        for pp in (s.get("paragraphs") or []))
        ftok = {w.lower() for w in TOK.findall(full)}
        for e in _find_content_list(dd) or []:
            if e.get("type") != "equation":
                continue
            t = element_text(e) or ""
            toks = {w.lower() for w in TOK.findall(t)}
            if not toks:
                continue
            n_eq += 1
            rates.append(sum(1 for w in toks if w in ftok) / len(toks))
    r = np.array(rates)
    print(f"  公式块 {n_eq} 个｜内容词命中正文比例：中位 {np.median(r):.2f}｜"
          f"p10 {np.percentile(r,10):.2f}｜p90 {np.percentile(r,90):.2f}")
    print(f"  完全冗余(>=0.9) {(r>=0.9).mean():.0%}｜高度冗余(>=0.7) {(r>=0.7).mean():.0%}｜"
          f"低冗余(<0.3) {(r<0.3).mean():.0%}")

    print("\n=== ③ gold 含数学标记的题量与通过率 ===")
    run = json.loads(Path("qa/qasper_run_20260911_072510.json").read_text(encoding="utf-8"))
    math_q = [x for x in run if MATH_GOLD.search(x.get("gold") or "")]
    other = [x for x in run if x not in math_q]
    P = lambda g: sum(1 for x in g if x.get("status") == "pass")  # noqa: E731
    print(f"  gold 含数学标记：{len(math_q)}/{len(run)}（{len(math_q)/len(run):.0%}）｜"
          f"通过率 {P(math_q)}/{len(math_q)} = {P(math_q)/max(len(math_q),1):.1%}")
    print(f"  其余题通过率：{P(other)}/{len(other)} = {P(other)/max(len(other),1):.1%}")
    for x in math_q[:8]:
        print(f"    - {x['paper'][-14:]} | {str(x.get('gold'))[:64]} | {x['status']} {x['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
