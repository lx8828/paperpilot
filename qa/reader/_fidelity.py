"""B 导读忠实度抽检：overview/guide/core_points 的数值事实 回论文原文回验。

方法：抽取三类数值（百分比 / 小数 / ≥4 位整数），逐个在 PDF 原文全文查找。
  命中=原文有该数（✅）；未命中=原文无该数 → 高误导风险候选（❌ 或格式差异，人工复核）。
局限：只查"数在不在"，不查"数与句子主语的关系对不对"；是忠实度的数值子集。
"""
from __future__ import annotations
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
from paperpilot.tools.pdf_parser import parse_pdf  # noqa: E402

PDFS = ["2609.04170v1.pdf", "2609.03035v1.pdf", "2608.31079v1.pdf",
        "2609.02056v1.pdf", "2609.02786v1.pdf"]
PAPERS = ROOT / "src" / "paperpilot" / "storage" / "papers"

# 百分比 或 小数 或 ≥4 位整数
NUM_RE = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d+%|\.\d+%|\d+\.\d+%|\d+%|"
                    r"\d{1,3}(?:,\d{3})*\.\d+|\.\d+|\d+\.\d+|\d{4,})")


def normalize(t: str) -> str:
    return t.replace(",", "").replace(" ", "").replace("\u2009", "")


def variants(num: str) -> list[str]:
    """数值 → 待查变体（去逗号/空格 + 带%原样）。"""
    n = normalize(num)
    if num.endswith("%"):
        return [n, n[:-1]]
    return [n, n + "%"]


def units_of(pdf: str) -> list[tuple[str, str]]:
    """(来源, 文本)。overview/guide 全文 + 每条 core_point。"""
    stem = pdf.replace(".pdf", "")
    out = []
    ov = json.load(open(ROOT / f"out_views/{stem}.overview.json", encoding="utf-8"))
    gd = json.load(open(ROOT / f"out_views/{stem}.guide.json", encoding="utf-8"))
    rep = json.load(open(ROOT / f"out_views/{stem}.report.json", encoding="utf-8"))
    out.append(("overview", str(ov.get("overview", ""))))
    out.append(("guide", str(gd.get("guide", ""))))
    for cp in (rep.get("core_points") or [])[:8]:
        out.append(("core_points", str(cp.get("text", ""))))
    return out


def main() -> int:
    rows = []
    all_miss: list[dict] = []
    for pdf in PDFS:
        raw = parse_pdf(str(PAPERS / pdf))["raw_text"]
        raw_n = normalize(raw)
        tot = hit = 0
        per_miss = []
        for src, text in units_of(pdf):
            for m in NUM_RE.finditer(text):
                num = m.group(1)
                if num in ("100%",):
                    pass
                ok = any(nv in raw_n for nv in variants(num))
                tot += 1
                hit += bool(ok)
                if not ok:
                    # 提取一句上下文用于人工判断
                    s = max(0, m.start() - 28)
                    per_miss.append({"num": num, "src": src,
                                     "ctx": text[s:m.end() + 30].replace("\n", " ")})
                    all_miss.append({"pdf": pdf, "num": num, "src": src,
                                     "ctx": text[s:m.end() + 30].replace("\n", " ")})
        rows.append({"pdf": pdf, "total": tot, "hit": hit,
                     "miss": tot - hit,
                     "miss_rate": round((tot - hit) / tot, 3) if tot else 0})
        print(f"{pdf}: 数值 {tot} | 原文命中 {hit} | 未命中 {tot-hit} "
              f"({(tot-hit)/tot:.0%})" if tot else f"{pdf}: 无数值")
    print("\n=== 未命中明细（高误导风险候选，需人工判断是真幻觉还是格式差异） ===")
    for m in all_miss:
        print(f"  [{m['pdf'][:12]}] {m['num']:>8} ({m['src']}) …{m['ctx']}")
    Path("qa/reader/fidelity_result.json").write_text(
        json.dumps({"per_pdf": rows, "misses": all_miss}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
