"""临时：chunk 粒度不均匀量化 + 与题目难度的混杂检查。

回答：用户猜"按标题切导致块大小悬殊、中位1000+、min~1000、max4000、不均隐藏难测"。
本脚本：
  A. 精确分布（min/p10/p25/中位/p75/p90/max、变异系数 CV、分桶占比）
  B. 超小块构成：是否多为摘要/短节；gold 落小块的题是否偏 hard
  C. "切到上限4000" 的大整节块占比
  D. 细标题碎块：单段=1的小块数量与题命中分布
只读 chunks，不加载 embedder / 不检索。
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402

BUCKETS = [(0, 300, "<300"), (300, 800, "300-800"), (800, 1500, "800-1500"),
           (1500, 2500, "1500-2500"), (2500, 4000, "2500-4000"), (4000, 10**9, ">4000")]


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    pids = sorted({it["pid"] for it in data["items"]})
    lens: list[int] = []
    blocks_n: list[int] = []
    single_para: list[int] = []   # 单段块的长
    cap_full: list[int] = []      # ≥3800 逼近上限的块长
    per_sec: dict[str, list[int]] = {}
    leaf_lens: list[int] = []
    for pid in pids:
        for c in ordered_chunks(f"qasper_{pid}.qpdf"):
            ln = len(c.text)
            lens.append(ln)
            blocks_n.append(c.n_blocks)
            if c.n_blocks == 1:
                single_para.append(ln)
            if ln >= 3800:
                cap_full.append(ln)
            leaf = (c.title_path or ["?"])[-1]
            leaf_lens.append(ln)
            per_sec.setdefault(leaf, []).append(ln)
    n = len(lens)
    ls = sorted(lens)
    mu = statistics.mean(ls)
    cv = statistics.stdev(ls) / mu
    def pct(p):
        return ls[min(n - 1, int(p / 100 * n))]
    L = ["# chunk 粒度不均匀量化（recall_set 208 篇，%d 块）" % n, ""]
    L.append(f"- 字符：min {ls[0]} | p10 {pct(10)} | p25 {pct(25)} | 中位 {pct(50)} | "
             f"p75 {pct(75)} | p90 {pct(90)} | max {max(ls)}")
    L.append(f"- 段落数：min {min(blocks_n)} | p25 {sorted(blocks_n)[len(blocks_n)//4]} | "
             f"中位 {statistics.median(blocks_n)} | p75 {sorted(blocks_n)[3*len(blocks_n)//4]} | max {max(blocks_n)}")
    L.append(f"- 均值 {mu:.0f} | 标准差 {statistics.stdev(ls):.0f} | 变异系数 CV {cv:.2f} "
             f"（CV>1=极不均匀；均匀文本分块通常 0.3-0.5）")
    L.append("")
    L.append("### 分桶占比")
    L.append("| 桶 | 块数 | 占比 | 累计字符占比 |")
    L.append("|---|---|---|---|")
    tot_chars = sum(lens)
    acc = 0
    for lo, hi, lab in BUCKETS:
        cnt = sum(1 for x in lens if lo <= x < hi)
        acc += sum(x for x in lens if lo <= x < hi)
        L.append(f"| {lab} | {cnt} | {cnt/n*100:.1f}% | {acc/tot_chars*100:.1f}% |")
    L.append("")
    L.append(f"- 单段碎块：{len(single_para)} 块（{len(single_para)/n*100:.0f}%），"
             f"长 中位 {statistics.median(single_para) if single_para else 0} | max {max(single_para) if single_para else 0}")
    L.append(f"- ≥3800 逼近上限（整节切满）：{len(cap_full)} 块（{len(cap_full)/n*100:.0f}%），"
             f"其中 >4000 的 {sum(1 for x in lens if x > 4000)} 块（超出阈值=单段超大）")
    L.append("")
    # 标题层级名出现的小块
    tiny = [x for x in lens if x < 400]
    L.append(f"- <400 字符的极小块：{len(tiny)} 块（{len(tiny)/n*100:.1f}%）")
    L.append("")
    L.append("### 细标题样例：同一篇论文里 leaf 标题对应的块大小波动")
    from collections import Counter
    leaf_occ = Counter({k: len(v) for k, v in per_sec.items()})
    print("\n".join(L))
    Path("qa/recall/CHUNK_GRAN_20260910.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
