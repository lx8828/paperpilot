"""临时：QASPER 段落长度分布——超长段(>1600/2400/4000)占比、落在哪些节、是否覆盖 gold。

决定 B2 是否需要句级再切：若超长段多且常命中 gold，均匀目标在这些题上失效。
只读文本统计，零模型零检索。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from paperpilot.qasper_source import (_clean_para, _split_title,  # noqa: E402
                                      gold_answer, load_papers)


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    pids = sorted({it["pid"] for it in data["items"]})
    tot = over = 0
    bucket = {">1600": 0, ">2400": 0, ">4000": 0, ">8000": 0}
    by_leaf: dict[str, int] = {}
    examples: list[tuple[int, str, str]] = []  # (len, leaf, head)
    gold_in_over = 0
    gold_total = 0
    # 题 → 是否 gold evidence 落在超长段
    qid_by_paper: dict[str, list[str]] = {}
    for it in data["items"]:
        qid_by_paper.setdefault(it["pid"], []).append(it["qid"])
    for pid in pids:
        paper = papers[pid]
        para_index: list[tuple[list[str], str, int]] = []  # (path, cleaned, idx)
        for sec in paper.get("full_text") or []:
            path = _split_title(sec.get("section_name") or "") or ["(PREAMBLE)"]
            for p in (sec.get("paragraphs") or []):
                cl = _clean_para(p)
                if not cl:
                    continue
                para_index.append((path, cl, len(cl)))
                n = len(cl)
                tot += 1
                if n > 1600:
                    over += 1
                    for lab, thr in ((">1600", 1600), (">2400", 2400),
                                     (">4000", 4000), (">8000", 8000)):
                        if n > thr:
                            bucket[lab] += 1
                    leaf = path[-1]
                    by_leaf[leaf] = by_leaf.get(leaf, 0) + 1
                    if len(examples) < 6:
                        examples.append((n, leaf, cl[:90].replace("\n", " ")))
        # gold evidence 落在超长段？
        qas = {str(q.get("question_id") or ""): q for q in paper.get("qas") or []}
        for qid in qid_by_paper.get(pid, []):
            q = qas.get(qid)
            if not q:
                continue
            _g, ev = gold_answer(q)
            if not ev:
                continue
            gold_total += 1
            if any(ev.replace("\n", " ") in cl or cl in ev.replace("\n", " ")
                   for _p, cl, _n in para_index if _n > 1600):
                gold_in_over += 1
    L = ["# QASPER 段落长度分布（recall_set %d 篇）" % len(pids), ""]
    L.append(f"- 段落总数 {tot} | 超 1600 字符 {over} 段（{over/tot*100:.1f}%）")
    L.append(f"- 分桶：>1600 {bucket['>1600']} | >2400 {bucket['>2400']} | "
             f">4000 {bucket['>4000']} | >8000 {bucket['>8000']}")
    L.append("")
    L.append("### 超长段所在叶标题 Top10")
    for leaf, c in sorted(by_leaf.items(), key=lambda x: -x[1])[:10]:
        L.append(f"- {leaf}: {c}")
    L.append("")
    L.append("### 样例")
    for n, leaf, head in examples:
        L.append(f"- [{n} 字符 | {leaf}] {head}")
    L.append("")
    L.append(f"> gold evidence 落在超长段的题：{gold_in_over}/{gold_total} "
             f"({gold_in_over/max(gold_total,1)*100:.1f}%)")
    txt = "\n".join(L)
    print(txt)
    Path("qa/recall/PARA_LEN_20260910.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
