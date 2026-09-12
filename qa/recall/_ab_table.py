"""任务①结果汇总：三臂（A0 纯 pymupdf / A1 注入 / A2 全链 MinerU）对照。

重点两份口径：
  · 全 30 题（整体）
  · **表格子集**（mineru_hard_set 里 src 含 "Table" 的题，n=12）——这才是"MinerU vs pymupdf"的关键轴

用法：uv run python qa/recall/_ab_table.py
产物：qa/recall/MINERU_AB_20260911.md
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SET = Path("qa/recall/mineru_hard_set_v1.json")
ARMS = [("plain", "A0 纯 pymupdf（无注入）"),
        ("inject", "A1 注入（线上默认）"),
        ("mineru", "A2 全链 MinerU")]
PASS = 4


def load(arm: str) -> dict[str, dict]:
    p = Path(f"qa/recall/ab_{arm}_r1.json")
    if not p.exists():
        return {}
    return {r["qid"]: r for r in json.loads(p.read_text(encoding="utf-8"))}


def main() -> int:
    items = {x["qid"]: x for x in json.loads(SET.read_text(encoding="utf-8"))["items"]}
    table_qids = [q for q, x in items.items() if "Table" in (x.get("src") or "")]
    arms = {a: load(a) for a, _ in ARMS}

    def stat(recs: dict, qids: list[str]) -> tuple[int, int, float]:
        sub = [recs[q] for q in qids if q in recs]
        if not sub:
            return 0, 0, 0.0
        p = sum(1 for r in sub if r["score"] >= PASS)
        return p, len(sub), sum(r["score"] for r in sub) / len(sub)

    L = ["# MinerU vs pymupdf：表格题 A/B（2026-09-11）", "",
         "> 链路：graph.ask（v3 两级 + nol3j + 闸门）｜裁判：glm-4-flash ≥4=pass",
         "> 题目：`qa/recall/mineru_hard_set_v1.json`（10 篇真实 PDF × 3 题 = 30）",
         "> 三臂均**只改解析源**，其余（检索/answer/裁判/题集）完全一致。", "",
         "## 0. 三臂定义", "",
         "| 臂 | 正文（chunk）源 | 表格/公式 | 触发 |",
         "|---|---|---|---|",
         "| **A0** plain | pymupdf 切分 | **无** | 评测开关 `PAPERPILOT_MINERU_INJECT=0` |",
         "| **A1** inject | pymupdf 切分 | **按页注入** MinerU 表格/公式 | **线上默认** |",
         "| **A2** mineru | MinerU 切分 | 原生在块里 | `PAPERPILOT_USE_MINERU=1` |", "",
         "## 1. 主指标", "",
         "| 口径 | A0 纯 pymupdf | A1 注入（默认） | A2 全链 MinerU |", "|---|---|---|---|"]

    all_q = list(items)
    rows = []
    for label, qids in (("全 30 题", all_q), (f"表格子集（{len(table_qids)} 题）", table_qids)):
        cells = []
        for a, _ in ARMS:
            p, n, m = stat(arms[a], qids)
            cells.append(f"{p}/{n}（{m:.2f}）")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    L += rows + ["", "## 2. 逐题矩阵（表格子集优先）", "",
                 "| qid | 论文 | src（金标准出处） | A0 | A1 | A2 |", "|---|---|---|---|---|---|"]
    order = table_qids + [q for q in all_q if q not in table_qids]
    for q in order:
        x = items[q]
        sc = []
        for a, _ in ARMS:
            r = arms[a].get(q)
            sc.append("—" if not r else str(r["score"]))
        mark = "**" if q in table_qids else ""
        L.append(f"| {mark}{q}{mark} | {x['pid']} | {x.get('src','')} | " + " | ".join(sc) + " |")

    L += ["", "## 3. 结论", ""]
    p0, _, m0 = stat(arms["plain"], all_q)
    p1, _, m1 = stat(arms["inject"], all_q)
    p2, _, m2 = stat(arms["mineru"], all_q)
    L.append(f"- 全 30 题：A0 {p0}/30（{m0:.2f}）→ A1 {p1}/30（{m1:.2f}）→ A2 {p2}/30（{m2:.2f}）")
    t0, tn, tm0 = stat(arms["plain"], table_qids)
    t1, _, tm1 = stat(arms["inject"], table_qids)
    t2, _, tm2 = stat(arms["mineru"], table_qids)
    L.append(f"- 表格子集（{tn} 题）：A0 {t0}/{tn}（{tm0:.2f}）→ A1 {t1}/{tn}（{tm1:.2f}）→ A2 {t2}/{tn}（{tm2:.2f}）")
    L += ["", "> 单次跑，n 小（表格子集 12 题）→ 翻动 2~3 题即 ±2 分，结论只认**跨臂一致的量级差**，",
          "> 单题翻动须复跑确认（参见 `mh-27843-q2` 曾 3/4/3 的历史）。"]

    out = Path("qa/recall/MINERU_AB_20260911.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
