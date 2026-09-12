"""三列对比的显著性检验：配对 McNemar 精确检验（同一批题、同一裁判）。

为什么要配对检验：三列跑的是**同一批题**，pass 率差异来自"同一道题上谁过谁不过"的翻转，
独立双样本检验会低估配对信息。McNemar 只看不一致对（仅 A 过 / 仅 B 过），
双尾 p 由二项分布精确算出（n 小，不做正态近似）。

用法：uv run python qa/compare/_signif.py [时间戳]
"""
from __future__ import annotations

import collections
import io
import json
import math
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PASS = 4
TS = sys.argv[1] if len(sys.argv) > 1 else "20260911_040708"
POOLS = [("常规 64 题（20 篇）", "run_{}_{}.json"), ("深水 30 题", "deep_{}_{}.json")]
COLS = ["B0", "B1", "B2"]
LABEL = {"B0": "直接 LLM", "B1": "朴素 RAG", "B2": "PaperPilot"}


def load(p: str) -> dict[str, dict]:
    return {r["qid"]: r for r in json.loads(Path(p).read_text(encoding="utf-8"))}


def is_pass(r: dict) -> bool:
    return r.get("score", 0) >= PASS


def mcnemar(b: int, c: int) -> float:
    """双尾精确 p（H0: 翻转对称）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    for name, tmpl in POOLS:
        d = {c: load(f"qa/compare/{tmpl.format(TS, c)}") for c in COLS}
        qids = sorted(set(d["B0"]) & set(d["B1"]) & set(d["B2"]))
        print(f"\n===== {name}  (n={len(qids)}，三列同题) =====")
        for c in COLS:
            p = sum(is_pass(d[c][q]) for q in qids)
            print(f"  {c} {LABEL[c]:<10} pass {p}/{len(qids)} = {p / len(qids):.1%}")
        for a, b in [("B2", "B0"), ("B2", "B1"), ("B0", "B1")]:
            only_a = sum(1 for q in qids if is_pass(d[a][q]) and not is_pass(d[b][q]))
            only_b = sum(1 for q in qids if is_pass(d[b][q]) and not is_pass(d[a][q]))
            pv = mcnemar(only_a, only_b)
            verdict = "**显著**" if pv < 0.05 else "不显著"
            print(f"  {a} vs {b}: 仅{a}过 {only_a}、仅{b}过 {only_b}、净 Δ={only_a - only_b:+d}"
                  f"  → McNemar 双尾 p={pv:.3f} {verdict}")
        # 分题型（若记录带 qtype）
        if any("qtype" in r for r in d["B0"].values()):
            print("  分题型 pass：")
            for qt in sorted({r.get("qtype", "") for r in d["B0"].values()}):
                cells = []
                for c in COLS:
                    s = [d[c][q] for q in qids if d[c][q].get("qtype") == qt]
                    cells.append(f"{sum(is_pass(x) for x in s)}/{len(s)}")
                print(f"    {qt:<14} B0 {cells[0]} | B1 {cells[1]} | B2 {cells[2]}")
            # 唯一"稳赢"的子桶：单独检验，避免用方向性叙述代替显著性
            sub = [q for q in qids if d["B0"][q].get("qtype") == "unanswerable"]
            if sub:
                print(f"  仅 unanswerable 子桶（n={len(sub)}）的配对检验：")
                for a, b in [("B2", "B0"), ("B2", "B1")]:
                    oa = sum(1 for q in sub if is_pass(d[a][q]) and not is_pass(d[b][q]))
                    ob = sum(1 for q in sub if is_pass(d[b][q]) and not is_pass(d[a][q]))
                    pv = mcnemar(oa, ob)
                    print(f"    {a} vs {b}: 仅{a}过 {oa}、仅{b}过 {ob}、净 Δ={oa - ob:+d}"
                          f"  → p={pv:.3f} {'**显著**' if pv < 0.05 else '不显著（样本太小）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
