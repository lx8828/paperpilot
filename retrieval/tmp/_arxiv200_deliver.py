"""**线上口径下的交付 N 曲线**——用 `LITSEARCH_ARXIV200_raw.csv`（零 LLM 重算）。

## 为什么必须做这一版（而不是拿 LitSearch 的结论）
线上与 LitSearch 是**两个不同世界**：

| | LitSearch（研究区） | arXiv200（线上口径） |
|---|---|---|
| 语料 | 63,269 篇（NLP，冻结 2023 底） | **540,052 篇**（arXiv cs，2026 最新） |
| 与对方重叠 | **13 篇（0.02%）** | 同上 |
| 提问语言 | **英文** | **中文**（`lang=zh`，跨语言检索） |
| 真值 | 人工审核 qrels | LLM 从论文合成（gold = 该篇） |

**已观测到的结论翻转**（同一套流水线）：

| 阶段 | LitSearch 597 | arXiv200 |
|---|---|---|
| hyb0.5 R@1 | 0.3548 | **0.2150** |
| + CE | 0.4154（+6.1pt） | **0.8500（+63.5pt）** |
| + LLM listwise | 0.6092（**+19.4pt**） | 0.8750（**+2.5pt**） |

→ LitSearch 说"listwise 是最大一跃"、线上说"**CE 才是主力**"。
**所以 LitSearch 上的池深/交付结论不能直接给线上定水位。**

## 本脚本做什么
CSV 里有**逐题位次**：`g_rank_retr`（hyb0.5 本语料位次）、`ce100/200/400`、`llm100/200/400`
（数字 = gold 在该臂下的位次；`进池` = 是否进了候选池）。
→ 直接算 `hit@N`（单 gold 集上 = `coverage@N`），并按 `抽象层级`/`语料`/`batch` 分层。

用法：uv run python retrieval/tmp/_arxiv200_deliver.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "retrieval" / "results" / "LITSEARCH_ARXIV200_raw.csv"
ARMS = ("g_rank_retr", "ce100", "ce200", "ce400", "llm100", "llm200", "llm400")
NS = (1, 5, 10, 20, 50, 100, 200, 400)


def curve(d: pd.DataFrame, col: str, n: int | None = None) -> dict[int, float]:
    r = d[col]
    if n is not None:
        r = r[:n]
    return {N: float((r <= N).mean()) for N in NS}


def show(d: pd.DataFrame, title: str) -> None:
    print(f"\n########## {title}（{len(d)} 题）")
    print(f"  {'臂':<16}" + "".join(f"{'N=' + str(N):>9}" for N in NS if N <= 100))
    for col in ARMS:
        c = curve(d, col)
        print(f"  {col:<16}" + "".join(f"{c[N]:>9.1%}" for N in NS if N <= 100))
    print(f"  {'进池（参考）':<14}{float(d['进池'].mean()):>9.1%}")


def main() -> int:
    d = pd.read_csv(CSV)
    print(f"\n{'=' * 112}\n【线上口径 · 交付 N 曲线】{CSV.name} ｜ {len(d)} 题 ｜ "
          f"语言 {sorted(d['lang'].unique())} ｜ 语料 {sorted(d['语料'].unique())}\n{'=' * 112}")
    print("⚠️ 单 gold 集 → `hit@N` 即 `coverage@N`；`ce*`/`llm*` 的数字后缀 = **候选池深**。")
    print("⚠️ 合成题（LLM 从论文写 query）有**乐观偏置**，绝对水位需人工抽查标定。")

    show(d, "总体（均匀主样本 + 旧语料补抽）")
    for name, sub in (("新语料", d[d["语料"] == "新"]), ("旧语料", d[d["语料"] == "旧"]),
                      ("具体型", d[d["抽象层级"] == "具体"]),
                      ("上位型", d[d["抽象层级"] == "上位"])):
        if len(sub) >= 10:
            show(sub, f"分层：{name}")

    print(f"\n{'#' * 112}\n关键读数")
    tot = curve(d, "g_rank_retr")
    for col in ("ce100", "ce200", "ce400", "llm100", "llm200", "llm400"):
        c = curve(d, col)
        print(f"  {col:<14} 交付5 {c[5]:>7.1%}（相对 hyb 交付5 {tot[5]:.1%}: "
              f"{100 * (c[5] - tot[5]):+.1f}pt）｜ 交付20 {c[20]:>7.1%}｜ 交付50 {c[50]:>7.1%}")
    b = d[d["抽象层级"] == "上位"]
    if len(b) >= 10:
        print("\n  **上位型（136 题级，最接近『问一个研究方向』）**：")
        for col in ("g_rank_retr", "ce200", "ce400", "llm200", "llm400"):
            c = curve(b, col)
            print(f"    {col:<14} 交付5 {c[5]:>7.1%}｜ 交付20 {c[20]:>7.1%}｜ 交付50 {c[50]:>7.1%}")
    print("\n读法：比较 `ce100/200/400` 与 `llm100/200/400` 的**列内差**= 池深收益；"
          "\n      比较 `ce200` 与 `llm200` = listwise 在**线上口径**下的增量（LitSearch 上是 +19.4pt）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
