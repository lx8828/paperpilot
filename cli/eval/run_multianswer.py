"""**多答案开放域检索 · 独立入口**（不在"方向 → 精读"那条链路上）。

用户口径：*"选择 N，然后问多答案开放域问题，然后跑出回答，完全分开的路线。"*
★ 这里的「选 N」是**离线跑批参数**（`--corpus-n`），**不是前端交互** ——
  前端（独立页 `/ma`）语料**固定 ≤10 篇**、**没有 N 选择器**。

## 它和 `run_group_qa.py` / `run_retrieval_eval.py` 的区别

| | 输入 | 输出 |
|---|---|---|
| 方向流程（现有） | 一个研究方向 | 一批论文的**深度报告**（单篇精读） |
| **本入口** | **一个问题 + 语料（N 篇）** | ★ **回答**（逐篇判定 → LLM 汇总）＋ 逐篇依据 |

## 用

    # 一次多答案（选 N=30）
    ./.venv/Scripts/python.exe cli/eval/run_multianswer.py \\
        --question "哪些论文用了对比学习做跨语言摘要" \\
        --chunks retrieval/data/r2dev/prodchunk50/mineru/c0.parquet \\
        --n 30

    # ★ 一次跑批扫多个 N（判全部候选 → 离线切前缀，不重跑判官）
    ./.venv/Scripts/python.exe cli/eval/run_multianswer.py \\
        --question "..." --chunks .../c0.parquet --sweep 10,20,30,50

⚠️ 要真调判官（LLM）→ 需要 key（`PAPERPILOT_LLM_*` / `PAPERPILOT_JUDGE_*`）。
   只探活可跑 `--help`（不加载重模块）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description="多答案开放域检索（独立入口）")
    ap.add_argument("--question", required=True, help="多答案开放域问题")
    ap.add_argument("--chunks", required=True,
                    help="已切块的 parquet（需含 `docid` / `text` 两列）")
    # ⚠️ 这两个**必须分开**（原先把它们混成一个 `--n`，跑出来不是"在 N 篇里找"）：
    #    `--corpus-n` 是**语料规模**（= 用户说的"选 N"，从多少篇里找）；
    #    `--topk`    是**候选篇上限**（判几篇）—— 定稿口径 30。
    ap.add_argument("--corpus-n", type=int, default=50,
                    help="语料规模：从多少篇论文里找（用户说的「选 N」）")
    ap.add_argument("--topk", type=int, default=0,
                    help="候选篇上限：只判定检索前 K 篇（0 = 判全部候选，默认）")
    ap.add_argument("--sweep", default="",
                    help="逗号分隔的候选上限列表 → 一次跑批**离线扫**（要求 --topk 0）")
    ap.add_argument("--b", type=int, default=12, help="每篇给判官的候选块数（生产默认 12）")
    ap.add_argument("--workers", type=int, default=8, help="判官并发")
    ap.add_argument("--extract", action="store_true", help="额外抽取逐篇内容（复合题需要）")
    ap.add_argument("--out", default="", help="结果 JSON 落盘路径（可回溯）")
    ap.add_argument("--out-md", default="",
                    help="**demo 存档**（Markdown）：问题 + 口径 + 答案 + 扫 N 表。"
                         "以后展示直接读它，不必重跑（省时间省钱）")
    args = ap.parse_args()

    from paperpilot.multianswer import CorpusIndex, answer, render, sweep_n

    idx = CorpusIndex.from_parquet(args.chunks, n=args.corpus_n)

    print("=" * 96)
    print(f"【多答案开放域检索】问题：{args.question}")
    print(f"  语料：{Path(args.chunks).name} ｜ **语料规模 N={len(idx.pdfs)}** 篇"
          f" ｜ 候选上限 K={args.topk or '不限（判全部）'} ｜ 每篇块数 b={args.b}")
    print("=" * 96)

    res = answer(args.question, idx, n=(args.topk or None), b=args.b,
                 workers=args.workers, extract=args.extract)

    print(f"\n候选 {res['n_papers']} 篇 ｜ 判 yes {res['n_yes']} 篇"
          f"（检索 {res.get('retr')}｜双判官 {res.get('dual')}｜窗口 {res.get('win_chars')} 字符）\n")
    for i, p in enumerate(res["papers"], 1):
        print(f"  [{i}] {p['pdf']}   (score {p['score']:.4f})")
        print(f"      理由：{p['why']}")
        if p.get("extract"):
            print(f"      内容：{p['extract'][:160]}")
        print(f"      证据：{str(p['snippet'])[:150]}…\n")

    if not res["papers"]:
        print("  （没有论文被判为做了这件事）")

    rows: list[dict] = []
    if args.sweep:
        ns = [int(x) for x in args.sweep.split(",") if x.strip()]
        print("-" * 96)
        print("★ 同一次跑批离线扫候选上限（不重跑判官）—— 前提：本次判了**全部**候选")
        try:
            rows = sweep_n(res, ns)
            for r in rows:
                print(f"  K={r['n']:>4}  候选 {r['n_candidates']:>4} 篇"
                      f"  →  判 yes {r['n_yes']:>3} 篇")
        except ValueError as e:
            print(f"  ✗ 扫不了：{e}")

    print("-" * 96)
    answer_text = render(args.question, res)
    print("【答案】\n" + answer_text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(res, question=args.question, chunks=str(args.chunks),
                       index_papers=len(idx.pdfs), answer_text=answer_text,
                       sweep=rows)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ 结果已写 {out}（{len(json.dumps(payload, ensure_ascii=False))} 字符）")

    if args.out_md:
        from datetime import datetime
        md = [
            "# 多答案开放域检索 · demo 存档", "",
            "> ★ **这是存档，不是评测报告** —— 目的：以后 demo 展示直接读它，"
            "不必重跑（省时间省钱）。", "",
            "| | |", "|---|---|",
            f"| **问题** | **{args.question}** |",
            f"| 语料 | `{Path(args.chunks).name}` ｜ **规模 N = {len(idx.pdfs)} 篇** |",
            f"| 候选上限 K | {res.get('top_papers') or '不限（判全部候选）'} |",
            f"| 命中 | **判 yes {res['n_yes']} 篇** / 候选 {res['n_papers']} 篇 |",
            f"| 检索 | `{res.get('retr')}` ｜ 双判官 {res.get('dual')}"
            f" ｜ 窗口 {res.get('win_chars')} 字符 ｜ 每篇块数 b={res.get('b')} |",
            f"| 生成于 | {datetime.now().strftime('%Y-%m-%d %H:%M')} |",
            "",
            "## 答案：哪几篇论文符合", "",
            "| # | 论文 | 检索分 | 判官理由 | 证据片段 |",
            "|---:|---|---:|---|---|",
        ]
        for i, p in enumerate(res["papers"], 1):
            snip = str(p.get("snippet") or "").replace("|", "\\|").replace("\n", " ")[:110]
            why = str(p.get("why") or "").replace("|", "\\|")
            md.append(f"| {i} | `{p['pdf']}` | {p['score']:.4f} | {why} | {snip}… |")
        if not res["papers"]:
            md.append("| — | （没有论文被判为做了这件事） | | | |")

        if rows:
            md += ["", "## 扫候选上限（★ 一次跑批，判官**没重跑**）", "",
                   "| 候选上限 K | 判断过的篇数 | 判 yes |", "|---:|---:|---:|"]
            for r in rows:
                md.append(f"| {r['n']} | {r['n_candidates']} | **{r['n_yes']}** |")
            md += ["", "★ 判果只依赖每篇自己的片段，与候选多少无关 → "
                       "改 K 只是**切前缀**，所以一张表能给出多个 K 的数。"]

        md += ["", "## 原始答案文本", "", "```", answer_text.strip(), "```", ""]
        p = Path(args.out_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(md), encoding="utf-8")
        print(f"→ demo 存档已写 {p}（{p.stat().st_size} 字节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
