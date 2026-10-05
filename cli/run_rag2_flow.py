"""**多答案开放域检索 · 端到端编排入口**
（方向问题 → rag1 检索 → arxiv-id → 论文取料下载 → 多答案开放问题 → rag2 → 答案）

## 用

    # ★ 目标架构：用户问方向 → rag1 检索 → 拿 id → 取料下载 → 问多答案问题 → 答案
    uv run python cli/run_rag2_flow.py `
        --direction "如何用检索增强生成做知识密集问答" --n 10 `
        --question "这些论文里，哪些报告了消融实验？"

    # CLI 离线档：直接用现成的簇（跳过 rag1 与取料）—— 语料现成、跑得快
    # ★ 前端（独立页 `/ma`）没有这个档位。
    uv run python cli/run_rag2_flow.py --cluster 1 --question "…"

    # 最快：直接用现成切块（纯验证 ④）
    uv run python cli/run_rag2_flow.py --chunks <切块.parquet> --question "…"

    # 看现成簇有哪些
    uv run python cli/run_rag2_flow.py --list-clusters

⚠️ 要真调判官（LLM）→ 需 key（`PAPERPILOT_LLM_*` / `PAPERPILOT_JUDGE_*`）。
★ `--n` 的含义：**语料规模**（rag1 交付几篇）—— 注意 rag1 本地**只精排前 10**
  （`corpus_search.N_OUT = 10`），见 `docs/RAG2_CORPUS_WIRING.md`。
  ★ 这是 **CLI 离线参数**；前端独立页 `/ma` 固定 ≤10 篇、**没有 N 选择器**。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="多答案开放域检索 · 端到端编排（方向 → rag1 → 取料 → rag2 → 答案）")
    src = ap.add_argument_group("语料来源（三选一）")
    src.add_argument("--direction", default="",
                     help="★ 目标架构：方向问题（走 rag1 → 取料 → 切块）")
    src.add_argument("--cluster", type=int, default=0,
                     help="「展示」档：用现成的簇（1/2/3，各 50 篇，跳过 rag1 与取料）")
    src.add_argument("--chunks", default="", help="最快：直接用现成切块 parquet")
    ap.add_argument("--n", type=int, default=10,
                    help="★ 语料规模 = rag1 交付几篇（本地只精排前 10）")
    ap.add_argument("--question", default="", help="多答案开放问题")
    ap.add_argument("--no-fetch", action="store_true",
                    help="目标架构下只检索不下载（按 arxiv 号找既有 PDF）")
    ap.add_argument("--no-llm", action="store_true", help="rag1 跳过 LLM 精排")
    ap.add_argument("--extract", action="store_true",
                    help="额外抽取逐篇内容（复合题：答案从「哪几篇」升级到「各自是什么」）")
    ap.add_argument("--b", type=int, default=12, help="每篇给判官的候选块数（生产默认 12）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="", help="结果 JSON 落盘")
    ap.add_argument("--list-clusters", action="store_true", help="列出可用的现成簇后退出")
    args = ap.parse_args()

    from paperpilot.multianswer import pipeline as P  # noqa: PLC0415
    from paperpilot.tools import llm  # noqa: PLC0415

    if args.list_clusters:
        for c in P.list_clusters():
            print(f"  簇{c['ci']}  {c['name']}  {'✓ 在' if c['exists'] else '✗ 缺'}"
                  f"  {c['path']}")
        return 0

    if not args.question:
        raise SystemExit("必须给 `--question`（多答案开放问题）")
    if not (args.direction or args.cluster or args.chunks):
        raise SystemExit("必须给一种语料来源：`--direction` / `--cluster` / `--chunks`")

    llm._load_dotenv(str(ROOT))          # ★ 直接跑脚本不会自动加载 .env

    tag = f"n{args.n}" if args.direction else (f"c{args.cluster}" or "flow")
    res = P.run_flow(args.question, direction=(args.direction or None),
                     n=args.n, cluster=(args.cluster or None),
                     chunks_parquet=(args.chunks or None), tag=tag,
                     use_llm=not args.no_llm, fetch=not args.no_fetch,
                     extract=(True if args.extract else None),
                     b=args.b, workers=args.workers)

    print("\n" + "=" * 96)
    print(f"【语料】{res['n']} 篇 ｜ 切块 {res['chunks_parquet']}")
    if args.direction:
        print(f"       方向 = {res['direction']}")
    print(f"【命中】判 yes {res['n_yes']} 篇：{res['yes']}")
    print("=" * 96)
    print("【答案】\n" + res["answer_text"])

    if args.out:
        p = P.save_run(res, args.out)
        print(f"\n→ 结果已写 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
