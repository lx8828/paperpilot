"""**多答案开放域问答**：一组论文 + 一个问题 → **回答**
（先逐篇判「哪几篇做了 X」，再把判定结果汇成一段回答；逐篇证据作为「依据」保留）。

这是**四个工具里的第 ④ 个**（多答案 / 集合问答），前三个是：
  ① `run_fetch.py` 取料 ｜ ② `run_search.py` 检索 ｜ ③ `run_pipeline.py` 端到端（直读+报告）

与 web 的 `/api/ask`（`PAPERPILOT_QA_READER=set`）走**同一份**生产实现
（`components.set_judge.run` + `nodes.set_answer.answer_set` 的判定部分），
只是把入口搬成命令行，便于脚本化与对照。

用法（在仓库根目录）：
    # 指定若干篇论文 + 一个问题
    uv run python cli/run_set.py --pdfs 2408.09017.pdf 2310.03184.pdf --question "哪些篇做了消融？"
    # 一整目录的论文（取 *.pdf）
    uv run python cli/run_set.py --dir assets/papers --question "哪些篇报告了显著性检验？"
    # 复合题（判定之外再要"各自具体内容"）
    uv run python cli/run_set.py --dir assets/papers --question "..." --extract
    # 候选篇上限（默认 0 = 判全部候选）
    uv run python cli/run_set.py --dir assets/papers --question "..." --top-papers 30

⏱ 首次调用要加载 embedding 模型 + 逐篇建向量缓存（`.cvec.npy`），之后秒级复用。
⚠️ 每篇判官是 1 次 LLM 调用（单判官），成本 = 候选篇数 × 单价（实测 ≈ ¥0.002/次）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.agents.embedder import MultiChunkIndex  # noqa: E402
from paperpilot.components import set_judge  # noqa: E402


def _collect_pdfs(args: argparse.Namespace) -> list[str]:
    """`--pdfs` 与 `--dir` 二选一（或同时：dir 里的全收，再追加 pdfs 去重）。"""
    out: list[str] = []
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"❌ 目录不存在：{d}")
            raise SystemExit(2)
        out += [str(p) for p in sorted(d.glob("*.pdf"))]
    if args.pdfs:
        for p in args.pdfs:
            for piece in p.split(","):
                piece = piece.strip()
                if piece:
                    out.append(piece)
    # 去重保序
    seen: set[str] = set()
    uniq = [p for p in out if not (p in seen or seen.add(p))]
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="多答案开放域问答：哪几篇做了 X")
    ap.add_argument("--pdfs", nargs="*", default=[], help="论文（路径或文件名，可逗号分隔）")
    ap.add_argument("--dir", default="", help="取该目录下全部 *.pdf")
    ap.add_argument("--question", required=True, help="要判断的论断 / 问题")
    ap.add_argument("--extract", action="store_true", help="复合题：判定之外再要逐篇具体内容")
    ap.add_argument("--top-papers", type=int, default=0,
                    help="候选篇上限（默认 0 = 判全部有命中的篇）")
    ap.add_argument("--b", type=int, default=0, help="每篇给判官的块数（默认 12）")
    ap.add_argument("--workers", type=int, default=0, help="判官并发（默认 6）")
    args = ap.parse_args()

    pdfs = _collect_pdfs(args)
    if not pdfs:
        print("❌ 没给论文：用 --pdfs 或 --dir 指定语料")
        return 2

    import os
    os.environ.setdefault("PAPERPILOT_QA_READER", "set")
    if args.extract:
        os.environ["PAPERPILOT_SET_EXTRACT"] = "1"

    print(f"[set] 语料 {len(pdfs)} 篇 ｜ 问题：{args.question[:120]}")
    idx = MultiChunkIndex(pdfs)
    res = set_judge.run(
        args.question, idx, claim=args.question,
        extract=args.extract,
        top_papers=(args.top_papers or None),
        b=(args.b or None), workers=(args.workers or None),
    )
    print("=" * 88)
    print(set_judge.render(args.question, res))
    print("=" * 88)
    print(f"[set] 候选 {res['n_papers']} 篇 ｜ 判为「做了」 {res['n_yes']} 篇 ｜ "
          f"未决 {len(res['unclear'])} 篇 ｜ b={res['b']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
