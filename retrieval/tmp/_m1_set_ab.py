"""**实验 2｜生产 5 篇语料（S/C ≈ 0.37）上：`fullctx` vs `set`（检索+逐篇判定）**

## 要回答的问题
「判定层」的价值是否取决于语料规模？
R2 实测（20 篇 / S/C≈2）：逐篇判定 F1 **0.817** vs 整档直读 **0.773**。
但生产是 **5 篇 / 48k token / S/C≈0.37** —— 直读的"主场"。
**若 `set` 在 5 篇上也赢 → 判定层与 S/C 无关，生产管线该接；若输 → 它确实只属于大语料线。**

## 协议（与 `run_fullctx_qa.py` 严格对齐，否则不可比）
· 语料 = `group_corpus(group)`（同 `papers()`）
· 题目 = `load_cross_questions(corpus)` 里 `kind == "M1"`，**与 `fullctx_m1all_group*.json` 同题**
· 判分 = `run_multi_qa.score_answer`（唯一口径）→ `ok` / `ok_strict`
· 对照臂 = `fullctx_m1all_group{1..5}.json`（50 题，`ok_strict` **30/50 = 60%**）

⚠️ **两版渲染**（关键，否则会冤枉判定层）：
`set_judge.render` 只给「篇名 + ≤80 字理由」，而 M1 判分锚点多为**数字/专名**，
`fullctx` 是长篇详答（会照抄数字）→ 直接比会**系统性低估 set**。
故同时测：
  · `shipped` = 原样 `render()`（生产现状）
  · `rich`    = `render()` + **逐篇证据片段原文**（判分文本里就有数字了）
两版**判定完全相同**（同一批 `set_judge.run` 结果）→ 差异 100% 来自渲染。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_m1_set_ab.py --limit 3 --group group1   # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_m1_set_ab.py                            # 全 5 组
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
from _qa_groups import papers  # noqa: E402
from run_fullctx_qa import group_corpus  # noqa: E402
from run_multi_qa import load_cross_questions, score_answer  # noqa: E402

OUT = ROOT / "qa" / "multi" / "_runs"


def rich_answer(question: str, res: dict) -> str:
    """`rich` 渲染：在 `render()` 后附上每篇的证据原文（让判分文本含数字/专名）。"""
    from paperpilot.components import set_judge
    lines = [set_judge.render(question, res)]
    lines.append("\n逐篇证据原文：")
    for p in res["papers"]:
        lines.append(f"· {p['pdf']}（第 {p['page']} 页 {p['section']}）：{p['snippet']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="", help="空=全 5 组")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--b", type=int, default=0, help="0=用 set_judge 默认（12）")
    ap.add_argument("--tag", default="set_m1all")
    args = ap.parse_args()

    from paperpilot.tools import llm
    from paperpilot.agents.embedder import MultiChunkIndex
    from paperpilot.components import set_judge
    from paperpilot.workflow import _ensure_env

    _ensure_env()
    llm._load_dotenv(str(ROOT))
    groups = [args.group] if args.group else [f"group{i}" for i in range(1, 6)]

    all_recs: list[dict] = []
    for g in groups:
        corpus = group_corpus(g)
        qs = [q for q in load_cross_questions(corpus)
              if q["group"] == "B" and str(q.get("kind") or "") == "M1"]
        if args.limit:
            qs = qs[: args.limit]
        print("=" * 108)
        print(f"[{g}] 语料 {len(corpus)} 篇 ｜ M1 题 {len(qs)} 道 ｜ b={args.b or set_judge.DEFAULT_B}")
        print(f"   · {', '.join(corpus)}", flush=True)

        t_idx = time.time()
        idx = MultiChunkIndex(corpus)
        n_chunks = len(idx._doc_chunks())                       # noqa: SLF001
        print(f"  索引就绪：{n_chunks} 块（{time.time() - t_idx:.0f}s）", flush=True)

        recs = []
        for q in qs:
            llm.reset_usage()
            t0 = time.time()
            try:
                res = set_judge.run(str(q.get("question") or ""), idx,
                                    b=(args.b or None))
                ans_s = set_judge.render(str(q.get("question") or ""), res)
                ans_r = rich_answer(str(q.get("question") or ""), res)
            except Exception as e:  # noqa: BLE001
                res = {"papers": [], "n_papers": len(corpus), "n_yes": 0, "unclear": [], "b": 0}
                ans_s = ans_r = f"ERR {type(e).__name__}: {e}"
            sec = time.time() - t0
            cites = [{"pdf": p["pdf"], "chunk_id": p["chunk_id"], "page": p["page"],
                      "section": p["section"], "evidence": p["snippet"]} for p in res["papers"]]
            okS, missS, stS = score_answer(q, ans_s, cites)
            okR, missR, stR = score_answer(q, ans_r, cites)
            us = llm.usage_stats()
            recs.append({**q, "reader": "set", "answer": ans_s, "answer_rich": ans_r,
                         "papers": [p["pdf"] for p in res["papers"]], "n_yes": res["n_yes"],
                         "unclear": res["unclear"], "b": res["b"],
                         "ok": okS, "ok_strict": stS, "miss": missS,
                         "ok_rich": okR, "ok_strict_rich": stR, "miss_rich": missR,
                         "n_papers": res["n_papers"], "seconds": round(sec, 1),
                         "llm_calls": us["calls"], "prompt_tokens": us["prompt_tokens"],
                         "completion_tokens": us["completion_tokens"]})
            r = recs[-1]
            print(f"  {'✅' if stS else ('🟡' if stR else '·')} {str(q.get('qid')):<12}"
                  f" 判yes={r['n_yes']}/5 unclear={len(r['unclear'])} {sec:>4.1f}s"
                  f" 原样{'✅' if okS else '·'}/{'✅' if stS else '·'}"
                  f" rich{'✅' if okR else '·'}/{'✅' if stR else '·'}", flush=True)
        all_recs += recs
        (OUT / f"{args.tag}_{g}.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 汇总 ──
    print(f"\n{'=' * 108}\n【实验 2｜生产 5 篇·M1】{len(all_recs)} 题")
    n = len(all_recs)
    okS = sum(1 for r in all_recs if r["ok"])
    stS = sum(1 for r in all_recs if r["ok_strict"])
    okR = sum(1 for r in all_recs if r["ok_rich"])
    stR = sum(1 for r in all_recs if r["ok_strict_rich"])
    def row(nm: str, ok: int, st: int) -> str:
        return (f"  {nm:<34} | ok {ok:>2}/{n} = {ok / n:>5.1%}"
                f" | ok_strict {st:>2}/{n} = **{st / n:>5.1%}**")

    print(f"  {'臂（同题：50 道 M1）':<34} | {'ok（答案+引用）':<16} | {'ok_strict（仅答案）'}")
    print(f"  {'fullctx 全上下文直读（存档）':<34} | ok 33/50 = 66.0% | ok_strict 30/50 = **60.0%**")
    print(row("set 检索+逐篇判定（原样渲染）", okS, stS))
    print(row("set 同判定·带证据原文渲染", okR, stR))
    print(f"\n  交付篇数 均 {sum(r['n_yes'] for r in all_recs) / n:.2f}/5"
          f" ｜ 有 unclear 的题 {sum(1 for r in all_recs if r['unclear'])}/{n}")
    print(f"  用量：均 {sum(int(r['prompt_tokens']) for r in all_recs) / n:,.0f} prompt tok"
          f" ｜ 均 {sum(int(r['llm_calls']) for r in all_recs) / n:.1f} 次调用"
          f" ｜ 均 {sum(float(r['seconds']) for r in all_recs) / n:.1f} 秒")
    print(f"\n已写 {OUT / (args.tag + '_group*.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
