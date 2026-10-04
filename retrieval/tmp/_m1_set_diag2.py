"""**实验 2b｜4 臂决定性测试**：`set` 的 12% 到底缺什么？

## 诊断假设
M1 题形 = **集合筛选 + 内容枚举** 的复合题：
  「这五篇里，哪些篇做了组件消融？**各自消融掉的是什么**？」
`set` 只答前半（哪几篇 ✓），不答后半（各自是什么 ✗）→ 锚点（数字/专名）全缺 → 12%。

## 4 臂（**判定完全共用**，只换"内容载体"→ 差异 100% 来自交付形态）
| 臂 | 答案文本 | 判分文本（`score_answer` = answer + cites[].evidence） |
|---|---|---|
| S `shipped` | `render()`：篇名 + ≤80 字理由 | + 各 yes 篇的 **300 字** snippet |
| R `rich300` | 同上（+ `render` 头） | 同上 |
| **Y `route_yes`** | `render()` + **yes 篇全部 b 块原文** | 同左（全文） |
| **A `route_all`** | `render()` + **全部 5 篇全部 b 块原文** | 同左（全文） |

对照臂：`fullctx`（全上下文直读·存档）`ok_strict` **30/50 = 60%**。

读法：
· **Y ≈ A ≈ 60%+** → 判定是对的，缺的只是内容载体 → `set` 该当**路由/约束**，不该当答案。
· **Y 明显 < A** → 判定漏篇真的伤了内容。
· **Y < fullctx** → 检索压缩（每篇只留 b 块）丢了内容 → 5 篇规模下**不该压缩**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
from run_fullctx_qa import group_corpus  # noqa: E402
from run_multi_qa import load_cross_questions, score_answer  # noqa: E402

OUT = ROOT / "qa" / "multi" / "_runs"


def blocks_text(hits: list[dict]) -> str:
    return "\n".join(f"[片段{i + 1}] {str(h.get('text') or '')}" for i, h in enumerate(hits))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--b", type=int, default=12)
    ap.add_argument("--tag", default="setdiag2")
    args = ap.parse_args()

    from paperpilot.tools import llm
    from paperpilot.agents.embedder import MultiChunkIndex
    from paperpilot.components import set_judge
    from paperpilot.workflow import _ensure_env

    _ensure_env()
    llm._load_dotenv(str(ROOT))
    groups = [args.group] if args.group else [f"group{i}" for i in range(1, 6)]

    recs: list[dict] = []
    for g in groups:
        corpus = group_corpus(g)
        qs = [q for q in load_cross_questions(corpus)
              if q["group"] == "B" and str(q.get("kind") or "") == "M1"]
        if args.limit:
            qs = qs[: args.limit]
        idx = MultiChunkIndex(corpus)
        print(f"\n[{g}] 语料 {len(corpus)} 篇 {len(idx._doc_chunks())} 块 ｜ M1 {len(qs)} 题",  # noqa: SLF001
              flush=True)
        for q in qs:
            qq = str(q.get("question") or "")
            hits = set_judge.per_paper_hits(idx, qq, b=args.b)
            res = set_judge.run(qq, idx, b=args.b)
            yes = [p["pdf"] for p in res["papers"]]
            base = set_judge.render(qq, res)

            def cite(ev_map: dict[str, list[dict]], pdf: str, snip: bool = False) -> list[dict]:
                hs = ev_map.get(pdf, [])
                if not hs:
                    return []
                return [{"pdf": pdf, "chunk_id": str(hs[0].get("chunk_id") or ""),
                         "page": int(hs[0].get("page") or 0),
                         "section": str(hs[0].get("section") or ""),
                         "evidence": (str(hs[0].get("text") or "")[:300] if snip
                                      else blocks_text(hs))}]

            arms = {}
            # S shipped / R rich300：判分文本只带 300 字 snippet（= 现行产品行为）
            c_ship = [x for p in yes for x in cite(hits, p, snip=True)]
            arms["S_shipped"] = score_answer(q, base, c_ship)
            # Y route_yes：yes 篇的全部 b 块原文（判定当路由 + 内容载体）
            body_y = base + "\n\n各篇证据原文：\n" + "\n".join(
                f"【{p}】\n{blocks_text(hits.get(p, []))}" for p in yes)
            c_y = [x for p in yes for x in cite(hits, p)]
            arms["Y_route_yes"] = score_answer(q, body_y, c_y)
            # A route_all：全部篇的 b 块原文（不靠判定路由）
            body_a = base + "\n\n全部语料证据原文：\n" + "\n".join(
                f"【{p}】\n{blocks_text(h)}" for p, h in hits.items())
            c_a = [x for p in hits for x in cite(hits, p)]
            arms["A_route_all"] = score_answer(q, body_a, c_a)

            rec = {**q, "yes": yes, "n_yes": len(yes),
                   **{f"{k}_ok": v[0] for k, v in arms.items()},
                   **{f"{k}_strict": v[2] for k, v in arms.items()},
                   "S_miss": arms["S_shipped"][1], "Y_miss": arms["Y_route_yes"][1]}
            recs.append(rec)
            print(f"  {str(q.get('qid')):<12} yes={len(yes)}/5  "
                  + "  ".join(f"{k[0]}{'✅' if arms[k][2] else '·'}" for k in
                              ("S_shipped", "Y_route_yes", "A_route_all")), flush=True)
        (OUT / f"{args.tag}_{g}.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    n = len(recs)
    print(f"\n{'=' * 116}\n【实验 2b｜4 臂·同判定换内容载体】{n} 题 M1（生产 5 篇）")
    print(f"  {'臂':<46}{'ok':>10}{'ok_strict':>14}")
    print(f"  {'fullctx 全上下文直读（存档对照）':<46}{'33/50':>10}{'30/50  60.0%':>14}")
    names = (("S_shipped", "S shipped：篇名+理由（现行 set 产品行为）"),
             ("Y_route_yes", "Y route_yes：判定当路由 + yes 篇原文"),
             ("A_route_all", "A route_all：全部 5 篇原文（不靠判定）"))
    for k, nm in names:
        ok = sum(1 for r in recs if r[f"{k}_ok"])
        st = sum(1 for r in recs if r[f"{k}_strict"])
        print(f"  {nm:<46}{f'{ok}/{n}':>10}{f'{st}/{n}  {st / n:.1%}':>14}")
    print(f"\n  均交付篇数 {sum(r['n_yes'] for r in recs) / n:.2f}/5")
    print(f"已写 {OUT / (args.tag + '_group*.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
