"""**验证 `extract` 补丁**（生产 5 篇 · 50 道 M1 复合题）

对照臂（全部同题）：
  · `fullctx` 全上下文直读（存档）          30/50 = 60.0%
  · S `shipped`（仅判定，无内容）            7/50 =  14.0%
  · Y 判定当路由 + yes 篇原文                34/50 =  68.0%
  · A 全部 5 篇原文倾倒（不靠判定）           44/50 =  88.0%  ← 假阳性上界
本臂：**E `extract`** = 逐篇判定 + 逐篇内容抽取 → `render()`（自然语言，非倾倒原文）

读法：
  · E ≈ 60% 上下 → **补对了**（判定保留 + 内容补上，且不靠倾倒原文）
  · E ≈ 14%     → `extract` 没生效
  · E ≈ 88%     → 又变成"贴原文"了（说明 `extract` 太长/在抄片段）
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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="setextract")
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
        idx = MultiChunkIndex(corpus)
        print(f"\n[{g}] {len(corpus)} 篇 / {len(idx._doc_chunks())} 块 ｜ M1 {len(qs)} 题",  # noqa: SLF001
              flush=True)
        recs = []
        for q in qs:
            qq = str(q.get("question") or "")
            llm.reset_usage()
            res = set_judge.run(qq, idx, extract=True)
            ans = set_judge.render(qq, res)
            cites = [{"pdf": p["pdf"], "chunk_id": p["chunk_id"], "page": p["page"],
                      "section": p["section"], "evidence": p["snippet"]} for p in res["papers"]]
            ok, miss, st = score_answer(q, ans, cites)
            us = llm.usage_stats()
            recs.append({**q, "answer": ans, "papers": [p["pdf"] for p in res["papers"]],
                         "extracts": {p["pdf"]: p["extract"] for p in res["papers"]},
                         "n_yes": res["n_yes"], "extract": res["extract"],
                         "ok": ok, "ok_strict": st, "miss": miss,
                         "n_papers": res["n_papers"], "prompt_tokens": us["prompt_tokens"],
                         "llm_calls": us["calls"]})
            print(f"  {'✅' if st else '·'} {str(q.get('qid')):<12} yes={res['n_yes']}/5"
                  f" 有extract={sum(1 for p in res['papers'] if p['extract'])}/{res['n_yes']}"
                  f" miss={('；'.join(map(str, miss[:1])) or '—')[:30]:<32}", flush=True)
            if args.limit:
                for p in res["papers"]:
                    print(f"       · {p['pdf']}: {p['extract'][:70]}")
        all_recs += recs
        (OUT / f"{args.tag}_{g}.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    n = len(all_recs)
    ok = sum(1 for r in all_recs if r["ok"])
    st = sum(1 for r in all_recs if r["ok_strict"])
    ne = sum(1 for r in all_recs if r["extract"])
    print(f"\n{'=' * 116}\n【`extract` 补丁验证｜生产 5 篇 M1】{n} 题")
    print(f"  {'臂':<44}{'ok':>9}{'ok_strict':>13}")
    for nm, a, b in (("fullctx 全上下文直读（存档对照）", 33, 30),
                     ("S shipped 仅判定·无内容", 8, 7),
                     ("Y 判定当路由 + yes 篇原文", 34, 34),
                     ("A 全部 5 篇原文倾倒（假阳性上界）", 44, 44)):
        print(f"  {nm:<44}{f'{a}/50':>9}{f'{b}/50  {b / 50:.1%}':>13}")
    print(f"  {'E extract 逐篇判定+逐篇内容（**本臂**）':<44}"
          f"{f'{ok}/{n}':>9}{f'{st}/{n}  {st / n:.1%}':>13}")
    print(f"\n  extract 生效题数 {ne}/{n} ｜ 均交付篇数 "
          f"{sum(r['n_yes'] for r in all_recs) / n:.2f}/5"
          f" ｜ 均 {sum(int(r['llm_calls']) for r in all_recs) / n:.1f} 次调用")
    print(f"已写 {OUT / (args.tag + '_group*.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
