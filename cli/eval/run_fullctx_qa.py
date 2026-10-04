"""**全上下文跑批**：语料一次性塞进单次调用，判分/题集加载与 `run_group_qa.py` 同源。

与 `run_group_qa.py` 的区别（这样两者才能同题对比）：
  · 走 `components.fullctx.answer`（**单次调用**，不做 judge_l0/judge_l3/validator 多轮门控）
  · 每题记录 **prompt cache 命中/未命中 tokens**（`Llm` 2026-09-27 新增）
  · 判分仍用 `run_multi_qa.score_answer`（唯一口径），因此同时得到
      `ok`（答案 + 引用通道）与 `ok_strict`（仅答案）

⚠️ 成本口径：前缀缓存命中价按未命中的 1/10 计（DeepSeek 常见档位），
    脚本里 `P_HIT/P_MISS/P_OUT` 可改。

用法：
    uv run python cli/eval/run_fullctx_qa.py --group group1 --skip-m1 --repeat 3 \
        --out qa/multi/_runs/fullctx_graph_group1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))

from _qa_groups import papers  # noqa: E402
from run_multi_qa import (load_cross_questions, load_single_questions,  # noqa: E402
                          score_answer)

P_MISS, P_HIT, P_OUT = 1.0, 0.1, 2.0        # ¥ / 百万 token


def group_corpus(group: str) -> list[str]:
    out = []
    for stem in papers(group):
        pdf = ROOT / "assets" / "papers" / f"{stem}.pdf"
        if not pdf.exists():
            print(f"  ✗ 语料缺失：{pdf.name}")
            continue
        out.append(f"{stem}.pdf")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group1")
    ap.add_argument("--qid", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--skip-m1", action="store_true")
    ap.add_argument("--only-m1", action="store_true",
                    help="**只跑跨篇 M1**（全上下文路线的真正用武之地）")
    ap.add_argument("--single-layer", action="store_true",
                    help="**按需塞**：单篇层题只塞它自己那一篇（多篇层仍塞全语料）")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--style", default="", choices=("", "base", "cover"),
                    help="`cover` = **强制逐篇覆盖**（M1 小样实测：失败模式多"
                         "为『只答了一两篇』→ 要求每篇都给结论或写『未涉及』）")
    args = ap.parse_args()

    from paperpilot.components import fullctx
    from paperpilot.tools import llm
    from paperpilot.workflow import _ensure_env

    _ensure_env()
    corpus = group_corpus(args.group)
    qs: list[dict[str, Any]] = load_single_questions(corpus)
    qs += load_cross_questions(corpus)
    if args.qid:
        want = {x.strip() for x in args.qid.split(",") if x.strip()}
        qs = [q for q in qs if str(q.get("qid")) in want]
        print(f"[过滤] --qid 命中 {len(qs)} 题（请求 {len(want)}）", flush=True)
    elif args.limit:
        a = [q for q in qs if q["group"] == "A"][: args.limit]
        b = [q for q in qs if q["group"] == "B"][: args.limit]
        qs = a + b
    if args.skip_m1:
        n0 = len(qs)
        qs = [q for q in qs
              if not (q["group"] == "B" and str(q.get("kind") or "") == "M1")]
        print(f"[过滤] --skip-m1：{n0} → {len(qs)}", flush=True)
    if args.only_m1:
        n0 = len(qs)
        qs = [q for q in qs
              if q["group"] == "B" and str(q.get("kind") or "") == "M1"]
        print(f"[过滤] --only-m1：{n0} → {len(qs)}", flush=True)

    print("=" * 108)
    print(f"[组] {args.group} ｜ [链路] fullctx（**单次调用**，语料全塞，"
          f"按需塞={'开' if args.single_layer else '关'}）")
    for p in corpus:
        print(f"   · {p}")
    print(f"[问题] A组 {sum(1 for q in qs if q['group'] == 'A')} 题 ｜ "
          f"B组 {sum(1 for q in qs if q['group'] == 'B')} 题", flush=True)

    recs: list[dict[str, Any]] = []
    for rep in range(max(args.repeat, 1)):
        if args.repeat > 1:
            print(f"\n{'━' * 20} 第 {rep + 1}/{args.repeat} 遍 {'━' * 20}", flush=True)
        for q in qs:
            scoped = corpus
            if args.single_layer and q["group"] == "A" and q.get("home"):
                scoped = [str(q["home"])]
            llm.reset_usage()
            t0 = time.time()
            try:
                out = fullctx.answer(scoped, str(q.get("question") or ""),
                                     max_tokens=args.max_tokens,
                                     style=(args.style or None))
            except Exception as e:  # noqa: BLE001
                out = {"answer": f"ERR {type(e).__name__}: {e}", "cites": [],
                       "ctx_chars": 0, "n_papers": len(scoped)}
            sec = time.time() - t0
            us = llm.usage_stats()
            ans = str(out.get("answer") or "")
            cites = list(out.get("cites") or [])
            ok, miss, ok_strict = score_answer(q, ans, cites)
            recs.append({**q, "rep": rep, "answer": ans, "ok": ok, "ok_strict": ok_strict,
                         "miss": miss, "route": ["FULLCTX"],
                         "n_cites": len(cites), "n_papers": out.get("n_papers"),
                         "ctx_chars": out.get("ctx_chars"), "seconds": round(sec, 1),
                         "llm_calls": us["calls"], "prompt_tokens": us["prompt_tokens"],
                         "completion_tokens": us["completion_tokens"],
                         "cache_hit_tokens": us["cache_hit_tokens"],
                         "cache_miss_tokens": us["cache_miss_tokens"]})
            r = recs[-1]
            print(f"  [{'✅' if ok else '⚠️'}] r{rep} {q['group']} "
                  f"{str(q.get('qid'))[:14]:<14} 塞{r['n_papers']}篇 "
                  f"{sec:>5.1f}s tok={r['prompt_tokens']}"
                  f"(hit {r['cache_hit_tokens']}/{r['cache_miss_tokens']})"
                  f" cites={len(cites)}", flush=True)

    n = len(recs)
    pt = sum(int(r["prompt_tokens"]) for r in recs)
    ct = sum(int(r["completion_tokens"]) for r in recs)
    hit = sum(int(r["cache_hit_tokens"]) for r in recs)
    miss_t = sum(int(r["cache_miss_tokens"]) for r in recs)
    cost = (miss_t * P_MISS + hit * P_HIT + ct * P_OUT) / 1e6
    print(f"\n{'=' * 108}\n【汇总·fullctx】{args.group}")
    for g, name in (("A", "指向某一篇的题@多篇语料"), ("B", "跨篇/组级题")):
        gs = [r for r in recs if r["group"] == g]
        if not gs:
            continue
        okn = sum(1 for r in gs if r["ok"])
        stn = sum(1 for r in gs if r["ok_strict"])
        print(f"  {name:<20}✅ {okn}/{len(gs)} = {okn / len(gs):.0%}"
              f" ｜ 仅答案 {stn}/{len(gs)} = {stn / len(gs):.0%}"
              + (f"（引用通道 +{okn - stn}）" if okn != stn else ""))
        ks = Counter(str(r.get("kind") or r.get("intent") or "?") for r in gs)
        if g == "B":
            parts = [f"{k} {sum(1 for r in gs if str(r.get('kind')) == k and r['ok'])}/{ks[k]}"
                     for k in ("M0", "M1", "X5") if ks.get(k)]
            print(f"  {'':<20} B 组分类：{' ｜ '.join(parts)}")
    print(f"\n  [用量] 均 prompt {pt / n:.0f} tok ｜ 均 completion {ct / n:.0f} tok"
          f" ｜ 均 {sum(float(r['seconds']) for r in recs) / n:.1f} 秒"
          f" ｜ 均调用 {sum(int(r['llm_calls']) for r in recs) / n:.1f} 次")
    hrate = hit / max(hit + miss_t, 1)
    print(f"  [缓存] **前缀命中率 {hrate:.0%}**（hit {hit} / miss {miss_t} tok）"
          f" ← 命中价按 1/10 计")
    print(f"  [成本] 合计 ¥{cost:.3f} ｜ **¥{cost / n:.4f}/题**"
          f"（全未命中价 ¥{(pt * P_MISS + ct * P_OUT) / 1e6 / n:.4f}/题）")
    if args.out:
        Path(args.out).write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"\n→ 落盘 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
