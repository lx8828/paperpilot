"""组级跑批（**走完整图** `paperpilot.graph.ask`）：路由 + 检索 + 输出闸门 + 修复回路。

## 为什么必须单开一个（2026-09-23 假测试排查）

| 脚本 | 走的链路 | 覆盖 |
|---|---|---|
| `run_multi_qa.py` | **直连节点**：自己调检索 → `generate_answer` | 只测"检索+生成" |
| 本脚本 | `web/app.py` 同款 **`graph.ask`** | 生产全链路 |

`run_multi_qa.py` 跳过了 **Router（`route_state`）/ `judge_l3` / `answer_unknown` /
`validator.gate`（默认**开**，会 repair / supplement / 数值裁决 / 兜底话术）**。
而范式文档里写的单篇层入口 `run_qa_v2.py` **已不存在**（该脚本已不在仓库）→ 这两组的题
（5 篇 × 14 + 组级 20 = 每组 90 题）此前**没有任何脚本走过完整图**，
即"测的路径 ≠ 用户实际用的路径"。本脚本补上这一环。

## 用法

    uv run python cli/eval/run_group_qa.py --group group2            # 全量（A 70 + B 20）
    uv run python cli/eval/run_group_qa.py --group group2 --limit 3  # A/B 各 3 题
    uv run python cli/eval/run_group_qa.py --group group2 --out qa/multi/_runs/group2_graph.json

判分口径与题集加载**复用 `run_multi_qa`**（`score_answer` / `load_*_questions`），
不另写一份 —— 判分散在两处必漂移（这正是 2026-09-23 发现的假测试之一）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))              # 同目录 runner 工具
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))               # _qa_groups

from _qa_groups import papers  # noqa: E402
from run_multi_qa import (  # noqa: E402
    assert_no_gold,
    load_cross_questions,
    load_single_questions,
    score_answer,
)


def group_corpus(group: str) -> list[str]:
    """组 → 语料 PDF 文件名（并**核实文件真在库里**）。

    ⚠️ 必须核实：`_qa_groups` 里存的是 stem，而库里可能有版本号（`2609.02094v1`）。
    名字对不上时 `graph.ask` 会走到"未摄取"分支**静默返回拒绝**，
    跑批结果看起来像"模型答不出"，实则是语料没接上（`run_multi_qa.auto_corpus`
    注释里记过同一个坑）。
    """
    out: list[str] = []
    for stem in papers(group):
        pdf = ROOT / "assets" / "papers" / f"{stem}.pdf"
        if not pdf.exists():
            print(f"  ✗ 语料缺失：{pdf.relative_to(ROOT)}"
                  f"（graph.ask 会判「未摄取」→ 静默返回拒绝）")
            continue
        out.append(f"{stem}.pdf")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group2", help="分组名（见 retrieval/scripts/_qa_groups.py）")
    ap.add_argument("--limit", type=int, default=0, help="A/B 组各最多跑 N 题（0=不限）")
    ap.add_argument("--qid", default="",
                    help="只跑指定 qid（逗号分隔）——定向验证用，比 --limit 省，且能挑到"
                         "**真的会下钻检索**的题（--limit 总先撞上 L0/M0 免检索题）")
    ap.add_argument("--out", default="", help="落盘路径；建议 qa/multi/_runs/<group>_graph.json")
    ap.add_argument("--skip-m1", action="store_true",
                    help="跳过 B 组的 M1（跨篇·需检索）——只测『免检索（单篇/M0/X5）+ 需检索的单篇』")
    ap.add_argument("--repeat", type=int, default=1,
                    help="同一批题**重复跑 N 遍**（测波动/噪声底；每题记录带 `rep` 字段）。"
                         "⚠️ 端到端同配置重跑实测有 8/50 翻盘 → 结论必须站在重复之上")
    args = ap.parse_args()

    from paperpilot.graph import ask as graph_ask
    from paperpilot.tools import llm
    from paperpilot.workflow import _ensure_env

    _ensure_env()

    corpus = group_corpus(args.group)
    if len(corpus) < 2:
        print(f"✗ 语料不足 2 篇（{corpus}）—— 本系统没有单篇路径")
        return 1

    qs: list[dict[str, Any]] = load_single_questions(corpus)
    qs += load_cross_questions(corpus)
    if args.qid:
        want = {x.strip() for x in args.qid.split(",") if x.strip()}
        n0 = len(qs)
        qs = [q for q in qs if str(q.get("qid") or "") in want]
        hit = {str(q.get("qid")) for q in qs}
        print(f"[过滤] --qid：请求 {len(want)} 个 → 命中 {len(qs)}/{n0} 题"
              f"（未命中：{sorted(want - hit) or '无'}）", flush=True)
    elif args.limit:
        a = [q for q in qs if q["group"] == "A"][: args.limit]
        b = [q for q in qs if q["group"] == "B"][: args.limit]
        qs = a + b
    if args.skip_m1:
        n0 = len(qs)
        qs = [q for q in qs
              if not (q["group"] == "B" and str(q.get("kind") or "") == "M1")]
        print(f"[过滤] --skip-m1：{n0} → {len(qs)} 题（去掉 {n0 - len(qs)} 道跨篇 M1）", flush=True)

    print("=" * 108)
    from paperpilot.agents.nodes.read_full import reader as _reader

    print(f"[组] {args.group} ｜ [链路] graph.ask（完整图：Router → "
          f"{'read_full 全文直读' if _reader() == 'fullctx' else 'search_l3 检索'} → "
          f"answer → 闸门+修复）")
    for p in corpus:
        print(f"   · {p}")
    print(f"[问题] A组(指向某一篇) {sum(1 for q in qs if q['group'] == 'A')} 题 ｜ "
          f"B组(跨篇) {sum(1 for q in qs if q['group'] == 'B')} 题", flush=True)

    # 生产链路守卫（缺了它，"测的不是生产路径"又会静默复发）
    import os
    gate = os.environ.get("PAPERPILOT_VALIDATOR_GATE", "1") != "0"
    nol3j = os.environ.get("PAPERPILOT_V3_NOL3J", "1") != "0"
    print(f"[配置] 输出闸门(validator.gate)={'开' if gate else '关'} ｜ "
          f"nol3j(删 judge_l3)={'开' if nol3j else '关'}", flush=True)

    recs: list[dict[str, Any]] = []
    reps = max(int(args.repeat), 1)
    for rep in range(reps):
        if reps > 1:
            print(f"\n{'━' * 20} 第 {rep + 1}/{reps} 遍 {'━' * 20}", flush=True)
        for q in qs:
            question = str(q.get("question") or "")
            # ⚠️ gold 边界守卫：`graph.ask` **只允许**拿「题干 + 语料」。
            #    `q` 里同时躺着 `hint`（=答案原文）与 `must_all`，一旦有人在这里
            #    把它拼进题干就会**瞬间满分**且看不出来（见 run_multi_qa「gold 隔离」）。
            assert question == str(q.get("question") or ""), "题干被拼接了非题干内容"
            assert_no_gold({"question": question, "pdfs": corpus},
                           f"{q.get('qid')}/graph_ask")
            t0 = time.time()
            llm.reset_usage()                          # 每题单独记账（2026-09-27）
            try:
                out = graph_ask(question, corpus)
            except Exception as e:  # noqa: BLE001
                out = {"answer": f"ERR {type(e).__name__}: {e}", "cites": [],
                       "route": ["error"]}
            us = llm.usage_stats()
            answer = str(out.get("answer") or "")
            cites = list(out.get("cites") or [])
            ok, miss, ok_strict = score_answer(q, answer, cites)
            dbg = out.get("debug") or {}
            # 问答读取器路径（2026-09-27）：`fullctx`（默认，全文直读）
            # 或 RAG-2 的 `layered_quota` / `layered_quota+xling`。旧产物无 `reader` → 回退读 `l3`。
            rpath = str(((dbg.get("reader") or dbg.get("l3") or {}).get("path")) or "")
            v = out.get("validator") or {}
            recs.append({
                **q, "rep": rep, "answer": answer, "ok": ok, "ok_strict": ok_strict,
                "miss": miss,
                "route": list(out.get("route") or []),
                "l3_path": rpath,                   # 生产应为 layered_quota / multi_hybrid
                "n_cites": len(cites),
                "validator_action": str(v.get("action") or ""),
                "parse_degraded": bool(out.get("parse_degraded")),
                "seconds": round(time.time() - t0, 1),
                # ── 每题用量（2026-09-27 新增：全上下文方案要拿它定预算）──
                "llm_calls": us["calls"],
                "prompt_tokens": us["prompt_tokens"],
                "completion_tokens": us["completion_tokens"],
            })
            tag = "✅" if ok else "⚠️"
            extra = (f"route={','.join(recs[-1]['route']) or '—'}"
                     f" l3={rpath or '—'} cites={len(cites)}"
                     f" tok={us['prompt_tokens']}/{us['completion_tokens']}"
                     + (f" 闸门={recs[-1]['validator_action']}"
                        if recs[-1]["validator_action"] else "")
                     + (" ⚠️解析降级" if recs[-1]["parse_degraded"] else ""))
            print(f"  [{tag}] r{rep} {q['group']} "
                  f"{str(q.get('qid') or '')[:14]:<14} {extra}", flush=True)
            print(f"        Q: {question[:88]}")
            print(f"        A: {answer[:150]}")

    print("\n" + "=" * 108)
    print(f"【汇总】{args.group} ｜ 链路=graph.ask（完整图）")
    for g, name in (("A", "指向某一篇的题@多篇语料"), ("B", "跨篇问题（新能力）")):
        gs = [r for r in recs if r["group"] == g]
        if not gs:
            continue
        okn = sum(1 for r in gs if r["ok"])
        stn = sum(1 for r in gs if r.get("ok_strict"))
        print(f"  {name:<18} ✅ {okn}/{len(gs)} = {okn / len(gs):.0%}"
              f"  ｜ 仅答案判分 {stn}/{len(gs)} = {stn / len(gs):.0%}"
              + (f"（引用通道 +{okn - stn}）" if okn != stn else ""))
        if g == "B":
            ks = Counter(str(r.get("kind") or "?") for r in gs)
            parts = [f"{k} {sum(1 for r in gs if str(r.get('kind')) == k and r['ok'])}/{ks[k]}"
                     for k in ("M0", "M1", "X5") if ks.get(k)]
            print(f"  {'':<18} B 组分类：{' ｜ '.join(parts)}"
                  f"   （M0 多篇免检索 / M1 跨篇需检索 / X5 语料级拒答）")

    # ── 生产链路健康度（这几项是"完整图有没有真的被走到"的硬证据）──
    print(f"\n  [链路] route 分布：{dict(Counter(','.join(r['route']) or '—' for r in recs))}")
    print(f"  [链路] 问答读取器：{dict(Counter(r['l3_path'] or '（未读数）' for r in recs))}"
          f"   ← 默认 fullctx（全文直读）；retrieval 派为 layered_quota(+xling)")
    # 读取器白名单：`fullctx`（默认）/ `layered_quota` / 带 L6 跨语言补充的 `layered_quota+xling`。
    if any(r["l3_path"] and r["l3_path"] not in ("fullctx", "layered_quota", "layered_quota+xling")
           for r in recs):
        print("  ⚠️ 有题走了**非生产**读取路径（multi_hybrid 需 PAPERPILOT_QUERY_LEVELS；"
              "或 XLING 被改坏）")
    va = Counter(r["validator_action"] or "（未过闸门）" for r in recs)
    print(f"  [闸门] 动作分布：{dict(va)}   ← pass 之外的都是**答案被改写过**的题")
    if sum(1 for r in recs if r["parse_degraded"]):
        print(f"  ⚠️ 解析降级（MinerU→pymupdf）题数：{sum(1 for r in recs if r['parse_degraded'])}")

    # ── 用量与上下文规模（2026-09-27 新增：给"全上下文直灌"方案定预算）──
    if recs and any(r.get("prompt_tokens") for r in recs):
        n = len(recs)
        pt = sum(int(r.get("prompt_tokens") or 0) for r in recs)
        ct = sum(int(r.get("completion_tokens") or 0) for r in recs)
        # 三档分开统计（2026-09-27）：L0 直答 / **全文直读(FULLCTX)** / L3 检索。
        # ⚠️ 不能把 FULLCTX 并进 L0 —— 两者输入规模差一个量级（12k vs 54k tok），
        #    合并后"全塞方案的输入规模"就看不出来了。
        def _pt(rs):
            return [int(r.get("prompt_tokens") or 0) for r in rs]

        rt = lambda r: (r.get("route") or [])  # noqa: E731
        l0 = _pt([r for r in recs if "L3" not in rt(r) and "FULLCTX" not in rt(r)])
        fc = _pt([r for r in recs if "FULLCTX" in rt(r)])
        l3 = _pt([r for r in recs if "L3" in rt(r)])
        avg = lambda xs: (sum(xs) / len(xs) if xs else 0.0)  # noqa: E731
        print(f"\n  [用量] 均 prompt {pt / n:.0f} tok ｜ 均 completion {ct / n:.0f} tok"
              f" ｜ 均调用 {sum(int(r.get('llm_calls') or 0) for r in recs) / n:.1f} 次/题")
        print(f"  [用量] **L0 直答** {len(l0)} 题 均 prompt {avg(l0):.0f} tok"
              f" ｜ **全文直读** {len(fc)} 题 均 prompt {avg(fc):.0f} tok"
              f" ｜ **L3 检索** {len(l3)} 题 均 prompt {avg(l3):.0f} tok"
              f"   ← 这就是『全塞』方案要对比的输入规模")
        print(f"  [用量] 合计 prompt {pt} tok / completion {ct} tok"
              f"（deepseek-chat 参考价 ¥1/¥2 每百万 → 约 ¥{(pt * 1 + ct * 2) / 1e6:.3f}）")

    if args.out:
        Path(args.out).write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"\n→ 落盘 {args.out}")

    _emit_report(args.group, recs)
    return 0


def _emit_report(group: str, recs: list[dict]) -> None:
    """把**汇总指标**落进统一记录格式（`evals/report.py`）—— 让各层数字可比。

    为什么单独做这一步：上面所有汇总**只 print 到 stdout，跑完就没了** →
    无法与历史比、也无法与 L1/L2 放在一张表里（要翻日志考古）。

    ⚠️ **失败绝不影响评测**：报告只是记录，跑批本身已经完成；这里 try 住并告警。
    """
    try:
        sys.path.insert(0, str(ROOT))
        from evals import report as R  # noqa: PLC0415

        n = len(recs)
        if not n:
            return
        ev = f"qa/multi/_runs/{group}_graph.json"
        common = dict(layer="L3", name=f"{group}_graph", evidence_path=ev)

        def rate(key: str, ok_key: str, subset: list[dict], note: str) -> dict:
            k = sum(1 for r in subset if r.get(ok_key))
            return dict(metric=key, value=(k / len(subset)) if subset else 0.0,
                        n=len(subset) or 1, note=note, **common)

        A = [r for r in recs if r["group"] == "A"]
        B = [r for r in recs if r["group"] == "B"]
        out = [
            rate("qa.ok", "ok", recs, "锚点命中率（含引用通道）／全部 A+B 题"),
            rate("qa.ok_strict", "ok_strict", recs,
                 "**仅答案判分**（不含引用通道）／全部题"),
            rate("qa.ok_a", "ok", A, "A 组：指向某一篇的题@多篇语料（单篇层）"),
            rate("qa.ok_b", "ok", B, "B 组：跨篇问题（新能力）"),
        ]
        for kind, desc in (("M0", "多篇·免检索（答案须落在 5 篇 L0 材料里）"),
                           ("M1", "跨篇·需下钻检索"),
                           ("X5", "语料级拒答（5 篇都没有）")):
            ks = [r for r in recs if str(r.get("kind")) == kind]
            if ks:
                out.append(rate(f"qa.kind_{kind.lower()}", "ok", ks, f"B 组 {kind}：{desc}"))

        # 健康度：这几项**应为 0**，非 0 就是链路出了问题（比准确率更该先看）
        off = sum(1 for r in recs if r["l3_path"] and r["l3_path"] not in (
            "fullctx", "layered_quota", "layered_quota+xling"))
        out += [
            dict(metric="qa.reader_offlabel", value=off, n=n,
                 note="走了**非生产**读取路径的题数（白名单 fullctx/layered_quota"
                      "/layered_quota+xling）；**应为 0**", **common),
            dict(metric="qa.validator_rewritten",
                 value=sum(1 for r in recs if (r.get("validator_action") or "pass") != "pass"),
                 n=n, note="答案被闸门**改写/拦下**过的题数（pass 之外都算）", **common),
            dict(metric="qa.parse_degraded",
                 value=sum(1 for r in recs if r["parse_degraded"]), n=n,
                 note="解析降级（MinerU→pymupdf）的题数", **common),
        ]
        if any(r.get("prompt_tokens") for r in recs):
            pt = sum(int(r.get("prompt_tokens") or 0) for r in recs)
            ct = sum(int(r.get("completion_tokens") or 0) for r in recs)
            out += [
                dict(metric="qa.avg_prompt_tokens", value=pt / n, n=n,
                     note="平均输入 token／题（含 fullctx 直灌的整篇上下文）", **common),
                dict(metric="qa.cost_cny", value=(pt * 1 + ct * 2) / 1e6, n=n,
                     note="按 deepseek-chat 参考价 ¥1/¥2 每百万 token 估算", **common),
            ]
        p = R.emit(*out)
        print(f"→ 汇总记录已落 {p.relative_to(ROOT)}（{len(out)} 条；"
              f"`python evals/report.py --table` 可看全层对比）")
    except Exception as e:  # noqa: BLE001  报告失败不能弄挂评测
        print(f"⚠️ 汇总记录落盘失败（不影响本次评测结果）：{type(e).__name__}: {e}",
              file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
