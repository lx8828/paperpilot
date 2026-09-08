"""三列对比结果汇总 → compare_<ts>.md。

用法：
    uv run python qa/compare/_summarize.py \
        --b1 qa/compare/run_<ts>_B1.json --b0 qa/compare/run_<ts>_B0.json \
        --b2 qa/compare/run_<ts>_B2.json --out qa/compare/compare_<ts>.md

口径：与 qa/COMPARE_DESIGN.md §2/§3 一致。成本按传入单价（元/1M token）估算，
默认 deepseek-chat 入 2 / 出 8（仅估算，报告注明假设；裁判 glm-4-flash 免费不计）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load(p: str) -> list[dict]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _bucket(rec: dict) -> str:
    st = rec.get("status", "")
    return {"pass": "pass", "overflow": "overflow", "error": "error"}.get(st, "fail")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", required=True)
    ap.add_argument("--b0", required=True)
    ap.add_argument("--b2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--price-in", type=float, default=2.0)
    ap.add_argument("--price-out", type=float, default=8.0)
    args = ap.parse_args()

    cols = {"B1": load(args.b1), "B0": load(args.b0), "B2": load(args.b2)}
    price = {"in": args.price_in, "out": args.price_out}

    def fmt_cost(rec: dict) -> float:
        g = rec.get("gen") or {}
        return (g.get("prompt_tokens", 0) * price["in"]
                + g.get("completion_tokens", 0) * price["out"]) / 1_000_000

    def agg(recs: list[dict]) -> dict:
        total = len(recs)
        buck = {k: sum(1 for r in recs if _bucket(r) == k) for k in
                ("pass", "fail", "overflow", "error")}
        scored = [r.get("score", 0) for r in recs if r.get("score", 0) > 0]
        unans_l = [r for r in recs if r.get("unanswerable")]
        unans_pass = sum(1 for r in unans_l if _bucket(r) == "pass")
        ans_l = [r for r in recs if not r.get("unanswerable")]
        ans_pass = sum(1 for r in ans_l if _bucket(r) == "pass")
        gen_t = sum((r.get("gen") or {}).get("prompt_tokens", 0)
                    + (r.get("gen") or {}).get("completion_tokens", 0) for r in recs)
        jdg_t = sum((r.get("judge") or {}).get("prompt_tokens", 0)
                    + (r.get("judge") or {}).get("completion_tokens", 0) for r in recs)
        calls = sum((r.get("gen") or {}).get("calls", 0) for r in recs)
        jcalls = sum((r.get("judge") or {}).get("calls", 0) for r in recs)
        times = [r.get("time_s", 0) for r in recs
                 if r.get("status") not in ("overflow", "error")]
        t_ans = [r.get("time_s_ans", 0) for r in recs
                 if r.get("status") not in ("overflow", "error")]
        cites_n = [r.get("cites_n", 0) for r in recs]
        cost = sum(fmt_cost(r) for r in recs)
        return {
            "total": total, "buck": buck,
            "pass_rate": ans_pass / len(ans_l) if ans_l else 0,
            "unans_pass": unans_pass, "unans_n": len(unans_l),
            "avg_score": round(sum(scored) / len(scored), 2) if scored else 0,
            "avg_t": round(sum(times) / len(times), 1) if times else 0,
            "avg_t_ans": round(sum(t_ans) / len(t_ans), 1) if t_ans else 0,
            "gen_pt": gen_t, "jdg_pt": jdg_t, "calls": calls, "judge_calls": jcalls,
            "avg_cites": round(sum(cites_n) / len(cites_n), 2) if cites_n else 0,
            "est_cost": round(cost, 4),
        }

    A = {c: agg(v) for c, v in cols.items()}

    L = [f"# 对比测试报告（直接 LLM / 朴素 RAG vs PaperPilot）\n"]
    L.append(f"> 生成: {time.strftime('%Y-%m-%d %H:%M:%S')} ｜ "
             f"题数: {A['B1']['total']}（与 B0/B2 同题）")
    L.append(f"> 裁判: glm-4-flash（异源，三列同 prompt 盲判）｜ 生成: deepseek-chat(v4-flash) "
             f"｜ 成本估算单价: 入 {price['in']} / 出 {price['out']} 元/M（假设，见 COMPARE_DESIGN §3.3）")
    L.append("")

    # 主表
    L.append("## 1. 主指标（三列同一批题）\n")
    L.append("| 指标 | B0 直接 LLM | B1 朴素 RAG | B2 PaperPilot | 说明 |")
    L.append("|---|---|---|---|---|")
    rows = [
        ("pass 通过数(总)", lambda c: f"{A[c]['buck']['pass']}/{A[c]['total']}"),
        ("有答案题 pass 率", lambda c: f"{A[c]['pass_rate']:.1%}"),
        ("无答案题 pass(诚实拒答被认可)", lambda c: f"{A[c]['unans_pass']}/{A[c]['unans_n']}"),
        ("overflow(放不下)", lambda c: f"{A[c]['buck']['overflow']}"),
        ("error", lambda c: f"{A[c]['buck']['error']}"),
        ("均分(1-5)", lambda c: f"{A[c]['avg_score']}"),
        ("平均耗时/题(s, 含判分)", lambda c: f"{A[c]['avg_t']}"),
        ("平均耗时/题(s, 仅作答)", lambda c: f"{A[c]['avg_t_ans']}"),
        ("生成 token 总量", lambda c: f"{A[c]['gen_pt']:,}"),
        ("生成调用总数", lambda c: f"{A[c]['calls']}"),
        ("裁判 token 总量", lambda c: f"{A[c]['jdg_pt']:,}"),
        ("平均 cites/题", lambda c: f"{A[c]['avg_cites']}"),
        ("估算生成成本(元)", lambda c: f"{A[c]['est_cost']:.3f}"),
    ]
    for name, fn in rows:
        L.append(f"| {name} | {fn('B0')} | {fn('B1')} | {fn('B2')} | — |")
    L.append("")

    # 分题型
    L.append("## 2. 分题型 pass 率\n")
    L.append("| qtype | B0 | B1 | B2 |")
    L.append("|---|---|---|---|")
    for qt in ("free_form", "yes_no", "extractive", "unanswerable"):
        line = f"| {qt} "
        for c in ("B0", "B1", "B2"):
            sub = [r for r in cols[c] if r.get("qtype") == qt]
            n = len(sub)
            p = sum(1 for r in sub if _bucket(r) == "pass")
            line += f"| {p}/{n}" if n else "| —"
        L.append(line + " |")
    L.append("")

    # 每题对比（三方不一致才列出核心差）
    by_qid = {}
    for c, recs in cols.items():
        for r in recs:
            by_qid.setdefault(r["qid"], {})[c] = r
    L.append("## 3. 关键差异题（B2 与 B1/B0 判分不一致且 B2 更高/更低）\n")
    L.append("| qid | 题型 | B0 | B1 | B2 | 问题(截断) |")
    L.append("|---|---|---|---|---|---|")
    n_shown = 0
    for qid, m in by_qid.items():
        s0 = m.get("B0", {}).get("score", 0)
        s1 = m.get("B1", {}).get("score", 0)
        s2 = m.get("B2", {}).get("score", 0)
        if s2 > max(s0, s1) and s2 >= 4 or (s2 < min(s0, s1) if s0 and s1 else False):
            q = (m.get("B2") or m.get("B1") or m.get("B0")).get("question", "")
            qt = (m.get("B2") or m.get("B1") or m.get("B0")).get("qtype", "")
            L.append(f"| {qid[:10]} | {qt} | {s0} | {s1} | {s2} | {q[:44]} |")
            n_shown += 1
            if n_shown >= 15:
                break
    if n_shown == 0:
        L.append("| — 无显著分歧 — | | | | |")
    L.append("")
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:2]))
    print(f"已写 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
