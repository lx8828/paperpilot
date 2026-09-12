"""A/B：新默认（nol3j+gate）下，查询改写 关(0) vs 开(1) 的端到端效果。

样本 = ab_v3base 同 100 题（50 hard + 50 normal，与历史 nol3j 同题可比）。
同题同进程、同外部裁判 glm-4-flash，两臂各跑一次完整 graph.ask。
记录：pass/score/level/calls/prompt + 历史 oldV（nol3j 73% 那批）。
用途：决定 PAPERPILOT_QUERY_REWRITE 是否升默认（工程上多篇必需，但需数据背书）。
"""
from __future__ import annotations
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/ab_rewrite_result.json")
PASS = 4

_JUDGE_SYS = (
    "你是严谨的论文问答裁判。你会看到：① 论文中的一道问题；② 人工标注的标准答案（可能含原文摘录）；"
    "③ 待评测系统回答与它引用的论文原文。请独立判断系统回答是否**准确**回答了问题（与标准答案一致、不编造、"
    '不遗漏关键信息）。只输出 JSON：{"score": 1-5, "reason": "一句话"}。5=完全正确且细节齐全；'
    "4=正确仅缺次要细节；3=部分正确有重要遗漏或轻微错误；2=明显错误或遗漏核心；1=答非所问或编造。")
_JUDGE_USER = """【问题】
{question}

【标准答案】
gold: {gold}
evidence: {evidence}

【系统回答】
{answer}

【引用原文】
{cites}

请打分(1-5)。引用必须真实支撑，否则视为编造降分。"""


def ext_judge(q, answer, cites):
    gold, ev = gold_answer(q)
    ct = "\n".join(f"[{i+1}](p{c.get('page','?')}) {c.get('evidence','')}"
                   for i, c in enumerate(cites[:5])) or "（无引用）"
    user = _JUDGE_USER.format(question=q.get("question", ""), gold=gold or "（无答案，应诚实说明）",
                              evidence=(ev or ""), answer=(answer or "")[:1500], cites=ct)
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        return int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
    except Exception:
        return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    base = json.load(open("qa/recall/ab_v3base_result.json", encoding="utf-8"))
    old = {}
    if Path("qa/recall/ab_nol3j_result.json").exists():
        old = {r["qid"]: bool(r["oldV_pass"]) for r in
               json.load(open("qa/recall/ab_nol3j_result.json", encoding="utf-8"))}
    papers = load_papers()
    qmap: dict[str, tuple[str, dict]] = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qid = str(q.get("question_id") or "")
            if qid:
                qmap[qid] = (pid, q)
    cases = [(r["qid"], r["grp"]) for r in base]
    if args.limit:
        cases = cases[: args.limit]

    done = set()
    recs: list[dict] = []
    if OUT.exists():
        try:
            recs = json.load(open(OUT, encoding="utf-8"))
            done = {r["qid"] for r in recs}
        except Exception:
            recs, done = [], set()

    t0 = time.time()
    for i, (qid, grp) in enumerate(cases, 1):
        if qid in done:
            continue
        pid, q = qmap[qid]
        qq = q.get("question", "")
        pdf = f"qasper_{pid}.qpdf"

        def run(rewrite: str) -> dict:
            os.environ["PAPERPILOT_QUERY_REWRITE"] = rewrite
            llm.reset_usage()
            try:
                r = graph_ask(qq, pdf)
                dbg = r.get("debug") or {}
                level = ((dbg.get("answer") or {}).get("level", "")) or ""
                v = r.get("validator") or {}
                ans, cites = r.get("answer") or "", list(r.get("cites") or [])
            except Exception as e:  # noqa: BLE001
                ans, cites, level, v = f"ERR {e}", [], "err", {}
            u = llm.usage_stats()
            return {"answer": ans, "cites": cites, "level": level,
                    "action": v.get("action"), "calls": u["calls"],
                    "prompt": u["prompt_tokens"], "completion": u["completion_tokens"]}

        ro = run("0")
        so = ext_judge(q, ro["answer"], ro["cites"])
        rn = run("1")
        sn = ext_judge(q, rn["answer"], rn["cites"])
        os.environ["PAPERPILOT_QUERY_REWRITE"] = "0"

        recs.append({
            "qid": qid, "grp": grp, "oldV_pass": old.get(qid),
            "off": {"score": so, "pass": so >= PASS, **{k: ro[k] for k in
                    ("level", "action", "calls", "prompt", "completion")}},
            "on": {"score": sn, "pass": sn >= PASS, **{k: rn[k] for k in
                   ("level", "action", "calls", "prompt", "completion")}},
        })
        flag = ("same" if (so >= PASS) == (sn >= PASS)
                else ("ON-GAIN" if sn >= PASS else "ON-LOSS"))
        print(f"[{i}/{len(cases)}] {qid[:10]} off={so} on={sn} {flag}", flush=True)
        if i % 5 == 0:
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    def agg(sub, arm):
        n = len(sub) or 1
        return (sum(1 for r in sub if r[arm]["pass"]),
                sum(r[arm]["prompt"] for r in sub) / n,
                sum(r[arm]["calls"] for r in sub) / n,
                sum(r[arm]["completion"] for r in sub) / n)

    L = ["# 查询改写 A/B（新默认 nol3j+gate，100 题）", ""]
    L.append(f"> 题 {len(recs)} | 裁判 glm-4-flash ≥4=pass")
    L.append("")
    L.append("| 分组 | off pass | on pass | Δ | off calls | on calls | off prompt | on prompt |")
    L.append("|---|---|---|---|---|---|---|---|")
    for nm, key in [("all", lambda r: True), ("hard", lambda r: r["grp"] == "hard"),
                    ("normal", lambda r: r["grp"] == "normal")]:
        sub = [r for r in recs if key(r)]
        if not sub:
            continue
        po, pro, co, cmo = agg(sub, "off")
        pn, prn, cn, cmn = agg(sub, "on")
        L.append(f"| {nm} | {po}/{len(sub)} = {po/len(sub)*100:.0f}% | "
                 f"{pn}/{len(sub)} = {pn/len(sub)*100:.0f}% | {pn-po:+d} | "
                 f"{co:.1f} | {cn:.1f} | {pro:.0f} | {prn:.0f} |")
    if old:
        ov = sum(1 for r in recs if r.get("oldV_pass"))
        L.append("")
        L.append(f"> 历史 oldV（nol3j 73% 那批）同题 pass：{ov}/{len(recs)} "
                 f"= {ov/len(recs)*100:.0f}%")
    gains = [r for r in recs if r["on"]["pass"] and not r["off"]["pass"]]
    losses = [r for r in recs if r["off"]["pass"] and not r["on"]["pass"]]
    L.append("")
    L.append(f"> 改写：救回 {len(gains)} 题 / 弄坏 {len(losses)} 题")
    L.append("")
    L.append("### ON 救回")
    for r in gains:
        L.append(f"  {r['qid'][:10]} [{r['grp']}] off={r['off']['score']} on={r['on']['score']}")
    L.append("")
    L.append("### ON 弄坏")
    for r in losses:
        L.append(f"  {r['qid'][:10]} [{r['grp']}] off={r['off']['score']} on={r['on']['score']}")
    txt = "\n".join(L)
    Path("qa/recall/REWRITE_AB_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
