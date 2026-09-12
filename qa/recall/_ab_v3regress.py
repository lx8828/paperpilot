"""v3 定稿回归：同裁判同进程 A(现状 v2 漏斗) vs V0(v3 纯两级)，权威规模样本。

样本：从 qasper_overnight_full（官方 1000 题口径，README 75.0%）中抽 n 题
（gold 有答案 + report 存在 + load_papers 可定位），seed 固定。另记录官方 status
作子集代表性参照。判分协议同 v3base 100 题（external_judge，glm-4-flash）。
定稿判据：V0 pass ≥ A pass（同 run 同裁判）→ v3 替换 v2 作默认；成本一并记。
"""
from __future__ import annotations
import io
import json
import random
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.graph.qa_graph_v3 import build_qa_graph_v3  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
OUT = Path("qa/recall/ab_v3regress_result.json")


def external_judge(q, answer, cites):
    gold, ev = gold_answer(q)
    ct = "\n".join(f"[{i+1}](p{c.get('page','?')}) {c.get('evidence','')}"
                   for i, c in enumerate(cites[:5])) or "（无引用）"
    user = _JUDGE_USER.format(question=q.get("question", ""), gold=gold or "（无答案，应诚实说明）",
                              evidence=(ev or ""), answer=answer[:1500], cites=ct)
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        return int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
    except Exception:
        return 0


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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    app0 = build_qa_graph_v3()
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)
    view = ROOT / "assets/artifacts/out_views"
    overnight = json.load(open("qa/qasper_overnight_full_20260907_142814.json", encoding="utf-8"))
    # 有答案(gold 可定位) + report 存在 的官方样本
    cand = []
    for rec in overnight:
        qid = rec["qid"]
        if not rec.get("gold") or rec.get("unanswerable"):
            continue
        if qid not in qmap:
            continue
        pid = qmap[qid][0]
        if not (view / f"qasper_{pid}.report.json").exists():
            continue
        cand.append((qid, pid, bool(rec.get("status") == "pass")))
    print(f"官方 1000 中可用有答案题: {len(cand)}")
    rng = random.Random(args.seed)
    rng.shuffle(cand)
    sel = cand[: args.n]
    if args.limit:
        sel = sel[: args.limit]

    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    recs = []
    t0 = time.time()
    for i, (qid, pid, official_pass) in enumerate(sel, 1):
        q = qmap[qid][1]
        qq = q.get("question", "")
        pdf = f"qasper_{pid}.qpdf"

        def run(graph):
            llm.reset_usage()
            try:
                out = graph_ask(qq, pdf) if graph == "A" else app0.invoke({"question": qq, "pdf": pdf})
                answer = (out.get("answer") or "").strip()
                cites = list(out.get("cites") or [])
                dbg = out.get("debug") or {}
                level = ((dbg.get("answer") or {}).get("level", "")) or ""
            except Exception as e:  # noqa: BLE001
                answer, cites, level = f"ERR {e}", [], "err"
            u = llm.usage_stats()
            return {"answer": answer, "cites": cites, "level": level,
                    "calls": u["calls"], "prompt": u["prompt_tokens"]}

        ra = run("A")
        llm.reset_usage()
        sa = external_judge(q, ra["answer"], ra["cites"])
        ua = llm.usage_stats()
        rv = run("V")
        llm.reset_usage()
        sv = external_judge(q, rv["answer"], rv["cites"])
        uv = llm.usage_stats()
        recs.append({
            "qid": qid, "pid": pid, "official_pass": official_pass,
            "A": {"score": sa, "pass": sa >= PASS, "level": ra["level"], "calls": ra["calls"],
                  "prompt": ra["prompt"] + ua["prompt_tokens"]},
            "V": {"score": sv, "pass": sv >= PASS, "level": rv["level"], "calls": rv["calls"],
                  "prompt": rv["prompt"] + uv["prompt_tokens"]},
        })
        print(f"[{i}/{len(sel)}] {qid[:10]} A={sa}({ra['level']}) V={sv}({rv['level']}) "
              f"off={int(official_pass)}", flush=True)
        if i % 10 == 0 or i == len(sel):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)

    def agg(recs, key):
        return {"pass": sum(1 for r in recs if r[key]["pass"]),
                "prompt": sum(r[key]["prompt"] for r in recs) / len(recs),
                "calls": sum(r[key]["calls"] for r in recs) / len(recs)}

    a, v = agg(recs, "A"), agg(recs, "V")
    off = sum(1 for r in recs if r["official_pass"])
    aw = [r for r in recs if r["A"]["pass"] and not r["V"]["pass"]]
    vw = [r for r in recs if not r["A"]["pass"] and r["V"]["pass"]]
    print("=" * 60)
    print(f"定稿回归 n={len(recs)}：A(现状v2) pass {a['pass']} ({a['pass']/len(recs):.0%})  "
          f"V0(v3) pass {v['pass']} ({v['pass']/len(recs):.0%})  差 {v['pass']-a['pass']}")
    print(f"官方口径参照: 该子集 official pass {off} ({off/len(recs):.0%})")
    print(f"A过V挂 {len(aw)} / A挂V过 {len(vw)}")
    print(f"成本: A prompt {a['prompt']:.0f}/calls {a['calls']:.1f} | "
          f"V0 prompt {v['prompt']:.0f}/calls {v['calls']:.1f}")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
