"""Step1 A/B：现状 v2 全漏斗 vs v3 纯两级（L0 → L3）。

样本 = QASPER 全池重抽 100 题（report+gold 可定位，hard 源 40 + normal 源 60，
normal 含大量 L0/L1 直答 easy 题 → 测"砍 L1/L2 把 easy 推 L3"的损益）。
A = 现状图（qa_graph.ask，含 L0→L1→L2→L3）
V = v3 图（qa_graph_v3，L0→L3）
同题同进程、同外部裁判 glm-4-flash、记录 calls/prompt/level。
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
OUT = Path("qa/recall/ab_v3base_result.json")


def build_pool():
    papers = load_papers()
    final_lock = json.load(open("qa/qasper_final_lock_20260907_183225.json", encoding="utf-8"))
    hard_qids = {r["qid"] for r in final_lock if r.get("new_status") in ("fail", "unknown_ok")}
    view = ROOT / "assets/artifacts/out_views"
    index: dict[str, tuple[str, dict]] = {}
    for pid, p in papers.items():
        if not (view / f"qasper_{pid}.report.json").exists():
            continue
        for q in p.get("qas") or []:
            gold, ev = gold_answer(q)
            qid = str(q.get("question_id") or "")
            if gold and ev:
                index[qid] = (pid, q)
    return index, hard_qids


def sample(index, hard_qids, n_hard, n_normal, seed):
    hard_pool = [q for q in index if q in hard_qids]
    normal_pool = [q for q in index if q not in hard_qids]
    rng = random.Random(seed)
    rng.shuffle(hard_pool)
    rng.shuffle(normal_pool)
    per_pid: dict[str, int] = {}
    sel: list[str] = []
    cnt = {"hard": 0, "normal": 0}

    def take(qid):
        pid = index[qid][0]
        grp = "hard" if qid in hard_qids else "normal"
        if per_pid.get(pid, 0) >= 2:
            return False
        if (grp == "hard" and cnt["hard"] >= n_hard) or \
           (grp == "normal" and cnt["normal"] >= n_normal):
            return False
        per_pid[pid] = per_pid.get(pid, 0) + 1
        cnt[grp] += 1
        sel.append(qid)
        return True

    for q in hard_pool + normal_pool:
        if cnt["hard"] >= n_hard and cnt["normal"] >= n_normal:
            break
        take(q)
    return sel


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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    app3 = build_qa_graph_v3()
    papers = load_papers()
    index, hard_qids = build_pool()
    n_hard = args.n // 2
    n_normal = args.n - n_hard
    sel = sample(index, hard_qids, n_hard, n_normal, args.seed)
    if args.limit:
        sel = sel[: args.limit]

    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    recs = []
    t0 = time.time()
    for i, qid in enumerate(sel, 1):
        pid, q = index[qid]
        qq = q.get("question", "")
        pdf = f"qasper_{pid}.qpdf"

        def run(graph):
            llm.reset_usage()
            try:
                out = graph_ask(qq, pdf) if graph == "A" else app3.invoke({"question": qq, "pdf": pdf})
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
            "qid": qid, "pid": pid, "grp": "hard" if qid in hard_qids else "normal",
            "A": {"score": sa, "pass": sa >= PASS, "level": ra["level"], "calls": ra["calls"],
                  "prompt": ra["prompt"] + ua["prompt_tokens"]},
            "V": {"score": sv, "pass": sv >= PASS, "level": rv["level"], "calls": rv["calls"],
                  "prompt": rv["prompt"] + uv["prompt_tokens"]},
        })
        print(f"[{i}/{len(sel)}] {qid[:10]} A={sa}({ra['level']},{ra['calls']}c) "
              f"V={sv}({rv['level']},{rv['calls']}c)", flush=True)
        if i % 10 == 0 or i == len(sel):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    def agg(recs, key):
        return {"pass": sum(1 for r in recs if r[key]["pass"]),
                "prompt": sum(r[key]["prompt"] for r in recs) / len(recs),
                "calls": sum(r[key]["calls"] for r in recs) / len(recs)}

    print("=" * 60)
    for name, key in [("A(现状 v2 漏斗)", "A"), ("V(v3 纯两级)", "V")]:
        a = agg(recs, key)
        print(f"{name}: pass {a['pass']}/{len(recs)}  prompt/题 {a['prompt']:.0f}  calls/题 {a['calls']:.1f}")
    for g in ("hard", "normal"):
        sub = [r for r in recs if r["grp"] == g]
        if sub:
            a, v = agg(sub, "A"), agg(sub, "V")
            print(f"  {g}: A pass {a['pass']}/{len(sub)}  V pass {v['pass']}/{len(sub)}")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
