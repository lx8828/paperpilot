"""A/B：answer-before-unknown —— judge 判不够时是否该先试答一次。

主测：recall hard 100 题，judge_l3(natural top12) 判不够 → 直接用自然 top12 answer
      （无 oracle 前置），外部裁判(glm) pass = "救回"。
风险组：10 条无答案题，若 answer 硬答造成编造 → 从"诚实拒答(pass)"掉成 fail，量化风险。
"""
from __future__ import annotations
import io
import json
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.agents.nodes.answer import generate_answer  # noqa: E402
from paperpilot.agents.nodes.judge import judge_l3  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
_REF_RE = re.compile(r"\b[A-Z]+REF\d+\b")
_FIG_RE = re.compile(r"\bFIGURE\d+\b|\bTABLE\d+\b", re.I)

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
    user = _JUDGE_USER.format(question=q.get("question", ""), gold=gold or "（该题在论文中无答案，系统应诚实说明）",
                              evidence=(ev or ""), answer=answer[:1500], cites=ct)
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        return int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
    except Exception:
        return 0


def judge_natural(q, pdf, title):
    hits = ChunkIndex(pdf).search_hybrid(q.get("question", ""), top_k=12)
    ents = [{"kind": "chunk", "chunk_id": h.get("chunk_id"), "page": h.get("page"),
             "section": str(list(h.get("title_path") or [])[-1] if h.get("title_path") else ""),
             "text": h.get("text", "")} for h in hits]
    st = {"question": q.get("question", ""), "l3_chunks": ents, "route": [], "debug": {}, "title": title or ""}
    v = (judge_l3(st) or {}).get("verdict") or {}
    return v.get("enough", False), ents


def answer_only(q, ents, title):
    st = {"question": q.get("question", ""), "l3_chunks": ents, "route": [], "debug": {}, "title": title or ""}
    try:
        r = generate_answer(st)
        return r.get("answer") or "", list(r.get("cites") or [])
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}", []


def main() -> int:
    llm._load_dotenv(str(ROOT))
    papers = load_papers()
    setd = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
    hard = [it for it in setd["items"] if it["group"] == "hard"]
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    t0 = time.time()
    res_main = []
    n_ne = 0
    n_rescued = 0
    for i, it in enumerate(hard, 1):
        q = next((qq for qq in papers[it["pid"]].get("qas") or []
                  if str(qq.get("question_id") or "") == it["qid"]), None)
        if not q:
            continue
        pdf = f"qasper_{it['pid']}.qpdf"
        enough, ents = judge_natural(q, pdf, papers[it["pid"]].get("title"))
        if enough:
            continue  # 现状已作答；本实验只管 not-enough 分支
        n_ne += 1
        ans, cites = answer_only(q, ents, papers[it["pid"]].get("title"))
        sc = external_judge(q, ans, cites)
        ok = sc >= PASS
        n_rescued += int(ok)
        res_main.append({"qid": it["qid"], "judge_ne": True, "score": sc, "pass": ok,
                         "answer": ans[:200]})
        print(f"[hard {n_ne}/{i}] qid {it['qid'][:10]} not-enough → answer score={sc} "
              f"{'RESCUE' if ok else 'fail'}", flush=True)
    # 风险组：无答案题
    unans_pool = []
    seen = set()
    for it in hard:
        for qq in papers[it["pid"]].get("qas") or []:
            qid = str(qq.get("question_id") or "")
            if qid in seen:
                continue
            gold, _ = gold_answer(qq)
            if not gold:
                seen.add(qid)
                unans_pool.append((it["pid"], qq))
            if len(unans_pool) >= 10:
                break
        if len(unans_pool) >= 10:
            break
    res_risk = []
    n_risk_honest = 0
    for pid, q in unans_pool:
        pdf = f"qasper_{pid}.qpdf"
        enough, ents = judge_natural(q, pdf, papers[pid].get("title"))
        if enough:
            continue
        ans, cites = answer_only(q, ents, papers[pid].get("title"))
        sc = external_judge(q, ans, cites)
        ok = sc >= PASS
        n_risk_honest += int(ok)
        res_risk.append({"qid": str(q.get("question_id", ""))[:12], "score": sc, "pass": ok,
                         "answer": ans[:160]})
        print(f"[unans] qid {str(q.get('question_id',''))[:10]} answer score={sc} "
              f"{'pass' if ok else 'fail(可能编造)'}", flush=True)
    out = {"main": {"n_not_enough": n_ne, "rescued": n_rescued,
                    "rescue_rate": round(n_rescued / max(n_ne, 1), 3),
                    "detail": res_main},
           "risk": {"n": len(res_risk), "honest_pass": n_risk_honest, "detail": res_risk},
           "t_s": round(time.time() - t0, 1)}
    Path("qa/recall/ab_answer_unknown_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nHARD: judge not-enough {n_ne} → answer 救回 {n_rescued}（{n_rescued/max(n_ne,1):.0%}）")
    print(f"RISK 无答案: {len(res_risk)} 题 answer-only 诚实通过 {n_risk_honest}"
          f"（若明显低于拒答线→有编造风险）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
