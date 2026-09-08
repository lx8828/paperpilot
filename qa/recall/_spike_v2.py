"""judge_v2 spike（最小闭环，不改生产代码）：两步法 judge vs 现状 L3 judge。

两臂同题跑：
  V1(现状)：judge_l3(自然 top12) enough → generate_answer；不够 → unknown(fail)
  V2(spike)：单次调用"起草→验证" → evidence_status
       complete/partial → draft 直接当答案（带 used chunk 作 cites）；unrelated → unknown(fail)
同一外部裁判(glm-4-flash)打 pass；比 pass、救回/回归、调用数。

用法：uv run python qa/recall/_spike_v2.py [--n-hard 50] [--n-unans 10]
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

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.agents.nodes.answer import generate_answer  # noqa: E402
from paperpilot.agents.nodes.judge import judge_l3  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4

_SYS_V2 = (
    "你是论文问答的起草-验证器。任务分两步，但只在一次回复中完成：\n"
    "第 1 步【起草】：通读全部检索段落，逐条抽出能回答用户问题的**候选事实**"
    "（数值/定义/断言/枚举），每一条必须来自某段原文，标注来源编号。\n"
    "第 2 步【验证】：逐个检查草稿里的论断是否**被对应原文支撑**、问题要求的要素是否都"
    "有着落。然后输出 JSON：\n"
    "{\n"
    '  "evidence_status": "complete | partial | unrelated",\n'
    '  "draft": "基于验证通过论断的完整回答（可直接作答，不要复述过程）",\n'
    '  "ev_used": [1,3],   // 实际支撑回答的段落编号（0 起，最多 6 条）\n'
    '  "gap": "一句话说明缺什么或为何无法作答"\n'
    "}\n"
    "判定规则：\n"
    "- 草稿所有论断都能被原文支撑且要素齐全 → complete；draft 就是最终答案。\n"
    "- 检索段落与问题是**同一对象/同一实验/同一主题**，但缺你要的某条具体信息"
    "（如未直接写出那个精确比较/具体数值/某个子项）→ **partial**：draft 给已能确定的结论，"
    "gap 说明缺什么。\n"
    "- 只有当检索段落**在讲别的对象/别的实验/与问题完全无关**，没有任何可作答实质内容 "
    "→ unrelated；此时 draft 必须留空字符串，不得编造。\n"
    "- 单要素题（只要一个数值/专名/定义）找到即可 complete，不要求额外列举。\n"
    "- 禁止用段落之外的任何知识补充。"
)

_V2_USER = """全文检索到的正文段落（独立检索）：
{chunks}

用户问题：{question}

请按上面的两步法起草并验证，然后输出 JSON。"""


def fmt_chunks(ents):
    parts = []
    for i, e in enumerate(ents):
        parts.append(f"[{i}] (p{e.get('page','?')} {e.get('section','')}) {e.get('text','')}")
    return "\n\n".join(parts)


def hits_to_ents(pdf, question, top_k=12):
    hits = ChunkIndex(pdf).search_hybrid(question, top_k=top_k)
    return [{"kind": "chunk", "chunk_id": h.get("chunk_id"), "page": h.get("page"),
             "section": str(list(h.get("title_path") or [])[-1] if h.get("title_path") else ""),
             "text": h.get("text", "")} for h in hits]


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


def arm_v1(q, pdf, ents, title):
    st = {"question": q.get("question", ""), "l3_chunks": ents, "route": [],
          "debug": {}, "title": title or ""}
    v = (judge_l3(st) or {}).get("verdict") or {}
    if not v.get("enough", False):
        return "unknown", None, 0
    r = generate_answer(st)
    ans = r.get("answer") or ""
    cites = list(r.get("cites") or [])
    return "answered", ans, external_judge(q, ans, cites)


def arm_v2(q, ents, title):
    user = _V2_USER.format(chunks=fmt_chunks(ents), question=q.get("question", ""))
    raw = llm.chat_json(_SYS_V2, user, temperature=0.0)
    if not isinstance(raw, dict):
        return "parse_err", None, 0, None, None
    status = str(raw.get("evidence_status", ""))
    draft = str(raw.get("draft", ""))
    used = raw.get("ev_used") or []
    gap = str(raw.get("gap", ""))[:160]
    if status == "unrelated" or not draft.strip():
        return status, None, 0, [], gap
    cites = [{"page": ents[i].get("page"), "evidence": (ents[i].get("text") or "")[:800]}
             for i in used if isinstance(i, int) and 0 <= i < len(ents)]
    sc = external_judge(q, draft, cites)
    return status, draft, sc, used, gap


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hard", type=int, default=50)
    ap.add_argument("--n-unans", type=int, default=10)
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    papers = load_papers()
    setd = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
    hard = [it for it in setd["items"] if it["group"] == "hard"][: args.n_hard]
    # 无答案池（从 hard 涉及的论文里找）
    unans_pool, seen = [], set()
    for it in hard:
        for qq in papers[it["pid"]].get("qas") or []:
            qid = str(qq.get("question_id") or "")
            if qid in seen:
                continue
            if not gold_answer(qq)[0]:
                seen.add(qid)
                unans_pool.append((it["pid"], qq))
            if len(unans_pool) >= args.n_unans:
                break
        if len(unans_pool) >= args.n_unans:
            break
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    t0 = time.time()
    recs = []
    print(f"hard={len(hard)} unans={len(unans_pool)}", flush=True)
    for i, it in enumerate(hard, 1):
        q = next(qq for qq in papers[it["pid"]].get("qas") or []
                 if str(qq.get("question_id") or "") == it["qid"])
        pdf = f"qasper_{it['pid']}.qpdf"
        ents = hits_to_ents(pdf, q.get("question", ""))
        llm.reset_usage()
        s1, a1, sc1 = arm_v1(q, pdf, ents, papers[it["pid"]].get("title"))
        u1 = llm.usage_stats()
        llm.reset_usage()
        s2, a2, sc2, used, gap = arm_v2(q, ents, papers[it["pid"]].get("title"))
        u2 = llm.usage_stats()
        recs.append({"qid": it["qid"], "kind": "hard",
                     "v1": {"state": s1, "score": sc1, "calls": u1["calls"]},
                     "v2": {"state": s2, "score": sc2, "calls": u2["calls"],
                            "gap": gap, "used": used}})
        if i % 10 == 0 or i == len(hard):
            print(f"[hard {i}/{len(hard)}] t={time.time()-t0:.0f}s", flush=True)
    for pid, q in unans_pool:
        pdf = f"qasper_{pid}.qpdf"
        ents = hits_to_ents(pdf, q.get("question", ""))
        llm.reset_usage()
        s1, a1, sc1 = arm_v1(q, pdf, ents, papers[pid].get("title"))
        u1 = llm.usage_stats()
        llm.reset_usage()
        s2, a2, sc2, used, gap = arm_v2(q, ents, papers[pid].get("title"))
        u2 = llm.usage_stats()
        recs.append({"qid": str(q.get("question_id", ""))[:14], "kind": "unans",
                     "v1": {"state": s1, "score": sc1, "calls": u1["calls"]},
                     "v2": {"state": s2, "score": sc2, "calls": u2["calls"],
                            "gap": gap, "used": used}})
    Path("qa/recall/spike_v2_result.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    hd = [r for r in recs if r["kind"] == "hard"]
    na = [r for r in recs if r["kind"] == "unans"]
    for name, grp in (("HARD", hd), ("UNANS", na)):
        p1 = sum(1 for r in grp if (r["v1"]["score"] or 0) >= PASS)
        p2 = sum(1 for r in grp if (r["v2"]["score"] or 0) >= PASS)
        rescue = sum(1 for r in grp if (r["v1"]["score"] or 0) < PASS and (r["v2"]["score"] or 0) >= PASS)
        regr = sum(1 for r in grp if (r["v1"]["score"] or 0) >= PASS and (r["v2"]["score"] or 0) < PASS)
        print(f"\n{name} n={len(grp)}: V1 pass {p1} | V2 pass {p2} | 救回 {rescue} | 回归 {regr}")
    st = {}
    for r in hd:
        st[r["v2"]["state"]] = st.get(r["v2"]["state"], 0) + 1
    print("V2 status(hard):", st)
    print("calls V1/V2 (hard avg):",
          round(sum(r["v1"]["calls"] for r in hd) / len(hd), 2),
          round(sum(r["v2"]["calls"] for r in hd) / len(hd), 2))
    print(f"t={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
