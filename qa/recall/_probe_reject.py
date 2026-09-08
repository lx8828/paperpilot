"""诊断探针：judge 双拒(自然&gold-first 都 enough=false)的题，跳过 judge 直接 answer。

若 answer(gold-first) 能 pass → judge 自我预测误杀，answer 层可以救（诊断1/3支撑）；
若仍 fail → judge 拒绝有据（材料不足，需要别的料）。
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


def norm(s):
    t = " ".join(str(s).split())
    t = _REF_RE.sub("", t)
    t = _FIG_RE.sub("", t)
    return " ".join(t.split())


def gold_chunk_id(chunks, evidence):
    ev = norm(evidence)
    ntexts = [norm(c.text) for c in chunks]
    for c, t in zip(chunks, ntexts):
        if ev in t:
            return c.chunk_id
    for seg in re.split(r"[.;:]\s|\n", evidence):
        seg = seg.strip()
        if len(seg) < 20:
            continue
        ns = norm(seg)
        for c, t in zip(chunks, ntexts):
            if ns in t:
                return c.chunk_id
    anchor = ev[:80]
    for c, t in zip(chunks, ntexts):
        if anchor in t:
            return c.chunk_id
    return None


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


def run_answer_only(q, entries, title):
    st = {"question": q.get("question", ""), "l3_chunks": entries,
          "route": [], "debug": {}, "title": title or ""}
    try:
        r = generate_answer(st)
        ans = r.get("answer") or ""
        cites = list(r.get("cites") or [])
    except Exception as e:  # noqa: BLE001
        return None, f"ERR {e}"
    sc = external_judge(q, ans, cites)
    return sc, ans[:260]


def main() -> int:
    llm._load_dotenv(str(ROOT))
    import glob
    order = sorted(glob.glob("qa/recall/order_ab_*.json"))[-1]
    recs = json.load(open(order, encoding="utf-8"))
    rejects = [r for r in recs if not r["natural"]["enough"] and not r["forced"]["enough"]]
    print(f"双拒题 {len(rejects)}")
    papers = load_papers()
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    out = []
    for i, r in enumerate(rejects, 1):
        qid = r["qid"]
        pid = next((p for p, pp in papers.items()
                    for qq in pp.get("qas") or []
                    if str(qq.get("question_id") or "") == qid), None)
        if not pid:
            continue
        q = next(qq for qq in papers[pid].get("qas") or []
                 if str(qq.get("question_id") or "") == qid)
        gold, ev = gold_answer(q)
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        gid = gold_chunk_id(chunks, ev)
        hits = ChunkIndex(pdf).search_hybrid(q.get("question", ""), top_k=12)
        ents = [{"kind": "chunk", "chunk_id": h.get("chunk_id"), "page": h.get("page"),
                 "section": str(list(h.get("title_path") or [])[-1] if h.get("title_path") else ""),
                 "text": h.get("text", "")} for h in hits]
        ge = next((e for e in ents if e["chunk_id"] == gid), None) if gid else None
        # 只跑 gold-first（能回答的"最好情况"）
        if ge is None:
            out.append({"qid": qid, "skip": "gold-not-in-top12-or-locate-fail", "score": None})
            print(f"[{i}] {qid[:10]} skip（gold 不在 top12/未定位）")
            continue
        rest = [e for e in ents if e["chunk_id"] != gid]
        sc, ans = run_answer_only(q, [ge] + rest, papers[pid].get("title"))
        out.append({"qid": qid, "score": sc, "pass": bool(sc and sc >= PASS), "answer": ans})
        print(f"[{i}] {qid[:10]} gold-first answer-only score={sc} "
              f"{'PASS' if sc and sc >= PASS else 'fail'}", flush=True)
        print(f"     A: {ans[:180].replace(chr(10),' ')}")
    Path("qa/recall/probe_reject_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    scored = [o for o in out if o.get("score") is not None]
    npass = sum(1 for o in scored if o["pass"])
    print(f"\nanswer-only(gold-first) 能 pass: {npass}/{len(scored)}"
          f"（judge 曾判不够）→ 若 >0 说明 judge 误杀、answer 可救")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
