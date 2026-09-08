"""L1 judge 专项诊断：判"够"的假阳率 + target_sections 命中答案节的比例。

对 recall hard 前 N 题：
  1) retrieve_claims(L1 检索) → judge_l1 → enough?
     enough → generate_answer(L1) + 外部裁判 → 判够但答不对 = 假阳（过早收口）
  2) not enough 时，gold 真实所在节是否 ∈ judge 给的 target_sections(top≤2) = 节引导命中率
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

from paperpilot.agents.document_cache import (ordered_chunks,  # noqa: E402
                                              section_from_path)
from paperpilot.agents.nodes.answer import generate_answer  # noqa: E402
from paperpilot.agents.nodes.judge import judge_l1  # noqa: E402
from paperpilot.agents.nodes.retrieve import retrieve_claims  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
_REF_RE = re.compile(r"\b[A-Z]+REF\d+\b")
_FIG_RE = re.compile(r"\bFIGURE\d+\b|\bTABLE\d+\b", re.I)


def norm(s):
    t = " ".join(str(s).split())
    t = _REF_RE.sub("", t)
    t = _FIG_RE.sub("", t)
    return " ".join(t.split())


def gold_chunk_top(chunks, evidence):
    ev = norm(evidence)
    ntexts = [norm(c.text) for c in chunks]
    for c, t in zip(chunks, ntexts):
        if ev in t:
            return c
    for seg in re.split(r"[.;:]\s|\n", evidence):
        seg = seg.strip()
        if len(seg) < 20:
            continue
        ns = norm(seg)
        for c, t in zip(chunks, ntexts):
            if ns in t:
                return c
    anchor = ev[:80]
    for c, t in zip(chunks, ntexts):
        if anchor in t:
            return c
    return None


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
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    papers = load_papers()
    hard = [it for it in json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))["items"]
            if it["group"] == "hard"][: args.n]
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    t0 = time.time()
    recs = []
    for i, it in enumerate(hard, 1):
        q = next(qq for qq in papers[it["pid"]].get("qas") or []
                 if str(qq.get("question_id") or "") == it["qid"])
        pdf = f"qasper_{it['pid']}.qpdf"
        gold, ev = gold_answer(q)
        chunks = ordered_chunks(pdf)
        gc = gold_chunk_top(chunks, ev)
        gold_top = section_from_path(list(gc.title_path)) if gc else None
        overview = ""
        ovf = Path(f"out_views/{Path(pdf).stem}.overview.json")
        if ovf.exists():
            try:
                overview = json.load(open(ovf, encoding="utf-8")).get("overview", "") or ""
            except Exception:
                pass
        # L1 检索 + judge
        r1 = retrieve_claims({"pdf": pdf, "question": q.get("question", "")})
        retrieved = r1.get("retrieved") or []
        st = {"question": q.get("question", ""), "overview": overview,
              "retrieved": retrieved, "route": [], "debug": {}, "title": papers[it["pid"]].get("title")}
        v = (judge_l1(st) or {}).get("verdict") or {}
        enough = bool(v.get("enough", False))
        targets = list(v.get("target_sections") or [])
        rec = {"qid": it["qid"], "enough": enough, "targets": targets,
               "gold_top": gold_top, "gap": (v.get("gap") or "")[:120]}
        if enough:
            try:
                r = generate_answer(st)
                ans = r.get("answer") or ""
                cites = list(r.get("cites") or [])
            except Exception as e:  # noqa: BLE001
                ans, cites = f"ERR {e}", []
            sc = external_judge(q, ans, cites)
            rec.update({"score": sc, "pass": sc >= PASS, "answer": ans[:120]})
        recs.append(rec)
        print(f"[{i}] enough={int(enough)} targets={targets[:2]} gold_top={gold_top} "
              f"{'score=' + str(rec.get('score')) if enough else '(下钻)'}", flush=True)
        if i % 10 == 0:
            print(f"   …t={time.time()-t0:.0f}s", flush=True)
    Path("qa/recall/diag_l1_result.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    ne = [r for r in recs if r["enough"]]
    nn = [r for r in recs if not r["enough"]]
    p = sum(1 for r in ne if r.get("pass"))
    print(f"\nL1 判够 {len(ne)}/{len(recs)} → L1 answer pass {p}（判够的假阳={len(ne)-p}）")
    # target 命中（gold_top 已知且 not enough）
    ok = [r for r in nn if r["gold_top"]]
    hit = sum(1 for r in ok if r["gold_top"] in r["targets"])
    print(f"判不够 {len(nn)}，其中可定位 gold 节 {len(ok)}，"
          f"target_sections(top{2}) 命中 gold 节 {hit}（{hit/max(len(ok),1):.0%}）")
    print(f"t={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
