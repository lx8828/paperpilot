"""排序影响端到端对照（A/B）：gold 强制第1 vs 自然顺序。

对 hard 组中 hybrid 首gold 名次∈[2,12] 的题：
  - 自然臂：search_hybrid top12 原序喂 judge_l3 → enough 则 answer
  - 强推臂：gold chunk 移到第 1 位，其余照旧
比较两臂的 judge enough 率 与 外部裁判(glm, QASPER gold) pass 率。
判读：若强推显著提升 → MRR/精排值得投；若几乎无差 → MRR 不伤端到端，转查 judge 读块能力。
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

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.agents.nodes.answer import generate_answer  # noqa: E402
from paperpilot.agents.nodes.judge import judge_l3  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
_REF_RE = __import__("re").compile(r"\b[A-Z]+REF\d+\b")
_FIG_RE = __import__("re").compile(r"\bFIGURE\d+\b|\bTABLE\d+\b", __import__("re").I)

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
    """严格文本定位 → gold chunk_id（不用 fallback）。"""
    ev = norm(evidence)
    ntexts = [norm(c.text) for c in chunks]
    for c, t in zip(chunks, ntexts):
        if ev in t:
            return c.chunk_id
    for seg in __import__("re").split(r"[.;:]\s|\n", evidence):
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


def run_arm(q, entries, title):
    """返回 (enough, pass?, answer)。enough=False → 不算 pass。"""
    st = {"question": q.get("question", ""), "l3_chunks": entries,
          "route": [], "debug": {}, "title": title or ""}
    try:
        v = judge_l3(st) or {}
    except Exception as e:  # noqa: BLE001
        return False, False, f"ERR judge {e}"
    if not (v.get("verdict") or {}).get("enough", False):
        return False, False, None
    try:
        r = generate_answer(st)
        ans = r.get("answer") or ""
        cites = list(r.get("cites") or [])
    except Exception as e:  # noqa: BLE001
        return True, False, f"ERR answer {e}"
    sc = external_judge(q, ans, cites)
    return True, sc >= PASS, ans


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    papers = load_papers()
    items = [it for it in json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))["items"]
             if it["group"] == "hard"]
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    cands = []
    seen = set()
    for it in items:
        if it["qid"] in seen:
            continue
        seen.add(it["qid"])
        pid, qid = it["pid"], it["qid"]
        q = next((qq for qq in papers[pid].get("qas") or []
                  if str(qq.get("question_id") or "") == qid), None)
        if not q:
            continue
        _gold, ev = gold_answer(q)
        if not ev:
            continue
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        gid = gold_chunk_id(chunks, ev)
        if not gid:
            continue
        try:
            hits = ChunkIndex(pdf).search_hybrid(q.get("question", ""), top_k=12)
        except Exception:
            continue
        pos = next((i for i, h in enumerate(hits) if h.get("chunk_id") == gid), None)
        if pos is None or pos < 1:
            continue  # 只要 rank∈[2,12] 且能文本定位
        cands.append({"pid": pid, "qid": qid, "rank": pos + 1, "hits": hits,
                      "gid": gid, "q": q, "title": papers[pid].get("title")})
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    sel = cands[: args.n]
    print(f"候选 {len(cands)}，选 {len(sel)}（rank 分布: "
          f"{sorted([c['rank'] for c in sel][:8])}…）", flush=True)

    recs = []
    t0 = time.time()
    for i, c in enumerate(sel, 1):
        ent_n = [{"kind": "chunk", "chunk_id": h.get("chunk_id"), "page": h.get("page"),
                  "section": str((list(h.get("title_path") or [])[-1] if (h.get("title_path")) else "")),
                  "text": h.get("text", "")} for h in c["hits"]]
        gold_e = next(e for e in ent_n if e["chunk_id"] == c["gid"])
        rest = [e for e in ent_n if e["chunk_id"] != c["gid"]]
        ent_f = [gold_e] + rest
        llm.reset_usage()
        n_enough, n_pass, n_ans = run_arm(c["q"], ent_n, c["title"])
        u_n = llm.usage_stats()
        llm.reset_usage()
        f_enough, f_pass, f_ans = run_arm(c["q"], ent_f, c["title"])
        u_f = llm.usage_stats()
        recs.append({"qid": c["qid"], "rank": c["rank"], "qtype": "hard",
                     "natural": {"enough": n_enough, "pass": n_pass},
                     "forced": {"enough": f_enough, "pass": f_pass},
                     "gen": {"natural": u_n, "forced": u_f},
                     "n_answer": bool(n_ans), "f_answer": bool(f_ans)})
        print(f"[{i}/{len(sel)}] rank={c['rank']:2} natural(e={int(n_enough)},p={int(n_pass)})"
              f" forced(e={int(f_enough)},p={int(f_pass)})", flush=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / f"qa/recall/order_ab_{ts}.json"
    out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(recs)
    e_n = sum(1 for r in recs if r["natural"]["enough"])
    e_f = sum(1 for r in recs if r["forced"]["enough"])
    p_n = sum(1 for r in recs if r["natural"]["pass"])
    p_f = sum(1 for r in recs if r["forced"]["pass"])
    both_e = sum(1 for r in recs if r["natural"]["enough"] and r["forced"]["enough"])
    print(f"\n总 {n}：enough natural {e_n} vs forced {e_f} ｜ pass natural {p_n} vs forced {p_f}")
    print(f"两臂都 enough {both_e}｜ forced 提升enough {(e_f-e_n)} pass {(p_f-p_n)}")
    print(f"已写 {out}｜ t={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
