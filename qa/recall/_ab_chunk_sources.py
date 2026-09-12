"""临时：两源 chunk（pymupdf vs MinerU）对检索影响的离线探针。

对 HyGRAIL 题集中需检索的题（expect 含 L3），用 BM25 top-12 在两源 chunks 上检索，
检查 must_have/must_all 关键词所在证据是否进入检索窗（近似 recall 探针，0 token）。
"""
from __future__ import annotations
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents import document_cache as dc  # noqa: E402
from paperpilot.agents.embedder import BM25Index  # noqa: E402

QUESTIONS = json.loads(Path("qa/questions/2609.02056v1.json").read_text(encoding="utf-8"))
PDF = "2609.02056v1.pdf"

# 需检索的题：expect 含 L3（深入/全局检索），排除 overview 直答
sel = [q for q in QUESTIONS if "L3" in (q.get("expect") or []) or "L2" in (q.get("expect") or [])]
print(f"题集总数 {len(QUESTIONS)}，选需检索题 {len(sel)}")


def kw(q) -> list[str]:
    out = list(q.get("must_have") or []) + list(q.get("must_all") or [])
    return [k for k in out if len(k) >= 2][:5]


def docs_for(env: str):
    os.environ["PAPERPILOT_USE_MINERU"] = env
    dc.ordered_chunks.cache_clear()
    dc.current_source.cache_clear()
    cs = dc.ordered_chunks(PDF)
    return [c.text for c in cs]


pym = docs_for("0")
minu = docs_for("1")
print(f"pymupdf chunks={len(pym)}  mineru chunks={len(minu)}")

idx_p = BM25Index(pym)
idx_m = BM25Index(minu)

hit_p_all = hit_m_all = 0
print("\n逐题 top12 证据命中（must 词是否在检索窗内）：")
for q in sel:
    qq = q.get("question", "")
    kws = kw(q)
    sp = idx_p.score(qq)
    sm = idx_m.score(qq)
    def top(idx, sc, n=12):
        return sorted(range(len(sc)), key=lambda i: -sc[i])[:n]
    tp = " ".join(pym[i] for i in top(idx_p, sp)).lower()
    tm = " ".join(minu[i] for i in top(idx_m, sm)).lower()
    hit_p = sum(1 for k in kws if k.lower() in tp)
    hit_m = sum(1 for k in kws if k.lower() in tm)
    hit_p_all += hit_p == len(kws)
    hit_m_all += hit_m == len(kws)
    tag = "=" if hit_p == hit_m else ("pymupdf优" if hit_p > hit_m else "mineru优")
    print(f"  {q.get('qid')} [{q.get('intent')}] must词 {hit_p}/{len(kws)} vs {hit_m}/{len(kws)} {tag}")
    if hit_p != hit_m:
        for k in kws:
            print(f"      词 {k!r}: pym={k.lower() in tp}  mineru={k.lower() in tm}")

print(f"\n全命中率（所有 must 词都进 top12）: pymupdf {hit_p_all}/{len(sel)}  mineru {hit_m_all}/{len(sel)}")
