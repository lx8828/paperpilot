"""RAG-2 语料/切块规模实测（写文档用，别再靠记忆）。"""
from __future__ import annotations

import importlib.util as _iu
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)

CACHE = HERE / "data" / "r2dev" / "clusters"
print("=" * 100)
print("【RAG-2 开发线语料】LitSearch corpus_clean.full_paper ｜ 切块 = `chunks_of` 1000 字符 / 步长 900")
print(f"  {'簇':>3}{'篇数':>6}{'字符/篇(中位)':>15}{'总字符':>14}{'≈token':>11}{'块数':>8}{'块/篇':>8}")
tot_c = tot_k = 0
for ci, f in enumerate(sorted(CACHE.glob("c*.parquet"))):
    sub = pd.read_parquet(f)
    ln = sub["full_paper"].astype(str).str.len()
    nch = 0
    for t in sub["full_paper"].tolist():
        nch += len(_m.chunks_of(t))
    tot_c += int(ln.sum()); tot_k += nch
    print(f"  {ci + 1:>3}{len(sub):>6}{int(ln.median()):>15,}{int(ln.sum()):>14,}"
          f"{int(ln.sum() / 4.94):>11,}{nch:>8}{nch / len(sub):>8.1f}")
print(f"  {'合计':>3}{'':>6}{'':>15}{tot_c:>14,}{int(tot_c / 4.94):>11,}{tot_k:>8}")

print()
print("【RAG-2 生产线语料】assets/papers/*.pdf ｜ 切块 = `retrieval_chunks`（语义切块 ≤4000 字符 + MinerU 注入）")
ROOT = HERE.parent
SRC = ROOT / "retrieval" / "scripts" / "_qa_groups.py"
sys.path.insert(0, str(SRC.parent))
from _qa_groups import papers  # noqa: E402

try:
    from paperpilot.agents.document_cache import (MAX_CHUNK_LEN, ordered_chunks,  # noqa: E402
                                                 retrieval_chunks)
    from paperpilot.workflow import _ensure_env  # noqa: E402

    _ensure_env()
    print(f"  MAX_CHUNK_LEN = {MAX_CHUNK_LEN}（document_cache；与 pipeline/run_qa_eval 同口径）")
    print(f"  {'组':<9}{'篇数':>6}{'基础块':>9}{'检索块':>9}{'注入块':>9}{'字符(检索视图)':>17}{'均值块长':>10}")
    for g in [f"group{i}" for i in range(1, 6)]:
        ps = papers(g)
        nb = nv = ch = 0
        for stem in ps:
            pdf = f"{stem}.pdf"
            b, v = ordered_chunks(pdf), retrieval_chunks(pdf)
            nb += len(b); nv += len(v)
            ch += sum(len(c.text) for c in v)
        print(f"  {g:<9}{len(ps):>6}{nb:>9}{nv:>9}{nv - nb:>9}{ch:>17,}"
              f"{(ch / max(nv, 1)):>10.0f}")
except Exception as e:  # noqa: BLE001
    print(f"  ⚠️ 生产切块不可直接枚举（{type(e).__name__}: {e}）")
