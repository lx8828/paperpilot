"""两条测试集的语料/单元规模实测（写清单用，别靠记忆）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 108)
print("【测试集 A：QAMPARI（LoFT 官方）】语料单位 = 官方 passage（**不切块**）")
print(f"  {'档位':<8}{'passage 数':>12}{'字符/段(中位)':>16}{'总字符':>13}{'≈token/段':>12}{'总≈token':>12}")
for tier in ("32k", "128k", "1m"):
    f = HERE / "data" / "loft" / "qampari" / tier / "corpus.jsonl"
    if not f.exists():
        print(f"  {tier:<8}（缺 {f}）")
        continue
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    ln = np.array([len(str(r.get("title_text") or "")) + len(str(r.get("passage_text") or ""))
                   for r in rows])
    print(f"  {tier:<8}{len(rows):>12,}{int(np.median(ln)):>16,}{int(ln.sum()):>13,}"
          f"{np.median(ln) / 4.94:>12.0f}{ln.sum() / 4.94:>12,.0f}")
    q = f.parent / "test_queries.jsonl"
    if q.exists():
        qs = [json.loads(x) for x in q.read_text(encoding="utf-8").splitlines() if x.strip()]
        ng = np.array([len(r.get("answers") or []) for r in qs])
        print(f"  {'':<8}问题 {len(qs)} 条 ｜ gold 答案数 均 {ng.mean():.2f}（{ng.min()}~{ng.max()}）")

print()
print("=" * 108)
print("【测试集 B：20 篇论文（我们自己出题）】切块 = `chunks_of` 1000 字符 / 步长 900")
CACHE = HERE / "data" / "r2dev" / "clusters"
sys.path.insert(0, str(HERE / "scripts"))
import importlib.util as _iu  # noqa: E402

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)

print(f"  {'簇':>3}{'篇数':>6}{'字符/篇(中位)':>15}{'块数':>8}{'块/篇':>8}  块长(字符) 步长")
tot = 0
for ci, f in enumerate(sorted(CACHE.glob("c*.parquet"))):
    sub = pd.read_parquet(f)
    ln = sub["full_paper"].astype(str).str.len()
    nch = sum(len(_m.chunks_of(t)) for t in sub["full_paper"].tolist())
    tot += nch
    print(f"  {ci + 1:>3}{len(sub):>6}{int(ln.median()):>15,}{nch:>8}{nch / len(sub):>8.1f}"
          f"{'1000':>10}{'900':>6}")
print(f"  {'合计':>3}{'':>6}{'':>15}{tot:>8}")
g = pd.read_csv(HERE / "data" / "r2dev" / "gold_recalib.csv")
g2 = pd.read_csv(HERE / "data" / "r2dev" / "gold_recalib_rich.csv")
print(f"  真值：gold_recalib {len(g)} 行（{g.groupby(['cluster','facet']).ngroups} 个真值）"
      f" ｜ gold_recalib_rich {len(g2)} 行（**争议子集**，覆盖用）")
print(f"  子查询：{len(json.loads((HERE / 'data/r2dev/subqueries.json').read_text(encoding='utf-8')))} 个 facet 有子查询")
