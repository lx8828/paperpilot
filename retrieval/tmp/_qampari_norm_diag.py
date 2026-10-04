"""**零 LLM 诊断**：`passage_union` 多列出来的条目里，有多少是"同一实体的不同写法"？

如果 `coverage` 在"子串/包含匹配"口径下明显上升 → 说明多列的 2 条主要是**归并未做**，
而不是"列了错答案" → **归并规范化（零 LLM）就能救 `em`**。
"""
from __future__ import annotations

import importlib.util as _iu
import json
import re
import string
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parents[1]
spec = _iu.spec_from_file_location("qr", HERE / "tmp" / "_qampari_run.py")
qr = _iu.module_from_spec(spec)
sys.modules["qr"] = qr
spec.loader.exec_module(qr)

D = HERE / "data" / "loft" / "qampari" / "1m"
queries = {json.loads(l)["qid"]: json.loads(l)
           for l in (D / "test_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

ARMS = ["passage_union", "grp10", "passage_verified"]
od = HERE / "results" / "official_tier1m_mr20"
res = {}
for arm in ARMS:
    p = od / arm / "preds.jsonl"
    if not p.exists():
        continue
    res[arm] = {json.loads(l)["qid"]: json.loads(l)["model_outputs"][0]
                for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

print("=" * 104)
print("【strict（现口径：规范化后完全相等） vs loose（互为子串）】")
print(f"  {'臂':<18}{'strict cov':>12}{'loose cov':>11}{'Δ':>8}{'strict P':>10}{'loose P':>10}")
rows = []
for arm, preds in res.items():
    sc, lc, sp, lp = [], [], [], []
    for qid, pr in preds.items():
        gold = [qr.normalize_answer(x) for x in queries[qid]["answers"]]
        gs = set(gold)
        ps = [qr.normalize_answer(x) for x in pr]
        pss = set(ps)
        # strict
        tp = len(gs & pss)
        sc.append(tp / len(gs) if gs else np.nan)
        sp.append(tp / len(pss) if pss else 0.0)
        # loose：pred 与 gold 互为子串即算命中（与官方 subspan_em 的匹配规则一致）
        def hit(a, b):
            return (a in b) or (b in a)
        lhit_g = sum(1 for g in gs if any(hit(g, p) for p in pss))
        lhit_p = sum(1 for p in pss if any(hit(p, g) for g in gs))
        lc.append(lhit_g / len(gs) if gs else np.nan)
        lp.append(lhit_p / len(pss) if pss else 0.0)
    rows.append(dict(arm=arm, strict_cov=float(np.nanmean(sc)), loose_cov=float(np.nanmean(lc)),
                     strict_P=float(np.mean(sp)), loose_P=float(np.mean(lp)),
                     pred_mean=float(np.mean([len(v) for v in preds.values()]))))
d = pd.DataFrame(rows)
for _, r in d.iterrows():
    print(f"  {r['arm']:<18}{r['strict_cov']:>12.3f}{r['loose_cov']:>11.3f}"
          f"{r['loose_cov'] - r['strict_cov']:>+8.3f}{r['strict_P']:>10.3f}{r['loose_P']:>10.3f}")

print("\n【样例：strict 未命中、但 loose 命中的预测（= 同一实体的不同写法）】")
for arm in ARMS:
    if arm not in res:
        continue
    shown = 0
    for qid, pr in res[arm].items():
        gold = [qr.normalize_answer(x) for x in queries[qid]["answers"]]
        gs = set(gold)
        for x in pr:
            n = qr.normalize_answer(x)
            if n in gs:
                continue
            for g in gs:
                if (n in g) or (g in n):
                    print(f"  [{arm}] 预测 {x!r}  ←→  gold {g!r}（qid={qid[:26]}）")
                    shown += 1
                    break
            if shown >= 4:
                break
        if shown >= 4:
            break
d.to_csv(HERE / "results" / "R2_QAMPARI_NORM_DIAG.csv", index=False, encoding="utf-8")
print(f"\n已写 results/R2_QAMPARI_NORM_DIAG.csv")
