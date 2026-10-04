"""**A1c｜生成最终真值**（生产切块口径 · 双 LLM 重标 + 剔除无分辨率题）

规则：
· 正例 = `gold_recalib2.csv` 的 `new_gold`
· **剔除**（无分辨率，无法算 P/R/F1）：
    - 正例数 = 0（判官全否 → 该题在语料上无真值）
    - 正例数 = 篇数（全正 → 同上）
· 只保留有 PDF 的 47 篇

产物：
  `data/r2dev/gold_final.csv`       (cluster, facet, docid, gold)
  `data/r2dev/gold_final_meta.json`  {usable, dropped, per_facet, n_docs}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

df = pd.read_csv(DEV / "gold_recalib2.csv")
df["anchor_gold"] = df["anchor_gold"].astype(bool)
df["new_gold"] = df["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
pmap = json.loads((DEV / "pdf_map.json").read_text(encoding="utf-8"))
keep = {d for d, v in pmap.items() if v["ok"]}
df = df[df.docid.isin(keep)].copy()

usable, dropped = [], []
rows = []
for (c, f), g in df.groupby(["cluster", "facet"]):
    n, npos = len(g), int(g.new_gold.sum())
    if npos == 0 or npos == n:
        dropped.append(dict(cluster=int(c), facet=str(f), n_paper=n, n_pos=npos,
                            why="无正例" if npos == 0 else "全正例",
                            n_anchor=int(g.anchor_gold.sum())))
        continue
    usable.append((int(c), str(f)))
    for _, r in g.iterrows():
        rows.append(dict(cluster=int(c), facet=str(f), docid=str(r.docid),
                         gold=bool(r.new_gold), anchor_gold=bool(r.anchor_gold)))

out = pd.DataFrame(rows).sort_values(["cluster", "facet", "docid"])
out.to_csv(DEV / "gold_final.csv", index=False, encoding="utf-8-sig")
meta = {
    "source": "gold_recalib2.csv（双 LLM 异源 + 第三轮对抗；生产章节段落切块口径）",
    "chunk_source": "prodchunk/mineru（MinerU 骨架，USE_MINERU=1）",
    "n_pairs": len(out),
    "n_usable_facets": len(usable),
    "usable": [f"{c}|{f}" for c, f in usable],
    "dropped": dropped,
    "per_facet": {f"{c}|{f}": int(g.gold.sum())
                  for (c, f), g in out.groupby(["cluster", "facet"])},
}
(DEV / "gold_final_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                          encoding="utf-8")

print("=" * 100)
print(f"【最终真值】配对 {len(out)} 条 ｜ 可用 facet **{len(usable)}**（原 35）")
print(f"  正例合计 {int(out.gold.sum())} ｜ 锚点正例 {int(out.anchor_gold.sum())}"
      f" ｜ 每篇被判正例率 {out.gold.mean():.1%}")
print(f"\n  【剔除 {len(dropped)} 个 facet】（无分辨率）")
print(f"  {'簇':>3}{'facet':<18}{'篇数':>5}{'正例':>5}{'锚点正例':>9}  原因")
for d in dropped:
    print(f"  {d['cluster']:>3}{d['facet']:<18}{d['n_paper']:>5}{d['n_pos']:>5}"
          f"{d['n_anchor']:>9}  {d['why']}  ← 锚点有 {d['n_anchor']} 个假阳性'撑起'了这道题")
print(f"\n  【保留 facet 的正例数】（均 {out.groupby(['cluster', 'facet']).gold.sum().mean():.2f} 篇/题）")
for (c, f), g in out.groupby(["cluster", "facet"]):
    print(f"    簇{c} {f:<18} {int(g.gold.sum()):>2}/{len(g)} 篇正例"
          f"（锚点 {int(g.anchor_gold.sum())}）")
print(f"\n→ {DEV / 'gold_final.csv'} ｜ {DEV / 'gold_final_meta.json'}")
