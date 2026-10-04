"""**S/C 分流结论**：汇总两条实验 → 定「检索线的价值起点」。

产出：
  1. 实验 1 的**配对显著性**（整档直读 vs 逐篇判定，同一批 30 真值）
  2. S/C 阶梯 × 实测判读（含新测点）
  3. 生产 5 篇（S/C≈0.4）上各臂对照（含 2b 的度量陷阱）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "retrieval" / "results"
RUNS = ROOT / "qa" / "multi" / "_runs"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(0)


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 20000) -> tuple[float, float, float]:
    """配对 bootstrap：Δ 均值、95% CI、P(Δ>0)。"""
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs > 0).mean())


print("=" * 112)
print("【实验 1｜配对检验】R2 语料 20 篇/簇 ≈266k token（S/C ≈ 2）")
f = pd.read_csv(RES / "R2_r2fullread.csv")
r = pd.read_csv(RES / "R2_r2reader_b6.csv")[["cluster", "facet", "reader_F1", "ret_F1",
                                              "reader_P", "reader_R", "ret_P", "ret_R"]]
m = f.merge(r, on=["cluster", "facet"])
n = len(m)
print(f"  同真值配对样本 n={n}\n")
print(f"  {'臂':<30}{'P':>8}{'R':>8}{'F1':>8}")
print(f"  {'A 整档直读（20 篇全文·266k tok）':<30}{m.P.mean():>8.3f}{m.R.mean():>8.3f}{m.F1.mean():>8.3f}")
print(f"  {'B 逐篇局部检索+判定（≈24k tok）':<30}{m.reader_P.mean():>8.3f}"
      f"{m.reader_R.mean():>8.3f}{m.reader_F1.mean():>8.3f}")
print(f"  {'C 篇级检索 top-10（无判定）':<30}{m.ret_P.mean():>8.3f}"
      f"{m.ret_R.mean():>8.3f}{m.ret_F1.mean():>8.3f}")
for nm, x in (("B − A（局部检索+判定的增益）", m.reader_F1 - m.F1),
              ("B − C（判定层的增益）", m.reader_F1 - m.ret_F1),
              ("A − C（整档相对篇级检索）", m.F1 - m.ret_F1)):
    mu, lo, hi, p = paired(x.values, np.zeros(len(x)))
    mark = "**显著**" if lo > 0 else ("不显著" if hi > 0 else "显著为负")
    print(f"\n  Δ {nm:<34} {mu:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  P(Δ>0)={p:.3f}  → {mark}")
print(f"\n  成本：整档直读 {f.prompt_tokens.mean():,.0f} tok/题（缓存命中 "
      f"{f.cache_hit.sum() / f.prompt_tokens.sum():.0%}）｜ 逐篇判定 ≈6 块×20 篇 ≈ 24k tok/题")
print(f"  注意力：整档直读 touched = {f.n_touched.mean():.1f}/20（答案显式提到【Pn】的篇数）")
print(f"  交付偏多：git直读 交付 {f.n_yes.mean():.2f} vs gold {f.n_gold.mean():.2f}"
      f"（比值 {f.n_yes.mean() / f.n_gold.mean():.2f}）")

print("\n" + "=" * 112)
print("【实验 2｜生产 5 篇（S/C ≈ 0.4）M1 50 题】")
fc = {}
for p in sorted(RUNS.glob("fullctx_m1all_group*.json")):
    for x in json.loads(p.read_text(encoding="utf-8")):
        fc[str(x.get("qid"))] = x
d2 = {}
for p in sorted(RUNS.glob("setdiag2_group*.json")):
    for x in json.loads(p.read_text(encoding="utf-8")):
        d2[str(x.get("qid"))] = x          # ⚠️ `_m1_set_diag2.py` 是**累积写入**（文件N 含前 N 组）
d2 = list(d2.values())                     #    → 必须按 qid 去重，否则重复计数 3×
sh = {}
for p in sorted(RUNS.glob("set_m1all_group*.json")):
    for x in json.loads(p.read_text(encoding="utf-8")):
        sh[str(x.get("qid"))] = x
qn = len(d2)
fc_ok = sum(1 for v in fc.values() if v.get("ok"))
fc_st = sum(1 for v in fc.values() if v.get("ok_strict"))
sh_ok = sum(1 for v in sh.values() if v.get("ok"))
sh_st = sum(1 for v in sh.values() if v.get("ok_strict"))
print(f"  {'臂':<44}{'ok':>9}{'ok_strict':>13}")
print(f"  {'fullctx 全上下文直读（1 次调用·全文 47k tok）':<44}"
      f"{f'{fc_ok}/{len(fc)}':>9}{f'{fc_st}/{len(fc)}  {fc_st / len(fc):.1%}':>13}")
print(f"  {'set shipped 篇名+理由（现行 set 产品形态）':<44}"
      f"{f'{sh_ok}/{len(sh)}':>9}{f'{sh_st}/{len(sh)}  {sh_st / len(sh):.1%}':>13}")
for k, nm in (("S_shipped", "S shipped = 同上（2b 复现）"),
              ("Y_route_yes", "Y 判定当路由 + yes 篇原文"),
              ("A_route_all", "A 全部 5 篇原文倾倒（**不靠判定**）")):
    st = sum(1 for x in d2 if x.get(f"{k}_strict"))
    ok = sum(1 for x in d2 if x.get(f"{k}_ok"))
    print(f"  {nm:<44}{f'{ok}/{qn}':>9}{f'{st}/{qn}  {st / qn:.1%}':>13}")
print(f"\n  ⚠️ **A 臂 = 88% 是假阳性上界**：把检索块原文倒进答案，锚点必然出现。")
print(f"     → `ok_strict` 在 M1 上是**信息出现率**，不是答案质量；**不能用来比较交付形态**。")
print(f"     → 但 fullctx(60%) 是诚实的（LLM 成稿，不倾倒）→ 该数可用。")

print("\n" + "=" * 112)
print("【S/C 阶梯 · 检索线的价值起点】")
rows = [
    ("生产 5 篇", 48_000, 0.37, "fullctx 30/50 **>** 检索线 20/50（read_full.py 既有实测）", "❌ 检索不值"),
    ("QAMPARI 128k", 105_322, 0.80, "直读 0.700 == 检索 0.700（既有实测）", "➖ 打平"),
    ("R2 20 篇/簇", 266_423, 2.03, "整档直读 0.773 **<** 局部检索+判定 0.817，且省 11× token", "✅ **检索值**"),
    ("QAMPARI 1m", 844_484, 6.44, "直读崩（24% 上下文只拿 3/5）", "✅✅ 必需"),
]
print(f"  {'设定':<16}{'≈token':>10}{'S/C':>8}  {'实测':<60}{'判定'}")
for nm, tk, sc, ev, v in rows:
    print(f"  {nm:<16}{tk:>10,}{sc:>8.2f}  {ev:<60}{v}")
print(f"\n  → **价值起点落在 S/C ≈ 0.8 与 2.0 之间**（不是 S/C=1）。")
print(f"  → 生产 5 篇（0.37）在价值区**之外** ⇒ 检索线**该独立**；")
print(f"     但独立理由不是「塞不下」，而是「语料大到逐篇局部化才划算」。")
