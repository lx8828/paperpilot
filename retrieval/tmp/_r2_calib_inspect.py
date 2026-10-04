"""抽查校准集原始行：分段长度不一致的问题（LLM 看 1200 字 vs NLI 看 1800 字）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parents[1]

df = pd.read_csv(HERE / "data" / "r2dev" / "nli_calibration_labeled.csv")
PAT = {"该论文做了消融实验": r"\bablation",
       "该论文公开了代码或数据": r"github\.com|code (?:is|will be) (?:publicly )?available|we (?:release|open[- ]source)"}

print(f"总行 {len(df)}｜证据长度: min {df['证据'].str.len().min()} / "
      f"中位 {int(df['证据'].str.len().median())} / max {df['证据'].str.len().max()}")

print("\n【逐行检查：证据里到底有没有那个锚点词】")
bad = 0
for i, r in df.iterrows():
    ev = str(r["证据"])
    claim = str(r["论断"])
    pat = next((v for k, v in PAT.items() if claim.startswith(k)), None)
    has = bool(re.search(pat, ev, re.I)) if pat else None
    tag = "有" if has else ("无" if has is False else "?")
    if has is False:
        bad += 1
    print(f"  {i:>2} b={r['引用块数']:>2} 构造={r['构造标签']:<3} 词面命中={str(r['词面命中']):<5} "
          f"证据里{tag}锚点 ｜ NLI_ent={r['NLI_entail']:.2f} con={r['NLI_contra']:.2f} "
          f"｜ LLM={r['LLM判定']:<10}")
print(f"  → 其中「构造=支持但证据里搜不到锚点」的行数：{bad}")

print("\n【完整看 3 行（构造=支持、LLM=neutral 的）】")
sub = df[(df["构造标签"] == "支持") & (df["LLM判定"] == "neutral")].head(3)
for _, r in sub.iterrows():
    print("=" * 96)
    print(f"论断：{r['论断']}   ｜ b={r['引用块数']} ｜ 词面命中={r['词面命中']}")
    print(f"NLI：entail={r['NLI_entail']:.3f} contra={r['NLI_contra']:.3f} ｜ "
          f"LLM：{r['LLM判定']}（{r['LLM理由']}）")
    print(f"证据（{len(str(r['证据']))} 字）：")
    print(str(r["证据"])[:1500])

print("\n【完整看 2 行（NLI 判矛盾、LLM 判 entail 的）】")
sub2 = df[(df["NLI_contra"] >= 0.5) & (df["LLM判定"] == "entail")].head(2)
for _, r in sub2.iterrows():
    print("=" * 96)
    print(f"论断：{r['论断']} ｜ b={r['引用块数']}")
    print(f"NLI：entail={r['NLI_entail']:.3f} contra={r['NLI_contra']:.3f} ｜ LLM：entail")
    print(f"证据（{len(str(r['证据']))} 字）：\n{str(r['证据'])[:1200]}")
