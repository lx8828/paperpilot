"""R2 回归对比：`SYS_SET`（基线）vs `SYS_SET_EXTRACT`（带 extract 的提示词）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(0)

a = pd.read_csv(RES / "R2_r2reader_b6.csv").set_index(["cluster", "facet"])
b = pd.read_csv(RES / "R2_r2reader_extract.csv").set_index(["cluster", "facet"])
m = a.join(b, lsuffix="_base", rsuffix="_ext", how="inner")
n = len(m)

print("=" * 104)
print(f"【R2 回归】纯集合题（`extract` 在此**无用**）｜同真值 {n} 个")
print(f"  {'指标':<26}{'基线 SYS_SET':>16}{'SYS_SET_EXTRACT':>20}{'Δ':>10}")
for nm, c in (("集合 P", "reader_P"), ("集合 R", "reader_R"), ("集合 F1", "reader_F1")):
    x, y = m[f"{c}_base"].values, m[f"{c}_ext"].values
    d = y - x
    idx = rng.integers(0, len(d), size=(20000, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"  {nm:<26}{x.mean():>16.3f}{y.mean():>20.3f}{d.mean():>+10.3f}"
          f"   95%CI [{lo:+.3f}, {hi:+.3f}]")
print(f"  {'交付篇数':<26}{m['n_yes_base'].mean():>16.2f}{m['n_yes_ext'].mean():>20.2f}"
      f"{m['n_yes_ext'].mean() - m['n_yes_base'].mean():>+10.2f}")
print(f"  {'gold 篇数':<26}{m['n_gold_base'].mean():>16.2f}")
print(f"\n  → 若 Δ 的 95%CI 跨 0 → **不退化**，`extract` 可安全共用同一提示词。")

print("\n" + "=" * 104)
print("【生产 5 篇 M1 汇总】`extract` 补丁的边际效果")
print(f"  {'臂':<44}{'ok_strict':>12}")
for nm, v in (("fullctx 全上下文直读（存档）", 30),
              ("E extract 逐篇判定+逐篇内容（**新补**）", 17),
              ("S shipped 仅判定·无内容（原）", 7),
              ("Y 判定当路由 + yes 篇原文", 34),
              ("A 全部 5 篇原文倾倒（假阳性上界）", 44)):
    print(f"  {nm:<44}{f'{v}/50  {v / 50:.1%}':>12}")
print(f"\n  → `extract` 把 14% 抬到 **34%**，但**仍远低于 fullctx 60%**。")
print(f"  → 且 RAG-2 检索管线既有实测为 **20/50 = 40%** → 三条「成稿」路线都低于 fullctx。")
print(f"  → 结论：生产 5 篇上 **fullctx 仍是唯一正解**，判定线**接不上**。")
