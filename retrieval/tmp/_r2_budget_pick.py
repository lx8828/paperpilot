"""**挑一个 demo 配置**：(单题预算 X, 语料规模 N) → 检索必要性 / 机器时间 / setF1。

需求（用户原话）：检索有必要，N 不能太大，只能靠降单题预算，找一个最合理的点，demo 级即可。

## 复用而不是重写
直接 import `_r2_budget_model` 的 `load_facets` / `ceil_sim` / `P_JUDGE` / `C_JUDGE`
（判据分两处必然漂移）。`ρ` 与 `MACRO`（Jensen 差）在主模型里是在 `main()` 内就地
实测的、没抽成函数，故此处**逐行照抄**其算法 —— 并用**自检**保证没抄错：
先复现报告里已公布的点（表 B `X=1.0, N=1000 → setF1 0.6427`）。

## ★ 两个必须说清的（否则挑点会挑错）
1. **机器时间 ∝ k，与 N 无关**（主模型：`判官秒 = k×0.8/8`）。
   N 大**只是让检索更难/外推更远**，**不会让机器变慢**。
2. 所以"N 不能太大"的**真实约束是外推可信度**（新干扰按现有 50 篇的得分分布采样，
   N 越大外推越远），而不是速度。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tmp"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("budget_model", HERE / "tmp" / "_r2_budget_model.py")

F = M.load_facets()
C, PJ = M.C_JUDGE, M.P_JUDGE
D = pd.read_csv(HERE / "results" / "R2_PROD_FINAL_DELIV.csv")
D["in_gold"] = D["in_gold"].astype(int)
D["label"] = D["label"].fillna("").astype(str).str.strip().str.lower()

# ── ρ(k/N)：与 `_r2_budget_model.main()` 同源（照抄）──
fx, fy = [], []
for kk in [5, 10, 15, 20, 25, 30, 40, 50]:
    ce = M.ceil_sim(F, 50, kk, 1)
    rrs = []
    for _, x in D.groupby(["cluster", "facet"]):
        g = set(x[x.in_gold == 1]["docid"])
        kx = x[x["rank"] <= kk]
        rrs.append(len(set(kx[kx.label == "yes"]["docid"]) & g) / max(len(g), 1))
    fx.append(kk / 50)
    fy.append(float(np.mean(rrs)) / ce)


def rho(fr: float) -> float:
    return float(np.interp(fr, fx, fy))


# ── MACRO：Jensen 口径差（照抄）──
f0 = pd.read_csv(HERE / "results" / "R2_PROD_FINAL.csv")
f0 = f0[f0.k == 0]
F1H = 2 * f0.setP.mean() * f0.setR.mean() / (f0.setP.mean() + f0.setR.mean())
MACRO = float(F1H - f0.setF1.mean())


def setF1(R: float) -> float:
    """端到端 setF1（扣 Jensen 差，与定稿报数同口径）。"""
    h = 2 * PJ * R / (PJ + R) if (PJ + R) else 0.0
    return h - MACRO


def m_of(X: float) -> int:
    return max(0, int((X - 2 * C) // C))


B = 120

# ── 自检：必须复现报告里的点 ──
print("【自检】复现报告已公布的点（对不上则下面的表不可信）")
for Xa, Na, want in ((1.0, 1000, 0.6427), (0.5, 1000, 0.5360), (1.0, 2000, 0.5410)):
    k = min(m_of(Xa), Na)
    ce = M.ceil_sim(F, Na, k, B)
    got = setF1(ce * rho(k / Na))
    flag = "✓" if abs(got - want) < 0.01 else "✗ 偏了"
    print(f"   X=¥{Xa:<5} N={Na:<7,} 报告 {want:.4f}  本脚本 {got:.4f}  {flag}")

# ── 网格 ──
XS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
NS = [100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1500, 2000]

rows = []
for X in XS:
    m = m_of(X)
    for N in NS:
        if N < 51:
            continue
        k = min(m, N)
        if k < 1:
            continue
        ce = M.ceil_sim(F, N, k, B)
        fr = k / N
        R = ce * rho(fr)
        rows.append(dict(X=X, m=m, N=N, k=k, fr=fr, need=1 - fr, ceil=ce,
                         rho=rho(fr), R=R, setF1=setF1(R), sec=k * 0.8 / 8))
T = pd.DataFrame(rows)

print("\n" + "=" * 104)
print("【全网格】need=检索筛掉的比例（=1−k/N）｜ sec=判官墙钟秒（k×0.8/8）")
print("=" * 104)
print(f"  {'X(元)':>6}{'m':>7}{'N':>8}{'k':>7}{'k/N':>7}{'need':>7}{'ceil':>8}"
      f"{'ρ':>7}{'setF1':>8}{'sec':>7}")
for X in XS:
    sub = T[T.X == X]
    if not len(sub):
        continue
    for _, r in sub.iterrows():
        print(f"  {r.X:>6.2f}{r.m:>7,}{r.N:>8,}{r.k:>7,}{r.fr:>7.1%}{r.need:>7.1%}"
              f"{r.ceil:>8.4f}{r.rho:>7.3f}{r.setF1:>8.4f}{r.sec:>7.1f}")
    print()

# ── 帕累托前沿（三目标：setF1↑ / sec↓ / need↑）──
def dominated(a, b) -> bool:
    """b 支配 a：三目标都不差且至少一个更好。"""
    return (b.setF1 >= a.setF1 - 1e-9 and b.sec <= a.sec + 1e-9
            and b.need >= a.need - 1e-9
            and (b.setF1 > a.setF1 + 1e-4 or b.sec < a.sec - 1e-9
                 or b.need > a.need + 1e-4))


front = [r for _, r in T.iterrows()
         if not any(dominated(r, o) for _, o in T.iterrows())]
front = pd.DataFrame(front).sort_values("sec")

print("=" * 104)
print("【帕累托前沿】三目标 setF1↑ / sec↓ / need↑（非支配解）")
print("=" * 104)
print(f"  {'X(元)':>6}{'m':>6}{'N':>7}{'k':>6}{'need':>7}{'setF1':>8}{'sec':>7}")
for _, r in front.iterrows():
    print(f"  {r.X:>6.2f}{r.m:>6,}{r.N:>7,}{r.k:>6,}{r.need:>7.1%}{r.setF1:>8.4f}{r.sec:>7.1f}")

# ── 按 demo 约束筛（need ≥ 2/3 = 检索真的在干活；sec ≤ 30s）──
print("\n" + "=" * 104)
print("【demo 结论】need ≥ 2/3（检索真的在干活）且 sec ≤ 30s（秒级）→ 取 setF1 最高")
print("=" * 104)
cand = T[(T.sec <= 30) & (T.need >= 2 / 3)].sort_values("setF1", ascending=False)
print(f"  {'X(元)':>6}{'N':>8}{'k':>7}{'need':>7}{'setF1':>8}{'sec':>7}{'100题费用':>11}")
for _, r in cand.head(10).iterrows():
    print(f"  {r.X:>6.2f}{r.N:>8,}{r.k:>7,}{r.need:>7.1%}{r.setF1:>8.4f}{r.sec:>7.1f}"
          f"{'¥%.1f' % (r.X * 100):>11}")
if not len(cand):
    print("  （无解 —— 需放宽约束）")

# ── 对照：不要求 need 时能到多高（说明「检索有必要」的代价）──
alt = T[T.sec <= 60].sort_values("setF1", ascending=False)
print("\n【对照】若**不强求** need ≥ 2/3（只要求 ≤60 秒），能到多高：")
for _, r in alt.head(3).iterrows():
    print(f"  X=¥{r.X:<5} N={r.N:<7,} need={r.need:>6.1%} setF1={r.setF1:.4f} sec={r.sec:.1f}")
print("  → 这些点 setF1 更高，但**检索只承担不到一半**，「检索有必要」这条讲不出来。")

