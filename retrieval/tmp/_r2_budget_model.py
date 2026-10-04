r"""**预算 → k → 检索效果 → 端到端效果** 的计算模型（零新调用、零新论文）。

用户的问题：*"...规定单次花费不大于 x 元，那么判官可以判的上限就有了，k 的上限也有了，对吧。
对应的单价对应的取 k 对应的检索效果对应的端到端效果，理论上是不是可以计算的？"*

**答案：可以。** 需求的两半我们都有现成的：

| 半边 | 来源 | 是否需要更多论文 |
|---|---|---|
| **成本 → 可判篇数 `m`** | 单价 × 次数；**证据窗口封顶 4,200 字符 → 每篇成本恒定** → **与论文大小无关** | ❌ |
| **检索效果 `ceil(N, k)`** | **重采样外推**（新干扰按"同难度"插入现有 50 篇的得分序） | ❌ |
| **端到端** | `R = ceil × ρ`，`F1 = 2PR/(P+R)`；`ρ` = 判官**条件召回**（实测） | ❌ |

## 成本侧（为什么不需要"平均论文大小"）
证据窗口 = `≤4,200` 字符 → **一篇论文再长，判官也只吃 4,200 字符** → **每次调用对"篇"是常数**：

    m = floor( (X − 2·c) / c )        # 2·c = 子查询 + 判据锚点 两次固定调用

## 效果侧（外推的三个假设，必须声明）
1. **新干扰与现有干扰"同难度"**：新篇得分从现有**非 gold 篇的得分分布**采样
   （另给"更易/更难"变体框住边界）
2. **`ρ` 只依赖预算比例 `k/N`**（实测 `ρ` 随 `k/N` 单调下降）
3. **`P` 为常数**（实测 0.71）← ⚠️ **最弱的一环**

## `ceil` 外推（解析 + 二项采样，O(1)/gold/次）
    p_i     = P(某个新干扰得分 > gold_i 的得分)
    rank_i  = 1 + #{原序中严格大于 s_i 的篇} + tie_i + Binomial(D, p_i)
    ceil@k  = mean_i [ rank_i ≤ k ]            D = N − 50
`D = 0` 时**逐位复现实测 `ceil@k`** → 这条当自检。

跑法：`./.venv/Scripts/python.exe -u retrieval/tmp/_r2_budget_model.py`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P_JUDGE = 0.71          # 实测交付精确率（判完全部候选后）
C_JUDGE = 0.002         # 元/次（`set_judge.py` 实测 600 次 ≈ ¥1.2；token 反算 ¥0.0016）


def load_facets() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """返回 `(scores, gold_mask, 真实名次)` 三元组。

    ⚠️ 需要**真实名次**：得分可能有并列，`D>0` 时并列只能**随机打散**（近似），
    而 `D=0` 用真实名次可让自检**严格通过**。
    """
    d = pd.read_csv(HERE / "results" / "R2_PROD_FINAL_DELIV.csv")
    d["in_gold"] = d["in_gold"].astype(int)
    out = []
    for _, x in d.groupby(["cluster", "facet"]):
        out.append((x["score"].to_numpy(float), x["in_gold"].to_numpy(bool),
                    x["rank"].to_numpy(int)))
    return out


def ceil_sim(facets, N: int, k: int, B: int, seed: int = 0,
             sub: str = "same") -> float:
    """在现有语料上加 `D = N − 50` 篇干扰后的 `ceil@k`（对 facet 平均）。

    `sub`：新干扰的难度来源 —— `same`（非 gold 全分布）/ `easy`（低分半）/ `hard`（高分半）。
    """
    rng = np.random.default_rng(seed)
    D = max(0, N - 50)
    acc = 0.0
    for s, g, rk in facets:
        gs, ns_all = s[g], np.sort(s[~g])
        if sub == "easy":
            ns = ns_all[: max(1, len(ns_all) // 2)]
        elif sub == "hard":
            ns = ns_all[len(ns_all) // 2:]
        else:
            ns = ns_all
        gt = np.array([(s > v).sum() for v in gs])
        eq = np.array([(s == v).sum() for v in gs])
        p = np.array([((ns > v).sum() + 0.5 * (ns == v).sum()) / len(ns) for v in gs])
        if D == 0:
            # ★ 无新增 → **直接用真实名次**（并列按数据的实际次序）→ 自检可严格通过
            acc += float((rk[g] <= k).mean())
            continue
        off = rng.integers(0, eq, size=(B, len(gs)))
        rank = 1 + gt[None, :] + off + rng.binomial(D, p, size=(B, len(gs)))
        acc += float((rank <= k).mean())
    return acc / len(facets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--c", type=float, default=C_JUDGE)
    args = ap.parse_args()
    F = load_facets()
    print("=" * 126)
    print(f"【预算 → k → 效果 计算模型】{len(F)} 个 facet ｜ 判官单价 ¥{args.c}/次 ｜ "
          f"自举 {args.boot} 次 ｜ P_judge 假设 {P_JUDGE}")

    # ── 0. 自检：D=0 必须复现实测 ceil@k ──
    d = pd.read_csv(HERE / "results" / "R2_PROD_FINAL_DELIV.csv")
    d["in_gold"] = d["in_gold"].astype(int)
    d["label"] = d["label"].fillna("").astype(str).str.strip().str.lower()
    print("\n【自检】D=0（不加干扰）时模型应逐位复现实测 ceil@k")
    print(f"  {'k':>5}{'模型':>10}{'实测':>10}")
    ok = True
    for k in (5, 10, 20, 30, 40, 50):
        real = float(np.mean([
            len(set(x[(x["rank"] <= k) & (x.in_gold == 1)]["docid"])) /
            max(len(set(x[x.in_gold == 1]["docid"])), 1)
            for _, x in d.groupby(["cluster", "facet"])]))
        sim = ceil_sim(F, 50, k, 1)
        ok &= abs(sim - real) < 1e-9
        print(f"  {k:>5}{sim:>10.4f}{real:>10.4f}{'' if abs(sim-real)<1e-9 else '  ✗'}")
    print(f"  → {'✓ 逐位一致（外推算法可信）' if ok else '✗ 不一致，外推不可用'}")

    # ── ρ(k)：判官条件召回（实测反推；DELIV 含**全部 50 篇判果** → 任意 k 都能算）──
    KS_RHO = [5, 10, 15, 20, 25, 30, 40, 50]
    fx, fy = [], []
    for kk in KS_RHO:
        ce = ceil_sim(F, 50, kk, 1)
        rrs = []
        for _, x in d.groupby(["cluster", "facet"]):
            gset = set(x[x.in_gold == 1]["docid"])
            kx = x[x["rank"] <= kk]
            pred = set(kx[kx["label"] == "yes"]["docid"])
            rrs.append(len(pred & gset) / max(len(gset), 1))
        fx.append(kk / 50)
        fy.append(float(np.mean(rrs)) / ce)
    print(f"\n【判官条件召回 ρ】实测（预算比例 k/N → ρ）：")
    print("        " + " ｜ ".join(f"{a:.1f}→{b:.3f}" for a, b in zip(fx, fy)))
    print("    （ρ 随 k/N 单调下降：被判的篇越靠后、证据越弱 → 判官越容易判错）")
    rho = lambda fr: float(np.interp(fr, fx, fy))          # noqa: E731

    # ── F1 口径标定：`2PR/(P+R)` 用的是**均值 P/R**，而我们对外报的 setF1 是
    #    **逐 facet F1 的平均**（F1 是凹函数 → Jensen 差）。在本档实测点上标定这个差。
    f0 = pd.read_csv(HERE / "results" / "R2_PROD_FINAL.csv")
    f0 = f0[f0.k == 0]
    F1H = 2 * f0.setP.mean() * f0.setR.mean() / (f0.setP.mean() + f0.setR.mean())
    MACRO = float(F1H - f0.setF1.mean())
    print(f"\n【F1 口径标定】k=∞ 实测：均值 P {f0.setP.mean():.3f} × 均值 R {f0.setR.mean():.3f} "
          f"→ `2PR/(P+R)` = {F1H:.4f}；而**逐 facet F1 的平均** = {f0.setF1.mean():.4f}"
          f" ｜ **差 {MACRO:+.4f}**（Jensen）")
    print(f"    → 下表 `setF1*` 为 `2PR/(P+R)`；`setF1` 已扣此差（{MACRO:+.4f}），"
          f"与定稿报数同口径")

    def f1_of(R: float) -> float:
        """端到端 setF1（**扣掉 Jensen 口径差**，与定稿报数可比）。"""
        h = 2 * P_JUDGE * R / (P_JUDGE + R) if (P_JUDGE + R) else 0.0
        return h - MACRO

    # ── 表 A ──
    print("\n" + "=" * 126)
    print("【表 A】可判篇数上限 m = (X − 2c) / c   （窗口封顶 4,200 字符 → **每篇成本恒定，"
          "与论文长短无关**）")
    CS = [0.0005, 0.001, 0.002, 0.004, 0.01]
    print(f"  {'X(元/题)':>10}" + "".join(f"{f'¥{c}/次':>13}" for c in CS))
    for X in (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0):
        print(f"  {X:>10.2f}" + "".join(
            f"{max(0, int((X - 2 * c) // c)):>13,}" for c in CS))

    # ── 表 B ──
    print("\n" + "=" * 126)
    print(f"【表 B】单价 ¥{args.c}/次 ｜ `k = min(m, N)` ｜ `R = ceil(k,N) × ρ(k/N)` ｜ "
          f"`F1 = 2·{P_JUDGE}·R/({P_JUDGE}+R)` ｜ 判官秒 = k×0.8/8")
    print(f"\n  {'X(元)':>7}{'m(可判)':>9}{'N(语料)':>9}{'k':>7}{'k/N':>7}{'ceil@k':>9}"
          f"{'ρ':>7}{'R':>7}{'P':>7}{'setF1':>8}{'判官秒':>8}")
    for X in (0.1, 0.5, 1.0, 2.0, 5.0):
        m = max(0, int((X - 2 * args.c) // args.c))
        for N in (50, 200, 1000, 10000):
            k = min(m, N)
            if k < 1:
                continue
            ce = ceil_sim(F, N, k, max(30, args.boot // 6))
            fr = k / N
            R = ce * rho(fr)
            F1 = f1_of(R)
            print(f"  {X:>7.2f}{m:>9,}{N:>9,}{k:>7,}{fr:>7.1%}{ce:>9.4f}"
                  f"{rho(fr):>7.3f}{R:>7.3f}{P_JUDGE:>7.2f}{F1:>8.4f}{k*0.8/8:>8.1f}")
        print()

    # ── 表 C ──
    X0, m0 = 1.0, max(0, int((1.0 - 2 * args.c) // args.c))
    print("=" * 126)
    print(f"【表 C ★ 衰减曲线】固定预算 **¥{X0:.2f}/题**（可判 {m0:,} 篇）｜ "
          f"**论文库变大**的代价")
    print(f"  {'N(语料)':>10}{'k=min(m,N)':>12}{'k/N':>7}{'ceil@k':>9}{'ρ':>7}{'R':>7}"
          f"{'setF1':>8}{'判官秒':>8}")
    for N in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 100000):
        k = min(m0, N)
        ce = ceil_sim(F, N, k, max(20, args.boot // 8))
        fr = k / N
        R = ce * rho(fr)
        F1 = f1_of(R)
        print(f"  {N:>10,}{k:>12,}{fr:>7.1%}{ce:>9.4f}{rho(fr):>7.3f}{R:>7.3f}"
              f"{F1:>8.4f}{k*0.8/8:>8.1f}")

    # ── 表 D ──
    print("\n" + "=" * 126)
    print(f"【表 D】外推假设的敏感性：新增干扰**更易 / 同难度 / 更难**"
          f"（X=¥{X0:.2f}，N=10,000，k={m0}）")
    print(f"  {'情形':<14}{'ceil@k':>9}{'R':>7}{'setF1':>8}")
    for mode, lab in (("easy", "更易（跨方向）"), ("same", "同难度（基线）"),
                      ("hard", "更难（同方向）")):
        ce = ceil_sim(F, 10000, m0, max(20, args.boot // 8), sub=mode)
        R = ce * rho(m0 / 10000)
        F1 = f1_of(R)
        print(f"  {lab:<14}{ce:>9.4f}{R:>7.3f}{F1:>8.4f}")

    print("\n" + "=" * 126)
    print("【三条读法】")
    print("  ① **k 由预算决定，不由语料决定**（只要语料 ≫ 可判篇数）")
    print("  ② **语料变大 → 同样的钱保住的召回比例下降**（`ceil` 只依赖 `k/N`）")
    print("  ③ 要在任何规模上维持 R，只有四条路：降单价 ｜ 加预算 ｜ "
          "**抬高 `ceil` 曲线（= 改进排序）** ｜ **便宜的预过滤（不伤 gold 地压 N）**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
