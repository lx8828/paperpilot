r"""**C3 级联离线分析**（零新 LLM 调用）

`_r2_c3_cascade.py` 已把**池内每块**的 `nli` 分、`regex` 命中、**判官 label** 落盘
（`results/R2_C3_LABELS.csv`）→ 于是所有变体、所有 τ 都能**离线复算**，一次调用都不用再花。

## 要判定什么
小样实跑显示级联 **省 58.8% 但 F1 −0.112**，病根怀疑是 **"regex 命中直接判正"**
（簇1：①判正 9 篇 / P 0.889 → ②判正 23 篇 / P 0.435）。
→ 这里把 **"regex 判正"** 与 **"NLI 粗筛"** 两个动作**拆开**，看谁在掉分：

| 变体 | 做法 | 调用数/题 |
|---|---|---|
| **V0 全判** | 池内全判（τ=0 就是它，作自检） | \|pool\| |
| **V1 = NLI粗筛 + regex判正** | 粗筛 → regex 判正 → 其余 LLM | \|surv ∧ ¬regex\| |
| **V2 = 只做 NLI 粗筛** | 粗筛 → **幸存块全部 LLM**（不引入 regex 判正） | \|surv\| |
| **V3 = regex 判正（不粗筛）** | 全池 regex 判正 + 其余 LLM | \|pool ∧ ¬regex\| |

doc 级判定：该篇有任一块被判正 → 该篇为正。指标 = 集合 **P / R / F1**（vs `gold_final3`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAUS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]


def main() -> int:
    L = pd.read_csv(HERE / "results" / "R2_C3_LABELS.csv")
    gg = pd.read_csv(DEV / "gold_final3.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    L["yes"] = L["label"].astype(str).str.lower().eq("entail")
    L["err"] = L["label"].astype(str).str.lower().eq("err")
    print(f"标签 {len(L)} 行 ｜ ERR {int(L['err'].sum())} ｜ facet {L.groupby(['cluster', 'facet']).ngroups}")
    print(f"判正块占比（全池）{L['yes'].mean():.1%} ｜ regex 命中占比 {L['regex'].mean():.1%}")

    rows = []
    for (ci, fa), x in L.groupby(["cluster", "facet"]):
        gold = NG.get((int(ci), str(fa)), set())
        pool = x                                # 全部都是池内块
        for tau in TAUS:
            surv = pool[pool["nli"] >= tau]
            # 变体 V1/V2/V3 的**判正块集合**
            v_llm = surv[surv["yes"]]                                  # 幸存块里 LLM 判正
            set2 = set(v_llm["docid"]) | set(surv[surv["regex"] == 1]["docid"])   # V1
            setA = set(v_llm["docid"])                                          # V2
            setR = (set(pool[pool["regex"] == 1]["docid"])
                    | set(pool[(pool["regex"] == 0) & pool["yes"]]["docid"]))    # V3
            set0 = set(pool[pool["yes"]]["docid"])                              # V0
            for tag, s, calls in (("V0_全判", set0, len(pool)),
                                  ("V1_NLI粗筛+regex判正", set2, len(surv[surv["regex"] == 0])),
                                  ("V2_只NLI粗筛", setA, len(surv)),
                                  ("V3_只regex判正", setR, len(pool[pool["regex"] == 0]))):
                inter = len(s & gold)
                p = inter / max(len(s), 1)
                r = inter / max(len(gold), 1)
                rows.append(dict(cluster=int(ci), facet=fa, tau=tau, variant=tag,
                                 calls=calls, n_pred=len(s), inter=inter,
                                 setP=p, setR=r,
                                 setF1=2 * p * r / (p + r) if (p + r) else 0.0))
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "results" / "R2_C3_OFFLINE.csv", index=False, encoding="utf-8-sig")

    # ── 自检：V0 应与 τ 无关 ──
    v0 = R[R.variant == "V0_全判"]
    print(f"\n[自检] V0_全判 在 9 个 τ 上的 F1 极差 = "
          f"{v0.groupby('tau')['setF1'].mean().max() - v0.groupby('tau')['setF1'].mean().min():.4f}"
          f"（应 ≈0，即与 τ 无关）")

    for tag in ("V0_全判", "V1_NLI粗筛+regex判正", "V2_只NLI粗筛", "V3_只regex判正"):
        t = R[R.variant == tag]
        base = t[t.tau == 0.0]["setF1"].mean()
        print("\n" + "=" * 108)
        print(f"【{tag}】")
        print(f"  {'τ':>5}{'调用/题':>9}{'节省':>8}{'setP':>8}{'setR':>8}{'setF1':>9}"
              f"{'ΔF1':>9}{'判正篇数':>10}")
        for tau in TAUS:
            s = t[t.tau == tau]
            if not len(s):
                continue
            pool = L[L.facet.isin(s.facet)]
            pn = pool.groupby(["cluster", "facet"]).size().mean()
            print(f"  {tau:>5.2f}{s['calls'].mean():>9.0f}"
                  f"{1 - s['calls'].mean() / pn:>8.1%}{s['setP'].mean():>8.3f}"
                  f"{s['setR'].mean():>8.3f}{s['setF1'].mean():>9.3f}"
                  f"{s['setF1'].mean() - base:>+9.3f}{s['n_pred'].mean():>10.1f}")

    # ── 选优：在"省 ≥50%"的约束下取 F1 最好的 ──
    print("\n" + "=" * 108)
    print("【在「省 ≥ 指定比例」约束下，各变体的最好 F1】")
    print(f"  {'节省门槛':>9}{'V1 F1':>9}{'@τ':>7}{'V2 F1':>9}{'@τ':>7}{'V3 F1':>9}{'@τ':>7}"
          f"　（V0 无约束 = {R[(R.variant == 'V0_全判') & (R.tau == 0)]['setF1'].mean():.3f}）")
    for thr in (0.0, 0.3, 0.5, 0.6, 0.7):
        line = f"  {thr:>9.0%}"
        for tag in ("V1_NLI粗筛+regex判正", "V2_只NLI粗筛", "V3_只regex判正"):
            t = R[R.variant == tag].copy()
            # 每 τ 一个 (省, F1)
            agg = t.groupby("tau").agg(f1=("setF1", "mean"), calls=("calls", "mean"))
            tot = L.groupby(["cluster", "facet"]).size().mean()
            agg["save"] = 1 - agg["calls"] / tot
            ok = agg[agg.save >= thr - 1e-9]
            if len(ok):
                b = ok.f1.idxmax()
                line += f"{ok.f1.max():>9.3f}{b:>7.2f}"
            else:
                line += f"{'—':>9}{'—':>7}"
        print(line)
    print("\n  ⚠️ 小样仅 3 个 facet（每簇 1 个）→ 数字噪声大，只用于判**方向**，别当定论。")
    print(f"  → 已写 results/R2_C3_OFFLINE.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
