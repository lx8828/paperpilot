"""**facet 谱审计**（零 LLM）：把 19 facet × 3 簇 = 57 个组合摊开，定"合理 facet"的筛选标准。

背景：当年语料构建用**词面锚点**筛 facet（正例率 15%~75%）→ **假阳性也能撑起一道题**
（实测 `deployment` 三簇新真值全为 0，全靠 12 个假阳性入选）。
现在有了**重标后的真值**（`gold_recalib2.csv`，893 对，覆盖全部 57 组合）→ 可以按真值重筛。

判据（都基于**重标真值**，锚点只作参考）：
  · 分辨率：正例率 ∈ [MIN_RATE, MAX_RATE]（否则题无意义）
  · 样本量：正例数 ≥ MIN_POS 且 负例数 ≥ MIN_NEG
  · 跨簇复现：同一 facet 在 ≥2 簇上有分辨率
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rate", type=float, default=0.10)
    ap.add_argument("--max-rate", type=float, default=0.90)
    ap.add_argument("--min-pos", type=int, default=3)
    ap.add_argument("--min-neg", type=int, default=3)
    ap.add_argument("--recalib", default="gold_recalib2.csv")
    ap.add_argument("--facets-file", default="", help="用另一份 facet 表（只影响标题计数）")
    args = ap.parse_args()

    df = pd.read_csv(DEV / args.recalib)
    df["anchor_gold"] = df["anchor_gold"].astype(bool)
    df["new_gold"] = df["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    df["A_err_s"] = df["A_err"].where(df["A_err"].notna(), "").astype(str)
    df["B_err_s"] = df["B_err"].where(df["B_err"].notna(), "").astype(str)

    rows = []
    for (c, f), g in df.groupby(["cluster", "facet"]):
        A, N = set(g[g.anchor_gold].docid), set(g[g.new_gold].docid)
        tp = len(A & N)
        n, npos = len(g), len(N)
        rate = npos / n if n else 0.0
        rows.append(dict(
            cluster=int(c), facet=str(f), n=n, n_anchor=len(A), n_new=npos, rate=rate,
            anchor_P=tp / len(A) if A else float("nan"),
            anchor_R=tp / len(N) if N else float("nan"),
            added=len(N - A), removed=len(A - N),
            agree=float(g.agree.mean()) if "agree" in g else float("nan"),
            third=float(g.need_third.mean()) if "need_third" in g else float("nan"),
            ok=(args.min_rate <= rate <= args.max_rate and npos >= args.min_pos
                and (n - npos) >= args.min_neg),
        ))
    s = pd.DataFrame(rows)

    print("=" * 118)
    print(f"【facet 谱】19 facet × 3 簇 = {len(s)} 个组合 ｜ 判据：正例率 ∈ "
          f"[{args.min_rate:.0%}, {args.max_rate:.0%}] ∧ 正例 ≥{args.min_pos} ∧ 负例 ≥{args.min_neg}")
    print(f"  {'簇':>3} {'facet':<18}{'篇':>4}{'锚点':>5}{'新真值':>7}{'正例率':>8}"
          f"{'锚点P':>7}{'锚点R':>7}{'新增':>5}{'删去':>5}{'第三轮':>8}  判定")
    for _, r in s.sort_values(["cluster", "facet"]).iterrows():
        if not r.n_new and not r.n_anchor:
            verdict = "❌ 锚点与真值都为空"
        elif not r.n_new:
            verdict = f"❌ 无正例（锚点 {int(r.n_anchor)} 个全是假阳性）"
        elif r.n_new == r.n:
            verdict = "❌ 全正例"
        elif r.rate < args.min_rate:
            verdict = f"⚠️ 正例率过低（{r.rate:.0%}）"
        elif r.rate > args.max_rate:
            verdict = f"⚠️ 正例率过高（{r.rate:.0%}）"
        elif r.n_new < args.min_pos:
            verdict = f"⚠️ 正例仅 {int(r.n_new)} 篇"
        elif r.n - r.n_new < args.min_neg:
            verdict = f"⚠️ 负例仅 {int(r.n - r.n_new)} 篇"
        else:
            verdict = "✅ 可用"
        if r.n_anchor == 0:
            verdict += "（锚点看不到→旧流程漏掉）"
        print(f"  {int(r.cluster):>3} {r.facet:<18}{int(r.n):>4}{int(r.n_anchor):>5}"
              f"{int(r.n_new):>7}{r.rate:>8.1%}{r.anchor_P:>7.2f}{r.anchor_R:>7.2f}"
              f"{int(r.added):>5}{int(r.removed):>5}{r.third:>8.1%}  {verdict}")

    print("\n" + "=" * 118)
    print("【按 facet 汇总：跨簇复现性】")
    piv = s.pivot_table(index="facet", columns="cluster", values="n_new", aggfunc="sum")
    rat = s.pivot_table(index="facet", columns="cluster", values="rate", aggfunc="mean")
    okc = s[s.ok].groupby("facet")["cluster"].nunique()
    print(f"  {'facet':<18}{'总正例':>7}{'簇数(有真值)':>13}{'✅可用簇数':>11}"
          f"  各簇正例(gold) / 正例率")
    for f in sorted(F.keys()):
        sub = s[s.facet == f]
        tot = int(sub.n_new.sum())
        nz = int((sub.n_new > 0).sum())
        nok = int(okc.get(f, 0))
        detail = " ｜ ".join(
            f"簇{int(r.cluster)} {int(r.n_new)}/{int(r.n)}={r.rate:.0%}"
            for _, r in sub.iterrows() if r.n_new or r.n_anchor)
        flag = "✅" if nok >= 2 else ("◻" if nok == 1 else "❌")
        print(f"  {flag} {f:<16}{tot:>7}{nz:>13}{nok:>11}  {detail[:70] or '（全空）'}")

    print(f"\n  **可用组合 {int(s.ok.sum())}/{len(s)}** ｜ 涉及 facet "
          f"{s[s.ok].facet.nunique()}/{len(F)}")
    print(f"  **与旧 usable 名单对比**：旧 35 个组合 / 新 {int(s.ok.sum())} 个")
    meta = json.loads((DEV / "clusters" / "meta.json").read_text(encoding="utf-8"))
    old = {(i + 1, f) for i, m in enumerate(meta) for f in m["usable"]}
    new = {(int(r.cluster), r.facet) for _, r in s[s.ok].iterrows()}
    print(f"\n  ➕ 新增（旧名单没有、新判据有）：{sorted(new - old)}")
    print(f"  ➖ 剔除（旧名单有、新判据没有）：{sorted(old - new)}")

    s.to_csv(HERE / "results" / f"R2_FACET_AUDIT_{args.recalib.split('.')[0].split('_')[-1].upper()}.csv",
             index=False, encoding="utf-8-sig")
    print(f"\n→ {HERE / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
