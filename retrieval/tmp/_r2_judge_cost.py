r"""**判官能不能只留一个？** —— 用已有逐篇 A/B/third 记录离线算（**零新调用**）

数据源：`results/R2_PROD_FINAL_DELIV.csv`（我上一轮评估时把**全部候选篇**的
`label / A / B / third / rank / in_gold` 都落盘了）→ 于是可以**离线**比较几种判官协议的
交付集合与指标，不必重跑任何 LLM。

比较的协议：
| 协议 | 调用/篇 | 规则 |
|---|---|---|
| **cur（现状）** | **2.1** | A/B 双方 yes→yes；双方 no→no；**其余走第三轮** |
| **onlyA（纯单判官）** | **1.0** | 只信 A 的 `yes` |
| onlyB | 1.0 | 只信 B 的 `yes` |
| strict2（双判官都须 yes，无第三轮） | 2.0 | A==yes 且 B==yes |
| anyYes（任一 yes） | 2.0 | A==yes 或 B==yes |
| **condA（条件触发）** | **≈1.15** | 只跑 A；**仅当 A 判 `unclear` 才补 B + 第三轮**（分歧不复核） |

指标：集合 P/R/F1 vs `gold_final3`（按 rank 前缀切 k，口径同评估脚本）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
KS = [20, 30, 50, 0]


def main() -> int:
    d = pd.read_csv(HERE / "results" / "R2_PROD_FINAL_DELIV.csv").reset_index(drop=True)
    for c in ("A", "B", "third", "label"):
        d[c] = d[c].fillna("").astype(str).str.strip().str.upper()
    gg = pd.read_csv(DEV / "gold_final3.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    print("=" * 116)
    print(f"逐篇记录 {len(d)} 行 ｜ facet {d.groupby(['cluster', 'facet']).ngroups} ｜ 候选均 "
          f"{d.groupby(['cluster', 'facet']).size().mean():.1f} 篇/题")
    ab = d[(d.A != "") & (d.B != "")]
    dis = ab[ab.A != ab.B]
    print(f"A/B 双方都有输出 {len(ab)} ｜ **分歧 {len(dis)}（{len(dis) / max(len(ab), 1):.1%}）**"
          f" ｜ A=unclear {int((d.A == 'UNCLEAR').sum())} ｜ B=unclear {int((d.B == 'UNCLEAR').sum())}")

    # 各协议下该篇是否交付
    yes = {
        "cur（现状 A/B/第三轮）": d.label == "YES",
        "onlyA（纯单判官）": d.A == "YES",
        "onlyB": d.B == "YES",
        "strict2（双都须yes）": (d.A == "YES") & (d.B == "YES"),
        "anyYes（任一yes）": (d.A == "YES") | (d.B == "YES"),
        "condA（A为unclear才补B）": np.where(d.A == "UNCLEAR",
                                            (d.B == "YES") | (d.third == "YES"),
                                            d.A == "YES"),
    }
    # ★ `condA` 的调用数按**实测** unclear 率算（不是拍脑袋）
    p_unc = float((d.A == "UNCLEAR").mean())
    CALLS = {"cur（现状 A/B/第三轮）": 2.1, "onlyA（纯单判官）": 1.0, "onlyB": 1.0,
             "strict2（双都须yes）": 2.0, "anyYes（任一yes）": 2.0,
             "condA（A为unclear才补B）": 1.0 + p_unc * 1.1}
    print(f"  → `condA` 的 A=unclear 率 = {p_unc:.1%} → 调用 "
          f"{CALLS['condA（A为unclear才补B）']:.2f}/篇")

    rows = []
    for k in KS:
        for name, m in yes.items():
            m = np.asarray(m, dtype=bool)
            for (ci, fa), x in d.groupby(["cluster", "facet"]):
                gold = NG.get((int(ci), str(fa)), set())
                g = x.sort_values("rank")
                if k > 0:
                    g = g[g["rank"] <= k]
                idx = g.index.to_numpy()
                pred = set(g.loc[idx[m[idx]], "docid"].astype(str))
                inter = len(pred & gold)
                P = inter / max(len(pred), 1)
                R = inter / max(len(gold), 1)
                rows.append(dict(k=k, proto=name, n_pred=len(pred), calls=CALLS[name],
                                 setP=P, setR=R,
                                 setF1=2 * P * R / (P + R) if (P + R) else 0.0))
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "results" / "R2_JUDGE_PROTO.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 116)
    print("【★ 判官协议对照（离线，零新调用）】")
    print(f"  {'k':>5}{'协议':<26}{'调用/篇':>8}{'交付/题':>9}{'P':>8}{'R':>8}{'F1':>8}{'ΔF1(对现状)':>13}")
    for k in KS:
        base = None
        for name in CALLS:
            t = R[(R.k == k) & (R.proto == name)]
            if not len(t):
                continue
            if base is None:
                base = t.setF1.mean()
            dlt = "" if name.startswith("cur") else f"{t.setF1.mean() - base:+.3f}"
            print(f"  {'∞' if k <= 0 else k:>5}{name:<26}{t.calls.mean():>8.2f}"
                  f"{t.n_pred.mean():>9.1f}{t.setP.mean():>8.3f}{t.setR.mean():>8.3f}"
                  f"{t.setF1.mean():>8.3f}{dlt:>13}")
        print()

    print("【结论要点】在 k=∞（判全部候选）下对比 —— 这是最不利单判官的档位（无排序挡板）")
    t = R[(R.k == 0)]
    for name in CALLS:
        s = t[t.proto == name]
        print(f"  {name:<26} 调用 {s.calls.mean():>4.2f}/篇 ｜ F1 {s.setF1.mean():.3f}"
              f" ｜ R {s.setR.mean():.3f} ｜ P {s.setP.mean():.3f}")
    print(f"\n  → 已写 results/R2_JUDGE_PROTO.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
