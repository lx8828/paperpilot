"""池子深度 → **召回天花板**：从已有逐题记录里现算，零成本。

用途：在决定要不要把候选池从 C=100 加到 C=300/500 之前，先看**上限能涨多少**。
`g_rank_retr` = gold 在**所属语料** hybrid 全序里的位次 → 池子宽 C 时
`撞线率 = P(g_rank_retr <= C)` 就是**该池子宽度下的召回天花板**（= 精排 R@1 的上界）。

⚠️ 这只是"上限"。上限涨了不等于收益能落地 —— CE 能不能把新进来的候选排到前面，
必须实跑。见 `LITSEARCH_RERANK_DEPTH.md` 的负结果（旧语料池子 100→300：
天花板 +5.5pt，R@1 +0.00pt，R@5 −0.84pt）。

用法：
    python retrieval/scripts/pool_ceiling.py                       # 读 MIXED200 的记录
    python retrieval/scripts/pool_ceiling.py --csv xxx_raw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
KS = (10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="LITSEARCH_MIXED200_raw.csv")
    args = ap.parse_args()

    d = pd.read_csv(RESULTS / args.csv, encoding="utf-8-sig")
    uni = d[d["batch"] == "uniform"]
    print(f"读入 {len(d)} 条（均匀主样本 {len(uni)}）\n")

    groups = [
        ("总体·均匀", uni),
        ("旧语料（全）", d[d["语料"] == "旧"]),
        ("新语料（全）", d[d["语料"] == "新"]),
        ("具体型（全）", d[d["抽象层级"] == "具体"]),
        ("上位型（全）", d[d["抽象层级"] == "上位"]),
    ]

    rows = []
    for name, sub in groups:
        if sub.empty:
            continue
        r = sub["g_rank_retr"].to_numpy()
        row = {"分层": f"{name} ({len(sub)})"}
        for k in KS:
            row[f"C={k}"] = float((r <= k).mean())
        rows.append(row)
    t = pd.DataFrame(rows)

    print("=" * 118)
    print("召回天花板 = P(gold 落在本语料前 C 名)  ← 池子宽 C 时精排 R@1 的**理论上界**")
    print("=" * 118)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n【边际收益：加宽池子能多捞回多少】（百分点）")
    mr = []
    for name, sub in groups:
        if sub.empty:
            continue
        r = sub["g_rank_retr"].to_numpy()
        mr.append({
            "分层": f"{name} ({len(sub)})",
            "100→200": 100 * ((r <= 200).mean() - (r <= 100).mean()),
            "100→300": 100 * ((r <= 300).mean() - (r <= 100).mean()),
            "100→500": 100 * ((r <= 500).mean() - (r <= 100).mean()),
        })
    print(pd.DataFrame(mr).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    print("\n【现在漏掉、但加宽能捞回的题】（100 < 本语料位次 ≤ 300）")
    miss = d[(d["g_rank_retr"] > 100) & (d["g_rank_retr"] <= 300)]
    print(f"  共 {len(miss)} 条 / {len(d)}（{100 * len(miss) / len(d):.1f}%）")
    if len(miss):
        print("  按抽象层级：" + str(miss["抽象层级"].value_counts().to_dict()))
        print("  按语料侧  ：" + str(miss["语料"].value_counts().to_dict()))
        print("  位次分布  ：")
        bins = [(100, 150), (150, 200), (200, 250), (250, 300)]
        for lo, hi in bins:
            n = int(((miss["g_rank_retr"] > lo) & (miss["g_rank_retr"] <= hi)).sum())
            print(f"    {lo:>4}-{hi:<4} {n:>3} 条")
        print(f"  中位位次 {int(miss['g_rank_retr'].median())} | "
              f"最大 {int(miss['g_rank_retr'].max())}")

    print("\n【>300 才有的（加宽到 300 也救不回）】")
    far = d[d["g_rank_retr"] > 300]
    print(f"  共 {len(far)} 条（{100 * len(far) / len(d):.1f}%）| "
          f"抽象层级 {far['抽象层级'].value_counts().to_dict()}")

    # ────────────────────────────────────────────────────────────
    # ★ 同等 CE 成本下：**合并语料 vs 只用 arXiv**
    #   池深是「每语料各取 C 个」→ 合并模式把一半预算花在了旧语料上。
    #   对 arXiv 侧的 gold 来说，旧语料那 100/200/300 个候选是**纯干扰项**
    #   （不可能命中），所以拿它们去换 arXiv 候选是净赚。
    # ⚠️ 该节只在**合并语料**的产物上有意义（arXiv-only 的池子没有"每个语料各取"）。
    # ────────────────────────────────────────────────────────────
    if (d["语料"] == "旧").sum() == 0:
        print("\n（本 CSV 是 arXiv-only 产物 —— 上面那张「同等成本」对比不适用，已跳过）")
        return 0
    print("\n" + "=" * 112)
    print("★ 同等 CE 打对数下的 arXiv 侧天花板（只对 gold 在 arXiv 的题）")
    print("=" * 112)
    arx = d[d["语料"] == "新"]
    r = arx["g_rank_retr"].to_numpy()
    print(f"  gold 在 arXiv 的题：{len(arx)} 条")
    print(f"  {'CE打对数':>9} | {'合并(旧+新)池构成':>18} {'arXiv侧天花板':>13} | "
          f"{'只用arXiv 池构成':>17} {'天花板':>10} | {'arXiv-only 增益':>15}")
    print("  " + "-" * 106)
    rows2 = []
    for C in (100, 200, 300, 400, 500, 600):
        cm = float((r <= C).mean())          # 合并模式：arXiv 侧拿到 C 个
        ca = float((r <= 2 * C).mean())      # 同成本：全部预算给 arXiv → 2C 个
        rows2.append({"CE打对数": 2 * C, "合并池": f"{C}旧+{C}新",
                      "合并_arXiv天花板": cm, "仅arXiv池": f"{2 * C}",
                      "仅arXiv天花板": ca, "增益pt": 100 * (ca - cm)})
        print(f"  {2 * C:>9} | {f'{C}旧+{C}新':>18} {cm:>13.4f} | "
              f"{f'{2 * C} arXiv':>17} {ca:>10.4f} | {100 * (ca - cm):>+14.2f}pt")
    print("\n⚠️ 同一个池深 C 下，两者的 arXiv 侧天花板**完全相同** ——")
    print("   因为 `g_rank_retr` 是「该语料**内部**位次」，与池里有没有旧语料无关。")
    print("   → 所以「排除旧语料」这条动作本身（不重新分配预算）只值**兑现率那一项**：")
    new = d[d["语料"] == "新"]
    conv = ((new["llm300"] > 0) & (new["llm300"] <= 1)).mean() / max(
        (new["g_rank_retr"] <= 300).mean(), 1e-9)
    print(f"     实测合并模式的新语料兑现率 = {100 * conv:.1f}% → **最多 +{100 * (1 - conv):.1f}pt**")
    print("   → 真正的增益来自**把省下的预算还给 arXiv**（见上表）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
