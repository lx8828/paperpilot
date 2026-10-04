"""**A3｜三判官多数票 → 最终真值 + 最终 facet 题集**

## 为什么要多数票（而不是挑一个"更好的定义"）
扩大抽检实测（`gold_adjudicate_gold_recalib3.csv`）：
· 收紧定义的 v3 与 v2 **总体一致率持平**（89.1% vs 89.5%）——
  只是把误差从"漏"挪到"误收"（正例错误率 8.3% → 19.2%，负例 11.6% → 7.4%）。
· 细读分歧发现：**很多是第三判官自己漏了**（`We train a Transformer decoder…` 被判非微调、
  `The average of 3 runs with different seeds` 被判非多次运行）。
→ **边界分歧是任务固有的主观性**，换定义/换判官都消不掉。
→ 正解 = **三个异源判官独立投票**（A `deepseek-chat` / B `glm-4-flash` / C `deepseek-reasoner`），
  取多数票；票型（3:0 / 2:1）本身就是**置信度**，可随指标一起报告。

## 题集选择（基于多数票真值）
· 分辨率：正例率 ∈ [10%, 90%]，正例 ≥2，负例 ≥2
· **核心集** = 同一 facet 在 **≥2 簇** 上都可用（跨簇复现 → 结论更可信）
· 单簇可用的进**扩展集**，与核心集分开报告

产物：`data/r2dev/gold_final2.csv`（cluster,facet,docid,gold,votes,A/B/C,anchor_gold）
      `data/r2dev/facets_v2_selected.json`（核心集/扩展集/剔除 + 逐题正例数与票型分布）
"""
from __future__ import annotations

import importlib.util as _iu
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = _iu.spec_from_file_location("facets2", HERE / "tmp" / "_r2_facets_v2.py")
_f = _iu.module_from_spec(spec)
sys.modules["facets2"] = _f
spec.loader.exec_module(_f)
F2 = _f.F2

RECAL = DEV / "gold_recalib3.csv"
ADJ = DEV / "gold_adjudicate_gold_recalib3.csv"
OUT = DEV / "gold_final2.csv"
SEL = DEV / "facets_v2_selected.json"
MIN_RATE, MAX_RATE, MIN_POS, MIN_NEG = 0.10, 0.90, 2, 2


def main() -> int:
    a = pd.read_csv(RECAL)
    c = pd.read_csv(ADJ)
    a["key"] = a.apply(lambda r: f"{int(r.cluster)}|{r.facet}|{r.docid}", axis=1)
    c["key"] = c["key"].astype(str)
    c = c.drop_duplicates("key")
    m = a.merge(c[["key", "third_label", "third_conf", "third_err"]], on="key", how="left")

    def yes(s) -> pd.Series:
        return s.astype(str).str.strip().str.lower().eq("entail")

    m["A_yes"] = yes(m["A_label"])
    m["B_yes"] = yes(m["B_label"])
    m["C_yes"] = yes(m["third_label"])
    for c_ in "ABC":
        m[f"{c_}_yes"] = m[f"{c_}_yes"].where(m[f"{c_}_yes"].notna(), False)
    m["votes"] = m["A_yes"].astype(int) + m["B_yes"].astype(int) + m["C_yes"].astype(int)
    m["gold"] = m["votes"] >= 2                      # 多数票
    m["anchor_gold"] = m["anchor_gold"].astype(bool)
    m["unanimous"] = m["votes"].isin([0, 3])

    print("=" * 112)
    print(f"【三判官多数票】配对 {len(m)} ｜ 判官 A=deepseek-chat / B=glm-4-flash / C=deepseek-reasoner")
    print(f"  {'票型':<10}{'条数':>7}{'占比':>8}   含义")
    for v, lab in ((3, "3:0 全票正"), (2, "2:1 多数正"), (1, "1:2 多数负"), (0, "0:3 全票负")):
        g = m[m.votes == v]
        print(f"  {lab:<10}{len(g):>7}{len(g) / len(m):>8.1%}"
              f"   {'高置信' if v in (0, 3) else '⚠️ 有分歧（置信度中）'}")
    print(f"\n  **正例 {int(m.gold.sum())}**（{m.gold.mean():.1%}）｜ 全票一致的配对 "
          f"{m.unanimous.mean():.1%}")
    print(f"  与锚点对比：锚点正例 {int(m.anchor_gold.sum())}"
          f" ｜ 锚点 P {m[m.anchor_gold].gold.mean():.3f}"
          f" ｜ 锚点 R {m[m.gold].anchor_gold.mean():.3f}")

    # ── 题集选择 ──
    rows = []
    for (cl, f), g in m.groupby(["cluster", "facet"]):
        n, npos = len(g), int(g.gold.sum())
        rate = npos / n
        ok = (MIN_RATE <= rate <= MAX_RATE and npos >= MIN_POS and (n - npos) >= MIN_NEG)
        rows.append(dict(cluster=int(cl), facet=str(f), n=n, n_pos=npos, rate=rate,
                         n_anchor=int(g.anchor_gold.sum()),
                         disrate=float((~g.unanimous).mean()),
                         unanimous_pos=int(((g.votes == 3)).sum()), ok=ok))
    s = pd.DataFrame(rows)
    nclu = s[s.ok].groupby("facet")["cluster"].nunique()
    core = sorted(nclu[nclu >= 2].index)
    ext = sorted(nclu[nclu == 1].index)
    dropped = sorted(set(F2.keys()) - set(core) - set(ext))

    print("\n" + "=" * 112)
    print(f"【最终题集】可用组合 {int(s.ok.sum())}/{len(s)}"
          f" ｜ facet {int(s[s.ok].facet.nunique())}/19")
    print(f"\n  ★ 核心集（≥2 簇复现）：**{len(core)} 个 facet**")
    print(f"  {'facet':<17}{'✅簇':>4}{'总正例':>7}{'各簇(正例/篇=率)':<44}{'分歧配对':>9}")
    for f in core:
        sub = s[s.facet == f].sort_values("cluster")
        det = " ｜ ".join(f"簇{int(r.cluster)} {int(r.n_pos)}/{int(r.n)}={r.rate:.0%}"
                          for _, r in sub.iterrows() if r.ok)
        dr = (sub[sub.ok].n_pos.sum() and
              (sub[sub.ok].disrate * sub[sub.ok].n).sum() / sub[sub.ok].n.sum())
        print(f"  {f:<17}{int(nclu[f]):>4}{int(sub.n_pos.sum()):>7}  {det[:42]:<44}{dr:>9.1%}")
    print(f"\n  ◻ 扩展集（仅 1 簇）：{ext}")
    print(f"  ❌ 剔除：{dropped}")

    m["combo"] = m.cluster.astype(str) + "|" + m.facet
    keepset = {f"{a_}|{b_}" for a_, b_ in s[s.ok][["cluster", "facet"]].apply(tuple, axis=1)}
    out = m[m.combo.isin(keepset)][
        ["cluster", "facet", "docid", "gold", "votes", "A_yes", "B_yes", "C_yes",
         "anchor_gold"]].sort_values(["cluster", "facet", "docid"])
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    SEL.write_text(json.dumps({
        "source": "三判官多数票（A deepseek-chat / B glm-4-flash / C deepseek-reasoner）",
        "facet_defs": "_r2_facets_v2.py::F2（操作化定义）",
        "criteria": f"正例率∈[{MIN_RATE},{MAX_RATE}] ∧ 正例≥{MIN_POS} ∧ 负例≥{MIN_NEG}",
        "core_facets": core, "extended_facets": ext, "dropped_facets": dropped,
        "n_pairs": len(out), "n_pos": int(out.gold.sum()),
        "per_combo": {f"{int(r.cluster)}|{r.facet}": {"n": int(r.n), "n_pos": int(r.n_pos),
                                                      "rate": round(r.rate, 3),
                                                      "disagree_rate": round(r.disrate, 3),
                                                      "n_anchor": int(r.n_anchor),
                                                      "tier": "core" if r.facet in core else "ext"}
                      for _, r in s[s.ok].iterrows()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}（{len(out)} 条 / {int(out.gold.sum())} 正例）")
    print(f"→ {SEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
