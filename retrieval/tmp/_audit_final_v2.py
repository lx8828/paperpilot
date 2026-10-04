"""**最终审计表**：官方 CLI 结果 + 答案数 + precision + coverage 分母陷阱 + 历史产物对账"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(__file__).resolve().parents[1]
OFF = R / "data" / "loft" / "official"
RUNS = R / "results" / "official_runs"


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


rows = []
for d in sorted(RUNS.iterdir()):
    if not d.is_dir():
        continue
    q = {x["qid"]: x for x in jl(d / "queries.jsonl")}
    p = {x["qid"]: x for x in jl(d / "preds.jsonl")}
    m = json.loads((d / "preds_metrics.json").read_text(encoding="utf-8"))
    n_empty = sum(1 for x in p.values() if not (x.get("model_outputs") or [[]])[0])
    pl = jl(d / "preds_metrics_per_line.jsonl")
    cov_rows = [x["coverage"] for x in pl if "coverage" in x]
    n_g = np.mean([len(q[k].get("answers", [])) for k in q])
    n_p = np.mean([len((p[k].get("model_outputs") or [[]])[0]) for k in q if k in p])
    rows.append(dict(
        运行=d.name, 题数=len(q), 空预测题=n_empty,
        gold均=n_g, 预测均=n_p, 比值=n_p / n_g,
        官方em=m["quality"]["em"], 官方subspan=m["quality"]["subspan_em"],
        官方coverage_报道=m["quality"]["coverage"],          # 官方脚本报的（只在非空行）
        coverage_全行=float(np.mean([x.get("coverage", 0.0) for x in pl])),  # 全 100 行口径
        coverage_非空行=float(np.mean(cov_rows)) if cov_rows else float("nan"),
    ))
t = pd.DataFrame(rows).sort_values("官方subspan", ascending=False)
print("=" * 126)
print("【最终审计表｜全部由官方 LoFT CLI 产出】`results/official_runs/<run>/`")
print(f"  {'运行':<20}{'空预测':>7}{'gold均':>8}{'预测均':>8}{'比值':>7}"
      f"{'官方em':>8}{'官方subspan':>11}{'coverage报道':>12}{'coverage全行':>12}")
for _, r in t.iterrows():
    print(f"  {r['运行']:<20}{int(r['空预测题']):>7}{r['gold均']:>8.2f}{r['预测均']:>8.2f}"
          f"{r['比值']:>7.3f}{r['官方em']:>8.3f}{r['官方subspan']:>11.3f}"
          f"{r['官方coverage_报道']:>12.4f}{r['coverage_全行']:>12.4f}")

print("\n" + "=" * 126)
print("【★ 陷阱 1：官方 coverage 的分母随空预测数变化 → 跨运行不可比】")
bad = t[t["空预测题"] > 5].sort_values("空预测题", ascending=False)
if len(bad):
    for _, r in bad.iterrows():
        print(f"  {r['运行']:<20} 空预测 {int(r['空预测题'])}/100 → 官方 coverage 只在"
              f" {100 - int(r['空预测题'])} 行上求均 = **{r['官方coverage_报道']:.4f}**"
              f"（看起来最好！）但全行口径只有 **{r['coverage_全行']:.4f}**，"
              f"且 em {r['官方em']:.2f} / subspan {r['官方subspan']:.2f} 是**最差**")
print("\n  → 结论：**coverage 绝不能单独报**；必须同时给 空预测题数 / em / subspan_em")

print("\n" + "=" * 126)
print("【★ 陷阱 2：ALCE(HELMET) 的 P/R 口径是安全的（空预测记 0，分母恒定 100）】")
oa = pd.read_csv(R / "results" / "R2_QAMPARI_OFFICIAL_ALLRUNS.csv")
for _, r in oa.sort_values("ALCE_F1", ascending=False).iterrows():
    flag = ""
    if r["比值"] > 1.2:
        flag = f"  ⚠️ 多列 {r['比值']:.2f}× → R {r['ALCE_R']:.3f} 有 {r['ALCE_R'] - r['ALCE_P']:+.3f} 水分"
    elif r["比值"] < 0.9:
        flag = f"  · 列不够 {r['比值']:.2f}× → R {r['ALCE_R']:.3f} 被低估"
    print(f"  {r['运行']:<20} P {r['ALCE_P']:.3f}  R {r['ALCE_R']:.3f}  F1 {r['ALCE_F1']:.3f}{flag}")

print("\n" + "=" * 126)
print("【历史产物对账：两份存档分别对应哪个运行？】")
hist = {
    "data/loft/official/audit/preds_metrics.json":
        json.loads((OFF / "audit" / "preds_metrics.json").read_text(encoding="utf-8"))["quality"],
    "results/official_audit_k40_ascii/preds_metrics.json":
        json.loads((R / "results" / "official_audit_k40_ascii" / "preds_metrics.json")
                   .read_text(encoding="utf-8"))["quality"],
}
for nm, q in hist.items():
    print(f"\n  存档 {nm}")
    print(f"    em={q['em']} coverage={q['coverage']:.4f} subspan_em={q['subspan_em']}")
    hit = t[(np.isclose(t["官方em"], q["em"])) & (np.isclose(t["官方subspan"], q["subspan_em"]))
            & (np.isclose(t["官方coverage_报道"], q["coverage"], atol=2e-3))]
    if len(hit):
        for _, r in hit.iterrows():
            print(f"    → ✅ 与官方重跑的 **{r['运行']}** 一致"
                  f"（em {r['官方em']} / cov {r['官方coverage_报道']:.4f} / subspan {r['官方subspan']}）")
    else:
        print("    → ⚠️ 无匹配运行（审计链待补）")
t.to_csv(R / "results" / "R2_QAMPARI_AUDIT_FINAL.csv", index=False, encoding="utf-8")
print(f"\n已写 {R / 'results' / 'R2_QAMPARI_AUDIT_FINAL.csv'}")
