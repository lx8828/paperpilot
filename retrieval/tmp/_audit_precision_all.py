"""跨运行 precision 审计（①）：答案数 + precision，双官方口径

- **LoFT 官方**（我们用的基准）：em / coverage / subspan_em  ← `evaluation.rag` 直接 import
- **ALCE/HELMET 官方**：`compute_qampari_f1`（逐字移植 eval_alce.py）
  → 输出 num_preds（平均预测答案数）、prec、rec、rec_top5、f1  ← 这正是缺失的 precision

用户判据：预测答案数 ≫ gold 答案数 → recall 是虚的。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(__file__).resolve().parents[1]
OFF = R / "data" / "loft" / "official"
sys.path.insert(0, str(OFF))
from evaluation import rag as orag  # noqa: E402
from evaluation import utils as ou  # noqa: E402

QQ = R / "data" / "loft" / "qampari" / "128k" / "test_queries.jsonl"
gold_all = {json.loads(l)["qid"]: json.loads(l)["answers"]
            for l in QQ.read_text(encoding="utf-8").splitlines() if l.strip()}
GO = [len(v) for v in gold_all.values()]
print(f"官方 query 文件：{QQ.name} ｜ {len(gold_all)} 题 ｜ gold 均数 {np.mean(GO):.2f}"
      f"（min {min(GO)} / max {max(GO)}）")


def parse_pred(s) -> list[str]:
    s = str(s)
    try:
        v = json.loads(s)
    except Exception:  # noqa: BLE001
        try:
            v = json.loads(s + '"]')
        except Exception:  # noqa: BLE001
            return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v] if isinstance(v, list) else []


def alce_qampari(gold: list[str], pred: list[str]) -> dict:
    """逐字移植 HELMET/ALCE `eval_alce.py::compute_qampari_f1` 的算术（单样本）。

    注意：ALCE 里 gold 是"答案组列表"，LoFT 是扁平表 → 每个 gold 自成一組。
    """
    answers = [[g] for g in gold]                 # 组列表
    flat_answers = [x for sub in answers for x in sub]
    preds = [p for p in pred if len(p) > 0]
    prec = (sum(1 for p in preds if p in flat_answers) / len(preds)) if preds else 0.0
    rec = (sum(1 for a in answers if any(x in preds for x in a)) / len(answers)) if answers else 0.0
    hit = sum(1 for a in answers if any(x in preds for x in a))
    rec5 = min(5, hit) / min(5, len(answers)) if answers else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    f15 = 0.0 if (prec + rec5) == 0 else 2 * prec * rec5 / (prec + rec5)
    return dict(num_preds=len(preds), alceP=prec, alceR=rec, alceR5=rec5, alceF1=f1, alceF1_5=f15)


rows = []
for f in sorted((R / "results").glob("R2_QAMPARI_*.csv")):
    if "PRECISION" in f.name:
        continue
    d = pd.read_csv(f)
    acc = []
    for _, r in d.iterrows():
        gold = [ou.normalize_answer(x) for x in gold_all.get(r["qid"], [])]
        pred = [ou.normalize_answer(x) for x in parse_pred(r["pred_raw"])]
        gs, ps = set(gold), set(pred)
        tp = len(gs & ps)
        acc.append(dict(
            n_gold=len(gs), n_pred=len(ps),
            setP=tp / len(ps) if ps else 0.0,
            setR=tp / len(gs) if gs else 0.0,
            setF1=2 * tp / (len(ps) + len(gs)) if (ps and gs) else 0.0,
            em=float(gs == ps),
            subspan=orag.compute_multi_value_subspan_em(gold, pred),
            **alce_qampari(gold, pred),
        ))
    a = pd.DataFrame(acc)
    rows.append(dict(
        运行=f.stem.replace("R2_QAMPARI_", ""), n=len(a),
        gold均=a["n_gold"].mean(), 预测均=a["n_pred"].mean(),
        比值=a["n_pred"].mean() / a["n_gold"].mean(),
        预测gtgold=int((a["n_pred"] > a["n_gold"]).sum()),
        P=a["setP"].mean(), R=a["setR"].mean(), F1=a["setF1"].mean(),
        em=a["em"].mean(), subspan_em=a["subspan"].mean(),
        ALCE_P=a["alceP"].mean(), ALCE_R=a["alceR"].mean(),
        ALCE_R5=a["alceR5"].mean(), ALCE_F1=a["alceF1"].mean(),
    ))
t = pd.DataFrame(rows).sort_values("F1", ascending=False)

print("\n" + "=" * 124)
print("【A】答案数审计（用户判据：预测数 ≫ gold 数 → recall 虚）")
print(f"  {'运行':<20}{'n':>4}{'gold均':>8}{'预测均':>8}{'比值':>7}{'预测>gold':>10}{'判定':>12}")
for _, r in t.iterrows():
    if r["比值"] > 1.3:
        v = "⚠️ 明显列多"
    elif r["比值"] > 1.1:
        v = "· 略多"
    elif r["比值"] < 0.9:
        v = "· 列不够"
    else:
        v = "✅ 1:1 附近"
    print(f"  {r['运行']:<20}{int(r['n']):>4}{r['gold均']:>8.2f}{r['预测均']:>8.2f}"
          f"{r['比值']:>7.3f}{int(r['预测gtgold']):>10}{v:>12}")

print("\n" + "=" * 124)
print("【B】precision 双口径对照")
print(f"  {'运行':<20}{'集合P':>8}{'集合R':>8}{'集合F1':>8}{'Δ(R-P)':>9}"
      f"{'ALCE_P':>9}{'ALCE_R':>8}{'ALCE_R@5':>10}{'ALCE_F1':>9}{'em':>7}{'subspan':>9}")
for _, r in t.iterrows():
    print(f"  {r['运行']:<20}{r['P']:>8.3f}{r['R']:>8.3f}{r['F1']:>8.3f}{r['R'] - r['P']:>+9.3f}"
          f"{r['ALCE_P']:>9.3f}{r['ALCE_R']:>8.3f}{r['ALCE_R5']:>10.3f}{r['ALCE_F1']:>9.3f}"
          f"{r['em']:>7.3f}{r['subspan_em']:>9.3f}")

print("\n" + "=" * 124)
print("【C】判据结论")
bad = t[t["比值"] > 1.2]
if len(bad) == 0:
    print("  ✅ 没有任何运行的预测条数超过 gold 的 1.2 倍 → **recall 不是靠多列刷出来的**")
else:
    for _, r in bad.iterrows():
        print(f"  ⚠️ {r['运行']:<20} 预测 {r['预测均']:.2f} vs gold {r['gold均']:.2f}"
              f"（{r['比值']:.2f}×）｜ P {r['P']:.3f} vs R {r['R']:.3f}"
              f"（Δ {r['R'] - r['P']:+.3f}）→ **recall 有 {r['R'] - r['P']:.3f} 的水分**")
print(f"\n  最健康的（Δ(R-P) 最小）：")
for _, r in t.assign(D=lambda x: (x["R"] - x["P"]).abs()).nsmallest(3, "D").iterrows():
    print(f"    {r['运行']:<20} P {r['P']:.3f} R {r['R']:.3f} Δ {r['R'] - r['P']:+.3f}")
t.to_csv(R / "results" / "R2_QAMPARI_PRECISION_ALLRUNS.csv", index=False, encoding="utf-8")
print(f"\n已写 {R / 'results' / 'R2_QAMPARI_PRECISION_ALLRUNS.csv'}")
