"""**F1 自算 + subspan_em 一致性检验**（用官方 runner 已验证过的那份预测）

用户给的判据：
  · F1 ≫ subspan_em（如 F1 0.85+ vs subspan_em 0.70）→ 部分匹配表现好，**合理**
  · F1 ≪ subspan_em → **subspan_em 的计算逻辑可能有问题**

额外做三件事：
  ① **合成单元测试**：完美/空/部分/超集/反向，看 subspan_em 是否按定义给分（尤其"超集=1"证明它不看 precision）
  ② **双向子串匹配**的宽松度：分别算「严格集合匹配的 recall」与「子串匹配的 recall」，看差多少
  ③ **逐题一致性**：subspan_em 应与「subspan 覆盖率==1」的题占比接近（同一逻辑的两种写法）
"""
from __future__ import annotations

import json
import string
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parents[1]


# ── 官方口径（与 evaluation/utils.py 逐行一致）──
def normalize_answer(s: str) -> str:
    import re
    s = unicodedata.normalize("NFD", str(s))
    s = " ".join(s.split())
    s = re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.lower().split())


def subspan_matrix(gold, pred):
    m = np.zeros([len(gold), len(pred)])
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if g in p or p in g:
                m[i, j] = 1
    return m


def subspan_em(gold, pred) -> float:
    """官方 evaluation/rag.py::compute_multi_value_subspan_em"""
    if not gold or not pred:
        return 0.0
    sc = subspan_matrix(gold, pred)
    r, c = linear_sum_assignment(-sc)
    al = np.zeros(len(gold))
    for ri, ci in zip(r, c):
        al[ri] = sc[ri, ci]
    return float(all(al))


def subspan_recall(gold, pred) -> float:
    """宽松版：每个 gold 是否**至少**被某个 pred 匹配上（不要求互斥分配）"""
    if not gold or not pred:
        return 0.0
    m = subspan_matrix(gold, pred)
    return float(m.max(axis=1).mean())


def set_prf(gold, pred):
    tp = len(set(gold) & set(pred))
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def answer_level_f1(gold, pred):
    """SQuAD 式：每个 pred 取与 gold 的最大 token-F1，再对 pred 平均"""
    import collections

    def t(s):
        return normalize_answer(s).split()

    out = []
    for p in pred:
        pt = t(p)
        best = 0.0
        for g in gold:
            gt = t(g)
            if not gt or not pt:
                best = max(best, float(gt == pt))
                continue
            n = sum((collections.Counter(gt) & collections.Counter(pt)).values())
            if n == 0:
                continue
            pr, rc = n / len(pt), n / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
        out.append(best)
    return float(np.mean(out)) if out else 0.0


def main() -> int:
    print("【① 合成单元测试：subspan_em 是否按定义给分】")
    cases = [
        ("完美", ["a b", "c d"], ["a b", "c d"], 1.0),
        ("少一个（partial）", ["a b", "c d"], ["a b"], 0.0),
        ("**超集（多答了很多）**", ["a b"], ["a b", "x", "y", "z"], 1.0),
        ("**把 gold 包在长句里**", ["paris"], ["the capital is paris indeed"], 1.0),
        ("全错", ["a b", "c d"], ["x y", "z w"], 0.0),
        ("空预测", ["a b"], [], 0.0),
        ("顺序打乱", ["a", "b", "c"], ["c", "b", "a"], 1.0),
    ]
    for nm, g, p, exp in cases:
        got = subspan_em(g, p)
        print(f"  {nm:<22} gold={g} pred={p} → {got:.0f}（预期 {exp:.0f}）{'✅' if got == exp else '❌'}")

    df = pd.read_csv(HERE / "results" / "R2_QAMPARI_audit_k40_k40.csv")
    qf = {json.loads(l)["qid"]: json.loads(l)
          for l in (HERE / "data" / "loft" / "qampari" / "128k" / "test_queries.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()}

    rows = []
    for _, r in df.iterrows():
        gold = [normalize_answer(x) for x in qf[r["qid"]]["answers"]]
        pred = [normalize_answer(x) for x in json.loads(str(r["_full_pred"]))] if "_full_pred" in df.columns \
            else [normalize_answer(x) for x in json.loads(str(r["pred_raw"]) + ('"]' if not str(r["pred_raw"]).rstrip().endswith("]") else ""))]
        p, rec, f1 = set_prf(gold, pred)
        rows.append(dict(qid=r["qid"], n_gold=len(gold), n_pred=len(pred),
                         subspan_em=subspan_em(gold, pred),
                         subspan_rec=subspan_recall(gold, pred),
                         exact_cov=rec, setP=p, setF1=f1,
                         ansF1=answer_level_f1(gold, pred)))
    a = pd.DataFrame(rows)

    n = len(a)
    print(f"\n【② 完整对照表】（同一份预测，{n} 题；官方 runner 已验证 subspan_em=0.69/em=0.47/coverage=0.8148）")
    print(f"  {'指标':<34}{'均值':>8}  说明")
    items = [
        ("subspan_em（官方主指标·全有或全无）", a["subspan_em"].mean(), "5 个 gold 全被匹配才给 1；**不看 precision**"),
        ("subspan 覆盖率（逐 gold 匹配率）", a["subspan_rec"].mean(), "把'all-or-nothing'拆成逐 gold 平均"),
        ("严格集合 recall（=官方 coverage）", a["exact_cov"].mean(), "只认字符串完全相等（官方分母=非空行）"),
        ("严格集合 precision", a["setP"].mean(), "预测里有多少是对的"),
        ("**集合级 F1**", a["setF1"].mean(), "2PR/(P+R)，逐题算再平均"),
        ("**答案级 F1**（SQuAD 式）", a["ansF1"].mean(), "逐 pred 取与 gold 的最大 token-F1"),
        ("frac(严格 coverage == 1)", (a["exact_cov"] >= 1 - 1e-9).mean(),
         "预测集合**完全覆盖** gold 的题占比"),
    ]
    for nm, v, note in items:
        print(f"  {nm:<34}{v:>8.4f}  {note}")

    print("\n【③ 你说的判据检验】")
    se, f1s, f1a = a["subspan_em"].mean(), a["setF1"].mean(), a["ansF1"].mean()
    print(f"  subspan_em {se:.4f} ｜ 集合级 F1 {f1s:.4f}（{f1s - se:+.4f}）"
          f" ｜ 答案级 F1 {f1a:.4f}（{f1a - se:+.4f}）")
    verdict = "✅ **合理**：F1 > subspan_em，且部分匹配拿到大量分数" if (f1s > se and f1a > se) \
        else "⚠️ 需要排查"
    print(f"  → F1 {'高于' if f1s > se else '低于'} subspan_em → {verdict}")

    print("\n【④ 三个内部一致性检查】")
    print(f"  a) subspan_em({se:.3f}) ≥ frac(coverage==1)({(a['exact_cov'] >= 1 - 1e-9).mean():.3f}) ?"
          f"  {'✅（子串匹配比严格相等宽松，必须成立）' if se >= (a['exact_cov'] >= 1 - 1e-9).mean() else '❌'}")
    print(f"  b) subspan 覆盖率({a['subspan_rec'].mean():.3f}) ≥ 严格 recall({a['exact_cov'].mean():.3f}) ?"
          f"  {'✅（同上）' if a['subspan_rec'].mean() >= a['exact_cov'].mean() else '❌'}")
    print(f"  c) subspan_em 与 frac(subspan覆盖率==1) 之差："
          f"{se - (a['subspan_rec'] >= 1 - 1e-9).mean():+.4f}"
          f"（互斥分配要求 → 前者可略低，**不应差很多**）")

    print("\n【⑤ 拿一题把 subspan_em 手算给你看】")
    r0 = a.iloc[0]
    gold = [normalize_answer(x) for x in qf[r0["qid"]]["answers"]]
    pred = [normalize_answer(x) for x in json.loads(str(df.iloc[0]["pred_raw"]))]
    m = subspan_matrix(gold, pred)
    print(f"  qid={r0['qid']} ｜ subspan_em={r0['subspan_em']:.0f} ｜ setF1={r0['setF1']:.3f}")
    print(f"  gold({len(gold)}): {gold}")
    print(f"  pred({len(pred)}): {pred}")
    print("  匹配矩阵（行=gold，列=pred，1=互为子串）：")
    for i, g in enumerate(gold):
        print(f"    {g[:44]:<46}{''.join(str(int(x)) for x in m[i])}")
    print(f"  逐 gold 是否被匹配：{m.max(axis=1).astype(int).tolist()} → all={bool(m.max(axis=1).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
