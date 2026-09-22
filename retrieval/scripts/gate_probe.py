"""门控信号探针：**能不能用"旧语料的最佳匹配分"判断该不该引入新语料？**

动机：直接把新语料并进候选，会让 LitSearch 的 R@5 掉 4.5pt（新语料占 top-100 的 40%）。
但那些是"语料内查询" —— 旧语料本来答得好。如果能在**查询级**先判断
"旧语料答得了吗"，就只在答不了时才放开新语料 → **理论上的兼得**。

候选信号（全部**免费**，检索时顺手就有）：
  · `s_old_top1`  = 旧语料 dense 余弦的**最高分**（查询与旧语料的最强匹配）
  · `s_old_gap`   = s_old_top1 − s_old_top1_in_new？不，用 s_old_top1 − s_new_top1
  · 两组的 s_old_top1 分布是否可分 → 用 AUC 衡量

判据：AUC 越接近 1，门控越可行；≈0.5 则不可分（门控没戏）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

from eval_multi_corpus import load_side  # noqa: E402

OLD = HERE / "data" / "litsearch" / "derived"
NEW = HERE / "data" / "arxiv"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(正例得分 > 负例得分)（Mann-Whitney，含并列按 0.5 计）。"""
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv-queries", default="arxiv_queries.parquet",
                    help="新语料查询集文件名（arxiv_queries_hard.parquet = 更难的那版）")
    ap.add_argument("--tag", default="", help="输出文件名后缀，用于区分两个查询集")
    args = ap.parse_args()
    print("载入索引…")
    m_old, _, _ = load_side(OLD, "corpusid")
    N_OLD = m_old.shape[0]
    m_new, _, _ = load_side(NEW, "docid")

    qa = pd.read_parquet(QFILE)
    qb = pd.read_parquet(NEW / "meta" / args.arxiv_queries)
    texts = qa["query"].astype(str).tolist() + qb["query"].astype(str).tolist()
    lab = np.array(["litsearch(旧语料能答)"] * len(qa) + ["arxiv(旧语料答不了)"] * len(qb))

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    model.max_seq_length = 512
    if dev == "cuda":
        model.half()
    print(f"编码 {len(texts)} 条查询…")
    QV = model.encode(texts, batch_size=128, normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    rows = []
    for i in range(len(texts)):
        so, sn = m_old @ QV[i], m_new @ QV[i]
        k = 5
        rows.append({
            "组": lab[i],
            "s_old_top1": float(so.max()),
            "s_new_top1": float(sn.max()),
            "s_old_top5_mean": float(np.sort(so)[-k:].mean()),
            "s_old_top1_minus_new_top1": float(so.max() - sn.max()),
        })
    d = pd.DataFrame(rows)

    print("\n" + "=" * 92)
    print("两组查询的信号分布（判断门控可行性）")
    print("=" * 92)
    print(d.groupby("组").describe().T.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 92)
    print("★ 门控可分性（AUC：用该信号区分'旧语料答得了/答不了'）")
    print("=" * 92)
    out = []
    for sig in ("s_old_top1", "s_old_top5_mean", "s_old_top1_minus_new_top1"):
        a = d[d["组"].str.startswith("litsearch")][sig].to_numpy()
        b = d[d["组"].str.startswith("arxiv")][sig].to_numpy()
        v = auc(a, b)               # 越大 = 该信号越高越可能是"旧语料能答"
        out.append({"信号": sig, "AUC": v, "判别力": "很强" if v > 0.9 else
                    "可用" if v > 0.75 else "弱" if v > 0.6 else "不可用"})
        print(f"  {sig:<28} AUC = {v:.4f}   ({out[-1]['判别力']})")

    best = max(out, key=lambda r: r["AUC"])
    print(f"\n  → 最佳信号 `{best['信号']}`，AUC = {best['AUC']:.4f}")

    # 用最佳信号扫阈值，看"正确放行率 vs 错误放行率"
    if best["AUC"] > 0.6:
        sig = best["信号"]
        a = d[d["组"].str.startswith("litsearch")][sig].to_numpy()
        b = d[d["组"].str.startswith("arxiv")][sig].to_numpy()
        print(f"\n  阈值扫描（信号={sig}；低于阈值 → 放行新语料）:")
        print(f"    {'阈值':>8} {'旧语料被误开新语料':>20} {'新查询正确放行':>16}")
        for th in np.percentile(np.concatenate([a, b]), [1, 5, 10, 20, 30, 50]):
            print(f"    {th:8.4f} {100 * (a < th).mean():19.1f}% {100 * (b < th).mean():15.1f}%")

    name = f"gate_probe{args.tag}.csv"
    d.to_csv(HERE / "results" / name, index=False, encoding="utf-8-sig")
    print(f"\n已写出：results/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
