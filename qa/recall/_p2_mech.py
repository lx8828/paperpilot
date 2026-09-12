"""验证 P2 双写变差的机制：摘要会不会让**同篇内的表块互相更像**（区分度下降）。

做法：直接用两臂已建好的向量缓存（`out_views` 原表 vs `out_views_p2` 摘要），
对每篇 A 桶论文取 xtbl-* 子集，算**块间平均余弦**与**目标块与其它表块的 margin**。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

import _tgt  # noqa: E402
from paperpilot.agents.document_cache import retrieval_chunks  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

OFF = Path("assets/artifacts/out_views")
ON = Path(os.environ.get("PP_ALT_DIR") or "assets/artifacts/out_views_p2")   # 摘要臂缓存目录


def mean_pairwise(m: np.ndarray) -> float:
    if len(m) < 2:
        return float("nan")
    s = m @ m.T
    iu = np.triu_indices(len(m), k=1)
    return float(s[iu].mean())


def margin(m: np.ndarray, tgt_idx: list[int], qvec: np.ndarray) -> float | None:
    """目标块向量相似度 − 其它表块最高相似度（正 = 目标能排第一）。"""
    if not tgt_idx or len(m) < 2:
        return None
    s = m @ qvec
    others = [j for j in range(len(m)) if j not in tgt_idx]
    if not others:
        return None
    return float(s[tgt_idx].max() - s[others].max())


def main() -> int:
    papers = load_papers()
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    from paperpilot.agents.embedder import encode_query
    rows = []
    for x in exp:
        pid, qid = x["pid"], x["qid"]
        stem = f"qasper_{pid}"
        vo, vn = OFF / f"{stem}.cvec.npy", ON / f"{stem}.cvec.npy"
        co, cn = OFF / f"{stem}.cidx.json", ON / f"{stem}.cidx.json"
        if not (vo.exists() and vn.exists() and co.exists() and cn.exists()):
            continue
        ids = json.loads(co.read_text(encoding="utf-8"))["ids"]
        a, b = np.load(vo), np.load(vn)
        if len(a) != len(b) or len(ids) != len(a):
            continue
        ext = [i for i, cid in enumerate(ids) if str(cid).startswith("xtbl")]
        if len(ext) < 2:
            continue
        q = next((qq for qq in papers[pid]["qas"]
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        gold, evs = gold_answer_full(q)
        cs = retrieval_chunks(f"{stem}.qpdf")
        tgt, _ = _tgt.locate(cs, evs, gold, mode="or")
        tids = [i for i, cid in enumerate(ids) if str(cid) in tgt]
        tgt_local = [ext.index(i) for i in tids if i in ext]
        qv = encode_query(q["question"])
        rows.append({
            "pid": pid, "qid": qid, "n_ext": len(ext),
            "raw_sim": mean_pairwise(a[ext]), "sum_sim": mean_pairwise(b[ext]),
            "raw_margin": margin(a[ext], tgt_local, qv),
            "sum_margin": margin(b[ext], tgt_local, qv),
        })
    print(f"n = {len(rows)} 题（有 xtbl 子集且两臂缓存齐）\n")
    rs = np.array([r["raw_sim"] for r in rows])
    ss = np.array([r["sum_sim"] for r in rows])
    print(f"同篇表块**块间平均余弦**：原表 {rs.mean():.3f} | 摘要 {ss.mean():.3f} "
          f"（+{ss.mean()-rs.mean():+.3f} = 更像 = 区分度更低）")
    print(f"  逐篇变高 {int((ss>rs).sum())}/{len(rows)} 篇\n")
    rm = np.array([r["raw_margin"] if r["raw_margin"] is not None else np.nan for r in rows])
    sm = np.array([r["sum_margin"] if r["sum_margin"] is not None else np.nan for r in rows])
    ok = ~np.isnan(rm) & ~np.isnan(sm)
    print(f"目标表 vs 其它表块的**向量侧 margin**（>0 = 目标排第一）：")
    print(f"  原表 中位 {np.nanmedian(rm):+.3f} | 摘要 中位 {np.nanmedian(sm):+.3f}")
    print(f"  变差 {int((sm[ok] < rm[ok]).sum())} / 变好 {int((sm[ok] > rm[ok]).sum())} / 共 {int(ok.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
