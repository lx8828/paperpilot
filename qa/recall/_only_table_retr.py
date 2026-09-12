"""只检索表块（table-only pool）能比混合池好多少？—— 验证"分开检索"的检索层上限。

对 A 桶可定位目标表的题，比较目标表块的位次：
  1) 混合池 + RRF(α=0.5)   —— 现状
  2) 混合池 + RRF(α=0.75)  —— 软加权
  3) **表池内** RRF(α=0.5) —— 等价于"硬路由到表池"
  4) **表池内** 纯向量 / 纯 BM25

用法：uv run python qa/recall/_only_table_retr.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))

import _tgt  # noqa: E402  目标表定位口径（编号 ∪ 内容），见 qa/recall/_tgt.py
from _table_policy_ab import encode_new, new_chunks, text_vectors  # noqa: E402
from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

TGT_MODE = os.environ.get("PP_TGT", "or")       # "num"（旧口径）/ "or"（新，默认）


def ranks(vs, bs, alpha_ext, is_ext):
    n = len(vs)
    rv = np.empty(n)
    rv[np.argsort(-vs, kind="stable")] = np.arange(n)
    rb = np.empty(n)
    rb[np.argsort(-bs, kind="stable")] = np.arange(n)
    tv, tb = 1.0 / (60 + rv + 1), 1.0 / (60 + rb + 1)
    wv, wb = np.ones(n), np.ones(n)
    wv[is_ext], wb[is_ext] = 2 * alpha_ext, 2 * (1 - alpha_ext)
    return np.argsort(-(wv * tv + wb * tb), kind="stable")


def main() -> int:
    papers = load_papers()
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    rows = []
    n_src = {"num": 0, "cnt": 0, "only_cnt": 0}      # 定位来源统计（诊断口径影响）
    for x in exp:
        pid, qid = x["pid"], x["qid"]
        q = next((qq for qq in papers[pid]["qas"]
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        _g, evs = gold_answer_full(q)
        want = _tgt.gold_keys(evs)
        if not want:
            continue
        texts = ordered_chunks(f"qasper_{pid}.qpdf")
        tv = text_vectors(pid, len(texts))
        if tv is None:
            continue
        ext = new_chunks(pid, "norm")
        if not ext:
            continue
        chunks = list(texts) + ext
        vecs = np.vstack([tv, encode_new(ext)])
        vs = (encode_query(q["question"]) @ vecs.T).ravel()
        bs = BM25Index([c.text for c in chunks]).score(q["question"])
        is_ext = np.array([str(c.chunk_id).startswith("xtbl") for c in chunks])
        ext_idx = np.flatnonzero(is_ext)
        # 目标表定位：改用共享口径（编号 ∪ 内容；见 _tgt.py）。`PP_TGT=num` 可复现旧口径。
        tgt_set, diag = _tgt.locate_idx([c.text for c in chunks], evs, _g, mode=TGT_MODE,
                                        ext_mask=[bool(v) for v in is_ext])
        tgt = sorted(tgt_set)
        if not tgt:
            continue                       # 该篇定位不到目标表 → 与检索无关
        n_src["num"] += 1 if diag["by_num"] else 0
        n_src["cnt"] += 1 if diag["by_cnt"] else 0
        n_src["only_cnt"] += 1 if (diag["by_cnt"] and not diag["by_num"]) else 0

        def pos_mixed(order):
            return next((k for k, c in enumerate(order) if c in tgt), None)

        def pos_ext(sub_order):
            """在**表池内**的位次（0-based）。"""
            return next((k for k, c in enumerate(sub_order) if c in tgt), None)

        o50 = ranks(vs, bs, 0.5, is_ext)
        o75 = ranks(vs, bs, 0.75, is_ext)
        # 表池内排序：只在 ext 子集里重排名次
        rv_e = np.empty(len(ext_idx))
        rv_e[np.argsort(-vs[ext_idx], kind="stable")] = np.arange(len(ext_idx))
        rb_e = np.empty(len(ext_idx))
        rb_e[np.argsort(-bs[ext_idx], kind="stable")] = np.arange(len(ext_idx))
        s_rrf = 1.0/(60+rv_e+1) + 1.0/(60+rb_e+1)
        s_vec = -rv_e
        s_bm = -rb_e
        order_in_pool = {k: [int(ext_idx[j]) for j in np.argsort(-v, kind="stable")]
                         for k, v in (("rrf", s_rrf), ("vec", s_vec), ("bm", s_bm))}
        rows.append({
            "qid": qid, "pid": pid, "n_ext": len(ext),
            "mixed50": pos_mixed(o50), "mixed75": pos_mixed(o75),
            "pool_rrf": pos_ext(order_in_pool["rrf"]),
            "pool_vec": pos_ext(order_in_pool["vec"]),
            "pool_bm": pos_ext(order_in_pool["bm"]),
        })
    print(f"n = {len(rows)} 题（可定位目标表）| 口径 PP_TGT={TGT_MODE}"
          f"（num=仅编号 / or=编号∪内容）")
    print(f"  定位来源：编号命中 {n_src['num']} 题 | 内容命中 {n_src['cnt']} 题 "
          f"| **仅靠内容救回** {n_src['only_cnt']} 题")
    print(f"表池平均大小 = {np.mean([r['n_ext'] for r in rows]):.1f} 个外部块\n")
    print("| 指标 | 混合池 α=0.5（现状） | 混合池 α=0.75 | **表池内** RRF | 表池内 纯向量 | 表池内 纯BM25 |")
    print("|---|---|---|---|---|---|")

    def agg(key):
        v = [r[key] for r in rows]
        v = [x if x is not None else 10 ** 6 for x in v]
        return v
    cols = [("mixed50", "混合 α=0.5"), ("mixed75", "混合 α=0.75"),
            ("pool_rrf", "表池 RRF"), ("pool_vec", "表池 向量"), ("pool_bm", "表池 BM25")]
    for name, th in [("进 top-1", 1), ("进 top-3", 3), ("进 top-5", 5), ("进 top-12", 12)]:
        cells = []
        for k, _ in cols:
            v = agg(k)
            cells.append(f"{sum(1 for x in v if x < th)}/{len(v)} = {sum(1 for x in v if x < th)/len(v):.0%}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    cells = []
    for k, _ in cols:
        cells.append(f"{np.median(agg(k)):.0f}")
    print("| 位次中位 | " + " | ".join(cells) + " |")
    Path("qa/recall/only_table_retr_20260911.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写 qa/recall/only_table_retr_20260911.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
