"""表块表征变体实测：去重 / 加 refs / 纯尺寸填充，谁在表池里排得更好？

变体（都基于 `element_text(el)` = cap+body+footnote）：
  V0 现状     : cap + "\\n" + t        ← 生产现状（caption 重复一遍）
  V1 去重     : t
  V2 +refs全  : t + "\\n[引用] " + 全部 refs（最长可达 ~2300 字）
  V3 +refs截断: t + "\\n[引用] " + 首段 refs（<=400 字）
  V4 纯填充   : t + "\\n" + 150 字无关填充（**只测尺寸效应**，回答"加大是否影响检索"）

指标：表池内目标表位次（进 top-1/3/5）、块长分布、是否触顶 MAX_CHUNK_LEN。

用法：uv run python qa/recall/_ext_variants.py
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
from _table_policy_ab import TBL, text_vectors  # noqa: E402

TGT_MODE = os.environ.get("PP_TGT", "or")     # "num"（旧口径）/ "or"（新，默认）
from paperpilot.agents.document_cache import MAX_CHUNK_LEN, ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, encode_query, encode_texts  # noqa: E402
from paperpilot.models.schema import Chunk  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402
from paperpilot.tools.mineru_bridge import (_find_content_list,  # noqa: E402
                                            element_text, extract_figures_from_mineru)

FILLER = ("This table reports the main quantitative results discussed in the paper. "
          "The values are grouped by experimental setting and are intended to be read "
          "together with the surrounding analysis and the corresponding figures.")
VARIANTS = ["V0_现状(含重复)", "V1_去重", "V2_去重+refs全", "V3_去重+refs截断400", "V4_去重+150字填充"]


def build(pid: str) -> dict[str, list[str]]:
    """返回 {变体名: [块文本, ...]}（只含 table 元素，顺序与 cl 一致）。"""
    cl = _find_content_list(TBL / f"{pid}v1")
    if not cl:
        return {}
    figs = [f for f in extract_figures_from_mineru(cl) if f["kind"] == "table"]
    out: dict[str, list[str]] = {v: [] for v in VARIANTS}
    fi = 0
    for el in cl:
        if el.get("type") != "table":
            continue
        t = element_text(el)
        fig = figs[fi] if fi < len(figs) else {"refs": []}
        fi += 1
        if not t:
            continue
        cap = " ".join(el.get("table_caption") or [])
        refs = [r for r in (fig.get("refs") or []) if r]
        out["V0_现状(含重复)"].append(f"{cap}\n{t}" if cap else t)
        out["V1_去重"].append(t)
        out["V2_去重+refs全"].append(t + ("\n[引用] " + " ".join(refs) if refs else ""))
        r1 = refs[0][:400] if refs else ""
        out["V3_去重+refs截断400"].append(t + (f"\n[引用] {r1}" if r1 else ""))
        out["V4_去重+150字填充"].append(t + "\n" + FILLER)
    return out


def pool_rank(qvec: np.ndarray, texts: list[str], q: str,
              cg: set[int]) -> tuple[int | None, int | None, int | None]:
    """表池内位次：(RRF, 纯向量, 纯 BM25)。"""
    if not texts:
        return None, None, None
    vecs = encode_texts(texts)
    vs = (qvec @ vecs.T).ravel()
    bs = BM25Index(texts).score(q)
    n = len(texts)

    def pos(order):
        return next((k for k, c in enumerate(order) if c in cg), None)

    rv = np.empty(n)
    rv[np.argsort(-vs, kind="stable")] = np.arange(n)
    rb = np.empty(n)
    rb[np.argsort(-bs, kind="stable")] = np.arange(n)
    rrf = np.argsort(-(1.0 / (60 + rv + 1) + 1.0 / (60 + rb + 1)), kind="stable")
    return pos(rrf), pos(np.argsort(-vs, kind="stable")), pos(np.argsort(-bs, kind="stable"))


def main() -> int:
    papers = load_papers()
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    res: dict[str, list] = {v: [] for v in VARIANTS}     # RRF 位次
    r_vec: dict[str, list] = {v: [] for v in VARIANTS}   # 纯向量位次
    r_bm: dict[str, list] = {v: [] for v in VARIANTS}    # 纯 BM25 位次
    lens: dict[str, list] = {v: [] for v in VARIANTS}
    n_trunc = {v: 0 for v in VARIANTS}
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
        texts_t = ordered_chunks(f"qasper_{pid}.qpdf")
        tv = text_vectors(pid, len(texts_t))
        if tv is None:
            continue
        variants = build(pid)
        if not variants:
            continue
        qvec = encode_query(q["question"])
        for v in VARIANTS:
            ts = variants[v]
            # 目标块下标：编号 ∪ 内容（见 _tgt.py）；PP_TGT=num 复现旧口径
            cg, _d = _tgt.locate_idx(ts, evs, _g, mode=TGT_MODE)
            if not cg:
                continue
            for t in ts:
                lens[v].append(len(t))
                if len(t) > MAX_CHUNK_LEN:
                    n_trunc[v] += 1
            a, b, c = pool_rank(qvec, ts, q["question"], cg)
            res[v].append(a)
            r_vec[v].append(b)
            r_bm[v].append(c)

    print("| 变体 | 表池 top-1 | top-3 | top-5 | 块长中位 | 块长 max | 超 4000 的块 |")
    print("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        r = [x if x is not None else 10 ** 6 for x in res[v]]
        if not r:
            continue
        L = np.array(lens[v])
        print(f"| {v} | {sum(1 for x in r if x < 1)}/{len(r)} = {sum(1 for x in r if x < 1)/len(r):.0%} "
              f"| {sum(1 for x in r if x < 3)}/{len(r)} = {sum(1 for x in r if x < 3)/len(r):.0%} "
              f"| {sum(1 for x in r if x < 5)}/{len(r)} = {sum(1 for x in r if x < 5)/len(r):.0%} "
              f"| {np.median(L):.0f} | {L.max()} | {n_trunc[v]} |")
    print(f"\n（块长为 **全部表块** 的分布，非仅目标块；MAX_CHUNK_LEN={MAX_CHUNK_LEN}）")

    def rk(d, v, key=""):
        return np.array([x if x is not None else 10 ** 6 for x in d[v]])

    print("\n### 敏感指标：目标块的**位次中位**（越小越好）+ 位次变差的题数")
    print("| 变体 | RRF 位次中位 | 向量位次中位 | BM25位次中位 | RRF变差题数(vs 去重) | 向量变差 | BM25变差 |")
    print("|---|---|---|---|---|---|---|")
    base_a, base_b, base_c = rk(res, "V1_去重"), rk(r_vec, "V1_去重"), rk(r_bm, "V1_去重")
    for v in VARIANTS:
        a, b, c = rk(res, v), rk(r_vec, v), rk(r_bm, v)
        print(f"| {v} | {np.median(a):.1f} | {np.median(b):.1f} | {np.median(c):.1f} "
              f"| {int((a > base_a).sum())} ({int((a > base_a).sum())-int((a < base_a).sum()):+d}) "
              f"| {int((b > base_b).sum())-int((b < base_b).sum()):+d} "
              f"| {int((c > base_c).sum())-int((c < base_c).sum()):+d} |")
    print("\n> 「变差题数」= 位次变大(更差)的题数 − 位次变小(更好)的题数，基准为 V1_去重；正数=整体变差。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
