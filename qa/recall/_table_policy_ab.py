"""表格通道策略 A/B（离线、零 LLM）：用**真指标**决定 #2「表格只走向量路」与 #3「cap 上限」要不要上。

两个面板：
  面板 1（**表格收益**）：A 桶 36 题（gold 出自表格）→ 目标表块**是否进 top-12**、平均位次
  面板 2（**挤占代价**）：召回集 250 题 → 正文 gold 的 R@12 / MRR@16，以及**表格块占掉几个槽**

变体（对应用户关心的两个开关）：
  A_eq_raw_mix     表格+公式(LaTeX 原始)，混池 RRF        ← 改动前形态
  B_eq_norm_mix    表格+公式(规范化)，混池 RRF            ← 只加规范化
  C_tbl_norm_mix   只表格，混池 RRF                       ← 去公式
  D_tbl_norm_vec   只表格 + **表格只走向量路**（#2）      ← C + #2
  E_D_cap2         同上 + **top-12 内表格块最多 2 个**（#3）← D + #3

工程：正文向量**复用 cvec 缓存**（缓存为"正文块 + xtbl-* 追加"），只编码新增块。

用法：uv run python qa/recall/_table_policy_ab.py [--panel 1|2|all]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, encode_query, encode_texts  # noqa: E402
from paperpilot.models.schema import Chunk  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402
from paperpilot.tools.mineru_bridge import (  # noqa: E402
    _find_content_list, element_text, latex_to_text)
from run_retrieval_eval import locate_gold, norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tgt  # noqa: E402  目标表定位口径（编号 ∪ 内容），见 qa/recall/_tgt.py

TGT_MODE = os.environ.get("PP_TGT", "or")     # "num"（旧口径）/ "or"（新，默认）
VIEW = Path("assets/artifacts/out_views")
TBL = Path("assets/artifacts/out_mineru")
KS, MRR_K = (8, 12, 16), 16
TOP_K = 12

VARIANTS = [
    ("A_eq_raw_mix", dict(eq="raw", vec_only=False, cap=0),
     "表格+公式(LaTeX 原始)，混池 RRF（改动前形态）"),
    ("B_eq_norm_mix", dict(eq="norm", vec_only=False, cap=0),
     "表格+公式(规范化)，混池 RRF（只加规范化）"),
    ("C_tbl_norm_mix", dict(eq="none", vec_only=False, cap=0),
     "只表格，混池 RRF（去公式）"),
    ("D_tbl_norm_vec", dict(eq="none", vec_only=True, cap=0),
     "C + **表格只走向量路**（#2）"),
    ("E_D_cap2", dict(eq="none", vec_only=True, cap=2),
     "D + **top-12 内表格块最多 2 个**（#3）"),
]

_enc_cache: dict[str, np.ndarray] = {}


def new_chunks(pid: str, eq: str) -> list[Chunk]:
    """按变体构建"外部块"（表格必然在；公式按 eq 模式 raw/norm/none）。"""
    cl = _find_content_list(TBL / f"{pid}v1")
    if not cl:
        return []
    out: list[Chunk] = []
    for i, el in enumerate(cl):
        typ = el.get("type")
        if typ == "table":
            t = element_text(el)
        elif typ == "equation" and eq != "none":
            tex = (el.get("text") or "").replace("$$", "").strip()
            t = f"公式: {latex_to_text(tex) if eq == 'norm' else tex}" if tex else None
        else:
            continue
        if not t:
            continue
        out.append(Chunk(chunk_id=f"xtbl-{i}", title_path=["(External)"],
                         page_span=(int(el.get("page_idx", 0)) + 1,) * 2, text=t[:4000],
                         n_blocks=1))
    return out


def text_vectors(pid: str, n_text: int) -> np.ndarray | None:
    """从 cvec 缓存取正文块向量（缓存 = 正文块 + xtbl-* 追加，正文在前 n_text 行）。"""
    f = VIEW / f"qasper_{pid}.cvec.npy"
    g = VIEW / f"qasper_{pid}.cidx.json"
    if not (f.exists() and g.exists()):
        return None
    try:
        ids = json.loads(g.read_text(encoding="utf-8")).get("ids")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(ids, list) or len(ids) < n_text:
        return None
    if any(str(x).startswith("xtbl") for x in ids[:n_text]):
        return None
    v = np.load(f)
    return v[:n_text] if len(v) >= n_text else None


def encode_new(chunks: list[Chunk]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 1024), dtype="float32")
    todo = [c for c in chunks if c.text not in _enc_cache]
    if todo:
        v = encode_texts([c.text for c in todo])
        for c, vec in zip(todo, v):
            _enc_cache[c.text] = vec
    return np.vstack([_enc_cache[c.text] for c in chunks])


def rank_order(vec_scores: np.ndarray, bm_scores: np.ndarray, is_ext: np.ndarray,
               vec_only: bool, cap: int) -> np.ndarray:
    """与生产 `embedder.rrf_order(skip_bm=...)` **同语义**：两路满权重求和，
    外部块在 vec_only 时**减去 BM25 路的贡献**（而不是给正文块减半）。

    ⚠️ 本函数首版写成 `a=1.0(外部块) / a=0.5(正文)`，等于**把正文块的贡献砍半**，
    使表块虚高到第 1 位——是公式 bug，不是 #2 的效果。已按生产语义修正。
    """
    n = len(vec_scores)
    rv = np.empty(n)
    rv[np.argsort(-vec_scores, kind="stable")] = np.arange(n)
    rb = np.empty(n)
    rb[np.argsort(-bm_scores, kind="stable")] = np.arange(n)
    term_v, term_b = 1.0 / (60 + rv + 1), 1.0 / (60 + rb + 1)
    rrf = term_v + term_b
    if vec_only:
        rrf[is_ext] -= term_b[is_ext]       # #2：外部块屏蔽 BM25 路
    order = list(np.argsort(-rrf, kind="stable"))
    if cap and cap > 0:                      # #3：top-12 内外部块最多 cap 个
        head, tail, used = [], [], 0
        for i in order:
            if is_ext[i]:
                if used < cap or len(head) >= TOP_K:
                    head.append(i)
                    used += 1
                else:
                    tail.append(i)
            else:
                head.append(i)
        order = head + tail
    return np.array(order)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="all", choices=["1", "2", "all"])
    args = ap.parse_args()
    papers = load_papers()
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    out: dict[str, dict] = {}

    # ── 面板 1：A 桶 36 题的"目标表块"能否进 top-12 ──
    if args.panel in ("1", "all"):
        print("=== 面板 1：A 桶 36 题，目标表块能否进 top-12 ===")
        abucket = exp
        res: dict[str, list] = {v[0]: [] for v in VARIANTS}
        for x in abucket:
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
            for name, cfg, _ in VARIANTS:
                ext = new_chunks(pid, cfg["eq"])
                chunks = list(texts) + ext
                vecs = np.vstack([tv, encode_new(ext)]) if ext else tv
                bm = BM25Index([c.text for c in chunks]).score(q["question"])
                is_ext = np.array([str(c.chunk_id).startswith("xtbl") for c in chunks])
                order = rank_order((encode_query(q["question"]) @ vecs.T).ravel(),
                                   bm, is_ext, cfg["vec_only"], cfg["cap"])
                # 目标表定位：编号 ∪ 内容（见 _tgt.py）；`PP_TGT=num` 复现旧口径
                tgt_set, _d = _tgt.locate(chunks, evs, _g, mode=TGT_MODE)
                pos = next((i for i, c in enumerate(order)
                            if str(chunks[c].chunk_id) in tgt_set), None)
                res[name].append(pos)
        for name, _, label in VARIANTS:
            r = res[name]
            hit = sum(1 for p in r if p is not None and p < TOP_K)
            med = np.median([p if p is not None else 10 ** 3 for p in r])
            print(f"  {name:<16} 目标表块进 top-12: {hit}/{len(r)} = {hit/max(len(r),1):.0%}"
                  f"｜位次中位 {med:.0f}｜{label}")
            out.setdefault(name, {})["panel1"] = {"n": len(r), "hit12": hit,
                                                  "median_pos": float(med)}

    # ── 面板 2：召回集 250 题的"挤占代价" ──
    if args.panel in ("2", "all"):
        print("\n=== 面板 2：召回集 250 题，正文 gold 的 R@k/MRR（看被挤占多少）===")
        data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
        by_pid: dict[str, list] = {}
        for it in data["items"]:
            by_pid.setdefault(it["pid"], []).append(it)
        pids = list(by_pid)
        # 归因隔离：B 与 F 只差一个开关（#2），才能把"MRR 恢复"归给 #2 而不是"去公式/去 cap"
        use = [("B_eq_norm_mix", dict(eq="norm", vec_only=False, cap=0)),
               ("F_eq_norm_vec", dict(eq="norm", vec_only=True, cap=0)),
               ("C_tbl_norm_mix", dict(eq="none", vec_only=False, cap=0)),
               ("D_tbl_norm_vec", dict(eq="none", vec_only=True, cap=0)),
               ("E_D_cap2", dict(eq="none", vec_only=True, cap=2))]
        acc = {n: {"first": [], "ext": []} for n, _ in use}
        t0 = time.time()
        for pno, pid in enumerate(pids, 1):
            texts = ordered_chunks(f"qasper_{pid}.qpdf")
            if not texts:
                continue
            tv = text_vectors(pid, len(texts))
            if tv is None:
                continue
            bm_texts = [c.text for c in texts]
            for name, cfg in use:
                ext = new_chunks(pid, cfg["eq"])
                chunks = list(texts) + ext
                vecs = np.vstack([tv, encode_new(ext)]) if ext else tv
                bm = BM25Index([c.text for c in chunks])
                is_ext = np.array([str(c.chunk_id).startswith("xtbl") for c in chunks])
                ntexts = [norm(c.text) for c in chunks]
                for it in by_pid[pid]:
                    q = next((qq for qq in papers[pid]["qas"]
                              if str(qq.get("question_id") or "") == it["qid"]), None)
                    if q is None:
                        continue
                    _g, evs = gold_answer_full(q)
                    cg: set[int] = set()
                    for e in evs:
                        gi, _ = locate_gold(ntexts, e)
                        if not gi:
                            try:
                                gi = {int(np.argmax(bm.score(norm(e))))}
                            except Exception:  # noqa: BLE001
                                gi = set()
                        cg |= gi
                    if not cg:
                        continue
                    order = rank_order((encode_query(q["question"]) @ vecs.T).ravel(),
                                       bm.score(q["question"]), is_ext, cfg["vec_only"], cfg["cap"])
                    first = next((i for i, c in enumerate(order) if c in cg), None)
                    acc[name]["first"].append(first if first is not None else 10 ** 6)
                    acc[name]["ext"].append(int(is_ext[order[:TOP_K]].sum()))
            if pno % 50 == 0:
                print(f"  [{pno}/{len(pids)}] t={time.time()-t0:.0f}s", flush=True)
        print("| 变体 | R@8 | R@12 | R@16 | MRR@16 | top-12 内表格块均值 |")
        print("|---|---|---|---|---|---|")
        for name, _ in use:
            f = np.array(acc[name]["first"])
            e = np.array(acc[name]["ext"])
            row = (f"{name} | " + " | ".join(f"{(f<k).mean():.3f}" for k in KS) +
                   f" | {np.mean([1.0/(x+1) if x < MRR_K else 0 for x in f]):.3f} | {e.mean():.2f} |")
            print("| " + row)
            out.setdefault(name, {})["panel2"] = {
                "R@12": float((f < 12).mean()),
                "MRR@16": float(np.mean([1.0/(x+1) if x < MRR_K else 0 for x in f])),
                "ext_in_top12": float(e.mean()), "n": int(len(f))}

    Path("qa/recall/table_policy_ab_20260911.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 qa/recall/table_policy_ab_20260911.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
