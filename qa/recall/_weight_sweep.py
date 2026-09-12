"""按块类型分权的 α 扫描：**文本块固定=生产权重（两路满权重）**，只调外部块（表格/公式）。

动机（用户 2026-09-11）：此前"给外部块屏蔽 BM25 路"被证否（表块恰恰靠 BM25 挣分）。
更细的做法是**只调外部块自身的两路权重**，文本块那一路完全不动 ——
代价只可能来自槽位竞争，而实测外部块平均只占 ~0.3 个槽，理论上代价极小。

RRF：score = Σ_routes weight_route / (60 + rank_route)
  · 文本块：权重 (1, 1) = 生产原样，**不动**
  · 外部块：权重 (2α, 2(1-α)) —— α=0.5 ⇒ (1,1)，与生产**逐位相同**
    α>0.5 偏向量路；α<0.5 偏 BM25 路

**关键性质：两路总量恒定** —— 外部块 (2α + 2(1-α)) ≡ 2 = 文本块 (1+1)。
所以 α 是「**配比**」旋钮，不是「放大」旋钮：α=0.75 ⇒ 外部块 向量:BM25 = 3:1；
α=0.25 ⇒ 1:3；α=1.0 ⇒ 纯向量路（总量仍为 2，与文本块等量）。

注：凸组合写法 `α·tv + (1-α)·tb`（文本块 α 固定 0.5）与本文件的 `(2α, 2(1-α))`
**只差一个全局因子 2 → argsort 完全相同**（已用两次运行的数值逐格核对确认）。
保留 (2α, 2(1-α)) 只是让 α=0.5 处**一眼看出**与生产的 (1,1) 相等。

两个面板（与 `_table_policy_ab.py` 同口径）：
  面板 1（表格收益）：A 桶 21 道可定位目标表的题 → 目标表块进 top-12 / 位次中位
  面板 2（挤占代价）：召回集 250 题，**按 gold 是否落在外部块切分**
     2b 纯文本 gold = 回答"挤占代价"；2c gold 在外部块 = 回答"表格收益"

用法：uv run python qa/recall/_weight_sweep.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))

# 注意：`_table_policy_ab` 在 import 时已设置 stdout 的 utf-8 wrapper；
# 这里**不要**再包一次 —— 两个 TextIOWrapper 共用一个 buffer 时，
# 先被 GC 的那个会关闭底层流，导致后续 `sys.stdout.reconfigure()` 抛
# "I/O operation on closed file"（首版就踩了这个坑）。
import _tgt  # noqa: E402  目标表定位口径（编号 ∪ 内容），见 qa/recall/_tgt.py
from _table_policy_ab import encode_new, new_chunks, text_vectors  # noqa: E402

TGT_MODE = os.environ.get("PP_TGT", "or")     # "num"（旧口径）/ "or"（新，默认）
from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402
from run_retrieval_eval import locate_gold, norm  # noqa: E402

ALPHAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
TOP_K, KS, MRR_K = 12, (8, 12, 16), 16
EQ = "norm"   # 与生产一致（表格 + 已规范化的公式）


def order_for(vs: np.ndarray, bs: np.ndarray, is_ext: np.ndarray,
              alpha_ext: float) -> np.ndarray:
    """文本块两路满权重；外部块权重 (2α, 2(1-α))。"""
    n = len(vs)
    rv = np.empty(n)
    rv[np.argsort(-vs, kind="stable")] = np.arange(n)
    rb = np.empty(n)
    rb[np.argsort(-bs, kind="stable")] = np.arange(n)
    tv, tb = 1.0 / (60 + rv + 1), 1.0 / (60 + rb + 1)
    wv = np.ones(n)
    wb = np.ones(n)
    wv[is_ext] = 2.0 * alpha_ext
    wb[is_ext] = 2.0 * (1.0 - alpha_ext)
    return np.argsort(-(wv * tv + wb * tb), kind="stable")


def main() -> int:
    papers = load_papers()
    L: list[str] = ["# 按块类型分权的 α 扫描（2026-09-11）", "",
                    "> **文本块**：两路满权重（= 生产 RRF，**完全不动**）；",
                    "> **外部块**（表格/公式）：权重 `(2α, 2(1-α))` —— α=0.5 ⇒ `(1,1)`，与生产**逐位相同**；",
                    "> α>0.5 偏向量路、α<0.5 偏 BM25 路。**两路总量恒定**（外部块 ≡ 文本块 = 2），",
                    "> 故 α 是「配比」旋钮而非「放大」旋钮：α=0.75 ⇒ 向量:BM25 = **3:1**；α=0.25 ⇒ 1:3。", ""]

    # ── 面板 1：表格收益 ──
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    p1: dict[float, list] = {a: [] for a in ALPHAS}
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
        ext = new_chunks(pid, EQ)
        chunks = list(texts) + ext
        vecs = np.vstack([tv, encode_new(ext)]) if ext else tv
        vs = (encode_query(q["question"]) @ vecs.T).ravel()
        bs = BM25Index([c.text for c in chunks]).score(q["question"])
        is_ext = np.array([str(c.chunk_id).startswith("xtbl") for c in chunks])
        # 目标表定位：编号 ∪ 内容（见 _tgt.py）；PP_TGT=num 复现旧口径
        tgt_set, _d = _tgt.locate(chunks, evs, _g, mode=TGT_MODE)
        for a in ALPHAS:
            order = order_for(vs, bs, is_ext, a)
            pos = next((i for i, c in enumerate(order)
                        if str(chunks[c].chunk_id) in tgt_set), None)
            p1[a].append(pos)
    L += ["## 面板 1：表格收益（A 桶 21 道可定位目标表的题）", "",
          "| 外部块 α | 目标表块进 top-12 | 位次中位 |", "|---|---|---|"]
    for a in ALPHAS:
        r = p1[a]
        hit = sum(1 for p in r if p is not None and p < TOP_K)
        med = np.median([p if p is not None else 10 ** 3 for p in r])
        L.append(f"| {a:.2f}{'（线上）' if a == 0.5 else ''} | {hit}/{len(r)} = "
                 f"{hit/max(len(r),1):.0%} | {med:.0f} |")

    # ── 面板 2：挤占代价（按 gold 位置切分） ──
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    by_pid: dict[str, list] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    acc = {a: {k: {"first": [], "ext": []} for k in ("all", "txt", "extg")} for a in ALPHAS}
    t0 = time.time()
    for pno, pid in enumerate(by_pid, 1):
        texts = ordered_chunks(f"qasper_{pid}.qpdf")
        if not texts:
            continue
        tv = text_vectors(pid, len(texts))
        if tv is None:
            continue
        ext = new_chunks(pid, EQ)
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
            vs = (encode_query(q["question"]) @ vecs.T).ravel()
            bs = bm.score(q["question"])
            key = "extg" if any(bool(is_ext[i]) for i in cg) else "txt"
            for a in ALPHAS:
                order = order_for(vs, bs, is_ext, a)
                first = next((i for i, c in enumerate(order) if c in cg), None)
                fv = first if first is not None else 10 ** 6
                n_ext = int(is_ext[order[:TOP_K]].sum())
                for k in ("all", key):
                    acc[a][k]["first"].append(fv)
                    acc[a][k]["ext"].append(n_ext)
        if pno % 60 == 0:
            print(f"  [{pno}/{len(by_pid)}] t={time.time()-t0:.0f}s", flush=True)

    def row(a: float, key: str) -> str:
        f = np.array(acc[a][key]["first"])
        e = np.array(acc[a][key]["ext"])
        if len(f) == 0:
            return f"| {a:.2f} | — | — | — | — | — |"
        return (f"| {a:.2f}{'（线上）' if a == 0.5 else ''} | "
                + " | ".join(f"{(f<k).mean():.3f}" for k in KS)
                + f" | {np.mean([1.0/(x+1) if x < MRR_K else 0 for x in f]):.3f} | {e.mean():.2f} |")

    for title, key in [("面板 2a：全部题（含 gold 落在外部块的题）", "all"),
                       ("面板 2b：仅纯文本 gold 子集 ← 回答「挤占代价」", "txt"),
                       ("面板 2c：gold 落在外部块的题 ← 回答「表格收益」", "extg")]:
        L += ["", f"## {title}（n={len(acc[ALPHAS[0]][key]['first'])}）", "",
              "| 外部块 α | R@8 | R@12 | R@16 | MRR@16 | top-12 内外部块均值 |",
              "|---|---|---|---|---|---|"]
        L += [row(a, key) for a in ALPHAS]

    txt = "\n".join(L)
    Path("qa/recall/WEIGHT_SWEEP_20260911.md").write_text(txt + "\n", encoding="utf-8")
    Path("qa/recall/weight_sweep_20260911.json").write_text(json.dumps(
        {"panel1": {str(a): {"hit12": sum(1 for p in p1[a] if p is not None and p < TOP_K),
                             "n": len(p1[a]),
                             "median_pos": float(np.median([p if p is not None else 10 ** 3
                                                            for p in p1[a]]))} for a in ALPHAS},
         "panel2": {str(a): {k: {
             "R@12": float((np.array(acc[a][k]["first"]) < 12).mean()) if acc[a][k]["first"] else 0.0,
             "MRR@16": float(np.mean([1.0/(x+1) if x < MRR_K else 0
                                      for x in acc[a][k]["first"]])) if acc[a][k]["first"] else 0.0,
             "ext_top12": float(np.mean(acc[a][k]["ext"])) if acc[a][k]["ext"] else 0.0,
             "n": len(acc[a][k]["first"])} for k in ("all", "txt", "extg")}
             for a in ALPHAS}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(txt)
    print("\n已写 qa/recall/WEIGHT_SWEEP_20260911.md + weight_sweep_20260911.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
