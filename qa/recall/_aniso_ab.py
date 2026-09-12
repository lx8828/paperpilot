"""各向异性去除 A/B（离线打分解耦实验，零 LLM 生成、零文档重编码）。

问题：单篇内 query→chunk 余弦分辨率低（gold 0.479 vs 干扰 0.426，仅差 0.053），
怀疑 bge-m3 各向异性（所有文本共享一个大共同方向）压窄了有效动态范围。

做法：**不改文档向量本身**，只在打分时对块向量做不同后处理，比较检索排序质量：
  raw            基线：cos(q, v)
  center_global  全局去均值：v'=normalize(v−μ)，μ=全语料均值；q'=normalize(q−μ)
  center_paper   篇内去均值：v'=normalize(v−μ_p)，μ_p=该篇块均值；score=cos(q, v')
  abtt_paper     篇内去第一主成分（all-but-the-top 简化）
  whiten_global  全局白化：v'=normalize(Σ^{-1/2}(v−μ))（Σ 加收缩）

同时输出「诊断量」解释为何 gold 不显著高：
  每题分数 spread(std/range) 与 gold−干扰 gap 的比值（gap/spread 越小越接近噪声）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

import numpy as np  # noqa: E402

import run_retrieval_eval as ree  # noqa: E402
from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

KS = (1, 8, 12, 16)
VARIANTS = ("raw", "center_global", "center_paper", "abtt_paper", "whiten_global")


def norm_rows(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def first_gold_rank(order: np.ndarray, cg: set[int]) -> int | None:
    for p in range(len(order)):
        if int(order[p]) in cg:
            return p
    return None


def auc1(vs: np.ndarray, lab: np.ndarray) -> float:
    pos, neg = vs[lab], vs[~lab]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    pids = list(by_pid.keys())

    # 预载全部块向量（含 chunk 文本用于 gold 定位）
    vec_by_pid: dict[str, np.ndarray] = {}
    ch_by_pid: dict[str, list] = {}
    all_vecs: list[np.ndarray] = []
    for pid in pids:
        ch = ordered_chunks(f"qasper_{pid}.qpdf")
        v = ChunkIndex(f"qasper_{pid}.qpdf").vectors()
        vec_by_pid[pid] = v
        ch_by_pid[pid] = ch
        all_vecs.append(v)
    V = np.concatenate(all_vecs, axis=0)
    mu = V.mean(axis=0, keepdims=True)
    mu_n = mu / (np.linalg.norm(mu) + 1e-9)
    # 白化矩阵（收缩：Σ + λ·mean(diag)·I）
    Xc = V - mu
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    lam = 0.1 * float(np.mean(np.diag(cov)))
    cov_s = cov + lam * np.eye(cov.shape[0], dtype=cov.dtype)
    w, U = np.linalg.eigh(cov_s.astype("float64"))
    W = U @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ U.T  # Σ^{-1/2}

    res = {k: {"hit": {x: 0 for x in KS}, "mrr": 0.0, "auc": [], "margin": [],
                "n_above": [], "top1": 0, "spread": [], "range": []}
           for k in VARIANTS}
    denom = 0
    for pno, pid in enumerate(pids, 1):
        its = by_pid[pid]
        ch = ch_by_pid[pid]
        n = len(ch)
        v_raw = vec_by_pid[pid]
        # 各变体的块向量
        v_glob = norm_rows(v_raw - mu)
        mu_p = v_raw.mean(axis=0, keepdims=True)
        v_pp = norm_rows(v_raw - mu_p)
        if n >= 3:
            Xp = v_raw - mu_p.ravel()
            # 第一主成分 = 最大右奇异向量（1024 维方向）
            _, _sp, Vtp = np.linalg.svd(Xp, full_matrices=False)
            u1 = Vtp[0]
            v_abtt = norm_rows(Xp - np.outer(Xp @ u1, u1))
        else:
            v_abtt = v_pp
        v_wh = norm_rows((v_raw - mu) @ W)
        VV = {"raw": v_raw, "center_global": v_glob, "center_paper": v_pp,
              "abtt_paper": v_abtt, "whiten_global": v_wh}

        ntexts = [ree.norm(c.text) for c in ch]
        bm = BM25Index([c.text for c in ch])
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        for it in its:
            q = qas.get(it["qid"])
            if not q:
                continue
            _g, evs = gold_answer_full(q)
            if not evs:
                continue
            cg: set[int] = set()
            for ev in evs:
                gi, _ = ree.locate_gold(ntexts, ev)
                if not gi:
                    try:
                        gi = {int(np.argmax(np.asarray(bm.score(ree.norm(ev)),
                                                       dtype="float64")))}
                    except Exception:  # noqa: BLE001
                        gi = set()
                cg |= gi
            if not cg:
                continue
            denom += 1
            q_raw = encode_query(it["question"])
            QV = {"raw": q_raw,
                  "center_global": (q_raw - mu.ravel()) /
                                   (np.linalg.norm(q_raw - mu.ravel()) + 1e-9),
                  "center_paper": q_raw,
                  "abtt_paper": q_raw,
                  "whiten_global": ((q_raw - mu.ravel()) @ W) /
                                   (np.linalg.norm((q_raw - mu.ravel()) @ W) + 1e-9)}
            for name in VARIANTS:
                vs = (VV[name] @ QV[name]).astype("float64")
                order = np.argsort(-vs, kind="stable")
                first = first_gold_rank(order, cg)
                r = res[name]
                if first is not None:
                    for x in KS:
                        if first < x:
                            r["hit"][x] += 1
                    if first < 16:
                        r["mrr"] += 1.0 / (first + 1)
                    if first == 0:
                        r["top1"] += 1
                bg = max(float(vs[i]) for i in cg)
                lab = np.zeros(n, dtype=bool)
                for i in cg:
                    lab[i] = True
                r["auc"].append(auc1(vs, lab))
                oth = vs[~lab]
                if len(oth):
                    r["margin"].append(bg - float(oth.max()))
                    r["n_above"].append(float((oth > bg).mean()))
                r["spread"].append(float(vs.std()))
                r["range"].append(float(vs.max() - vs.min()))
        if pno % 40 == 0 or pno == len(pids):
            print(f"  [{pno}/{len(pids)}] q={denom}", flush=True)

    d = max(denom, 1)
    L = ["# 各向异性去除 A/B（同 250 题 / 同块向量，仅改打分后处理）", ""]
    L.append(f"> 样本 {denom} 题｜文档向量**未重新编码**（纯打分变换）｜AUC=每题各算再平均")
    L.append("")
    L.append("| 打分方式 | gold@1 | R@8 | R@12 | R@16 | MRR@16 | AUC | margin>0 | margin均 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name in VARIANTS:
        r = res[name]
        L.append(f"| {name} | {r['top1']/d:.3f} | {r['hit'][8]/d:.3f} | {r['hit'][12]/d:.3f} | "
                 f"{r['hit'][16]/d:.3f} | {r['mrr']/d:.3f} | "
                 f"{np.nanmean(r['auc']):.3f} | "
                 f"{np.mean([x > 0 for x in r['margin']]):.3f} | {np.nanmean(r['margin']):+.4f} |")
    L.append("")
    r0 = res["raw"]
    L.append("## 诊断：为什么 gold 不显著高")
    L.append(f"- 每题「块分数 std」均 **{np.mean(r0['spread']):.4f}**｜"
             f"「块分数极差」均 **{np.mean(r0['range']):.4f}**")
    L.append(f"- gold − 干扰最高分（margin）均 **{np.nanmean(r0['margin']):+.4f}**")
    L.append(f"- **gap / spread = {np.nanmean(r0['margin'])/np.mean(r0['spread']):+.2f}**"
             f"（≈0 说明 gold 的优势幅度与块间分数波动同量级 → 排序接近噪声）")
    L.append(f"- 干扰块分数高于 gold 的比例（均）：{np.mean(r0['n_above']):.1%}")
    txt = "\n".join(L)
    Path("qa/recall/ANISO_AB_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print("\n已写 qa/recall/ANISO_AB_20260910.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
