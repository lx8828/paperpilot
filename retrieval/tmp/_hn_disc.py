"""打分判别力 + **hard negative** 诊断（零 LLM，5 组沙盒，走生产打分函数）。

回答四个问题：
  ① 打分到底有多差？（AUC：0.5 = 无判别力）
  ② 差在哪一层？把"非 gold 块"按难度拆成两类 **hard negative**：
       · **篇内 HN**：与某 gold 块**同篇**的非 gold 块（同主题/同写作风格 → 最难分）
       · **跨篇 HN**：其他篇的块（姊妹论文 → 也难分）
  ③ **第二路能不能补**？中文提问 vs 英文语料的**词法路**（BM25）在 zh / en 两侧各有多少信号、
     判别力多少 —— `en` 侧就是用**英文检索式**做词法匹配（现成资产，零增量成本）。
  ④ **融合端还有多少空间**？dense × BM25(en) 的**最优线性融合**（oracle 权重）AUC 上界。

用法：uv run python retrieval/tmp/_hn_disc.py --groups group1,group2,group3,group4,group5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval" / "tmp"))

import _sandbox as sbx  # noqa: E402

_GOLD = sbx._gold_of


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    """tie-aware ROC AUC（与 `_discrim.py` 同实现）。"""
    npos, n = int(pos.sum()), len(scores)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype="float64")
    ranks[order] = np.arange(1, n + 1, dtype="float64")
    s = scores[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def mm(s: np.ndarray) -> np.ndarray:
    """每题内 min-max 归一到 [0,1]（融合前的可比化；全同分 → 全 0）。"""
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(s)


def subgroup(scores: np.ndarray, pos: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把打分限制在 `pos ∪ other` 上（= 换一批对手重算 AUC）。"""
    keep = pos | other
    return scores[keep], pos[keep]


def med(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]          # 去 nan
    return float(np.median(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="group1,group2,group3,group4,group5")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    per: list[dict] = []
    for g in groups:
        sb = sbx.Sandbox(g)
        for q in sb.queries:
            qid = q["qid"]
            gold = set(_GOLD(sb, qid)["qchunks"])
            if not gold:
                continue
            n = sb.N
            pos = np.asarray([i in gold for i in range(n)], dtype=bool)
            gpap = {sb.pdf_of[i] for i in gold}
            same = np.asarray([(not pos[i]) and sb.pdf_of[i] in gpap for i in range(n)], dtype=bool)
            cross = np.asarray([(not pos[i]) and sb.pdf_of[i] not in gpap for i in range(n)], dtype=bool)

            v_zh = sb.vec_scores(qid, "zh").astype("float64")
            b_zh = sb.bm_global(qid, "zh").astype("float64")
            has_en = bool(q.get("translation"))
            v_en = sb.vec_scores(qid, "en").astype("float64") if has_en else None
            b_en = sb.bm_global(qid, "en").astype("float64") if has_en else None

            r: dict = {"qid": qid, "m1": "-M1-" in qid, "ngold": len(gold),
                       "n_same_hn": int(same.sum()), "n_cross_hn": int(cross.sum()),
                       "bm_zh_live": float(b_zh.max()) > 0,
                       "bm_en_live": bool(has_en and b_en is not None and float(b_en.max()) > 0)}
            # ① 全部对手
            r["a_v_zh"] = auc(v_zh, pos)
            r["a_b_zh"] = auc(b_zh, pos) if r["bm_zh_live"] else float("nan")
            r["a_b_en"] = auc(b_en, pos) if r["bm_en_live"] else float("nan")
            r["a_v_en"] = auc(v_en, pos) if v_en is not None else float("nan")
            # ② 分对手难度
            s, p = subgroup(v_zh, pos, same)
            r["a_v_same"] = auc(s, p)
            s, p = subgroup(v_zh, pos, cross)
            r["a_v_cross"] = auc(s, p)
            s, p = subgroup(b_en, pos, same) if r["bm_en_live"] else (None, None)
            r["a_b_en_same"] = auc(s, p) if s is not None else float("nan")
            # ③ margin：gold 最高分 − 最强对手（全语料 / 篇内）
            r["m_all"] = float(v_zh[pos].max() - v_zh[~pos].max())
            r["m_same"] = float(v_zh[pos].max() - v_zh[same].max()) if same.any() else float("nan")
            # ④ 融合上界：dense(zh) × BM25(en) 最优线性权重（oracle，逐题）
            if r["bm_en_live"]:
                best = max(auc(w * mm(v_zh) + (1 - w) * mm(b_en), pos)
                           for w in np.arange(0, 1.001, 0.1))
                r["a_fuse"] = best
            else:
                r["a_fuse"] = float("nan")
            per.append(r)

    def block(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        ac = lambda k: [x[k] for x in rows if x[k] == x[k]]  # noqa: E731
        print(f"  {name:<8}{len(rows):>4} ｜ AUC dense(zh) {np.mean(ac('a_v_zh')):>5.3f}"
              f" ｜ vs篇内HN {np.mean(ac('a_v_same')):>5.3f} ｜ vs跨篇HN {np.mean(ac('a_v_cross')):>5.3f}"
              f" ｜ BM25(en) {np.mean(ac('a_b_en')):>5.3f}({len(ac('a_b_en'))}题)"
              f" ｜ **融合上界** {np.mean(ac('a_fuse')):>5.3f}")
        print(f"  {'':<8}{'':>4} ｜ margin 全局中位 {med(ac('m_all')):>+.3f}"
              f" ｜ margin 篇内中位 {med(ac('m_same')):>+.3f}"
              f" ｜ BM25(zh) 有信号 {sum(1 for x in rows if x['bm_zh_live'])}/{len(rows)}"
              f" ｜ BM25(en) 有信号 {sum(1 for x in rows if x['bm_en_live'])}/{len(rows)}")

    print(f"\n{'=' * 110}\n########## 打分判别力 + hard negative（{len(groups)} 组，{len(per)} 题）")
    print(f"  {'层':<8}{'题数':>4} ｜ 指标（AUC 0.5=无判别力 / 1.0=完美）")
    block("全部", per)
    block("M1", [x for x in per if x["m1"]])
    block("非M1", [x for x in per if not x["m1"]])

    print(f"\n########## 判别力最差 12 道（dense AUC 升序；看 hard negative 有多少）")
    print(f"  {'qid':<11}{'层':<5}{'AUC d':>7}{'vs篇内':>7}{'vs跨篇':>7}{'BM25en':>8}"
          f"{'融合':>7}{'margin':>8}{'gold':>5}{'篇内HN':>7}{'跨篇HN':>7}")
    for x in sorted(per, key=lambda t: t["a_v_zh"])[:12]:
        f = lambda v: ("—" if v != v else f"{v:.3f}")  # noqa: E731
        print(f"  {x['qid']:<11}{'M1' if x['m1'] else '—':<5}{x['a_v_zh']:>7.3f}"
              f"{f(x['a_v_same']):>7}{f(x['a_v_cross']):>7}{f(x['a_b_en']):>8}"
              f"{f(x['a_fuse']):>7}{x['m_all']:>+8.3f}{x['ngold']:>5}"
              f"{x['n_same_hn']:>7}{x['n_cross_hn']:>7}")
    print("\n读法：`AUC dense(zh)` 低 → 一路打分本身不行；`vs篇内HN` 更低 → 同篇内分不开（hard negative 主战场）；"
          "\n      `融合上界` 比 dense 高多少 → **融合端还有多少空间**（这是 oracle，真实值会低一些）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
