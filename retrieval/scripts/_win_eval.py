"""窗口视图验收（**等预算口径**，见 RAG_COMPONENT_NOTES 铁律 6）。

对照三档：chunk（现状） / win1000 / win800。

指标（**不需要 gold**）：
  · 语料均匀度：各篇 unit 数、中位长 → 篇间是否对等
  · 平均篇数     ：top12 来自几篇（跨篇可见性）
  · top1覆盖     ：各篇**自己**的 top1 有几篇进了最终 top12
  · 指向性       ：目标命中均 / 首名正确（相关性不能被"公平"换掉）
  · **等预算覆盖**：按 rank 序累计字符达 B 时，覆盖"各篇单独 top12 并集(≤60)"多少条
                   ← 公平口径（固定 k 会奖励大块）
"""
import os
import sys
from collections import Counter
from statistics import median

sys.stdout.reconfigure(encoding="utf-8")

from paperpilot.workflow import _ensure_env

_ensure_env()

from paperpilot.agents.embedder import ChunkIndex, MultiChunkIndex  # noqa: E402

PDFS = ["2408.09273.pdf", "2305.14205.pdf", "2403.13240.pdf",
        "2112.08804.pdf", "2305.09220.pdf"]
MODES = [("chunk", 0), ("win1000", 1000), ("win800", 800)]
BUDGETS = (8000, 16000, 24000)
TOPK = 12

TARGETED = [
    ("2408.09273.pdf", "对比学习 contrastive learning 做跨语言摘要"),
    ("2305.14205.pdf", "content plan 内容规划作为跨语言桥接是怎么用的"),
    ("2403.13240.pdf", "可微分流水线 differentiable pipeline 少样本 few-shot"),
    ("2112.08804.pdf", "CrossSum 1500 多种语言对的数据集怎么构建的"),
    ("2305.09220.pdf", "统一多语言与跨语言摘要 unify multilingual cross-lingual"),
]
GENERIC = [
    "如何用对比学习做跨语言摘要生成",
    "训练数据是怎么构造的",
    "使用了哪些数据集和评价指标",
    "模型的架构是怎样的",
    "实验结果如何",
]


def _s(p: str) -> str:
    return p.replace(".pdf", "")


def _key(h: dict, pdf: str | None = None) -> tuple[str, str]:
    return (str(pdf or h.get("pdf") or ""), str(h["chunk_id"]))


def set_mode(target: int) -> None:
    os.environ["PAPERPILOT_RETRIEVAL_WINDOWS"] = "1" if target else "0"
    if target:
        os.environ["PAPERPILOT_WINDOW_TARGET"] = str(target)


def main() -> int:
    rows = []
    for name, target in MODES:
        set_mode(target)
        m = MultiChunkIndex(PDFS)
        units = m._doc_chunks()
        per = {p: ChunkIndex(p) for p in PDFS}
        m.search_hybrid("预热", top_k=1)          # 触发编码（窗口首次会 encode）

        n_by_p = {p: len(per[p]._doc_chunks()) for p in PDFS}
        lens = [len(u.text) for u in units]
        agg = {"np": [], "cov": 0, "tgt": 0, "first": 0, "k": 0,
               "bud": {b: 0.0 for b in BUDGETS}}
        detail = []

        for label, items in (("指向性", TARGETED), ("泛化", [(None, q) for q in GENERIC])):
            for tgt_pdf, q in items:
                own = {p: per[p].search_hybrid(q, top_k=TOPK) for p in PDFS}
                tset = {_key(h, p) for p, hs in own.items() for h in hs}
                full = m.search_hybrid(q, top_k=len(units))     # 全序
                top = full[:TOPK]
                dist = Counter(_s(h["pdf"]) for h in top)
                got = {_key(h) for h in top}
                cov = sum(1 for p in PDFS if _key(own[p][0], p) in got)

                agg["np"].append(len(dist))
                agg["cov"] += cov
                agg["k"] += 1
                tgt = dist.get(_s(tgt_pdf), 0) if tgt_pdf else None
                first = (top[0]["pdf"] == tgt_pdf) if tgt_pdf else None
                if tgt_pdf:
                    agg["tgt"] += tgt
                    agg["first"] += bool(first)

                bud = {}
                for B in BUDGETS:
                    chars, acc = 0, set()
                    for h in full:
                        acc.add(_key(h))
                        chars += len(h["text"])
                        if chars >= B:
                            break
                    bud[B] = len(acc & tset)
                    agg["bud"][B] += len(acc & tset)
                detail.append((label, q, tgt_pdf, dist, cov, tgt, first, bud, len(tset)))

        med = int(median(lens))
        row = {
            "name": name, "units": len(units), "med": med,
            "n_by_p": n_by_p,
            "np": sum(agg["np"]) / len(agg["np"]),
            "cov": agg["cov"] / agg["k"],
            "tgt": agg["tgt"] / 5, "first": agg["first"],
            "bud": {b: agg["bud"][b] / len(detail) for b in BUDGETS},
        }
        rows.append(row)

        print(f"\n{'=' * 112}")
        print(f"【{name}】unit {len(units)} ｜ 中位长 {med} ｜ 各篇 unit 数 "
              f"{ {_s(p): v for p, v in n_by_p.items()} }")
        for label, q, tgt_pdf, dist, cov, tgt, first, bud, n_t in detail:
            tag = f"[目标 {_s(tgt_pdf)}]" if tgt_pdf else "[泛化]"
            ts = "-" if tgt is None else str(tgt)
            fs = "-" if first is None else ("✓" if first else "✗")
            print(f"  {label[:3]} {q[:34]:<34} {tag:<16} 目标{ts:>3} {fs} "
                  f"top1覆盖{cov}/5 篇数{len(dist)} 等预算"
                  f"{bud[8000]}/{bud[16000]}/{bud[24000]}of{n_t} {dict(dist)}")

    print(f"\n{'=' * 112}\n【汇总】等预算口径")
    print(f"  {'mode':<9}{'unit':>6}{'中位长':>8}{'平均篇数':>9}{'top1覆盖':>10}"
          f"{'目标命中':>9}{'首名':>6}{'R@8k':>8}{'R@16k':>8}{'R@24k':>8}")
    for r in rows:
        b = r["bud"]
        print(f"  {r['name']:<9}{r['units']:>6}{r['med']:>8}{r['np']:>9.1f}"
              f"{r['cov']:>8.1f}/5{r['tgt']:>9.1f}{r['first']:>4}/5"
              f"{b[8000]:>8.1f}{b[16000]:>8.1f}{b[24000]:>8.1f}")
    print("\n  [读法] R@Bk = 累计 B 字符时覆盖『各篇单独 top12 并集』的条数（上限 60/题）。")
    print("         等预算下比大小才公平；同时看『目标命中/首名』不能掉。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
