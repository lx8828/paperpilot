"""**污染检查 v2**：我们调用的模型（deepseek-chat）预训练是否见过 QAMPARI？

用户要求重点：**半查询法**（看是不是在背答案）。

四路探针（**全部不给任何文档**）：
  P1 无上下文直答    query → answers（**同 `--exhaustive` 提示**，便于与 exh_k40 可比）
                     ← 若接近"带检索"分数 → 模型本来就会答
  P2 半题面补全      query 前半 → 要求逐字补全后半 ← 数据集**文本**是否被背下来
  P3 半答案补全      query + gold 一半 → 列出**其余**（留出集）+ **盲猜基线对照** ← 最关键
  P4 配对判别        query + 真答案集 vs **同域**干扰答案集 ← 是否知道"题↔答案集"绑定

v2 相对 v1 的三处修正（v1 冒烟暴露的设计缺陷）：
  ① P3 加**盲猜基线**：模型给 k 个答案不算本事，要比"从全体 gold 池随机抽 k 个"高多少
  ② P4 干扰项改**同域**（选题面 token 重合最高的另一题）→ 排除"靠领域匹配蒙对"
  ③ P1 改用 **exhaustive 提示**，与 `exh_k40_k40` 口径一致
"""
from __future__ import annotations

import importlib.util as _iu
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"

spec = _iu.spec_from_file_location("qr", HERE / "tmp" / "_qampari_run.py")
qr = _iu.module_from_spec(spec)
sys.modules["qr"] = qr
spec.loader.exec_module(qr)

DATA = HERE / "data" / "loft" / "qampari" / "128k"
queries = [json.loads(l) for l in (DATA / "test_queries.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]

ap = __import__("argparse").ArgumentParser()
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--limit2", type=int, default=60)
ap.add_argument("--model", default="deepseek-chat")
ap.add_argument("--fresh", action="store_true", help="忽略缓存重跑")
args = ap.parse_args()

reader, RUSE = qr.make_reader(True, args.model, 2000)
print(f"污染检查 v2 ｜ 模型 {args.model}（no-think）｜ P1 {args.limit} 题 / P2-P4 {args.limit2} 题\n")

FMT = ("Your final answer should be in a list, in the following format:\n"
       "Final Answer: ['answer1', 'answer2', ...]")
# 与 `_qampari_run.py --exhaustive` 同款措辞（只把 "documents" 换成 "your own knowledge"）
EXH_TAIL = ("====== Now let's start! ======\n"
            "This question has MANY correct answers (typically more than five). You must list "
            "**ALL** answers you know — do not stop after finding a few, and do not omit an "
            "answer just because it looks less important.\n"
            "Think step-by-step, then format the answers into a list.\nquery: {query}")

ALL_GOLD = [qr.normalize_answer(a) for q in queries for a in q["answers"]]
rnd0 = random.Random(0)


def score(gold_raw, pred_raw):
    g = [qr.normalize_answer(x) for x in gold_raw]
    p = [qr.normalize_answer(x) for x in pred_raw]
    gs, ps = set(g), set(p)
    tp = len(gs & ps)
    return dict(n_pred=len(ps), coverage=tp / len(gs) if gs else 0.0,
                subspan_em=qr.compute_subspan_em(g, p), em=qr.compute_em_multi(g, p),
                setP=tp / len(ps) if ps else 0.0)


# ── P1 ─────────────────────────────────────────────────────────────────────
print("=" * 108)
print("【P1】无上下文直答（exhaustive 提示、零文档）")
r1, t0 = [], time.time()
for i, q in enumerate(queries[: args.limit], 1):
    p = ("You will be given a query. You have **no documents** — answer from your own knowledge "
         "only, as best you can.\n\n" + FMT + "\n\n" + EXH_TAIL.format(query=q["query_text"]))
    r1.append(dict(qid=q["qid"], n_gold=len(q["answers"]), **score(q["answers"], qr.parse_answers(reader(p)))))
    if i % 25 == 0:
        print(f"  {i}/{args.limit} … subspan={np.mean([x['subspan_em'] for x in r1]):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
d1 = pd.DataFrame(r1)
print(f"\n  {'指标':<14}{'P1 无上下文':>12}{'带检索 exh_k40':>16}")
for nm, v, ref in (("coverage", d1["coverage"].mean(), 0.8127),
                   ("subspan_em", d1["subspan_em"].mean(), 0.700),
                   ("em", d1["em"].mean(), 0.440), ("precision", d1["setP"].mean(), 0.8304)):
    print(f"  {nm:<14}{v:>12.4f}{ref:>16.4f}")
print(f"  平均预测数 {d1['n_pred'].mean():.2f}（gold {d1['n_gold'].mean():.2f}）")

# ── P2 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 108)
print("【P2】半题面补全（给前半，逐字补全后半）→ 数据集**文本**是否被背下来")
r2 = []
for q in queries[: args.limit2]:
    qt = str(q["query_text"]); cut = len(qt) // 2; pre = qt[:cut]
    comp = reader("下面是某个公开问答数据集里一条问题的**前半部分**。请**逐字补全**这条问题的"
                  "后半部分（不要解释、不要作答，只输出补全的内容）：\n\n" + pre
                  ).strip().strip('"').strip("'")
    full = (pre + " " + comp)
    a = set(re.findall(r"[a-z0-9]+", full.lower()))
    b = set(re.findall(r"[a-z0-9]+", qt.lower()))
    r2.append(dict(qid=q["qid"], token_f1=2 * len(a & b) / (len(a) + len(b)) if (a or b) else 0.0,
                   tail_hit=float(qt[cut:].strip().lower()[:30] in full.lower()), comp=comp[:160]))
d2 = pd.DataFrame(r2)
print(f"  token-F1 {d2['token_f1'].mean():.3f} ｜ 含原题后半（30 字符）的比例 "
      f"**{d2['tail_hit'].mean():.1%}** → "
      f"{'⚠️ 高' if d2['tail_hit'].mean() > 0.3 else '✅ 低（未复现数据集文本）'}")
for _, r in d2.head(3).iterrows():
    print(f"    {r['qid'][:22]:<24} 补全：{r['comp'][:120]}")

# ── P3（+ 盲猜基线）─────────────────────────────────────────────────────────
print("\n" + "=" * 108)
print("【P3】半答案补全：给 gold 一半，列出**其余**（留出集）｜**对照盲猜基线**")
r3 = []
for q in queries[: args.limit2]:
    gold = list(q["answers"]); k = len(gold)
    half, rest = gold[:(k + 1) // 2], gold[(k + 1) // 2:]
    pr = qr.parse_answers(reader(
        f"Question: {q['query_text']}\n\nSome **known correct answers** to this question are: "
        f"{json.dumps(half, ensure_ascii=False)}\n\nList ALL of the **remaining** correct answers "
        f"(do not repeat the ones given). {FMT}"))
    gs = {qr.normalize_answer(x) for x in rest}
    ps = {qr.normalize_answer(x) for x in pr}
    tp = len(gs & ps)
    # 盲猜基线：从全体 gold 池随机抽 len(ps) 个（排本题）
    own = {qr.normalize_answer(x) for x in gold}
    pool = [x for x in ALL_GOLD if x not in own]
    n = len(ps)
    base = np.mean([len(gs & set(rnd0.sample(pool, min(n, len(pool))))) / len(gs)
                    for _ in range(20)]) if (pool and gs and n) else 0.0
    r3.append(dict(qid=q["qid"], n_held=len(gs), n_pred=n, held_recall=tp / len(gs) if gs else 0.0,
                   held_prec=tp / len(ps) if ps else 0.0, blind_recall=float(base),
                   pred=json.dumps(pr, ensure_ascii=False)[:260]))
d3 = pd.DataFrame(r3)
print(f"  **留出集召回 {d3['held_recall'].mean():.3f}** ｜ 留出集精度 {d3['held_prec'].mean():.3f}"
      f" ｜ 模型给 {d3['n_pred'].mean():.1f} 个 vs 留出 {d3['n_held'].mean():.1f} 个")
print(f"  **盲猜基线（同数量随机抽）留出集召回 {d3['blind_recall'].mean():.3f}**")
gain = d3["held_recall"].mean() - d3["blind_recall"].mean()
print(f"  → 相对盲猜 **{gain:+.3f}** → "
      f"{'⚠️ 显著领先盲猜：模型确实知道部分答案' if gain > 0.1 else '✅ 与盲猜无异（撒网，非背答案）'}")
print(f"  （精度 {d3['held_prec'].mean():.3f} 说明绝大多数输出是噪声）")

# ── P4（同域 vs 随机 双干扰对照）────────────────────────────────────────────
print("\n" + "=" * 108)
print("【P4】配对判别：真答案集 vs 干扰｜**同域干扰 vs 随机干扰 对照**")
print("  ⚠️ 先声明构念效度：「选出与问题相符的答案集」**可由语义拟合解决**，")
print("     因此**即使准确率高也不能证明记住了数据集**。故加随机干扰臂做对照：")
print("     若两臂都 ≈100% → 测得的是「语义拟合能力」，不是「题↔答案集绑定」。")
STOP = {"which", "what", "who", "whom", "whose", "when", "where", "why", "how", "did", "does",
        "do", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "and", "or", "with", "by", "from", "that", "this", "it", "its", "their", "his", "her",
        "man", "woman", "people", "person", "work", "works", "name", "called", "known", "after",
        "before", "during", "many", "much", "more", "most", "other", "same", "both", "all", "any"}
rnd = random.Random(1)


def content(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if t not in STOP and len(t) > 2}


qtok = {q["qid"]: content(q["query_text"]) for q in queries}


def distractor(q, mode: str):
    own = {qr.normalize_answer(x) for x in q["answers"]}
    cands = [o for o in queries if o["qid"] != q["qid"]
             and not ({qr.normalize_answer(x) for x in o["answers"]} & own)]
    if not cands:
        return None, 0
    if mode == "random":
        o = cands[rnd.randrange(len(cands))]
        return o, len(qtok[q["qid"]] & qtok[o["qid"]])
    best, sc = None, -1
    for o in cands:
        s = len(qtok[q["qid"]] & qtok[o["qid"]])
        if s > sc:
            best, sc = o, s
    return best, sc


r4 = []
for q in queries[: args.limit2]:
    row = dict(qid=q["qid"])
    for mode in ("domain", "random"):
        o, sc = distractor(q, mode)
        if o is None:
            row[f"{mode}_ok"] = np.nan
            continue
        swap = rnd.random() < 0.5
        x, y = ((list(o["answers"]), list(q["answers"])) if swap
                else (list(q["answers"]), list(o["answers"])))
        a = str(reader(f"Question: {q['query_text']}\n\nTwo candidate answer sets:\n"
                       f"1) {json.dumps(x, ensure_ascii=False)}\n2) {json.dumps(y, ensure_ascii=False)}\n\n"
                       "Which one is the correct answer set for the question? 只输出数字 1 或 2。"))
        m = re.search(r"[12]", a)
        pick = int(m.group()) if m else 0
        row[f"{mode}_ok"] = float(bool((pick == 2) if swap else (pick == 1)))
        row[f"{mode}_shared"] = sc
    r4.append(row)
d4 = pd.DataFrame(r4)
print(f"  {'干扰类型':<12}{'准确率':>9}{'题干内容词共享':>14}{'n':>5}")
for mode, nm in (("domain", "同域干扰"), ("random", "随机干扰")):
    print(f"  {nm:<12}{d4[f'{mode}_ok'].mean():>9.3f}{d4[f'{mode}_shared'].mean():>14.2f}"
          f"{int(d4[f'{mode}_ok'].notna().sum()):>5}")
gap4 = d4["domain_ok"].mean() - d4["random_ok"].mean()
print(f"  → 同域 − 随机 = {gap4:+.3f}")
print(f"  → 判读：{'⚠️ 两臂都高 → **该任务测得的是语义拟合，不能证明数据集绑定**' if min(d4['domain_ok'].mean(), d4['random_ok'].mean()) > 0.8 else '两臂有差异，可部分归因于领域贴合度'}")

# ── 汇总 ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 108)
print("【汇总】污染信号")
print(f"  P1 无上下文 coverage {d1['coverage'].mean():.3f} / subspan_em {d1['subspan_em'].mean():.3f}"
      f"（带检索 0.813 / 0.700）｜预测数 {d1['n_pred'].mean():.1f} vs gold 5.0")
print(f"  P2 半题面逐字补全命中 {d2['tail_hit'].mean():.1%}")
print(f"  P3 留出集召回 {d3['held_recall'].mean():.3f} vs 盲猜基线 {d3['blind_recall'].mean():.3f}"
      f"（净增益 {gain:+.3f}）｜精度 {d3['held_prec'].mean():.3f}")
print(f"  P4 配对判别 同域 {d4['domain_ok'].mean():.3f} / 随机干扰 {d4['random_ok'].mean():.3f}"
      f"（随机 0.5）→ 两臂差 {gap4:+.3f}")
u = RUSE
cost = (u["cache_miss_tokens"] * 1.07 + u["cache_hit_tokens"] * 0.021
        + u["completion_tokens"] * 4.26) / 1e6
print(f"  调用 {u['calls']} 次 ｜ 成本 ¥{cost:.3f}")
for nm, d in (("P1", d1), ("P2", d2), ("P3", d3), ("P4", d4)):
    d.to_csv(OUT / f"R2_CONTAM_{nm}.csv", index=False, encoding="utf-8")
print("已写 R2_CONTAM_P1..P4.csv")
