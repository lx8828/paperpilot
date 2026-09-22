"""「只输出前 N 名」能否把 K=50 的延迟压回 K=20 水平，同时保住 R@5？

由来（`latency_probe.py` 的实测结论）：K=50 的总耗时比 K=20 多 **295ms**，
而拆开看：
    prompt token  ×2.38  →  TTFT 只 +24ms（不显著）   ← prefill 几乎免费
    completion tok ×2.43 →  总耗时 +295ms（**全部增量**）← decode 是瓶颈

→ 所以瓶颈是「LLM 要多输出 30 个数字」，而交付只需要 **top-5**。
→ 修法：**让 LLM 只吐前 N 名**，别吐 50 篇的全排列。**把「池子深度 K」与「输出长度」解耦。**

三个臂（都在**同一个池子**上，即 hyb0.5 top-100 → CE top-K）：
    k20       K=20 + 全排列（现用基线）
    k50       K=50 + 全排列
    k50p10    K=50 + **只输出前 10 名**   ← 本实验的主角

两个用法（分开跑，别混）：

  ① 延迟对比（三臂**交错**，同会话内，排除服务器负载漂移）：
     python partial_out_probe.py --arms k20,k50,k50p10 --n 30 --repeats 3

  ② 效果对比（只看 k50p10 的 R@5 是否保住；全量 597 才有统计效力）：
     python partial_out_probe.py --arms k50p10 --n 0 --repeats 1

⚠️ 不写 `llm_rerank_cache.json`（避免污染正式缓存）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from llm_rerank import (  # noqa: E402
    CE_SCORES, CORPUS, DERIVED, POOL_DEPTH, QFILE, RESULTS, TOPK,
    _SYS, build_user, doc_repr, load_env,
)
from significance import mcnemar  # noqa: E402

VIEW = "full"

_SYS_PARTIAL = (
    "你是学术文献检索的排序专家。研究者给出一个「文献检索需求」，"
    "下面是若干候选论文的标题与摘要。请挑出**最符合该需求的前 {n} 篇**，"
    "按「该论文对满足这个检索需求的有用程度」**从高到低**给出。\n"
    "判断依据：论文研究的**问题/方法/任务**是否与需求匹配。\n"
    "注意：\n"
    "  · 不要因为出现相同关键词就判为相关 —— 要看它是否真的解决需求描述的那个问题；\n"
    "  · 若需求包含多个条件（如「既用 A 又用 B」），优先选**同时满足更多条件**的论文；\n"
    "  · 你看到的候选顺序是随机的，与相关性无关；\n"
    "  · 候选有 {k} 篇，但**你只需要给出前 {n} 名**。\n"
    '只输出 JSON：{{"ranking": [编号, 编号, ...]}}，'
    "必须恰好 {n} 个编号、每个只出现一次。"
)


def build_user_partial(query: str, docs: list[tuple[str, str]], n: int) -> str:
    """与 `build_user` 同构，只把结尾的"共 K 个"换成"共 n 个"。"""
    lines = [f"检索需求：{query}", "", "候选论文："]
    for j, (title, body) in enumerate(docs, 1):
        lines.append(f"{j}. 标题：{title}" + (f"\n   摘要：{body}" if body else ""))
    lines += ["", '请输出 JSON：{"ranking": [...]}（编号从高到低，'
              f"**只给前 {n} 名**）"]
    return "\n".join(lines)


def parse_partial(text: str, k: int, n: int) -> tuple[list[int], bool]:
    """解析"前 n 名"。返回 (编号列表(1-based，已在候选内)，是否干净)。

    干净 = 恰好 n 个、全部落在 1..k、且互不重复。
    """
    m = re.search(r"\[[\s\d,]*\]", text)
    if m:
        try:
            nums = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
    else:
        nums = [int(x) for x in re.findall(r"\d+", text)]
    out, seen = [], set()
    for x in nums:
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= x <= k and x not in seen:
            seen.add(x)
            out.append(x)
    clean = (len(out) == n) and (len(nums) == n)
    return out[:n], clean


def llm_stream(system: str, user: str, timeout: int = 180):
    """流式调用：返回 (text, ttft_ms, total_ms, prompt_tok, completion_tok)。"""
    base = os.environ["PAPERPILOT_LLM_BASE_URL"].rstrip("/")
    body = json.dumps({
        "model": os.environ["PAPERPILOT_LLM_MODEL"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {os.environ['PAPERPILOT_LLM_API_KEY']}",
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, parts, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
            if d.get("usage"):
                usage = d["usage"]
            delta = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(delta)
    total = (time.perf_counter() - t0) * 1000
    return ("".join(parts), ttft if ttft is not None else total, total,
            int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))


def md_table(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    cols = list(df.columns)
    fmt = lambda v: (f"{v:{floatfmt}}" if isinstance(v, (int, float, np.floating))
                     else str(v))  # noqa: E731
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="k20,k50,k50p10",
                    help="逗号分隔：k20 / k50 / k50p10")
    ap.add_argument("--partial-n", type=int, default=10, help="k50p10 臂输出的名次数")
    ap.add_argument("--n", type=int, default=30, help="查询数；0=全部 597")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out-md", default="LITSEARCH_PARTIAL_LAT.md")
    ap.add_argument("--start", type=int, default=0,
                    help="从 qidx 的第几个开始（断点续跑；配合 --resume-from 补齐）")
    ap.add_argument("--resume-from", default="",
                    help="逗号分隔的已有 raw csv（results/ 下），先并入再一起分析")
    ap.add_argument("--only-q", default="",
                    help="只跑这些查询下标（逗号分隔）。**补单题请用这个，别用 --start** —— "
                         "当 --n > 0 时 qidx 是 rng.choice 的随机样本，其下标与题号不一致")
    args = ap.parse_args()

    load_env()
    if not all(os.environ.get(k) for k in
               ("PAPERPILOT_LLM_BASE_URL", "PAPERPILOT_LLM_API_KEY", "PAPERPILOT_LLM_MODEL")):
        print("未配置 PAPERPILOT_LLM_*，退出")
        return 2

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ("k20", "k50", "k50p10")]
    if bad:
        print(f"未知臂 {bad}；可选 k20 / k50 / k50p10")
        return 2

    q = pd.read_parquet(QFILE)
    truth = q["specificity"].to_numpy()
    ids = np.load(DERIVED / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[g] for g in row if int(g) in id2row} for row in q["corpusids"]]

    dfc = pd.read_parquet(CORPUS, columns=["text", "title"])
    dfc = dfc[dfc["text"] != ""].reset_index(drop=True)
    titles, texts = dfc["title"].astype(str).tolist(), dfc["text"].astype(str).tolist()

    pool = np.load(TOPK)["hyb0.5"][:, :POOL_DEPTH]
    ce = np.load(CE_SCORES)[:, :POOL_DEPTH]
    ce_order = [pool[i][np.argsort(-ce[i])] for i in range(len(q))]

    def K_of(arm: str) -> int:
        return 20 if arm == "k20" else 50

    rng = np.random.default_rng(args.seed)
    qidx = (np.arange(len(q)) if args.n <= 0
            else np.sort(rng.choice(len(q), min(args.n, len(q)), replace=False)))
    if args.only_q:                                # 精确补单题
        want = [int(x) for x in args.only_q.split(",") if x.strip()]
        qidx = np.array([i for i in want if 0 <= i < len(q)], dtype=int)
    n_all = len(qidx)
    qidx = qidx[args.start:]                       # 断点续跑：跳过已跑过的
    rows: list[dict] = []
    for p in filter(None, (s.strip() for s in args.resume_from.split(","))):
        prev = pd.read_csv(RESULTS / p, encoding="utf-8-sig")
        rows += prev.to_dict("records")
        print(f"  并入 {p}：{len(prev)} 行（{prev['q'].nunique()} 题）")
    ncall = len(qidx) * len(arms) * args.repeats
    print(f"本次处理 qidx[{args.start}:{n_all}] = {len(qidx)} 题 × 臂 {arms} "
          f"× 重复 {args.repeats} = {ncall} 次调用（约 ${0.006 * ncall / 10:.2f}）")

    print("预热中…", end=" ", flush=True)
    d0 = doc_repr(VIEW, titles[ce_order[0][0]], texts[ce_order[0][0]], {}, ce_order[0][0])
    llm_stream(_SYS, build_user("warmup", [d0]))
    print("完成\n")

    for t, i in enumerate(qidx, 1):
        query = str(q.loc[i, "query"])
        for rep in range(1, args.repeats + 1):
            # 交错：排除服务器负载漂移。用**每题独立**的 rng → 断点续跑时顺序也确定
            order = list(arms)
            per = np.random.default_rng(args.seed + i * 1000 + rep).permutation(len(order))
            order = [order[j] for j in per]
            for arm in order:
                kk, pn = K_of(arm), (args.partial_n if arm == "k50p10" else 0)
                cand = ce_order[i][:kk]
                disps = np.random.default_rng(args.seed + i * 100 + kk).permutation(kk)
                docs = [doc_repr(VIEW, titles[cand[d]], texts[cand[d]], {}, cand[d])
                        for d in disps]
                if arm == "k50p10":
                    system = _SYS_PARTIAL.format(n=pn, k=kk)
                    user = build_user_partial(query, docs, pn)
                else:
                    system, user = _SYS, build_user(query, docs)
                try:
                    txt, ttft, total, ptok, ctok = llm_stream(system, user)
                except Exception as e:  # noqa: BLE001
                    print(f"  [题{i} {arm} r{rep}] 失败：{type(e).__name__}: {e}")
                    continue
                if arm == "k50p10":
                    nums, clean = parse_partial(txt, kk, pn)
                else:
                    from llm_rerank import parse_ranking
                    nums, clean = parse_ranking(txt, kk)
                ranked = [int(cand[disps[x - 1]]) for x in nums]
                rows.append({"q": int(i), "arm": arm, "rep": rep, "ttft_ms": ttft,
                             "total_ms": total, "prompt_tok": ptok, "comp_tok": ctok,
                             "clean": int(clean), "out_len": len(ranked),
                             "R@1": 1.0 if (golds[i] & set(ranked[:1])) else 0.0,
                             "R@5": (len(golds[i] & set(ranked[:5])) / len(golds[i])
                                     if golds[i] else 0.0),
                             "R@10": (len(golds[i] & set(ranked[:10])) / len(golds[i])
                                      if golds[i] else 0.0)})
        if t % 25 == 0 or t == len(qidx):
            print(f"  {t}/{len(qidx)} 题 | 累计 {len(rows)} 次调用")

    d = pd.DataFrame(rows)
    if d.empty:
        print("没有成功调用")
        return 1
    # 去重（续跑时可能重复并入），保留最后一条
    before = len(d)
    d = d.drop_duplicates(subset=["q", "arm", "rep"], keep="last").reset_index(drop=True)
    if before != len(d):
        print(f"  去重：{before} → {len(d)} 行")

    # ── 指标（按臂聚合；每条取中位仅用于延迟）────────────────────
    print("\n" + "=" * 100)
    print("① 效果：R@1 / R@5 / R@10（recall 口径，与论文一致）")
    print("=" * 100)
    eff = d.groupby("arm").agg(n=("q", "nunique"), R1=("R@1", "mean"),
                               R5=("R@5", "mean"), R10=("R@10", "mean"),
                               out_len=("out_len", "mean"),
                               clean=("clean", "mean")).reset_index()
    eff["clean"] = (100 * eff["clean"]).round(1)
    print(eff.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 100)
    print("② 延迟：TTFT / 总耗时（每条先取中位，再跨题统计）")
    print("=" * 100)
    lat = d.groupby(["arm", "q"]).agg(ttft_ms=("ttft_ms", "median"),
                                      total_ms=("total_ms", "median")).reset_index()
    lt = lat.groupby("arm").agg(TTFT中位=("ttft_ms", "median"),
                                总耗时中位=("total_ms", "median")).reset_index()
    print(lt.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    tk = d.groupby("arm").agg(prompt_tok=("prompt_tok", "median"),
                              comp_tok=("comp_tok", "median")).reset_index()
    print("\n" + tk.to_string(index=False))

    # ── ③ 配对显著性：k50p10 vs k20 / vs k50 ──────────────────
    def paired(x_arm: str, y_arm: str, metric: str):
        """y_arm − x_arm 的配对差（同题逐次平均后配对）。"""
        px = d[d.arm == x_arm].groupby("q")[metric].mean()
        py = d[d.arm == y_arm].groupby("q")[metric].mean()
        common = px.index.intersection(py.index)
        return px.loc[common].to_numpy(), py.loc[common].to_numpy()

    print("\n" + "=" * 100)
    print("③ 配对显著性（同题；hit@5 用 McNemar，连续量用配对 bootstrap）")
    print("=" * 100)
    sig = []
    for xa, ya in (("k20", "k50p10"), ("k50", "k50p10")):
        if xa not in set(d.arm) or ya not in set(d.arm):
            continue
        for metric in ("R@1", "R@5", "R@10"):
            if metric == "R@10" and ya == "k50p10" and args.partial_n < 10:
                continue
            va, vb = paired(xa, ya, metric)
            diff = vb - va
            rb = np.random.default_rng(0)
            boots = np.array([diff[rb.integers(0, len(diff), len(diff))].mean()
                              for _ in range(10000)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ha = (va > 0).astype(int)
            hb = (vb > 0).astype(int)
            o1, o0 = int(((hb == 1) & (ha == 0)).sum()), int(((hb == 0) & (ha == 1)).sum())
            pv = mcnemar(o1, o0)
            # ⚠️ 列名必须固定 —— 用 xa/ya 当列名会让两组对比的列错开、填满 nan
            sig.append({"对比": f"{ya} − {xa}", "指标": metric,
                        "左": va.mean(), "右": vb.mean(), "Δ": diff.mean(),
                        "Δpt": 100 * diff.mean(), "CI下": lo, "CI上": hi,
                        "显著": "是" if (lo > 0 or hi < 0) else "否",
                        "翻转(仅右/仅左)": f"{o1}/{o0}", "p": pv})
            print(f"  [{metric}] {ya} − {xa}: Δ={100 * diff.mean():+.2f}pt  "
                  f"CI=[{100 * lo:+.2f}, {100 * hi:+.2f}]  "
                  f"{'显著' if lo > 0 or hi < 0 else '不显著'}  p={pv:.3g}")
    ts = pd.DataFrame(sig)

    # ── ④ 决策 ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("④ 结论")
    print("=" * 100)
    lm = dict(zip(lt["arm"], lt["总耗时中位"]))
    if "k20" in lm and "k50p10" in lm:
        print(f"  总耗时：k20 {lm['k20']:.0f}ms | k50p10 {lm['k50p10']:.0f}ms "
              f"→ {'✅ 压回 K=20 水平' if lm['k50p10'] < lm['k20'] * 1.1 else '❌ 未压回'}")
    if "k50" in lm and "k50p10" in lm:
        print(f"  相对 k50：{lm['k50p10'] - lm['k50']:+.0f}ms")
    c = dict(zip(eff["arm"], eff["clean"]))
    if "k50p10" in c:
        print(f"  k50p10 输出格式干净率：{c['k50p10']:.1f}%"
              + ("（⚠️ 有退化，需加校验/重试）" if c["k50p10"] < 99 else ""))

    out = RESULTS / args.out_md
    out.write_text(
        f"# 「只输出前 {args.partial_n} 名」能否同时拿到 K=50 的召回与 K=20 的延迟\n\n"
        f"- 臂：{arms}；{len(qidx)} 条查询 × 重复 {args.repeats}\n"
        f"- 三臂共用同一池子（hyb0.5 top-100 → CE top-K），**只改喂给 LLM 的候选数与输出长度**\n\n"
        "## ① 效果\n\n" + md_table(eff, ".4f") +
        "\n\n## ② 延迟\n\n" + md_table(lt) + "\n\n" + md_table(tk) +
        "\n\n## ③ 配对显著性\n\n" + (md_table(ts, ".4g") if len(ts) else "（单臂，无对比）") +
        "\n", encoding="utf-8")
    d.to_csv(RESULTS / (Path(args.out_md).stem + "_raw.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已写出：results/{args.out_md}\n         results/{Path(args.out_md).stem}_raw.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
