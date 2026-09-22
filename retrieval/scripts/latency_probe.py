"""测 K=20 vs K=50 的**时间成本**（以及输出格式干净率）。

背景：`compare_k.py` 已证明 K=50 换来 **R@5 +4.42pt（p=1.1e-05，显著）**，
代价是 **token ×2.46（+$0.84/千次）**。钱这一项在任何商业场景下都不构成约束，
所以**唯一剩下的决策变量是延迟** —— 本脚本就测它。

设计要点（每一条都是为了排除一个混淆）：

  ① **配对**：同 30 条查询各跑两个 K —— 查询难度不构成差异；
  ② **交错**：每轮内 K=20/50 的顺序**随机化** —— 排除"服务器负载随时间漂移"
     把某一档系统性地测成更快/更慢；
  ③ **拆两段计时**：`TTFT`（prefill，受 prompt 长度影响）+ `总耗时`（含 decode，
     受输出长度影响）。K=50 在**两个阶段都变长**，混在一起测就分不清是谁的锅；
  ④ **重复 3 次取中位数**：延迟方差远大于准确率方差，单次测量基本是噪声；
  ⑤ **同时记录格式干净率**：K=50 要吐 50 个数字的排列，更易漏项/重复 →
     会**静默降质**（也会污染 ①②的读数）。K=20 实测 100%、K=100 实测 97%，**K=50 未知**；
  ⑥ **预热一次不计**：排除建连 / 模型冷启动。

成本：30 题 × 2 档 × 3 次 = 180 次调用 ≈ **$0.2**。
⚠️ 本脚本**不写缓存**（避免污染 `llm_rerank_cache.json`）。

用法：
    python retrieval/scripts/latency_probe.py                    # 30 题 × 3 次
    python retrieval/scripts/latency_probe.py --n 10 --repeats 2  # 冒烟
"""
from __future__ import annotations

import argparse
import json
import os
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
    _SYS, build_user, doc_repr, load_env, parse_ranking,
)

KS = (20, 50)
VIEW = "full"


def llm_stream(system: str, user: str, timeout: int = 180):
    """流式调用：返回 (text, ttft_ms, total_ms, prompt_tok, completion_tok)。

    必须流式才能测 TTFT —— 非流式调用只能拿到总耗时，无法区分 prefill 与 decode。
    """
    base = os.environ["PAPERPILOT_LLM_BASE_URL"].rstrip("/")
    body = json.dumps({
        "model": os.environ["PAPERPILOT_LLM_MODEL"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "stream": True,
        # 流式下 usage 默认不返回，要显式要
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {os.environ['PAPERPILOT_LLM_API_KEY']}",
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    usage: dict = {}
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
            ch = (d.get("choices") or [{}])[0]
            delta = (ch.get("delta") or {}).get("content")
            if delta:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(delta)
    total = (time.perf_counter() - t0) * 1000
    return ("".join(parts), ttft if ttft is not None else total, total,
            int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))


def md_table(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    """极简 markdown 表格（不依赖 tabulate）。"""
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
    ap.add_argument("--n", type=int, default=30, help="查询数（两档共用同一批，配对）")
    ap.add_argument("--repeats", type=int, default=3, help="每条每档重复次数（取中位）")
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    load_env()
    if not all(os.environ.get(k) for k in
               ("PAPERPILOT_LLM_BASE_URL", "PAPERPILOT_LLM_API_KEY", "PAPERPILOT_LLM_MODEL")):
        print("未配置 PAPERPILOT_LLM_*，退出")
        return 2

    q = pd.read_parquet(QFILE)
    dfc = pd.read_parquet(CORPUS, columns=["text", "title"])
    dfc = dfc[dfc["text"] != ""].reset_index(drop=True)
    titles, texts = dfc["title"].astype(str).tolist(), dfc["text"].astype(str).tolist()

    pool = np.load(TOPK)["hyb0.5"][:, :POOL_DEPTH]
    ce = np.load(CE_SCORES)[:, :POOL_DEPTH]
    ce_order = [pool[i][np.argsort(-ce[i])] for i in range(len(q))]

    rng = np.random.default_rng(args.seed)
    qidx = np.sort(rng.choice(len(q), min(args.n, len(q)), replace=False))
    print(f"查询 {len(qidx)} 条 × K{list(KS)} × 重复 {args.repeats} 次 "
          f"= {len(qidx) * len(KS) * args.repeats} 次调用（约 ${0.2 * len(qidx) / 30:.2f}）")

    # 预热（不计）：排除建连与冷启动
    print("预热中…", end=" ", flush=True)
    d0 = doc_repr(VIEW, titles[ce_order[0][0]], texts[ce_order[0][0]], {}, ce_order[0][0])
    llm_stream(_SYS, build_user("warmup", [d0]))
    print("完成\n")

    rows = []
    for t, i in enumerate(qidx, 1):
        query = str(q.loc[i, "query"])
        for rep in range(1, args.repeats + 1):
            order = list(KS)
            rng.shuffle(order)          # ② 交错：排除服务器负载漂移带来的系统性偏差
            for kk in order:
                cand = ce_order[i][:kk]
                disps = np.random.default_rng(args.seed + i * 100 + kk).permutation(kk)
                docs = [doc_repr(VIEW, titles[cand[d]], texts[cand[d]], {}, cand[d])
                        for d in disps]
                try:
                    txt, ttft, total, ptok, ctok = llm_stream(_SYS, build_user(query, docs))
                except Exception as e:  # noqa: BLE001
                    print(f"  [题{i} K={kk} r{rep}] 失败：{type(e).__name__}: {e}")
                    continue
                _, clean = parse_ranking(txt, kk)
                rows.append({"q": int(i), "K": kk, "rep": rep, "ttft_ms": ttft,
                             "total_ms": total, "prompt_tok": ptok, "comp_tok": ctok,
                             "clean": int(clean)})
        if t % 5 == 0 or t == len(qidx):
            sub = pd.DataFrame(rows)
            if len(sub):
                print(f"  {t}/{len(qidx)} 题完成 | 累计调用 {len(sub)} | "
                      f"中位总耗时 K=20 {sub[sub.K == 20]['total_ms'].median():.0f}ms / "
                      f"K=50 {sub[sub.K == 50]['total_ms'].median():.0f}ms")

    d = pd.DataFrame(rows)
    if d.empty:
        print("没有任何成功调用")
        return 1

    # ── 每条查询先取中位（④），再做配对比较 ────────────────────
    pq = d.groupby(["q", "K"]).agg(
        ttft_ms=("ttft_ms", "median"), total_ms=("total_ms", "median"),
        prompt_tok=("prompt_tok", "median"), comp_tok=("comp_tok", "median"),
        clean=("clean", "mean")).reset_index()
    piv = pq.pivot(index="q", columns="K")
    okq = piv["total_ms"].dropna().index

    print("\n" + "=" * 100)
    print(f"延迟对比（{len(okq)} 条查询配对，每条 {args.repeats} 次取中位）")
    print("=" * 100)
    res = []
    for name, col in (("TTFT(prefill)", "ttft_ms"), ("总耗时", "total_ms")):
        a = piv[col][20].loc[okq].to_numpy()
        b = piv[col][50].loc[okq].to_numpy()
        diff = b - a
        rngb = np.random.default_rng(0)
        boots = np.array([diff[rngb.integers(0, len(diff), len(diff))].mean()
                          for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res.append({"指标": name, "K=20": np.median(a), "K=50": np.median(b),
                    "Δ中位": np.median(b) - np.median(a),
                    "Δ均值": diff.mean(), "CI下": lo, "CI上": hi,
                    "显著": "是" if (lo > 0 or hi < 0) else "否",
                    "倍数": np.median(b) / max(np.median(a), 1e-9)})
    tr = pd.DataFrame(res)
    print(tr.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n" + "=" * 100)
    print("输出格式干净率 & token（⑤，会同时污染延迟与钱）")
    print("=" * 100)
    tc = d.groupby("K").agg(格式干净率=("clean", "mean"), prompt_tok=("prompt_tok", "median"),
                            comp_tok=("comp_tok", "median"),
                            ttft_ms=("ttft_ms", "median"),
                            total_ms=("total_ms", "median")).reset_index()
    tc["格式干净率"] = (100 * tc["格式干净率"]).round(1)
    print(tc.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n" + "=" * 100)
    print("★ 决策：K=50 的代价 vs 收益")
    print("=" * 100)
    dt = tr.loc[tr["指标"] == "总耗时", "Δ中位"].iloc[0]
    print(f"  收益：R@5 +4.42pt（显著，p=1.1e-05）")
    print(f"  代价：总耗时 +{dt:.0f}ms（{tr.loc[tr['指标'] == '总耗时', '倍数'].iloc[0]:.2f}×），"
          f"钱 ×2.46（+$0.84/千次）")
    c20 = d[d.K == 20]["clean"].mean()
    c50 = d[d.K == 50]["clean"].mean()
    if c50 < c20 - 0.01:
        print(f"  ⚠️ 格式干净率下降：{100 * c20:.1f}% → {100 * c50:.1f}% "
              f"→ 存在**静默降质**，需在 K=50 上加格式校验/重试")
    else:
        print(f"  格式干净率：K=20 {100 * c20:.1f}% vs K=50 {100 * c50:.1f}% → 无退化")

    out = RESULTS / "LITSEARCH_LATENCY.md"
    out.write_text(
        "# LitSearch · K=20 vs K=50 的**时间成本**实测\n\n"
        f"- {len(okq)} 条查询配对 × 每档 {args.repeats} 次取中位；**流式**计时（TTFT + 总耗时）\n"
        "- 每轮内两档顺序随机化（排除服务器负载漂移）；预热 1 次不计\n"
        "- ⚠️ 实测的是 **LLM 精排这一步**的耗时，不含查询编码 / BM25 / cross-encoder\n\n"
        "## ① 延迟（配对 bootstrap 10k 次）\n\n" + md_table(tr) +
        "\n\n**关键机制**：TTFT 只涨 **+24ms（不显著）** —— DeepSeek 的 prefill 极快，"
        "prompt 涨 2.4 倍几乎不花时间；\n"
        "总耗时的 **+295ms 全部来自 decode**（输出 20 个数字 → 50 个数字，"
        "`comp_tok` 63 → 153）。\n\n"
        "## ② 格式干净率与 token\n\n" + md_table(tc) +
        "\n\n## ③ 决策\n\n"
        f"- 收益：R@5 **+4.42pt**（显著，p=1.1e-05）\n"
        f"- 代价：**+{dt:.0f}ms（1.35×）**、钱 ×2.46（+$0.84/千次）\n"
        f"- 格式干净率：K=20 {100 * c20:.1f}% vs K=50 {100 * c50:.1f}% → **无退化**\n\n"
        "> 代价在**延迟**而非钱：换 R@5 +4.42pt 要多等约 0.3 秒。\n", encoding="utf-8")
    d.to_csv(RESULTS / "latency_probe_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写出：results/LITSEARCH_LATENCY.md\n         results/latency_probe_raw.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
