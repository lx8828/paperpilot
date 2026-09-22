"""从 arXiv 新语料生成"研究者会问的"检索查询 —— 用来测**语料外查询的增益**。

为什么需要它：
  旧语料（LitSearch 63k）里**没有**新语料的 gold（重叠仅 13 篇），所以拿 LitSearch 的
  597 题只能测"扩语料有没有害"，测不出"有没有用"。要测收益，必须有一组
  **gold 落在新语料里**的查询。

做法（镜像 LitSearch 的 `inline` 集如何构造）：
  给 LLM 一篇论文的标题 + 摘要 → 让它写一条"研究者会拿去检索的需求"，
  **不许出现标题里的原词、不许点名论文** → 得到查询-论文对，gold 就是那篇。

成本：N 篇 × ~600 token ≈ $0.01 / 100 条。

用法：
    python retrieval/scripts/gen_arxiv_queries.py --n 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from llm_rerank import load_env  # noqa: E402

META = HERE / "data" / "arxiv" / "meta"
SRC = META / "corpus_text.parquet"
OUT = META / "arxiv_queries.parquet"

_SYS = (
    "你是学术检索基准的构造者。给定一篇论文的标题与摘要，"
    "你要写出一条**研究者会拿去文献检索的提问**。\n"
    "硬性要求：\n"
    "  · 只描述**研究需求**（想要什么方法/任务/该做什么），**不要**复述论文标题；\n"
    "  · **不要**出现论文的专有名词（自创的方法名、数据集名、作者名）；\n"
    "  · 是一条自然语言的问句或需求句，15~40 词，英文；\n"
    "  · 要具体到能区分出这一篇（例如带上任务、手段、约束），不要泛泛而谈。\n"
    '只输出 JSON：{"query": "..."}'
)

# 「更难」的版本：**刻意拉开与原文的表面距离** —— 用来检验上一版是不是"太容易"
_SYS_HARD = (
    "你是学术检索基准的构造者。给定一篇论文的标题与摘要，"
    "你要写出一条**刚进入这个方向的人会问的**检索需求。\n"
    "硬性要求：\n"
    "  · **禁止**出现论文里的具体方法名、数据集名、指标名、数字、专有缩写；\n"
    "  · **禁止**复用摘要里的原句或半句 —— 必须换成**上位概念**来表述；\n"
    "  · 描述「要解决什么问题、在什么场景、想要什么性质的方法」，20~35 词，英文；\n"
    "  · 要有方向性（能区别于整个 CS），但**不要**具体到一眼就能对上某一篇。\n"
    '只输出 JSON：{"query": "..."}'
)


def call(sys_p: str, usr: str, timeout: int = 120) -> str:
    body = json.dumps({
        "model": os.environ["PAPERPILOT_LLM_MODEL"],
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr}],
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(
        os.environ["PAPERPILOT_LLM_BASE_URL"].rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {os.environ['PAPERPILOT_LLM_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--style", default="normal", choices=["normal", "hard"],
                    help="hard = 刻意拉开与原文的表面距离，用于检验'是不是太容易'")
    ap.add_argument("--out", default="", help="输出文件名（默认按 style 命名）")
    args = ap.parse_args()
    sys_p = _SYS_HARD if args.style == "hard" else _SYS
    out_path = META / (args.out or
                       ("arxiv_queries_hard.parquet" if args.style == "hard"
                        else "arxiv_queries.parquet"))

    load_env()
    if not all(os.environ.get(k) for k in
               ("PAPERPILOT_LLM_BASE_URL", "PAPERPILOT_LLM_API_KEY", "PAPERPILOT_LLM_MODEL")):
        print("未配置 PAPERPILOT_LLM_*，退出")
        return 2

    df = pd.read_parquet(SRC)
    rng = np.random.default_rng(args.seed)
    # 只从"正文完整"的里抽，且限定 2026 年（这批新语料的主体）
    pool = df[df["n_chars"] > 800].reset_index(drop=True)
    idx = np.sort(rng.choice(len(pool), min(args.n, len(pool)), replace=False))
    print(f"从 {len(pool)} 篇中抽样 {len(idx)} 篇生成查询")

    def work(i: int):
        r = pool.loc[i]
        usr = f"标题：{r['title']}\n\n摘要：{r['abstract'][:2000]}"
        try:
            txt = call(sys_p, usr)
            m = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            q = str(m.get("query") or "").strip()
            return i, q or None
        except Exception as e:  # noqa: BLE001
            return i, f"__ERR__{type(e).__name__}"

    rows, bad = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, (i, q) in enumerate(ex.map(work, idx.tolist()), 1):
            if q and not q.startswith("__ERR__"):
                r = pool.loc[i]
                rows.append({"query": q, "gold_docid": int(r["docid"]),
                             "gold_arxiv_id": r["arxiv_id"], "gold_title": r["title"],
                             "gold_categories": r["categories"], "gold_created": r["created"]})
            else:
                bad += 1
            if k % 30 == 0:
                print(f"  {k}/{len(idx)}  ({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"\n已写出：{out_path}  （{len(out)} 条，失败 {bad}，style={args.style}）")
    print("样例：")
    for _, r in out.head(5).iterrows():
        print(f"  Q: {r['query'][:110]}")
        print(f"     → {r['gold_title'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
