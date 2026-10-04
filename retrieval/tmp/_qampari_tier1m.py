"""**1M 档**：语料 5,878 段 ≈ 844k token —— 判决"检索到底值不值"

## 为什么必须跑这一档（见 `QAMPARI_AUDIT_REPORT_20260929.md §2.8`）
`128k` 档语料（755 段 ≈ 105k token）**可以整档塞进上下文** →
实测 `direct128k_k20`（整档直读）与 `exh_k40_k40`（检索 top-40）**subspan_em 都是 0.700** →
**在 128k 档，检索没有证明自己的价值**。

`1M` 档：844k token **装不进常规上下文** → 检索必须创造价值。这是判决场。

## 做法（与 128k 档严格同口径）
- **完全复用** `_qampari_run.py` 的官方 prompt 常量（`CORPUS_INSTRUCTION` / `FORMATTING` /
  exhaustive `QUERY_COT`）与官方指标（em / coverage / subspan_em）
- 检索：bge-m3 dense + BM25 → RRF → top-k
- reader：`deepseek-chat` no-think（与之前一致）
- **并行**（8 线程）、写出官方格式 `preds.jsonl` → 用**官方 CLI** 出指标

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_tier1m.py --limit 3        # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_tier1m.py --k 40 --workers 8
"""
from __future__ import annotations

import importlib.util as _iu
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "tmp"))

spec = _iu.spec_from_file_location("qr", HERE / "tmp" / "_qampari_run.py")
qr = _iu.module_from_spec(spec)
sys.modules["qr"] = qr
spec.loader.exec_module(qr)

OUT = HERE / "results"
OFFICIAL = HERE / "data" / "loft" / "official" / "run_evaluation.py"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--length", default="1m", choices=["128k", "1m"])
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--pool", type=int, default=0, help="0 → =k")
    ap.add_argument("--tag", default="tier1m_k40")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--direct-chars", type=int, default=0,
                    help=">0 时额外跑一个「直读前 N 字符」臂（用于测上下文上限）")
    ap.add_argument("--rerank-from", default="",
                    help="复用 `_qampari_rerank.py` 导出的重排序列（{qid:[pids]}），免重复跑重排")
    args = ap.parse_args()

    DATA = HERE / "data" / "loft" / "qampari" / args.length
    corpus = [json.loads(l) for l in (DATA / "corpus.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    queries = [json.loads(l) for l in (DATA / "test_queries.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        queries = queries[: args.limit]
    full = [str(c.get("title_text") or "") + " \n " + str(c.get("passage_text") or "")
            for c in corpus]
    pid = [str(c.get("pid")) for c in corpus]
    chars = sum(len(x) for x in full)
    print(f"档位 {args.length} ｜ 语料 {len(corpus):,} 段 / {chars:,} 字符 "
          f"≈ {chars / 4 / 1000:.0f}k token ｜ 问题 {len(queries)} ｜ k={args.k}\n")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    t0 = time.time()
    D = enc.encode(full, batch_size=64, normalize_embeddings=True,
                   show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    Q = enc.encode([q["query_text"] for q in queries], batch_size=16,
                   normalize_embeddings=True, show_progress_bar=False,
                   convert_to_numpy=True).astype(np.float32)
    print(f"编码完成 {time.time() - t0:.0f}s")

    # BM25（与 `_qampari_run.py` 同实现）
    import math
    import re
    from collections import Counter

    def tok(s):
        return re.findall(r"[a-z0-9]+", str(s).lower())

    tf = [Counter(tok(t)) for t in full]
    ln = [max(1, sum(t.values())) for t in tf]
    avg = float(np.mean(ln))
    dfc = Counter()
    for t in tf:
        dfc.update(t.keys())
    N = len(tf)
    idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in dfc.items()}

    def bm25s(q):
        out = np.zeros(N)
        for t in set(tok(q)):
            i_ = idf.get(t)
            if i_ is None:
                continue
            for i, tt in enumerate(tf):
                f = tt.get(t, 0)
                if f:
                    out[i] += i_ * f * 2.2 / (f + 1.2 * (1 - 0.75 + 0.75 * ln[i] / avg))
        return out

    reader, RUSE = qr.make_reader(True, args.model, 2500)
    pool = args.pool or args.k
    direct_head = ""
    if args.direct_chars:
        direct_head = "\n".join(
            f"ID: {pid[i]} | TITLE: {corpus[i].get('title_text')} | CONTENT: {full[i]}"
            for i in range(len(corpus)))[: args.direct_chars]

    lock = threading.Lock()
    prog = {"n": 0}
    # 重排序列（若给定）：{qid: [pids]}；pid → 扁平下标
    rr_map: dict[str, list[str]] = {}
    if args.rerank_from:
        rr_map = json.loads(Path(args.rerank_from).read_text(encoding="utf-8"))
        print(f"使用重排序列 {args.rerank_from}（{len(rr_map)} 题）\n")
    pid2i = {p: i for i, p in enumerate(pid)}

    def one(i: int) -> dict:
        q = queries[i]
        dense = D @ Q[i]
        sparse = bm25s(q["query_text"])
        # RRF（同 `_qampari_run.py`）
        score = (1.0 / (60 + np.argsort(np.argsort(-dense)) + 1)
                 + 1.0 / (60 + np.argsort(np.argsort(-sparse)) + 1))
        if rr_map and q["qid"] in rr_map:
            order = np.array([pid2i[p] for p in rr_map[q["qid"]][: args.k] if p in pid2i])
        else:
            order = np.argsort(-score)[:pool][: args.k]
        blocks = "\n".join(f"ID: {pid[j]} | TITLE: {corpus[j].get('title_text')} | "
                           f"CONTENT: {full[j]}" for j in order)
        head = f"{qr.CORPUS_INSTRUCTION}\n\n{qr.FORMATTING}\n\n{blocks}\n\n"
        prompt = head + qr.QUERY_COT.format(query=q["query_text"])
        pred = qr.parse_answers(reader(prompt))
        out = dict(qid=q["qid"], n_pred=len(pred), prompt_chars=len(prompt),
                   n_ctx_passages=len(order), pred=pred, mode="retrieval")
        if direct_head:
            p2 = (f"{qr.CORPUS_INSTRUCTION}\n\n{qr.FORMATTING}\n\n{direct_head}\n\n"
                  + qr.QUERY_COT.format(query=q["query_text"]))
            out["direct_pred"] = qr.parse_answers(reader(p2))
            out["direct_prompt_chars"] = len(p2)
        with lock:
            prog["n"] += 1
            if prog["n"] % 20 == 0:
                print(f"  {prog['n']}/{len(queries)} …", flush=True)
        return out

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(one, range(len(queries))))
    print(f"\n调用完成 {time.time() - t1:.0f}s")
    df = pd.DataFrame(recs)
    maxpc = int(df["prompt_chars"].max())
    print(f"检索臂 prompt 字符：中位 {int(df['prompt_chars'].median()):,} ｜ 最大 {maxpc:,}"
          f" ≈ {maxpc / 4 / 1000:.0f}k token")

    # ── 官方格式导出 + 官方 CLI 出指标 ──
    tag = args.tag
    d = OUT / f"official_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    qs = {q["qid"]: q for q in queries}
    with (d / "preds.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps({"qid": r["qid"], "model_outputs": [list(r["pred"])],
                                 "num_turns": 1}, ensure_ascii=True) + "\n")
    with (d / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps(qs[r["qid"]], ensure_ascii=True) + "\n")

    import subprocess
    cp = subprocess.run([PY, str(OFFICIAL), "--answer_file_path", str(d / "queries.jsonl"),
                         "--pred_file_path", str(d / "preds.jsonl"),
                         "--task_type", "multi_value_rag"],
                        capture_output=True, text=True, encoding="utf-8",
                        cwd=str(OFFICIAL.parent))
    if not (d / "preds_metrics.json").exists():
        print("官方 CLI 失败：", cp.stderr[-400:])
        return 1
    m = json.loads((d / "preds_metrics.json").read_text(encoding="utf-8"))
    print(f"\n{'=' * 110}\n【官方指标（{args.length} 档，k={args.k}）】")
    print(json.dumps(m, indent=2, ensure_ascii=False))
    df.drop(columns=["pred"]).to_csv(OUT / f"R2_QAMPARI_{tag}.csv", index=False,
                                     encoding="utf-8-sig")
    u = RUSE
    cost = (u["cache_miss_tokens"] * 1.07 + u["cache_hit_tokens"] * 0.021
            + u["completion_tokens"] * 4.26) / 1e6
    print(f"\n平均预测答案数 {df['n_pred'].mean():.2f} ｜ 调用 {u['calls']} ｜ "
          f"prompt {u['prompt_tokens']:,}（命中 {u['cache_hit_tokens']:,}）"
          f" / completion {u['completion_tokens']:,} ｜ 成本 ¥{cost:.3f}")
    print(f"产物：{d} ／ {OUT / f'R2_QAMPARI_{tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
