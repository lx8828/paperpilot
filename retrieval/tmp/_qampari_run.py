"""**LoFT-RAG-QAMPARI 128k 档 · 检索 pipeline 首跑**

## 目标
在我们自己的硬件/模型上跑通"检索 → LLM reader → 多答案列表"，并与公开数字对标：
- 论文 Table 2（128k test）：**专用 RAG pipeline 0.55** / **Gemini 1.5 Pro 直读 0.44** / GPT-4o 0.27 / Claude 0.25
- ⚠️ 它们是**别的模型** → 我们的数字是"deepseek-flash + 我们的检索"的**新数据点**，公开数字是**参照线**

## 严格照抄官方的部分（口径一致）
- 语料格式：`ID: {pid} | TITLE: {title} | CONTENT: {passage}`（`prompts/constants/common.py`）
- 指令/收尾：`CORPUS_INSTRUCTION` + `FORMATTING_INSTRUCTION` + `QUERY_FORMAT_WITH_COT`（`prompts/constants/rag.py`）
- 指标（`evaluation/rag.py` + `evaluation/utils.py`）：
  - `em` = 集合相等；`coverage` = |pred∩gold|/|gold|；
  - **`subspan_em`** = 匹配矩阵(`gold in pred or pred in gold`)→ `linear_sum_assignment` → **全对齐才 1**
  - 先做 SQuAD 式 `normalize_answer`

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_run.py            # 检索 top-20
  ./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_run.py --k 40
  ./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_run.py --limit 10 # 冒烟
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import string
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
DATA = HERE / "data" / "loft" / "qampari" / "128k"
OUT = HERE / "results"

from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))

CORPUS_INSTRUCTION = ("You will be given a list of documents. You need to read carefully and "
                      "understand all of them. Then you will be given a query, and your goal is "
                      "to answer the query based on the documents you have read.")
FORMATTING = ("Your final answer should be in a list, in the following format:\n"
              "Final Answer: ['answer1', 'answer2', ...]\n"
              "If there is only one answer, it should be in the format:\n"
              "Final Answer: ['answer']")
QUERY_COT = ("====== Now let's start! ======\nWhich document is most relevant to the query and can "
             "answer the query? Think step-by-step and then format the answers into a list.\n"
             "query: {query}")


# ── 官方指标（照抄 evaluation/utils.py 与 evaluation/rag.py）──
def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))

    def rm_art(t):
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", t)

    def ws(t):
        return " ".join(t.split())

    def rm_punc(t):
        return "".join(c for c in t if c not in set(string.punctuation))

    return ws(rm_art(rm_punc(s.lower())))


def compute_coverage(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return float("nan")
    return len(set(pred) & set(gold)) / float(len(gold))


def compute_subspan_em(gold: list[str], pred: list[str]) -> float:
    if not gold or not pred:
        return 0.0
    sc = np.zeros([len(gold), len(pred)])
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if g in p or p in g:
                sc[i, j] = 1
    r, c = linear_sum_assignment(-sc)
    aligned = np.zeros(len(gold))
    for ri, ci in zip(r, c):
        aligned[ri] = sc[ri, ci]
    return float(all(aligned))


def compute_em_multi(gold: list[str], pred: list[str]) -> float:
    return float(set(gold) == set(pred))


def parse_answers(text: str) -> list[str]:
    """从模型输出里抽 'Final Answer: [...]'。"""
    m = re.findall(r"Final Answer\s*:\s*\[(.*?)\]", str(text), re.S | re.I)
    if not m:
        return []
    raw = m[-1]
    parts = re.split(r"['\"]\s*,\s*['\"]", raw)
    out = []
    for p in parts:
        s = p.strip().strip("'\"").strip()
        if s:
            out.append(s)
    return out


def make_reader(no_think: bool, model: str, max_tokens: int):
    """直连 API 的 reader：可显式关掉思考模式。

    ⚠️ 为什么不用 `llm.chat_text`：DeepSeek V4.1 的**思考模式默认可能自动开启**，
    长 prompt 下会把 token 预算全部花在 reasoning 上 → `content` 为空
    （实测 k=80 / 池深 200 两格整批 0 答案，就是这样挂的）。
    """
    import os
    import urllib.request

    base, key, _ = llm.config("PAPERPILOT_LLM")
    url = base.rstrip("/") + "/chat/completions"
    usage = {"prompt_tokens": 0, "completion_tokens": 0,
             "cache_hit_tokens": 0, "cache_miss_tokens": 0, "calls": 0, "empty": 0}

    def call(prompt: str) -> str:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "stream": False, "max_tokens": max_tokens}
        if no_think:
            body["thinking"] = {"type": "disabled"}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"},
                                     method="POST")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.loads(r.read().decode())
                u = d.get("usage") or {}
                pt = int(u.get("prompt_tokens") or 0)
                hit = int(u.get("prompt_cache_hit_tokens") or 0)
                usage["calls"] += 1
                usage["prompt_tokens"] += pt
                usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                usage["cache_hit_tokens"] += hit
                usage["cache_miss_tokens"] += max(pt - hit, 0)
                msg = d["choices"][0]["message"]
                c = (msg.get("content") or "").strip()
                if not c:
                    usage["empty"] += 1
                return c or str(msg.get("reasoning_content") or "")
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return ""
                time.sleep(2 * (attempt + 1))
        return ""

    return call, usage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    # ⚠️ 默认用**非思考**模型：deepseek-flash 默认思考，长 prompt 下会把预算吃光 → content 为空
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--tag", default="retr")
    ap.add_argument("--length", choices=["32k", "128k", "1m"], default="128k",
                    help="语料档位（LoFT：32k/128k/1m，对应语料 27k/105k/844k token）")
    ap.add_argument("--mode", choices=["retr", "direct"], default="retr",
                    help="retr = 检索 top-k；direct = 整档语料全塞（同模型对照，测「检索值不值」）")
    ap.add_argument("--exhaustive", action="store_true",
                    help="加一段「列出全部答案」的强指令（QAMPARI 是多答案题，模型容易给少）")
    ap.add_argument("--ground", action="store_true",
                    help="确定性「答案接地」过滤：丢掉在检索到的段落里找不到出处的预测答案（免费，治过预测）")
    ap.add_argument("--verify", action="store_true",
                    help="二轮自检：把首轮候选交给 LLM 严格复核（去掉不是答案的、补上漏的）")
    ap.add_argument("--pool", type=int, default=0,
                    help="检索池深（默认 = k）。配合 --rerank 时先取 pool 再重排到 k")
    ap.add_argument("--rerank", action="store_true",
                    help="用 bge-reranker-v2-m3 对池深做 cross-encoder 重排（RAG-1 里 +6.56pt 的那招）")
    ap.add_argument("--retriever", choices=["rrf", "dense", "bm25"], default="rrf",
                    help="检索器：rrf（dense+BM25 融合）/ dense / bm25")
    ap.add_argument("--no-think", action="store_true",
                    help="显式关闭思考模式（长 prompt 下思考会吃光预算 → content 为空）")
    ap.add_argument("--dump", action="store_true",
                    help="额外导出官方格式 preds.jsonl（全保真，供官方 runner 复核）")
    args = ap.parse_args()
    import os

    os.environ["PAPERPILOT_LLM_MODEL"] = args.model          # 让 --model 真正生效

    global QUERY_COT
    if args.exhaustive:
        QUERY_COT = ("====== Now let's start! ======\n"
                     "Read every document carefully. This question has MANY correct answers "
                     "(typically more than five). You must list **ALL** answers that appear "
                     "anywhere in the documents — do not stop after finding a few, and do not "
                     "omit an answer just because it looks less important.\n"
                     "Think step-by-step, go document by document, then format the answers into "
                     "a list.\nquery: {query}")

    data_dir = DATA.parent / args.length
    corpus = pd.DataFrame([json.loads(x) for x in (data_dir / "corpus.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()])
    queries = pd.DataFrame([json.loads(x) for x in (data_dir / "test_queries.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()])
    if args.limit:
        queries = queries.head(args.limit)
    corpus["full"] = corpus["title_text"].fillna("") + " \n " + corpus["passage_text"].fillna("")
    print(f"语料 {len(corpus)} 段 ｜ 问题 {len(queries)} 条 ｜ k={args.k} ｜ 模型 {args.model}")
    print(f"语料字符合计 {corpus['full'].str.len().sum():,} ≈ "
          f"{corpus['full'].str.len().sum() / 4 / 1000:.0f}k token")

    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    t0 = time.time()
    D = enc.encode(corpus["full"].tolist(), batch_size=32, normalize_embeddings=True,
                   show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    print(f"语料编码完成 {time.time() - t0:.1f}s")
    ce = None                                   # cross-encoder 惰性加载（--rerank 时才用）

    # ── 检索（dense + BM25 → RRF 融合，复用 RAG-1 的做法）──
    from collections import Counter
    import math

    def tok(s):
        return re.findall(r"[a-z0-9]+", str(s).lower())

    tf = [Counter(tok(t)) for t in corpus["full"]]
    ln = [max(1, sum(t.values())) for t in tf]
    avg = float(np.mean(ln))
    df = Counter()
    for t in tf:
        df.update(t.keys())
    N = len(tf)
    idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def bm25(q: str) -> np.ndarray:
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

    qv = enc.encode(queries["query_text"].tolist(), batch_size=16, normalize_embeddings=True,
                    show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    pid_of = corpus["pid"].tolist()

    recs = []
    reader, RUSE = make_reader(args.no_think, args.model, args.max_tokens)
    t_start = time.time()
    for qi, (_, q) in enumerate(queries.iterrows()):
        dense = D @ qv[qi]
        sparse = bm25(q["query_text"])
        if args.retriever == "dense":
            score = dense
        elif args.retriever == "bm25":
            score = sparse
        else:
            score = (1.0 / (60 + np.argsort(np.argsort(-dense)) + 1)
                     + 1.0 / (60 + np.argsort(np.argsort(-sparse)) + 1))
        pool = args.pool or args.k
        order = np.argsort(-score)[:pool]
        # ── cross-encoder 重排（bge-reranker-v2-m3；RAG-1 里 +6.56pt 的那招）──
        if args.rerank and len(order) > args.k:
            if ce is None:
                from sentence_transformers import CrossEncoder
                ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device=dev)
            pairs = [(q["query_text"], str(corpus["full"].iloc[i])) for i in order]
            ce_s = np.asarray(ce.predict(pairs, batch_size=16, show_progress_bar=False))
            order = order[np.argsort(-ce_s)][:args.k]
        else:
            order = order[:args.k]
        qrels = [str(x[0]) for x in q["metadata"]["qrels"]]
        gold_pass = set(qrels)
        hit = len(set(pid_of[i] for i in order) & gold_pass)
        if args.mode == "direct":
            order = np.arange(len(corpus))          # 全塞（直读对照）
        blocks = "\n".join(f"ID: {pid_of[i]} | TITLE: {corpus['title_text'].iloc[i]} | "
                           f"CONTENT: {corpus['full'].iloc[i]}" for i in order)
        # ⚠️ 两轮调用共用同一前缀（head）→ 保住前缀缓存（第二轮几乎不额外计费）
        head = f"{CORPUS_INSTRUCTION}\n\n{FORMATTING}\n\n{blocks}\n\n"
        prompt = head + QUERY_COT.format(query=q["query_text"])
        ans_text = reader(prompt)
        pred_raw = parse_answers(ans_text)
        # ── 二轮自检（LLM judge 环节）：严格复核首轮候选 ──
        if args.verify and pred_raw:
            vprompt = (head +
                       f"Query: {q['query_text']}\n\n"
                       f"A first pass produced this candidate list:\n"
                       f"{json.dumps(pred_raw, ensure_ascii=False)}\n\n"
                       "Now VERIFY each candidate against the documents above. Remove any candidate "
                       "that is NOT actually an answer to the query (wrong entity, belongs to a "
                       "different relation, or not supported by the documents), and ADD any correct "
                       "answer that the first pass missed. Be exhaustive but strict — every item in "
                       "your final list must be a genuine answer to the query.\n"
                       f"{FORMATTING}")
            vtext = reader(vprompt)
            vpred = parse_answers(vtext)
            if vpred:
                pred_raw = vpred
        gold_n = [normalize_answer(x) for x in q["answers"]]
        pred_n = [normalize_answer(x) for x in pred_raw]
        # ── 确定性「答案接地」过滤：只保留能在检索到的段落原文里找到出处的答案 ──
        ctx = normalize_answer(" \n ".join(str(corpus["full"].iloc[i]) for i in order))
        pred_g = [a for a in pred_n if a and a in ctx]
        cov = compute_coverage(gold_n, pred_n)
        sse = compute_subspan_em(gold_n, pred_n)
        tp = len(set(pred_n) & set(gold_n))
        prec = tp / len(pred_n) if pred_n else 0.0
        f1 = 2 * prec * cov / (prec + cov) if (prec + cov) else 0.0
        cov_g = compute_coverage(gold_n, pred_g)
        sse_g = compute_subspan_em(gold_n, pred_g)
        tpg = len(set(pred_g) & set(gold_n))
        prec_g = tpg / len(pred_g) if pred_g else 0.0
        f1g = 2 * prec_g * cov_g / (prec_g + cov_g) if (prec_g + cov_g) else 0.0
        recs.append(dict(qid=q["qid"], k=args.k, n_gold=len(gold_n),
                         recall_pass=hit / len(gold_pass) if gold_pass else float("nan"),
                         n_pred=len(pred_n), coverage=cov, subspan_em=sse,
                         em=compute_em_multi(gold_n, pred_n),
                         setP=prec, setR=cov, setF1=f1,
                         n_pred_g=len(pred_g), coverage_g=cov_g, subspan_em_g=sse_g,
                         em_g=compute_em_multi(gold_n, pred_g),
                         setP_g=prec_g, setR_g=cov_g, setF1_g=f1g,
                         pred_raw=json.dumps(pred_raw, ensure_ascii=False)[:600],
                         _full_pred=json.dumps(pred_raw, ensure_ascii=False),
                         pred_grounded=json.dumps(pred_g, ensure_ascii=False)[:600],
                         ans_tail=str(ans_text)[-260:].replace("\n", " ")))
        if (qi + 1) % 10 == 0:
            print(f"  {qi + 1}/{len(queries)} … cov={np.mean([r['coverage'] for r in recs]):.3f} "
                  f"sse={np.mean([r['subspan_em'] for r in recs]):.3f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    df = pd.DataFrame(recs)
    OUT.mkdir(exist_ok=True)
    # 全保真导出官方格式（供官方 runner 复核；避免 CSV 截断导致回读失真）
    if args.dump:
        d = OUT / f"official_{args.tag}"
        d.mkdir(exist_ok=True)
        qs = {json.loads(l)["qid"]: json.loads(l)
              for l in (data_dir / "test_queries.jsonl").read_text(
                  encoding="utf-8").splitlines() if l.strip()}
        with open(d / "preds.jsonl", "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                ans = json.loads(str(r["_full_pred"]))
                f.write(json.dumps({"qid": r["qid"], "model_outputs": [ans],
                                    "num_turns": 1}, ensure_ascii=False) + "\n")
        with open(d / "queries.jsonl", "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps(qs[r["qid"]], ensure_ascii=False) + "\n")
        print(f"  已导出官方格式（全保真）→ {d}")
    dst = OUT / f"R2_QAMPARI_{args.tag}_k{args.k}.csv"
    df.to_csv(dst, index=False, encoding="utf-8-sig")
    u = RUSE
    cost = (u["cache_miss_tokens"] * 1.07 + u["cache_hit_tokens"] * 0.021
            + u["completion_tokens"] * 4.26) / 1e6
    print(f"  ⚠️ 空 content 次数：{u['empty']}（思考模式吃光预算的征兆；`--no-think` 可避免）")
    print(f"\n{'=' * 100}\n【结果】{len(df)} 题 ｜ k={args.k} ｜ 模型 {args.model}")
    print(f"  **subspan_em {df['subspan_em'].mean():.4f}**（LoFT 主指标）")
    print(f"  coverage    {df['coverage'].mean():.4f}  ｜ em {df['em'].mean():.4f}"
          f"  ｜ set-F1 {df['setF1'].mean():.4f} ｜ set-P {df['setP'].mean():.4f}")
    print(f"  段落级 recall@{args.k} {df['recall_pass'].mean():.4f}"
          f"  ｜ 平均预测答案数 {df['n_pred'].mean():.2f}（gold {df['n_gold'].mean():.2f}）")
    print(f"\n  【确定性「答案接地」过滤后】（丢掉段落里找不到出处的答案，免费）")
    print(f"  **subspan_em {df['subspan_em_g'].mean():.4f}**（Δ {df['subspan_em_g'].mean() - df['subspan_em'].mean():+.4f}）"
          f"  ｜ coverage {df['coverage_g'].mean():.4f}（Δ {df['coverage_g'].mean() - df['coverage'].mean():+.4f}）"
          f"  ｜ set-P {df['setP_g'].mean():.4f}（Δ {df['setP_g'].mean() - df['setP'].mean():+.4f}）"
          f"  ｜ set-F1 {df['setF1_g'].mean():.4f}")
    print(f"  平均保留答案数 {df['n_pred_g'].mean():.2f}（过滤前 {df['n_pred'].mean():.2f}）")
    print(f"  用时 {time.time() - t_start:.0f}s ｜ 调用 {u['calls']} 次 ｜ "
          f"prompt {u['prompt_tokens']:,}（命中 {u['cache_hit_tokens']:,}）"
          f" / completion {u['completion_tokens']:,} ｜ **成本 ¥{cost:.3f}**（均 ¥{cost / len(df):.4f}/题）")
    print(f"\n  参照（论文 Table 2，128k test，别的模型）：专用 RAG 0.55 ｜ Gemini 1.5 Pro 直读 0.44 "
          f"｜ GPT-4o 0.27 ｜ Claude 0.25")
    print(f"  → 已写 {dst}")
    print(f"\n  样例（前 2 题）：")
    for _, r in df.head(2).iterrows():
        print(f"   {r['qid']} ｜ cov={r['coverage']:.2f} sse={r['subspan_em']:.0f} "
              f"｜ pred={r['pred_raw'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
