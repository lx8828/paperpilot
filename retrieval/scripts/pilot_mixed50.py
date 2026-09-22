"""混合语料试点：**从语料里随机抽论文 → 按论文出题 → 看 gold 排第几**。

为什么这样测（纠正前一版的设计缺陷）：
  旧版分别在「gold 全在旧语料」和「gold 全在新语料」两个**单侧**查询集上测，
  Δ 都是构造性的（加干扰项必然掉、加答案源必然从 0 涨），两者不可相加，
  **"净效应"从未被测量**。这里改成：

    统一索引空间（旧 0..N_OLD-1，新 N_OLD..N_OLD+N_NEW-1）
      → 从**全集**随机抽 n 篇（均匀主样本，对语料构成无偏）
      → 每篇用 LLM 生成 1 条"研究员会问的检索问题"（prompt 见 _SYS_GEN）
      → 跑完整流水线，记录该篇在**每一级**的位次

  这样查询天然按真实语料比例混合，且 gold 是**真实论文**（不是人造 qrels）。

流水线（与最终配置一致）：
    hyb0.5 本语料 top-100  →  两语料候选合并(200)  →  CE  →  top-50  →  LLM 只输出前 10

三层分析（都从原始记录现算，**零额外成本**）：
  ① **抽样口径**：均匀主样本（总体指标只用它）+ `--old-quota` 补抽旧语料
     （只进分侧统计）。因为旧语料只占语料 10.5%，不补抽的话分侧 CI 宽到 ±17pt。
  ② **分层切面**：语料侧 / **抽象层级**（含不含技术锚点）。两者都**只读查询文本
     或抽样来源**，不看检索结果 → **非循环**。（"难/易"不能这么分：没有独立定义，
     只能从结果反推，属于循环论证。）
  ③ **失败归因**：`召回失败`（gold 没进 200 池）vs `排序失败`（进池但没进 top-5）
     —— 两者修法完全不同。
  ④ **泄漏审计**：上一版"新语料查询"实测 R@1 = 1.0，逐条看是**标题改写**。这里
     量化它（标题稀有词被查询复现的比例）。注意该指标是**词面**的，中文查询天然低。

用法：
    # ① 建题（均匀 200 + 补抽旧语料 60）
    python retrieval/scripts/pilot_mixed50.py --n 200 --old-quota 60 \
        --out-md LITSEARCH_MIXED200.md
    # ② 复用同一批题，比池子深度（精确配对；语料不变时才可复用）
    python retrieval/scripts/pilot_mixed50.py --reuse-queries --pool 300 \
        --out-md LITSEARCH_POOL300.md
    # 冒烟（**务必换 queries-file**，否则会覆盖正式攒下的查询集）
    python retrieval/scripts/pilot_mixed50.py --n 3 --pool 300 \
        --queries-file _smoke_q.parquet --out-md _smoke.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from _hfcache import ensure_hf_home  # noqa: E402

# ⚠️ 必须在 transformers / sentence_transformers 导入**之前**
ensure_hf_home()

from eval_retrieval import RESULTS, SparseBM25, rrf_fuse  # noqa: E402
from llm_rerank import LLM, load_env  # noqa: E402

OLD = HERE / "data" / "litsearch" / "derived"
NEW = HERE / "data" / "arxiv"
QF = RESULTS / "pilot_mixed50_queries.parquet"

ALPHA = 0.5                  # 臂内 hybrid 的向量权重
DEPTH = 1000                 # RRF 组件列表深度
POOL_BASE = 100              # 对照臂：每语料进 CE 的候选数（= 现用参数，合并 200）
K_CE = 50                    # CE 后取前 K 给 LLM
N_OUT = 10                   # LLM 只输出前 N 名（交付 top-5 只需要这些）

# ⚠️ 池子深度对比为什么能"一次跑两臂"：
#    `bge-reranker-v2-m3` 是 **pointwise** cross-encoder —— 每对 (query, doc) 独立打分，
#    候选之间**不交互**。所以在宽池（如 300+300）上打的分，**限制到窄池子集**
#    （如 100+100）后重排序，结果与"只在窄池上打分"**逐位相同**。
#    → 一次宽池运行 = 两个池子深度的**精确配对**结果，不必跑两遍。
POOLS: list[int] = []        # 运行时由 --pools 决定（含 POOL_BASE，升序）

DISPLAY = 0                  # LLM 候选的呈现顺序打乱用（0 = 按 CE 序，便于逐级对照）

# ── 出题 prompt：对用户给的原文只做三处工程化处理 ────────────────
#   ① "2024 年" → **该论文的真实年份**（否则是给 LLM 一个假前提）
#   ② **钉死语言** —— prompt 是中文，LLM 会不自觉地中英混出（冒烟实测 4 中 1 英），
#      语言一混，"泄漏审计"和跨题比较全都失真
#   ③ 末尾追加 JSON 输出约束（解析需要）
_SYS_GEN = (
    "请根据这篇 {year}论文的贡献，生成一个研究员在寻找这篇论文时会问的搜索问题。"
    "要求包含具体的任务、方法或评估指标，避免直接使用标题词汇。{lang_hint}\n"
    '只输出 JSON：{{"query": "..."}}'
)

# ⚠️ 语料是英文论文，但**用户提问是中文** —— 这两个选择测的是不同能力：
#    zh → 跨语言检索（贴近真实用户，但与 LitSearch 的英文口径不可直接比）
#    en → 单语言检索（与语料同语言，可与 LitSearch 论文口径对比）
LANG_HINT = {
    "zh": "请用**中文**提出这个问题（不要混入英文，方法名等专有名词可保留原文）。",
    "en": "Please write the question in **English**.",
}

# LLM 精排（只输出前 N）—— 与 `partial_out_probe.py` 的 k50p10 臂同构
_SYS_RANK = (
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

_STOP = set("""a an the of for and or to in on with without using use used via by from as at is are be
this that these those we our it its their can could may might will would should not no than then
paper study approach method methods model models results show shows propose proposed new novel
based towards toward into over under between across during about which who whom whose what when
where why how all any both each few more most other some such only own same so too very s t just
also however therefore thus while whereas although because since if else one two three first second""".split())


def content_words(text: str) -> set[str]:
    """内容词 = 小写字母数字 token，长度 ≥ 4，去掉停用词。"""
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(t) >= 4 and t not in _STOP}


def leak_ratio(query: str, title: str) -> tuple[float, list[str]]:
    """泄漏度 = gold 标题的**稀有内容词**中被查询复现的比例（0~1）。

    高 = 查询把标题的词照搬了 → 这题等于"答案写在题干里"，位次无参考价值。
    用长度 ≥ 5 的词做分母（更接近"专有名词/方法名"，短词易偶然命中）。
    """
    rare = {t for t in content_words(title) if len(t) >= 5}
    if not rare:
        return 0.0, []
    hit = sorted(rare & content_words(query))
    return len(hit) / len(rare), hit


def abstract_level(q: str) -> str:
    """把查询分成「具体」/「上位」——对应 LitSearch 的 specificity。

    **只读查询文本**，不看任何检索结果 → 因此是**非循环**的分层依据
    （"难/易"不能这么分：它没有独立定义，只能从结果反推）。

    具体 = 含拉丁技术词（方法名/数据集名：BART、GrailQA、LSTM）、数字（2009、1,883）
           或短引号专名 → 这些往往是从摘要里搬来的**可检索锚点**；
    上位 = 纯中文概念描述（"如何用互信息抽取新术语"）。
    """
    lat = len(re.findall(r"[A-Za-z]{3,}", q))
    num = len(re.findall(r"\d", q))
    return "具体" if (lat >= 1 or num >= 1) else "上位"


def rank_of(order, target: int) -> int:
    """target 在 order 里的 1-based 位次；不在则 0。"""
    a = np.asarray(order)
    pos = np.where(a == target)[0]
    return int(pos[0]) + 1 if len(pos) else 0


def topk_frac(ranks: list[int], k: int) -> float:
    return float(np.mean([1.0 if 0 < r <= k else 0.0 for r in ranks]))


def call_gen(llm: LLM, sysp: str, title: str, abstract: str) -> str | None:
    usr = f"标题：{title}\n\n摘要：{str(abstract)[:2000]}"
    try:
        txt = llm.chat(sysp, usr, temperature=0.3)
        m = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        q = str(m.get("query") or "").strip()
        return q or None
    except Exception:  # noqa: BLE001
        return None


def parse_partial(text: str, k: int, n: int) -> tuple[list[int], bool]:
    """解析"前 n 名"。干净 = 恰好 n 个、全在 1..k、互不重复。"""
    m = re.search(r"\[[\s\d,]*\]", text)
    nums = json.loads(m.group(0)) if m else [int(x) for x in re.findall(r"\d+", text)]
    out, seen = [], set()
    for x in nums:
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= x <= k and x not in seen:
            seen.add(x)
            out.append(x)
    return out[:n], (len(out) >= n)


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [f"{r[c]:{floatfmt}}" if isinstance(r[c], (float, np.floating)) else str(r[c])
                 for c in df.columns]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--langs", default="zh",
                    help="出题语言，逗号分隔。zh=跨语言检索（贴近中文用户）；"
                         "en=与语料同语言（可与 LitSearch 口径对比）。默认 zh")
    ap.add_argument("--corpus", default="merged", choices=["merged", "arxiv"],
                    help="merged = 旧 6.3w + 新 54w（历史口径）；"
                         "**arxiv = 只用 arXiv 54w（= 生产口径）**。"
                         "arxiv 模式下不载入旧语料索引、从 arXiv 均匀抽题、池子全来自 arXiv")
    ap.add_argument("--pools", default="",
                    help=f"池深列表（**每语料**各取多少候选），逗号分隔，如 100,200,400。"
                         f"脚本会同时跑这些深度的**精确配对**对比 —— 因为 CE 是 pointwise，"
                         f"一次宽池打分限制到窄池子集即等价。默认只用 {POOL_BASE}")
    ap.add_argument("--old-quota", type=int, default=0,
                    help="**额外**强制补抽的旧语料题数。⚠️ 旧语料在 60w 里只占 10.5%%，"
                         "均匀抽 200 题只有 ~21 题旧语料 → CI ±17pt，分侧结论等于白抽。"
                         "补抽的题**只进**分侧统计，**不进**总体（总体仍用均匀主样本，保持无偏）")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 精排级")
    ap.add_argument("--reuse-queries", action="store_true",
                    help="复用已生成的查询（results/pilot_mixed50_queries.parquet），不重新出题")
    ap.add_argument("--queries-file", default="pilot_mixed50_queries.parquet",
                    help="查询集落盘文件名（results/ 下）。**冒烟时务必换名**，"
                         "否则会覆盖正式跑攒下的查询集（语料不变时它可精确复用）")
    ap.add_argument("--out-md", default="LITSEARCH_MIXED50.md")
    args = ap.parse_args()

    global POOLS
    want = [int(x) for x in str(args.pools).split(",") if x.strip()] or [POOL_BASE]
    POOLS = sorted(set([POOL_BASE, *want]))                  # 升序，窄→宽
    if len(POOLS) > 1:
        print(f"池子深度配对对比：{POOLS} —— CE 打分只在最大池 {max(POOLS)} 上做一次；"
              f"因为 CE 是 pointwise，各窄池的结果可**精确**读出来\n")

    load_env()
    llm = LLM()
    if not llm.ok():
        print("未配置 PAPERPILOT_LLM_*，退出")
        return 2

    # ── 语料 ────────────────────────────────────────────────
    # `--corpus arxiv`（= 生产口径）：**不载入旧语料**。把 N_OLD 置 0 之后，
    # 下游所有「旧/新」分支自动退化成「只有新」——包括池子（`r_old[:C]` 变空）。
    arxiv_only = (args.corpus == "arxiv")
    print("载入语料（**仅 arXiv，= 生产口径**）…" if arxiv_only else "载入两侧语料…")

    d_new = pd.read_parquet(NEW / "meta" / "corpus_text.parquet",
                            columns=["docid", "arxiv_id", "title", "abstract",
                                     "created", "categories"])
    ids_new = np.load(NEW / "emb" / "docid.npy")
    assert len(ids_new) == len(d_new), f"新侧行数不一致 {len(ids_new)} vs {len(d_new)}"

    if arxiv_only:
        d_old, ids_old = None, None
        N_OLD, N_NEW = 0, len(d_new)
        titles = d_new["title"].astype(str).tolist()
        abstracts = d_new["abstract"].astype(str).tolist()
        years = d_new["created"].astype(str).str[:4].tolist()
    else:
        # ⚠️ 向量是按 `text != ""` 过滤后建的；必须用同一口径对齐（否则行号全错）
        _full = pd.read_parquet(OLD / "corpus_text.parquet",
                                columns=["corpusid", "title", "abstract", "year", "text"])
        d_old = _full[_full["text"] != ""].reset_index(drop=True)
        ids_old = np.load(OLD / "emb" / "corpusid.npy")
        assert len(ids_old) == len(d_old), f"旧侧行数不一致 {len(ids_old)} vs {len(d_old)}"
        N_OLD, N_NEW = len(d_old), len(d_new)
        titles = pd.concat([d_old["title"], d_new["title"]],
                           ignore_index=True).astype(str).tolist()
        abstracts = pd.concat([d_old["abstract"], d_new["abstract"]],
                              ignore_index=True).astype(str).tolist()
        years = pd.concat([d_old["year"], d_new["created"].astype(str).str[:4]],
                          ignore_index=True).tolist()

    N_ALL = N_OLD + N_NEW
    print(f"  旧 {N_OLD:,} + 新 {N_NEW:,} = **{N_ALL:,}** 篇")

    # ── 抽样 + 出题 ─────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    qf = RESULTS / args.queries_file
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    bad = [s for s in langs if s not in LANG_HINT]
    if bad:
        print(f"未知语言 {bad}；可选 {list(LANG_HINT)}")
        return 2

    if args.reuse_queries and QF.exists():
        qd_prev = pd.read_parquet(qf)
        pg = qd_prev["gold_row"].astype(int).tolist()
        pl = (qd_prev["lang"].astype(str).tolist() if "lang" in qd_prev.columns
              else ["zh"] * len(pg))
        pb = (qd_prev["batch"].astype(str).tolist() if "batch" in qd_prev.columns
              else ["uniform"] * len(pg))
        pairs = list(zip(pg, pl))
        batches = pb
        queries = qd_prev["query"].astype(str).tolist()
        print(f"\n复用已有查询：{len(queries)} 条（seed / langs 忽略）")
    else:
        picks = np.sort(rng.choice(N_ALL, min(args.n, N_ALL), replace=False)).tolist()
        batch_of = {int(i): "uniform" for i in picks}
        n_str = sum(1 for i in picks if i < N_OLD)
        print(f"\n主样本：从 {N_ALL} 篇**均匀**抽 {len(picks)} 篇（seed={args.seed}）")
        print(f"  其中旧语料 {n_str} 篇 / 新语料 {len(picks) - n_str} 篇")

        if args.old_quota > 0 and N_OLD > 0:
            # ⚠️ 旧语料只占语料 10.5% → 均匀抽 200 题只落到 ~21 题，分侧 CI 会宽到 ±17pt。
            #    这里**额外**补抽，且明确标 batch=extra_old → 只并入"分侧统计"，
            #    **不进总体**（总体保持均匀无偏）。见脚本头部的说明。
            rest = np.setdiff1d(np.arange(N_OLD, dtype=np.int64),
                                np.asarray(picks, dtype=np.int64))
            extra = np.sort(rng.choice(rest, min(args.old_quota, len(rest)),
                                       replace=False)).tolist()
            for i in extra:
                batch_of[int(i)] = "extra_old"
            picks = sorted(set(picks) | set(extra))
            print(f"  补抽旧语料 **{len(extra)}** 篇（batch=extra_old；只进分侧，不进总体）")

        todo = [(i, lg) for i in picks for lg in langs]     # 每篇 × 每种语言 = 1 条

        def gen(job):
            i, lg = job
            yr = str(years[i])[:4]
            sysp = _SYS_GEN.format(year=f"{yr} 年" if yr.isdigit() else "",
                                   lang_hint=LANG_HINT[lg])
            return (i, lg), call_gen(llm, sysp, titles[i], abstracts[i])

        print(f"\n生成 {len(todo)} 条查询（{len(picks)} 篇 × {langs}，并行 {args.workers}）…")
        t0 = time.time()
        got: dict[tuple[int, str], str] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, (key, q) in enumerate(ex.map(gen, todo), 1):
                if q:
                    got[key] = q
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s")
        print(f"  完成 {time.time() - t0:.0f}s | 成功 {len(got)}/{len(todo)}")
        pairs = sorted(got)
        batches = [batch_of[g] for g, _ in pairs]
        queries = [got[k] for k in pairs]
        pd.DataFrame({"gold_row": [g for g, _ in pairs],
                      "lang": [lg for _, lg in pairs],
                      "batch": batches,
                      "query": queries}).to_parquet(qf, index=False)
        print(f"  已存 {qf.relative_to(HERE)}")

    # ── 索引 ────────────────────────────────────────────────
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print("\n载入向量索引…")
    m_new = np.concatenate([np.load(p) for p in sorted((NEW / "emb").glob("part_*.npy"))], axis=0)
    if arxiv_only:
        m_old, bm_old = None, None       # 不载入旧语料（省 259 MB + 时间）
        print(f"  仅新 {m_new.shape}")
    else:
        m_old = np.concatenate([np.load(p) for p in sorted((OLD / "emb").glob("part_*.npy"))],
                               axis=0)
        bm_old = SparseBM25.load(OLD / "bm25")
    bm_new = SparseBM25.load(NEW / "bm25")
    print(f"  旧 {m_old.shape if m_old is not None else '—'} | 新 {m_new.shape}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    print(f"编码 {len(queries)} 条查询（{dev}）…")
    QV = enc.encode(queries, batch_size=32, normalize_embeddings=True,
                    show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    print("载入 cross-encoder…")
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=512)
    if dev == "cuda":
        try:
            ce.model.half()
        except Exception:  # noqa: BLE001
            pass

    # ── 逐题跑流水线 ────────────────────────────────────────
    print(f"\n跑流水线（{len(pairs)} 题）…")
    t0 = time.time()
    rows = []
    llm_err: list[str] = []
    for qi, ((gi, lg), qs) in enumerate(zip(pairs, queries)):
        bc = batches[qi]
        side = "旧" if gi < N_OLD else "新"
        row_g = gi if gi < N_OLD else gi - N_OLD

        s_new = m_new @ QV[qi]
        r_new = rrf_fuse([np.argsort(-s_new)[:DEPTH], bm_new.topk(qs, DEPTH)],
                         [ALPHA, 1 - ALPHA], N_NEW)
        if arxiv_only:
            r_old = None                      # 没有旧语料这一路
        else:
            s_old = m_old @ QV[qi]
            r_old = rrf_fuse([np.argsort(-s_old)[:DEPTH], bm_old.topk(qs, DEPTH)],
                             [ALPHA, 1 - ALPHA], N_OLD)

        # ① 检索级：gold 在**本语料** hybrid 列表里的位次（池子的理论上界）
        r_retr = (rank_of(r_old, row_g) if side == "旧" else rank_of(r_new, row_g))

        # ② CE 级：候选合并 → CE 重排（**在最大池上打一次分**）
        #    arxiv 模式下 N_OLD=0 且 r_old 为 None → 池子自然只剩 arXiv 侧
        pools = {}
        for C in POOLS:
            cand = r_old[:C].tolist() if r_old is not None else []
            cand += (r_new[:C] + N_OLD).tolist()
            pools[C] = list(dict.fromkeys(cand))
        big = pools[POOLS[-1]]
        docs = [titles[g].replace("\n", " ")[:300] + ". " + abstracts[g][:1500].replace("\n", " ")
                for g in big]
        sc = np.asarray(ce.predict([(qs, d) for d in docs],
                                   batch_size=64, show_progress_bar=False), dtype=np.float64)
        sc_map = dict(zip(big, sc.tolist()))

        # ③ LLM 级：各池子的 CE top-50 → LLM 只输出前 10
        got: dict[str, int] = {}
        dirty = 0
        for C in POOLS:
            order = sorted(pools[C], key=lambda g: -sc_map[g])   # pointwise → 精确等价
            got[f"ce{C}"] = rank_of(order, gi)
            got[f"llm{C}"] = 0
            if args.no_llm or got[f"ce{C}"] == 0:
                continue
            cand = order[:K_CE]
            disp = list(range(len(cand)))
            if DISPLAY:
                disp = list(np.random.default_rng(args.seed + qi).permutation(len(cand)))
            cdocs = [(titles[cand[d]], (abstracts[cand[d]][:1000]).replace("\n", " ").strip())
                     for d in disp]
            ul = [f"检索需求：{qs}", "", "候选论文："]
            for j, (t, b) in enumerate(cdocs, 1):
                ul.append(f"{j}. 标题：{t}" + (f"\n   摘要：{b}" if b else ""))
            ul += ["", f'请输出 JSON：{{"ranking": [...]}}（编号从高到低，**只给前 {N_OUT} 名**）']
            try:
                out = llm.chat(_SYS_RANK.format(n=N_OUT, k=len(cand)), "\n".join(ul))
                perm, clean = parse_partial(out, len(cand), N_OUT)
                dirty = max(dirty, 0 if clean else 1)
                got[f"llm{C}"] = rank_of([cand[disp[p - 1]] for p in perm], gi)
            except Exception as e:  # noqa: BLE001
                # ⚠️ 之前这里只写 `dirty = 1` —— 异常被**静默吞掉**，日志里查不出
                # 是哪条题、什么错（上次出现一段 880s 的尖峰就无法归因）。
                dirty = 1
                llm_err.append(f"q{qi}/C={C} {type(e).__name__}: {str(e)[:120]}")

        lk, lw = leak_ratio(qs, titles[gi])
        rows.append({
            "q": qi, "gold_row": gi, "lang": lg, "batch": bc, "语料": side,
            "抽象层级": abstract_level(qs),
            "g_rank_retr": r_retr,
            **{f"ce{C}": got[f"ce{C}"] for C in POOLS},
            **{f"llm{C}": got[f"llm{C}"] for C in POOLS},
            "进池": int(got[f"ce{POOLS[-1]}"] > 0),
            "泄漏度": lk, "泄漏词": ",".join(lw[:6]), "格式脏": dirty,
            "query": qs, "gold_title": titles[gi],
        })
        if (qi + 1) % 10 == 0:
            print(f"  {qi + 1}/{len(pairs)}  {time.time() - t0:.0f}s")

    d = pd.DataFrame(rows)

    # ── 汇总 ────────────────────────────────────────────────
    # ⚠️ 口径：**总体只用均匀主样本**（batch=uniform）→ 保持对语料构成无偏；
    #    补抽的旧语料题（batch=extra_old）只进「分侧 / 分层」统计。
    uni = d[d["batch"] == "uniform"]
    C_BASE, C_MAIN = POOLS[0], POOLS[-1]
    # 主臂池子的**总大小**：arxiv 模式只有一个语料来源，合并模式是两路相加
    POOL_N = C_MAIN if arxiv_only else 2 * C_MAIN
    CE_M, LLM_M = f"ce{C_MAIN}", f"llm{C_MAIN}"
    CE_B, LLM_B = f"ce{C_BASE}", f"llm{C_BASE}"
    ks = (1, 5, 10, 20, 50, 100)

    stages = [("① hyb0.5（本语料）", "g_rank_retr")]
    sz = (lambda C: C if arxiv_only else 2 * C)      # 池子**总大小**
    for C in POOLS:
        stages.append((f"② + CE（池 {sz(C)}）", f"ce{C}"))
    if not args.no_llm:
        for C in POOLS:
            stages.append((f"③ + LLM（池 {sz(C)}）", f"llm{C}"))

    def stage_table(sub: pd.DataFrame, tag: str) -> list[dict]:
        return [{"分组": tag, "阶段": name,
                 **{f"R@{k}": topk_frac(sub[col].tolist(), k) for k in ks}}
                for name, col in stages]

    stage_rows = []
    for lg in sorted(uni["lang"].unique()):
        stage_rows += stage_table(uni[uni["lang"] == lg], f"总体·均匀({lg})")
    st = pd.DataFrame(stage_rows)

    # ── 池子深度配对对比：**同一批题、同一批 CE 分数** → 精确配对 ────
    pool_rows = []
    if len(POOLS) > 1:
        for name, sub in (("总体·均匀", uni),
                          ("旧语料", d[d["语料"] == "旧"]),
                          ("新语料", d[d["语料"] == "新"]),
                          ("具体型", d[d["抽象层级"] == "具体"]),
                          ("上位型", d[d["抽象层级"] == "上位"])):
            if sub.empty:
                continue
            row = {"分层": f"{name} ({len(sub)})",
                   f"天花板@{C_BASE}": topk_frac(sub["g_rank_retr"].tolist(), C_BASE),
                   f"天花板@{C_MAIN}": topk_frac(sub["g_rank_retr"].tolist(), C_MAIN)}
            for tag, cb, cm in (("CE", CE_B, CE_M), ("LLM", LLM_B, LLM_M)):
                if tag == "LLM" and args.no_llm:
                    continue
                row[f"{tag}R@1@{C_BASE}"] = topk_frac(sub[cb].tolist(), 1)
                row[f"{tag}R@1@{C_MAIN}"] = topk_frac(sub[cm].tolist(), 1)
                row[f"{tag}ΔR@1"] = 100 * (topk_frac(sub[cm].tolist(), 1)
                                           - topk_frac(sub[cb].tolist(), 1))
                row[f"{tag}ΔR@5"] = 100 * (topk_frac(sub[cm].tolist(), 5)
                                           - topk_frac(sub[cb].tolist(), 5))
            pool_rows.append(row)
    pool_tbl = pd.DataFrame(pool_rows)

    # ── 分层切面（零额外成本：全从原始记录现算）──────────────
    strat = []
    for name, sub in (("旧语料", d[d["语料"] == "旧"]),
                      ("新语料", d[d["语料"] == "新"]),
                      ("具体型（带技术锚点）", d[d["抽象层级"] == "具体"]),
                      ("上位型（纯概念）", d[d["抽象层级"] == "上位"])):
        if sub.empty:
            continue
        strat.append({
            "分层": f"{name} ({len(sub)})",
            "① 检索R@1": topk_frac(sub["g_rank_retr"].tolist(), 1),
            "① 检索R@100": topk_frac(sub["g_rank_retr"].tolist(), 100),
            "② CE R@1": topk_frac(sub[CE_M].tolist(), 1),
            "② CE R@5": topk_frac(sub[CE_M].tolist(), 5),
            "③ LLM R@1": topk_frac(sub[LLM_M].tolist(), 1),
            "进池率": float(sub["进池"].mean()),
        })
    strat_tbl = pd.DataFrame(strat)

    # ── 失败归因：失败到底发生在「召回」还是「排序」──────────
    #   召回失败 = gold 压根没进 200 候选池 → 加宽池子/换索引才能救
    #   排序失败 = 进了池子但没排进 top-5 → 精排的问题
    def fault(sub: pd.DataFrame, name: str) -> dict:
        if sub.empty:      # ⚠️ 列名必须与下面完全一致，否则 pandas 会多出 NaN 列
            return {"分层": f"{name} (0)", "n": 0, "召回失败(未进池)": "-",
                    "排序失败(进池未进top5)": "-", "成功(CE top5)": "-"}
        pool = sub["进池"] == 1
        ok = (sub[CE_M] > 0) & (sub[CE_M] <= 5)
        return {"分层": f"{name} ({len(sub)})", "n": len(sub),
                "召回失败(未进池)": f"{100 * (~pool).mean():.1f}%",
                "排序失败(进池未进top5)": f"{100 * (pool & ~ok).mean():.1f}%",
                "成功(CE top5)": f"{100 * ok.mean():.1f}%"}
    fault_tbl = pd.DataFrame([
        fault(uni, "总体·均匀"),
        fault(d[d["语料"] == "旧"], "旧语料"),
        fault(d[d["语料"] == "新"], "新语料"),
        fault(d[d["抽象层级"] == "具体"], "具体型"),
        fault(d[d["抽象层级"] == "上位"], "上位型"),
    ])

    def grp(sub: pd.DataFrame, name: str) -> dict:
        if sub.empty:                      # 空分组直接给 nan，别让 numpy 报警
            return {"分组": f"{name} (0)", "R@1": float("nan"), "R@5": float("nan"),
                    "平均泄漏度": float("nan")}
        return {"分组": f"{name} ({len(sub)})",
                "R@1": topk_frac(sub[CE_M].tolist(), 1),
                "R@5": topk_frac(sub[CE_M].tolist(), 5),
                "平均泄漏度": float(sub["泄漏度"].mean())}

    leak_rows = pd.DataFrame([
        grp(d, "全部"),
        grp(d[d["泄漏度"] >= 0.5], "泄漏高 ≥0.5"),
        grp(d[(d["泄漏度"] > 0) & (d["泄漏度"] < 0.5)], "泄漏中 0~0.5"),
        grp(d[d["泄漏度"] == 0], "泄漏 0（完全没抄标题）"),
    ])

    # 逐题明细（按分层 → 泄漏度降序）
    det = d[["q", "lang", "batch", "语料", "抽象层级", "泄漏度",
             "g_rank_retr", CE_M, LLM_M, "query", "gold_title"]].copy()
    if len(POOLS) > 1:
        det[CE_B] = d[CE_B]
    det["batch"] = det["batch"].map({"uniform": "均匀", "extra_old": "补旧"})
    det = det.sort_values(["batch", "语料", "泄漏度"], ascending=[True, True, False]) \
             .reset_index(drop=True)
    det = det.rename(columns={"batch": "批"})
    det["g_rank_retr"] = det["g_rank_retr"].replace(0, ">1000")
    det[CE_M] = det[CE_M].replace(0, f">{POOL_N}")
    det[LLM_M] = det[LLM_M].replace(0, "未进前10")
    det["query"] = det["query"].str.slice(0, 80)
    det["gold_title"] = det["gold_title"].str.slice(0, 46)
    det["泄漏度"] = det["泄漏度"].round(2)

    n_uni = len(uni)
    print("\n" + "=" * 100)
    print(f"混合语料试点 | 语料 {N_ALL:,} 篇 | 总查询 {len(d)} 条"
          f"（均匀主样本 {n_uni} + 补抽旧语料 {len(d) - n_uni}）")
    print("=" * 100)
    print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(POOLS) > 1:
        print(f"\n【★ 池子深度配对对比（对照 C={C_BASE} vs 主臂 C={C_MAIN}）"
              f"—— 同一批题、同一批 CE 分数】")
        print(pool_tbl.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n【分层切面】（分侧/分层用全量；总体见上表）")
    print(strat_tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n【失败归因：召回 vs 排序】")
    print(fault_tbl.to_string(index=False))
    print("\n【泄漏审计：CE 级 R@k 按泄漏度分组】")
    print(leak_rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[ 成本 ] 调用 {llm.calls} 次 | prompt {llm.ptok:,} | completion {llm.ctok:,}")
    if llm_err:
        print(f"[ ⚠ LLM 失败 {len(llm_err)} 次 ]（这些题的该臂按'未命中'计）")
        for e in llm_err[:8]:
            print(f"    {e}")
        if len(llm_err) > 8:
            print(f"    … 另有 {len(llm_err) - 8} 次")
    else:
        print("[ LLM ] 无失败")

    raw_csv = RESULTS / (Path(args.out_md).stem + "_raw.csv")
    d.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    md = RESULTS / args.out_md
    if arxiv_only:
        hdr = (f"# arXiv-only 基准：**从 {N_ALL:,} 篇 arXiv 均匀抽论文 → 出题 → 看 gold 排第几**\n\n"
               f"- **语料 = 生产口径**：仅 arXiv **{N_NEW:,}** 篇（seed={args.seed}）；"
               f"旧语料不参与\n"
               f"- 抽样：**均匀 {n_uni}** 条（无一侧过采样 —— arXiv 语料单一）\n")
        pipe = f"`hyb0.5 arXiv top-C` → 候选 C 篇 → `CE` → top-{K_CE} → `LLM 只输出前 {N_OUT}`"
    else:
        hdr = (f"# 混合语料试点：**从 {N_ALL:,} 篇随机抽论文 → 按论文出题 → 看 gold 排第几**\n\n"
               f"- 索引空间：旧 {N_OLD:,} + 新 {N_NEW:,} = **{N_ALL:,}** 篇（seed={args.seed}）\n"
               f"- **抽样口径**：均匀主样本 **{n_uni}** 条（对语料构成无偏，**总体指标只用它**）\n"
               f"  + 补抽旧语料 **{len(d) - n_uni}** 条（`batch=extra_old`，**只进分侧统计**）\n"
               f"  ——因为旧语料只占语料 10.5%，不补抽的话分侧 CI 会宽到 ±17pt、等于白抽\n")
        pipe = (f"`hyb0.5 本语料 top-C` → 两语料候选合并(2C) → `CE` → top-{K_CE} → "
                f"`LLM 只输出前 {N_OUT}`")
    md.write_text(
        hdr +
        f"- **池子深度**：{POOLS}（每语料各取 C）| 流水线：{pipe}\n"
        f"- 出题 prompt：`" + _SYS_GEN.format(year="{该论文年份}", lang_hint="{语言约束}") + "`\n"
        f"- 调用 {llm.calls} 次 | prompt {llm.ptok:,} token | completion {llm.ctok:,}\n\n"
        "## ① 分级命中率（总体 · 均匀主样本）\n\n" + md_table(st) +
        "\n\n> `① hyb0.5（本语料）` = gold 在**所属语料**的 hybrid 列表里的位次 —— "
        "这一级**不含跨语料竞争**，即池子的理论上界。\n"
        "> `② + CE` 起才是跨语料合并后的真实位次。\n\n"
        + (f"## ② ★ 池子深度配对对比（C={C_BASE} → C={C_MAIN}）\n\n"
           + md_table(pool_tbl, floatfmt="+.4f") +
           f"\n\n> **这是精确配对**：同一批查询、**同一批 CE 分数**（CE 是 pointwise，"
           f"宽池打分限制到窄池子集逐位等价）。\n"
           f"> 拆开看两件事是否互相抵消：\n"
           f">  · `天花板`列 = 池子加宽能多捞回多少（**召回那一半**）\n"
           f">  · `CEΔR@1` / `LLMΔR@1` = 实际落地多少（**排序那一半有没有把收益吃掉**）\n"
           f">  · 若 `天花板 +4pt` 而 `ΔR@1 ≈ 0` → 新进来的候选**排不上去**，加宽无效\n\n"
           if len(POOLS) > 1 else "") +
        "## ③ 分层切面\n\n" + md_table(strat_tbl) +
        "\n\n> 分层都**只读查询文本 / 抽样来源**，不看检索结果 → **非循环**。\n"
        "> 抽象层级：`具体` = 查询含拉丁技术词 / 数字 / 专名（可检索锚点）；"
        "`上位` = 纯中文概念描述。\n\n"
        "## ④ 失败归因：**失败发生在召回还是排序**\n\n" + md_table(fault_tbl) +
        f"\n\n> `召回失败` = gold 压根没进 {POOL_N} 候选池 → 要**加宽池子 / 换索引**才救得回；\n"
        "> `排序失败` = 进了池子但没排进 top-5 → 是**精排**的问题。两者修法完全不同。\n\n"
        "## ⑤ 泄漏审计（**没有这张表，上面数字不可解释**）\n\n" + md_table(leak_rows) +
        "\n\n> 泄漏度 = gold **标题**里长度 ≥5 的内容词被查询复现的比例。\n"
        "> ⚠️ 该指标是**词面**的：中文查询用上位概念转述英文标题时，词面重叠天然为 0，\n"
        "> 所以「泄漏 0」**不等于没有语义指向**，只是没照抄词。\n\n"
        "## ⑥ 逐题位次明细\n\n" + md_table(det, floatfmt=".2f") + "\n",
        encoding="utf-8")
    print(f"\n已写出：\n  results/{md.name}\n  results/{raw_csv.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
