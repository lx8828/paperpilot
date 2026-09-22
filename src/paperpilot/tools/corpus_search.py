"""语料检索（工具 ①）：自然语言查询 → 最相关的 N 篇论文。

## 流水线（与评测口径**逐位一致**，2026-09-20 定稿）

    hyb0.5 = RRF(BM25, dense)  α=0.5，组件深度 1000
      → top-200                 （池深 C=200；arXiv-only 语料）
      → bge-reranker-v2-m3      （pointwise cross-encoder）→ top-50
      → deepseek-chat           （listwise，只输出前 10）
      → 交付 top-5

    基准（arXiv-only 200 题）：R@1 0.875 / R@5 0.885，端到端 ~4.0 秒，100% 可取 PDF。

## 架构位置（**方案 A：索引留在原地**）

索引在 `retrieval/data/arxiv/`（研究区），应用在 `src/paperpilot/`（生产区）。
现在**按路径引用**，把所有跨区耦合**收在文件底部的「适配段」**里 ——
将来上方案 C（常驻 HTTP 服务）只替换那一段，本文件其余部分不动。

## 常驻单例

索引与模型在**首次调用时加载一次**并常驻（`_State`）：
    稠密矩阵 2.21 GB（RAM）· BM25 ~0.6 GB（RAM）· 两个模型 ~2.4 GB（显存）

## 为什么现在**不用** FAISS

评测要的是**精确**召回 —— 池内丢一篇，后面的 CE/LLM 位次全受影响，指标会被污染
且难以归因。45 万篇下暴力扫约 250 ms/查询，用于评测完全够（597 题 ≈ 2.5 分钟）。
**要服务多并发时再上 HNSW**，且上线前必须先验证 `ANN top-100 vs 精确 top-100`
的重合率 ≥ 99.5%。

用法：
    from paperpilot.tools.corpus_search import search
    res = search("如何用对比学习做跨语言摘要", k=5)
    for p in res.papers:
        print(p.rank, p.arxiv_id, p.title)
"""
from __future__ import annotations

import sys
import datetime as _dt
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]           # src/paperpilot/tools/x.py → 根


def _to_day(s: str) -> int:
    """`'2026-04-01'` / `'2026-04'` / `'2026'` → **天数**（自 1970-01-01）。

    只接受这三种粗粒度写法（时间窗本来就是粗筛）。无法解析时**抛 ValueError**
    —— 刻意**不静默当成"不限制"**：那会让调用方以为筛了、其实没筛。
    """
    t = str(s).strip()
    if len(t) == 4:
        t += "-01-01"
    elif len(t) == 7:
        t += "-01"
    try:
        d = _dt.date.fromisoformat(t)
    except ValueError as e:
        raise ValueError(f"无法解析日期 {s!r}（用 `2026` / `2026-04` / `2026-04-01`）") from e
    return (d - _dt.date(1970, 1, 1)).days

# ── 定稿参数（改动必须同步更新评测报告，否则指标不可迁移）────────────────────
ALPHA = 0.5          # 臂内 hybrid 的向量权重
DEPTH = 1000         # RRF 组件列表深度
POOL = 200           # 进 CE 的候选数（C=200，2026-09-20 定稿）
K_CE = 50            # CE 后取前 K 交给 LLM
N_OUT = 10           # LLM 只输出前 N 名（交付 top-5 只需要这些）
DOC_TITLE_CHARS = 300
DOC_ABS_CHARS = 1500     # 喂给 CE 的摘要长度
LLM_ABS_CHARS = 1000     # 喂给 LLM 的摘要长度


# ── 返回值 ───────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    """一条检索结果。`arxiv_id` 可直接交给 `arxiv_fetch.fetch_arxiv()`。"""
    docid: int
    arxiv_id: str
    title: str
    abstract: str
    score: float                 # CE 分（未归一，仅用于同查询内比较）
    rank: int                    # 交付位次（1-based）
    stage: str                   # 该篇走到的最后一级：llm / ce
    created: str = ""            # 首次提交日（arXiv `created`，`YYYY-MM-DD`）

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"


@dataclass
class SearchResult:
    papers: list[Paper] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)   # 各阶段秒数
    n_candidates: int = 0        # 进 CE 的候选数
    n_llm_input: int = 0         # 交给 LLM 的候选数
    n_in_window: int = 0         # 时间窗内的语料篇数（不加时间窗时 = 全集）
    degraded: str = ""           # 非空 = 降级说明（如 LLM 失败）

    @property
    def total_ms(self) -> float:
        return 1000 * sum(self.timings.values())


# ── 常驻单例 ─────────────────────────────────────────────────────────────────

class _State:
    """索引 + 模型的常驻容器。首次 `get()` 时加载，之后复用。"""
    _inst: "_State | None" = None

    def __init__(self) -> None:
        _ensure_hf_before_torch()                   # ⚠️ 必须在 st/torch 之前
        import numpy as np
        import pandas as pd
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from _adapter import (INDEX_DIR, SparseBM25, load_env, meta_columns)

        load_env()
        self.np = np
        # (title, abstract, arxiv_id, created) 四个 ndarray —— **行号与向量严格对齐**
        self.meta = meta_columns(INDEX_DIR)
        self.n = len(self.meta[0])
        # `created` 预转成**天数**（自 1970-01-01）：时间窗掩码于是只是一次整数比较
        # （~1ms）。若每次现比字符串，540k 次会白烧几十毫秒。
        self.cdays = np.asarray(self.meta[3], dtype="datetime64[D]").astype("int64")

        parts = sorted((INDEX_DIR / "emb").glob("part_*.npy"))
        if not parts:
            raise RuntimeError(f"找不到向量分片：{INDEX_DIR / 'emb'}（索引没建好？）")
        self.vectors = np.concatenate([np.load(p) for p in parts], axis=0)
        self.bm25 = SparseBM25.load(INDEX_DIR / "bm25")
        if self.bm25 is None:
            raise RuntimeError(f"找不到 BM25 索引：{INDEX_DIR / 'bm25'}")

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc = SentenceTransformer("BAAI/bge-m3", device=self.dev, trust_remote_code=True)
        self.enc.max_seq_length = 512
        if self.dev == "cuda":
            self.enc.half()
        self.ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device=self.dev, max_length=512)
        if self.dev == "cuda":
            # ⚠️ **必须与评测口径一致**（`pilot_mixed50.py` 同样做 half()）。
            #    不 half 会退到 fp32：慢 2~3 倍，**且打分有细微差异**
            #    → 生产排序 ≠ 评测排序，指标不可迁移。
            try:
                self.ce.model.half()
            except Exception:  # noqa: BLE001
                pass

        from _adapter import LLM
        self.llm = LLM()

    @classmethod
    def get(cls) -> "_State":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst


# ── 主入口 ───────────────────────────────────────────────────────────────────

def search(query: str, *, k: int = 5, use_llm: bool = True,
           pool: int = POOL, since: str | None = None,
           until: str | None = None) -> SearchResult:
    """检索与 `query` 最相关的 `k` 篇论文。

    Args:
        query: 自然语言查询（中文/英文均可，跨语言检索）。
        k: 交付篇数（产品取 5；内部候选池不受影响）。
        use_llm: False 时只到 CE 级（省 ~0.8 秒与 LLM 费用；用于消融/降级）。
        pool: 进 CE 的候选数，默认 `POOL`（定稿 200）。
        since: 只要 `created >= since` 的论文（`2026` / `2026-04` / `2026-04-01`）。
        until: 只要 `created <= until`。

    Returns:
        `SearchResult`。**不抛异常**（LLM 失败 → 降级为 CE 结果，`degraded` 说明原因）；
        但 `since`/`until` **格式非法会抛 `ValueError`**（宁可响亮失败）。

    ⚠️ **时间窗是前置过滤**（在 dense / BM25 取 top 之前就 mask 掉），不是事后过滤：
    近半年只占语料约 13%，事后过滤会把 DEPTH=1000 的候选池掏空到百来篇 → 召回断崖。
    代价只有一次整数比较（`cdays` 已预转成天数）。
    """
    q = str(query or "").strip()
    if not q:
        return SearchResult(degraded="空查询")

    st = _State.get()
    np = st.np
    t: dict[str, float] = {}
    res = SearchResult(timings=t)

    # ⓿ 时间窗掩码（**前置**；两个参数都空 = 不限制）
    mask = None
    if since or until:
        mask = np.ones(st.n, dtype=bool)
        if since:
            mask &= st.cdays >= _to_day(since)
        if until:
            mask &= st.cdays <= _to_day(until)
        res.n_in_window = int(mask.sum())
        if res.n_in_window == 0:
            return SearchResult(timings=t, degraded=f"时间窗 {since or ''}~{until or ''} 内没有论文")
    else:
        res.n_in_window = st.n

    # ① 查询向量
    t0 = time.time()
    qv = st.enc.encode([q], batch_size=1, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)[0].astype(np.float32)
    t["query_enc"] = time.time() - t0

    # ② hyb0.5：dense 与 BM25 各自取 top-DEPTH 后做 RRF
    t0 = time.time()
    s_dense = st.vectors @ qv
    bm_s = st.bm25.score(q)
    depth = min(DEPTH, res.n_in_window)          # 窗口内不足 DEPTH 时别多取
    if mask is not None:                         # 前置过滤：窗口外置 -inf，自然排到最后
        s_dense = np.where(mask, s_dense, -np.inf)
        bm_s = np.where(mask, bm_s, -np.inf)
    dense_list = np.argsort(-s_dense)[:depth]
    bm_list = np.argsort(-bm_s)[:depth]
    from _adapter import rrf_fuse
    ranked = rrf_fuse([dense_list, bm_list], [ALPHA, 1 - ALPHA], st.n)
    t["retrieve"] = time.time() - t0

    # ③ CE 精排（pointwise，逐篇独立打分）
    cand = ranked[:pool].tolist()
    res.n_candidates = len(cand)
    titles, abstracts = st.meta[0], st.meta[1]
    docs = [f"{titles[g].replace(chr(10), ' ')[:DOC_TITLE_CHARS]}. "
            f"{abstracts[g][:DOC_ABS_CHARS].replace(chr(10), ' ')}" for g in cand]
    t0 = time.time()
    sc = st.ce.predict([(q, d) for d in docs], batch_size=64, show_progress_bar=False)
    t["ce"] = time.time() - t0
    order = [cand[i] for i in np.argsort(-np.asarray(sc, dtype=np.float64)).tolist()]
    score_of = dict(zip(cand, np.asarray(sc, dtype=np.float64).tolist()))
    stage_of: dict[int, str] = {}

    # ④ LLM listwise 精排（只输出前 N_OUT）
    final = order
    if use_llm and st.llm.ok():
        head = order[:K_CE]
        res.n_llm_input = len(head)
        ul = [f"检索需求：{q}", "", "候选论文："]
        for j, g in enumerate(head, 1):
            body = str(abstracts[g])[:LLM_ABS_CHARS].replace("\n", " ").strip()
            ul.append(f"{j}. 标题：{titles[g]}" + (f"\n   摘要：{body}" if body else ""))
        ul += ["", f'请输出 JSON：{{"ranking": [...]}}（编号从高到低，**只给前 {N_OUT} 名**）']
        t0 = time.time()
        try:
            from _adapter import parse_partial, sys_rank
            out = st.llm.chat(sys_rank(N_OUT, len(head)), "\n".join(ul))
            perm, clean = parse_partial(out, len(head), N_OUT)
            if perm:
                picked = [head[p - 1] for p in perm]
                final = picked + [g for g in order if g not in set(picked)]
                for g in picked:
                    stage_of[g] = "llm"
                if not clean:
                    res.degraded = f"LLM 输出不完整（{len(perm)}/{N_OUT} 名），已用解析到的部分"
            else:
                res.degraded = "LLM 未解析出有效排序，已回退 CE 结果"
        except Exception as e:  # noqa: BLE001
            res.degraded = f"LLM 精排失败，已回退 CE 结果：{type(e).__name__}: {e}"
        t["llm"] = time.time() - t0

    # ⑤ 组装
    for r, g in enumerate(final[:k], 1):
        res.papers.append(Paper(
            docid=int(g),
            arxiv_id=str(st.meta[2][g]),
            title=str(titles[g]),
            abstract=str(abstracts[g]),
            score=float(score_of.get(g, 0.0)),
            rank=r,
            stage=stage_of.get(g, "ce"),
            created=str(st.meta[3][g]),
        ))
    return res


# ── HF 离线（必须在 torch / sentence_transformers 导入之前）────────────────────

def _ensure_hf_before_torch() -> None:
    """把 HF 缓存指到本地并切离线 —— 否则每次加载模型都会卡在 Hub 的 5 次重试。

    ⚠️ 必须在 `import torch` / `sentence_transformers` **之前**调用：
    `huggingface_hub` 在 import 期就把这些开关读成常量。
    """
    from _adapter import ensure_hf_home
    ensure_hf_home()


# ════════════════════════════════════════════════════════════════════════════
# 适配段 —— 与「研究区」`retrieval/` 的**唯一**耦合点
#
#   方案 A（当前）：索引与评测代码都留在 `retrieval/`，这里按路径引用。
#   方案 C（以后）  ：把本段整体换成「对常驻检索服务的 HTTP 调用」，上层不动。
#
#   ⚠️ prompt 与解析函数**故意 import 而不复制**：生产 prompt 必须与评测
#      prompt 同源，否则评测出来的指标不可迁移到线上。
# ════════════════════════════════════════════════════════════════════════════

def _install_adapter() -> None:
    """建一个名为 `_adapter` 的伪模块，把研究区符号集中到一处。"""
    import types

    rroot = ROOT / "retrieval"
    scripts = rroot / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    import numpy as np
    import pandas as pd
    from _hfcache import ensure_hf_home                     # noqa: F401
    from eval_retrieval import SparseBM25, rrf_fuse         # noqa: F401
    from llm_rerank import LLM, load_env                    # noqa: F401
    from pilot_mixed50 import _SYS_RANK, parse_partial      # noqa: F401

    INDEX_DIR = rroot / "data" / "arxiv"
    _CACHE: dict[str, tuple] = {}

    def meta_columns(index_dir: Path) -> tuple:
        """按**向量行序**取 `(titles, abstracts, arxiv_ids, created)`。

        ⚠️ 四者必须与 `emb/docid.npy` 同序 —— 建索引时已校验过
        （向量 / docid / 语料表 / BM25 四者行数一致），这里只做一次缓存。
        `created` 是 arXiv 首次提交日（`YYYY-MM-DD`），**时间检索用它**。
        """
        key = str(index_dir)
        if key not in _CACHE:
            df = pd.read_parquet(index_dir / "meta" / "corpus_text.parquet",
                                 columns=["docid", "arxiv_id", "title",
                                          "abstract", "created"])
            assert (df["docid"].to_numpy() == np.arange(len(df))).all(), \
                "corpus_text 行序与 docid 不一致 —— 索引被改过，请重建"
            _CACHE[key] = (df["title"].astype(str).to_numpy(),
                           df["abstract"].astype(str).to_numpy(),
                           df["arxiv_id"].astype(str).to_numpy(),
                           df["created"].astype(str).to_numpy())
        return _CACHE[key]

    def sys_rank(n: int, k: int) -> str:
        return _SYS_RANK.format(n=n, k=k)

    m = types.ModuleType("_adapter")
    m.INDEX_DIR = INDEX_DIR
    m.SparseBM25 = SparseBM25
    m.rrf_fuse = rrf_fuse
    m.LLM = LLM
    m.load_env = load_env
    m.ensure_hf_home = ensure_hf_home
    m.parse_partial = parse_partial
    m.meta_columns = meta_columns
    m.sys_rank = sys_rank
    sys.modules["_adapter"] = m


_install_adapter()
