"""Claim embedding 索引：BAAI/bge-m3 本地向量检索（L2 Claim Retrieval）。

职责：
    - 以 report.groups 的 rep_text（中文主张）为检索单元构建向量
    - 向量缓存到 assets/artifacts/out_views/<stem>.gvec.npy（+ <stem>.gidx.json 记顺序），
      避免每次问答都重算（CPU encode 一篇约几十秒，缓存后秒回）
    - search(query) 返回 topK 命中（带 score，供组装 ClaimHit）

注意：
    - 模型单例懒加载（首次问答加载 ~15s，之后复用）
    - 缓存失效条件：groups 数量或 group_id 顺序与索引时不一致 → 重建
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

import math
import re
from collections import Counter

import numpy as np

from paperpilot.tools import mock_llm   # 演示模式（PAPERPILOT_MOCK_EMBED=1）的实现

MODEL_NAME = "BAAI/bge-m3"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 模型已在本地 huggingface 缓存（避免每次加载联网访问 hub）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── 轻量 BM25（无第三方依赖）：与向量检索做 RRF 融合 ─────────────────────
# 英文按词切、中文按字切；QASPER/长文场景的精确专名（AMI IHM、Meta-LSTM 等）
# 向量召回弱，BM25 的精确匹配可互补。
_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for t in _WORD_RE.findall(text.lower()):
        out.append(t)
    return out


class BM25Index:
    """Okapi BM25，doc 级词频统计，支持英/中混合。构造 O(N*len)。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        n = len(docs)
        self.tfs: list[Counter[str]] = []
        self.lens: list[int] = []
        df: Counter[str] = Counter()
        for d in docs:
            toks = _tokenize(d)
            c = Counter(toks)
            self.tfs.append(c)
            self.lens.append(len(toks))
            df.update(c.keys())
        self.n = n
        self.avgdl = (sum(self.lens) / n) if n else 0.0
        # idf（bm25+ 平滑，避免 df>n）
        self.idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in df}

    def score(self, query: str) -> np.ndarray:
        """返回每 doc 的 bm25 分数（未归一）。"""
        q = Counter(_tokenize(query))
        out = np.zeros(self.n, dtype="float64")
        for term, qf in q.items():
            idf = self.idf.get(term)
            if idf is None or qf == 0:
                continue
            for i in range(self.n):
                f = self.tfs[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl) if self.avgdl else 1.0
                out[i] += idf * qf * f / denom
        return out


def rrf_order(vec_scores: np.ndarray, bm_scores: np.ndarray | None,
              k: int = 60, w_vec: np.ndarray | None = None,
              w_bm: np.ndarray | None = None) -> np.ndarray:
    """RRF 全序（不截断）：score = Σ weight/(k + rank)。返回按 RRF 降序的全部索引。

    w_vec/w_bm: 每块的两路权重（默认 None = 全 1，即生产原样）。

    注 1：曾尝试对**外部块**（MinerU 表格/公式，`xtbl-*`）**屏蔽 BM25 路**，
    理由是表块的 BM25 位次看起来偏差。**实测证否、已回退**：混池 RRF 里外部块
    恰恰靠 BM25 路挣分，屏蔽后目标表块 top-12 命中 14/21→**4/21**、位次中位 6→18
    （见 `qa/recall/TABLE_POLICY_AB_20260911.md`）。**不要**再走这条路。
    注 2：正确做法是**只调外部块的两路权重（加总）**，文本块完全不动 ——
    见 `_ext_weights` 与 `qa/recall/WEIGHT_SWEEP_20260911.md`。
    """
    n = len(vec_scores)
    rrf = np.zeros(n, dtype="float64")
    order_v = np.argsort(-vec_scores)
    for r, i in enumerate(order_v):
        rrf[i] += (1.0 if w_vec is None else float(w_vec[i])) / (k + r + 1)
    if bm_scores is not None:
        order_b = np.argsort(-bm_scores)
        for r, i in enumerate(order_b):
            rrf[i] += (1.0 if w_bm is None else float(w_bm[i])) / (k + r + 1)
    return np.argsort(-rrf)


def _ext_weights(chunks: list[Any], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """外部块两路权重 (2α, 2(1-α))；文本块恒 (1,1)。

    外部块 = MinerU 表格/公式（`xtbl-*`）。**α=0.5 ⇒ 外部块也是 (1,1) = 生产原样**，
    此时直接返回全 1（零额外开销，不构建掩码）。
    离线扫描（250 题 + A 桶 21 题，`qa/recall/WEIGHT_SWEEP_20260911.md`）：
    α=0.75 时表格块 MRR 0.468→0.591、进 top-12 14/21→15/21（位次中位 7→2），
    而**纯文本 gold 的 R@k 完全不变**、MRR 仅 −0.3pt。
    默认 α=0.5（线上行为逐位不变）；调大通过 env `PAPERPILOT_EXT_RRF_ALPHA`。
    """
    n = len(chunks)
    wv = np.ones(n, dtype="float64")
    wb = np.ones(n, dtype="float64")
    if abs(alpha - 0.5) < 1e-9:
        return wv, wb
    from paperpilot.tools.mineru_bridge import EXT_CHUNK_PREFIX
    for i, c in enumerate(chunks):
        if str(getattr(c, "chunk_id", "")).startswith(EXT_CHUNK_PREFIX):
            wv[i], wb[i] = 2.0 * alpha, 2.0 * (1.0 - alpha)
    return wv, wb


def _ext_quota() -> int:
    """外部块（表/公式）名额：env `PAPERPILOT_EXT_QUOTA`，**默认 2 = 开启并集**（2026-09-12 起）。

    正数 m：**并集** —— 正文 top_k 不变，额外追加表池 top-m（返回 top_k+m 块，**正文零损失**）。
    负数 m：**替换** —— 正文 top-(top_k-|m|) ∪ 表池 top-|m|（总数仍是 top_k，正文让出槽位）。
    `=0` 可退回改动前行为（候选严格 = 混池 top_k）。

    **为什么默认开**（证据见 `qa/recall/UNION_AND_CONTEXT_20260911.md` §7/§8、
    汇总见 `qa/recall/TABLE_LINE_STATUS_20260912.md`）：
      · 机制：表块在**混合池**里要和 25~42 个正文块比分数，命中被压低；在**表池内**目标表进
        top-2 达 86%。故不靠调权重（改不了"和谁比"），而是**给表池独立名额**。
      · 检索层（生产链路，n=22，口径修正后）确定性：目标表进候选 15/22 → **19/22（+4、零回退）**。
      · 端到端（A 桶 36 题单轮，同裁判）：15/36 → **19/36（+4）**，churn 6（新增 5 / 回退 1），
        与检索层预测同向同量；上下文真变长的 4 题里 3 题变好。
      · 代价：上下文 +m 块（top_k=12 时 +17% 体量）。
    不采用"替换"模式：m=3 要净丢 11 个正文题才换 +3 个表题。
    """
    try:
        return int(os.environ.get("PAPERPILOT_EXT_QUOTA", "2") or 0)
    except ValueError:
        return 2


def _ext_mask(chunks: list[Any]) -> np.ndarray:
    from paperpilot.tools.mineru_bridge import EXT_CHUNK_PREFIX
    return np.array([str(getattr(c, "chunk_id", "")).startswith(EXT_CHUNK_PREFIX)
                     for c in chunks])


def rrf_merge(vec_scores: np.ndarray, bm_scores: np.ndarray | None,
              top_k: int, k: int = 60) -> list[int]:
    """RRF 融合：score = Σ 1/(k + rank)。vec 必给；bm 可选（None 时只按 vec 排）。"""
    return list(rrf_order(vec_scores, bm_scores, k)[:top_k].tolist())

# agents/embedder.py → 项目根
ROOT = Path(__file__).resolve().parents[3]
VIEW_DIR = ROOT / "assets/artifacts/out_views"
# ChunkIndex 的向量缓存目录（**只这一项**可被 env 覆盖）。
# 用途：A/B 两臂的表文本不同 → cvec 指纹不同 → 共用目录会来回覆盖重建（每轮白烧 40 分钟）。
# 用 PAPERPILOT_CHUNK_VIEW_DIR 让每臂各用一份；ClaimIndex 的 gvec 与 report.json 仍在 VIEW_DIR。
CHUNK_VIEW_DIR = Path(os.environ.get("PAPERPILOT_CHUNK_VIEW_DIR") or VIEW_DIR)
MAX_DOCS_PER_PDF = 400      # 单篇主张数上限（防御异常大文件）
EMBED_DIM = 1024

_model = None


def _get_model():
    """模型单例（懒加载，避免 import 即下模型）。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print("  [embedder] 加载 BAAI/bge-m3…")
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    return _model


def _mock_vec_dir(base: Path) -> Path:
    """向量缓存目录；**演示模式单独一份**（`*__mock/`）。

    为什么必须分开：演示模式用"词袋哈希"假向量，真模式用 bge-m3。
    两者共用缓存目录会**静默互相污染**（假向量被真查询复用 → 检索结果毫无意义，且不报错）。
    """
    if mock_llm.embed_enabled():
        return base.parent / f"{base.name}__mock"
    return base


def encode_texts(texts: list[str]) -> np.ndarray:
    """批量 encode（归一化，shape (n, 1024)）。

    `PAPERPILOT_MOCK_EMBED=1`（演示模式）→ 用内置"词袋哈希"向量，
    **不加载 bge-m3**（省 2 GB 下载），保证无模型也能跑通检索链路。
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    if mock_llm.embed_enabled():
        return mock_llm.encode_texts(list(texts), EMBED_DIM)
    model = _get_model()
    vecs = model.encode(list(texts), normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def encode_query(query: str) -> np.ndarray:
    """query 加 bge 官方指令前缀后 encode。"""
    if mock_llm.embed_enabled():
        return mock_llm.hash_vector(query, EMBED_DIM)
    return encode_texts([QUERY_PREFIX + query])[0]


class ClaimIndex:
    """单篇论文的主张向量索引。"""

    def __init__(self, pdf: str):
        self.pdf = pdf
        self.stem = Path(pdf).stem
        self._report: dict[str, Any] | None = None
        vec_dir = _mock_vec_dir(VIEW_DIR)
        self._vec_file = vec_dir / f"{self.stem}.gvec.npy"
        self._gid_file = vec_dir / f"{self.stem}.gidx.json"

    # ── report 加载 ──────────────────────────────────────
    @property
    def report(self) -> dict[str, Any]:
        if self._report is None:
            path = VIEW_DIR / f"{self.stem}.report.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"缺 {path.name}——请先跑 `uv run python cli/main.py {self.pdf}` 生成报告")
            self._report = json.loads(path.read_text(encoding="utf-8"))
        rep = self._report
        assert rep is not None  # 上面 if 分支已赋值
        return rep

    def _groups(self) -> list[dict[str, Any]]:
        groups = self.report.get("groups") or []
        return groups[:MAX_DOCS_PER_PDF]

    # ── 向量构建/缓存 ────────────────────────────────────
    def _group_ids(self) -> list[str]:
        return [g["group_id"] for g in self._groups()]

    def vectors(self) -> np.ndarray:
        """返回 (n, 1024) 向量；缓存失效时重建。"""
        if self._vec_file.exists() and self._gid_file.exists():
            old_ids = json.loads(self._gid_file.read_text(encoding="utf-8"))
            if old_ids == self._group_ids():
                return np.load(self._vec_file)
        texts = [g.get("rep_text", "") for g in self._groups()]
        print(f"  [embedder] 构建索引：{len(texts)} 条主张（encode…）")
        vecs = encode_texts(texts)
        # 按**实际目标目录**建目录（演示模式写 `out_views__mock/`，首次不存在）
        self._vec_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._vec_file, vecs)
        self._gid_file.write_text(json.dumps(self._group_ids(), ensure_ascii=False),
                                  encoding="utf-8")
        return vecs

    # ── 检索 ─────────────────────────────────────────────
    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """返回 topK 命中（含 group 字段 + score）。"""
        q = encode_query(query)
        vecs = self.vectors()
        if len(vecs) == 0:
            return []
        scores = (q @ vecs.T).astype("float64")  # 已归一化，点积=cosine
        order = np.argsort(-scores)[: min(top_k, len(scores))]
        groups = self._groups()
        hits = []
        for i in order:
            g = groups[int(i)]
            hits.append({**g, "score": round(float(scores[int(i)]), 4)})
        return hits


class ChunkIndex:
    """单篇论文的正文 chunk 向量索引（L3 Global Chunk 检索）。

    L2/L3 的正文检索单元是 parse_pdf → chunk_document → extractable 的 chunk
    （复用 document_cache 的 ordered_chunks，避免重复 parse）。
    向量缓存到 assets/artifacts/out_views/<stem>.cvec.npy（+ <stem>.cidx.json 记 chunk_id 顺序），
    首次 encode 一篇约 3~10s，之后秒级复用。

    缓存失效条件：chunk_id 列表与索引时不一致 → 重建。
    与 ClaimIndex 的关系：ClaimIndex 检索主张（L1），ChunkIndex 检索正文（L3），
    二者独立索引、独立缓存，模型单例共享。
    """
    def __init__(self, pdf: str):
        self.pdf = pdf
        self.stem = Path(pdf).stem
        self._chunks: list[Any] | None = None
        vec_dir = _mock_vec_dir(CHUNK_VIEW_DIR)
        self._vec_file = vec_dir / f"{self.stem}.cvec.npy"
        self._cid_file = vec_dir / f"{self.stem}.cidx.json"

    # document_cache 延迟 import：Chunk 模型只在 L3 需要，避免 L0 热路径背负解析模块
    def _doc_chunks(self) -> list[Any]:
        if self._chunks is None:
            from paperpilot.agents.document_cache import retrieval_chunks
            # 检索视图：chunk_id 同 pymupdf 空间，文本额外含 MinerU 表格/公式
            self._chunks = retrieval_chunks(self.pdf)
        return self._chunks

    def _fingerprint(self) -> dict[str, Any]:
        """缓存判据：chunk_id 顺序 + **文本指纹**。

        只比 chunk_id 不够：检索视图会在**同一批 id** 上把 MinerU 表格/公式注入文本，
        文本变了而 id 没变 → 旧向量会被静默复用（错且不可见）。故加文本指纹。
        """
        import hashlib
        chunks = self._doc_chunks()
        h = hashlib.md5()
        for c in chunks:
            h.update(c.chunk_id.encode("utf-8", "ignore"))
            h.update(b"\x00")
            # 指纹必须含**实际参与 encode 的文本**（`embed_text` 优先）：
            # P2 双写下向量侧喂的是摘要，若只 hash `text`，改摘要不会让缓存失效 → 静默复用旧向量。
            h.update((c.embed_text or c.text).encode("utf-8", "ignore"))
            h.update(b"\x00")
        out = {"ids": [c.chunk_id for c in chunks], "fp": h.hexdigest()}
        if mock_llm.embed_enabled():
            # 只在自己这侧加标记：真模式的指纹保持**逐字不变**（不触发无意义重建）
            out["enc"] = "mock"
        return out

    def vectors(self) -> np.ndarray:
        """返回 (n, 1024) chunk 向量；缓存失效时重建。"""
        if self._vec_file.exists() and self._cid_file.exists():
            try:
                old = json.loads(self._cid_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 缓存损坏 → 重建
                old = None
            # 旧格式是纯 id 列表 → 视为失效（一次性迁移重建）
            if isinstance(old, dict) and old == self._fingerprint():
                return np.load(self._vec_file)
        # **P2 双写**：向量侧优先用 `embed_text`（表块的一行语义摘要），
        # BM25/作答仍读 `c.text`（原表）——两路文本解耦，见 `Chunk.embed_text`。
        texts = [c.embed_text or c.text for c in self._doc_chunks()]
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype="float32")
        print(f"  [embedder] 构建 ChunkIndex：{len(texts)} 个 chunk（encode…）")
        vecs = encode_texts(texts)
        # 按**实际目标目录**建目录（演示模式写 `out_views__mock/`，首次不存在）
        self._vec_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._vec_file, vecs)
        self._cid_file.write_text(json.dumps(self._fingerprint(), ensure_ascii=False),
                                  encoding="utf-8")
        return vecs

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """全文 chunk 语义检索（L3 独立保底），返回命中（含 chunk 字段 + score）。"""
        q = encode_query(query)
        vecs = self.vectors()
        if len(vecs) == 0:
            return []
        scores = (q @ vecs.T).astype("float64")
        order = np.argsort(-scores)[: min(top_k, len(scores))]
        chunks = self._doc_chunks()
        hits = []
        for i in order:
            c = chunks[int(i)]
            hits.append({
                "chunk_id": c.chunk_id,
                "title_path": list(c.title_path),
                "page": c.page_span[0],
                "text": c.text,
                "score": round(float(scores[int(i)]), 4),
            })
        return hits

    @staticmethod
    def _section_of(c: Any) -> str:
        """chunk 顶层节名（与 document_cache.top_section 同口径，内联避免环依赖）。"""
        tp = list(getattr(c, "title_path", None) or [])
        if not tp:
            return ""
        head = str(tp[0])
        return head.split(" · ", 1)[1].strip() if " · " in head else head.strip()

    def _cap_select(self, chunks: list[Any], order: list[int], top_k: int) -> list[int]:
        """节级配额去重（保序）：每顶层节最多 cap 块；cap<=0 → 直接取前 top_k。

        背景（2026-09-09 离线扫描）：top12 覆盖全篇 78%，但 Introduction×4 / Experiments×3
        等"同节重复块"挤占配额，把 Methods/Experiments 深处的 gold 挤到 2-5 名。
        cap=1 离线 Recall/MRR 全胜（R@8 0.897→0.945, MRR 0.490→0.548）。env 默认关。
        """
        cap = int(os.environ.get("PAPERPILOT_RETRIEVE_SECTION_CAP", "0") or 0)
        if cap <= 0:
            return order[:top_k]
        out: list[int] = []
        used: dict[str, int] = {}
        for idx in order:
            sec = self._section_of(chunks[int(idx)])
            if used.get(sec, 0) >= cap:
                continue
            out.append(int(idx))
            used[sec] = used.get(sec, 0) + 1
            if len(out) >= top_k:
                break
        return out

    def _select(self, chunks: list[Any], order: list[int], top_k: int) -> list[int]:
        """取最终候选：**默认（quota=2）加表池名额**；`PAPERPILOT_EXT_QUOTA=0` 退回原 `_cap_select`。

        见 `_ext_quota()`：正数 = 并集（正文满额 + 追加表池名额），负数 = 替换（正文让槽）。
        """
        q = _ext_quota()
        if q == 0:
            return self._cap_select(chunks, order, top_k)
        is_ext = _ext_mask(chunks)
        # 外部块共用同一伪节名 ("(External Tables)")，走 _cap_select 会被节配额砍到 1 个
        # → 表池部分直接按表池内 RRF 序取，不套节配额。
        ext_order = [int(i) for i in order if is_ext[int(i)]]
        if q > 0:
            # 并集：**基线候选原样保留**，再"追加"表池前 q 名里基线没有的。
            # ⚠️ 首版写成「正文池 top_k + 表池 top-q」→ 把"原本混在混池 top_k 里的表块"
            #    挤掉了（实测 20 题里有 2 题目标表因此掉出候选）。必须做**加法**而非替换。
            base = [int(i) for i in self._cap_select(chunks, order, top_k)]
            out = list(base)
            for i in ext_order[:q]:
                if i not in out:
                    out.append(i)
            return out
        m = min(-q, max(top_k - 1, 0))
        text_order = [int(i) for i in order if not is_ext[int(i)]]
        return self._cap_select(chunks, text_order, top_k - m) + ext_order[:m]

    def search_multi_hybrid(self, queries: list[str], top_k: int = 8) -> list[dict[str, Any]]:
        """多 query × 向量+BM25 混合，全部按 RRF 融合成一份 top_k（查询改写主用）。

        对每个查询同时累积"向量位次"与"BM25 位次"的 RRF 分；最后按 RRF 取 top_k。
        命中结构与 search_hybrid 一致（score 存平均向量 cosine）。
        env PAPERPILOT_RETRIEVE_SECTION_CAP=N>0 → 保序节级配额去重。
        """
        chunks = self._doc_chunks()
        vecs = self.vectors()
        n = len(chunks)
        if n == 0:
            return []
        texts = [c.text for c in chunks]
        rrf = np.zeros(n, dtype="float64")
        avg = np.zeros(n, dtype="float64")
        bm_idx = BM25Index(texts)
        wv, wb = _ext_weights(chunks, float(os.environ.get("PAPERPILOT_EXT_RRF_ALPHA", "0.5") or 0.5))
        used = 0
        for q in queries:
            if not q:
                continue
            used += 1
            qv = encode_query(q)
            v = (vecs @ qv).astype("float64")
            b = np.asarray(bm_idx.score(q), dtype="float64")
            for r, i in enumerate(np.argsort(-v)):
                rrf[int(i)] += (1.0 if wv is None else float(wv[int(i)])) / (60 + r + 1)
                avg[int(i)] += float(v[int(i)])
            for r, i in enumerate(np.argsort(-b)):
                rrf[int(i)] += (1.0 if wb is None else float(wb[int(i)])) / (60 + r + 1)
        order = self._select(chunks, list(np.argsort(-rrf)), top_k)
        hits = []
        for i in order:
            c = chunks[int(i)]
            hits.append({
                "chunk_id": c.chunk_id,
                "title_path": list(c.title_path),
                "page": c.page_span[0],
                "text": c.text,
                "score": round(float(avg[int(i)]) / max(used, 1), 4),
            })
        return hits

    def search_hybrid(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """向量 + BM25 RRF 融合检索（专名/术语精确匹配互补）。

        命中结果结构与 search 一致；score 存向量 cosine（便于沿用现有阈值/排序逻辑），
        bm_rank 字段额外记录 bm25 贡献名次，供分析。
        """
        chunks = self._doc_chunks()
        vecs = self.vectors()
        n = len(chunks)
        if n == 0:
            return []
        q = encode_query(query)
        vec_scores = (q @ vecs.T).astype("float64")
        bm = BM25Index([c.text for c in chunks])
        bm_scores = bm.score(query)
        wv, wb = _ext_weights(chunks, float(os.environ.get("PAPERPILOT_EXT_RRF_ALPHA", "0.5") or 0.5))
        order = self._select(chunks, list(rrf_order(vec_scores, bm_scores,
                                                    w_vec=wv, w_bm=wb)), top_k)
        hits = []
        for i in order:
            c = chunks[int(i)]
            hits.append({
                "chunk_id": c.chunk_id,
                "title_path": list(c.title_path),
                "page": c.page_span[0],
                "text": c.text,
                "score": round(float(vec_scores[int(i)]), 4),
                "bm_rank": int(np.sum(bm_scores > bm_scores[int(i)])) + 1,
            })
        return hits

    def search_multi(self, queries: list[str], top_k: int = 8) -> list[dict[str, Any]]:
        """多 query 分别向量检索后 RRF 融合（query 改写方案2）。

        queries[0] 通常为原问题。每 query 向量打分一次，RRF 合并 top_k。
        命中带 score(原 query cosine) 与 src_query（命中来自哪个 query 的 top 位次信息）。
        """
        chunks = self._doc_chunks()
        vecs = self.vectors()
        n = len(chunks)
        if n == 0:
            return []
        all_vec = np.zeros(n, dtype="float64")
        rrf = np.zeros(n, dtype="float64")
        for q in queries:
            if not q:
                continue
            qv = encode_query(q)
            scores = (vecs @ qv).astype("float64")
            order = np.argsort(-scores)
            for r, i in enumerate(order):
                rrf[int(i)] += 1.0 / (60 + r + 1)
                all_vec[int(i)] += float(scores[int(i)])
        order = np.argsort(-rrf)[: min(top_k, n)]
        hits = []
        for i in order:
            c = chunks[int(i)]
            hits.append({
                "chunk_id": c.chunk_id,
                "title_path": list(c.title_path),
                "page": c.page_span[0],
                "text": c.text,
                "score": round(float(all_vec[int(i)]) / max(len(queries), 1), 4),
            })
        return hits
