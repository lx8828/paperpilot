"""Claim embedding 索引：BAAI/bge-m3 本地向量检索（L2 Claim Retrieval）。

职责：
    - 以 report.groups 的 rep_text（中文主张）为检索单元构建向量
    - 向量缓存到 out_views/<stem>.gvec.npy（+ <stem>.gidx.json 记顺序），
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


def rrf_merge(vec_scores: np.ndarray, bm_scores: np.ndarray | None,
              top_k: int, k: int = 60) -> list[int]:
    """RRF 融合：score = Σ 1/(k + rank)。vec 必给；bm 可选（None 时只按 vec 排）。"""
    n = len(vec_scores)
    rrf = np.zeros(n, dtype="float64")
    order_v = np.argsort(-vec_scores)
    for r, i in enumerate(order_v):
        rrf[i] += 1.0 / (k + r + 1)
    if bm_scores is not None:
        order_b = np.argsort(-bm_scores)
        for r, i in enumerate(order_b):
            rrf[i] += 1.0 / (k + r + 1)
    return list(np.argsort(-rrf)[:top_k].tolist())

# agents/embedder.py → 项目根
ROOT = Path(__file__).resolve().parents[3]
VIEW_DIR = ROOT / "out_views"
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


def encode_texts(texts: list[str]) -> np.ndarray:
    """批量 encode（归一化，shape (n, 1024)）。"""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    model = _get_model()
    vecs = model.encode(list(texts), normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def encode_query(query: str) -> np.ndarray:
    """query 加 bge 官方指令前缀后 encode。"""
    return encode_texts([QUERY_PREFIX + query])[0]


class ClaimIndex:
    """单篇论文的主张向量索引。"""

    def __init__(self, pdf: str):
        self.pdf = pdf
        self.stem = Path(pdf).stem
        self._report: dict[str, Any] | None = None
        self._vec_file = VIEW_DIR / f"{self.stem}.gvec.npy"
        self._gid_file = VIEW_DIR / f"{self.stem}.gidx.json"

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
        VIEW_DIR.mkdir(parents=True, exist_ok=True)
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
    向量缓存到 out_views/<stem>.cvec.npy（+ <stem>.cidx.json 记 chunk_id 顺序），
    首次 encode 一篇约 3~10s，之后秒级复用。

    缓存失效条件：chunk_id 列表与索引时不一致 → 重建。
    与 ClaimIndex 的关系：ClaimIndex 检索主张（L1），ChunkIndex 检索正文（L3），
    二者独立索引、独立缓存，模型单例共享。
    """
    def __init__(self, pdf: str):
        self.pdf = pdf
        self.stem = Path(pdf).stem
        self._chunks: list[Any] | None = None
        self._vec_file = VIEW_DIR / f"{self.stem}.cvec.npy"
        self._cid_file = VIEW_DIR / f"{self.stem}.cidx.json"

    # document_cache 延迟 import：Chunk 模型只在 L3 需要，避免 L0 热路径背负解析模块
    def _doc_chunks(self) -> list[Any]:
        if self._chunks is None:
            from paperpilot.agents.document_cache import ordered_chunks
            self._chunks = ordered_chunks(self.pdf)
        return self._chunks

    def _chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self._doc_chunks()]

    def vectors(self) -> np.ndarray:
        """返回 (n, 1024) chunk 向量；缓存失效时重建。"""
        if self._vec_file.exists() and self._cid_file.exists():
            old = json.loads(self._cid_file.read_text(encoding="utf-8"))
            if old == self._chunk_ids():
                return np.load(self._vec_file)
        texts = [c.text for c in self._doc_chunks()]
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype="float32")
        print(f"  [embedder] 构建 ChunkIndex：{len(texts)} 个 chunk（encode…）")
        vecs = encode_texts(texts)
        VIEW_DIR.mkdir(parents=True, exist_ok=True)
        np.save(self._vec_file, vecs)
        self._cid_file.write_text(json.dumps(self._chunk_ids(), ensure_ascii=False),
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
        order = rrf_merge(vec_scores, bm_scores, min(top_k, n))
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
