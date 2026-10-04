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


def _bm_or_none(bm_scores: np.ndarray | None) -> np.ndarray | None:
    """BM25 分数**无有效信号**时返回 None（即：只用向量路）。

    ⚠️ 2026-09-21 实测（原脚本 `retrieval/scripts/_bm_check.py` **已不在仓库**，故此处
       把结论数据一并落在注释里，避免"引用不可复核"）：5 篇 / 146 chunk
        `BM25Index` 的分词是「英文词 + **单个汉字**」，而语料是英文论文
        → **纯中文查询的 token 全部无匹配 → `score()` 全 0**。
        此时若照常融合，`np.argsort(-bm_scores)` 对全 0 数组返回**索引序**
        → 每个块按"它在拼接语料里的先后"拿到 `1/(60+i)` 的**伪分**，
        **拼接序第一位的那篇被系统性加分**（多篇下直接表现为"第一篇霸占"）。

    实测数据（阈值 = 是否有非零分）：
        "实验结果如何" / "训练数据是怎么构造的" / … → max **0.000**、非零 0/146 ❌
        "CrossSum 1500 多种语言对…" → max 0.911、非零 44/146 ✅
        "content plan 内容规划…"   → max 3.212、非零 28/146 ✅

    ⚠️ **2026-09-26 修正（原写"产品是中文提问 → 必现"，这句过强）**：
        按本判据逐题实测 120 道中文题（`retrieval/tmp/_lang_effect.py`）——
        **104/120（87%）的题 BM25 有非零分**，只有 16 道是"全 0"。
        原因：中文问句普遍**照抄英文实体**（`MAMuJoCo` / `VAE` / `w/o Expand` /
        `Table 1` / `token`…），这些 token 在英文正文里有匹配。
        所以准确表述是：**纯中文 token 的查询才会全 0**（占 13%），不是"中文提问必现"。

    闸门的**价值**（Step 1 消融，`retrieval/tmp/_sandbox.py --ablate`，120 题）：
        伪装信号放行（= 模拟"没有闸门"）→ 未命中 11 → **12**，且**只落在那 16 道 dead 题上**
        （live 层 5→5、dead 层 6→7）——与上面"索引序伪位次"的机制**完全吻合**。
        即：闸门的价值 = **1 道**（在 N=24 下）。仍必须留（真 bug），但别高估它。
        → 附注：正因为闸门只在 dead 层生效，**"调 BM25 参数"（k1/b/权重）只影响 live 层**，
          而 live 层才是主力（104/120）→ 见 `_sandbox.py --ablate` 的 A1a/A6a 对照。
    """
    if bm_scores is None or bm_scores.size == 0:
        return None
    return bm_scores if float(np.max(bm_scores)) > 0 else None


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
    bm_scores = _bm_or_none(bm_scores)      # 无 BM25 信号 → 只用向量（防"索引序伪位次"）
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

# 批量 encode 的 batch_size（**不要调回 32**，2026-09-20 实测）。
#
# `model.encode()` 默认 32：chunk 最长可达 **2806 token**（单段超 4000 字符的兜底块，
# 见 chunker._split_by_paragraph），32 × 长序列的激活值会把 6 GB 显存**吃穿**
# （实测峰值 6276 / 6141 MiB）→ Windows 驱动回退到共享内存（PCIe 换页）
# → **单篇首次建索引 6.7s 变 237.9s（35 倍）**。
#
# 降到 8 后：峰值显存 6276 → 3200 MiB，耗时 237.9 → 6.7s。
#
# ⚠️ 这**不改变向量**：与 32 的最大绝对差 3.2e-07（fp32 舍入噪声级），
#    平均余弦 1.00000000 → 900+ 个已建 `.cvec.npy` 缓存**继续有效**。
# ⚠️ 也**不要改 `max_seq_length` 来省显存**（现为默认 8192）：那是**截断**，
#    实测会把 19% 的块砍掉（最长那块 2806 → 512 token，砍 82%），
#    正是 `nodes/judge.py` 与 `nodes/pull_chunk.py` 记录的
#    **"Run1 真漏检根因"**（900/3000 截断均已移除，2026-09 结论）。
#    8192 是**必需的**：QASPER 极端块 15981 字符 ≈ 4200 token，低于此值就会截断。
ENCODE_BATCH = 8

# ★ 每批的 **token 预算**（2026-09-30 实测）。`padding=True` 会补到批内最长 → 驱动激活显存的
#   真正指标是 **`batch_size × 批内最长 token`**。用真 tokenizer 在**本语料 3,284 块**上实测：
#
#   | 分批策略 | 总 token | **单批 token 峰值** |
#   |---|---|---|
#   | 乱序（原行为） | 3,838,068 | **60,864** ← 打满 6 GB → 驱动换页 |
#   | 仅按长度排序 | 2,587,840 | **30,432** ← 仍偏高（最长的几个块被排到一起） |
#   | **排序 + 本预算** | 2,485,939 | **8,925** ← 降 6.8× |
#
#   ⚠️ 别用"批内最长 token"当指标 —— 两种排序下它都是 7,608（最长的块总在某个批里），
#      必须看 **batch_size × 批内最长**。
ENCODE_TOKEN_BUDGET = 8192

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

    ## ★ 按长度排序再批（2026-09-30）
    `SentenceTransformer.encode` 会**补到批内最长**，而我们的块长度跨度很大
    （语料实测：中位 1,794 字符、p90 3,751、最长可达 2,806 token）→ **乱序时每个批
    都被同批最长的块拖满**。改成"**先按长度排序 → encode → 还原原序**"：

    | | 峰值显存 | 单簇 1,079 块耗时 |
    |---|---|---|
    | 乱序（原） | 5,634 / 6,144 MiB（**仅剩 287** → 驱动换页） | >1,000s 且不收敛 |
    | 排序（现） | 见 `ENCODE_BATCH` 注释的一致量级 | 见 verify 输出 |

    ⚠️ **不截断**（挨着 `max_seq_length=8192` 那条"不许截断"的结论）、**不改精度**
    （不引入 fp16，故不失效已有 `.cvec.npy` 缓存）。批组成变化带来的数值差异与
    当年 `32→8` 同量级（fp32 舍入噪声 ≈3e-7），`_fingerprint()` 只看文本 → 缓存继续有效。
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    if mock_llm.embed_enabled():
        return mock_llm.encode_texts(list(texts), EMBED_DIM)
    model = _get_model()
    ts = list(texts)
    if len(ts) <= ENCODE_BATCH:                    # 小集合（如 `encode_query` 的 1 条）直接走
        return np.asarray(model.encode(ts, normalize_embeddings=True,
                                       batch_size=ENCODE_BATCH), dtype="float32")
    # ⚠️ 必须显式给 batch_size：默认 32 会让单篇首次建索引慢 35 倍（见 ENCODE_BATCH 注释）
    order = sorted(range(len(ts)), key=lambda i: len(ts[i]))      # 批内长度相近 → 少 padding
    lens = _token_lens(model, [ts[i] for i in order])             # 真 tokenize（约 1s/3000 块）
    out = np.empty((len(ts), EMBED_DIM), dtype="float32")
    for b in _plan_batches(lens, ENCODE_BATCH, ENCODE_TOKEN_BUDGET):
        sub = [order[i] for i in b]
        v = model.encode([ts[i] for i in sub], normalize_embeddings=True,
                         batch_size=len(sub))
        out[sub] = np.asarray(v, dtype="float32")
    return out


def _token_lens(model: Any, texts: list[str]) -> list[int]:
    """真实 token 长度（截断到 `max_seq_length`，与实际编码口径一致）。"""
    try:
        tk = model.tokenizer
        mx = int(getattr(model, "max_seq_length", 8192) or 8192)
        out: list[int] = []
        for i in range(0, len(texts), 256):
            enc = tk(texts[i:i + 256], truncation=True, max_length=mx,
                     add_special_tokens=True)
            out += [len(x) for x in enc["input_ids"]]
        return out
    except Exception:  # noqa: BLE001（tokenizer 不可用 → 按字符估，3.5 字符/token 实测均值）
        return [max(1, int(len(t) / 3.5)) for t in texts]


def _plan_batches(lens: list[int], max_bs: int, budget: int) -> list[list[int]]:
    """在**已按长度升序**的前提下切批：每批 ≤`max_bs` 条，且 `条数 × 批内最长` ≤`budget`。

    → 既保住"批内长度相近"，又给**超长块**单独设上限（否则它会把整批拉满）。
    """
    out: list[list[int]] = []
    cur: list[int] = []
    mx = 0
    for i, n in enumerate(lens):
        m2 = max(mx, n)
        if cur and (len(cur) >= max_bs or m2 * (len(cur) + 1) > budget):
            out.append(cur)
            cur, mx = [i], n
        else:
            cur.append(i)
            mx = m2
    if cur:
        out.append(cur)
    return out


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
        """**检索单元**：`retrieval_chunks`（语义切块，chunk_id 同 pymupdf 空间）。"""
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
                **self._hit_extra(int(i)),
            })
        return hits

    def _hit_extra(self, i: int) -> dict[str, Any]:
        """命中结果的**附加字段钩子**。

        · chunk 视图（默认）：返回空 dict → 命中结构与引入前**逐字节一致**（评测口径不变）。
        · 窗口视图：额外带 `chunk_ids`（本窗口覆盖的原 chunk）→ cites / 页码可映射回原文。
          多篇语料库再由 `MultiChunkIndex` 覆写补上来源篇。
        """
        chunks = self._doc_chunks()
        if 0 <= int(i) < len(chunks):
            cids = getattr(chunks[int(i)], "chunk_ids", None)
            if cids:
                return {"chunk_ids": list(cids)}
        return {}

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
            b = _bm_or_none(np.asarray(bm_idx.score(q), dtype="float64"))
            for r, i in enumerate(np.argsort(-v)):
                rrf[int(i)] += (1.0 if wv is None else float(wv[int(i)])) / (60 + r + 1)
                avg[int(i)] += float(v[int(i)])
            if b is not None:                # 无 BM25 信号 → 只用向量路（见 `_bm_or_none`）
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
                **self._hit_extra(int(i)),
            })
        return hits

    # ── ★ R2 定稿：多路加权 RRF（1 中文 dense + n 子查询 dense + n 子查询 BM25）──
    #    依据 `retrieval/results/R2_FINAL_SPEC_20260930.md`：7 路 RRF(C=60)，ρ=1:4，
    #    每路截断 d=(zh 100, sq-dense 50, sq-bm25 50)（三类体积比 1:1.10:1.25 已配平）。
    #    与 `search_multi_hybrid` 的差别**只有**三点：逐路截断 d、逐类权重、路数可分离。
    def search_multiroute(self, *, zh: str, subs: list[str],
                          d_zh: int = 100, d_sd: int = 50, d_sb: int = 50,
                          w_dense: float = 1.0, w_sparse: float = 4.0,
                          rrf_k: int = 60, top_k: int = 8,
                          use_select: bool = True) -> list[dict[str, Any]]:
        """**多路加权 RRF**：`zh` 走 1 路 dense，`subs` 各走 dense + BM25，按类加权。

        · 每路**先截 top-d 再计入 RRF**（`d<=0` = 不截）；RRF 名次用**该路内的全局秩**；
        · 权重：两类 dense = `w_dense`，BM25 路 = `w_sparse`（定稿 4.0 ← bm25 主导）；
        · `score` 存 **RRF 分**（定稿按此排序；其他 reader 用 cosine，这里不沿用）；
        · `use_select=False` → 不做节级配额（`_select`），返回**纯 RRF 序**（定稿口径）。
        env 覆盖：`PAPERPILOT_R2_{D_ZH,D_SD,D_SB,W_SPARSE,RRF_K}`。

        ⚠️ 中文查询的 BM25 通常全 0（`_bm_or_none` 会跳过），故 `zh` **不另开 BM25 路**
        ——与定稿一致，也避免"索引序伪位次"（见 `_bm_or_none` 的长注释）。
        """
        def _env_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, "") or default)
            except ValueError:
                return default

        def _env_f(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, "") or default)
            except ValueError:
                return default

        d_zh = _env_int("PAPERPILOT_R2_D_ZH", d_zh)
        d_sd = _env_int("PAPERPILOT_R2_D_SD", d_sd)
        d_sb = _env_int("PAPERPILOT_R2_D_SB", d_sb)
        w_sparse = _env_f("PAPERPILOT_R2_W_SPARSE", w_sparse)
        rrf_k = _env_int("PAPERPILOT_R2_RRF_K", rrf_k)

        chunks = self._doc_chunks()
        vecs = self.vectors()
        n = len(chunks)
        if n == 0:
            return []
        rrf = np.zeros(n, dtype="float64")
        # `q_sim` = 该块在**全部查询上的最大 cosine** —— 供 C3"按语义选块"构造证据窗口用
        # （定稿窗口 = 该篇内 sim 最大的若干块拼到 ≤4,200 字符；生产侧没有 facet 正则，
        #  故窗口只用语义选块这一支）。见 `R2_FINAL_SPEC_20260930.md` / `set_judge.build_user`。
        q_sim = np.full(n, -np.inf, dtype="float64")
        bm_idx = BM25Index([c.text for c in chunks])

        def _add(scores: np.ndarray, d: int, w: float) -> None:
            if scores is None:
                return
            order = np.argsort(-scores)
            lim = n if d <= 0 else min(int(d), n)
            for r, i in enumerate(order[:lim]):        # 名次 = 该路内全局秩 r（与定稿一致）
                rrf[int(i)] += w / (rrf_k + r + 1)

        if zh:
            v = (vecs @ encode_query(zh)).astype("float64")
            q_sim = np.maximum(q_sim, v)
            _add(v, d_zh, w_dense)
        for s in subs:
            if not s:
                continue
            v = (vecs @ encode_query(s)).astype("float64")
            q_sim = np.maximum(q_sim, v)
            _add(v, d_sd, w_dense)
            _add(_bm_or_none(np.asarray(bm_idx.score(s), dtype="float64")), d_sb, w_sparse)

        order = list(np.argsort(-rrf))
        sel = self._select(chunks, order, top_k) if use_select else order[: min(top_k, n)]
        hits = []
        for i in sel:
            c = chunks[int(i)]
            hits.append({
                "chunk_id": c.chunk_id,
                "title_path": list(c.title_path),
                "page": c.page_span[0],
                "text": c.text,
                "score": round(float(rrf[int(i)]), 6),
                "q_sim": (None if not np.isfinite(q_sim[int(i)])
                          else round(float(q_sim[int(i)]), 4)),
                **self._hit_extra(int(i)),
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
        bm_sig = _bm_or_none(bm_scores)      # 纯中文查询 → 全 0，跳过 BM25 路（见 `_bm_or_none`）
        wv, wb = _ext_weights(chunks, float(os.environ.get("PAPERPILOT_EXT_RRF_ALPHA", "0.5") or 0.5))
        order = self._select(chunks, list(rrf_order(vec_scores, bm_sig,
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
                "bm_rank": (0 if bm_sig is None         # 0 = 本次**无** BM25 信号（别当"第 1 名"读）
                            else int(np.sum(bm_scores > bm_scores[int(i)])) + 1),
                **self._hit_extra(int(i)),
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
                **self._hit_extra(int(i)),
            })
        return hits


class MultiChunkIndex(ChunkIndex):
    """多篇论文合并成的**检索语料库**（N 篇 → 一个扁平 chunk 空间）。

    ## 设计要点（2026-09-21）

    · **不新增缓存产物**：每篇仍由 `ChunkIndex` 建自己的 `<stem>.cvec.npy`，
      本类**只在读取时拼接** —— 所以 900+ 个已有 `.cvec.npy` 全部继续有效，
      而且"单篇"与"多篇"共用同一批索引文件（单篇路径零影响、零迁移）。
    · **只重写 `_doc_chunks()` 与 `vectors()` 两处**，其余（`search_hybrid` /
      `search_multi_hybrid` / BM25 / RRF / `_select` / `_ext_weights`）全部继承
      —— 它们只看见一个扁平 chunk 列表，与篇数无关。
    · 命中额外带 `pdf` 字段（跨篇引用 / 前端跳转用），由 `_hit_extra` 注入。

    ## 已知待观察项（跑数据后再决定，别预先优化）

    · `_section_of` 返回顶层节名（"Introduction"/"Experiments"…），**跨篇会撞名**
      → 节级配额 `PAPERPILOT_RETRIEVE_SECTION_CAP`（默认 0=关）在多篇下语义会变。
    · 可能出现**单篇霸占 top_k**（某篇特别长/特别贴题），此时需要"篇级配额"。
    · 语料变大后 `top_k` 需重调（单篇 top12 覆盖 ~85% 的结论不自动迁移）。
    """

    def __init__(self, pdfs: list[str]):
        self.pdfs = [str(p) for p in pdfs if p]
        # 兼容父类属性（日志/调试用，**不参与检索、不作为缓存键**）
        self.pdf = "+".join(Path(p).stem for p in self.pdfs)
        self.stem = self.pdf
        self._chunks: list[Any] | None = None
        self._vecs: np.ndarray | None = None
        self._owner: list[str] = []      # 扁平下标 → 来源 pdf（与 _chunks 同序）

    def _doc_chunks(self) -> list[Any]:
        """各篇 `retrieval_chunks` 按 pdfs 顺序拼成一个扁平 chunk 空间。"""
        if self._chunks is None:
            from paperpilot.agents.document_cache import retrieval_chunks
            chunks: list[Any] = []
            owner: list[str] = []
            for p in self.pdfs:
                cs = retrieval_chunks(p)
                chunks.extend(cs)
                owner.extend([p] * len(cs))
            self._chunks, self._owner = chunks, owner
        return self._chunks

    def vectors(self) -> np.ndarray:
        """各篇**已缓存**向量按同一顺序 `vstack` —— 不落新缓存、不重新 encode。"""
        if self._vecs is None:
            parts = [ChunkIndex(p).vectors() for p in self.pdfs]
            parts = [v for v in parts if len(v)]
            self._vecs = (np.vstack(parts).astype("float32") if parts
                          else np.zeros((0, EMBED_DIM), dtype="float32"))
        return self._vecs

    def _fingerprint(self) -> dict[str, Any]:
        """多篇**不落缓存**（向量来自各篇自己的缓存），故不需要指纹。"""
        return {}

    def _hit_extra(self, i: int) -> dict[str, Any]:
        return {**super()._hit_extra(int(i)),
                "pdf": self._owner[int(i)] if 0 <= int(i) < len(self._owner) else ""}

    # ── 分层检索：篇内先检索 → 跨篇融合 ────────────────────────────────

    def search_layered(self, query: str, *, per_paper_k: int = 4, top_k: int = 12,
                       mode: str = "quota", floor: int = 1) -> list[dict[str, Any]]:
        """多篇检索主路径：**每篇内先检索（同粒度可比）→ 跨篇融合**。

        ⚠️ **2026-09-25 重测后改写的动机说明**（此前引用的"谁切得细"是 2026-09-21 的
        **历史结论，已不适用** —— 见下"历史"一节，别再拿它当现状）：
            粒度**已被抹平**：本套语料 group1 中位块长 886~1853（2.09×）、块数 20~28；
            group2 1046~1780（1.70×）、块数 20~24。（2026-09-22 那次"与
            `chunker.chunk_document` 对齐"的分块改动生效了。）
            **但全局混池仍系统性漏篇**（N=24，两组各 60 题）：
                global/hybrid  平均命中 3.60 / 4.18 篇 ｜ 覆盖满 5 篇仅 15/60、23/60
                本方法(quota)  平均命中 **4.90 / 4.93** 篇 ｜ 覆盖满 5 篇 **54/60、56/60**
            原因现在是**两条叠加**：
              ① **粒度残留**（2× 未归零）：实测两组里**块最粗的那篇恰好被漏最多**
                 （29290 中位 1853 → 漏 23/60；25389 中位 1780 → 漏 25/60）；
              ② **同主题分数集中**：5 篇是姊妹论文，cos 全挤在 0.48~0.55（只差 0.07）
                 → 全局 top-k 会被"字面更贴近查询"的那篇成片占掉。这一条与块长无关。
            所以本方法的作用应理解为**「保证每篇可见」（补漏）**，不是"修正粒度偏差"：
            它很轻 —— 实测平均只注入 1.07~1.52 块/题，12~17/60 题**完全不注入**。

        历史（2026-09-21，**仅存档，勿引用**）：当时 5 篇 chunk 数 22~52、中位块长差 5.8×
        → 全局 cosine 实际在比"谁切得细"（泛化查询 top12 被单篇 100% 霸占、top48 仍占 71%，
        把 top_k 从 12 加到 48 其他篇覆盖只从 0 涨到 1~7）。粒度问题后来已修，该结论作废。

        mode:
            "quota"  —— **保底 + 全局补足**：各篇 top-`floor` 保底进候选，
                        其余名额按**全局分数**排。**默认（实测最优，见下）。**
            "rrf"    —— 只比**篇内名次**（1/(60+rank)）。篇内第 1 名各篇等权
                        → 粒度造成的"整篇分数偏移"天然抵消，但**过度均衡**
                        （实测目标命中从 7.2 掉到 2.8，一篇只剩 3 块）。
            "znorm"  —— 篇内全量分数 z 标准化。（⚠️ 实测**最差**且引入新偏差：
                        它偏好"篇内方差大"的篇 —— 恰好是块长跨度大的那篇，
                        于是查询里写着专名时首名反而给错。）
            "zmean"  —— 篇内分数减去该篇均值（去篇级偏移、保留 cosine 单位）。
            "global" —— 全局索引（对照基线，等价 `self.search_hybrid`）。

        5 篇实测（指向性 5 题 + 泛化 5 题，per_paper_k=4 / top_k=12）:
            mode     目标命中均   首名正确   top1覆盖   平均篇数
            global      7.2       3/5      2.2/5     2.2   ← 排序好，泛化查询 100% 单篇
            rrf         2.8       2/5      5.0/5     5.0   ← 跨篇了，排序被稀释
            znorm       3.4       1/5      2.8/5     4.4   ← 两头不讨好
            zmean       5.0       2/5      2.7/5     3.9
            → 所以才要 "quota"：**保底**保证跨篇可见，**全局补足**保住排序。

        Returns: 命中列表，每条额外带 `pdf` / `rank_in_paper` / `fuse` / `fuse_score`。
        """
        if mode == "global":
            return self.search_hybrid(query, top_k=top_k)
        if mode == "quota":
            return self._search_quota(query, floor=floor, top_k=top_k)
        if mode not in ("rrf", "znorm", "zmean"):
            raise ValueError(f"未知 mode: {mode}")

        per: list[tuple[str, list[dict[str, Any]]]] = []
        for p in self.pdfs:
            idx = ChunkIndex(p)
            n = len(idx._doc_chunks())
            if n == 0:
                continue
            # znorm/zmean 需要该篇**全量**分数分布才能做篇级校准；rrf 只要前 k'
            k = n if mode in ("znorm", "zmean") else min(per_paper_k, n)
            per.append((p, idx.search_hybrid(query, top_k=k)))

        return self._fuse(per, mode)[:top_k]

    @staticmethod
    def _fuse(per: list[tuple[str, list[dict[str, Any]]]],
              mode: str) -> list[dict[str, Any]]:
        """篇内命中 → 跨篇融合排序。"""
        from statistics import mean, pstdev

        rows: list[tuple[float, float, str, int, dict[str, Any]]] = []
        for p, hits in per:
            ss = [float(h.get("score") or 0.0) for h in hits]
            mu = mean(ss) if ss else 0.0
            sd = (pstdev(ss) if len(ss) > 1 else 0.0)
            for rank, h in enumerate(hits, 1):
                s = float(h.get("score") or 0.0)
                if mode == "rrf":
                    fs = 1.0 / (60 + rank)
                elif mode == "znorm":
                    fs = ((s - mu) / sd) if sd > 1e-9 else 0.0
                else:                                   # zmean
                    fs = s - mu
                rows.append((fs, s, p, rank, h))

        # 主键融合分降序；次键**原始 cosine**（跨篇虽不可比，但同分时是合理的次级依据）
        rows.sort(key=lambda t: (-t[0], -t[1]))
        return [{**h, "pdf": p, "rank_in_paper": rank, "fuse": mode,
                 "fuse_score": round(fs, 6)} for fs, _s, p, rank, h in rows]

    def _search_quota(self, query: str, *, floor: int, top_k: int
                      ) -> list[dict[str, Any]]:
        """**保底 + 全局补足**：各篇 top-`floor` 保底进候选，其余名额按全局排序。

        为什么不是等权融合（rrf）：那会让每篇都塞满，目标篇 12 条里只剩 2.8 条。
        这里保留全局序的**头部**（排序质量），只把**尾部**若干槽位换成保底项
        （保证每篇至少可见一次）→ 目标篇仍占多数，但跨篇不会盲。
        """
        glob = self.search_hybrid(query, top_k=len(self._doc_chunks()))
        out = [dict(h) for h in glob[:top_k]]
        have = {(str(h.get("pdf") or ""), str(h["chunk_id"])) for h in out}

        slot = len(out) - 1                    # 从**尾部**往前替换，保护头部排序
        for p in self.pdfs:
            if slot < 0:
                break
            for h in ChunkIndex(p).search_hybrid(query, top_k=floor):
                key = (p, str(h["chunk_id"]))
                if key in have:
                    continue
                if slot < 0:
                    break
                out[slot] = {**h, "pdf": p, "rank_in_paper": 1,
                             "fuse": "quota",
                             "fuse_score": round(float(h.get("score") or 0.0), 6)}
                have.add(key)
                slot -= 1
        return out
