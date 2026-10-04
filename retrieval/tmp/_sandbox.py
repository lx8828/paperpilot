"""Step 0：检索侧**可复算沙盒**（冻结一次 → 之后所有配置秒级复算，零模型 / 零 LLM）。

## 为什么要它

调参困境的根源不是"参数多"，而是**每次试都要重建池子**：
`_scan_quota.py` 每跑一轮都要重新 encode 查询、重建 BM25、重排全量候选（分钟级），
所以"耦合的 88 个配置"根本没法随手试。

本沙盒把**唯一昂贵的一步**（encode + 全量打分）冻结到磁盘，之后一切都在冻结数据上复算：

| 改什么 | 需要重算什么 | 耗时 |
|---|---|---|
| `top_k` / `floor` / 节级 cap / 并集配比 | 对已存的**排名**切片 | 毫秒 |
| RRF `k` / 两路权重 / 关掉 BM25 路 | 对已存的 **rank** 复算（RRF 只看名次）| 毫秒 |
| BM25 `k1`/`b` | 从**文本**重建 BM25（本脚本缓存 BM25Index）| 秒 |
| 视图 / 分块 / 模型 / 查询前缀 | **必须重建沙盒**（指纹会失效并报警）| 分钟 |

## 保真原则

**不重新实现生产逻辑**，直接调用 `embedder` 里的真函数/真方法：
  `encode_query` / `BM25Index` / `_bm_or_none` / `_ext_weights` / `rrf_order`
  / `ChunkIndex._select`（含节级 cap 与表池并集）/ `ChunkIndex._section_of`
唯一需要自己写的是 `_search_quota` 的**融合循环**（它绑在 `MultiChunkIndex` 实例上），
由 `--selftest` 与真 `search_layered` **逐块比对**来保证一致。

## 用法

    uv run python retrieval/tmp/_sandbox.py --build                 # 建（含指纹与耗时）
    uv run python retrieval/tmp/_sandbox.py --selftest              # 与真检索逐块比对（保真证明）
    uv run python retrieval/tmp/_sandbox.py --reproduce             # 只用沙盒复现生产头条数字
    uv run python retrieval/tmp/_sandbox.py --bench                 # 全网格耗时（说明"秒级"）
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))
sys.path.insert(0, str(ROOT / "retrieval" / "tmp"))
sys.path.insert(0, str(ROOT / "src"))

import _scan_quota as sq  # noqa: E402
import _xling_ab as xa  # noqa: E402
from _qa_groups import papers  # noqa: E402
from paperpilot.agents.embedder import (  # noqa: E402
    BM25Index, ChunkIndex, MultiChunkIndex, _bm_or_none, _ext_weights,
    _tokenize, encode_query, rrf_order,
)

SBX = ROOT / "retrieval" / "tmp" / "_sbx"
MODEL = "BAAI/bge-m3"

# 生产默认（与 `pull_chunk.py` 同步；沙盒只做**默认值对齐**，不替代它）
PROD = {"n": 16, "xling_k": 16, "floor": 1, "k": 60, "alpha": 0.5, "k1": 1.5, "b": 0.75,
        "xling_w": 2.0}
# 2026-09-26：`n` 24 → 16、`xling_k` → 16，且多路合并由"整块尾部追加"改为
# **加权交错**（`pull_chunk._rrf_interleave_hits`）—— 与生产默认同步。
# 2026-09-27：`xling_w` 1.0 → **2.0**（与 `pull_chunk.XLING_W` 同步）。理由：en 臂判别力
# 强于 zh 臂，等权交错系统性偏向弱臂 → 严口径「每个有 gold 的篇都有 gold 块进前 12」
# 6/71 → 13/71（`retrieval/tmp/_archive/_fuse_sweep.py`，5 组 164 题，零 LLM）。**1.1~4.0 是平台**。


# ─────────────────────────── 建（唯一昂贵的一步）───────────────────────────

def build(group: str) -> Path:
    t0 = time.time()
    d = SBX / group
    d.mkdir(parents=True, exist_ok=True)

    stems = papers(group)
    idx = MultiChunkIndex([f"{s}.pdf" for s in stems])
    chunks = idx._doc_chunks()
    vecs = np.asarray(idx.vectors(), dtype="float32")
    texts = [str(c.text) for c in chunks]
    print(f"[{group}] 语料 {len(chunks)} 块 / {vecs.shape[1]} 维（{time.time() - t0:.1f}s）")

    chunk_meta = [{
        "chunk_id": str(c.chunk_id),
        "pdf": str(getattr(c, "pdf", "") or ""),
        "thead": (list(getattr(c, "title_path", None) or [""])[0]),   # `_section_of` 的输入原文
        "tlen": len(str(c.text)),
        "ntok": len(_tokenize(str(c.text))),
    } for c in chunks]
    # `_doc_chunks` 只带（无 pdf 字段）→ 用 MultiChunkIndex._owner 还原来源篇
    for m, p in zip(chunk_meta, idx._owner):
        m["pdf"] = str(p)

    ps = sq.load_probes(group)
    qvecs, tvecs, qmeta = [], [], []
    for i, q in enumerate(ps, 1):
        raw = str(q["question"])
        qvecs.append(encode_query(raw))
        en = xa.english_queries(str(q["qid"]), raw)
        tr = str(en.get("translation") or "").strip()
        tvecs.append(encode_query(tr) if tr else np.zeros(vecs.shape[1], "float32"))
        qmeta.append({
            "qid": str(q["qid"]), "question": raw,
            "must_all": [str(x) for x in q["must"]],
            "quotes": [str(x) for x in q["quotes"]],
            "translation": tr,
        })
        if i % 20 == 0 or i == len(ps):
            print(f"[{group}] 查询 {i}/{len(ps)}（{time.time() - t0:.1f}s）", flush=True)

    np.savez_compressed(
        d / "vectors.npz",
        vecs=vecs,
        qvec=np.asarray(qvecs, dtype="float32"),
        tvec=np.asarray(tvecs, dtype="float32"),
    )
    (d / "chunks.json").write_text(json.dumps(chunk_meta, ensure_ascii=False), encoding="utf-8")
    (d / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    (d / "queries.json").write_text(json.dumps(qmeta, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({
        "group": group, "model": MODEL, "dim": int(vecs.shape[1]),
        "stems": stems, "n_chunks": len(chunks), "n_queries": len(qmeta),
        "n_with_en": sum(1 for m in qmeta if m["translation"]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_secs": round(time.time() - t0, 1),
        "fingerprint": fingerprint(chunk_meta, qmeta),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{group}] 沙盒落盘 {d}（{time.time() - t0:.1f}s）")
    return d


def fingerprint(chunk_meta: list[dict], qmeta: list[dict]) -> str:
    """语料 + 题目（题干/锚点/引文/译文）的内容指纹。

    ⚠️ 任何一项变了（改 gold 锚点、重切块、换视图）都必须 `--build` 重建；
    否则沙盒里的排名是**旧世界的**，扫描结果会静默失真。
    """
    blob = json.dumps({
        "chunks": [(m["chunk_id"], m["tlen"], m["thead"]) for m in chunk_meta],
        "queries": [(m["qid"], m["question"], m["must_all"], m["quotes"], m["translation"])
                    for m in qmeta],
        "model": MODEL,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────── 沙盒（加载后可反复复算）───────────────────────────

class _FakeIdx(ChunkIndex):
    """只为复用 `_select` / `_cap_select` 的**真实现**，不建任何索引或缓存。"""

    def __init__(self, chunks: list[Any]):
        self._chunks = chunks

    def _doc_chunks(self) -> list[Any]:
        return self._chunks


class Sandbox:
    def __init__(self, group: str, check: bool = True):
        d = SBX / group
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"沙盒不存在：{d}（先跑 --build）")
        self.group, self.dir = group, d
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.chunk_meta = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.texts = json.loads((d / "texts.json").read_text(encoding="utf-8"))
        self.queries = json.loads((d / "queries.json").read_text(encoding="utf-8"))
        with np.load(d / "vectors.npz") as z:
            self.vecs = z["vecs"]
            self.qvec = z["qvec"]
            self.tvec = z["tvec"]
        assert len(self.texts) == len(self.chunk_meta) == self.vecs.shape[0], "沙盒内部不一致"

        if check and meta["fingerprint"] != fingerprint(self.chunk_meta, self.queries):
            raise RuntimeError(
                f"沙盒指纹已失效（{group}）：语料/题目改动过 → 请 `--build` 重建，"
                "否则扫描结果基于旧世界的排名")

        # 真 `_select` 需要的假 chunk（只需 chunk_id 与 title_path 头部）
        self.chunks = [SimpleNamespace(chunk_id=m["chunk_id"],
                                       title_path=[m["thead"]] if m["thead"] else [],
                                       text=t)
                       for m, t in zip(self.chunk_meta, self.texts)]
        self.idxsel = _FakeIdx(self.chunks)
        self.N = len(self.chunks)
        self.cids = [m["chunk_id"] for m in self.chunk_meta]
        self.tlens = np.asarray([m["tlen"] for m in self.chunk_meta], dtype="int64")
        self.pdf_of = [m["pdf"] for m in self.chunk_meta]
        self.pdfs = list(dict.fromkeys(self.pdf_of))                 # 保序 = `idx.pdfs`
        self.seg = {p: np.asarray([i for i, x in enumerate(self.pdf_of) if x == p], dtype="int64")
                    for p in self.pdfs}
        self.qidx = {q["qid"]: i for i, q in enumerate(self.queries)}
        self._bm: dict = {}
        self._mask: dict = {}

    # ── 打分（零模型：查询向量已冻结）──
    def vec_scores(self, qid: str, side: str = "zh") -> np.ndarray:
        i = self.qidx[qid]
        v = self.qvec[i] if side == "zh" else self.tvec[i]
        return (v @ self.vecs.T).astype("float64")

    def _bm_index(self, seg: str | None, k1: float, b: float) -> BM25Index:
        key = (seg, round(k1, 4), round(b, 4))
        bi = self._bm.get(key)
        if bi is None:
            texts = ([self.texts[int(i)] for i in self.seg[seg]] if seg else self.texts)
            bi = BM25Index(texts, k1=k1, b=b)
            self._bm[key] = bi
        return bi

    def bm_scores(self, qid: str, side: str = "zh", *, seg: str | None = None,
                  k1: float = PROD["k1"], b: float = PROD["b"]) -> np.ndarray:
        """BM25 分（真 `BM25Index`）。`seg=None` → 全语料；否则某篇内（IDF 不同，勿混）。"""
        i = self.qidx[qid]
        q = self.queries[i]["question"] if side == "zh" else self.queries[i]["translation"]
        return np.asarray(self._bm_index(seg, k1, b).score(q), dtype="float64")

    def bm_global(self, qid: str, side: str = "zh", **kw) -> np.ndarray:
        return self.bm_scores(qid, side, seg=None, **kw)

    # ── 排名（真 `rrf_order`）──
    def order(self, qid: str, side: str = "zh", *, k: float = PROD["k"],
              use_bm: bool = True, use_vec: bool = True, fake_bm: bool = False,
              alpha: float = PROD["alpha"],
              k1: float = PROD["k1"], b: float = PROD["b"]) -> np.ndarray:
        """全量全局排名（不截断）。消融开关：

        · `use_bm=False`   → 关掉词法路（只剩向量）
        · `use_vec=False`  → 关掉向量路（只剩词法）；⚠️ 无 BM25 信号的题降级为向量序
        · `fake_bm=True`   → **模拟"没有闸门"**：把全 0 的 BM25 伪装成极小正分，
          让 `rrf_order` 内部的 `_bm_or_none`（第 127 行，**闸门其实在那儿**）放行
          → 复现"`argsort` 全 0 数组退化成索引序 → 伪位次"那个 bug。
          ⚠️ 光在调用侧"绕过"是无效的：闸门在 `rrf_order` **内部**再拦一次（踩过）。
        """
        v = self.vec_scores(qid, side)
        bsig = None
        if use_bm:
            raw = self.bm_global(qid, side, k1=k1, b=b)
            if fake_bm:
                raw = np.where(raw > 0, raw, 1e-9)
            bsig = _bm_or_none(raw)
        wv, wb = _ext_weights(self.chunks, alpha)
        if not use_vec:
            n = len(v)
            zero = np.zeros(n, dtype="float64")
            wv = zero if wv is None else np.zeros_like(wv, dtype="float64")
            if bsig is None:
                return np.asarray(rrf_order(v, None, k=k, w_vec=None, w_bm=None))
        return np.asarray(rrf_order(v, bsig, k=k, w_vec=wv, w_bm=wb))

    def paper_orders(self, qid: str, side: str = "zh", *, k: float = PROD["k"],
                     use_bm: bool = True, use_vec: bool = True, fake_bm: bool = False,
                     alpha: float = PROD["alpha"],
                     k1: float = PROD["k1"], b: float = PROD["b"]) -> dict[str, list[int]]:
        """每篇内的全量排名（映射回全局块下标）——`quota` 的保底项来源。

        ⚠️ 篇内 BM25 的 IDF 与全局**不同**（各自的分段统计）→ 不能复用全局分数。
        """
        out: dict[str, list[int]] = {}
        v_all = self.vec_scores(qid, side)
        for p in self.pdfs:
            s = self.seg[p]
            raw = self.bm_scores(qid, side, seg=p, k1=k1, b=b)
            if fake_bm:
                raw = np.where(raw > 0, raw, 1e-9)
            bsig = _bm_or_none(raw) if use_bm else None
            wv, wb = _ext_weights([self.chunks[int(i)] for i in s], alpha)
            if not use_vec:                      # 关掉向量路（消融臂）
                wv = np.zeros(len(s)) if wv is None else np.zeros_like(wv, dtype="float64")
                if bsig is None:                 # 篇内也无词法信号 → 降级为向量序
                    o = rrf_order(v_all[s], None, k=k, w_vec=None, w_bm=None)
                    out[p] = [int(s[int(i)]) for i in o]
                    continue
            o = rrf_order(v_all[s], bsig, k=k, w_vec=wv, w_bm=wb)
            out[p] = [int(s[int(i)]) for i in o]
        return out

    # ── 选择：**真实现**（节级 cap + 表池并集）──
    def select(self, order: Iterable[int], top_k: int) -> list[int]:
        return list(ChunkIndex._select(self.idxsel, self.chunks,
                                       [int(i) for i in order], int(top_k)))

    # ── 篇级融合 `quota`：逐行复刻 `MultiChunkIndex._search_quota` 的替换循环 ──
    def cap_by_paper(self, idxs: Iterable[int], c: int) -> list[int]:
        """**硬上限**：按顺序走，某篇已入选 `c` 块就跳过它；不足 `top_k` 也不放宽。

        ⚠️ 与生产里旧的 `_cap_select` **不同**：旧实现有 `eff = max(cap, ceil(n/npap))`
        的**自动放宽**（踩过：于是测的从来不是真 cap）。这里是硬上限。
        """
        if not c or c <= 0:
            return [int(i) for i in idxs]
        cnt: dict[str, int] = {}
        out: list[int] = []
        for i in idxs:
            p = self.pdf_of[int(i)]
            if cnt.get(p, 0) >= c:
                continue
            cnt[p] = cnt.get(p, 0) + 1
            out.append(int(i))
        return out

    def quota(self, glob_order: Iterable[int], per_orders: dict[str, list[int]],
              *, top_k: int = PROD["n"], floor: int = PROD["floor"],
              cap: int = 0) -> list[int]:
        gl = [int(i) for i in glob_order]
        if cap and cap > 0:
            # ① 先按**硬上限**取（某篇满 c 就跳过）→ 这直接打"单篇霸占"，
            #    但也可能**凑不满 top_k**（正是"硬"的代价，必须让指标自己说）
            out = self.cap_by_paper(gl, cap)[: min(top_k, len(gl))]
        else:
            out = gl[: min(top_k, len(gl))]
        have = set(out)
        slot = len(out) - 1                       # 尾部往前替换（保护头部排序）
        for p in self.pdfs:
            if slot < 0:
                break
            for i in per_orders[p][: max(int(floor), 0)]:
                if i in have:
                    continue
                if slot < 0:
                    break
                out[slot] = i
                have.add(i)
                slot -= 1
        return out

    # ── 判分（与 `_check_reachable --search` 同口径）──
    def masks(self, qid: str) -> tuple[np.ndarray, np.ndarray]:
        """(锚点位掩码, 引文位掩码) 逐块——`must_all` 全中才算题命中。"""
        if qid not in self._mask:
            q = self.queries[self.qidx[qid]]
            am = np.zeros(self.N, dtype="int64")
            qm = np.zeros(self.N, dtype="int64")
            for i, t in enumerate(self.texts):
                tn = " ".join(t.split())
                for j, a in enumerate(q["must_all"]):
                    if a in t:
                        am[i] |= 1 << j
                for j, qt in enumerate(q["quotes"]):
                    if qt and qt in tn:
                        qm[i] |= 1 << j
            self._mask[qid] = (am, qm)
        return self._mask[qid]

    def evaluate(self, qid: str, idxs: Iterable[int]) -> dict[str, int]:
        am, qm = self.masks(qid)
        a = q_ = 0
        sel = [int(i) for i in idxs]
        for i in sel:
            a |= int(am[i])
            q_ |= int(qm[i])
        full = (1 << len(self.queries[self.qidx[qid]]["must_all"])) - 1
        return {"miss": 0 if a == full else 1, "qhit": 1 if q_ else 0,
                "papers": len({self.pdf_of[i] for i in sel}),
                "chars": int(sum(int(self.tlens[i]) for i in sel)),
                "n": len(sel)}

    def sel_quota(self, qid: str, side: str = "zh", *, top_k: int = PROD["n"],
                  floor: int = PROD["floor"], cap: int = 0, **kw) -> list[int]:
        """`search_layered(mode='quota')` 的完整复刻（排名 → quota 融合）。"""
        return self.quota(self.order(qid, side, **kw),
                          self.paper_orders(qid, side, **kw),
                          top_k=top_k, floor=floor, cap=cap)

    def sel_xling(self, qid: str, *, top_k: int = PROD["n"],
                  xling_k: int = PROD["xling_k"], merge: str = "inter",
                  w_en: float = PROD["xling_w"], out_n: int | None = None,
                  cover: bool = True, cap_final: int = 0, **kw) -> list[int]:
        """生产路径：中文主路 + 英文检索式（默认**加权交错**，见 `pull_chunk.XLING_MERGE`）。

        ⚠️ **直接调用生产的合并函数**，不在这里重写一遍 —— 沙盒与生产的语义漂移
        （`_xling_ab` 的 A2 口径与生产不一致）就是这么来的。
        """
        zh_i = self.sel_quota(qid, "zh", top_k=top_k, **kw)
        if not self.queries[self.qidx[qid]]["translation"]:
            return self.cap_by_paper(zh_i, cap_final) if cap_final else zh_i
        en_i = self.sel_quota(qid, "en", top_k=xling_k, **kw)
        from paperpilot.agents.nodes.pull_chunk import (_quota_union_hits,
                                                       _rrf_interleave_hits)
        enc = lambda js: [{"pdf": self.pdf_of[i], "chunk_id": self.cids[i]} for i in js]  # noqa: E731
        if merge == "append":
            hits = _quota_union_hits([enc(zh_i), enc(en_i)], [top_k, xling_k])
            if out_n is not None:                     # 公平对照：同输出上限
                hits = hits[:out_n]
        else:
            hits = _rrf_interleave_hits([enc(zh_i), enc(en_i)], [1.0, w_en],
                                        top_n=(top_k if out_n is None else out_n),
                                        cover=cover)
        pos = {(self.pdf_of[i], self.cids[i]): i for i in range(self.N)}
        out = [pos[(h["pdf"], h["chunk_id"])] for h in hits]
        return self.cap_by_paper(out, cap_final) if cap_final else out


# ─────────────────────────── 自证：与真检索逐块比对 ───────────────────────────

def selftest(groups: list[str], step: int = 7) -> int:
    """① 沙盒排名/融合 vs **真** `MultiChunkIndex`（逐块 ID 比对）
       ② 指纹一致性（= load 时不报错）
    """
    bad = 0
    for group in groups:
        sb = Sandbox(group)
        idx = MultiChunkIndex([f"{s}.pdf" for s in sb.meta["stems"]])
        print(f"\n########## {group} 保真自证（每 {step} 题抽 1）")
        n = 0
        for i in range(0, len(sb.queries), step):
            q = sb.queries[i]
            qid, raw = q["qid"], q["question"]
            n += 1
            # ① 全局全量排名
            real = [str(h["chunk_id"]) for h in idx.search_hybrid(raw, top_k=sb.N)]
            mine = [sb.cids[int(j)] for j in sb.order(qid, "zh")]
            ok_full = real == mine
            # ② 生产路径 quota（中文主路）
            real_q = [str(h["chunk_id"]) for h in idx.search_layered(raw, top_k=PROD["n"],
                                                                     floor=PROD["floor"])]
            mine_q = [sb.cids[int(j)] for j in sb.sel_quota(qid, "zh")]
            ok_q = real_q == mine_q
            # ③ 生产路径 quota（英文检索式，**生产调的是 top_k=xling_k**）
            ok_en = True
            if q["translation"]:
                real_en = [str(h["chunk_id"]) for h in idx.search_layered(
                    q["translation"], top_k=PROD["xling_k"], floor=PROD["floor"])]
                mine_en = [sb.cids[int(j)] for j in sb.sel_quota(qid, "en",
                                                                 top_k=PROD["xling_k"])]
                ok_en = real_en == mine_en
                # ③b 大 K 英文臂（`_xling_ab.py` 的口径，用于暴露口径差异）
                real_en2 = [str(h["chunk_id"]) for h in idx.search_layered(
                    q["translation"], top_k=PROD["n"], floor=PROD["floor"])]
                ok_en = ok_en and real_en2 == [sb.cids[int(j)]
                                               for j in sb.sel_quota(qid, "en")]
            if not (ok_full and ok_q and ok_en):
                bad += 1
                print(f"  ❌ {qid}  全量={ok_full} quota_zh={ok_q} quota_en={ok_en}")
                if not ok_full:
                    d = next((k for k, (a, b) in enumerate(zip(real, mine)) if a != b), None)
                    print(f"       全量首个分歧位 {d}：真={real[d] if d is not None else '-'} "
                          f"沙盒={mine[d] if d is not None else '-'}")
        print(f"  抽样 {n} 题 → {'✅ 全部逐块一致' if not bad else f'❌ {bad} 题不一致'}")
    return bad


# ────────────────────── 只用沙盒复现生产头条数字 ──────────────────────

def union_variants(groups: list[str]) -> int:
    """并集的**英文补充臂口径**对结果的影响（毫秒级）。

    ⚠️ 这是 Step 0 抓到的第一处真问题：`_xling_ab.py` 的 A2 臂把英文臂算成
    "24 选的前 8"，而**生产**调的是 `search_layered(xling, top_k=8)` ——
    后者在 K=8 时会用 floor **从尾部替换 5 个槽位**（5 篇 × floor=1），
    于是"英文臂的前 8 名"根本不是英文排名的前 8。两种口径结果不同。
    """
    rows: dict[str, list[int]] = {}
    for group in groups:
        sb = Sandbox(group)
        for q in sb.queries:
            qid = q["qid"]
            zh = sb.sel_quota(qid, "zh")
            if not q["translation"]:
                continue
            en_full = sb.sel_quota(qid, "en")                       # 24 选（含 floor）
            en_small = sb.sel_quota(qid, "en", top_k=PROD["xling_k"])  # 生产实际调用
            en_glob = [int(i) for i in sb.order(qid, "en")][:PROD["xling_k"]]
            arms = {
                "① 生产：英文臂 top8（含 floor 替换）": en_small,
                "② `_xling_ab` 口径：24 选的前 8": en_full[:PROD["xling_k"]],
                "③ 英文臂 = 纯全局 top8": en_glob,
                "④ 英文臂 = 全 24（不截断）": en_full,
            }
            for name, extra in arms.items():
                seen = set(zh)
                sel = zh + [i for i in extra if i not in seen]
                rows.setdefault(name, [0, 0, 0])           # [题数, 未命中, 字符]
                r = sb.evaluate(qid, sel)
                rows[name][0] += 1
                rows[name][1] += r["miss"]
                rows[name][2] += r["chars"]
    print("\n\n########## 并集的「英文补充臂口径」对比（中文主路固定 = 生产 quota@24）")
    print(f"  {'口径':<38}{'题数':>6}{'未命中':>8}{'平均字符':>10}")
    for name, (n, m, c) in rows.items():
        print(f"  {name:<36}{n:>6}{m:>8}{c / max(n, 1):>10.0f}")
    return 0


def _gold_of(sb: "Sandbox", qid: str) -> dict[str, Any]:
    """把一道题的 gold 拆到**块级**：

    · `qchunks`：引文块集合（chunk 含 gold evidence quote 的逐字匹配）← 最接近"gold 证据块"
    · `achunks`：锚点块集合（chunk 含 `must_all` 锚点）
    · `nq/nа`：引文条数 / 锚点个数
    · `qvec[j]`：第 j 条引文命中的所有块（用于"这条证据覆盖了吗"）
    逐题缓存，避免重复扫描。
    """
    key = ("gold", qid)
    cached = sb._mask.get(key)
    if cached is not None:
        return cached
    am, qm = sb.masks(qid)
    q = sb.queries[sb.qidx[qid]]
    nq, na = len(q["quotes"]), len(q["must_all"])
    qmap = {j: {i for i in range(sb.N) if int(qm[i]) >> j & 1} for j in range(nq)}
    amap = {j: {i for i in range(sb.N) if int(am[i]) >> j & 1} for j in range(na)}
    out = {"qchunks": {i for s in qmap.values() for i in s},
           "achunks": {i for s in amap.values() for i in s},
           "nq": nq, "na": na, "qmap": qmap, "amap": amap}
    sb._mask[key] = out
    return out


def oracle(groups: list[str]) -> int:
    """**重排上界**：假想一个"完美重排器"能把候选里的 gold 块排到最前 → 量 headroom。

    这是"重排值不值得做"的判定器，**零成本、确定性**（不需要真训/真跑任何模型）：
      · 当前@k        —— 现状：gold 块进前 k 的比例
      · 重排上界@k    —— 候选集**不变**、排序完美：min(候选内 gold 块数, k)
      · 绝对上限@k    —— 候选集**也不设限**（= 召回也完美）：min(该题全部 gold 块数, k)
    两个差值分别回答："排序还有多少空间" 与 "召回还有多少空间"。
    """
    kns = (1, 5, 8, 12)
    arms: list[tuple[str, Any]] = [
        ("旧生产（英文臂 K=8）", lambda sb, q: sb.sel_xling(q, xling_k=8)),
        ("新生产（英文臂 K=24）", lambda sb, q: sb.sel_xling(q, xling_k=24)),
        ("无跨语言（zh only）", lambda sb, q: sb.sel_quota(q, "zh")),
    ]
    acc: dict[tuple[str, str], dict] = {}

    def _a(arm: str, layer: str) -> dict:
        return acc.setdefault((arm, layer), {
            "gold": 0, **{f"cur{k}": 0 for k in kns},
            **{f"orc{k}": 0 for k in kns}, **{f"ceil{k}": 0 for k in kns},
            "n": 0, "newg12": 0})

    for group in groups:
        sb = Sandbox(group)
        for q in sb.queries:
            qid = q["qid"]
            g = _gold_of(sb, qid)
            layer = "M1 跨篇" if qid.startswith(("G1-M1", "G2-M1")) else "其余"
            for nm, fn in arms:
                sel = [int(i) for i in fn(sb, qid)]
                pos = {i: r + 1 for r, i in enumerate(sel)}
                in_sel = [i for i in g["qchunks"] if i in pos]
                for lay in (layer, "全部"):
                    d = _a(nm, lay)
                    d["gold"] += len(g["qchunks"])
                    d["n"] += 1
                    for k in kns:
                        d[f"cur{k}"] += sum(1 for i in in_sel if pos[i] <= k)
                        d[f"orc{k}"] += min(len(in_sel), k)
                        d[f"ceil{k}"] += min(len(g["qchunks"]), k)
                    d["newg12"] += max(0, min(len(in_sel), 12) - sum(1 for i in in_sel if pos[i] <= 12))

    print("\n\n########## 重排上界（oracle）：假想完美重排器 —— "
          "回答『排序还有多少空间』")
    print(f"  {'臂 / 分层':<26}{'gold块':>7}"
          f"{'当前@1':>8}{'@5':>7}{'@8':>7}{'@12':>7}"
          f"{'重排@5':>8}{'@8':>7}{'@12':>7}{'绝对@12':>9}{'重排可增@12':>12}")
    for nm, _ in arms:
        for lay in ("全部", "M1 跨篇", "其余"):
            d = acc[(nm, lay)]
            if not d["gold"]:
                continue
            gt = d["gold"]
            print(f"  {(nm[:12] + '·' + lay):<26}{gt:>7}"
                  f"{d['cur1'] / gt:>8.0%}{d['cur5'] / gt:>7.0%}"
                  f"{d['cur8'] / gt:>7.0%}{d['cur12'] / gt:>7.0%}"
                  f"{d['orc5'] / gt:>8.0%}{d['orc8'] / gt:>7.0%}"
                  f"{d['orc12'] / gt:>7.0%}{d['ceil12'] / gt:>9.0%}"
                  f"{d['newg12']:>6}/{d['n']:<5}")
    print("  ↑ 『重排可增@12』= 完美重排后**新增进入前 12** 的 gold 块数 / 题数")
    print("  ↑ 『绝对@12』= 连召回也完美（该题全部 gold 块都能参与排序）时的 @12")
    return 0


def chunk_level(groups: list[str]) -> int:
    """**块级** gold 覆盖：回答"喂给回答 LLM 的 chunk 到底变好没有"。

    为什么要它：题级判据（`must_all` 全中 = 0/1）会把"3 个锚点进了 2 个"算成 0 分，
    既看不到**进了多少块**，也看不到**块在第几位**。而产品的本质是"让更多 gold 块进来"。
    """
    import re as _re
    import statistics as st
    arms: list[tuple[str, Any]] = [
        ("① 无跨语言（zh only）", lambda sb, q: sb.sel_quota(q, "zh")),
        ("② 旧生产（英文臂 K=8）", lambda sb, q: sb.sel_xling(q, xling_k=8)),
        ("③ 新生产（英文臂 K=24）", lambda sb, q: sb.sel_xling(q, xling_k=24)),
    ]
    rows: dict[str, list[float]] = {}
    per_q: dict[str, dict[str, dict]] = {}
    # ★ 候选规模（vs 语料）｜ gold 块**完整排名分布**（含"未进候选"）｜ gold 密度 lift
    cand_size: dict[str, dict[str, list[int]]] = {}
    corpus_n: dict[str, int] = {}
    rank_acc: dict[str, dict] = {}          # arm -> {flag(None/M1/其它) -> 统计}

    def _acc(nm: str, flag: Any) -> dict:
        return rank_acc.setdefault(nm, {}).setdefault(
            flag, {"gold": 0, "in": 0, "ranks": [], "bucket": [0] * 4,
                   "base": 0.0, "dens": 0.0, "n": 0})
    for group in groups:
        sb = Sandbox(group)
        for q in sb.queries:
            qid = q["qid"]
            g = _gold_of(sb, qid)
            for nm, fn in arms:
                sel = fn(sb, qid)
                pos = {int(i): r for r, i in enumerate(sel)}
                qcov = sum(1 for j in range(g["nq"]) if g["qmap"][j] & set(pos))
                acov = sum(1 for j in range(g["na"]) if g["amap"][j] & set(pos))
                got_q = g["qchunks"] & set(pos)
                got_a = g["achunks"] & set(pos)
                first = [pos[i] for i in g["qchunks"] if i in pos]
                b = rows.setdefault(nm, [0] * 14)
                b[0] += 1                       # 题数
                b[1] += qcov                    # 引文条数覆盖
                b[2] += g["nq"]                 # 引文条数总数
                b[3] += acov                    # 锚点个数覆盖
                b[4] += g["na"]                 # 锚点总数
                b[5] += len(got_q)              # gold 引文块命中数
                b[6] += len(g["qchunks"])       # gold 引文块总数
                b[7] += len(got_a)              # gold 锚点块命中数
                b[8] += len(g["achunks"])       # gold 锚点块总数
                b[9] += len(sel)                # 候选块数
                b[10] += (min(first) if first else len(sel))   # gold 块首位次
                b[11] += 1 if first and min(first) < 8 else 0
                b[12] += 1 if first and min(first) < 12 else 0
                b[13] += 1 if first and min(first) < 24 else 0
                per_q.setdefault(nm, {})[qid] = {
                    "sel": [int(i) for i in sel], "got_q": got_q, "got_a": got_a,
                    "tot_q": len(g["qchunks"]), "tot_a": len(g["achunks"]),
                    "nq": g["nq"], "na": g["na"],
                    "first": min(first) if first else -1,
                }
                cand_size.setdefault(nm, {}).setdefault(group, []).append(len(sel))
                corpus_n[group] = sb.N
                m1flag = bool(_re.match(r"^G\d+-M1-", qid))
                rk = sorted(pos[i] + 1 for i in g["qchunks"] if i in pos)   # 1-based 名次
                for flag in (None, "M1" if m1flag else "其余"):
                    d = _acc(nm, flag)
                    d["gold"] += len(g["qchunks"])
                    d["in"] += len(rk)
                    d["ranks"].extend(rk)
                    for r in rk:
                        d["bucket"][0 if r <= 8 else 1 if r <= 12 else 2 if r <= 24 else 3] += 1
                    d["base"] += len(g["qchunks"]) / sb.N          # 全语料 gold 密度
                    d["dens"] += len(got_q) / max(len(sel), 1)     # 候选内 gold 密度
                    d["n"] += 1

    print("\n\n########## 块级 gold 覆盖（**回答『喂给回答 LLM 的 chunk 变好没有』**）")
    print(f"  {'臂':<26}{'引文条覆盖':>12}{'锚点覆盖':>11}{'gold块覆盖':>12}"
          f"{'候选块数':>10}{'gold密度':>10}{'gold首位次':>11}{'进前8':>8}{'进前12':>8}")
    for nm, _ in arms:
        b = rows[nm]
        n = max(b[0], 1)
        print(f"  {nm:<24}{b[1] / max(b[2], 1):>11.1%}{b[3] / max(b[4], 1):>11.1%}"
              f"{b[5] / max(b[6], 1):>12.1%}"
              f"{b[9] / n:>10.1f}{b[5] / max(b[9], 1):>10.1%}"
              f"{b[10] / n:>11.1f}{b[11] / n:>8.0%}{b[12] / n:>8.0%}")

    # ── 按题型拆：M1（组级「五篇里哪些篇…」类）与其余 ──
    is_m1 = {qq: bool(_re.match(r"^G\d+-M1-", qq)) for qq in per_q[arms[0][0]]}
    print("\n########## 按题型拆（M1 = 组级「五篇里哪些篇 / 各自…」类）")
    print(f"  {'臂':<24}{'M1 gold块覆盖':>15}{'其余 gold块覆盖':>17}"
          f"{'M1 候选块数':>13}{'其余 候选块数':>15}")
    for nm, _ in arms:
        pq = per_q[nm]
        cell = []
        for m1flag in (True, False):
            ks = [k for k in pq if is_m1[k] is m1flag]
            got = sum(len(pq[k]["got_q"]) for k in ks)
            tot = sum(pq[k]["tot_q"] for k in ks)
            cell.append(f"{got / max(tot, 1):>15.1%}")
            sel_avg = sum(len(pq[k]["sel"]) for k in ks) / max(len(ks), 1)
            cell.append(f"{sel_avg:>13.1f}" if m1flag else f"{sel_avg:>15.1f}")
        print(f"  {nm:<22}{cell[0]}{cell[2]}{cell[1]}{cell[3]}")

    # ── ① 候选块数 vs 语料规模（担心『塞满 → 召回变假』）──
    xl = PROD["xling_k"]
    print("\n########## ① 候选块数 vs 语料规模（担心『塞满 → 召回虚高』）")
    print(f"  {'组':<9}{'语料块数':>9}{'旧K=8 均/最大':>16}{'占比(均)':>10}"
          f"{'新K=24 均/最大':>16}{'占比(均)':>10}")
    allsz: dict[str, list[int]] = {}
    for g, n in corpus_n.items():
        cells = []
        for nm, _ in arms[1:]:
            szs = cand_size[nm][g]
            allsz.setdefault(nm, []).extend(szs)
            cells.append(f"{sum(szs) / max(len(szs), 1):.1f} / {max(szs)}")
            cells.append(f"{sum(szs) / max(len(szs), 1) / n:.1%}")
        print(f"  {g:<9}{n:>9}{cells[0]:>16}{cells[1]:>10}{cells[2]:>16}{cells[3]:>10}")
    a2, a3 = arms[1][0], arms[2][0]
    print(f"  → 硬上限 = 主路 K + 补充 K = {xl}+{xl} = {2 * xl} 块 → 占最小语料"
          f" {2 * xl / max(min(corpus_n.values()), 1):.0%}；实测均值："
          f"旧 {sum(allsz[a2]) / len(allsz[a2]):.1f} 块（"
          f"{sum(allsz[a2]) / len(allsz[a2]) / (sum(corpus_n.values()) / len(corpus_n)):.0%}）、"
          f"新 {sum(allsz[a3]) / len(allsz[a3]):.1f} 块（"
          f"{sum(allsz[a3]) / len(allsz[a3]) / (sum(corpus_n.values()) / len(corpus_n)):.0%}）")

    # ── ② gold 块的**排名分布**（全部 gold 块，含"未进候选"）──
    print("\n########## ② gold 块的排名分布（**全部** gold 块；跨篇题有多个，不能只看首位）")
    print(f"  {'臂 / 分层':<24}{'gold块':>8}{'进候选':>8}{'均名次':>8}{'中位':>7}"
          f"{'1-8':>8}{'9-12':>8}{'13-24':>8}{'≥25':>8}{'未进':>8}")
    for nm, _ in arms:
        for flag, label in ((None, "全部"), ("M1", "M1 跨篇"), ("其余", "其余")):
            d = _acc(nm, flag)
            if not d["gold"]:
                continue
            rk = d["ranks"]
            gtot = max(d["gold"], 1)
            print(f"  {(nm[:12] + '·' + label):<24}{d['gold']:>8}{d['in']:>8}"
                  f"{(sum(rk) / len(rk) if rk else 0):>8.1f}"
                  f"{(st.median(rk) if rk else 0):>7.1f}"
                  f"{d['bucket'][0] / gtot:>8.0%}{d['bucket'][1] / gtot:>8.0%}"
                  f"{d['bucket'][2] / gtot:>8.0%}{d['bucket'][3] / gtot:>8.0%}"
                  f"{(d['gold'] - d['in']) / gtot:>8.0%}")

    # ── ③ 召回选择性：候选内 gold 密度 vs 全语料 gold 密度（lift）──
    print("\n########## ③ 召回『虚不虚』：候选内 gold 密度 vs 全语料 gold 密度")
    for nm, _ in arms:
        d = _acc(nm, None)
        base = d["base"] / max(d["n"], 1)
        dens = d["dens"] / max(d["n"], 1)
        print(f"  {nm:<24}候选 {dens:>6.1%} ｜ 全语料 {base:>6.1%} ｜ "
              f"**lift {dens / max(base, 1e-9):>4.1f}×**（1× = 与随机抽无区别）")

    # ── 结构核对 + K=8 → K=24 到底多了哪些块 ──
    a, c = per_q[arms[1][0]], per_q[arms[2][0]]
    same_head = drop_q = drop_gold = 0
    added_gold = 0
    buckets = [0, 0, 0, 0]                  # 新增 gold 块的位次分布
    gains: list[tuple[int, str]] = []
    for qid in a:
        s8, s24 = set(a[qid]["sel"]), set(c[qid]["sel"])
        if a[qid]["sel"][:24] == c[qid]["sel"][:24]:
            same_head += 1
        drop = s8 - s24
        if drop:
            drop_q += 1
            drop_gold += len(drop & (a[qid]["got_q"] | a[qid]["got_a"]))
        pos24 = {i: r for r, i in enumerate(c[qid]["sel"])}
        newgold = [i for i in (s24 - s8) if i in (c[qid]["got_q"] | c[qid]["got_a"])]
        added_gold += len(newgold)
        for i in newgold:
            r = pos24[i]
            buckets[0 if r < 8 else 1 if r < 12 else 2 if r < 24 else 3] += 1
        gains.append((len(newgold), qid))
    print("\n########## K=8 → K=24：块级净变化")
    print(f"  主路前 24 块**逐块相同**的题：{same_head}/{len(a)}"
          f"（并集只在尾部追加 → 原有块位次不变）")
    print(f"  有块被**丢掉**的题：{drop_q}（其中丢掉 gold 块：{drop_gold}）")
    print(f"  新增 gold 块合计：**{added_gold}** 块")
    print(f"  新增 gold 块的位次分布：进前 8 {buckets[0]} ｜ 8~11 {buckets[1]} ｜ "
          f"12~23 {buckets[2]} ｜ **≥24（尾部）{buckets[3]}**")
    gains.sort(reverse=True)
    print(f"  新增 gold 块最多的题："
          f"{'、'.join(f'{q}(+{n})' for n, q in gains[:8] if n)}")
    return 0


def ablate(groups: list[str]) -> int:
    """Step 1：**消融矩阵** —— 一次只关一个组件，量它的边际贡献。

    为什么先做这个：**扫参（Step 2）的收益上限由消融决定**。若关掉某组件指标不变
    （Δ=0），那它的参数怎么调都是 0 收益，先扫它会得出"这个参数没用"的错误结论。

    分层 `live`/`dead` = 该题 BM25 是否有非零信号（`_bm_or_none` 的**真实判据**）。
    这一分层同时**验证机制**：关掉词法路的 Δ 应当**只来自 live 层**（dead 层本来就被闸门丢了）。
    检索层是确定性的 → Δ 是精确值，不需要重复采样。
    """
    xl = PROD["xling_k"]
    arms: list[tuple[str, dict]] = [
        ("基线（生产默认 N=24/floor=1/k=60）", {}),
        ("A1a 关词法路（只剩向量）", {"use_bm": False}),
        ("A1b 关向量路（只剩词法）", {"use_vec": False}),
        ("A2 关 `_bm_or_none` 闸门（伪装词法信号放行）", {"fake_bm": True}),
        ("A3 关篇级保底（floor=0 → 退化成 global）", {"floor": 0}),
        ("A4 保底加强（floor=2）", {"floor": 2}),
    ]
    xarms: list[tuple[str, int, dict]] = [
        ("A5a 跨语言并集（**旧生产**：英文臂 K=8）", 8, {}),
        (f"A5b 跨语言并集（**现生产**：英文臂不截断 K={xl}）", xl, {}),
        # 交互项：词法路存在的**意义**就是被英文译式救活 → 必须交叉看（A1a 说它净贡献只有 1）
        (f"A6a = A5b + 关词法路（英文臂不截断 K={xl}）", xl, {"use_bm": False}),
        ("A6b = A5a + 关词法路（英文臂 K=8）", 8, {"use_bm": False}),
    ]

    res: dict[str, dict[str, dict | None]] = {nm: {} for nm, *_ in arms + xarms}
    layer: dict[str, bool] = {}
    for group in groups:
        sb = Sandbox(group)
        for q in sb.queries:
            qid = q["qid"]
            layer[qid] = float(sb.bm_global(qid).max()) > 0
            for nm, kw in arms:
                if nm.startswith("A1b") and not layer[qid]:
                    res[nm][qid] = None              # 无词法信号 → 该臂**不适用**
                    continue
                res[nm][qid] = sb.evaluate(qid, sb.sel_quota(qid, "zh", **kw))
            for nm, xk, kw2 in xarms:
                res[nm][qid] = (sb.evaluate(qid, sb.sel_xling(qid, xling_k=xk, **kw2))
                                if q["translation"] else None)

    base = res[arms[0][0]]
    n_all = len(layer)
    n_live = sum(1 for v in layer.values() if v)
    print(f"\n\n########## Step 1 消融矩阵（{n_all} 题：词法路开火 {n_live} ／ 死 {n_all - n_live}）")
    print(f"  {'臂':<42}{'适用':>5}{'未命中':>8}{'Δ':>6}{'live漏':>7}{'dead漏':>8}"
          f"{'篇覆盖':>8}{'字符':>8}")
    delta: dict[str, int] = {}
    papn: dict[str, float] = {}
    for nm, *_ in arms + xarms:
        rec = res[nm]
        qs = [q for q, r in rec.items() if r is not None]
        miss = sum(int(rec[q]["miss"]) for q in qs)
        bmiss = sum(int(base[q]["miss"]) for q in qs if base[q])
        lv = sum(int(rec[q]["miss"]) for q in qs if layer[q])
        dd = sum(int(rec[q]["miss"]) for q in qs if not layer[q])
        pap = sum(int(rec[q]["papers"]) for q in qs) / max(len(qs), 1)
        ch = sum(int(rec[q]["chars"]) for q in qs) / max(len(qs), 1)
        delta[nm] = miss - bmiss
        papn[nm] = pap
        print(f"  {nm:<40}{len(qs):>5}{miss:>8}{delta[nm]:>+6}{lv:>7}{dd:>8}"
              f"{pap:>8.2f}{ch:>8.0f}")

    print(f"\n########## 逐题差异（相对基线）")
    for nm, *_ in arms[1:] + xarms:
        rec = res[nm]
        worse = [q for q, r in rec.items() if r is not None and r["miss"] and not base[q]["miss"]]
        better = [q for q, r in rec.items() if r is not None and not r["miss"] and base[q]["miss"]]
        if not worse and not better:
            print(f"  {nm}：无差异")
            continue
        print(f"  {nm}")
        if worse:
            print(f"      变漏 {len(worse)} 道：{'、'.join(worse[:12])}"
                  f"{'…' if len(worse) > 12 else ''}")
        if better:
            print(f"      变中 {len(better)} 道：{'、'.join(better[:12])}"
                  f"{'…' if len(better) > 12 else ''}")

    print("\n########## 判定（**必须多指标**：Δ未命中 为主，Δ篇覆盖 为辅）")
    print("  为什么不能只看未命中：题目命中只问『锚点在不在候选里』，而有的组件"
          "（如篇级保底）\n  的价值是『每篇都可见』 → 它的收益在**答案层**兑现，"
          "在本指标上可能恰好为 0。")
    bp = papn[arms[0][0]]
    for nm, *_ in arms[1:]:
        d, dp = delta[nm], papn[nm] - bp
        if d > 0:
            v = "留（关掉确实变差）"
        elif d < 0:
            v = "**有害（关掉反而更好）→ 应删/该改机制**"
        elif dp < -0.2:
            v = f"留（未命中不变，但**篇覆盖 {dp:+.2f}** → 收益在答案层兑现）"
        else:
            v = "无效或不可触发（未命中与篇覆盖都无变化）→ 可删/挂起"
        print(f"  {nm:<40}{d:>+6}{dp:>+8.2f}   {v}")
    for nm, *_ in xarms:
        d, dp = delta[nm], papn[nm] - bp
        print(f"  {nm:<40}{d:>+6}{dp:>+8.2f}   "
              f"{'留（净收益）' if d < 0 else '无收益'}")

    # ── 区分度：这一批题**能提供多少信息**？────────────────────────────
    # 题集的价值不在"多少道"，而在"多少道能区分方案"。所有臂都命中的题 = 零信息。
    # 若区分度只有个位数，那任何结论都建立在个位数道题上 → 扩量应服务"区分度"，不是"样本量"。
    print("\n########## 区分度（决定题量该不该扩；**扩量要加的是『能区分的题』**）")
    miss: dict[str, set[str]] = {}
    for nm, *_ in arms + xarms:
        miss[nm] = {q for q, r in res[nm].items() if r is not None and r["miss"]}
    bad_arms = [nm for nm, S in miss.items() if S]
    union = set().union(*(miss[nm] for nm in bad_arms)) if bad_arms else set()
    always = n_all - len(union)
    print(f"  对所有臂都命中的题：**{always}/{n_all}**（这些题对本次比较**零区分度**）")
    print(f"  至少被一个臂漏掉的题：**{len(union)}/{n_all}**"
          f"（{100 * len(union) / max(n_all, 1):.0f}% ← 我们的全部结论只建立在这些题上）")
    print("  逐臂的『专属区分题』（只有它漏、基线不漏）：")
    for nm in bad_arms:
        only = miss[nm] - set().union(*(miss[o] for o in bad_arms if o != nm))
        tag = "、".join(sorted(only)[:10]) + ("…" if len(only) > 10 else "")
        print(f"      {nm:<40}{len(only):>4} 道  {tag}")
    return 0


def _acc_arms(sbs: dict, groups: list[str],
              arms: list) -> dict[str, list]:
    """`arms = [(名字, sel_of(g, sb, qid) -> 下标列表)]` → 与 `--prod` 同口径的累计表。

    ⚠️ 沙盒**只加载一次**（每个 Sandbox 加载 + 打分要 ~4~20s），否则扫 20 组要十几分钟。
    """
    rows: dict[str, list] = {}
    for nm, fn in arms:
        # 0 题数 ｜ 1 未命中 ｜ 2 篇覆盖 ｜ 3 字符 ｜ 4/5 M1篇全(任意块)/题数 ｜ 6/7 gold@12
        # 8/9 M1块@12 ｜ 10 (题,有gold的篇) 对数 ｜ 11 该对**有 gold 块**进前12
        # 12 位次样本(list) ｜ 13 **全篇都有 gold 块**的 M1 题数 ｜ 14 单篇最大独占样本(list)
        # 15 候选内 gold 块数 ｜ 16 语料块总数 ｜ 17 候选块总数（15~17 用来算**体积 lift**）
        acc: list = ([0] * 12) + [[]] + [0] + [[]] + [0, 0, 0]
        for g in groups:
            sb = sbs[g]
            for q in sb.queries:
                qid = q["qid"]
                sel = fn(g, sb, qid)
                r = sb.evaluate(qid, sel)
                gd = _gold_of(sb, qid)
                pos = {int(i): k for k, i in enumerate(sel)}
                acc[0] += 1
                acc[1] += int(r["miss"])
                acc[2] += int(r["papers"])
                acc[3] += int(r["chars"])
                acc[6] += sum(1 for i in gd["qchunks"] if i in pos and pos[i] < 12)
                acc[7] += len(gd["qchunks"])
                if "-M1-" in qid:
                    goldp = {sb.pdf_of[i] for i in gd["qchunks"]}
                    topp = {sb.pdf_of[i] for i in sel[:12]}
                    acc[5] += 1
                    acc[4] += 1 if goldp <= topp else 0
                    acc[8] += sum(1 for i in gd["qchunks"] if i in pos and pos[i] < 12)
                    acc[9] += len(gd["qchunks"])
                    byp: dict[str, list[int]] = {}
                    for i in gd["qchunks"]:
                        byp.setdefault(sb.pdf_of[i], []).append(pos.get(i, 10 ** 6))
                    nohit = 0
                    for v in byp.values():
                        acc[10] += 1
                        if min(v) < 12:
                            acc[11] += 1
                            acc[12].append(min(v))
                        else:
                            nohit += 1
                    acc[13] += 1 if nohit == 0 else 0        # **每篇的 gold 块都进前12**
                    acc[15] += len(gd["qchunks"] & set(sel))  # 候选内 gold 块（**体积 lift**）
                    acc[16] += sb.N                           # M1 语料块总数
                    acc[17] += len(sel)                       # M1 候选块总数
                    dist: dict[str, int] = {}
                    for i in sel[:12]:
                        dist[sb.pdf_of[i]] = dist.get(sb.pdf_of[i], 0) + 1
                    acc[14].append(max(dist.values()) if dist else 0)   # 单篇最大独占
        rows[nm] = acc
    return rows


def _show_metrics(rows: dict[str, list], arms: list, title: str) -> None:
    print(f"\n########## {title}")
    print(f"  {'配置':<30}{'全篇有gold':>11}{'篇缺gold':>10}{'独占':>5}"
          f"{'篇内命中':>9}{'每篇位次':>9}{'M1块@12':>9}{'gold@12':>9}"
          f"{'M1篇全(任意)':>13}{'未命中':>7}{'体积lift':>9}{'字符':>7}")
    for nm, *_ in arms:
        a = rows[nm]
        med = sorted(a[12])[len(a[12]) // 2] if a[12] else -1
        dom = sorted(a[14])[len(a[14]) // 2] if a[14] else 0
        lift = ((a[15] / max(a[17], 1)) / (a[9] / max(a[16], 1))) if a[16] else 0.0
        print(f"  {nm:<28}{f'{a[13]}/{a[5]}':>11}"
              f"{f'{a[10] - a[11]}/{a[10]}':>10}{dom:>5}"
              f"{a[11] / max(a[10], 1):>9.1%}{med:>9.1f}"
              f"{a[8] / max(a[9], 1):>9.1%}{a[6] / max(a[7], 1):>9.1%}"
              f"{f'{a[4]}/{a[5]}':>13}{a[1]:>7}{lift:>8.1f}×{a[3] / max(a[0], 1):>7.0f}")
    print("  判据（**前四列是主判据**，后四列是参考/诊断）：\n"
          "    全篇有gold   M1 题里「**每篇的 gold 块**都进前 12」的题数 ← 最贴近目标\n"
          "    篇缺gold     (题, 有 gold 的篇) 对里，该篇 gold 块**一个都没进前 12** 的对数\n"
          "    独占         前 12 里**同篇最多块数**的中位（单篇独占程度）\n"
          "    篇内命中     1 - 篇缺gold 比例\n"
          "    每篇位次     命中时该篇最好 gold 块位次中位（越小越好）\n"
          "    M1块@12/gold@12  **按块**平均 → ⚠️ **会奖励单篇独占**，只作参考\n"
          "    M1篇全(任意) 只要求该篇有**任意块**（不是 gold 块）→ 会高估，只作参考")


def ksweep(groups: list[str]) -> int:
    """**输出大小 K 扫描**（用跨篇判据，而不是 `gold@12`）。

    为什么要单独扫：`Kz=Ke=16`（+交错封顶 16）是**按 `gold@12` 位次口径**选出来的，
    那时还没做 M1 扩量、也没用「每篇的 gold 块」这套判据。现在要重新问：
      · 池子变大（Kz/Ke 与输出同步涨）→ 跨篇 gold 能不能回来？代价多少字符？
      · **深挖再截**（Kz/Ke 大、输出仍 16）→ 区分是"臂内截断"还是"最终截断"砍掉了 gold
    """
    sbs = {g: Sandbox(g) for g in groups}
    def kw(Kz: int, Ke: int, out: int) -> dict:
        return dict(top_k=Kz, xling_k=Ke, out_n=out, merge="inter",
                    w_en=1.0, cover=True, floor=1)

    arms: list = []
    for K in (16, 24, 32, 48):
        arms.append((f"Kz=Ke={K} 输出{K}",
                     lambda g, sb, qid, K=K: sb.sel_xling(qid, **kw(K, K, K))))
    for Kz, out in ((24, 16), (32, 16), (48, 16), (32, 24), (48, 32)):
        arms.append((f"Kz=Ke={Kz} 输出{out}（深挖再截）",
                     lambda g, sb, qid, Kz=Kz, out=out:
                     sb.sel_xling(qid, **kw(Kz, Kz, out))))
    # **交错合并不改会怎样**：同臂深下换成"整块尾部追加"（旧语义）
    for Kz, out in ((16, 16), (32, 16), (32, 32)):
        arms.append((f"Kz=Ke={Kz} 输出{out} **append（不交错）**",
                     lambda g, sb, qid, Kz=Kz, out=out:
                     sb.sel_xling(qid, **{**kw(Kz, Kz, out), "merge": "append"})))
    rows = _acc_arms(sbs, groups, arms)
    _show_metrics(rows, arms, f"输出大小 K 扫描（{len(groups)} 组，"
                              f"{rows[arms[0][0]][5]} 道 M1；判据=每篇的 gold 块）")
    print("  → 看 `全篇有gold` / `篇缺gold` / `独占`；`字符` 是代价。"
          "『深挖再截』若比『同步涨』更好，说明瓶颈在**最终截断**而非臂内召回。")
    return 0


def m1audit(groups: list[str], cfgs: list[tuple[str, dict]], only: str = "") -> int:
    """**逐题审**：把每道 M1 的「前 12 篇分布 / 每篇 gold 块命中 / 缺篇」摊开。

    为什么必须逐题看（聚合指标会骗人，两个方向都会）：
      · `gold@12` **按块**平均 → **单篇独占反而得分高**（同篇 9 个 gold 块 = 9 次成功），
        于是"跨篇覆盖差"被平均掉；cap 这种"保每篇有名额"的改动会在它上面看起来更差。
      · `M1篇全` 用的是「前 12 里出现的**篇**」vs「有 gold 的**篇**」→ 只要每篇有**任意块**
        就算"覆盖"，**不保证那篇的 gold 块进来了**（cap 的 71/71 就是这么来的）。
      → 所以要看**每篇的 gold 块命中**（分子=该篇进前12的 gold 块数，分母=该篇 gold 块总数）。
    """
    pat = only.strip().upper()
    summ: dict[str, list] = {nm: [0, 0, 0, 0] for nm, _ in cfgs}
    for g in groups:
        sb = Sandbox(g)
        for q in sb.queries:
            qid = q["qid"]
            if "-M1-" not in qid:
                continue
            if pat and pat not in qid.upper():
                continue
            gd = _gold_of(sb, qid)
            gp: dict[str, int] = {}
            for i in gd["qchunks"]:
                gp[sb.pdf_of[i]] = gp.get(sb.pdf_of[i], 0) + 1
            if not gp:
                continue
            print(f"\n### {qid}　gold 篇 {len(gp)} ｜ " +
                  " ｜ ".join(f"{p.replace('v1','')}×{n}" for p, n in gp.items()))
            for nm, kw in cfgs:
                sel = sb.sel_xling(qid, **kw)
                top = sel[:12]
                dist: dict[str, int] = {}
                for i in top:
                    k = sb.pdf_of[i].replace("v1", "")
                    dist[k] = dist.get(k, 0) + 1
                pos = {int(i): k for k, i in enumerate(sel)}
                hit = {p.replace("v1", ""): [sum(1 for i in gd["qchunks"]
                                                if sb.pdf_of[i] == p and i in pos
                                                and pos[i] < 12), n]
                       for p, n in gp.items()}
                miss = [p.replace("v1", "") for p, n in gp.items()
                        if hit[p.replace("v1", "")][0] == 0]
                print(f"    {nm:<26}前12: " +
                      " ".join(f"{k}×{v}" for k, v in sorted(dist.items(), key=lambda x: -x[1])) +
                      f" ｜ 每篇gold块 " +
                      " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in hit.items()) +
                      (f" ｜ **缺 {','.join(miss)}**" if miss else " ｜ ✅全篇有gold"))
                s = summ[nm]
                s[0] += 1                                   # 题数
                s[1] += 1 if not miss else 0                # **全篇都有 gold 块**的题数
                s[2] += len(miss)                           # 缺（篇,题）对数
                s[3] += len(gp)                             # (篇,题) 对总数
                s.append(max(dist.values()) if dist else 0)  # 单篇最大独占块数
    print(f"\n{'=' * 104}\n汇总（判据 = **每篇的 gold 块**是否进前 12，"
          f"不是『该篇有没有任意块』）")
    print(f"  {'配置':<28}{'全篇都有gold':>14}{'该篇缺gold(对)':>16}{'单篇最大独占(中位)':>20}")
    for nm, _ in cfgs:
        s = summ[nm]
        mx = sorted(s[4:])[len(s[4:]) // 2] if s[4:] else 0
        print(f"  {nm:<26}{f'{s[1]}/{s[0]}':>14}{f'{s[2]}/{s[3]}':>16}{mx:>20}")
    return 0


def capsweep(groups: list[str]) -> int:
    """**①每篇配额 cap**（硬上限、去掉 floor）—— 直接打"漏篇 + 单篇霸占"。

    基线 = 刚接线的生产（16/16 加权交错 + 篇保底 + floor=1）。
    扫 `cap ∈ {1..5} × floor ∈ {0,1}`（**臂内**硬上限，即每路 `search_layered` 里就限）；
    另扫**合并后**再 cap（`cap_final`），因为交错会把两路的同篇块重新聚到一起。
    判据用块级 + 跨篇那套（`gold@12` / `M1块@12` / `M1篇全` / `篇内命中` / `每篇位次`）。

    ## 实测结论（2026-09-27，5 组 / 71 道 M1）：**中性 —— 治住了独占，但治不了"缺 gold"**

        配置                        全篇有gold  篇缺gold   独占  篇内命中  每篇位次   候选
        基线 floor=1 无 cap              6/71   121/190     6   36.3%    2.0    24
        臂内 cap=2 floor=0               7/71   121/190     3   36.3%    4.0    23
        臂内 cap=4 floor=1               7/71   120/190     4   36.8%    3.0    24
        合并后 cap=4/5 floor=0           8/71   123/121     4~5 35.3/36.3% 2.0~3.0 15~17
        臂内 cap=1 floor=0               2/71   144/190     2   24.2%    2.0    13
        floor=0 无 cap                  3/71   125/190     6   34.2%    2.0    24

    ⚠️ **本判据是"每篇的 gold 块是否进前 12"，不是"该篇有没有任意块"** —— 后者（`M1篇全`）
    会高估：cap 下它显示 71/71，但真正的"全篇有 gold"只有 6~8/71。

    · **cap 确实治住了单篇独占**：独占中位 6 → **3**（cap=2）；但
      **"篇缺 gold" 纹丝不动（121/190）** → 说明**缺 gold 不是被独占挤掉的**；
    · `每篇位次` 从 2.0 变 3.0~4.0（cap 放进来的块排得更靠后）→ 整体**中性**；
    · **`floor=0` 明显更差**（全篇 6→3、缺 121→125）→ **floor 必须保留**（"去掉 floor"
      这个方向实测被否）；
    · cap=1 是崩的（全篇 2/71、缺 144/190）→ 一题 5 篇、每篇只给 1 格 = 没得选。
    """
    sbs = {g: Sandbox(g) for g in groups}
    base = dict(top_k=16, xling_k=16, merge="inter", w_en=1.0, cover=True)
    arms: list = [("基线 16/16 inter+保底（floor=1, 无 cap）",
                   lambda g, sb, qid: sb.sel_xling(qid, **base))]
    for fl in (0, 1):
        for cap in (1, 2, 3, 4, 5):
            arms.append((f"臂内 cap={cap} floor={fl}",
                         lambda g, sb, qid, c=cap, f=fl: sb.sel_xling(
                             qid, **{**base, "floor": f, "cap": c})))
    for cap in (2, 3, 4, 5):
        arms.append((f"**合并后** cap={cap} floor=0",
                     lambda g, sb, qid, c=cap: sb.sel_xling(
                         qid, **{**base, "floor": 0, "cap_final": c})))
    rows = _acc_arms(sbs, groups, arms)
    _show_metrics(rows, arms, f"①每篇配额 cap 扫描（{len(groups)} 组，"
                              f"{rows[arms[0][0]][5]} 道 M1；基线=新生产）")
    print("  读法：`篇内命中`↑ / `每篇位次`↓ = 跨篇覆盖与排序变好；"
          "`候选`↓ = 硬上限把总数压下来（**不满 16 也算**，这就是'硬'的代价）")
    return 0


def prodcheck(groups: list[str]) -> int:
    """**生产路径回归**：新旧默认配置的 Tier-1 对比（零 LLM，确定性）。

    旧 = Kz 24 / Ke 24 / 整块尾部追加（`XLING_MERGE=append`）
    新 = Kz 16 / Ke 16 / **加权交错** `w_en=PROD["xling_w"]`（**2026-09-27 起 = 2.0**）

    为什么必须跑它：`_inter_ab.py` 用的是它自己的实现；本函数走的是**生产代码**
    （`pull_chunk._rrf_interleave_hits`），所以它是"改完生产后有没有真生效"的证据。
    """
    arms = [
        ("旧生产 24/24 append（原默认）", dict(top_k=24, xling_k=24, merge="append")),
        ("16/16 inter **无篇保底**",
         dict(top_k=16, xling_k=16, merge="inter", w_en=1.0, cover=False)),
        ("16/16 inter +篇保底 · w_en=1.0（改前默认）",
         dict(top_k=16, xling_k=16, merge="inter", w_en=1.0, cover=True)),
        (f"16/16 inter +篇保底 · w_en={PROD['xling_w']}（**现默认**）",
         dict(top_k=16, xling_k=16, merge="inter", cover=True)),
        ("16/16 inter +篇保底 · 输出24", dict(top_k=16, xling_k=16, merge="inter",
                                             w_en=1.0, cover=True, out_n=24)),
        ("16/16 inter +篇保底 · 输出32", dict(top_k=16, xling_k=16, merge="inter",
                                             w_en=1.0, cover=True, out_n=32)),
        ("24/24 inter +篇保底", dict(top_k=24, xling_k=24, merge="inter",
                                     w_en=1.0, cover=True)),
    ]
    rows: dict[str, list] = {}
    for nm, kw in arms:
        # 0 题数 ｜ 1 未命中 ｜ 2 篇覆盖 ｜ 3 字符 ｜ 4/5 M1篇全（任意块）/题数 ｜ 6/7 gold@12
        # 8/9 M1块@12 ｜ 10 (题,有gold的篇) 对数 ｜ 11 该对**有 gold 块**进前12 ｜ 12 位次样本
        # 13 **全篇都有 gold 块**的 M1 题数 ｜ 14 单篇最大独占样本（前 12 里同篇最多几块）
        # ⚠️ 布局必须与上面注释一致（0~11 计数、12 位次样本、13 严口径计数…）。
        #    曾经写成 `[0]*14 + [[], []]` → `acc[12]` 是 int，遇到 M1 题就 `append` 崩。
        acc: list = [0] * 12 + [[]] + [0] + [[]] + [0, 0, 0]
        for g in groups:
            sb = Sandbox(g)
            for q in sb.queries:
                qid = q["qid"]
                sel = sb.sel_xling(qid, **kw)
                r = sb.evaluate(qid, sel)
                gd = _gold_of(sb, qid)
                pos = {int(i): k for k, i in enumerate(sel)}
                acc[0] += 1
                acc[1] += int(r["miss"])
                acc[2] += int(r["papers"])
                acc[3] += int(r["chars"])
                acc[6] += sum(1 for i in gd["qchunks"] if i in pos and pos[i] < 12)
                acc[7] += len(gd["qchunks"])
                if "-M1-" in qid:
                    goldp = {sb.pdf_of[i] for i in gd["qchunks"]}
                    topp = {sb.pdf_of[i] for i in sel[:12]}
                    acc[5] += 1
                    acc[4] += 1 if goldp <= topp else 0
                    # 块级位次（M1 子集）：gold 块进前 12 的比例
                    acc[8] += sum(1 for i in gd["qchunks"] if i in pos and pos[i] < 12)
                    acc[9] += len(gd["qchunks"])
                    # **跨篇块的排序效果**：对 (题, 有 gold 的篇) 每个组合，看该篇 gold 块
                    # 的最好位次。分母 = 组合数；命中 = 该篇至少有 1 个 gold 块进前 12；
                    # 位次样本**只取命中时**（否则 10^6 会污染中位数 —— 踩过）。
                    byp: dict[str, list[int]] = {}
                    for i in gd["qchunks"]:
                        byp.setdefault(sb.pdf_of[i], []).append(pos.get(i, 10 ** 6))
                    for v in byp.values():
                        acc[10] += 1
                        if min(v) < 12:
                            acc[11] += 1
                            acc[12].append(min(v))
                    # **严口径**：每个"有 gold 的篇"都有 gold 块进前 12（M1 的真实需求；
                    # `M1篇全` 只要求该篇有**任意块** → 会高估，见 `--prod` 说明）
                    acc[13] += 1 if all(min(v) < 12 for v in byp.values()) else 0
        rows[nm] = acc
    m1n = rows[arms[0][0]][5]
    print(f"\n########## 生产路径回归（{len(groups)} 组，{rows[arms[0][0]][0]} 题，"
          f"其中 **M1 {m1n} 道**；走生产合并函数）")
    print(f"  {'配置':<42}{'gold@12':>9}{'M1块@12':>9}{'**全篇有gold**':>14}{'M1篇全':>8}"
          f"{'篇内命中':>9}{'每篇位次':>9}{'未命中':>7}{'篇覆盖':>8}{'字符':>8}")
    for nm, _ in arms:
        a = rows[nm]
        med = sorted(a[12])[len(a[12]) // 2] if a[12] else -1
        print(f"  {nm:<40}{a[6] / max(a[7], 1):>9.1%}{a[8] / max(a[9], 1):>9.1%}"
              f"{f'{a[13]}/{a[5]}':>14}{f'{a[4]}/{a[5]}':>8}"
              f"{a[11] / max(a[10], 1):>9.1%}{med:>9.1f}"
              f"{a[1]:>7}{a[2] / max(a[0], 1):>8.2f}{a[3] / max(a[0], 1):>8.0f}")
    print("  判据（**块级 + 排序**，跨篇；前四列越大/越小越好）：\n"
          "    gold@12  全部 gold 块进前 12 的比例（块级位次）\n"
          "    M1块@12  M1 子集同一口径\n"
          "    **全篇有gold** M1 里**每个有 gold 的篇都有 gold 块进前 12** 的题数 ← **M1 主判据**\n"
          "    M1篇全   只要求该篇有**任意块**进前 12（**会高估**，仅参考）\n"
          "    篇内命中 对每个 (题, 有 gold 的**篇**) 组合，该篇**至少一个 gold 块**进前 12 的比例\n"
          "    每篇位次 命中时，该篇最好 gold 块的位次中位数（**越小越好** = 跨篇块的排序效果）")
    return 0


def reproduce(groups: list[str]) -> int:
    """与**已记录的实测数字**对表（`_xling_ab.py` 的 A0/A1 臂）。"""
    known = {"A0 中文原问": 11, "A1 英文译式": 5}
    tot = {k: 0 for k in known}
    en_n = 0
    print(f"\n########## 只用沙盒复现（N={PROD['n']}）")
    for group in groups:
        sb = Sandbox(group)
        m0 = m1 = n_en = 0
        for q in sb.queries:
            qid = q["qid"]
            m0 += sb.evaluate(qid, sb.sel_quota(qid, "zh"))["miss"]
            if q["translation"]:
                m1 += sb.evaluate(qid, sb.sel_quota(qid, "en"))["miss"]
                n_en += 1
        en_n += n_en
        n = len(sb.queries)
        print(f"  {group:<8}{n:>4} 题 ｜ A0 中文 {m0:>3}/{n} ｜ A1 英文 {m1:>3}/{n_en}")
        tot["A0 中文原问"] += m0
        tot["A1 英文译式"] += m1
    print(f"\n  {'臂':<14}{'沙盒':>8}{'实测':>8}   一致性")
    for k, v in tot.items():
        print(f"  {k:<14}{v:>8}{known[k]:>8}   {'✅' if v == known[k] else '❌ 不一致'}")
    ok = all(tot[k] == known[k] for k in known)
    return 0 if ok else 1


# ─────────────────────────── 提速验证 ───────────────────────────

def bench(groups: list[str]) -> int:
    ks = (30, 60, 120)
    tops = (12, 24, 32)
    print(f"\n########## 全网格耗时（{len(ks)} k × 2 词法路 × {len(tops)} top_k "
          f"= {len(ks) * 2 * len(tops)} 配置 × 120 题）")
    t0 = time.time()
    n_cfg = n_ev = 0
    for group in groups:
        sb = Sandbox(group)
        for k in ks:
            for use_bm in (True, False):
                ords = {q["qid"]: sb.order(q["qid"], "zh", k=k, use_bm=use_bm)
                        for q in sb.queries}                       # 每个配置重算一次排名
                pers = {q["qid"]: sb.paper_orders(q["qid"], "zh", k=k, use_bm=use_bm)
                        for q in sb.queries}
                for tk in tops:
                    for q in sb.queries:
                        sb.evaluate(q["qid"], sb.quota(ords[q["qid"]], pers[q["qid"]],
                                                       top_k=tk))
                        n_ev += 1
                    n_cfg += 1
    dt = time.time() - t0
    print(f"  {n_cfg} 配置 × 2 组 = {n_cfg // 2} 配置/组 ｜ 判分 {n_ev} 次 ｜ "
          f"耗时 {dt:.1f}s（{dt / max(n_cfg, 1) * 1000:.0f} ms/配置）")

    # BM25 `k1`/`b` 网格：这一项会**重建 BM25 索引**（按 (段, k1, b) 缓存）→ 单独计时
    t1 = time.time()
    n_kb = 0
    for group in groups:
        sb = Sandbox(group)
        for k1 in (1.2, 1.5, 2.0):
            for b in (0.5, 0.75, 1.0):
                for q in sb.queries:
                    sb.order(q["qid"], "zh", k1=k1, b=b)
                    n_kb += 1
    print(f"  BM25 k1×b 网格 9 组 × 120 题 ｜ 全量排名 {n_kb} 次 ｜ "
          f"耗时 {time.time() - t1:.1f}s（含 BM25 索引重建）")
    print("  对照：`_scan_quota.py` 同类网格需**重建池子**（分钟级/组，每次都要重新 encode）")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="group1,group2")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--step", type=int, default=7, help="自证抽样间隔（1 = 全量逐题）")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--prod", action="store_true", help="生产路径回归（新旧默认对比）")
    ap.add_argument("--cap", action="store_true", help="①每篇配额 cap 扫描（硬上限/去 floor）")
    ap.add_argument("--m1audit", action="store_true", help="逐题审：前12篇分布/每篇gold块命中/缺篇")
    ap.add_argument("--ksweep", action="store_true", help="输出大小 K 扫描（跨篇判据）")
    ap.add_argument("--only", default="", help="只看含该子串的 qid（如 G1-M1）")
    ap.add_argument("--union", action="store_true", help="并集的英文补充臂口径对比")
    ap.add_argument("--ablate", action="store_true", help="Step 1：消融矩阵（组件去留）")
    ap.add_argument("--chunk", action="store_true", help="**块级** gold 覆盖（召回到底变好没有）")
    ap.add_argument("--oracle", action="store_true", help="重排上界：完美重排器能带来多少 headroom")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    if not any((args.build, args.selftest, args.reproduce, args.bench)):
        args.build = True
    if args.build:
        for g in groups:
            build(g)
    if args.selftest:
        return 1 if selftest(groups, max(args.step, 1)) else 0
    if args.prod:
        return prodcheck(groups)
    if args.cap:
        return capsweep(groups)
    if args.ksweep:
        return ksweep(groups)
    if args.m1audit:
        base = dict(top_k=16, xling_k=16, merge="inter", w_en=1.0, cover=True)
        return m1audit(groups, [
            ("基线(floor=1 无 cap)", base),
            ("臂内 cap=2 floor=0", {**base, "floor": 0, "cap": 2}),
            ("臂内 cap=3 floor=0", {**base, "floor": 0, "cap": 3}),
            ("floor=0 无 cap", {**base, "floor": 0}),
        ], only=args.only)
    if args.reproduce:
        return reproduce(groups)
    if args.union:
        return union_variants(groups)
    if args.ablate:
        return ablate(groups)
    if args.chunk:
        return chunk_level(groups)
    if args.oracle:
        return oracle(groups)
    if args.bench:
        return bench(groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
