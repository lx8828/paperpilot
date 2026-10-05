"""语料适配层：把「一批论文的文本块 + 向量」包成 `set_judge` 要的**鸭子契约**。

## 为什么要单独一层

`set_judge.run()` 只依赖鸭子类型（`tests/test_set_judge.py` 已证明）：

    idx.pdfs              论文 id 列表
    idx._doc_chunks()     全部块（每个块要有 `text` / `chunk_id` / …）
    idx.search_hybrid()   单查询检索

而**多路检索**（中文题面 + n 路英文子查询 → 加权 RRF）由 `idx.search_multiroute()` 提供。

生产里这些方法长在 `embedder.ChunkIndex` 上 —— 但它绑在"**已入库的那批论文**"上。
本模块要能对**任意一批论文**（例如"这次检索到的 10 篇"）做同样的事，于是：

    search_multiroute = ChunkIndex.search_multiroute     # ★ 生产方法，**原样借**
    search_hybrid     = ChunkIndex.search_hybrid

★★ 这是**赋值**而不是"照抄一份实现" —— 于是**检索逻辑零分叉**：
   改生产的检索，本模块自动跟着变（判据只放一处）。

## 语料从哪来

从**已切块**的 parquet 读（`docid` / `text` 两列），例如
`retrieval/data/r2dev/prodchunk50/mineru/c0.parquet`（生产切块口径）。
★ 向量用生产的 `embedder.encode_texts` 现场算（bge-m3，本机已缓存）。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from paperpilot.agents import embedder as E


def load_chunks_from_parquet(path: str | Path, n: int | None = None,
                             docids: Iterable[str] | None = None,
                             ) -> tuple[list[str], list[str], list[str]]:
    """读切块 parquet → `(论文 id 列表, 每块所属论文, 每块文本)`。

    `docids`：**显式指定篇集与顺序**（优先级最高）。★ 需要它是因为
        "语料篇集"与"切块文件里的篇集"**不是一回事**：既有评测（`_r2_prod_eval.py`）
        的篇集是 `corpus50` 的 `docid` ∩ 切块 ∩ `pdf_map_all` 的 `ok`，
        **不是**切块 parquet 里的出现顺序。不照那个来，结果与既有数**不可比**。
    `n`：只取**前 n 篇**（= 用户说的"选 N"）。按出现顺序切，保证可复现。
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"切块文件不存在：{p}")
    df = pd.read_parquet(p)
    missing = [c for c in ("docid", "text") if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name} 缺列 {missing}（需要 `docid` / `text`）")

    if docids is not None:
        order = [str(d) for d in docids]
        avail = set(df["docid"].astype(str))
        order = [d for d in order if d in avail]        # 切块里没有的篇自然落空
        if not order:
            raise ValueError(f"给定的 docids 在 {p.name} 里一篇都对不上")
    else:
        # 保持**出现顺序**（不是 sorted）—— 与切块的原始排序一致
        order = []
        seen: set[str] = set()
        for d in df["docid"].astype(str).tolist():
            if d not in seen:
                seen.add(d)
                order.append(d)
        if n is not None:
            if n <= 0:
                raise ValueError(f"n 必须 > 0，收到 {n}")
            order = order[:n]

    keep = set(order)
    owner, texts = [], []
    for d, t in zip(df["docid"].astype(str), df["text"].astype(str)):
        if d in keep:
            owner.append(d)
            texts.append(t)
    if not owner:
        raise ValueError(f"{p.name} 里没读到任何块（n={n}, docids={len(order) if docids else None}）")
    return order, owner, texts


class CorpusIndex:
    """把任意一批「论文块 + 向量」包成 `set_judge` 的鸭子契约。

    ★ 检索方法直接借用生产实现（见模块 docstring）：**不是**复制一份逻辑。
    """

    # ★ 生产方法，原样借 —— 逻辑零分叉
    search_multiroute = E.ChunkIndex.search_multiroute
    search_hybrid = E.ChunkIndex.search_hybrid

    def __init__(self, pdfs: Iterable[str], owner: Iterable[str],
                 texts: Iterable[str], vecs: np.ndarray) -> None:
        self.pdfs = [str(p) for p in pdfs]
        self._owner = [str(d) for d in owner]
        self._texts = [str(t) for t in texts]
        self._chunks = [
            SimpleNamespace(text=t, chunk_id=f"{d}#{i}", title_path=[""],
                            page_span=(1, 1), embed_text=t)
            for i, (t, d) in enumerate(zip(self._texts, self._owner))
        ]
        self._vecs = np.asarray(vecs, dtype="float32")
        if len(self._chunks) != len(self._vecs):
            raise ValueError(f"块数({len(self._chunks)}) 与向量行数({len(self._vecs)}) 不一致")

    # ── 以下为 `ChunkIndex` 内部会用到的方法（鸭子契约的一部分）──

    @classmethod
    def from_parquet(cls, path: str | Path, n: int | None = None,
                     docids: Iterable[str] | None = None,
                     *, cache_dir: str | Path | None = None,
                     verbose: bool = True) -> "CorpusIndex":
        """从切块 parquet 建索引（向量**带缓存**）。

        `docids`：**显式指定篇集与顺序**。★ 与既有评测对齐时必须用它 ——
            既有口径的篇集 ≠ 切块 parquet 里的篇集（见 `load_chunks_from_parquet`）。
            不指定就按切块里的出现顺序取前 `n` 篇。

        ## ⚠️ 编码是这一步的瓶颈 —— 而它 ∝ **语料规模 N**
        实测（RTX 4050）：**50 篇 / 1079 块 → 156 秒**（bge-m3，`max_seq_length=8192`
        是生产值）。据此推算 300 篇 ≈ 3~4 分钟、1000 篇 ≈ 15 分钟。

        ★ 这条很关键，且容易被忽略：**"判官耗时"与 N 无关**（`k×0.8/8`，只看 k），
          但**建索引耗时与 N 成正比**。所以 N 的代价在**准备阶段**，不在判官阶段。

        ★ 因为是**一次性成本**，这里做缓存：同一 `(文件, mtime, n)` 第二次直接读 npz
          （秒级）。"选不同的 N"因此不必反复重编码。

        ## 缓存
        默认落在 `retrieval/data/r2dev/_ma_cache/`（该目录已被 .gitignore 挡住）。
        传 `cache_dir=""` 可关闭缓存。
        """
        import hashlib
        import time

        t0 = time.time()
        pdfs, owner, texts = load_chunks_from_parquet(path, n=n, docids=docids)

        p = Path(path).resolve()
        if cache_dir is None:
            root = Path(__file__).resolve().parents[3]
            cache_dir = root / "retrieval" / "data" / "r2dev" / "_ma_cache"
        cdir = Path(cache_dir) if cache_dir else None
        key = ""
        if cdir is not None:
            # ★ 缓存 key 必须含 **docids**：篇集不同 → 块不同 → 向量不同。
            #   漏了它会让"换了篇集"命中旧缓存，静默用错语料的向量。
            dkey = "" if docids is None else ",".join(str(d) for d in docids)
            raw = (f"{p}|{p.stat().st_mtime_ns}|{p.stat().st_size}|{n}|{dkey}|bge-m3")
            key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        vecs = None
        if cdir is not None:
            f = cdir / f"ma_{key}.npz"
            if f.exists():
                try:
                    vecs = np.load(f)["v"].astype("float32")
                    if len(vecs) != len(texts):      # 文件被换过 → 缓存作废
                        vecs = None
                    elif verbose:
                        print(f"[multianswer] 缓存命中 {f.name}"
                              f"（{time.time() - t0:.1f}s）", flush=True)
                except Exception:  # noqa: BLE001  坏缓存不该弄挂跑批
                    vecs = None

        if vecs is None:
            vecs = np.asarray(E.encode_texts(texts), dtype="float32")
            if cdir is not None:
                cdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cdir / f"ma_{key}.npz", v=vecs)

        if verbose:
            print(f"[multianswer] 语料就绪：{len(pdfs)} 篇 / {len(texts)} 块 "
                  f"（向量 {vecs.shape}）｜ {time.time() - t0:.1f}s", flush=True)
        return cls(pdfs, owner, texts, vecs)

    def _doc_chunks(self) -> list[Any]:
        return self._chunks

    def vectors(self) -> np.ndarray:
        return self._vecs

    def _select(self, chunks: Any, order: Any, top_k: int) -> list[int]:
        """把检索序切成前 `top_k` —— 与生产的选取语义一致。"""
        return list(order)[: min(top_k, len(order))]

    def _hit_extra(self, i: int) -> dict[str, Any]:
        """给命中块附加"来自哪篇/哪节"——`per_paper_hits` 靠它归并到篇。"""
        return {"pdf": self._owner[i], "section": ""}
