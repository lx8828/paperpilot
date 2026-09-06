"""QASPER 数据源适配：从 QASPER full_text 确定性构造 Chunk / 论文上下文。

QASPER（allenai）给出论文的章节化全文（full_text = [{section_name, paragraphs}]），
section_name 用 " ::: " 表示层级（如 "Proposed Method ::: Polarity Function"）。
本模块把每篇论文转成与 PDF 产物同构的数据：
    Chunk[]          → claims 提取 / L2 / L3 检索共用（chunk_id 可复现）
    qas 题目         → 评测时按 paper 拉取
    gold evidence    → 引用核对参考

设计约束（与 pipeline 一致性）：
    - title_path 顶层 = section 根名（如 "Introduction"），与分析器 _skip_chunk 兼容
    - 超大 section（>MAX_CHUNK_LEN）按段落二级切分（part 编号），与 chunker 同策略
    - chunk_id 形如 "c001"（全局自增），同一 paper 恒定 → 缓存/向量索引可复用
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paperpilot.models.schema import Chunk, ChunkSpan

ROOT = Path(__file__).resolve().parents[2]
QASPER_JSON = ROOT / "qasper_data" / "json" / "qasper-train-v0.3.json"

MAX_CHUNK_LEN = 4000          # 与 chunker.DEFAULT_MAX_LEN / pipeline.MAX_LEN 一致
REF_MARKERS = ("References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY")

# S2ORC 引用占位符（"BIBREF0" / "FIGREF1"）→ 归一，避免进 LLM 上下文
_REF_RE = re.compile(r"\b[A-Z]+REF\d+\b")
_FIG_RE = re.compile(r"\bFIGURE\d+\b|\bTABLE\d+\b", re.I)


def _clean_para(p: str) -> str:
    """段落清洗：折叠空白、剥离 S2ORC 占位引用（保留原文主体）。"""
    t = " ".join(p.split())
    t = _REF_RE.sub("", t)
    t = _FIG_RE.sub("", t)
    return " ".join(t.split())


def _split_title(name: str) -> list[str]:
    """'Proposed Method ::: Polarity Function' → ['Proposed Method', 'Polarity Function']。"""
    return [x.strip() for x in name.split(":::") if x.strip()]


def load_papers() -> dict[str, dict[str, Any]]:
    """加载 QASPER train 全量（paper_id → paper dict）。"""
    if not QASPER_JSON.exists():
        raise FileNotFoundError(f"缺 QASPER 数据: {QASPER_JSON}")
    return json.loads(QASPER_JSON.read_text(encoding="utf-8"))


def build_chunks(paper: dict[str, Any]) -> list[Chunk]:
    """从 QASPER full_text 确定性构造 Chunk[]（与 claims/QA 检索共用）。

    - 每 section 至少一个 chunk（含标题行文本）
    - 无 section_name 的前置段归 "(PREAMBLE)"（与分析器过滤规则一致）
    - 超长 section 按段落二级切分，共享 title_path，part="1/2"
    - chunk_id 全局自增 c001..；同一 paper 恒定
    """
    chunks: list[Chunk] = []
    full = paper.get("full_text") or []
    seq = 1
    for sec in full:
        parts = _split_title(sec.get("section_name") or "")
        title_path = parts or ["(PREAMBLE)"]
        paras = [_clean_para(p) for p in (sec.get("paragraphs") or []) if _clean_para(p)]
        if not paras:
            continue
        # 标题 + 段落拼成 raw 文本，按 MAX_CHUNK_LEN 段落累加切分
        raw = "\n".join(paras)
        if len(raw) <= MAX_CHUNK_LEN:
            chunks.append(Chunk(
                chunk_id=f"c{seq:04d}",
                title_path=list(title_path),
                page_span=(0, 0),
                text=raw,
                n_blocks=len(paras),
                spans=[ChunkSpan(block_id=f"c{seq:04d}", page=0, text=raw[:200])],
            ))
            seq += 1
            continue
        # 超长：段落级二级切分（子块共享 title_path）
        cur: list[str] = []
        cur_len = 0
        n_sub = 0
        for para in paras:
            if cur and cur_len + len(para) + 1 > MAX_CHUNK_LEN:
                n_sub += 1
                chunks.append(Chunk(
                    chunk_id=f"c{seq:04d}",
                    title_path=list(title_path),
                    page_span=(0, 0),
                    text="\n".join(cur),
                    n_blocks=len(cur),
                    part=f"{n_sub}/?",
                    spans=[ChunkSpan(block_id=f"c{seq:04d}", page=0,
                                     text="\n".join(cur)[:200])],
                ))
                seq += 1
                cur = []
                cur_len = 0
            cur.append(para)
            cur_len += len(para) + 1
        if cur:
            n_sub += 1
            chunks.append(Chunk(
                chunk_id=f"c{seq:04d}",
                title_path=list(title_path),
                page_span=(0, 0),
                text="\n".join(cur),
                n_blocks=len(cur),
                part=f"{n_sub}/?",
                spans=[ChunkSpan(block_id=f"c{seq:04d}", page=0,
                                 text="\n".join(cur)[:200])],
            ))
            seq += 1
    # 回填 part 总数（"1/?") → 确定性"1/N"
    per_path: dict[tuple[str, ...], list[int]] = {}
    for i, c in enumerate(chunks):
        if c.part:
            per_path.setdefault(tuple(c.title_path), []).append(i)
    for path, idxs in per_path.items():
        for j, i in enumerate(idxs, 1):
            chunks[i].part = f"{j}/{len(idxs)}"
    return chunks


def gold_answer(q: dict[str, Any]) -> tuple[str | None, str | None]:
    """取该题最可信的人工答案 → (答案文本, 证据段落)。

    优先级：free_form_answer > yes_no > extractive_spans 拼装。
    多 answerer 取第一个非 unanswerable；evidence 取对应答案的证据段（首段）。
    返回 (answer_text, evidence_text)；unanswerable 返回 (None, None)。
    """
    for a in q.get("answers") or []:
        inner = a.get("answer") or {}
        if inner.get("unanswerable"):
            continue
        if inner.get("free_form_answer"):
            return (str(inner["free_form_answer"]).strip(),
                    (inner.get("evidence") or [None])[0] or None)
        if inner.get("yes_no") is not None:
            return ("yes" if inner["yes_no"] else "no",
                    (inner.get("evidence") or [None])[0] or None)
        spans = inner.get("extractive_spans") or []
        if spans:
            return (" ".join(str(s) for s in spans).strip(),
                    (inner.get("evidence") or [None])[0] or None)
    return None, None
