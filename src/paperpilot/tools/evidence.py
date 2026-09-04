"""Claims 的 evidence 回原文核对（纯函数，无 LLM）。

ev 三档：
    hit   逐字/弱命中（归一大小写/引号/空白后为原文子串；
          弱命中：断行连字符/数学表格符号字形差异，剥离非字母数字后可命中）
    loose 省略式摘录（带 ... 且分段均可在原文找到）
    miss  未回原文（可能改写/跨段/符号差异，需人工看）

run_claims / run_summary / viewer 共享此模块。
"""
from __future__ import annotations

import re

from paperpilot.models.schema import Claim

# 弯引号/破折号等排版差异归一到 ASCII（PDF 提取与 LLM 输出常见此噪声）
_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
})
# 强归一：剥离一切非字母数字（吸收断行连字符/数学符号字形/表格符号等噪声）
_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_text(s: str) -> str:
    """归一化：去大小写、弯引号转直引号、空白折叠为单空格。"""
    return " ".join(s.translate(_QUOTE_MAP).casefold().split())


def alnum_norm(s: str) -> str:
    """强归一：剥离所有非字母数字（用于吸收字形/断行/符号噪声）。"""
    return _ALNUM.sub("", s)


def ev_segments(q: str) -> list[str]:
    """把 evidence 按省略号(…)拆成若干段，用于判断"省略式摘录"。"""
    return [p for p in norm_text(q).split("...") if p.strip()]


def evidence_state(claim: Claim, chunk_text_map: dict[str, str]) -> str:
    """单条 claim 的 evidence 状态：hit / loose / miss。"""
    src = norm_text(chunk_text_map.get(claim.chunk_id, ""))
    nq = norm_text(claim.evidence_quote)
    if not nq:
        return "miss"
    if nq in src:
        return "hit"
    if alnum_norm(nq) and alnum_norm(nq) in alnum_norm(src):
        return "hit"  # 字符级噪声：本质是原文
    segs = ev_segments(claim.evidence_quote)
    if len(segs) > 1 and all(s in src for s in segs):
        return "loose"
    return "miss"


def verify_evidence(claims: list[Claim],
                    chunk_text_map: dict[str, str]) -> tuple[list[Claim], list[Claim], list[Claim]]:
    """核对 claims 的 evidence 能否回原文。返回 (hit, loose, miss)。"""
    hit: list[Claim] = []
    loose: list[Claim] = []
    miss: list[Claim] = []
    for c in claims:
        st = evidence_state(c, chunk_text_map)
        if st == "hit":
            hit.append(c)
        elif st == "loose":
            loose.append(c)
        else:
            miss.append(c)
    return hit, loose, miss


# ── Claim <-> dict 序列化（落盘/读取共用）──


def claim_to_dict(c: Claim) -> dict:
    return {
        "claim_id": c.claim_id,
        "type": c.type,
        "text": c.text,
        "evidence_quote": c.evidence_quote,
        "chunk_id": c.chunk_id,
        "title_path": c.title_path,
        "page": c.page,
        "page_span": list(c.page_span),
    }


def dict_to_claim(d: dict) -> Claim:
    return Claim(
        claim_id=d["claim_id"], type=d["type"], text=d["text"],
        evidence_quote=d["evidence_quote"], chunk_id=d["chunk_id"],
        title_path=list(d["title_path"]), page=d["page"],
        page_span=tuple(d["page_span"]),
    )
