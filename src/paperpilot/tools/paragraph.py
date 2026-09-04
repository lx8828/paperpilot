"""段落检测（正文段落切分）。

从 PDF 文本块的**行级**信息中恢复段落边界，供超大 chunk 的二级切分使用。
段落边界信号：
  1. 段间距：相邻行垂直间距明显大于正常行距（28433 等无缩进论文）
  2. 首行缩进：行 x0 相对所在列起点明显右移（30023/30938 等缩进论文）

处理规则：
  - 按页、按列（x0 聚类）分别处理，避免双栏串行
  - 已知页眉/页脚噪声行（Page X of Y、Preprint、arXiv 行等）不参与段落
  - 跨页段落自然断开（页边界视为段边界）

用法：
    paras = split_paragraphs(ordered_blocks)
    # paras: [{text, page, y0, rows: [...]}, ...]
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, TypedDict

from paperpilot.tools.heading import BlockDict

# ================= 噪声行过滤（轻量，不追求完整页眉处理） =================
# 明确的页眉/页脚/无意义行，不参与段落切分
NOISE_LINE_RE = [
    re.compile(r"^Page\s+\d+\s+of\s+\d+", re.IGNORECASE),   # Page 12 of 18
    re.compile(r":\s*Preprint submitted to", re.IGNORECASE),  # Elsevier 页眉
    re.compile(r"^Preprint$", re.IGNORECASE),
    re.compile(r"^arXiv:\d+\.\d+"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),                     # 日期
    re.compile(r"^Page\s+\d+$", re.IGNORECASE),
]

# 列聚类容差：x0 距离 ≤12 视为同一列
_COL_TOL = 12.0
# 首行缩进判定阈值（相对列起点）
_INDENT_MIN = 8.0
# 段间距判定：> max(正常行距 × 系数, 下限)
_GAP_FACTOR = 2.5
_GAP_FLOOR = 4.0


def _is_noise_line(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    return any(r.match(t) for r in NOISE_LINE_RE)


class _Row(TypedDict):
    block_id: str
    y0: float
    y1: float
    x0: float
    x1: float
    text: str


class Paragraph(TypedDict):
    page: int
    y0: float
    text: str
    rows: list[_Row]


def _split_column(col: list[_Row], page: int) -> list[Paragraph]:
    col = sorted(col, key=lambda r: r["y0"])
    col_min_x = min(r["x0"] for r in col)

    # 正常行距：行间 gap 中位数（取 0.3~12 区间，排除段间/页间隙）
    gaps = []
    for i in range(1, len(col)):
        g = col[i]["y0"] - col[i - 1]["y1"]
        if 0.3 <= g <= 12:
            gaps.append(g)
    norm_gap = gaps[len(gaps) // 2] if gaps else 2.0
    gap_th = max(norm_gap * _GAP_FACTOR, _GAP_FLOOR)

    paras: list[Paragraph] = []
    cur: list[_Row] = []
    for r in col:
        if not cur:
            cur.append(r)
            continue
        gap = r["y0"] - cur[-1]["y1"]
        indent = r["x0"] - col_min_x
        prev_x = cur[-1]["x0"]
        is_start = (
            gap > gap_th
            or (indent >= _INDENT_MIN and prev_x < r["x0"] - _INDENT_MIN)
        )
        if is_start:
            paras.append(_mk_para(cur, page))
            cur = [r]
        else:
            cur.append(r)
    if cur:
        paras.append(_mk_para(cur, page))
    return paras


def _mk_para(rows: list[_Row], page: int) -> Paragraph:
    text = "\n".join(r["text"].rstrip() for r in rows).strip()
    return {"page": page, "y0": rows[0]["y0"], "text": text, "rows": rows}


def split_paragraphs(blocks: list[BlockDict]) -> list[Paragraph]:
    """把文本块切成段落。

    Args:
        blocks: 文本块列表（顺序不限，内部按页/y 重排）

    Returns:
        按 (页, y) 排序的段落列表，每段 {page, y0, text, rows}。
        页眉/页脚噪声行被跳过；跨页段落自然断开。
    """
    # 收集行，按页分组
    by_page: dict[int, list[_Row]] = defaultdict(list)
    for b in blocks:
        page = b["page"]
        for ln in b.get("lines", []):
            t = ln["text"].strip()
            if _is_noise_line(t):
                continue
            by_page[page].append({
                "block_id": b["block_id"],
                "y0": ln["y0"], "y1": ln["y1"],
                "x0": ln["x0"], "x1": ln["x1"], "text": t,
            })

    paras: list[Paragraph] = []
    for page in sorted(by_page):
        rows = by_page[page]
        rows.sort(key=lambda r: r["x0"])
        # 列聚类
        cols: list[list[_Row]] = []
        for r in rows:
            if cols and r["x0"] - cols[-1][-1]["x1"] <= _COL_TOL:
                cols[-1].append(r)
            else:
                cols.append([r])
        for col in cols:
            paras.extend(_split_column(col, page))

    paras.sort(key=lambda p: (p["page"], p["y0"]))
    return paras
