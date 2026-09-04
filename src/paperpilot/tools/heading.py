"""论文标题识别（V3 正式版）。

从 tests/_heading_scoring.py 迭代沉淀而来，经 14 篇规范论文验收（含 2 批测试集）：
  - 硬性：References 之后无非附录标题、L1 一级标题全覆盖、正文不产生碎片 chunk
  - 覆盖风格：编号(1 / 1.1 / A)与无编号、全大写 / title case / sentence case、
    单行与多行标题、不同字号（r 1.0x~1.3x）

核心判定：
  1. 弱编号匹配：编号每段 ∈ [1,20]，rest（标题文字）3~120 字符、首字母大写
  2. 多特征打分（字号/编号/独占行/留白/标点/排除前缀），用于阈值判定与排序
  3. 编号标题（L1/L2/L3/APPENDIX）rest 校验通过后强制判定；L1 若为 sentence case 低分则排除
  4. References 之后只允许附录标题；附录标题只可能出现在 References 之后
  5. 页眉/页脚噪声过滤（"2 Inoue et al." 页码+固定文本在 ≥3 页重复）

用法：
    blocks = parse_pdf(...)["blocks"]
    headings = find_headings(blocks)
    # headings: [{block_id, page, kind, no, level, text, score}, ...]（按页序）
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, TypedDict

# ================= 类型 =================


class BlockLine(TypedDict):
    """文本块内的一行。"""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


class BlockDict(TypedDict):
    """parse_pdf 输出的文本块结构。"""
    block_id: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    lines: list[BlockLine]


# 打分中间产物 detail / 标题候选，键值类型混合，用宽松 dict
DetailDict = dict[str, Any]

# ================= 常量 =================

# 特殊标题（无编号）
SPECIAL_SET = {
    "Abstract", "ABSTRACT",
    "References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY",
    "Acknowledgements", "ACKNOWLEDGEMENTS", "Acknowledgments", "ACKNOWLEDGMENTS",
    "Appendix", "APPENDIX",
}

# 参考文献标题（References 隔离边界）
REF_SET = {"References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY"}

# 排除前缀：编号命中但以这些开头的不是 section 标题（定理/图表环境）
EXCLUDE_PREFIX_TUP = tuple(
    w + x
    for w in [
        "Theorem", "Lemma", "Definition", "Corollary", "Proposition",
        "Remark", "Example", "Assumption", "Claim", "Conjecture",
        "Proof", "Notation", "Convention", "Fact", "Note",
        "Table", "Figure", "Fig", "Algorithm",
    ]
    for x in [" ", "."]
)

# 弱编号：1、3.1、4.1.2（允许末尾点）；附录 A./A ~ Z
_RE_WEAK_NO = re.compile(r"^(\d+(?:\.\d+){0,2})(\.)?$")
_RE_WEAK_APP = re.compile(r"^([A-Z])(\.)?$")

# 单个 \n → 空格（多行标题合成单行文本），连续 \n\n 保留
_SINGLE_NL = re.compile(r"(?<!\n)\n(?!\n)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# 阈值：非编号标题（SPECIAL/无编号）判定所需的最低总分
SCORE_THRESHOLD = 7
# L1（纯数字编号）为 sentence case 时所需的最低总分（挡贡献列表/表格说明/页眉）
L1_CASING_FLOOR = 6


# ================= 工具函数 =================

def _is_artword(text: str) -> bool:
    """艺术字：整块由单字母空格分隔组成（"A R T I C L E I N F O"）。只认字母。"""
    words = text.split()
    return bool(words) and all(len(w) == 1 and w.isalpha() for w in words)


def preprocess_text(raw: str) -> str:
    """单换行合并为空格，压缩多余空白。"""
    t = raw.strip()
    t = _SINGLE_NL.sub(" ", t)
    t = _MULTI_SPACE.sub(" ", t)
    return t.strip()


def doc_text_h_median(blocks: list[BlockDict]) -> float:
    """整篇文档正文行高（y1-y0）中位数，LaTeX 正文行高全局一致。"""
    hs = [b["y1"] - b["y0"] for b in blocks]
    hs = [h for h in hs if 7.0 <= h <= 16.0]
    if not hs:
        return 10.0
    hs.sort()
    return hs[len(hs) // 2]


def page_width(blocks: list[BlockDict]) -> float:
    """页面内容宽度。"""
    if not blocks:
        return 500.0
    return max(b["x1"] for b in blocks) - min(b["x0"] for b in blocks)


def _extract_weak_number(first_line: str):
    """弱编号匹配。

    Returns:
        (kind, no_str, score_add):
          kind: 'L1'/'L2'/'L3'/'APPENDIX' 或 None
          score_add: L2/L3 编号 +3，L1 纯数字 +2，附录 +3；rest 不合标题格式返回 None
    """
    if not first_line:
        return None, None, 0
    tokens = first_line.split(maxsplit=2)
    if not tokens:
        return None, None, 0
    t0 = tokens[0].rstrip(":")
    m = _RE_WEAK_NO.match(t0)
    if m:
        no_str = m.group(1)
        parts = [int(x) for x in no_str.split(".")]
        if any(p < 1 or p > 20 for p in parts):
            return None, None, 0
        level = len(parts)
        has_dot = bool(m.group(2))
        rest = first_line[len(t0):].strip()
        if has_dot and rest.startswith("."):
            rest = rest[1:].strip()
        # rest 必须像标题文字：3~120 字符、首字母大写
        if not rest or len(rest) < 3 or len(rest) > 120 or not rest[0].isupper():
            return None, None, 0
        return f"L{level}", no_str, 3 if level >= 2 else 2
    m = _RE_WEAK_APP.match(t0)
    if m:
        letter = m.group(1)
        has_dot = bool(m.group(2))
        rest = first_line[len(t0):].strip()
        if has_dot and rest.startswith("."):
            rest = rest[1:].strip()
        if not rest or len(rest) < 3 or len(rest) > 120 or not rest[0].isupper():
            return None, None, 0
        if _is_artword(rest):
            return None, None, 0
        return "APPENDIX", letter, 3
    return None, None, 0


# ================= 打分 =================

def _score_block(b: BlockDict, text_h_med: float, pg_width: float,
                 prev_b: BlockDict | None, next_b: BlockDict | None,
                 after_refs: bool) -> tuple[int, DetailDict]:
    """六特征打分：字号 / 弱编号 / 独占行 / 上下留白 / 末尾标点 / 排除前缀。"""
    raw = b["text"]
    text = preprocess_text(raw)
    detail: DetailDict = {
        "text": text,
        "multiline": "\n" in raw.strip(),
        "after_refs": after_refs,
    }
    if not text:
        return -99, detail

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first_line = lines[0] if lines else text

    h = b["y1"] - b["y0"]
    width = b["x1"] - b["x0"]
    textlen = len(text)
    detail.update({"h": h, "width": width, "pg_width": pg_width, "len": textlen})

    # 特征 6：排除前缀（一票否决级）
    f6 = 0
    for pref in EXCLUDE_PREFIX_TUP:
        if first_line.startswith(pref):
            f6 = -8
            break
    detail["f6_excl_prefix"] = f6

    # 特殊集合（Abstract/References 等）强分数
    special_bonus = 3 if text in SPECIAL_SET else 0
    detail["special"] = special_bonus

    # 特征 1：字号权重（r≥1.6 大行高 = 公式/图表标签，扣分）
    ratio = h / text_h_med if text_h_med > 0 else 1.0
    if ratio < 0.95:
        f1 = -3
    elif ratio < 1.10:
        f1 = 0
    elif ratio < 1.30:
        f1 = 3
    elif ratio < 1.60:
        f1 = 1
    else:
        f1 = -4
    detail["f1_font"] = (round(ratio, 2), f1)

    # 特征 2：弱编号 + special
    # 先按"合并后首行"提取；若失败且 block 含换行，说明可能是"标题行+正文段"被
    # pymupdf 并成一个 block（如 30938 的 '6.1.3. Experimental Details\nFor Large
    # Language Models, ...'），preprocess 把换行合并导致 rest 超长 → 回退用原始首行重试。
    kind, no_str, f2 = _extract_weak_number(first_line)
    if kind is None and "\n" in raw.strip():
        raw_first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        if raw_first and raw_first != first_line:
            k2, n2, s2 = _extract_weak_number(raw_first)
            if k2 is not None:
                kind, no_str, f2 = k2, n2, s2
    if after_refs and kind != "APPENDIX":
        f2 = -8
    if special_bonus > 0:
        f2 = max(f2, special_bonus)
    elif kind is None:
        f2 = -3  # 无编号且非特殊：正文短块几乎不是标题
    detail.update({"kind": kind, "no_str": no_str, "f2_number": f2})

    # 特征 3：独占一行（短 + 窄）
    f3 = 2 if textlen <= 120 and (pg_width < 1 or width / pg_width < 0.7) else 0
    detail["f3_shortnarrow"] = f3

    # 特征 4：上下留白
    f4_top = f4_bot = 0
    gap_th = 2.0 * text_h_med
    if prev_b is not None and prev_b["page"] == b["page"]:
        if b["y0"] - prev_b["y1"] > gap_th:
            f4_top = 2
    if next_b is not None and next_b["page"] == b["page"]:
        if next_b["y0"] - b["y1"] > gap_th:
            f4_bot = 2
    detail["f4_whitespace"] = (f4_top, f4_bot)

    # 特征 5：末尾标点（冒号结尾扣分，防列表引导句）
    last = text[-1] if text else ""
    if last in ".:：":
        f5 = 0 if last == "." else -2
    else:
        f5 = 2
    detail["f5_endpunct"] = f5

    total = f1 + f2 + f3 + f4_top + f4_bot + f5 + f6
    detail["total"] = total
    return total, detail


# ================= 判定 =================

def _title_words(detail: DetailDict) -> str:
    """去掉编号前缀取标题文字，如 '2 Inoue et al.' → 'Inoue et al.'。"""
    text = detail.get("text") or ""
    parts = text.split(maxsplit=1)
    return parts[1] if len(parts) > 1 else text


def _is_heading(total: int, detail: DetailDict) -> bool:
    kind = detail.get("kind")
    no_str = detail.get("no_str")
    after_refs = detail.get("after_refs")

    # References 之后：只允许附录标题
    if after_refs and kind != "APPENDIX":
        return False
    if kind == "APPENDIX":
        if not after_refs:
            # 附录标题只可能出现在 References 之后（正文里的 "A One individual..." 是字母标签）
            return False
        # 参考文献条目作者列表（"D. Sculley, Gary Holt, ..."）以 "D." 开头被误判为附录
        if _title_words(detail).count(",") >= 2:
            return False

    if kind and no_str:
        rest = _title_words(detail)
        # L1 纯数字编号误报多（贡献列表 "1. A controlled..."、表格说明 "1 The bold text..."、
        # 页眉 "2 Inoue et al."）：sentence case 且低分 → 排除；全大写标题（ICML/AAAI 风格）信任
        if kind == "L1" and not rest.isupper() and total < L1_CASING_FLOOR:
            return False
        if detail.get("multiline"):
            # 多行标题：y1-y0 是整块高度不代表字号，信任编号 + rest 校验
            return True
        f1 = detail["f1_font"][1]
        if f1 >= 0:
            return True

    return total >= SCORE_THRESHOLD


def filter_repeated_headers(headings: list[DetailDict]) -> list[DetailDict]:
    """页眉/页脚过滤：同一标题文字在 ≥3 个不同页重复 → "2 Inoue et al." 类噪声。"""
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for h in headings:
        key = _title_words(h["detail"]).strip()
        if key:
            pages_by_text[key].add(h["page"])
    repeat = {k for k, v in pages_by_text.items() if len(v) >= 3}
    if not repeat:
        return headings
    return [h for h in headings if _title_words(h["detail"]).strip() not in repeat]


# ================= 主入口 =================

def find_headings(blocks: list[BlockDict]) -> list[DetailDict]:
    """识别论文标题。

    Args:
        blocks: parse_pdf 输出的文本块（含 page/x0..y1/text），顺序不限

    Returns:
        按页序排列的标题列表，每项：
            {block_id, page, kind, no, level, text, score}
          kind: 'L1'/'L2'/'L3'/'APPENDIX'/'SPECIAL'
          no: 编号字符串或特殊标题名（如 '3.1' / 'References'）
          level: 层级数字（L1=1，L2=2，L3=3，APPENDIX/SPECIAL=None）
    """
    ordered = sorted(blocks, key=lambda b: (b["page"], b["y0"], b["x0"]))
    by_page: dict[int, list[BlockDict]] = defaultdict(list)
    for b in ordered:
        by_page[b["page"]].append(b)

    text_h_med = doc_text_h_median(ordered)
    page_pg_w = {p: page_width(bs) for p, bs in by_page.items()}

    # References 位置（其后视为参考文献/附录正文区域）
    ref_idx = -1
    for i, b in enumerate(ordered):
        if b["text"].strip() in REF_SET:
            ref_idx = i
            break

    candidates = []
    for i, b in enumerate(ordered):
        prev_b = ordered[i - 1] if i > 0 else None
        next_b = ordered[i + 1] if i < len(ordered) - 1 else None
        after_refs = ref_idx >= 0 and i > ref_idx
        total, detail = _score_block(b, text_h_med, page_pg_w[b["page"]],
                                     prev_b, next_b, after_refs=after_refs)
        detail["after_refs"] = after_refs
        if not _is_heading(total, detail):
            continue
        candidates.append({
            "block_id": b["block_id"],
            "page": b["page"],
            "detail": detail,
        })

    # 页眉过滤
    candidates = filter_repeated_headers(candidates)

    # 输出规整
    result = []
    for c in sorted(candidates, key=lambda x: (x["page"], x["detail"].get("y0", 0))):
        detail = c["detail"]
        kind = detail.get("kind")
        if kind is None:
            kind = "SPECIAL" if detail.get("special") else "UNKNOWN"
        no = detail.get("no_str") or (detail.get("text") or "").strip() or ""
        level = None if kind in ("SPECIAL", "APPENDIX") else int(kind[1:])
        result.append({
            "block_id": c["block_id"],
            "page": c["page"],
            "kind": kind,
            "no": no,
            "level": level,
            "text": detail.get("text", "").strip(),
            "score": detail.get("total", 0),
        })
    return result
