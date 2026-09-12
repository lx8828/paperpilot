"""目标表定位口径 v2：**编号 ∪ 表体内容/数值**（联合判定）。

## 为什么要改

旧口径只有一条：用证据行里的 `Table N` 去匹配「**块文本首行**的 `Table N`」。
它在两类真实情况下会失败（`qa/recall/MINERU_COVERAGE_20260912.md` §5）：

1. **编号错位**（上游 MinerU 行为；可辩护的约 2~3/33 篇，另有若干是匹配器假象）；
2. **caption 缺失 / 被上游拼了噪声**（16.7% 表块无 caption；少数块 caption 里混进了
   图的标题或表头行文字）→ 编号取不到，或取到**错的号**。

后果是双向的：
- **假阳性**：号对错时，目标表被定位到**另一张表** → 指标可能"命中"了错的块；
- **假阴性**：号取不到时，该题被当作"该篇没产出目标表"而**从分母里丢掉**（n 会变化）。

## 新口径

目标表块 = 满足**任一**条件：
  a. **编号命中**：块文本前若干字符里出现的 `Table/Figure N` ∩ 证据里的编号
     （解析版文本已清洗 caption 噪声 → 这里的编号比旧口径可靠）；
  b. **内容命中**：块文本含**答案里的显著数值**（长度 ≥3 的数值 token），
     或短答案（≤60 字、无数字）的实词**全部**命中。

两种口径都能跑：`mode="num"`（旧）/ `mode="or"`（新，默认），便于看清"改口径本身"
带来了多少变化。

⚠️ 这是**评测口径**，只改测量、不改生产：线上检索不依赖编号（`cites` 是 chunk 锚点）。
"""
from __future__ import annotations

import re
from typing import Any

try:                                        # 允许在无 src 路径时退化（纯文本脚本）
    from paperpilot.tools.mineru_bridge import EXT_CHUNK_PREFIX
except Exception:                           # noqa: BLE001
    EXT_CHUNK_PREFIX = "xtbl-"

KEY_RE = re.compile(r"(table|figure)\s*([IVXLC]{1,7}|\d+[A-Za-z]?)", re.I)
EV_RE = re.compile(r"\s*(?:FLOAT SELECTED:)?\s*(table|figure)\s*([A-Za-z]?\d+)", re.I)
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
         "ix": 9, "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15,
         "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20, "xxi": 21, "xxii": 22}
HEAD = 300          # 只看块头：caption/编号都在开头，避免表体内的偶然命中


def _norm(n: str) -> str:
    t = str(n).lower()
    return str(ROMAN[t]) if t in ROMAN else (re.sub(r"[^0-9]", "", t) or t)


def keys_in(text: str, head: int = HEAD) -> set[str]:
    """块文本（前 `head` 字）里出现的 `tableN`/`figureN` 编号集合。"""
    out: set[str] = set()
    for m in KEY_RE.finditer(str(text)[:head]):
        kind, num = m.group(1).lower(), _norm(m.group(2))
        if num:
            out.add(f"{kind}{num}")
    return out


def gold_keys(evs: list[str]) -> set[str]:
    """证据行里的 `Table/Figure` 编号（= 旧口径的 `want`）。"""
    out: set[str] = set()
    for e in evs or []:
        m = EV_RE.match(str(e))
        if m:
            out.add(f"{m.group(1).lower()}{m.group(2).lower()}")
    return out


def gold_numbers(gold: str | None) -> set[str]:
    """答案里的**显著**数值 token（≥3 字符），用于内容命中。

    阈值 3 是为了滤掉 "1"/"12" 这类会大面积误命的短数（表格里到处是）。
    """
    if not gold:
        return set()
    out: set[str] = set()
    for t in re.findall(r"\d+(?:[.,]\d+)*%?", str(gold)):
        t = t.rstrip("%").replace(",", "")
        if len(t) >= 3:
            out.add(t)
    return out


def _content_hit(text: str, gold: str | None, nums: set[str]) -> bool:
    t = str(text)
    if any(n in t for n in nums):
        return True
    g = (gold or "").strip()
    # 短答案（无数字）：要求其实词**全部**出现 —— 宁可漏，不可误
    if 0 < len(g) <= 60 and not re.search(r"\d", g):
        ws = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", g)][:6]
        if ws and all(w.lower() in t.lower() for w in ws):
            return True
    return False


def locate(chunks: list[Any], evs: list[str], gold: str | None,
           mode: str = "or", ext_only: bool = True) -> tuple[set[str], dict[str, set[str]]]:
    """返回 (目标块 id 集合, 诊断)。

    诊断键：`by_num`（编号命中）、`by_cnt`（内容命中）、`keys`（证据编号）、
    `nums`（答案显著数值）——用于统计"这次是靠什么定位到的"。
    """
    keys = gold_keys(evs)
    nums = gold_numbers(gold)
    tgt: set[str] = set()
    by_num: set[str] = set()
    by_cnt: set[str] = set()
    for c in chunks:
        cid = str(getattr(c, "chunk_id", "") or "")
        if ext_only and not cid.startswith(EXT_CHUNK_PREFIX):
            continue
        text = str(getattr(c, "text", "") or "")
        a = bool(keys_in(text) & keys)
        b = _content_hit(text, gold, nums)
        if a:
            by_num.add(cid)
        if b:
            by_cnt.add(cid)
        if a or (mode == "or" and b):
            tgt.add(cid)
    return tgt, {"by_num": by_num, "by_cnt": by_cnt, "keys": keys, "nums": nums}


def locate_idx(texts: list[str], evs: list[str], gold: str | None, mode: str = "or",
               ext_mask: list[bool] | None = None) -> tuple[set[int], dict[str, set[int]]]:
    """同上，但输入是**纯文本列表**（离线脚本用），返回索引集合。"""
    keys = gold_keys(evs)
    nums = gold_numbers(gold)
    tgt: set[int] = set()
    by_num: set[int] = set()
    by_cnt: set[int] = set()
    for i, text in enumerate(texts):
        if ext_mask is not None and not ext_mask[i]:
            continue
        a = bool(keys_in(text) & keys)
        b = _content_hit(text, gold, nums)
        if a:
            by_num.add(i)
        if b:
            by_cnt.add(i)
        if a or (mode == "or" and b):
            tgt.add(i)
    return tgt, {"by_num": by_num, "by_cnt": by_cnt, "keys": keys, "nums": nums}
