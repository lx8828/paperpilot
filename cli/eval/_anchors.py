"""锚点匹配的**文本归一**（`run_qa_v2` / `run_multi_qa` 共用）。

## 为什么必须归一（2026-09-23 冒烟实测）

答案带 markdown 强调与中英混排空格，naive 子串匹配会把锚点**切断**：

    答案：「根据提供的证据，论文**没有**报告被修复代码的运行时开销…」
    锚点：「没有报告」                      → `"没有报告" in text` = False  ✗ 假 ⚠️

剥掉 `* _ ~ `` ` 后即可命中。两个 runner 原来各自用 `re.search(re.escape(k), text, re.I)`
（大小写不敏感），这里保持一致，只是**匹配前先归一**。

## 口径（与 `_validate_questions.py` 的引文归一分开）

引文核对面对的是**论文原文**（要处理折行连字符），本模块面对的是**模型答案**
（要处理 markdown）。两者的坑不同，故不共用。

    variants(text)  → ('归一后', '再去掉全部空白')
                      第二个变体防中英混排插空格（`MIDR 是…` / `0.1125 → 0.1379`）。

用法：
    from _anchors import hit_all, hit_any
    hit_all(['0.1125'], text)      # 全部命中
    hit_any(['未报告', '没有报告'], text)   # 任一命中
"""
from __future__ import annotations

import re
from typing import Iterable

_MD = re.compile(r"[*_~`]+")      # markdown 强调 / 行内代码符号
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """剥 markdown 强调符 + 折叠空白（不去大小写）。"""
    return _WS.sub(" ", _MD.sub("", text or "")).strip()


def variants(text: str) -> tuple[str, ...]:
    """匹配用文本变体（已 casefold）：① 归一 ② 再去掉全部空白。"""
    n = normalize(text).casefold()
    return (n, _WS.sub("", n))


def hit_any(anchors: Iterable[str], text: str) -> bool:
    """任一锚点命中即 True；锚点列表为空 → True（无约束）。

    ⚠️ 与旧行为一致：锚点为空表示"无该档约束"，不是"必须没有"。
    """
    ks = [normalize(k).casefold() for k in (anchors or []) if k]
    if not ks:
        return True
    for v in variants(text):
        if any(k in v for k in ks):
            return True
    return False


def hit_all(anchors: Iterable[str], text: str) -> bool:
    """全部锚点命中才 True；锚点列表为空 → True。"""
    ks = [normalize(k).casefold() for k in (anchors or []) if k]
    if not ks:
        return True
    vs = variants(text)
    return all(any(k in v for v in vs) for k in ks)


def missing(anchors: Iterable[str], text: str) -> list[str]:
    """没命中的锚点（诊断/落盘用，便于归因"缺哪一条"）。"""
    ks = [k for k in (anchors or []) if k]
    vs = variants(text)
    return [k for k in ks if not any(normalize(k).casefold() in v for v in vs)]
