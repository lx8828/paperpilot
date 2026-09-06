"""query 改写（方案2）：把原始问题改写成多个检索友好的变体。

背景（QASPER 分析）：大量 fail 题的原始问题含指代（their baseline / first experiment /
this corpus）或问法含糊，bge 检索按字面匹配会偏。改写产出多个变体 → 分别检索 → RRF 融合，
可把"答案段落被低排"的题拉进 topK。

改写不依赖论文内容（成本约束：一次 LLM 调用），只做"问题理解 + 检索词扩展"：
- 补全/去指代（能判断时）
- 拆解复合问
- 换用论文里常见的表述（"compare against" / "is used for" / "what dataset"）

失败降级：任何异常返回 [原问题]。
"""
from __future__ import annotations

import json
import re
from typing import Any

from paperpilot.tools import llm

_SYS = (
    "你是检索词改写助手。给定一个论文问答问题，生成 2~3 个**用于向量检索**的变体。\n"
    "改写目标：让检索更容易命中论文原文中回答该问题的段落。技巧：\n"
    "1. 若问题含指代（their/this/it 等），尝试还原成具体对象（如 'their baseline methods'\n"
    "   → 'baseline methods used in the experiments'）；还原不了就保留原词。\n"
    "2. 若问题含两个子问（'what is X and how does it work'），拆成独立的子问题。\n"
    "3. 换成论文正文常见表述（如 'compare to'→'compared with/outperform/against'）。\n"
    "4. 保留原问题中的专名、数字、术语，不要改丢。\n"
    '只输出 JSON 字符串数组，如 ["变体1", "变体2"]；最多 3 个，不要多余文字。'
)

_TPL = "原始问题：{question}\n请输出检索变体 JSON 数组。"


def _try_json(text: str) -> list[str] | None:
    text = text.strip()
    # 去掉可能的 markdown 围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\[.*\]", text, flags=re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
    if isinstance(obj, list):
        out = [str(x).strip() for x in obj if str(x).strip()]
        return out[:3]
    return None


def expand_queries(question: str) -> list[str]:
    """返回 [原问题] + 改写变体（去重、保序）。失败只回 [原问题]。"""
    try:
        raw = llm.chat_json(_SYS, _TPL.format(question=question), temperature=0.3)
    except Exception:  # noqa: BLE001
        return [question]
    # chat_json 已解析；raw 可能是 list / dict / 意外 str
    if isinstance(raw, list):
        variants = [str(x).strip() for x in raw if str(x).strip()]
    else:
        variants = _try_json(json.dumps(raw, ensure_ascii=False)
                             if not isinstance(raw, str) else raw)
    if not variants:
        return [question]
    out = [question]
    for v in variants:
        v = v.strip()
        if v and v != question and v not in out:
            out.append(v)
    return out
