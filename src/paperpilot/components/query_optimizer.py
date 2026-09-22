"""查询侧优化（Query-Side Optimization）：把"用户问句"变成"检索用查询集"。

**架构（重要，别做成串行五步）**：L1 是**路由器**，L2~L5 是**并列的可选变换器**。
串行跑满五步 = 5 次 LLM 调用 + 互相矛盾的重写（重述过的句子再分解、分解后每段再 HyDE…），
成本翻倍且各变换会互相稀释。正确形态是"先分类 → 按类别选 1~2 个变换器"。

五级：

| 级 | 函数 | 作用 | 状态 |
|---|---|---|---|
| L1 | `classify()` | 意图识别与问题分类 → **路由**（决定跑哪几级） | 未实现（见该函数 docstring） |
| L2 | `restate()` | 问题重述：口语化 → 检索友好 | 未实现 |
| **L3** | **`expand()`** | **查询改写与扩展：补同义/上位/领域表述** | ✅ **已实现**（原 `pull_chunk._rewrite_queries`） |
| L4 | `hyde()` | HyDE：生成"假设答案文档"，用其向量检索 | 未实现 |
| L5 | `decompose()` | 查询分解与多查询生成：高概念密度查询拆子查询 | 未实现 |

**融合策略（这是本组件存在的核心教训）**

旧实现把"原问题 + 变体"做**等权 RRF**，两次 A/B 都是**负收益**。诊断结论是：

    **不是"改写无用"，是"RRF 平权融合有害"** —— 偏题变体会稀释原问题的正确排序
    （与"融合低于单路 oracle"同因）。

故本模块**只负责产出查询集**，并给出三种融合原语，由检索侧按场景选择：

    equal_rrf     原问题与变体等权 RRF              ← 旧默认，**仅作对照（已证伪）**
    quota_union   原问题取 k1 + 各变体各取 k2 的并集   ← 推荐：变体只做召回补充
    rerank_orig   变体只扩召回，再用**原问题**重排     ← 推荐（未实现，见 TODO）

**证据（沿用，勿删）**

    2026-09-07（40 题，同裁判）  ：ON pass 9  vs OFF 12 （救 2 / 坏 5）
    2026-09-10（100 题配对）     ：ON 68%     vs OFF 71%（救 3 / 坏 6），calls/题 +30%
    → **默认全关**。多篇/语料库场景召回面大、单查询带偏风险高时重开，**届时必须重新 A/B**。
    （参考：混合检索在单篇上"无增益"、在 63k 语料上变成 +6.7pt 显著 —— 规模会改变结论，
      所以改写在大语料上重测是有意义的，但不能预设它会翻盘。）

**开关**

    PAPERPILOT_QUERY_LEVELS=l3          # 只跑 L3（等价于旧的 PAPERPILOT_QUERY_REWRITE=1）
    PAPERPILOT_QUERY_LEVELS=l2,l3,l5    # 跑多级（变换器结果按序并入计划）
    PAPERPILOT_QUERY_REWRITE=1          # 向后兼容：等价于 levels=l3
    未设置 / 空                          # 全关，只回原问题（**行为与旧默认完全一致**）

启用**尚未实现**的级别会抛 `NotImplementedError` —— 这是刻意的：
宁可响亮失败，也不要"开了却什么也没发生"的沉默失败（教训见 worker 那条：吞异常最贵）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from paperpilot.tools import llm

LEVELS: tuple[str, ...] = ("l1", "l2", "l3", "l4", "l5")

# 融合策略枚举（稳定公开）
FUSION_EQUAL_RRF = "equal_rrf"
FUSION_QUOTA_UNION = "quota_union"
FUSION_RERANK_ORIG = "rerank_orig"
FUSION_NAMES = (FUSION_EQUAL_RRF, FUSION_QUOTA_UNION, FUSION_RERANK_ORIG)

L3_MAX_VARIANTS = 2          # 原问题 + 最多 2 条变体（与旧实现一致）
L3_MAX_QUERY_CHARS = 200     # 变体长度上限（旧实现同）


# ─────────────────────────── 计划对象 ───────────────────────────
@dataclass
class QueryPlan:
    """一次查询侧优化的结果。检索侧只需读 `.queries`。"""

    original: str
    intent: str = ""                       # L1
    restated: str = ""                     # L2
    variants: list[str] = field(default_factory=list)      # L3
    hyde_doc: str = ""                     # L4
    subqueries: list[str] = field(default_factory=list)    # L5
    levels: tuple[str, ...] = ()           # 本次实际启用的级别
    fusion: str = FUSION_EQUAL_RRF         # 融合**意图**（由检索侧执行）
    max_queries: int = 3

    @property
    def primary(self) -> str:
        """主查询：重述优先，否则原问句（融合时主查询权重最高）。"""
        return self.restated or self.original

    @property
    def queries(self) -> list[str]:
        """送去检索的查询列表：主查询在前，其余按 L5→L3 顺序，去重保序，截到 max_queries。"""
        out: list[str] = []
        for q in [self.primary, *self.subqueries, *self.variants]:
            s = str(q or "").strip()
            if s and s not in out:
                out.append(s)
        return out[: self.max_queries] if self.max_queries else out

    @property
    def multi(self) -> bool:
        """是否产生了多查询（决定检索侧走单查询还是多查询路径）。"""
        return len(self.queries) > 1

    def summary(self) -> dict[str, Any]:
        """写进 debug 的紧凑摘要（可观测性：知道这次到底做了什么）。"""
        return {"levels": list(self.levels), "fusion": self.fusion,
                "n_queries": len(self.queries), "multi": self.multi,
                "has_intent": bool(self.intent), "has_hyde": bool(self.hyde_doc)}


# ─────────────────────────── 开关 ───────────────────────────
def enabled_levels() -> tuple[str, ...]:
    """读取启用的级别（未配置 → 空，即全关）。"""
    raw = os.environ.get("PAPERPILOT_QUERY_LEVELS", "").strip()
    if not raw and os.environ.get("PAPERPILOT_QUERY_REWRITE", "0") == "1":
        raw = "l3"                                    # 向后兼容旧开关
    return tuple(x for x in (s.strip().lower() for s in raw.split(",")) if x in LEVELS)


# ─────────────────────────── L1 意图识别与问题分类 ───────────────────────────
def classify(question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """L1：意图识别与问题分类 —— **未实现**。

    设计（待实现，先写清约束，避免走 Router 的老路）：
      · 输出应当是**路由信号**而不是最终决定：如 `{intent, needs_decompose, needs_hyde,
        needs_history, has_reference}`，由 `optimize()` 按它挑选后续变换器；
      · **绝不做"分类定生死"**（`RAG_COMPONENT_NOTES §2` 的教训：LLM 预测当路由，
        L2 时代 judge 判够精度仅 ≈61%）—— 每级都要有"判错可降级"的兜底；
      · **可用的弱真值**：LitSearch 自带 `specificity`（0/1）与 `query_set`（4 类来源），
        可以当意图分类的先验/评测对象，先量化"分类准不准"再让它参与路由；
      · 针对本项目的实际信号：查询含指代（their/this/it）、含 "and" 多条件、
        含专名/缩写/数字（→ 走 L3 术语展开而非 L5 分解）。
    """
    raise NotImplementedError(
        "L1 意图识别与问题分类未实现；请先只启用已实现的 l3（PAPERPILOT_QUERY_LEVELS=l3）")


# ─────────────────────────── L2 问题重述 ───────────────────────────
def restate(question: str, history: list[dict[str, Any]] | None = None) -> str:
    """L2：把口语化提问重述成检索友好表示 —— **未实现**。

    设计：与 L3 的区别是"**不改语义方向、只去噪**"（去虚词、补被省略的主语；
    多人称指代用 history 还原）。预期收益在**口语化/带指代**的查询上；
    对 LitSearch 这种"已由专家审核过的高质量查询"预期收益很小 ——
    **先分层看有多少查询属于"口语化"再决定值不值得做**。
    """
    raise NotImplementedError(
        "L2 问题重述未实现；请先只启用已实现的 l3（PAPERPILOT_QUERY_LEVELS=l3）")


# ─────────────────────────── L3 查询改写与扩展（已实现）───────────────────────────
_L3_SYS = (
    "你是检索查询改写器。给定一个面向学术论文的问题，生成 2~3 条**仅供检索**的查询变体。\n"
    "规则：\n"
    "1. 抽取并保留关键实体（数据集/语料名、模型/方法名、指标、语言、数字等），这是检索的锚；\n"
    "2. 补同义/上位/常见论文表述（如 'how many utterances' → 'corpus size / number of utterances'；\n"
    "   'is the dataset multilingual?' → 'dataset language composition'）；\n"
    "3. 若问题是英文/面向英文论文，变体用英文；口语或长句改短、去虚词；每条 ≤12 词；\n"
    "4. 变体必须与问题同一语义方向，不要自创新问题。\n"
    '只输出 JSON：{"queries": ["...", "...", "..."]}'
)


def expand(question: str, n: int = L3_MAX_VARIANTS) -> list[str]:
    """L3：生成查询变体（**不含原问题**）。任何异常 → 空列表（不影响主链路）。

    这是**唯一已实现**的一级，逻辑与原 `pull_chunk._rewrite_queries` 一致
    （一次 LLM 调用、保留实体为锚、≤200 字符、与原问题去重）。
    """
    try:
        obj = llm.chat_json(_L3_SYS, f"问题：{question}", temperature=0.2)
    except Exception:  # noqa: BLE001（改写失败不影响主链路）
        return []
    if not isinstance(obj, dict):
        return []
    out: list[str] = []
    for q in (obj.get("queries") or []):
        s = str(q).strip()
        if s and len(s) <= L3_MAX_QUERY_CHARS and s.lower() != question.lower() and s not in out:
            out.append(s)
    return out[:n]


# ─────────────────────────── L4 HyDE ───────────────────────────
def hyde(question: str) -> str:
    """L4：生成"假设答案文档"，用其向量去检索 —— **未实现**。

    设计要点/风险（必须先想清再写）：
      · 收益前提是**答案文档与目标文档在语义空间邻近**。本项目的文档是
        `title + abstract`（中位 936 字符），而"假设答案"通常更长 →
        **长度错配可能让 HyDE 反而变远**，上线前必须实测（可用同一套指标链路）；
      · 成本是"生成 1 次 + 编码 1 次"（比 L3 贵），且**无法与原问题简单融合**——
        它的用法是"替换查询向量"或"作为额外一路召回"，两种都该单独 A/B；
      · 在 LitSearch 上建议先测"HyDE 向量 vs 原问题向量"的**单路**对比，
        再测"并集召回"。
    """
    raise NotImplementedError(
        "L4 HyDE 未实现；请先只启用已实现的 l3（PAPERPILOT_QUERY_LEVELS=l3）")


# ─────────────────────────── L5 查询分解 ───────────────────────────
def decompose(question: str, max_subs: int = 3) -> list[str]:
    """L5：把高概念密度查询拆成子查询 —— **未实现**。

    设计（本项目里**最可能有正收益**的一级）：
      · LitSearch 的查询大量是多条件复合句（"methods that contain **both** manually
        translated comments **and** additional data augmented…"）→ 拆开后每条子查询的
        语义密度更高，召回的互补性更强（这正是混合检索生效的同一个机制）；
      · 与 L3 的区别：L3 产**同义变体**（同方向），L5 产**互补子查询**（不同侧面）——
        所以 L5 更适合配 `quota_union`（各子查询各占配额），而不是等权 RRF；
      · 拆几段、要不要保留原问题一起检索，都应当**当参数测**，不要拍脑袋定。
    """
    raise NotImplementedError(
        "L5 查询分解未实现；请先只启用已实现的 l3（PAPERPILOT_QUERY_LEVELS=l3）")


# ─────────────────────────── 融合原语 ───────────────────────────
def quota_union(rank_lists: list[list[int]], quotas: list[int]) -> list[int]:
    """按配额合并多路排序（**变体只做召回补充，不稀释主查询排序**）。

    做法：第 i 路只取其前 `quotas[i]` 个，**按路顺序整块拼接**并去重
    （即"主路配额整块在最前，其余各路依次接在后面"）。返回顺序即最终顺序。

    ⚠️ **不能交错取**（"主路第1名、变体第1名、主路第2名…"）：那样变体的高名次项
    会把主路的第 2、3 名往后挤，**正是我们要避免的"稀释主查询排序"**
    （与 `equal_rrf` 负收益同一个机制）。第一版实现犯过这个错，被测试逮住。

    Args:
        rank_lists: 多路排序（每路是文档 id 列表，已按相关度降序）。**第一路视为主路。**
        quotas:     每路的配额（与 rank_lists 等长；<=0 表示该路不取）。
    """
    if len(rank_lists) != len(quotas):
        raise ValueError("rank_lists 与 quotas 长度必须一致")
    out: list[int] = []
    seen: set[int] = set()
    for rl, q in zip(rank_lists, quotas):
        for d in rl[:max(q, 0)]:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


# ─────────────────────────── 门面 ───────────────────────────
def optimize(question: str, *, levels: tuple[str, ...] | None = None,
             history: list[dict[str, Any]] | None = None,
             fusion: str | None = None, max_queries: int = 3) -> QueryPlan:
    """跑启用的级别，返回 `QueryPlan`（检索侧只读 `.queries`）。

    Args:
        levels: 显式指定级别（不传则读 env；空 → 全关）。**全关时只回原问题**，
                与旧默认行为完全一致。
        fusion: 融合意图（`FUSION_NAMES` 之一）；不传则按 env
                `PAPERPILOT_QUERY_FUSION`，默认 `equal_rrf`（旧行为）。
    """
    lv = enabled_levels() if levels is None else tuple(levels)
    fus = fusion or os.environ.get("PAPERPILOT_QUERY_FUSION", FUSION_EQUAL_RRF)
    if fus not in FUSION_NAMES:
        fus = FUSION_EQUAL_RRF
    plan = QueryPlan(original=question, levels=lv, fusion=fus, max_queries=max_queries)
    if not lv:
        return plan                                   # 全关：只有原问题

    if "l1" in lv:
        info = classify(question, history)
        plan.intent = str(info.get("intent") or "")
    if "l2" in lv:
        plan.restated = restate(question, history) or ""
    if "l3" in lv:
        plan.variants = expand(question)
    if "l4" in lv:
        plan.hyde_doc = hyde(question)
    if "l5" in lv:
        plan.subqueries = decompose(question)
    return plan


__all__ = ["LEVELS", "FUSION_NAMES", "FUSION_EQUAL_RRF", "FUSION_QUOTA_UNION",
           "FUSION_RERANK_ORIG", "QueryPlan", "enabled_levels", "classify", "restate",
           "expand", "hyde", "decompose", "quota_union", "optimize"]
