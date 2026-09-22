"""graph：LangGraph 编排层（只接线，节点逻辑在 agents/）。

    graph/qa_graph_v3.py  v3 两级图（L0 总览 → L3 全局检索）——2026-09-10 起唯一图
    components/           输出前闸门等（env PAPERPILOT_VALIDATOR_GATE=1 时 ask 自动校验）
    将来：多篇对比图、主题追踪图等在此扩展。

定稿依据（qa/recall/ab_v3regress_result.json，250 题同裁判 A/B）：
    V0(v3) 203/250 (81.2%) ≥ A(v2) 201/250 (80.4%)，calls 3.7 vs 5.6 (-34%)，prompt +17%。

**2026-09-10：v2 四层漏斗图（qa_graph.py）连同 L1 claims / L2 扩窗节点已下线归档**
（archive/qa_funnel_v2/），不再提供 ask_v2 回退——v3 是唯一检索链。
"""
from __future__ import annotations

import os
from typing import Any

from paperpilot.graph import qa_graph_v3

build_qa_graph_v3 = qa_graph_v3.build_qa_graph_v3


def ask(question: str, pdfs: list[str],
        history: list[dict[str, Any]] | None = None) -> dict:
    """一键问答（默认 v3 两级 + nol3j + 输出闸门）。

    **语料是多篇 `pdfs`（≥2 篇）**：本系统没有单篇路径（2026-09-23 删）。

    **2026-09-10 起 Validator 输出闸门默认开**（与 nol3j 默认配套：删 L3 judge 后由闸门管质量）。
    high 问题 → 对症 repair，修不动统一兜底话术；mid/low → 原样输出，结果记录在 out['validator']。
    关：env PAPERPILOT_VALIDATOR_GATE=0。升级接口见 components/validator.gate(repair=…)。

    history: 之前轮次对话 [{role, content}]，供 judge/answer 理解"这个方法/它"等指代
    （web /api/ask 多轮追问用）。

    **降级与容灾（2026-09-22 改）**：解析层有**两条独立的路** —— 默认 MinerU
    / 备用 pymupdf（见 `document_cache._chunks_via_mineru` / `_chunks_via_pymupdf`）。
    默认路失败（产物缺失/损坏/解析抛错）→ **自动走备用路**，问答**仍然可用**
    （代价：表格数值拿不到），降级原因记在返回值的 `parse_degraded` 字段里。
    ⚠️ 旧行为是"MinerU 失败 → 问答直接拒绝" —— 那会让备用路**永远走不到**。
    现在仍会被拦的只有一种：**摄取进行中**（产物马上就好，不是错误）。
    """
    # 摄取闸门：只剩"摄取进行中"（MinerU 失败已改为**降级**而非拒绝）。
    # 多篇时**逐篇**检查：任一篇没摄取完就明确拒绝（不拿半成品语料作答）。
    from paperpilot.ingest import qa_blocked_reason
    corpus = [str(p) for p in (pdfs or []) if p]
    for _p in corpus:
        blocked = qa_blocked_reason(_p)
        if blocked:
            return {
                "answer": blocked,
                "cites": [],
                "route": ["ingest_blocked"],
                "validator": {"action": "blocked", "issues": [], "supplements": []},
                "debug": {"ingest": {"blocked": True, "pdf": _p}},
            }

    out = qa_graph_v3.ask(question, corpus, history=history)
    if os.environ.get("PAPERPILOT_VALIDATOR_GATE", "1") != "0":
        from paperpilot.components import repairer, validator

        def _rep(q: str, ans: str, issues, cites):
            """修复回路：high 检出 → Self-Refine / CRAG 对症重试（修不动返回 None → 兜底）。"""
            return repairer.repair(q, corpus, ans, issues, cites)

        def _sup(q: str, ans: str, supplements, cites):
            """补充复核：数字在被引缺失但全篇命中 → 喂回 Generator 判断是否需改。"""
            return repairer.supplement(q, corpus, ans, supplements, cites)

        def _adj(q: str, ans: str, nums, cites):
            """数值裁决：数字在被引与全篇都找不到 → LLM 判是否支持（机器不判死）。"""
            return repairer.adjudicate_numbers(q, ans, nums, cites)

        dbg = out.get("debug") or {}
        n_ctx = ((dbg.get("answer") or {}).get("n_entries")) or None
        # 全篇 chunks：机器数字层溯源"被引缺失数"的命中块（补充上下文候选）。
        # 用**检索视图**（含 MinerU 注入的表格/公式）——否则答案引用表值会被判"无证据"。
        chunks = []
        try:
            from paperpilot.agents.document_cache import retrieval_chunks
            for _p in corpus:                      # 全语料的检索视图（多篇拼接）
                chunks.extend(retrieval_chunks(_p))
        except Exception:  # noqa: BLE001
            chunks = []
        g = validator.gate(question, out.get("answer") or "",
                           list(out.get("cites") or []), repair=_rep,
                           supplement=_sup, adjudicate=_adj,
                           n_entries=n_ctx, extra_chunks=chunks)
        if g["action"] != "pass":
            out["answer"] = g["answer"]
            out["cites"] = list(g.get("cites") or [])
        out["validator"] = g
    return out


__all__ = ["ask", "build_qa_graph_v3"]
