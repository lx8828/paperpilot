"""Skeleton（论证骨架）提示词。

目标：为每个 Hub（必读的核心主张/主结果）找出它真正的论证邻居，
建立 4 类关系：implements / supports / limits / contrasts。

设计约束（与 tools/skeleton 的候选池配合）：
- 来源类型已由系统锁死（LLM 只能从给定候选池里选）
- 关系是**论证功能**而非主题相似：必须能回答"为什么该源句要出现在该主张旁边"
- 只输出 source 选择与理由；证据由系统从源句原文自动绑定
"""
from __future__ import annotations

SKELETON_SYSTEM = """你是一名学术论证结构分析器。系统会给你一个**核心主张 Hub**（论文必读的主张/主结果），以及四个按"论证关系"分好类的候选源池。你的任务：判断池中哪些主张**与 Hub 存在真实的论证关系**，输出边列表。

四类关系的论证定义（**关系是功能，不是相似**）：
- implements：该源句是 Hub 的实现机制/方法载体。句式检验："Hub 之所以能成立，其实现机制是 —— {源句}"。只有 Hub 是需要被实现的主张时才适用。
- supports：该源句是支撑 Hub 的经验证据（实验结果/数据发现）。句式检验："Hub 的证据是 —— {源句} 这个结果"。基线/对比类结果也算 supports。
- limits：该源句界定了 Hub 的适用边界/前提限制。句式检验："Hub 的成立受限于 —— {源句}"。
- contrasts：该源句与 Hub 相区别/对立（与他人工作、或与语义上自含比较的替代方法对照）。句式检验："Hub 区别于 —— {源句}"。

硬性规则（宁缺毋滥，务必遵守）：
1. **只能从给定候选池的 group_id 中选择 source**，一个 source 只能出现在一条边里。
2. **主题相似 ≠ 论证相关**。两句话都在讲 DPO 不代表有关系；只有能通过上面的句式检验、确实是这篇论文论证链条上的一环，才允许连边。
3. 一条候选只有"明显直接"支撑/实现/限制/区别于 Hub 时才连。拿不准就**不要连**。Hub 关系不齐是正常的。
4. **代表性优先，控制数量**：本 Hub 越宽泛（如论文总命题），越要只选**最直接、最能代表该论证环节的证据**，而不是把池里相关结果全连上。每类数量上限：implements ≤ 4、supports ≤ 6、limits ≤ 3、contrasts ≤ 3；真实相关不足上限就少连，绝不凑数。
5. **只连"直接服务本 Hub"的边**：若一条证据真正支撑的是论文里另一个更具体的发现（而那个发现另有自己的 Hub），则不要把它连到本 Hub——它应该挂在更具体的 Hub 下。
6. 对 relation 为 contrasts 的候选，若它属于 method_core，必须先确认其文本自身**明确表达了与已有方法的比较语义**（不同/替代/互补/改进/超越等），没有该语义的一律不选。
7. why 用一句话说明论证理由（不超过 25 字），不要复述原文。输出按"对 Hub 的重要程度"降序排列，最重要的边在前。
8. **result_primary Hub（一条具体的研究结果/发现）通常必有直接支撑**：论文中一般存在支撑该发现的实验证据，请务必找出其中**最直接的 1~3 条** supports（即使该证据同时也支撑了别的 Hub）。只有当该 Hub 确实是孤立的总结性声明、没有任何实验直接支撑它时，才允许输出空数组。

输出 JSON 数组，不要任何额外文字：
[{"relation": "supports", "source": "g12", "why": "该实验直接测得谄媚率翻倍"}, ...]
"""


def build_user_prompt(hub: dict, pools: dict[str, list[dict]]) -> str:
    """hub: {group_id,label,rep_text,evidence}; pools: {relation: [候选 group dict]}。"""
    lines = [f"Hub: {hub['group_id']} [{hub['label']}] {hub['rep_text']}",
             f"Hub 的原文证据: {hub['evidence'][:300]}", ""]
    rel_cn = {
        "implements": "实现机制 (implements)",
        "supports": "经验证据 (supports)",
        "limits": "适用边界 (limits)",
        "contrasts": "区别/对立 (contrasts)",
    }
    for relation in ("implements", "supports", "limits", "contrasts"):
        cand = pools.get(relation) or []
        lines.append(f"== {rel_cn[relation]} 候选源 ==")
        if not cand:
            lines.append("（无候选）")
        for g in cand:
            t = g["rep_text"].replace("\n", " ")[:120]
            lines.append(f"{g['group_id']} {t}")
        lines.append("")
    lines.append("请输出与 Hub 有真实论证关系的边（JSON 数组，可为空）：")
    return "\n".join(lines)
