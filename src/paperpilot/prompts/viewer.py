"""Viewer（去重归并 / 角色打标）提示词。

去重：LLM 判断哪些 claims 是同一主张的重复表述并归组。
      只输出 best_id + claim_ids，rep 文本由本地从 best_id 取原句
      （避免 LLM 新写引入不可溯源文本）。
打标：LLM 只做"论证角色"离散分类，不直接打分；
      分数由本地规则根据 label + 位置/重复/量化/证据状态计算。
"""
from __future__ import annotations

DEDUPE_SYSTEM = """你是一名论文主张去重引擎。系统会给你一篇论文**按章节组织**的全部 claims，每条带 claim_id、类型(type)、所属章节(section)。你的任务：找出**表达同一主张/陈述同一事实**的重复 claims，把同义的那些归并成一组。

「同义主张」的判断标准：两条 claim 措辞不同，但**陈述的是同一个事实/结论**。典型场景：Abstract、Introduction、Discussion 会对同一核心结论做重述（如 Abstract 说"方法 X 把准确率从 80% 提升到 90%"，Introduction 又说"X 带来显著提升"——数字一致可视为同一结果的重复表述）。

必须严格遵守的边界（**宁可少并，不可误并**）：
1. 泛化 vs 具体 不是同义。如"6 种偏好优化目标都会诱导谄媚"与"DPO 会诱导谄媚"，一宽一窄，**不要合并**。
2. 不同实验、不同数据集、不同模型、不同设定的各自结果，即使主题相似，**不要合并**。
3. 同一研究中不同指标、不同对比对象、不同子发现的表述，**不要合并**。
4. 一条讲"方法怎么做"，一条讲"结果如何"，不是同义，**不要合并**。
5. 只把确实陈述同一事实的句子并在一起；不能把主题相近但内容不同的 claims 塞进同一组。
6. **论文核心贡献/目标/愿景的"换皮表述"必须合并**——这类最容易漏并。判断要点：若几条 claim 都是"本文/我们/该系统 提出/构建/介绍/设计了 X"或"X 旨在/目标是/目标是让…"，且指向**同一个系统及其总体目标**，无论措辞差多少都要并成一组。例如："提出Prove2Me，一个开源协作平台"、"平台目标是将数学形式化转变为可扩展的众包努力"、"Prove2Me旨在提供高效且可扩展的验证托管" 都指向 Prove2Me 与其总体目标 → 必须并组（选信息最全的一条作 best_id）。注意区分：该系统的**具体方法组件**（如不同的机制、不同模块）仍各自独立、不并入此组。

输出要求：
- 每条 claim 必须**恰好**出现在一个组里：同义的并组，无法并组的单独成组（组内只有它自己）。
- best_id：从组内选择**信息最完整、最能代表该主张**的那条 claim_id（组里通常不止一条时，优先选信息量最大/数字最具体的）。
- 只输出 JSON 数组，不要任何解释文字：
[{"best_id": "c0042", "claim_ids": ["c0042", "c0103", "c0122"]}, ...]
"""


def build_dedupe_user(claims_by_section: list[tuple[str, list[dict]]]) -> str:
    """claims_by_section: [(section_tail, [{claim_id, type, text}])]。"""
    lines = [f"共 {sum(len(c) for _, c in claims_by_section)} 条 claims，请归并。", ""]
    for section, items in claims_by_section:
        lines.append(f"## {section}")
        for it in items:
            lines.append(f"{it['claim_id']} [{it['type']}] {it['text']}")
        lines.append("")
    return "\n".join(lines)


LABEL_SYSTEM = """你是一名学术论文分析器。系统会给你论文中**已去重**的若干"主张组"，每组是一个唯一主张（可能被论文多处重申），带：group_id、原始类型 type、出现的章节 sections、代表句 text。

你的任务：判断**这句话在这篇论文的论述中扮演什么角色**，从下列 8 类中选 1 类：

- core_claim：论文核心命题/主结论（如"本文证明了…""本文提出/主张…"的总体论断，未必含数字）。
- result_primary：headline 量化结果——论文的"卖点"级结论，通常带关键数字或显著对比（提升 X%、从 A 涨到 B、显著优于…）。
- result_supporting：支撑性/次级实证发现——某个环节的发现或特征（相关性、分布、差异），但不是论文主卖点。
- method_core：核心方法设计——模型/架构/机制/流程是怎么做的。
- ablation：消融 / 敏感性 / 变体分析得到的结论。
- detail：实现细节、公式设定、数据构建细节、背景铺垫（说明性，非主张）。
- limitation：**本文自己**承认的局限/适用范围/假设/未覆盖场景。判据：主语是"我们/我们的/本文/该平台"，且语义是限制、未覆盖、留待未来、早期阶段、非受控等。
- related：与已有工作的对比 / 引用 / 领域定位（相关工作性质）。

重要反例（最易错）：**批评"现有/他人工作"的缺点不是 limitation**。引言里"现有方法难以复用、早期工作缺乏审计、已有系统依赖单点"这类句子是**论文动机/背景**，标 related（或 detail），绝不标 limitation。判断是否 limitation 先问："这是作者在说**自己**的不足，还是在说**别人的**问题？"

注意：选"角色"而不是"话题"——例如同样在讲某方法，"它如何设计"是 method_core，"用了它准确率提升到 90%"是 result_primary。

输出 JSON 数组，每项带简短理由：
[{"group_id": "g3", "label": "result_primary", "why": "带12%→32%数字"}, ...]
"""


def build_label_user(groups: list[dict]) -> str:
    """groups: [{group_id, type, sections, text}]。"""
    lines = []
    for g in groups:
        secs = ", ".join(g["sections"]) if g["sections"] else "?"
        lines.append(
            f"gid={g['group_id']} | type={g['type']} | sections=[{secs}]"
            f"\n     text: {g['text']}"
        )
    return "\n".join(lines)
