from typing import Any, Literal

from pydantic import BaseModel


class Page(BaseModel):
    page: int
    text: str


class BlockLine(BaseModel):
    """文本块内的一行（视觉行），带坐标。"""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


class Block(BaseModel):
    """PDF 文本块，带页码和坐标，支撑 Chunker 语义合并和 Verifier 溯源。"""
    block_id: str   # 形如 "p1_b0"，全局唯一
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    lines: list[BlockLine] = []  # 行级明细，支撑段落检测与精确溯源


class Heading(BaseModel):
    """识别出的论文标题。"""
    block_id: str
    page: int
    kind: str          # L1/L2/L3/APPENDIX/SPECIAL
    no: str            # 编号（'3.1'）或特殊标题名（'References'）
    level: int | None  # L1=1, L2=2, L3=3；附录/特殊为 None
    text: str          # 标题完整文本（含编号）
    score: float


class ChunkSpan(BaseModel):
    """Chunk 内的文本锚点（块级），支撑 Verifier 溯源到具体 block/页。"""
    block_id: str
    page: int
    text: str


class Chunk(BaseModel):
    """按标题切分的语义块，携带完整标题路径支撑溯源。

    超大 chunk 会被二级切分（段落级）拆成多个子块，子块之间通过
    part（如 "1/3"）区分，且共享父 chunk 的 title_path。
    """
    chunk_id: str          # 形如 "c1" / 二级切分子块 "c1-p1"
    title_path: list[str]  # 完整根路径，如 ["3 Method", "3.2 RLOO"]；PREAMBLE 为 ["(PREAMBLE)"]
    page_span: tuple[int, int]  # (起始页, 结束页)
    text: str              # 标题行 + 全部正文文本
    n_blocks: int          # 组成该 chunk 的 block 数
    part: str | None = None      # 二级切分子块序号 "1/3"；整块为 None
    spans: list[ChunkSpan] = []  # 组成该 chunk 的 block 锚点（块级溯源）


class ParserOutput(BaseModel):
    """Parser 节点写入 State 的数据结构，字段与 parse_pdf() 工具输出对齐。"""

    title: str = ""  # 部分 PDF 的 metadata 会缺失，给默认值
    authors: list[str] = []
    page_count: int
    pages: list[Page]
    blocks: list[Block]
    raw_text: str


ClaimType = Literal["contribution", "method", "result", "limitation"]


class Claim(BaseModel):
    """Analyzer 从 chunk 提取的原子主张（contribution/method/result/limitation）。

    每条 claim 绑定来源 chunk（chunk_id/title_path/page_span），
    evidence_quote 为原文逐字支持片段，供 Verifier 做提取-验真闭环。
    """

    claim_id: str
    type: ClaimType
    text: str            # 用户可读主张（中文概括）
    evidence_quote: str  # 原文支持片段（verbatim，须能在原 chunk 中找到）
    chunk_id: str
    title_path: list[str]
    page: int                       # 起始页，page_span[0]
    page_span: tuple[int, int]


ClaimLabel = Literal[
    "core_claim",          # 论文核心命题 / 主结论（总体主张）
    "result_primary",      # 主量化结果（headline，带关键数字/对比）
    "result_supporting",   # 支撑/次级结果（具体实证点，非 headline）
    "method_core",         # 主方法 / 架构设计
    "ablation",            # 消融 / 敏感性分析
    "detail",              # 实现细节 / 公式 / 背景 / 设定
    "limitation",          # 局限 / 适用边界
    "related",             # 相关工作 / 对比
]


class ClaimGroup(BaseModel):
    """去重归并 + 角色打标 + 规则算分后的"唯一主张组"。

    一组 = 同一主张在论文不同 section 的重复表述集合；
    rep 取自组内某条原始 claim 的 text（可溯源），其余成员记入 claim_ids。
    """

    group_id: str
    rep_claim_id: str      # rep_text 取自哪条原 claim
    rep_text: str
    claim_ids: list[str]   # 组内全部原始 claim_id（含 rep）
    type: str = ""         # 代表 claim 的类型（contribution/method/result/limitation）
    sections: list[str] = []   # 去重后的顶层 section 名（出现的不同章节）
    pages: list[int] = []
    label: ClaimLabel = "detail"
    label_why: str = ""
    importance: int = 0        # 1-5
    ev_state: str = ""         # 组代表 claim 的 evidence 状态 hit/loose/miss
    score: dict[str, float] = {}   # 各分量：base/pos/occ/quant/ev_adj，供解释


# ═══════════════════════ PaperReport：一步封装的最终报告结构 ═══════════════════════

NARRATIVE_SECTION_HINTS = ("abstract", "introduction", "conclusion", "discussion",
                           "summary", "concluding")


class GroupBrief(BaseModel):
    """报告里一条主张的展示摘要（渲染/问答共用）。

    rep_claim_id 让展示条目可直接定位代表 claim（evidence_quote/page），
    问答层引用 cite 回原文只需一跳。
    """

    gid: str
    rep_claim_id: str = ""
    label: str
    importance: int
    text: str
    ev: str = ""           # hit/loose/miss
    pages: list[int] = []


class EdgeView(BaseModel):
    """骨架一条关系边（source 组 + 绑定证据，供前端展开与溯源）。"""

    relation: str          # implements/supports/limits/contrasts
    source: str            # source 组 id
    text: str = ""         # source 组 rep_text
    why: str = ""
    evidence: str = ""     # source 组的原文证据
    page: int = 0
    ev: str = ""


class HubView(BaseModel):
    """骨架一个核心主张 Hub。"""

    hub: str
    label: str = ""
    importance: int = 0
    text: str = ""
    isolated_reason: str = ""   # "" / "no_sources" / "no_edges"
    edges: list[EdgeView] = []
    rep_claim_id: str = ""      # hub 对应主张组的代表 claim（直达原文证据）


class SectionView(BaseModel):
    """章节精读：一节下的主张——高分直展，低分默认折叠可展开。

    groups      高分主张（importance>=4，直接展示）
    detail_groups 低分主张（importance<4，默认折叠；前端 <details> 展开）
    """

    title: str
    groups: list[GroupBrief] = []
    n_detail: int = 0      # 低分细节数（= len(detail_groups)，兼容旧字段）
    detail_groups: list[GroupBrief] = []


class PaperReport(BaseModel):
    """单篇最终结构化报告——"一个 PDF 进，一个 report 出"的统一产物。

    设计原则：
      - 展示层字段（core_points/sections/…）由 summary 聚合而成，渲染/问答直接消费
      - claims/groups 全量保留，自足一份文件（不依赖 out_claims/out_views 中间产物）
      - 每个展示条目都携带 gid → 可回溯到 claims/groups → 原文证据
    """

    pdf: str
    title: str
    generated_at: str
    stats: dict[str, int] = {}      # n_claims / n_groups / n_hubs / n_edges / n_must / n_lim
    guide: str = ""                 # 一分钟导读（小白向）
    overview: str = ""              # 概述（研究者向）
    core_points: list[GroupBrief] = []   # 必读主张（importance>=5，按分排序）
    limitations: list[GroupBrief] = []   # 局限（按重要性）
    skeleton: list[HubView] = []         # 论证骨架
    sections: list[SectionView] = []     # 章节精读
    figures: list[dict[str, Any]] = []   # 图表一览（含读图指南）
    claims: list[Claim] = []             # 全量 claims（evidence/chunk/page/title_path，强类型）
    groups: list[ClaimGroup] = []        # 全量主张组（label/score/sections/claim_ids，强类型）
