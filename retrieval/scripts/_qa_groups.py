"""QA 题集**分组清单**（唯一真源）—— 组相关的脚本都从这里取论文名单与配额。

## 范式（加一组只做这几件事）

    1. 往 `GROUPS` 里加一条（组名 → 论文 stem 列表）
    2. 单篇层（`<stem>.questions.json`，每题绑一篇）：
        python retrieval/scripts/_dump_papers.py    --group groupN   # 抽原文 → 素材
        python retrieval/scripts/_render_tables.py  --group groupN   # 渲染表格页 → 素材
        （人工出题，写 retrieval/tmp/<group>/<stem>.questions.json）
        python retrieval/scripts/_validate_questions.py --group groupN   # 引文逐字 + 配额
        python retrieval/scripts/_check_reachable.py    --group groupN   # 可答性/层级自检
        python retrieval/scripts/_export_questions.py   --group groupN   # → qa/questions/
        python cli/run_qa_v2.py --pdf <stem>.pdf                         # 跑批（完整图，含路由/闸门）

    3. 组级层（`_group.questions.json`，**语料级**：一次问 5 篇）—— 单篇层测不到的能力：
        python retrieval/scripts/_validate_group_questions.py --group groupN  # 配额 + 跨篇引文 + reach
        python retrieval/scripts/_export_questions.py         --group groupN  # → qa/multi/<group>.json
        python cli/run_multi_qa.py --pdfs <5 篇> ...        # A 组(单篇题@5篇语料：干扰损耗) + B 组(组级题)
        python cli/run_multi_qa.py --pdfs <5 篇> ... --l0   # M0 应判「够」/ M1 应判「不够」← 负向对照

    覆盖矩阵（两类文件合起来才算"系统所有都被测到"）：
        单篇·免检索   L0 题（category=L0）        → run_qa_v2；不走检索，测 overview/core_points
        单篇·需检索   single / table 题           → run_qa_v2（完整图）/ run_multi_qa（A 组，5 篇语料干扰）
        单篇·拒答     negative 题                 → run_qa_v2；"论文没给"本身就是答案
        多篇·免检索   M0 题（组级）               → run_multi_qa --l0；合并总览 + judge_l0
        多篇·跨篇     M1 题（组级）               → run_multi_qa；只能在检索模式下答对
        语料级拒答    X5 题（组级）               → run_multi_qa；5 篇都没有，不许编造

## 两类文件（**入库策略不同**，别混）

    素材（**不入库**，含论文原文与版面图）：`retrieval/tmp/<group>/`
        <stem>.txt              pymupdf 逐页抽取原文（出题素材；gold 引文逐字核对用）
        <stem>.questions.json   **gold 题集（唯一真源）**，每题带逐字原文引文
        pages/<stem>_pN.png     表格页渲染（表格题**必须看版面**，不看任何解析器输出）
      → 为什么不入库：含大段论文原文与版面图，公开仓库不宜转载（与 .gitignore 既有策略一致）

    产物（**入库**，runner 直接消费）：`qa/questions/<stem>.json`
        由 `_export_questions.py` 从 gold 生成；字段是 `cli/run_qa_v2.py` 认的那套
        （`must_all` = 全部命中，`must_have` = 任一命中）

## gold 题集的判定锚点（**与 runner 的语义陷阱**）

    must_all  必现锚点（数字/专名）—— 全部命中才算过
    must_any  备选措辞 —— 任一命中即可（容忍中英表述差异）
    must_not  出现即判错（只用于**安全的**定性陷阱；纯取数题留空，硬塞会误伤）
    导出时：gold.must_all → runner.must_all，**gold.must_any → runner.must_have**（名字不同、语义对齐）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # scripts → retrieval → 仓库根
MATERIAL_ROOT = ROOT / "retrieval" / "tmp"          # 素材（不入库）
EXPORT_DIR = ROOT / "qa" / "questions"              # 导出产物（入库）

# 每组**每篇**的配额（单篇层）；组总题数 = 配额和 × 篇数
QUOTA: dict[str, int] = {"L0": 3, "single": 8, "table": 2, "negative": 1}

# 每组**每语料**的配额（组级层，见 `_group.questions.json`）；与单篇层**相加**
#   M0 多篇·免检索（答案须落在 5 篇 L0 材料里，--l0 下 judge_l0 应判「够」）
#   M1 跨篇·需检索（必须下钻；--l0 下 judge_l0 应判「不够」= 负向对照）
#   X5 语料级拒答（正确答案 = 五篇都没有）
GROUP_QUOTA: dict[str, int] = {"M0": 5, "M1": 10, "X5": 5}

# 组级题集的固定文件名（与 `<stem>.questions.json` 同目录）
GROUP_GOLD_NAME = "_group.questions.json"

GROUPS: dict[str, dict] = {
    "group1": {
        "title": "第一组 · cs.IR 五篇（生成式推荐 / 检索增强代码修复 / 复购解释 / 多模态文档检索 / 组合图像检索）",
        "papers": [
            "2608.29179v1",   # TAAL：生成式推荐的早期 beam 剪枝
            "2608.29290v1",   # Database-Augmented RAG：REST API 误用修复
            "2608.30333v1",   # LLM-Cited Feature Rationales：复购推荐的特征理由
            "2609.01316v1",   # MIDR：索引时富化的多模态文档检索
            "2609.01456v1",   # AutoConcept：元数据可用的 CIR 重排
        ],
    },
    "group2": {
        "title": "第二组 · Agent Skill 持续学习五篇（技能生成基准 / 技能共演化 / "
                 "元技能演化 / 离线持续协作 / 持续技能优化）",
        "papers": [
            "2604.20087",     # SkillLearnBench：技能生成的持续学习基准
            "2605.09341",     # SkillMAS：多智能体系统的技能共演化
            "2606.18837",     # Skill-MAS：自动多智能体系统的元技能演化
            "2606.25389",     # Offline Multi-agent Continual Cooperation：技能划分与复用
            "2609.02094v1",   # MASkills：多智能体 LLM 系统的持续技能优化
        ],
    },
}


def group(name: str = "group1") -> dict:
    if name not in GROUPS:
        raise SystemExit(f"未知分组 {name!r}；已有：{sorted(GROUPS)}")
    return GROUPS[name]


def papers(name: str = "group1") -> list[str]:
    return list(group(name)["papers"])


def material_dir(name: str = "group1") -> Path:
    return MATERIAL_ROOT / name


def gold_files(name: str = "group1") -> list[Path]:
    return [material_dir(name) / f"{stem}.questions.json" for stem in papers(name)]


def expected_total(name: str = "group1") -> int:
    return sum(QUOTA.values()) * len(papers(name))


def group_gold_file(name: str = "group1") -> Path:
    """组级题集（语料级）路径。"""
    return material_dir(name) / GROUP_GOLD_NAME


def group_expected_total(name: str = "group1") -> int:
    """组级层题数（与 `expected_total` **相加** 才是整组题量）。"""
    return sum(GROUP_QUOTA.values())


def export_dir_multi() -> Path:
    """组级题集的导出目录（`cli/run_multi_qa.py` 的 B 组读这里）。"""
    return ROOT / "qa" / "multi"
