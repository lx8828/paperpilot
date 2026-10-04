"""QA 题集**分组清单**（唯一真源）—— 组相关的脚本都从这里取论文名单与配额。

## 范式（加一组只做这几件事）

    1. 往 `GROUPS` 里加一条（组名 → 论文 stem 列表）
    2. 单篇层（`<stem>.questions.json`，每题绑一篇）：
        python retrieval/scripts/_dump_papers.py    --group groupN   # 抽原文 → 素材
        python retrieval/scripts/_render_tables.py  --group groupN   # 渲染表格页 → 素材
        （人工出题，写 retrieval/tmp/<group>/<stem>.questions.json）
        python retrieval/scripts/_validate_questions.py --group groupN   # 引文逐字 + 配额
        python retrieval/scripts/_check_reachable.py    --group groupN   # 可答性/层级自检（材料级）
        python retrieval/scripts/_export_questions.py   --group groupN   # → qa/questions/
        python retrieval/scripts/_check_export_sync.py --group groupN    # ⚠️ gold↔导出 同步（防假测试）

    3. 组级层（`_group.questions.json`，**语料级**：一次问 5 篇）—— 单篇层测不到的能力：
        python retrieval/scripts/_validate_group_questions.py --group groupN  # 配额 + 跨篇引文 + reach
        python retrieval/scripts/_export_questions.py         --group groupN  # → qa/multi/<group>.json
        python retrieval/scripts/_check_export_sync.py        --group groupN  # ⚠️ 同上，跑批前必跑
        python cli/eval/run_multi_qa.py --pdfs <5 篇> ... --retrieval layered  # 检索层对照（直连节点）
        python cli/eval/run_multi_qa.py --pdfs <5 篇> ... --retrieval layered --l0  # M0 应判「够」/ M1 应判「不够」
        python cli/eval/run_group_qa.py --group groupN                        # ⚠️ **完整图**（生产口径，见下）

    覆盖矩阵（两类文件合起来才算"系统所有都被测到"）：

        链路维度（**2026-09-23 假测试排查后新增，最容易被漏**）：
            `run_multi_qa.py`  = **直连节点**（自己调检索 → generate_answer），
                                 跳过 Router / judge_l3 / validator 闸门 / 修复回路
            `run_group_qa.py`  = `web/app.py` 同款 `graph.ask`，**生产全链路** ← 验收看这个
        ⚠️ 旧范式里写的单篇层完整图入口 `run_qa_v2.py` **已不存在**，
           不要再按它跑；完整图一律用 `cli/eval/run_group_qa.py`（本系统没有单篇路径，语料恒为 5 篇）。

        单篇·免检索   L0 题（category=L0）        → run_group_qa；不走检索，测 overview/core_points
        单篇·需检索   single / table 题           → run_group_qa（完整图）/ run_multi_qa（A 组）
        单篇·拒答     negative 题                 → run_group_qa；"论文没给"本身就是答案
        多篇·免检索   M0 题（组级）               → run_multi_qa --l0；合并总览 + judge_l0
        多篇·跨篇     M1 题（组级）               → run_group_qa / run_multi_qa；只能在检索模式下答对
        语料级拒答    X5 题（组级）               → run_group_qa / run_multi_qa；5 篇都没有，不许编造

## 两类文件（**入库策略不同**，别混）

    素材（**不入库**，含论文原文与版面图）：`retrieval/tmp/<group>/`
        <stem>.txt              pymupdf 逐页抽取原文（出题素材；gold 引文逐字核对用）
        <stem>.questions.json   **gold 题集（唯一真源）**，每题带逐字原文引文
        pages/<stem>_pN.png     表格页渲染（表格题**必须看版面**，不看任何解析器输出）
      → 为什么不入库：含大段论文原文与版面图，公开仓库不宜转载（与 .gitignore 既有策略一致）

    产物（**入库**，runner 直接消费）：`qa/questions/<stem>.json`
        由 `_export_questions.py` 从 gold 生成；字段是 `cli/eval/run_group_qa.py` 认的那套
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
        # 2026-09-27：M1 扩量 10 → 16（**专为跨篇 gold 召回与排序**加题：消融 / 超参敏感度 /
        # 推理·检索开销 / 评测规模 / 精度-部署权衡 / 离线 vs 查询期分工 六轴，
        # 与原有 10 道不重叠；新增题 evidence 跨 2~3 篇、锚点用正文数字/专名）。
        "quota": {"M0": 5, "M1": 16, "X5": 5},
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
        # 2026-09-27：M1 扩量 10 → 14（跨篇 gold 召回与排序）：组件消融 / 评测规模 /
        # 反向迁移 / 「无累积记忆·无共享证据面」对照 四轴。
        "quota": {"M0": 5, "M1": 14, "X5": 5},
        "papers": [
            "2604.20087",     # SkillLearnBench：技能生成的持续学习基准
            "2605.09341",     # SkillMAS：多智能体系统的技能共演化
            "2606.18837",     # Skill-MAS：自动多智能体系统的元技能演化
            "2606.25389",     # Offline Multi-agent Continual Cooperation：技能划分与复用
            "2609.02094v1",   # MASkills：多智能体 LLM 系统的持续技能优化
        ],
    },
    # ── 2026-09-25 新增 3 组：**只出 M1**（专测跨篇检索召回，不出单篇层）──────────
    # 论文来源：`cli/run_search.py`（本地 arXiv 索引 540k）→ `cli/run_fetch.py` 抓取。
    # 选篇口径：5 篇同题不同机制，且在**多个正交维度**上可切分（这样 M1 的
    # "哪些篇做了 X" 才有区分度）；并用"实验密度"（Table/百分比/消融/小数个数）
    # 剔除实验过薄的篇（详见 `retrieval/tmp/_exp_density.py` 的实测）。
    "group3": {
        "title": "第三组 · 智能体长期记忆五篇（动态互联笔记 / 层级记忆 / 加权记忆树 / "
                 "可维护主题文档 / 学习式记忆管理）",
        # 2026-09-27：M1 扩量 10 → 14（跨篇 gold 块级召回与排序）。新增四轴**避开**原 10 道
        # 已覆盖的「组件消融 / 上下文开销」：投毒·鲁棒性 / 对外部专家模型与预定操作的依赖 /
        # 写入·整理的触发条件 / 人类可读可编辑的原子笔记。
        "quota": {"M1": 14},      # 只出 M1
        "single_layer": False,    # 不出单篇层 → 单篇层的校验/导出/同步检查跳过
        "papers": [
            "2502.12110",     # A-MEM：Zettelkasten 式动态互联笔记
            "2601.06377",     # HiMem：层级长期记忆（认知理论启发）
            "2608.20631",     # Weighted Memory Tree：树 + 重要性权重
            "2606.10677",     # Infini Memory：可维护主题文档 + 事实修订
            "2601.01885",     # Agentic Memory (AgeMem)：LTM/STM 统一、学习式
        ],
    },
    "group4": {
        "title": "第四组 · LLM 工具使用五篇（生成式工具检索 / 依赖图检索 / "
                 "经验回放 / 推理注入检索器 / 统一表示与评测）",
        # 2026-09-27：M1 扩量 10 → 14（跨篇 gold 块级召回与排序）。新增四轴避开原 10 道：
        # 超参·组件消融 / 未见工具·跨模型族泛化 / 自改进与经验库更新 / 检索-执行解耦。
        "quota": {"M1": 14},      # 只出 M1
        "single_layer": False,
        "papers": [
            "2410.03439",     # ToolGen：工具统一为生成式 token，检索=生成
            "2512.17052",     # DTDR：动态工具依赖检索（端上轻量）
            "2508.15214",     # SEER：逐步经验回放（免人工示例）
            "2510.19791",     # ToolDreamer：把 LLM 推理注入工具检索器
            "2604.11557",     # UniToolCall：统一表示 / 数据 / 评测
        ],
    },
    "group5": {
        "title": "第五组 · 测试时验证与扩展五篇（生成式验证器 / 生成式过程奖励 / "
                 "长 CoT 验证器 / RL 训练验证器 / 验证即改进）",
        # 2026-09-27：M1 扩量 10 → 13（跨篇 gold 块级召回与排序）。新增三轴避开原 10 道：
        # 同计算预算下的测试时扩展 / 训练配比与标签量 / 低成本合成批评与标签。
        # ⚠️ 本组只有 2 篇有消融材料（且一篇落在附录图里）→ 不出消融题。
        "quota": {"M1": 13},      # 只出 M1
        "single_layer": False,
        "papers": [
            "2408.15240",     # GenRM：生成式验证器（奖励建模当 next-token 预测）
            "2504.00891",     # GenPRM：生成式过程奖励模型
            "2504.16828",     # ThinkPRM：长 CoT 验证器，1% PRM800K 标签
            "2509.23152",     # Mirror-Critique：RL 训练验证器（诚实性）
            "2609.19515",     # LLM-as-an-Improver：Verify–Repair–Reselect
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


def has_single_layer(name: str = "group1") -> bool:
    """该组是否有**单篇层**（`<stem>.questions.json`，每篇 14 题）。

    ⚠️ 新 3 组（group3/4/5）是为专测 M1 召回建的，声明 `"single_layer": False`
    → 单篇层的**校验 / 导出 / 同步检查**都应跳过；否则会一直报
    「缺 5 个 gold / 5 处不同步」的**假警报**，把真问题淹掉（2026-09-25 实测踩到）。
    """
    return bool(group(name).get("single_layer", True))


def expected_total(name: str = "group1") -> int:
    """单篇层期望题数（无单篇层 → 0）。"""
    if not has_single_layer(name):
        return 0
    return sum(QUOTA.values()) * len(papers(name))


def group_gold_file(name: str = "group1") -> Path:
    """组级题集（语料级）路径。"""
    return material_dir(name) / GROUP_GOLD_NAME


def group_quota(name: str = "group1") -> dict[str, int]:
    """该组的**组级层配额**；组条目里的 `"quota"` 覆盖默认 `GROUP_QUOTA`。

    ⚠️ 为什么要能覆盖（2026-09-25）：新 3 组（group3/4/5）是为**专测 M1 跨篇检索召回**建的，
    按要求**只出 M1**（各 10 道），不出 M0/X5。不能覆盖的话，校验器会一直报
    「⚠️ M0 0/5、⚠️ X5 0/5」的**噪声告警**，把真问题淹掉（实测踩过）。
    """
    q = group(name).get("quota")
    return dict(q) if q else dict(GROUP_QUOTA)


def group_expected_total(name: str = "group1") -> int:
    """组级层题数（与 `expected_total` **相加** 才是整组题量）。"""
    return sum(group_quota(name).values())


def export_dir_multi() -> Path:
    """组级题集的导出目录（`cli/eval/run_multi_qa.py` 的 B 组读这里）。"""
    return ROOT / "qa" / "multi"
