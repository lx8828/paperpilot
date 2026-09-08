# PaperPilot · 论文帮读器（LLM 精读 + 可溯源）

输入一篇 PDF 学术论文 → 系统自动完成 **解析 → claims 提取与证据溯源 → 语义去重 → 角色打标 → 重要性打分 → 论证骨架 → 图表指南 → 结构化精读报告**，并支持**带引用溯源的论文问答**（L0→L1→L2→L3 逐级下钻漏斗）。

**给谁用**：想读论文、却被英文和专业门槛劝退的普通人。系统把论文**讲成能懂的话**（一分钟导读 / 大白话概述 / 引导式问答），同时每个关键结论都带回原文出处——让外行"读得懂、不跑偏、能自己点回原文核实"。

> **核心信条：所有结论必须能回到原文。**
> 任何产出如果不能指向原文页码/证据片段，就视为不合格——用工程手段对抗 LLM 在学术场景下的幻觉，而不是靠 "prompt 让它别编"。

---

## ✨ 能力一览

| 能力 | 说明 |
|---|---|
| 📄 **帮读型报告** | 一分钟导读 / 大白话概述 / 核心要点 / 论证骨架 / 章节精读（低分默认折叠可展开）/ 图表一览 |
| 🔍 **证据溯源闭环** | 每条主张绑定原文 `evidence_quote` + 页码，报告里可一键跳回原文并**整行高亮** |
| 💬 **漏斗式问答** | L0(报告层) → L1(claims) → L2(章节邻域扩展) → L3(独立全文检索) → unknown(诚实收尾)，答案带 [n] 引用可点跳原文 |
| 🛡 **防编造/诚实拒答** | 该说不确定就说不确定；诱导/篡改数值/论文没有的东西 → 不编（负向组 20/20、对抗 12/12） |
| 🖥 **三栏工作台** | 左：结构导航 ｜ 中：报告 / 原文 PDF ｜ 右：多轮追问 |
| ✅ **评测体系** | 帮读问答 + 导读忠实度 + 防幻觉/压力/稳定性 + 三列公平对比（详见下） |

---

## 📊 评测成绩（标准版 · 2026-09-07 定版）

**产品定位**：论文**帮读器**（面向小白），**不是"答题得分器"**。对"帮高手精确摘原文细节"这类需求，我们如实承认不占优——那是产品范围决策，不是缺陷。

### 帮读定位评测（主口径）

| 测试 | 结果 | 报告 |
|---|---|---|
| A 小白外行问答（3 篇 × 5 题） | **15/15**：讲懂、不编造、给阅读路径 | `qa/reader/READER_REPORT_20260907.md` |
| B 导读数值忠实抽检（5 篇） | **102/102** 数值可回原文、0 缺失 | `qa/reader/FIDELITY_REPORT_20260907.md` |
| 防幻觉负向组（篡改数值/不存在物） | **20/20**：0 编造、0 误断言缺失 | `qa/negqa/NEGQA_REPORT_20260907.md` |
| 对抗/诱导压力（12 题） | **12/12**：不顺着假前提、评价缺失项不编 | `qa/stress/STRESS_REPORT_20260907.md` |
| 压力/边界 | 超长 168K 字符 **8/8** pass；坏 PDF 干净报错不崩溃；扫描件不可用（需 OCR/MinerU） | 同上 |
| 稳定性 / 同义改写（30×3） | 3 问法 pass 一致 **70%**（平均跨度 0.53 → 报数自带 ±） | `qa/robust/ROBUST_REPORT_20260907.md` |

### 三列公平对比（B0 直接 LLM / B1 朴素 RAG / B2 本系统 · 同题同裁判盲判）

| 池 | B0 | B1 | B2 | 结论 |
|---|---|---|---|---|
| 常规 20 篇 64 题 | 46 | 49 | **51** | 领先集中在**诚实拒答**（无答案 6/8 vs 4/8）与**证据引用**（1.9 cites/题）；extractive 与朴素 RAG 相当 |
| 硬题 30 题（按我们已知弱点选） | **10** | 8 | 9 | 无整体优势；**无答案诚实拒答 3/3 vs 1/3** 是唯一稳赢点；hard extractive 摘取是短板（1/12） |

> **诚实结论**：我们不是"更会答题"的系统。价值 = 该拒答就拒答、不编造、每句话可溯源、给小白讲得懂；精确摘取 extractive 细值是已知短板（也是将来 MinerU 表格接入的真正动机）。对比详情：`qa/compare/compare_20260907_191606.md`、`qa/compare/deep_20260907_compare.md`。

### 历史问答口径（工程边界，不再作主成绩）

| 口径 | 成绩 | 说明 |
|---|---|---|
| QASPER 1000 题（全新样本，异源裁判） | **75.0%** | 权威全量口径（旧 233 题 75.5% 双样本一致） |
| 定版估计 | **~79–82%** | 250 条非 pass 当前代码重跑 26% 翻绿外推 |
| 中文 QA 回归（21 篇手写题集） | 176/176 ✅ | 关键词自动判定（v2.0） |
| 检索层改动 | L3 `search_hybrid`+topK12、unknown 收尾"提炼原文给用户" | 查询改写 A/B 负收益已默认关闭 |

**版本说明**：标准版 = 链路模型 `deepseek-chat`（供应商侧 v4-flash），无 MinerU 表格接入、无更强模型；评测裁判默认 `glm-4-flash`（免费、与主链路异源）。完整决策史见 [`qa/CAMPAIGN_20260906-07.md`](qa/CAMPAIGN_20260906-07.md)（§10 定版、§11 定位收尾）。

---

## 🚀 快速开始

### 1. 环境准备

- Python **≥ 3.13**（项目用 [uv](https://docs.astral.sh/uv/) 管理依赖，建议安装 uv）
- 一个 OpenAI 兼容的 LLM API（本项目默认 DeepSeek）

```bash
git clone <repo-url> && cd paperpilot
uv sync                       # 安装依赖（pyproject.toml + uv.lock）
```

### 2. 配置 LLM

```bash
cp .env.example .env          # 然后编辑 .env，填入你的 key
```

```ini
PAPERPILOT_LLM_BASE_URL=https://api.deepseek.com/v1
PAPERPILOT_LLM_API_KEY=sk-你的key
PAPERPILOT_LLM_MODEL=deepseek-chat
PAPERPILOT_LLM_TIMEOUT=120
```

### 3. 启动 Web 工作台

```bash
uv run python web/app.py      # 或 .venv\Scripts\python.exe web\app.py（Windows）
```

打开 **http://127.0.0.1:8000** → 拖入一篇 PDF → 等报告生成（**新论文首次 1~3 分钟**，需 LLM；处理过且产物在 `out_views/` 的论文秒回）→ 中间读报告/原文，右侧提问。

---

## 🛠 CLI 命令

> 论文 PDF 放在 `src/paperpilot/storage/papers/` 下。

```bash
# 一步处理单篇（PDF → report.json）
uv run python cli/main.py <pdf名>

# 只装配已有产物（不调 LLM，缺环节报错）
uv run python cli/main.py <pdf名> --skip-llm

# 全链路强制重跑（调 LLM，较贵）
uv run python cli/main.py <pdf名> --force
```

分环节子命令（claims / view / skeleton / figures / report 等）见 `cli/` 目录。

---

## 🔬 评测与回归护栏

| 命令 | 作用 |
|---|---|
| `uv run python cli/run_qa_v2.py` | 中文 QA 回归（176 题，L0→L3 漏斗）。`--pdf`/`--limit` 定向冒烟；`--save-baseline` 固化基线 |
| `uv run python cli/run_qasper_eval.py --papers N` | QASPER 第三方基准（异源裁判 1~5 分，≥4 pass） |
| `uv run python cli/run_bench.py` | report 层回归护栏：本地重算 21 篇产物与 `bench/baseline.json` 做勾叉 diff（不调 LLM） |
| `uv run python cli/run_compare.py sample/run` | 三列公平对比 harness（B0 直接 LLM / B1 朴素 RAG / B2 本系统，见 `qa/COMPARE_DESIGN.md`） |
| `uv run python qa/negqa/_run_neg.py` 等 | 防幻觉负向 / 稳定性改写 / 导读数值忠实 等一次性评测 runner（可复用） |

**回归状态**（v2.0 定稿）：中文 QA 176/176 ✅；bench 21 篇 4 篇带"已知边界叉"（数学/定理证明型论文，产品决策为不纳入范围）；产物不入库（`out_claims/`、`out_views/`、PDF、`.env` 均 `.gitignore`），`qa/questions/`、`bench/baseline.json` 是资产入库。

---

## 🏗 架构与数据流

```
PDF
 │  pdf_parser → heading → chunker        # 版面 → 标题树 → 语义块
 ▼
claims（LLM 提取四类主张 + 原文证据锚定）
 │  verify_evidence                       # 证据回原文核对（hit/loose/miss）
 ▼
summary（LLM 语义去重 → 角色打标 → 本地规则算分）
 │
 ├─ skeleton（论证骨架：implements/supports/limits/contrasts）
 ├─ figures（图表识别 + 读图指南）
 └─ report（一分钟导读 / 概述 / 章节精读）
 ▼
out_views/<pdf>.report.json              # 一步封装，最终报告
```

### 代码分层

```
src/paperpilot/
├── pipeline.py        # 编排层：一个 PDF → report.json（process_pdf）
├── tools/             # 引擎：pdf_parser/heading/chunker/analyzer/
│                      #       evidence/viewer/skeleton/figures/report/llm
├── prompts/           # 各环节 LLM 提示词（与逻辑分离）
├── models/schema.py   # pydantic 强类型（Claim/ClaimGroup/PaperReport…）
├── agents/            # QA 漏斗节点（纯函数，不依赖 langgraph）
│   └── nodes/         #   report/judge/retrieve/expand/answer…
└── graph/             # LangGraph 接线（qa_graph.py）
web/                   # FastAPI + 前端三栏工作台
cli/                   # 命令行入口 + 评测工具
qa/                    # 手写问题集 + QA/评测结果与报告
bench/                 # 21 篇回归基线
```

### 问答漏斗（L0→L3）

```
用户问题
  L0 Report     报告层够不够？         够 → answer（快，不唤醒 embedding）
  L1 Claims     claims 检索判够？      够 → answer
  L2 Section    按命中章节逐步扩窗口    够 → answer
  L3 Global     独立全文检索（不继承）  够 → answer
                                    不够 → unknown（诚实收尾，不编造）
```

设计取舍（详见 `QA_FUNNEL_DESIGN.md`）：
- **证据驱动逐级下钻**，弃用"问题分类 router"（额外推理任务换毫秒级检索不值）
- **省预算只控块数**（topK/圆心/半径），绝不二次截断 chunk 文本（曾致真实漏检）
- **L3 独立保底**：错误不向上层传导

---

## 📚 文档导航

| 文档 | 内容 |
|---|---|
| `qa/CAMPAIGN_20260906-07.md` | **决策总账**：优化史 + 定版（§10）+ 定位收尾/评测矩阵（§11） |
| `qa/COMPARE_DESIGN.md` | 三列公平对比设计（控制变量/口径/成本记录） |
| `qa/reader/READER_REPORT…` + `FIDELITY_REPORT…` | 帮读主口径：小白问答 15/15、导读数值 102/102 |
| `qa/negqa/` `qa/robust/` `qa/stress/` | 防幻觉 / 稳定性 / 压力边界 报告 |
| `QA_FUNNEL_DESIGN.md` | 问答漏斗架构设计 + 铁律 |
| `DEVELOPMENT_LOG.md` | 开发踩坑记录、决策背景、遗留项、命令速查 |
| `qa/QA_V2_RUN1_REPORT.md` | QA v2 测试报告（Run1→Run2 演进与校准） |
| `design.md` | 早期设计 |

---

## 🧰 开发约定

- **类型检查**：`uv run basedpyright src cli web`（当前 **0 errors / 0 warnings**）。
- **改核心逻辑后**：先 `run_bench.py` 看产物无退化，再 `run_qa_v2.py` 看 QA 不回归。
- **产物不入库**：`out_claims/`、`out_views/`、论文 PDF、`.env` 均在 `.gitignore` 内；`qa/questions/` 与 `bench/baseline.json` 是资产，入库。

---

## 📌 当前定位

> 给被英文/专业门槛劝退的普通人**读论文**：报告讲得懂 + 问答不编造 + 每句可溯源。我们对"帮小白建立正确理解"负责，明确不做"帮高手精确摘原文细节"（那是产品范围决策）。
> 将来可选（证据驱动，已记录在 CAMPAIGN §11.4）：extractive 按题型直取原文、缺失复核扩展到 L3、MinerU 表格接入（需先验证 16G 笔记本体量）、MCP 工具化/多篇对比（远期）。

作者：lx（834659376@qq.com）
