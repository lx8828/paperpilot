# PaperPilot · LLM 论文深度精读系统

输入一篇 PDF 学术论文 → 系统自动完成 **解析 → claims 提取与证据溯源 → 语义去重 → 角色打标 → 重要性打分 → 论证骨架 → 图表指南 → 结构化精读报告**，并支持**带引用溯源的论文问答**（L0→L1→L2→L3 逐级下钻漏斗）。

> **核心信条：所有结论必须能回到原文。**
> 任何产出如果不能指向原文页码/证据片段，就视为不合格——用工程手段对抗 LLM 在学术场景下的幻觉，而不是靠 "prompt 让它别编"。

---

## ✨ 能力一览

| 能力 | 说明 |
|---|---|
| 📄 **单篇精读报告** | 一分钟导读 / 概述 / 核心要点 / 论证骨架 / 章节精读（低分默认折叠可展开）/ 图表一览 |
| 🔍 **证据溯源闭环** | 每条主张绑定原文 `evidence_quote` + 页码，报告里可一键跳回原文并**整行高亮** |
| 💬 **漏斗式问答** | L0(报告层) → L1(claims) → L2(章节邻域扩展) → L3(独立全文检索) → unknown(诚实收尾)，答案带 [n] 引用可点跳原文 |
| 🖥 **三栏工作台** | 左：结构导航 ｜ 中：报告 / 原文 PDF ｜ 右：多轮追问 |
| ✅ **评测体系** | 176 题全量 QA 回归（21 篇 × 双视角问题集）+ bench 回归护栏 |

---

## 📊 评测成绩（标准版 · 2026-09-07 定版）

| 口径 | 成绩 | 说明 |
|---|---|---|
| QASPER 1000 题（全新样本，异源裁判，flash） | **75.0%** | 权威全量口径（旧 233 题 75.5% 双样本一致） |
| 定版估计 | **~79–82%** | 250 条非 pass 用当前代码重跑 26% 翻绿，按旧 pass 不回退外推（未含回归，保守五折 ~78%） |
| 检索层改动 | L3 `search_hybrid`+topK12、unknown 收尾"提炼原文给用户" | 查询改写 A/B 负收益已默认关闭 |

**版本说明**：本版为标准版（链路模型 `deepseek-chat`/v4-flash，无 MinerU 表格接入，无更强模型）。
完整优化/评测/决策史见 [`qa/CAMPAIGN_20260906-07.md`](qa/CAMPAIGN_20260906-07.md)（含 §10 定版）；
逐级实验记录在 `qa/` 下 `QA_V2_NEW10…`、`QASPER_OVERNIGHT_MORNING_REPORT.md`、`QASPER_EVAL_LOG.md` 等。

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

## 📊 评测与回归护栏

| 命令 | 作用 |
|---|---|
| `uv run python cli/run_qa_v2.py` | **176 题全量 QA 回归**（L0→L3 漏斗）。`--pdf`/`--limit` 可定向单篇冒烟；`--save-baseline` 固化基线 |
| `uv run python cli/run_qa_v2.py --pdf x.pdf --limit 6` | 单篇冒烟 |
| `uv run python cli/run_qasper_eval.py --papers N` | **QASPER 第三方基准**（233 题，异源裁判 1~5 分，≥4 pass）|
| `uv run python cli/run_bench.py` | report 层回归护栏：本地重算 21 篇产物，与 `bench/baseline.json` 做勾叉 diff（不调 LLM） |
| `uv run python cli/run_bench.py --save-baseline` | 固化新基线 |

**当前状态**（v2.0 定稿）：
- 中文 QA 回归（21 篇手写题集）：校准后 **176/176 ✅**（关键词自动判定，见 `qa/QA_V2_RUN1_REPORT.md`）
- **QASPER 233 第三方基准：176/233 = 75.5% pass**（V1 基线 50.6% → v2.0 75.5%，优化链与逐级归因见 `qa/QASPER_EVAL_LOG.md`、`qa/FULL_RERUN_20260906_REPORT.md`）
- bench 21 篇中 4 篇带"已知边界叉"（数学/定理证明型论文，产品决策为不纳入范围，见 `DEVELOPMENT_LOG.md`）

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
qa/                    # 手写问题集 + QA 运行结果/汇总
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
| `QA_FUNNEL_DESIGN.md` | 问答漏斗架构设计 + 铁律 |
| `DEVELOPMENT_LOG.md` | 开发踩坑记录、决策背景、遗留项、命令速查 |
| `design.md` | 早期设计 |
| `qa/QA_V2_RUN1_REPORT.md` | QA v2 测试报告（Run1→Run2 演进与校准） |

---

## 🧰 开发约定

- **类型检查**：`uv run basedpyright src cli web`（当前 **0 errors / 0 warnings**）。`Any/Unknown` 噪音已在 `pyproject.toml [tool.basedpyright]` 收敛，保留 unused/拼接等卫生告警。
- **改核心逻辑后**：先 `run_bench.py` 看产物无退化，再 `run_qa_v2.py` 看 QA 不回归。
- **产物不入库**：`out_claims/`、`out_views/`、论文 PDF、`.env` 均在 `.gitignore` 内；`qa/questions/`（手写问题集）与 `bench/baseline.json`（回归基线）是资产，入库。

---

## 📌 当前定位

> 输入一篇论文 → 深度精读报告 + 可溯源问答。下一步规划：MCP 工具化（论文问答作为独立工具供外部 agent 调用）、多篇对比 / 主题追踪 / 论文库（远期）。

作者：lx（834659376@qq.com）
