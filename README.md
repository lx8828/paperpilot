# PaperPilot · 论文帮读器（LLM 精读 + 可溯源问答）

输入一篇 PDF 学术论文 → 系统自动完成 **解析 → claims 提取与证据溯源 → 语义去重 → 角色打标 → 重要性打分 → 论证骨架 → 图表指南 → 结构化精读报告**，并支持**带引用溯源的论文问答**（**v3 两级**：报告层直答 → 全局检索兜底，输出前经**事实闸门**校验）。

**给谁用**：想读论文、却被英文和专业门槛劝退的普通人。系统把论文**讲成能懂的话**（一分钟导读 / 大白话概述 / 引导式问答），同时每个关键结论都带回原文出处——让外行"读得懂、不跑偏、能自己点回原文核实"。

> **核心信条：所有结论必须能回到原文。**
> 任何产出如果不能指向原文页码/证据片段，就视为不合格——用工程手段对抗 LLM 在学术场景下的幻觉，而不是靠 "prompt 让它别编"。

---

## ✨ 能力一览

| 能力 | 说明 |
|---|---|
| 📄 **帮读型报告** | 一分钟导读 / 大白话概述 / 核心要点 / 论证骨架 / 章节精读（低分默认折叠可展开）/ 图表一览 |
| 🔍 **证据溯源闭环** | 每条主张绑定原文 `evidence_quote` + 页码，报告里可一键跳回原文并**整行高亮** |
| 💬 **两级问答（v3）** | `Router(L0 报告层)` 够 → 直答；不够 → `全局检索(L3)` → 判够 → 答 / 诚实收尾（不编造）。答案带 `[n]` 引用，可点跳原文 |
| 🛡 **输出前事实闸门** | 机器件（引用越界、编数/漏数）**硬拦** → 对症修复或兜底拒答；LLM 体检（无支撑断言/偏题/矛盾/含糊/零引用）**软标注** → 前端显示"答案自检（AI 复核）" |
| 📊 **表格内容可检索** | MinerU 表格/公式注入**检索视图**（表格数值可被召回、可被作答读到）；表池**并集候选**（正文 top-12 原样保留 + 追加表池前 2 名） |
| 🖥 **三栏工作台** | 左：结构导航 ｜ 中：报告 / 原文 PDF ｜ 右：多轮追问（含答案自检提示条） |
| ✅ **评测体系** | 帮读问答 + 导读忠实度 + 防幻觉/压力/稳定性 + 三列公平对比 + QASPER 第三方基准（详见下） |

---

## 📊 评测成绩

**产品定位**：论文**帮读器**（面向小白），**不是"答题得分器"**。对"帮高手精确摘原文细节"这类需求，我们如实承认不占优——那是产品范围决策，不是缺陷。

### 1) 帮读定位评测（主口径 · 2026-09-07 定版）

| 测试 | 结果 | 报告 |
|---|---|---|
| A 小白外行问答（3 篇 × 5 题） | **15/15**：讲懂、不编造、给阅读路径 | `qa/reader/READER_REPORT_20260907.md` |
| B 导读数值忠实抽检（5 篇） | **102/102** 数值可回原文、0 缺失 | `qa/reader/FIDELITY_REPORT_20260907.md` |
| 防幻觉负向组（篡改数值/不存在物） | **20/20**：0 编造、0 误断言缺失 | `qa/negqa/NEGQA_REPORT_20260907.md` |
| 对抗/诱导压力（12 题） | **12/12**：不顺着假前提、评价缺失项不编 | `qa/stress/STRESS_REPORT_20260907.md` |
| 压力/边界 | 超长 168K 字符 **8/8** pass；坏 PDF 干净报错不崩溃；扫描件不可用（需 OCR/MinerU） | 同上 |
| 稳定性 / 同义改写（30×3） | 3 问法 pass 一致 **70%**（平均跨度 0.53 → 报数自带 ±） | `qa/robust/ROBUST_REPORT_20260907.md` |

### 2) 三列公平对比（B0 直接 LLM / B1 朴素 RAG / B2 本系统 · 同题同裁判盲判）

| 池 | B0 | B1 | B2 | 结论 |
|---|---|---|---|---|
| 常规 20 篇 64 题 | 50 | 48 | **53** | 领先集中在**诚实拒答**（无答案 6/8 vs 3/8 / 2/8）与**证据引用**（2.62 cites/题 vs 0）；extractive 与朴素 RAG 相当 |
| 硬题 30 题（按已知弱点选） | 15 | 8 | 15 | 与直读**打平**；**无答案诚实拒答 3/3 vs 1/3、0/3** 仍是唯一稳赢点 |

> **诚实结论**：我们不是"更会答题"的系统。价值 = 该拒答就拒答、不编造、每句话可溯源、给小白讲得懂。
> **显著性**（配对 McNemar，`qa/compare/_signif.py`）：单池里没有任何一对能证明 B2 显著优于 B0/B1；
> 两池合并后 **B2 vs B1 翻转 16:4 → p=0.012 显著**（claims 证据链 + 漏斗 + 收尾的真实效应），**B2 vs B0 12:9 → p=0.664**（边际）。
> **扩样本所需题量**（`qa/compare/_power.py`）：B2vsB1 ≈ 107 题、B2vsB0 ≈ 365 题。
> ⚠️ **口径警告**：B0 的 prompt/模型未改却从 46→50（深水 10→15）——上游 `deepseek-chat` **本身在漂移**，
> 跨时间的数字**不可归因于我们的改动**；只有**同一次运行内**的三列可比。报告：`qa/compare/compare_20260911_040708.md`。

### 3) 检索/问答水位（**注意口径与日期**）

| 口径 | 成绩 | 说明 |
|---|---|---|
| **v3 两级定稿 A/B**（250 题，同裁判） | V0(v3) **203/250 = 81.2%** ≥ A(v2 四层) 201/250 = 80.4% | LLM 调用 **3.7 vs 5.6（−34%）**；据此 v2 四层图下线归档（`archive/qa_funnel_v2/`），**不提供 ask_v2 回退** |
| QASPER 503 题（1000 随机样本的无偏子集，**2026-09-11 口径**） | **424/503 = 84.3%**｜有答案题 **388/451 = 86.0%** | 异源裁判（GLM）1~5 分，≥4 pass。原始记录 `qa/qasper_run_20260911_072510.json` |
| └ 剔除不可归因项后〔能力上限口径〕 | 有答案题 **388/413 = 93.9%** | 剔除 36 题"表格=PNG 无解" + 2 题判分存疑；**须与上一行并列，不得单独宣称**｜`qa/recall/FAIL_ATTRIB_20260911.md` |
| └ ＋ arXiv 原始 PDF **表格通道**（MinerU） | **438/503 = 87.1%**｜有答案题 402/451 = 89.1% | 那 36 题在 QASPER 里 gold 出自表格、而 QASPER 只给 PNG；补原始 PDF 后**回收 14/36（39%）**｜`qa/recall/QASPER_TBL_EXP_20260911.md` |
| 中文 QA 回归（21 篇手写题集） | **176/176** ✅ | 关键词自动判定（v2.0） |
| **表格能力线（2026-09-12 起）** | 检索层 **目标表进候选 +4、零回退**；端到端单轮 **15/36 → 19/36** | P1 表表示 / P3 读表 prompt / caption 修复 / 表池并集候选（**已设为默认**）；P2 双写为负结果已收口。汇总：`qa/recall/TABLE_LINE_STATUS_20260912.md` |

> ⚠️ 表格线的改动（并集默认开、caption 修复、P1/P3）**尚未在 503 全量上复跑**——上表 503 数字是 09-11 口径，
> 不要与 09-12 之后的检索层改动混算。

**版本说明**：链路模型 `deepseek-chat`；评测裁判默认 `glm-4-flash`（免费、与主链路异源）。完整决策史见 [`qa/CAMPAIGN_20260906-07.md`](qa/CAMPAIGN_20260906-07.md)。

---

## 🚀 快速开始

### 1) 环境准备

- Python **≥ 3.13**（项目用 [uv](https://docs.astral.sh/uv/) 管理依赖）
- 一个 OpenAI 兼容的 LLM API（默认 DeepSeek）
- 〔可选〕**MinerU**：用于把表格/公式解析进**检索视图**（见「架构」）。缺失时报告照常生成、**问答不可用**（会明确告知原因，不静默降级）

```bash
git clone <repo-url> && cd paperpilot
uv sync                       # 安装依赖（pyproject.toml + uv.lock）
```

论文 PDF 放 **`assets/papers/`**；产物落在 **`assets/artifacts/`**（`out_claims` / `out_views` / `out_mineru`，均不入库）。

### 2) 配置 LLM

```bash
cp .env.example .env          # 然后编辑 .env，填入你的 key
```

```ini
PAPERPILOT_LLM_BASE_URL=https://api.deepseek.com/v1
PAPERPILOT_LLM_API_KEY=sk-你的key
PAPERPILOT_LLM_MODEL=deepseek-chat
PAPERPILOT_LLM_TIMEOUT=120
```

### 3) 启动 Web 工作台

```bash
uv run python web/app.py      # 或 .venv\Scripts\python.exe web\app.py（Windows）
```

打开 **http://127.0.0.1:8000** → 拖入一篇 PDF → 等报告生成（**新论文首次 1~3 分钟**，需 LLM + MinerU；
已有产物的论文秒回）→ 中间读报告/原文，右侧提问。

---

## 🛠 CLI 命令

```bash
# ── 摄取 / 报告 ────────────────────────────────────────────────
uv run python cli/main.py <pdf名>            # 一步处理单篇（PDF → report.json）
uv run python cli/main.py <pdf名> --skip-llm # 只装配已有产物（不调 LLM，缺环节报错）
uv run python cli/main.py <pdf名> --force    # 全链路强制重跑（调 LLM，较贵）
uv run python cli/run_ingest.py <pdf名>      # 摄取流水线：论文库 → MinerU(检索) → 报告 → 产物库（幂等）

# ── 分环节（调试用）────────────────────────────────────────────
uv run python cli/run_claims.py <pdf名>      # claims 提取 + 证据回核
uv run python cli/run_summary.py <pdf名>     # 语义去重 / 角色打标 / 打分
uv run python cli/run_skeleton.py <pdf名>    # 论证骨架
uv run python cli/run_figures.py <pdf名>     # 图表识别 + 读图指南
uv run python cli/run_report.py <pdf名>      # 结构化精读报告
uv run python cli/run_view.py <pdf名>        # 装配视图 / 产物检查
```

---

## ⚙️ 行为开关（env）

**这一节是"现在实际在跑什么"的权威说明**：默认值直接决定线上行为，回退通常只改一行/一个 env。

### 默认**生效**的改动

| 开关 | 默认 | 作用 |
|---|---|---|
| `PAPERPILOT_VALIDATOR_GATE` | `1`（开） | 输出前事实闸门：HIGH → 对症修复 / 兜底拒答；MID/LOW → 前端"答案自检"标注 |
| `PAPERPILOT_VALIDATOR_LLM` | `1`（开） | 闸门的 LLM 体检（无支撑断言/偏题/矛盾/含糊；零引用走**格式体检**） |
| `PAPERPILOT_EXT_QUOTA` | `2` | **表池并集候选**：正文 top-12 **原样保留**，追加表池前 2 名里没有的（候选 12 → 13/14）。`=0` 回退 |
| `PAPERPILOT_EXT_RRF_ALPHA` | `0.5` | 外部块（表格/公式）两路权重 `(2α, 2(1-α))`；0.5 = 生产原样（文本块恒 1:1 不动） |
| `PAPERPILOT_MINERU_INJECT` | `1`（开） | 把 MinerU 表格/公式文本按页注入**检索视图**（`=0` 可关，评测用） |

### 默认**关**的实验（需显式开启）

| 开关 | 默认 | 作用 / 为什么默认关 |
|---|---|---|
| `PAPERPILOT_QASPER_TABLES` | 关 | QASPER 外部表格通道（评测用；线上走 MinerU 注入） |
| `PAPERPILOT_TABLE_EMBED_SUMMARY` | 关 | **P2 双写**（表块向量侧改喂语义摘要）——两次实测均**不及现状**，已收口：`qa/recall/P2_DUALWRITE_20260912.md` |
| `PAPERPILOT_TABLE_V1` | 关 | `=1` 复现改造前的表表示 + prompt 布局（同日配对 A/B 用） |
| `PAPERPILOT_QUERY_REWRITE` | `0` | 查询改写（实测负收益，默认关） |
| `PAPERPILOT_RETRIEVE_SECTION_CAP` | `0` | 保序节级配额去重（>0 开启） |
| `PAPERPILOT_VALIDATOR_REPAIR_MID` | 关 | mid 级问题是否也触发修复（默认只标注不修） |
| `PAPERPILOT_VALIDATOR_MISSING` | 关 | 是否提示"原文可能还有未答要点"（好答案上误报多，默认关） |
| `PAPERPILOT_USE_MINERU` | 关 | `=1` 走遗留"全链 MinerU"模式（chunks 直接用 MinerU） |
| `PAPERPILOT_CHUNK_VIEW_DIR` | 空 | 向量缓存目录覆盖（A/B 两臂各用一份，避免来回覆盖重建） |

MinerU 相关：`PAPERPILOT_MINERU_VENV`（默认 `.venv-mineru`）、`PAPERPILOT_MINERU_BACKEND`（`pipeline`）、
`PAPERPILOT_MINERU_CMD`（覆盖可执行入口）、`PAPERPILOT_MINERU_TIMEOUT`（900s）。

---

## 🏗 架构与数据流

### 两条解析通道（2026-09-10 定，方案 B3）

```
PDF ──┬─ pymupdf  ──→ 页码 / 版面 / chunk 空间 / claims 锚点 ──→ 报告（report.json）
      │                （cites、前端高亮、claims 锚点都以此为唯一 id 空间）
      └─ MinerU   ──→ 表格 / 公式文本 ──按页注入「检索视图」──→ 问答检索与作答
                       （claims 仍读 pymupdf 原文：表格只进检索、不进 claims）
```

MinerU 失败或无 GPU 时：**报告照常生成**（pymupdf 渲染 + 图表回退 pymupdf 抽取），
**问答直接失败并说明原因**（`qa_blocked_reason()`，不静默降级）。摄取状态落 `ingest.json`（版本/时间戳，幂等）。

### 问答链路（v3 两级 + 输出闸门）

```
用户问题
  Router   L0 报告层（overview + core_points）够不够？── 够 ──→ 直答（快，不唤醒 embedding）
                                            └─ 不够 ──→ 全局检索 L3（向量 + BM25 混合，RRF 融合）
                                                        └─ 判够 ──→ 答（带 [n] 引用）
                                                        └─ 不够 ──→ 诚实收尾 / 提炼原文
  输出前闸门 validator.gate()：
     HIGH（引用越界 / 编数漏数）→ repairer 对症修复（Self-Refine / CRAG）→ 修不动则兜底话术
     MID / LOW（无支撑断言 / 偏题 / 矛盾 / 含糊 / 零引用）→ 原样输出 + 前端"答案自检"标注
```

> v2 四层漏斗（L0→L1 claims→L2 扩窗→L3）**已下线归档**到 `archive/qa_funnel_v2/`（含设计稿与旧节点快照）；
> 定稿依据见上表「v3 两级定稿 A/B」。

### 代码分层

```
src/paperpilot/
├── ingest.py          # 摄取流水线：论文库 → MinerU(检索) → pipeline(报告) → 产物库
├── pipeline.py        # 报告编排层：一个 PDF → 各环节产物 → report.json（process_pdf）
├── tools/             # 引擎：pdf_parser/heading/chunker/analyzer/evidence/viewer/
│                      #       skeleton/figures/report/llm/mineru_bridge
├── prompts/           # 各环节 LLM 提示词（与逻辑分离）
├── models/schema.py   # pydantic 强类型（Chunk/Claim/ClaimGroup/PaperReport…）
├── agents/            # 问答节点（纯函数，不依赖 langgraph）
│   ├── nodes/         #   report/judge/search_l3/generate_answer/answer_unknown…
│   └── document_cache.py # 两套视图：ordered_chunks（报告/溯源） / retrieval_chunks（检索，含表公式）
├── components/        # 组件门面（显式化）：router / retriever / generator / validator /
│                      #   repairer / context_builder / query_rewriter / splitter / numbers / reranker
└── graph/             # LangGraph 接线：qa_graph_v3.py（v3 两级是**唯一**检索链）
web/                   # FastAPI + 前端三栏工作台（index.html）
cli/                   # 命令行入口 + 评测工具
qa/                    # 手写问题集 + 各轮评测报告与复算脚本
bench/                 # 回归基线（baseline.json）
assets/papers/         # 论文 PDF（不入库）
assets/artifacts/      # 产物（不入库）
archive/qa_funnel_v2/  # v2 四层漏斗快照（含设计稿），只作对照
```

---

## 🔬 评测与回归护栏

| 命令 | 作用 |
|---|---|
| `uv run python cli/run_qasper_eval.py --papers N` | QASPER 第三方基准（异源裁判 1~5 分，≥4 pass） |
| `uv run python cli/run_qa_v2.py` | 中文 QA 回归（手写题集）。`--pdf`/`--limit` 定向冒烟；`--save-baseline` 固化基线 |
| `uv run python cli/run_bench.py` | report 层回归护栏：本地重算产物与 `bench/baseline.json` 勾叉 diff（不调 LLM） |
| `uv run python cli/run_compare.py sample/run` | 三列公平对比 harness（B0/B1/B2，见 `qa/COMPARE_DESIGN.md`） |
| `uv run python cli/run_chunk_eval.py` / `run_retrieval_eval.py` | 分块 / 检索层专项评估 |
| `uv run python qa/recall/_selftest_tables_20260912.py` | 表块关键函数**边界自测**（caption 清噪含罗马数字、表头/摘要、口径解析、权重） |
| `uv run python qa/recall/_selftest_validator_20260913.py` | 闸门自测（零引用分级、格式体检、gate 动作不变回归）；`PP_LLM=1` 加测真实体检 |

**回归纪律**：改核心逻辑后 → 先 `run_bench.py` 看产物无退化 → 再跑对应自测 / 定向 QA。
本 README 的评测数字**必须带口径与日期**；单次端到端运行的 churn 在 8%~25%，
**<3 题的效应不可判**（见 `qa/recall/UNION_AND_CONTEXT_20260911.md` §2）。

---

## 📚 文档导航

| 文档 | 内容 |
|---|---|
| `qa/CAMPAIGN_20260906-07.md` | **决策总账**：优化史 + 定版 + 定位收尾 |
| `DEVELOPMENT_LOG.md` | 开发踩坑记录、决策背景、遗留项、命令速查 |
| `docs/RAG_COMPONENT_NOTES.md` | 组件层设计依据（Router/Retriever/Generator/Validator 的取舍与实测） |
| `qa/recall/TABLE_LINE_STATUS_20260912.md` | **表格能力线总索引**：哪些改动在生效 / 哪些是默认关的实验 / 哪些做法已被否证 |
| `qa/review/RESPONSES_20260913.md` | 外部代码审查**逐条核验与处置**台账 |
| `qa/COMPARE_DESIGN.md` | 三列公平对比设计（控制变量/口径/成本） |
| `qa/reader/` `qa/negqa/` `qa/robust/` `qa/stress/` | 帮读主口径 / 防幻觉 / 稳定性 / 压力边界 报告 |
| `archive/qa_funnel_v2/QA_FUNNEL_DESIGN.md` | v2 四层漏斗架构设计 + 铁律（**已归档**，只作对照） |
| `design.md` | 早期设计 |

---

## 🧰 开发约定

- **类型检查**：`pyproject.toml` 里配了 `[tool.basedpyright]`，但**当前环境未安装**该工具
  （`uv run basedpyright` 会失败）。需要类型门禁时先 `uv add --dev basedpyright`；
  日常改动用 `python -m compileall src cli web qa` + 上面的自测脚本兜底。
- **产物不入库**：`assets/artifacts/`、`assets/papers/*.pdf`、`*.log`、`.env` 均在 `.gitignore` 内；
  `qa/questions/`、`qa/**/*.md`（报告）、`bench/baseline.json`、复算脚本 `_*.py` 是**资产，入库**。
- **评测口径**：目标表定位用 `qa/recall/_tgt.py`（编号 ∪ 内容联合判定），不要在各脚本里另写 `cap_key`。

---

## 📌 当前定位

> 给被英文/专业门槛劝退的普通人**读论文**：报告讲得懂 + 问答不编造 + 每句可溯源。
> 我们对"帮小白建立正确理解"负责，明确不做"帮高手精确摘原文细节"（产品范围决策）。
>
> 已做：MinerU 表格/公式进检索视图、表池并集候选（默认开）、输出闸门前端可见。
> 已知短板：extractive 细值抽取、表格题在上下文里仍答不出的那几道（属答案抽取层）、
> 以及"跨时间的三列对比不可归因"这类评测口径问题——都记在 `DEVELOPMENT_LOG.md` 与各专题报告里。

作者：lx（834659376@qq.com）
