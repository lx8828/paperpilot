# PaperPilot · 论文帮读器（LLM 精读 + 可溯源问答）

输入一篇 PDF 学术论文 → 系统自动完成 **解析 → claims 提取与证据溯源 → 语义去重 → 角色打标 → 重要性打分 → 论证骨架 → 图表指南 → 结构化精读报告**，并支持**带引用溯源的论文问答**（**v3 两级**：报告层直答 → 不够时 **全文直读**（默认，`PAPERPILOT_QA_READER=fullctx`）/ **块级检索**（RAG2，`=retrieval` 切回），输出前经**事实闸门**校验）。

**给谁用**：想读论文、却被英文和专业门槛劝退的普通人。系统把论文**讲成能懂的话**（一分钟导读 / 大白话概述 / 引导式问答），同时每个关键结论都带回原文出处——让外行"读得懂、不跑偏、能自己点回原文核实"。

> **核心信条：所有结论必须能回到原文。**
> 任何产出如果不能指向原文页码/证据片段，就视为不合格——用工程手段对抗 LLM 在学术场景下的幻觉，而不是靠 "prompt 让它别编"。

---
## 🧭 检索能力地图

| 能力 | 角色 | 当前状态 | 对应评测 / 说明 |
|---|---|---|---|
| **RAG1** | **论文级检索**：方向问题 → 候选论文 | 现行；稳定交付工作区是 **top-5**，LLM listwise 精排上限为 **top-10** | **LitSearch**（64,183 篇 / 597 查询） |
| **RAG2** | **论文块级 reader 检索**：块召回 → 生成答案 | 代码保留，**默认关闭**；只作 A/B 对照与回退 | r2dev 证据召回 / reader 判定 |
| **MultiAnswer** | **论文集合级多答案检索**：哪几篇满足问题 → 给出**回答** | **独立页 `/ma`**（工作台左下角进入），语料 = 本工作区那批论文，**固定 ≤10 篇** | **QAMPARI**（对外 passage 基准）+ 三簇论文集（内部） |
| **Fullctx** | 报告层不够时，全文直读作答 | **论文问答默认 reader** | 三列公平对比与端到端 QA |

> **QAMPARI 不是独立模块**。它属于 MultiAnswer，用来检查多答案检索能力；关系就像 **LitSearch 是 RAG1 的评测**。
> **RAG2 也不是 MultiAnswer**：前者是块级 reader 检索，后者是论文集合级多答案检索。

---

## ✨ 能力一览

| 能力 | 说明 |
|---|---|
| 📄 **帮读型报告** | 一分钟导读 / 大白话概述 / 核心要点 / 论证骨架 / 章节精读（低分默认折叠可展开）/ 图表一览 |
| 🔍 **证据溯源闭环** | 每条主张绑定原文 `evidence_quote` + 页码，报告里可一键跳回原文并**整行高亮** |
| 💬 **论文问答（v3）** | `Router(L0 报告层)` 够 → 直答；不够 → **`fullctx` 全文直读（默认）**；`PAPERPILOT_QA_READER=retrieval` 可切回 **RAG2 块级检索**（对照/回退，不在前端暴露）。`fullctx` 引用形如 `[P{n}·§...·¶cid]`，检索路径为 `[n]`，前端两者都可点回原文；输出前经事实闸门 |
| 🧩 **MultiAnswer 多答案** | 论文集合级“哪几篇满足条件”：从工作台**左下角**进入**独立页 `/ma`**，语料 = 本工作区那批论文（**固定 ≤10 篇**），产出**回答**（逐篇判定 → LLM 汇总）+ 依据。**QAMPARI 是它的对外检索评测，不是独立模块** |
| 🛡 **输出前事实闸门** | 机器件（引用越界、编数/漏数）**硬拦** → 对症修复 → 修不动则 **LLM「补充说明」**（带「非系统作答」标注）→ 才兜底拒答；LLM 体检（无支撑断言/偏题/矛盾/含糊/零引用）**软标注** → 前端显示“答案自检（AI 复核）” |
| 📊 **表格内容可检索** | MinerU 表格/公式注入**检索视图**（表格数值可被召回、可被作答读到）；表池**并集候选**（正文 top-12 原样保留 + 追加表池前 2 名） |
| 🖥 **三栏工作台** | 左：结构导航 ｜ 中：报告 / 原文 PDF ｜ 右：多轮追问（含答案自检提示条） |
| ✅ **评测体系** | 帮读问答 + 导读忠实度 + 防幻觉/压力/稳定性 + 三列公平对比 + QASPER；RAG1 对应 LitSearch，MultiAnswer 对应 QAMPARI + 三簇论文集（详见下） |

---

## 📊 评测成绩

> **要一份完整汇总（做了什么 / 上线了什么 / 上线效果 / 没上线的为什么 / 大 N 重采样 / 可信度支撑）？
> 看 [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md)** —— 对外汇报的第一入口，每个数字都带出处。
> 下面是口径与水位明细。

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
| QASPER 503 题（1000 随机样本的无偏子集，**2026-09-11 口径**） | **424/503 = 84.3%**｜有答案题 **388/451 = 86.0%** | 异源裁判（GLM）1~5 分，≥4 pass。原始记录 `qa/qasper_run_20260911_072510.json`（**本地运行记录，未入库**） |
| └ 剔除不可归因项后〔能力上限口径〕 | 有答案题 **388/413 = 93.9%** | 剔除 36 题"表格=PNG 无解" + 2 题判分存疑；**须与上一行并列，不得单独宣称**｜`qa/recall/FAIL_ATTRIB_20260911.md` |
| └ ＋ arXiv 原始 PDF **表格通道**（MinerU） | **438/503 = 87.1%**｜有答案题 402/451 = 89.1% | 那 36 题在 QASPER 里 gold 出自表格、而 QASPER 只给 PNG；补原始 PDF 后**回收 14/36（39%）**｜`qa/recall/QASPER_TBL_EXP_20260911.md` |
| 中文 QA 回归（手写题集：现 **31 篇 / 304 题**） | 历史成绩 **176/176**（21 篇口径 · 09-09 基线） | 关键词自动判定；⚠️ **2026-09-22 第一组 5 篇的题集已按 gold 引文口径重建（70 题 / 5 篇）**，旧数字与现题集不再一一对应 |
| **表格能力线（2026-09-12 起）** | 检索层 **目标表进候选 +4、零回退**；端到端单轮 **15/36 → 19/36** | P1 表表示 / P3 读表 prompt / caption 修复 / 表池并集候选（**已设为默认**）；P2 双写为负结果已收口。汇总：`qa/recall/TABLE_LINE_STATUS_20260912.md` |

> ⚠️ 表格线的改动（并集默认开、caption 修复、P1/P3）**尚未在 503 全量上复跑**——上表 503 数字是 09-11 口径，
> 不要与 09-12 之后的检索层改动混算。

**版本说明**：链路模型 `deepseek-chat`；评测裁判默认 `glm-4-flash`（免费、与主链路异源）。完整决策史见 [`qa/CAMPAIGN_20260906-07.md`](qa/CAMPAIGN_20260906-07.md)。

---

## 🧩 MultiAnswer · 多答案开放域检索（独立页 `/ma` · 2026-10-06）

> **定位**：MultiAnswer 是论文集合级“哪几篇满足条件”的能力。
> **QAMPARI 属于 MultiAnswer 的评测体系**，用于检查多答案检索与判官能力；
> 它不是独立模块——对应关系是 **QAMPARI : MultiAnswer = LitSearch : RAG1**。
> **RAG2 另有其人**：RAG2 是论文块级 reader 检索，默认关闭、只作对照/回退。

**入口与形态（2026-10-06 简化：不再有第二条路、没有 N）**

| | |
|---|---|
| **入口** | 工作台**左栏左下角**「🎯 多答案展示」→ 跳**独立页** `GET /ma?pdfs=…` |
| **语料** | **本工作区的那批论文**（与单篇问答同一口径 `askSources()`），**固定 ≤10 篇** |
| **N** | **没有 N 选择器** —— 篇集由 URL 的 `pdfs` 唯一确定，后端强制 `n = len(pdfs)` |
| **产物** | **回答**（把逐篇判定**塞进 LLM 汇总**）＋「依据」（逐篇明细，可折叠） |

入口这样收拢，是因为两者都是「**主题类查询 → 10 篇论文**」—— 入口统一、下游复用同一批语料。

**离线评测（能力仍在，只是不经前端）**

| 口径 | 语料 | 规模 |
|---|---|---|
| 实时档（离线 CLI） | RAG1 现场检索 → 取料下载 PDF → 切块 | ≤10 篇 |
| 论文域评测 | 预生成的固定论文簇 | **50 篇/簇**（3 簇 × 50 = 150） |

**规模口径（统一按这版读）**

- RAG1 的稳定交付工作区是 **top-5**；LLM listwise 精排上限是 **top-10**。
- 在线一次**最多 10 篇**；它表示“链路可跑通”，**不表示已验证 10 篇效果好**。
- **3 簇 × 50 = 150 篇**只是离线评测总量，不等于在线一次输入 150 篇。
- 换检索源、换语料或换规模后，既有 P/R/F1 **不可直接比较**，必须重标真值。

**评测归属**

| 能力 | 评测 | 数据 / 报告入口 |
|---|---|---|
| RAG1 | **LitSearch**（64,183 篇 / 597 查询） | `retrieval/results/` 的 LitSearch 报告 |
| RAG2 | r2dev 证据召回 / reader 判定 | `evals/RESULTS.md` 的 L2 + `retrieval/results/` |
| MultiAnswer · 对外 | **QAMPARI**（LoFT，100 题，128k passage 档） | `evals/RESULTS.md` 的 L4 + `retrieval/results/R2_QAMPARI_*.md` |
| MultiAnswer · 论文域 | 三簇 × 50 篇 = 150 篇、28 组合、三判官多数票 | `retrieval/results/R2_PROD_FINAL.csv` + `R2_PROD_FINAL_20261001.md` |

> QAMPARI 验证的是 multi-answer / coverage 方法在 passage 语料上的表现，**不能替代论文语料的端到端结论**。
> `evals/reports/*.jsonl` 是本地生成物；干净 clone 应看已入库的 `evals/RESULTS.md`、`evals/baselines/metrics.json` 和 `retrieval/results/` 报告。

---

## 🚀 快速开始

**先选一条路**（不想配 Key 就走 A —— 2 分钟能看到全部效果）：

| | 安装（一条命令） | 启动（一条命令） | 需要什么 |
|---|---|---|---|
| **A. 演示模式**（推荐先试） | `uv sync --no-install-package sentence-transformers` | `uv run python web/app.py --mock` | **只要 uv** |
| **B. 真跑**（真 LLM + 真向量） | `uv sync` | 配好 `.env` → `uv run python web/app.py` | OpenAI 兼容 API Key（可选 GPU + MinerU） |

### 0) 前置：装 uv（一行，不需要先装 Python）

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

uv 会按 `requires-python` 自动装好 Python 3.13。

### 1) 安装（一条命令）

```bash
git clone <repo-url> && cd paperpilot
uv sync            # 依赖 + 锁文件（uv.lock）
```

> 演示模式用不到向量模型（bge-m3 / torch，约 2~3 GB）→ 可以省掉：
> `uv sync --no-install-package sentence-transformers`（`uv run pytest -q` 也用这条）。

### 2) 启动（一条命令，免 Key）

```bash
uv run python web/app.py --mock
```

打开 **http://127.0.0.1:8000** → 拖入仓库自带的 **[`demo/demo_paper.pdf`](demo/demo_paper.pdf)** →
**几秒**出报告 → 右侧提问。

**演示模式（`--mock`）的诚实边界**（页面顶部会有黄色横幅提示）：

| | |
|---|---|
| 内容是 | 概述 / 主张 / 导读 / 答案是**固定示例**（带"（演示内容）"前缀） |
| 仍然是**真的** | 摄取 → 索引 → 检索 → 输出闸门 → **引用锚点**（答案里的 `[n]` 解析成真实页码 + 原文片段） |
| 不需要 | API Key、GPU、模型下载、MinerU（实测一篇 **0.3 秒**跑完，`sentence_transformers` 从未加载） |
| 细粒度开关 | `PAPERPILOT_MOCK_LLM=1`、`PAPERPILOT_MOCK_EMBED=1`、`PAPERPILOT_MINERU=0`（`--mock` 一键打开前两个） |

### 3) 真跑：配置 API Key

```bash
cp .env.example .env          # 然后填入你的 key
```

```ini
PAPERPILOT_LLM_BASE_URL=https://api.deepseek.com/v1    # 任何 OpenAI 兼容端点
PAPERPILOT_LLM_API_KEY=sk-你的key
PAPERPILOT_LLM_MODEL=deepseek-chat
PAPERPILOT_LLM_TIMEOUT=120

# 〔可选〕异源裁判模型：评测打分用（不配也能跑产品）
PAPERPILOT_JUDGE_BASE_URL=https://open.bigmodel.cn/api/paas/v4
PAPERPILOT_JUDGE_API_KEY=...
PAPERPILOT_JUDGE_MODEL=glm-4-flash
```

没配 key 就启动 → 上传时返回**明确的**"LLM 未配置"提示（不静默失败）。

### 4) 模型预下载（可选，建议）

向量检索用 `BAAI/bge-m3`（约 2.2 GB）。**首次提问**会自动下载；想提前下好 / 离线部署：

```bash
uv run python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-m3')"
```

- 缓存位置：`~/.cache/huggingface`（Windows：`C:\Users\<你>\.cache\huggingface`）
- 已缓存后想彻底离线：设 `HF_HUB_OFFLINE=1`
- **不想下**：用演示模式（`PAPERPILOT_MOCK_EMBED=1`，确定性词袋哈希向量）

### 5) MinerU（**默认 chunk 骨架**；不装则自动降级）

**2026-09-22 起 MinerU 是默认骨架**（报告链与检索**同源**读它）→ **属于"强烈建议装"**：不装不报错，但表格数值会缺。

| 场景 | 行为 |
|---|---|
| 装好且成功 | 报告与检索/问答读**同一份** MinerU chunks：表格数值**可召回、可作答** |
| 没装 / **失败**（无 GPU、显存不足、产物损坏） | **自动走 pymupdf 备用路**：报告与问答**都照常可用**，只是**表格数值类问题会弱**；降级原因记在 `parse_degraded` 与前端提示里（**不再拒答**） |
| **主动跳过**（`PAPERPILOT_MINERU=0`） | 同"没装"，但状态是 `skipped` 而非 `failed`（区分"我不用"与"跑挂了"） |
| 摄取**进行中**时提问 | 唯一会被拦的情形：明确告知"正在解析，请稍候"（不是失败） |

**安装**（独立 venv，与主环境隔离；实测 MinerU 3.4.5 / Python 3.12 / torch cu128）：

```bash
uv venv .venv-mineru --python 3.12
.venv-mineru/Scripts/pip install -U "mineru[all]"     # Windows
# .venv-mineru/bin/pip install -U "mineru[all]"       # macOS / Linux
uv run python web/app.py        # 摄取时自动调用（默认 backend=pipeline）
```

需要 GPU/显存；包名与版本以 [MinerU 官方文档](https://github.com/opendatalab/MinerU) 为准。
已装在别处可用 `PAPERPILOT_MINERU_VENV` / `PAPERPILOT_MINERU_CMD` 指过来。

### 6) 上传后的接口

**摄取是后台任务**：上传**立即返回**（`202 + job_id`），页面实时显示阶段与耗时，可**取消 / 重试**；
问答在产物就绪前会被闸门明确告知"正在解析，请稍候"（不是失败）。

| 接口 | 作用 |
|---|---|
| `POST /api/report` | 上传 → 提交后台 job（同内容重传：**幂等秒回报告**，不排任务） |
| `GET /api/job/{id}` | 任务状态：`queued/running/ready/failed/cancelled` + 各阶段耗时 |
| `GET /api/jobs/latest?pdf=` | 该论文最近一次任务（刷新页面后恢复进度） |
| `POST /api/job/{id}/cancel` | 请求取消（阶段边界生效；MinerU 会真终止子进程） |
| `POST /api/job/{id}/retry` | 重试（已完成的阶段**自动复用**，通常快很多） |
| `GET /api/report/{name}` | 取报告 JSON（含 `upload_note` / `mineru_warning`） |
| `GET /ma` | **多答案展示独立页**（`web/multianswer.html`）。URL 带 `?pdfs=a.pdf,b.pdf,…`（裸文件名，相对 `assets/papers/`）—— 这就是它的**全部语料**；缺参数时页面给引导 |
| `GET /api/meta` | 运行模式（演示模式横幅用它）。另含 `ask_wait_s` / `ask_concurrency`，以及 MultiAnswer 契约 `ma_live_max` / `ma_stages`（`ma_n_options` / `ma_n_max` 仍返回，但**只服务于离线评测**，独立页不使用） |
| `GET /pdf/{name}` | 取原始 PDF（前端 pdf.js 渲染与"点引用跳原文高亮"用）。另有 `GET /`（工作台页面）与 `/vendor`（静态资源挂载） |
| `POST /api/ask` | 提问（**不阻塞事件循环**：同步端点走线程池 + 并发闸门；摄取**进行中**时明确告知"正在解析，请稍候"——那不是失败） |
| `POST /api/direction/search` | 方向问题 → RAG1 检索候选论文（离线编排 CLI `run_rag2_flow.py` 复用该入口） |
| `POST /api/multianswer/fetch` | 候选 arXiv id → 取料下载 + 摄取（后台 job；失败篇逐条返回） |
| `POST /api/multianswer/search` | 独立页 `/ma` 用：`{question, pdfs}` —— 传了 `pdfs` 就以**它**为语料，后端**强制 `n=len(pdfs)`** 且 `≤ ma_live_max`；★ `corpus` **不是请求字段**。**完成后 job 带** `answer`（LLM 汇总的回答）/ `answer_ok` / `answer_error` / `evidence_text`（逐篇依据）/ `candidates`。**不带 `pdfs`** 时用服务端语料（`n` 生效）—— 那是**离线评测**路径，前端不用 |

**上传门（2026-09-14）**：只做两条"边界上花 3 行、省掉一次注定失败的分钟级摄取"的校验 ——
① 非 `.pdf` 扩展名 → `400`；② 缺 `%PDF-` 文件头（把 `.docx`/`.txt` 改名成 `.pdf`）→ `400`
（**允许前 1KB 有前导垃圾**，规范与真实文件都允许，避免误杀）；
③ 超过大小上限 → `413`（默认 200MB，`PAPERPILOT_MAX_PDF_MB=0` 关闭）。
**刻意不做**：页数上限（要**先解析一遍**才知道，而 3000 页的合集往往是合法需求）、
解压炸弹/恶意 PDF 防护（本地单用户没有攻击者模型，真缓解是子进程沙箱）——
若哪天变成多用户服务，这套要整体重做。

论文 PDF 放 **`assets/papers/`**；产物落在 **`assets/artifacts/`**（`out_claims` / `out_views` /
`out_mineru` / `out_jobs`，均不入库）。

---

## 🛠 CLI 命令

```bash
# ── 摄取 / 报告 ────────────────────────────────────────────────
uv run python cli/main.py <pdf名>            # 一步处理单篇（PDF → report.json）
uv run python cli/main.py <pdf名> --skip-llm # 只装配已有产物（不调 LLM，缺环节报错）
uv run python cli/main.py <pdf名> --force    # 全链路强制重跑（调 LLM，较贵）
uv run python cli/run_ingest.py <pdf名>      # 摄取流水线：论文库 → MinerU(默认 chunk 骨架) → 报告 → 产物库（幂等）

# ── 分环节（调试用）────────────────────────────────────────────
uv run python cli/run_claims.py <pdf名>      # claims 提取 + 证据回核
uv run python cli/run_summary.py <pdf名>     # claims 全量汇总 + 证据命中(ev✗)审计（不调 LLM）
uv run python cli/run_view.py <pdf名>        # 语义去重 / 角色打标 / 打分 / 渲染摘要视图
uv run python cli/run_skeleton.py <pdf名>    # 论证骨架
uv run python cli/run_figures.py <pdf名>     # 图表识别 + 读图指南
uv run python cli/run_report.py <pdf名>      # 结构化精读报告

# ── 检索 / MultiAnswer ────────────────────────────────────────
uv run python cli/run_fetch.py 1706.03762 --ingest     # 论文取料：arXiv id → 论文库 PDF（--ingest 顺手摄取）
uv run python cli/run_search.py "问题"                 # RAG1：问题 → top-k 论文（支持时间窗）
uv run python cli/run_pipeline.py "问题"               # 论文直读：检索 → 抓取 → 精读 → 报告（端到端）
uv run python cli/run_set.py --dir assets/papers --question "哪些篇做了消融？"   # MultiAnswer：已有论文集合上判定哪几篇满足，并汇成回答
uv run python cli/run_rag2_flow.py --direction "RAG 如何做知识密集问答" --n 10 --question "哪些篇做了消融？"  # 离线编排：RAG1 → 取料 → MultiAnswer（--n 是离线参数；文件名历史遗留）
uv run python cli/eval/run_multianswer.py --question "哪些篇做了消融？" --chunks <切块.parquet> --corpus-n 50        # 离线评测：固定语料上跑 MultiAnswer
# 评测/跑批脚本已归入 cli/eval/（见「评测与回归护栏」节）
```

---

## ⚙️ 行为开关（env）

**默认值直接决定线上行为**；回退通常只改一行或一个 env。

### 默认**生效**的改动

| 开关 | 默认 | 作用 |
|---|---|---|
| `PAPERPILOT_VALIDATOR_GATE` | `1`（开） | 输出前事实闸门：HIGH → 对症修复 / 补充说明 / 兜底拒答；MID/LOW → 前端"答案自检"标注 |
| `PAPERPILOT_VALIDATOR_SALVAGE` | `1`（开） | 修复失败后**叫一次 LLM 出「补充说明」**替代直接拒答：把"能确证 / 不能确证"分开写清，文本强制带 `【补充说明 · 非系统作答】` 标注（**由闸门层强制**，不依赖模型或实现自觉）。`=0` 回到纯拒答 |
| `PAPERPILOT_VALIDATOR_LLM` | `1`（开） | 闸门的 LLM 体检（无支撑断言/偏题/矛盾/含糊；零引用走**格式体检**） |
| `PAPERPILOT_EXT_QUOTA` | `2` | **表池并集候选**：正文 top-12 **原样保留**，追加表池前 2 名里没有的（候选 12 → 13/14）。`=0` 回退 |
| `PAPERPILOT_EXT_RRF_ALPHA` | `0.5` | 外部块（表格/公式）两路权重 `(2α, 2(1-α))`；0.5 = 生产原样（文本块恒 1:1 不动） |
| `PAPERPILOT_MINERU_INJECT` | `1`（开） | 把 MinerU 表格/公式文本按页注入**检索视图**（`=0` 可关，评测用） |

### 其它开关（**以「默认」列为准** —— 不少默认是"开"）

| 开关 | 默认 | 作用 / 为什么默认关 |
|---|---|---|
| `PAPERPILOT_QASPER_TABLES` | 关 | QASPER 外部表格通道（评测用；线上走 MinerU 注入） |
| `PAPERPILOT_TABLE_EMBED_SUMMARY` | 关 | **P2 双写**（表块向量侧改喂语义摘要）——两次实测均**不及现状**，已收口：`qa/recall/P2_DUALWRITE_20260912.md` |
| `PAPERPILOT_TABLE_V1` | 关 | `=1` 复现改造前的表表示 + prompt 布局（同日配对 A/B 用） |
| `PAPERPILOT_QUERY_LEVELS` | （空） | 查询侧优化启用的级别（`l3`...，五级见 `components/query_optimizer.py`）；**空 = 全关**。旧开关 `PAPERPILOT_QUERY_REWRITE=1` 等价于 `l3`。单篇实测负收益，默认关 |
| `PAPERPILOT_QUERY_FUSION` | `equal_rrf` | 多查询融合策略：`equal_rrf`（等权，已证伪，仅对照）/ `quota_union`（变体只做召回补充）/ `rerank_orig`（待实现） |
| `PAPERPILOT_V3_NOL3J` | 开 | 问答图**默认走"跳过 L3 裁判"的变体**（消融实测 64%→73%，见 `qa/REPORT_2.0_20260911.md`）；`=0` 回到带 `judge_l3` 的图（对比用） |
| `PAPERPILOT_RETRIEVE_SECTION_CAP` | `0` | 保序节级配额去重（>0 开启） |
| `PAPERPILOT_VALIDATOR_REPAIR_MID` | 关 | mid 级问题是否也触发修复（默认只标注不修） |
| `PAPERPILOT_VALIDATOR_MISSING` | 关 | 是否提示"原文可能还有未答要点"（好答案上误报多，默认关） |
| `PAPERPILOT_PDF_ID_STRICT` | 关 | 论文身份按**内容指纹**校验；默认对**历史产物**（无指纹记录）按 mtime 收编并补写指纹，`=1` 时连收编也强制重建（存量库排雷用） |
| ⚠️ `PAPERPILOT_USE_MINERU` | **开（`1`）** | MinerU 作 **chunk 骨架**（报告链与检索**同源**）；`=0` 退回 pymupdf 骨架（容灾 / A-B 对照）。**默认即开 → worker 默认串行** |
| `PAPERPILOT_CHUNK_VIEW_DIR` | 空 | 向量缓存目录覆盖（A/B 两臂各用一份，避免来回覆盖重建） |
| `PAPERPILOT_ASK_CONCURRENCY` | `4` | 问答并发上限：`/api/ask` 在线程池里最多同时跑几个（排队的是**线程**，不是事件循环）。**必须 ≥ 1**：`0`/负数/非法值一律回退默认 —— 0 会让每个问答**永久阻塞**，所以它**没有**"关闭闸门"的语义 |
| `PAPERPILOT_ASK_WAIT_S` | `300` | 闸门满时的**排队上限**（秒）；超时返回 `503`（可重试），而不是让页面无限转圈。`0` = 不排队（满了立刻 503，属于**更严格**的方向）。**与摄取（MinerU）耗时无关**：解析期间问答是**直接拒答**（不排队），这里的等待只发生在"前面的问答还没跑完"。单题实测 p50 5s / p90 25s / max 96s（`qa/qasper_run_20260911_072510.json`）→ 单人场景并发 4 > 1，**永远不排队** |
| `PAPERPILOT_MAX_PDF_MB` | `200` | 上传 PDF 大小上限；超限直接 `413`，在**读进内存之前**就拒。**只有显式 `0`** 才是不限制；负数/非数字/空串 → **回退默认 200MB** 并提示一次（配置异常不放开权限） |
| `PAPERPILOT_MOCK_LLM` | 关 | **演示模式·LLM**：固定响应，无需 API Key（`web/app.py --mock` 会打开） |
| `PAPERPILOT_MOCK_EMBED` | 关 | **演示模式·向量**：词袋哈希，不加载 bge-m3（缓存写 `out_views__mock/`，与真向量**物理隔离**） |
| `PAPERPILOT_MINERU` | `1` | `=0` 跳过 MinerU（`skipped`，**不**判问答不可用）；与"失败"区分：失败才拦问答 |

MinerU 相关：`PAPERPILOT_MINERU_VENV`（默认 `.venv-mineru`）、`PAPERPILOT_MINERU_BACKEND`（`pipeline`）、
`PAPERPILOT_MINERU_CMD`（覆盖可执行入口）、`PAPERPILOT_MINERU_TIMEOUT`（900s）。

---

## 🏗 架构与数据流

### 两条解析通道（**2026-09-22 翻转：MinerU 作默认骨架**）

```
PDF ──┬─ MinerU（默认）──→ content_list → chunk 骨架（报告 + claims + 检索**同源**）
      │                    （表格/公式文本天然进骨架 → L0 也能看到表值）
      └─ pymupdf（容灾）──→ 版面块 → 标题树切块 → 同一份骨架的下游
                           （只在 MinerU 产物缺失/损坏/未装时走；**表值拿不到**）
```

**默认骨架 = MinerU**（`PAPERPILOT_USE_MINERU` 未设即**开**）：报告链与检索读**同一份** chunks，
所以旧版"表格只进检索视图、claims 只读 pymupdf"的分工**已取消**。
设 `=0` 才退回 pymupdf 骨架（容灾 / A-B 对照）。`PAPERPILOT_MINERU_INJECT` 的"按页注入检索视图"
只在**备用路**下才有意义（默认路本来就是 MinerU，注入是 no-op）。

**降级与容灾（2026-09-22 改）**：MinerU 失败 / 未装 / 无 GPU → **自动走 pymupdf 备用路，
报告与问答都照常可用**（代价：表格数值拿不到），降级原因记在 `mineru_status()` 与返回值的
`parse_degraded` 里。仍会被拦的只剩一种：**摄取进行中**（产物马上就好，不是错误）。
摄取状态落 `ingest.json`（版本/时间戳，幂等）。

### 摄取 job（异步，2026-09-13）

```
POST /api/report ──→ job_id（202，立即返回）
                      │
       worker（后台，**job 之间串行**：max_workers=1，只有一块 GPU）
         MinerU（GPU，默认 chunk 骨架）──→ 报告链（读同一份 MinerU chunks + LLM）
                                            └─→ 向量索引（cvec，**必须在 MinerU 之后**）
                                            └─→ ready（可问答）
         （仅 `PAPERPILOT_USE_MINERU=0` 时：MinerU ∥ 报告链 两路互不依赖，可真并行）
前端轮询 /api/job/{id}：显示阶段 + 各阶段耗时；可取消（阶段边界生效）/ 重试（已完成阶段复用）
```

> **默认即串行**（`PAPERPILOT_USE_MINERU` 未设 = 开）：报告链也读 MinerU 产物 →
> 必须先 MinerU 后报告，`worker.py` 据此把 lane 并行降级为顺序执行。
> 索引放进 worker 后，"首次提问还要现建向量"这件事也提前做完了。

> **论文身份 = 内容指纹（sha256），不是文件名**（2026-09-13）。
> 产物按内容复用：换掉同名 PDF 会被识别（整链重建；`--skip-llm` 下直接报错），
> 上传"同名但内容不同"的 PDF 会自动另存为 `<原名>__<sha8>.pdf` 当新论文处理并明确告知——
> **不会**再出现"拿到另一篇论文的报告"。

### 问答链路（v3 两级 + 输出闸门）

```
用户问题
  Router   L0 报告层（overview + core_points）够不够？── 够 ──→ 直答（快，不唤醒 embedding）
                                            └─ 不够 ──→ reader
                                                          ├─ 默认：fullctx 全文直读
                                                          └─ `PAPERPILOT_QA_READER=retrieval`：RAG2 块级检索（对照/回退）
                                                                    └─→ **一律试答**（默认无 judge_l3）
  输出前闸门 validator.gate()：
    HIGH（引用越界 / 编数漏数）→ repairer 对症修复（Self-Refine / CRAG）
                                └─ 修不动 → **LLM「补充说明」**（文本强制带「非系统作答」标注）
                                           └─ 补充也失败 → 才走兜底拒答话术
    MID / LOW（无支撑断言 / 偏题 / 矛盾 / 含糊 / 零引用）→ 原样输出 + 前端"答案自检"标注
```

> v2 四层漏斗（L0→L1 claims→L2 扩窗→L3）**已下线归档**到 `archive/qa_funnel_v2/`（含设计稿与旧节点快照）；
> 定稿依据见上表「v3 两级定稿 A/B」。
> ⚠️ **默认配置下没有 L3 裁判**（`PAPERPILOT_V3_NOL3J` 未设即开：删 `judge_l3`、一律试答，消融 64%→73%），
> 因此 `answer_unknown` 节点在默认图里**不可达** —— "诚实拒答"由**输出闸门兜底话术**承担。
> `=0` 才回到带 `judge_l3` 的图（够→答 / 不够→answer_unknown）。

### 归档：v2 四层漏斗（为什么下线）

`archive/qa_funnel_v2/` 是旧版检索架构的完整快照
（`L0 总览 → L1 claims → L2 圆心扩窗（自环）→ L3 全局检索 → unknown`），**已从活代码彻底移除**，
不提供 `ask_v2` 回退；现行唯一检索链是 `src/paperpilot/graph/qa_graph_v3.py`。

| 证据 | 结论 |
|---|---|
| 250 题同裁判 A/B | v3 两级 **203** ≥ v2 四层 **201**；calls 3.7 vs 5.6（**−34%**） |
| L2 目标题群 67 题五连 A/B | L2 可赢空间**≈1 题**（噪声内），却要付"圆心+半径+预算自环"的整块复杂度 |
| L1 claims 分支 A/B | V1=62 < V0=64 —— claims 分支捞不回 L2 那 8 题（那 8 题实为 L3 检索召回短板） |
| judge 精度 | L2 曾出现 **61%** 判够准确率 —— LLM 预测当路由**危险**，判"够"的对象必须简单 |

一句话：**四层漏斗买到的准度 ≈ 0，复杂度却真实存在**（单篇场景的瓶颈在"全局限检索召回 + 答案层"，
不在"多一层判够"）。可复用教训：① 中间层要用数据赎买；② judge 是 LLM 预测，必须配降级链
（判"不够"只是多花钱，判"够"才会答错）；③ 诊断先分"没送到"与"送到了没用好"；④ 融合平坦即有害。

### 代码分层

```
src/paperpilot/
├── jobs.py            # 摄取任务状态：落盘/可查/可取消/可重试（out_jobs/<job_id>.json）
├── worker.py          # 后台 worker：MinerU ∥ 报告链 → 索引收口（job 间串行、job 内并行）
├── ingest.py          # 摄取流水线：论文库 → MinerU(检索) → pipeline(报告) → 产物库
├── pipeline.py        # 报告编排层：一个 PDF → 各环节产物 → report.json（process_pdf）
├── tools/             # 引擎：pdf_parser/heading/chunker/analyzer/evidence/viewer/
│                      #       mock_llm.py：演示模式（无 Key/模型的假 LLM + 假向量）
│                      #       skeleton/figures/report/llm/mineru_bridge
├── prompts/           # 各环节 LLM 提示词（与逻辑分离）
├── models/schema.py   # pydantic 强类型（Chunk/Claim/ClaimGroup/PaperReport…）
├── agents/            # 问答节点（纯函数，不依赖 langgraph）
│   ├── nodes/         #   report/judge/search_l3/generate_answer/answer_unknown…
│   └── document_cache.py # 两套视图：ordered_chunks（报告/溯源） / retrieval_chunks（检索，含表公式）
├── components/        # 组件门面（显式化）：router / retriever / generator / validator /
│                      #   repairer / context_builder / query_optimizer / splitter / numbers / reranker
└── graph/             # LangGraph 接线：qa_graph_v3.py（v3 两级是**唯一**检索链）
web/                   # FastAPI + 前端三栏工作台（index.html）
cli/                   # 命令行入口 + 评测工具
tests/                 # pytest 测试套件（离线：假 LLM/假向量/假 MinerU；CI 跑这个）
demo/                  # 自带 demo 论文（合成内容、无版权）+ 生成脚本 make_demo_pdf.py
qa/                    # 手写问题集 + 各轮评测报告与复算脚本（质量评测，需真模型）
bench/                 # 回归基线（baseline.json）
assets/papers/         # 论文 PDF（不入库）
assets/artifacts/      # 产物（不入库）
archive/qa_funnel_v2/  # v2 四层漏斗快照（含设计稿），只作对照
```

---

## 🧪 测试与 CI

**一条命令，5 分钟内出确定结果，不需要任何 API Key / GPU / 本地模型：**

```bash
uv run pytest -q        # 203 passed in ~10s（本地；CI 上含装依赖约 1~2 分钟）
```

**三层命令分三类事**（后两档默认跳过，缺环境只会 skip、不会红）：

| 命令 | 跑什么 | 需要什么 |
|---|---|---|
| **`uv run pytest -q`** | **离线**：纯逻辑 / 契约边界 / 状态机 / 缓存命中失效 / API / **mock LLM 端到端**（203 用例） | 无（CI 与"别人 clone 后自证"都用这条） |
| `uv run pytest -m local` | **真模型冒烟**：真向量检索位次（中文问英文论文）/ 真 LLM 作答带引用 / 真 MinerU 解析 / 真实裁判体检 | 本机 key + 模型 + `.venv-mineru`（实测 13s / 22s / 68s） |
| `uv run pytest -m ui` | **浏览器级冒烟**：真前端 JS + 真 HTTP（上传 → 报告 → 提问 → 点引用跳原文），并捕获未捕获的 JS 异常 | 本机 Edge/Chrome（或 `playwright install chromium`；**不必**下 130MB） |

外部依赖的处理方式（这是"不需要 key"的实现）：

| 依赖 | 测试里怎么处理 |
|---|---|
| **LLM** | `fake_llm`：替换最底层 `llm._chat`，按 prompt 返回固定结果（`chat_json/chat_text/裁判` 全自动跟随） |
| **embedding** | `fake_embed`：确定性"词袋哈希"向量（跨平台稳定，不依赖 `hash()` 随机化） |
| **MinerU** | 假 `content_list.json` + 假环境：走**真实**摄取代码路径（版本比对/产物校验/meta 落盘），只是不跑 GPU 子进程 |
| **论文语料** | 现场用 pymupdf 生成小 PDF；所有 `assets/**` 路径被重定向到 `tmp_path`（**不碰真实论文与产物**） |
| **网络** | `urllib.request.urlopen` 被换成"一用就炸"：漏了替身会**明确失败**，而不是偶发联网成功 |

覆盖清单（`uv run pytest --collect-only -q` 实测：**默认档 203 + local/ui 档 5 = 208**）：

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `test_tables.py` | 31 | 表块纯函数：caption 清噪（含罗马数字）、表头指纹、摘要、并集配额、按块权重、RRF |
| `test_api.py` | 30 | API 集成：202 早返回、幂等 200、任务查询、取消/重试、404/400、拒答话术、状态快照；**上传门**（`%PDF-` 魔数 / 大小上限，含**配置 fail-safe**：非法值不放宽权限） |
| `test_jobs_worker.py` | 25 | 任务状态机：阶段耗时、取消（含"编码中取消"）、重试、重启恢复、**异常兜底不卡死**、**并发读写不丢状态**、MinerU 真状态被采纳 |
| `test_api_concurrency.py` | 18 | **`/api/ask` 不阻塞事件循环**（同步端点 + 并发闸门）；闸门配置 **fail-safe**、排队**有界**（无名额 → 503 而不是无限等待） |
| `test_identity_cache.py` | 17 | 内容指纹判定（ok/stale/adopt）、**缓存命中与失效**（进程内 + 磁盘 cvec）、表格只进检索视图 |
| `test_llm_client.py` | 16 | JSON 解析容错、**坏 JSON**、**超长输入**、未配置时明确报错 |
| `test_tgt.py` | 15 | 目标表定位口径（编号 ∪ 内容）—— 所有表格类指标的尺子 |
| `test_validator_offline.py` | 13 | 闸门机器判据：**零引用分级**、**引用越界**、gate 动作不变回归（+1 条 `local`：真裁判模型） |
| `test_mock_mode.py` | 10 | **演示模式**（`--mock`）：无 Key / 无模型 / 无 MinerU 也能跑通（演示模式承诺的技术保障） |
| `test_e2e_mock.py` | **4** | **mock LLM 端到端**：上传 → 后台 job → 报告 → 提问 → 带 `[n]` 引用的答案；含"MinerU 失败 → 不建索引 + 问答被闸门拦" |
| `test_local_smoke.py` | 3 | `local` 档**真模型冒烟**：真向量检索位次 / 真 LLM 作答带引用 / 真 MinerU 解析（各带 skip 条件） |
| `test_ui_smoke.py` | 1 | `ui` 档**浏览器冒烟**：playwright 驱动真页面走完上传 → 报告 → 提问 → 点引用跳原文，并捕获未捕获 JS 异常 |

CI：`.github/workflows/ci.yml`（每次 push 自动跑同一套，**不设任何 secret**）。

[![CI](https://github.com/lx8828/paperpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/lx8828/paperpilot/actions/workflows/ci.yml)
<!-- 若 GitHub 用户名或仓库名不同，把上面两处 lx8828 / paperpilot 一起改掉 -->

**回归纪律**：改核心逻辑后 → `uv run pytest -q`（秒级，离线）→ 再 `run_bench.py` 看产物无退化 → 最后跑定向评测。

---

## 🔬 评测与回归护栏（**质量**类 · 需要真实模型 / LLM）

> 与上面的测试是**分工关系**：CI 判"对错"（确定、免费），这里判"高低"（有噪声、要钱）。

| 命令 | 作用 |
|---|---|
| `uv run python cli/eval/run_bench.py` | report 层回归护栏：本地重算产物与 `bench/baseline.json` 勾叉 diff（不调 LLM） |
| `uv run python cli/eval/run_compare.py sample/run` | 三列公平对比 harness（B0/B1/B2，见 `qa/COMPARE_DESIGN.md`） |
| `uv run python cli/eval/run_qa_eval.py` | QA 验证：报告结构化分层是否支撑真实问答 |
| `uv run python cli/eval/run_rag_eval.py` | RAG 评估：对 `qa_set` 每道题真实跑 LangGraph 问答图 |
| `uv run python cli/eval/run_chunk_eval.py` / `run_retrieval_eval.py` / `run_chunk_recall.py` | 分块 / 检索层专项评估（零 LLM 尺子） |

> 评测/跑批脚本 2026-10-01 已统一归入 **`cli/eval/`**（15 个 + `_anchors.py`）；`cli/` 根下只留
> 四个工具与报告小组件入口。旧入口 `run_qasper_eval.py` / `run_qa_v2.py` 已不在。

**评测数字必须带口径与日期**；单次端到端运行的 churn 在 8%~25%，
**<3 题的效应不可判**（见 `qa/recall/UNION_AND_CONTEXT_20260911.md` §2）。

### 评测分层（L0–L4）：每层一个触发条件

| 层 | 判什么 | 触发 | 成本 | 现状 |
|---|---|---|---|---|
| **L0 · 不变量** | 仓库状态与契约（**不跑模型**）：干净 clone 能不能评测、路径引用是否悬空、gold 与导出是否同步 | **每次改动**（CI 默认） | 秒级 / 免费 | ✅ 已有 |
| **L1 · 篇级检索（RAG1）** | LitSearch（64,183 篇）：Recall@k / nDCG | 发版前 / 手动 | 分钟级 | 🟡 能力已有（`cli/eval/run_retrieval_eval.py`） |
| **L2 · 块级 reader（RAG2）** | r2dev 28 个 facet：证据召回 / reader 判定 | 发版前 / 手动 | 分钟~小时 | 🟡 能力已有（`retrieval/tmp/_r2_*.py`） |
| **L3 · 端到端** | 产品链路（`graph.ask`）+ 5 组 164 题 | 发版前 / 手动 | 小时级 | 🟡 能力已有（`cli/eval/run_group_qa.py`） |
| **L4 · 对外可比** | **MultiAnswer 的外部评测（主用 QAMPARI）**；runner 同时支持 LoFT 五任务 | **按需** | 小时级 | ✅ 已有（`evals/runners/l4_loft.py`） |

**为什么这么分层**：把「**免费且必须每次跑**」（L0）与「**贵且只需发版前跑**」（L1–L3）分开 ——
否则一套评测要么贵到没人跑、要么便宜到抓不住问题。L4 单独一档，因为它的价值在**对外可比**
（与公开榜同口径），触发节奏与内部回归不同。

**能力 ↔ 评测归属**

| 能力模块 | 内部 / 端到端 | 对外可比 |
|---|---|---|
| **RAG1** | 篇级检索回归、LitSearch 消融 | **LitSearch（L1）** |
| **RAG2** | 块级证据召回 / reader 判定（L2） | —（默认关闭的对照路径） |
| **MultiAnswer** | 3 簇 × 50 篇 = 150 篇、28 组合（论文域） | **QAMPARI（L4）** |
| **Fullctx / 产品链路** | L3 端到端 + 三列公平对比 | QASPER 等报告 |

**`evals/` 目录约定**

```
evals/
  baselines/  基线（棘轮）：path_liveness.json（悬空引用）/ tmp_scripts.json（tmp 脚本分类）/ metrics.json（指标）
  checks/     L0 不变量（只读工具）：path_liveness.py / tmp_scripts_audit.py / metrics_ratchet.py
  report.py   统一记录格式 —— 让各层数字能放一张表里比
  reports/    落盘（gitignore；入库的是 .md 摘要）
  runners/    l4_loft.py（LoFT 五任务）
```

三条硬约定：① **真值只放一份** —— `evals/datasets/` 只写**指针**（指向 `retrieval/tmp/group*/` 与
`retrieval/data/r2dev/`），**不复制**（复制必然漂移）；② **判据只放一处** —— 例如"gold ↔ 导出同步"
只由 `retrieval/scripts/_check_export_sync.py` 定义，`evals/` 与 `tests/` 都**调用**它、不重写
（重写 = 漂移 = 假测试）；③ **报告入库，产物不入库** —— `retrieval/results/*.md` 入库（不可再生的记录），
跑批产出（`results/*/`、`qa/multi/_runs/`）不入库。

### L0 在查什么（免费 · 每次改动都跑）

| 检查 | 在哪 | 断言什么 |
|---|---|---|
| gold 是否全部入库 | `tests/test_eval_assets.py` | 磁盘上**每个** gold 都已在 git（曾 15 个只入库 7 个） |
| r2dev 真值是否入库 | 同上 | 现役真值 + 题集 + 子查询在库；大件/可再生件**不**在库 |
| gold ↔ 导出是否同步 | 同上 + `retrieval/scripts/_check_export_sync.py` | 不同步 = 跑批**静默测另一份题** |
| 悬空路径引用 | `evals/checks/path_liveness.py` + `tests/test_path_liveness.py` | 无**新增**悬空引用（棘轮，历史债不阻塞）。判据是**版本库视图**（不是磁盘）—— 否则本机绿、CI 红 |
| `retrieval/tmp` 脚本归属 | `evals/checks/tmp_scripts_audit.py` + `tests/test_tmp_scripts_audit.py` | 保留脚本**已入库**；归档**不造断 import** |
| 统一记录格式 | `evals/report.py` + `tests/test_evals_report.py` | 缺 `n`/`note` 的记录**写不进去**；NaN/inf 被拒 |
| **指标棘轮** | `evals/checks/metrics_ratchet.py` + `tests/test_metrics_ratchet.py` | 跑批后指标**掉出容差**就报错；带容差（±16pt）与**方向** —— `reader_offlabel` / `gold_map_failed` 这类**越低越好**，一律"降了就红"会把**修好了**判成回退 |
| **L4 口径完整性** | `evals/runners/l4_loft.py` + `tests/test_l4_loft.py` | 坏指标文件（**0KB**/缺字段）**被跳过而非当 0 分** |
| **跑批入口的 emit 接线** | `tests/test_eval_emit_wiring.py` | 记录**真落盘**（断言**副作用**，不是"没抛异常"）。曾因 `_emit_report` 漏传 `args` → `NameError` 被 `except` 静默吞掉 → **一整天没记录也没人发现** |

### L4：LoFT 五个任务（口径不同，不可混谈）

```bash
uv run python evals/runners/l4_loft.py --check                      # 五任务前置（离线秒级）
uv run python evals/runners/l4_loft.py --collect                    # 汇总已有运行
uv run python evals/runners/l4_loft.py --official --task sql --name my_run
```

| `--task` | 官方指标 | 口径要点 |
|---|---|---|
| `multi_value_rag`（QAMPARI） | `em` / `coverage` / `subspan_em` | `coverage` **只有 recall**；官方 `f1` **恒为 0**（多值分支未赋值）→ 已从记录里**排除** |
| `rag` | `em` / `f1` | SQuAD 风格**单值** |
| `retrieval` | `recall@k` / `mrecall@k` | **Capped**：gold 数 > `k` 时**除以 `k`** |
| `sql` | `execution_accuracy` | **不强制顺序**（建集时已滤掉需排序的题） |
| `icl` | `em` | 预测**多值被忽略**、实例**多轮** |

指标**透传**（官方输出里有什么数值指标就落什么）—— 硬编码清单会"官方加指标而这里**静默漏报**"。
健康度 `<前缀>.unanswered`（空预测题数）应为 0：官方口径会把它们**剔出分母** → 虚高。

### 评测侧已知未决

- **L2 只跑了检索侧，判定质量未测**：`_r2_retr_eval.py`（证据召回）已落 18 条；reader 侧
  `_r2_reader.py` 要**真调 LLM 判官**，**尚未跑** → `r2.reader_*` 无基线，"判得对不对"这一半**还是空白**。
- **L4 当前有数据的只有 QAMPARI**：`rag` / `retrieval` / `sql` / `icl` 四个任务**无数据无运行**（`--check` 显示"尚无运行"）；
  QAMPARI 的 **`32k` 档有数据但无运行**（补跑要真调 LLM）。
- **L4 容差待实测**：默认 `0.02` 在 `n=100` 上 = **2 题翻转**。只跟"**重收同一批 preds**"比是**确定性**的（没问题）；
  **重跑 preds**（LLM 采样）则可能误报。放宽容差等于把闸门关小 —— **要先量同配置两次跑的离散度**再定。
- **`evals/reports/` 是 gitignore 的** → 干净 clone 里指标棘轮**无数据可比**（会跳过并说明）。
  要让"指标历史"进版本库，应入库 `--md` 摘要（`uv run python evals/report.py --md`），当前摘要落在 **`evals/RESULTS.md`**。

### `qa/` 目录（评测与实验记录 —— **不是测试套件**）

测试在 `tests/`（`uv run pytest -q`，203 用例、无需 key）；`qa/` 放的是**评测与实验记录**：
判"高低"，有噪声、要 key/模型，**按需手动跑**。

**三份主要文档**

| 文档 | 内容 |
|---|---|
| `qa/QASPER_EVAL_LOG.md` | QASPER 第三方基准的逐次记录（**84.3%** 那条线的来龙去脉、每次改动的归因） |
| `qa/RETRIEVAL_EVAL_FRAMEWORK.md` | **评测脚本清单**：每个脚本的定位 / 样本量 / 得分 / 是否可复用 |
| `qa/CAMPAIGN_20260906-07.md` | v2→v3 的**决策史**（为什么砍 L2、为什么表池并集设为默认、什么是负结果） |

| 目录 / 文件 | 是什么 | 入库 |
|---|---|---|
| `recall/` | 检索与表格线的主战场：报告（`.md`）、评测口径模块（`_tgt.py`）、保留的 runner | ✅ 141 个 |
| `questions/` | 中文 QA 题集（31 个 json / 304 题；第一组 5 篇 = 70 题按 gold 引文口径重建） | ✅ |
| `compare/` | B0 / B1 / B2 三列对照（设计 + 报告 + 配对 McNemar / 检验功效脚本） | ✅ 12 个 |
| `reader/` `negqa/` `robust/` `stress/` | 四个专项：外行问答 / 防幻觉 / 同义改写稳定性 / 对抗诱导 | ✅ |
| `review/` | 代码审查的复现脚本与应答台账 | ✅ |
| `snapshots/`、`_archive/`（136 个一次性诊断脚本）、`_scratch/` | 场景快照 / 归档诊断 / 临时脚本 | ❌ 本机留档 |

**两个容易误判的地方**：

1. 报告里引用的 `qa/*.json` / `*.jsonl` 运行记录**多数没入库**（本机另有约 250 个运行记录与日志）。
   看到"某个 `qa/qasper_run_*.json` 不存在"，通常不是文档坏了，而是**那份记录只在本机**。
2. **2026-09-14 归档了 136 个一次性脚本**到 `qa/_archive/recall/`（本机留档、不入库，git 历史仍在），
   **清单见 `qa/recall/ARCHIVED.md`** → 老报告里指向 `qa/recall/_xxx.py` 的路径**可能已经不在**。

---

## 📚 文档导航

| 文档 | 内容 |
|---|---|
| 🗂 **`docs/DOC_INDEX.md`** | **文档总地图**：哪些是现状、哪些是历史、冲突时信谁（建立 2026-09-22，**最近校验 2026-10-06**；与代码冲突时**信代码**） |
| `qa/CAMPAIGN_20260906-07.md` | **决策总账**：优化史 + 定版 + 定位收尾 |
| `DEVELOPMENT_LOG.md` | 开发踩坑记录、决策背景、遗留项、命令速查 |
| `docs/RAG_COMPONENT_NOTES.md` | 组件层设计依据（Router/Retriever/Generator/Validator 的取舍与实测） |
| `qa/recall/TABLE_LINE_STATUS_20260912.md` | **表格能力线总索引**：哪些改动在生效 / 哪些是默认关的实验 / 哪些做法已被否证 |
| `qa/review/RESPONSES_20260913.md` | 外部代码审查**逐条核验与处置**台账 |
| `qa/COMPARE_DESIGN.md` | 三列公平对比设计（控制变量/口径/成本） |
| `qa/reader/` `qa/negqa/` `qa/robust/` `qa/stress/` | 帮读主口径 / 防幻觉 / 稳定性 / 压力边界 报告 |
| `archive/qa_funnel_v2/QA_FUNNEL_DESIGN.md` | v2 四层漏斗架构设计 + 铁律（**已归档**，只作对照） |
| `docs/RAG2_CORPUS_WIRING.md` | **MultiAnswer 语料接线与能力边界**（文件名 RAG2 为历史遗留；QAMPARI 是 MultiAnswer 的评测） |
| `docs/RETRIEVAL_LOG.md` | **RAG1 研究记录**：数据事实 / 完整消融 / 池深与精排扫描 / 阶段 ①~③ 的正负结果 |
| `archive/design.md` | 早期设计（已归档，只作对照） |

---

## 🧰 开发约定

- **类型检查**：`pyproject.toml` 里配了 `[tool.basedpyright]`，但**当前环境未安装**该工具
  （`uv run basedpyright` 会失败）。需要类型门禁时先 `uv add --dev basedpyright`；
  日常改动用 `uv run python -m compileall src cli web tests` + `uv run pytest -q` 兜底。
- **产物不入库**：`assets/artifacts/`、`assets/papers/*.pdf`、`*.log`、`.env` 均在 `.gitignore` 内。
- **入库的是"资产"**：`qa/questions/`（手写题集）、`qa/**/*.md`（报告）、`qa/**/_*.py`（复算脚本）、
  `qa/*_set*.json`（题集）、`bench/baseline.json`（回归基线）、`tests/`（测试套件）。
- **不入库的是"运行记录"**（2026-09-13 起统一排除）：`qa/recall/*.json`、`qa/compare/{deep_,run_}*.json`、
  `qa/negqa/neg_run_*.json`、`qa/reader|robust|stress/*_result.json`、`qa/qasper_*` 等——
  机器产出、可重跑、体积约 7 MB，且**含论文原文片段**（gold 证据/答案）不宜大段转载。
  ⚠️ 代价：复算脚本若依赖某个记录文件，**先在本地跑上游脚本生成**（README 里引用的原始记录均为本地文件）。
- **评测口径**：目标表定位用 `qa/recall/_tgt.py`（编号 ∪ 内容联合判定），不要在各脚本里另写 `cap_key`。
  `qa/` 下的测试类断言已迁到 `tests/`（pytest）；测试类断言不再新增散装 `_selftest_*.py`。

### 论文线真值（`retrieval/data/r2dev/`）：读哪个 gold

真值由人工 + 多判官标注，**不能靠重跑脚本再生**（原则：**gold 不重标**）→ **必须入库**。
按**语料口径**选，**别按文件名新旧猜**：

| 语料口径 | 真值 | 题集 | PDF 映射 | 语料 / 切块 |
|---|---|---|---|---|
| **20 篇**（3 簇 × 20 池 = 60，47 篇有 PDF） | `gold_final2.csv` | `facets_v2_selected.json`（28 组合） | `pdf_map.json` | `clusters/` + `prodchunk/mineru/` |
| **50 篇**（3 簇 × 50 = 150，含同领域干扰项） | **`gold_final3.csv`** | 同上 | `corpus50/pdf_map_all.json` | `corpus50/` + `prodchunk50/mineru/` |

**对外 / 汇总口径统一用 50 篇**；20 篇那套是最早的语料口径、**不再用于对外报数**，但**保留在库**
（真值不可再生；`retrieval/tmp/_r2_retr_eval.py --corpus 20|50` 等分析脚本仍会读它）。
`gold_final2.csv` **不是"旧版本"**（它比 `gold_final.csv` 新）—— 它是 `gold_final3.csv` 的**标注基础**：
`gold_final3` = 它的 **451 对** + 新增 **949 对** = **1,400 对**（**50 篇就是在 20 篇上扩出来的**）。

```
gold_final.csv ──(题集重选 29→28 组合 + 补三判官票数)──▶ gold_final2.csv ──(换语料 47→150 篇)──▶ gold_final3.csv
   第一代                                                    第二代(20篇)                      第三代·对外口径(50篇)
```

**入库策略**（本目录 27.9MB，只有约 1.3MB 入库）：三代 gold / 推导链 / 题集 / 子查询 / 映射 / 校准表入库；
`*_evidence.json`（11.6MB）、`clusters/` `corpus50/` `pdftext*/` `prodchunk*/`（14.3MB）、运行缓存**不入库**
（可按 gold 的 `ev_idx` 从切块重抽 / 从 PDF 重建）。

⚠️ **踩过的坑**：① 本目录曾**整体被 `.gitignore` 的 `retrieval/data/` 挡掉** → 干净 clone 拿不到真值、
论文线评测**完全无法复现**（2026-10-05 改为"挡可再生大件、放行手工标注真值"）；
② **`git check-ignore` 不能用来验证 `!` 反选**（反选命中时它**仍会打印路径**）——
要验证请用 `git ls-files --others --exclude-standard retrieval/data`。

### `retrieval/tmp/`：为什么"临时目录"里装着真源

目录名看着像"临时"，其实装着**两类不可再生的东西**：

| 内容 | 是什么 | 入库？ |
|---|---|---|
| `group*/<stem>.questions.json`、`group*/_group.questions.json` | **出题 gold（唯一真源）** —— 每题带逐字原文引文 | ✅ **必须** |
| `_*.py`（135 个） | 评测/实验脚本：**库 14 个 + 有出处的入口 ~120 个** | ✅ |
| `_archive/`（161 个 `.py`） | **一次性实验**的归档（无出处、无人引用） | ❌ gitignore |
| `_gold_history/` | 出题的草稿/备份 | ❌ gitignore |

分类**不是猜的** —— 由 `evals/checks/tmp_scripts_audit.py` 算出，结果落
`evals/baselines/tmp_scripts.json`（含每个脚本被谁引用、被哪份报告提到），可重跑复核。

⚠️ **这 14 个是「库」**（被别的脚本 import，删/改会连带一大片）：
`_r2_std_metrics.py`（被引用 **30**）、`_r2_facets_v2.py`（22）、`_sandbox.py`（15）、
`_qampari_run.py`（8）、`_r2_gold_recalib.py`（8）……

**为什么留在 `retrieval/tmp/` 而没搬去 `evals/`**：它们按**相对 `retrieval/`** 定位
（`HERE = parents[1]` → `retrieval/`；`sys.path` 插 `retrieval/scripts`；用 `spec_from_file_location`
动态加载兄弟脚本）。实测 **133/135 个用 `parents[N]`**、88 个改过 `sys.path`、50 个动态加载 ——
**换目录会让这些语义变化，不是换名能修的**（得在 100+ 个文件里逐处判断"这个相对路径想要的是谁"）。
零功能收益、全是风险，所以**不搬**。真要拆 `lib/` + `runners/`，前提是**先改成包导入**（`import evals.lib.x`）。

---

## 📌 当前定位

> 给被英文/专业门槛劝退的普通人**读论文**：报告讲得懂 + 问答不编造 + 每句可溯源。
> 我们对"帮小白建立正确理解"负责，明确不做"帮高手精确摘原文细节"（产品范围决策）。
>
> 已做：MinerU 表格/公式进检索视图、表池并集候选（默认开）、输出闸门前端可见。
> 已知短板：extractive 细值抽取、表格题在上下文里仍答不出的那几道（属答案抽取层）、
> 以及"跨时间的三列对比不可归因"这类评测口径问题——都记在 `DEVELOPMENT_LOG.md` 与各专题报告里。

作者：lx（834659376@qq.com）

许可证：[MIT](LICENSE)——可自由使用 / 修改 / 分发（保留版权声明即可）。

