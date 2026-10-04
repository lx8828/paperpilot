# 文档总地图（DOC_INDEX）· 2026-09-22 建立

> **为什么有这份文件**：项目几次大改后，文档里混着「现行口径」「当天口径」「早期设想」，
> 分不清哪些还成立。本表由 **2026-09-22 的逐条核对（一律以代码为准）** 产出：
> 标出每份文档的**性质、口径日期、已知过期点**，并记录本轮改了什么（§5）。
>
> **冲突时的优先级（硬规则）**：
>
> ```
> 代码  >  README.md（根）  >  docs/*.md  >  qa/*.md、retrieval/results/*.md（历史报告）
> ```
> 任何文档与代码冲突，**改文档**，不要改结论。

---

## 0. 分层总览

| 层 | 谁 | 性质 | 可当现状依据？ |
|---|---|---|---|
| ① **现状权威** | `README.md`（根） | 行为、开关、接口、评测成绩的唯一入口 | ✅ **是** |
| ② **设计依据** | `docs/RAG_COMPONENT_NOTES.md`、`qa/RETRIEVAL_EVAL_FRAMEWORK.md`、`qa/COMPARE_DESIGN.md`、`qa/RAG_ENGINE_DESIGN.md` | "为什么这么配"的证据账 | ⚠️ 配置结论可用；**数字口径看日期** |
| ② **研究区** | `retrieval/README.md` + `retrieval/results/*.md` | 篇级检索的探索与消融 | ⚠️ 只代表研究区语料，与线上不可混用 |
| ③ **流水账** | `DEVELOPMENT_LOG.md` | 决策史 / 踩坑 / 遗留项 | ❌ 只代表当时；已加性质声明 |
| ④ **历史报告** | `qa/**/*.md`（91 + 51 份）、`retrieval/results/*.md`（27 份） | 带日期的实验记录 | ❌ 口径只对当天生效 |
| ⑤ **已归档** | `archive/qa_funnel_v2/`、`design.md` | 早期架构与设想 | ❌ 只作对照 |
| ⑥ **不入库** | `assets/artifacts/**`（含 **932 份** `out_views/*.md` 论文产物）、`qa/snapshots/`、`qa/_archive/`、`qa/_scratch/`、`qa/baseline.json` | 本机产物 / 运行记录 | ❌ 可再生 |

---

## 1. 现行文档（要改代码前先看这层）

| 文档 | 看什么 | 已知过期点 / 注意 |
|---|---|---|
| **`README.md`** | 能力、快速开始、env 开关、CLI、架构与数据流、三层测试、评测成绩 | ✅ 2026-09-22 已按代码修正 **11 处**（见 §5）。评测成绩表**必须带日期读** |
| `docs/RAG_COMPONENT_NOTES.md` | 每个组件的**默认配置与实测依据**（Splitter/Router/QueryOptimizer/Retriever/Reranker/ContextBuilder/Generator/Validator） | ✅ 组件默认值已逐条对过源码；题集规模已更新为 31 篇 / 304 题 |
| `qa/RETRIEVAL_EVAL_FRAMEWORK.md` | 评测脚本清单：定位/样本量/是否可复用 | 样本量随题集变了，**以脚本 `--help` 为准** |
| `qa/COMPARE_DESIGN.md` | B0/B1/B2 三列对比的控制变量与口径 | 设计稳定；**数字看最新 `compare_*.md`** |
| `qa/README.md` | `qa/` 目录政策：什么入库、怎么跑、归档了哪些脚本 | ✅ 用例数已更新；`recall/ARCHIVED.md` 是一次性脚本的去向表 |
| `retrieval/README.md` | 篇级检索的设计与完整消融（LitSearch 64,183 篇 / 597 查询） | ⚠️ **研究区口径**：线上工具① 走 arXiv 语料，两者不可混算（已在文首声明） |
| `retrieval/scripts/`（脚本 docstring） | 出题/评测流水线的**契约**（`_qa_groups.py` 是分组唯一真源） | ✅ 现行 |

---

## 2. 历史报告（只在"某天的结论是怎么来的"时查）

| 目录 / 文件 | 数量 | 口径 | 什么时候看 |
|---|---|---|---|
| `qa/recall/`（含 `TABLE_LINE_STATUS_20260912.md` 总索引） | 91 | 2026-09-06 ~ 09-12 | 检索/表格线的每一次 A/B 与负结果；**表格线总索引**优先 |
| `qa/qasper_summary_*.md` / `qasper_deep_*.md` / `qasper_overnight_*.md` | ~20 | 2026-09-05 ~ 09-11 | QASPER 逐次运行；**看最新那次 + `QASPER_EVAL_LOG.md` 的归因** |
| `qa/qa_summary_*.md` | ~27 | 2026-09-04 ~ 09-21 | 中文题集的逐次跑批（多为 1KB 摘要）；**基线是 `qa_summary_20260909_003404.md`** |
| `qa/REPORT_2.0_20260911.md` / `REPORT_NIGHT_20260911.md` | 2 | 2026-09-11 | v2→v3 定稿与夜间四项验证 |
| `qa/CAMPAIGN_20260906-07.md` | 1 | 2026-09-06~07 | **决策总账**（优化史 + 定版 + 定位收尾） |
| `qa/QA_V2_RUN1_REPORT.md` / `QA_V2_NEW10_20260906_REPORT.md` / `FULL_RERUN_20260906_REPORT.md` / `QASPER_OVERNIGHT_MORNING_REPORT.md` | 4 | 2026-09-06~07 | 早期 v2 口径 |
| `qa/RETRIEVAL_EXPLORATION_20260908.md` | 1 | 2026-09-08 | **L2 存废之争全过程**（阶段 Q）——v2 四层为何被砍 |
| `qa/reader/` `qa/negqa/` `qa/robust/` `qa/stress/` | 各 1~2 | 2026-09-07 | 帮读主口径 / 防幻觉 / 稳定性 / 压力边界 |
| `qa/review/RESPONSES_20260913.md` | 1 | 2026-09-13 | 外部代码审查逐条核验台账 |
| `retrieval/results/*.md` | 29 | 2026-09-18 ~ 09-27 | 篇级检索消融（LITSEARCH_* / ARXIV_*）；**全部指标一页汇总**=`LITSEARCH_METRICS_20260927.md`（09-27 重跑）；**交付数 N / 池深 K / 覆盖**见 `LITSEARCH_DELIVER_N_20260927.md`；⚠️ **6 份是 0KB 空文件**（`ARXIV_CE100-100_*`、`LITSEARCH_SCORE100-100_*` 等），需重跑或删除 |

---

## 3. 已归档（只作对照，勿作现状依据）

| 位置 | 内容 | 状态 |
|---|---|---|
| `archive/qa_funnel_v2/` | v2 四层漏斗（L0→L1 claims→L2 扩窗→L3）的图/节点快照 + 设计稿 + 下线原因 | ✅ 已在文内标注"已下线"，**不要照它改代码** |
| `design.md` | 最早的设计稿（pymupdf-only、5 节点、Ollama、Chroma、Neo4j…） | ✅ 已加横幅：**多数"预留"已实现或已被替代**；保留是因为 `workflow.py` 还引用它的"MVP 范围" |

---

## 4. 不入库（可再生；看到"文档引用的文件不存在"先看这里）

| 位置 | 是什么 | 为什么不在库里 |
|---|---|---|
| `assets/artifacts/**` | 论文产物：`out_views/`（932 份报告 md + 向量）、`out_mineru/`、`out_claims/`、`out_jobs/` | 体积 + 含论文原文 |
| `assets/papers/*.pdf` | 论文 PDF | 体积 + 版权 |
| `qa/qasper_*.json`、`qa/qa_run_*.json`、`qa/robust/robust_run_*.json` 等 | 逐题运行记录 | 体积（~7MB）+ **含论文原文片段** |
| `qa/snapshots/`、`bench/snapshots/`、`qa/baseline.json` | 本机快照与回归基线 | 机器产出 |
| `qa/_archive/`、`qa/_scratch/`、`retrieval/_scratch/`、`retrieval/data/` | 一次性脚本留档 / 语料与索引 | 已由 `.gitignore` 排除 |
| `retrieval/tmp/<group>/` | 出题素材：pymupdf 原文 `<stem>.txt`、表格页 `pages/*.png`、gold 题集 `*.questions.json` | 含论文原文与版面图（**但 gold 是唯一真源，见 §5.4**） |

> **入库的"资产"**：`qa/questions/`（手写题集）、`qa/**/*.md`（报告）、`qa/**/_*.py`（复算脚本）、
> `qa/*_set*.json`、`bench/baseline.json`、`tests/`、`retrieval/scripts/*.py`、`retrieval/results/*.md`。

---

## 5. 本轮修正记录（2026-09-22 · 全部以代码为证据）

核对方式：把文档里的**可证伪陈述**（默认值、分支是否存在、路径、数字）逐条拉去源码里对。
代码证据的行号见下（改动后行号可能位移，按**符号名**搜）。

### 5.1 `README.md`（11 处）

| # | 位置 | 原陈述（错/过期） | 事实（代码证据） |
|---|---|---|---|
| 1 | 能力一览 · v3 问答 | "不够 → L3 → **判够** → 答 / 诚实收尾" | 默认 `PAPERPILOT_V3_NOL3J=1` **删了 `judge_l3`**，L3 后**一律试答**（`graph/qa_graph_v3.py` 选图 + `build_qa_graph_v3_nol3j` 节点表） |
| 2 | 问答链路图 | 同上，且"不够 → 诚实收尾"分支 | 默认图里 **`answer_unknown` 无入边、不可达**（`qa_graph_v3.py` 模块 docstring 自证）；拒答由**输出闸门兜底话术**承担 |
| 3 | env 表 · `PAPERPILOT_USE_MINERU` | 默认**关**（"=1 走遗留全链 MinerU"） | 实际默认 **`"1"`（开）**：`agents/document_cache.py` 的 `mineru_skeleton_enabled()`；语义也反了 —— 现在是 **MinerU 作 chunk 骨架**，`=0` 才退回 pymupdf |
| 4 | 两条解析通道 | "pymupdf → 报告；MinerU 只注检索；claims 只读 pymupdf" | 默认骨架已是 MinerU，**报告 / claims / 检索同源**；注入只在**备用路**生效（`document_cache.retrieval_chunks` / `PAPERPILOT_MINERU_INJECT`） |
| 5 | 降级语义 | "MinerU 失败 → **问答直接失败并拒答**" | 2026-09-22 起**自动走 pymupdf 备用路，问答仍可用**（`ingest.qa_blocked_reason` 只剩"摄取进行中"一种拦截；`graph/__init__.py` 同口径） |
| 6 | 摄取 job 图 | "job 内并行 / 两路互不依赖" | `worker.py` 的 `serial = mineru_skeleton_enabled()` 默认 True → **默认串行**；`=0` 才并行 |
| 7 | env 表标题 | "默认**关**的实验（需显式开启）" | 表里混着多个默认**开**的项（`V3_NOL3J`、`MINERU`、`ASK_CONCURRENCY=4`、`ASK_WAIT_S=300`、`MAX_PDF_MB=200`）→ 改标题 |
| 8 | CLI · `run_summary.py` | "语义去重 / 角色打标 / 打分" | 实际是 **claims 全量汇总 + 证据命中(ev✗)审计（不调 LLM）**；去重/打标/打分在 `run_view.py` |
| 9 | CLI · `run_view.py` | "装配视图 / 产物检查" | 实际是 **去重 → 打标 → 算分 → 渲染摘要视图** |
| 10 | API 表 | 缺 `GET /pdf/{name}` | 该端点是前端 pdf.js 与"跳原文"的必需项（`web/app.py`） |
| 11 | 测试数 / 中文 QA 成绩 | "178 用例"、"21 篇 176/176" | 实测 `pytest --collect-only` = **203（+5 deselected）**；题集现为 **31 篇 / 304 题**（第一组 5 篇已按 gold 引文口径重建为 70 题） |

### 5.2 其它文档

| 文档 | 改动 |
|---|---|
| `DEVELOPMENT_LOG.md` | ① 文首加**性质声明**（流水账 ≠ 现状；冲突信 README/代码）；② "摄取异步化"的**并行依据**就地更正（前提已被 MinerU 默认骨架推翻 → 默认串行）；③ 两处 `178 用例` → 203 |
| `docs/RAG_COMPONENT_NOTES.md` | ① 文首加**现状核对横幅**（组件默认值已逐条对过源码）；② 中文回归"31 篇 / 276 题" → **304 题**（并注明 276 是当时口径） |
| `qa/README.md` | 用例数 178 → **203**；`questions/` 行补"304 题 / 第一组 70 题 gold 口径" |
| `design.md` | 加**早期设计稿横幅**（哪些"预留"已实现/被替代），避免被当成现状 |
| `retrieval/README.md` | 文首加**研究区 / 生产区分离**声明（LitSearch 口径 ≠ 线上工具① 的 arXiv 口径） |

### 5.3 本轮**没有**做的（诚实交代）

- `qa/recall/*.md`（91 份）、`retrieval/results/*.md`（27 份）里的**逐条数字**没有复核 ——
  它们是**带日期的历史记录**，复核成本高而价值低；**引用它们时必须带日期**，且不要与最新口径混算。
- `qa/` 下 20+ 份 `qa_summary_*.md`（多为 1KB）体积虽小但**互相引用**（README/组件笔记会点名某一份当基线），
  故**未做物理搬移** —— 归档语义由本文件承担（§2 已说明哪份是基线）。
- `retrieval/results/` 里 **6 份 0KB 空文件**未删（`ARXIV_CE100-100_*`、`ARXIV_SCORE100-100_*`、
  `ARXIV_GATE100-100_*`、`LITSEARCH_CE100-*_MULTI_CORPUS`、`LITSEARCH_SCORE100-100_*`、
  `LITSEARCH_SCREEN_PROBE`）→ 建议**重跑或删除**，别当"已测过"。

### 5.4 ⚠️ 本轮发现的**入库异常**（需要你决定，我没动 git）

`git ls-files retrieval` = **0**，`git ls-files --others retrieval` = **119 且未被 ignore** ——
**整个 `retrieval/` 都没进 git**：

| 未入库的东西 | 为什么该入库 | 为什么可能不该 |
|---|---|---|
| `retrieval/README.md`（56KB 研究文档） | 是篇级检索的**唯一设计说明** | — |
| `retrieval/scripts/*.py`（46 个，含题集流水线 5 个） | README「开发约定」明写 **`retrieval/scripts/*.py` 要入库** | — |
| `retrieval/results/*.md`（27 份报告） | 同上明写 **`retrieval/results/*.md` 要入库** | — |
| `retrieval/tmp/group1/` | gold 题集（唯一真源） | **含论文原文全文与版面图** → 与"含原文片段不宜入库"的既定策略冲突 |

**即：README 说的"入库"，实际没入**（文档与现实又一处偏差）。建议处置：

```bash
# ① 资产入库（与 README 开发约定一致）
git add retrieval/README.md retrieval/scripts retrieval/results/*.md

# ② 素材不入库（把 tmp 显式排除，避免误提交论文原文）
echo 'retrieval/tmp/' >> .gitignore

# ③ 那 6 份 0KB 空报告：重跑或删掉，别当"已测过"
```
> 代价（必须先知道）：`retrieval/tmp/<group>/*.questions.json` 是**唯一真源**，一旦不入库就只在本机；
> 入库的 `qa/questions/*.json` 是**导出产物**（无逐字引文）。若想给 gold 上版本控制，
> 就把 **`*.questions.json` 单独 un-ignore**（它只有短引文，不含全文）。

---

## 5.5 本轮整理（2026-10-01 · 四工具收尾 + cli 归类）

| 改动 | 内容 |
|---|---|
| 代码 | `cli/` 分两类：**工具/报告入口留根**（`main`/`run_fetch`/`run_search`/`run_pipeline`/`run_ingest`/`run_set` + `run_report`/`run_claims`/`run_summary`/`run_view`/`run_skeleton`/`run_figures`）；**评测·跑批移入 `cli/eval/`**（14 个 + `_anchors.py`） |
| 代码 | 新增 `cli/run_set.py`（**工具④ 多答案开放域问答**入口，走 `set_judge.run` + `render`） |
| 代码 | `components/query_optimizer.py` 的 **L5 `decompose()` 已实现**（此前 docstring 标"未实现"）；测试 `test_unimplemented_level_raises` 去掉 `l5` |
| 代码 | `components/set_judge.py` 判官默认 **单判官 A**（`PAPERPILOT_SET_DUAL` 默认 0；见 `retrieval/results/R2_JUDGE_PROTO_20261001.md`） |
| 文档 | `design.md` → `archive/`（早期设计稿，此前已标注"只作对照"） |
| 文档 | README：CLI 段改为"四个工具" + `cli/eval/` 说明；评测表路径同步 |

⚠️ §5.3 提到的"6 份 0KB 空报告"**现已不存在**（2026-10-01 扫描 0 命中）—— 该条作废。

---

## 6. 维护约定（下次改完代码顺手做）

1. **改行为就改 README 的对应段**（尤其是 §行为开关 / 数据流 / 问答链路三处）；
2. 新增报告 → 文件名带日期，并在本文件 §2 登记一勾；
3. 结论被推翻 → **不要删旧报告**，在旧报告顶部加一行"已被 X 取代"并链到新报告；
4. 文档里凡是**数字**，必须写清**口径 + 日期 + 语料**（这句是复现事故最多的地方）。
