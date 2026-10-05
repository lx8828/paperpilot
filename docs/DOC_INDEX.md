# 文档总地图（DOC_INDEX）

> **建立**：2026-09-22 ｜ **最近校验**：2026-10-06
>
> 这份文件只回答两个问题：**当前应该信哪份文档**、**历史报告去哪里查**。
> 冲突时的优先级：

```
代码  >  README.md（根）  >  docs/*.md  >  qa/*.md、retrieval/results/*.md（历史报告）
```

> 代码行为与文档冲突时，先改文档，不要把历史结论当现状。

---

## 1. 先看能力地图

| 能力 | 定位 | 当前状态 | 对应评测 |
|---|---|---|---|
| **RAG1** | 论文级检索：方向问题 → 候选论文 | 现行；稳定交付 **top-5**，LLM 精排上限 **top-10** | **LitSearch**（64,183 篇 / 597 查询） |
| **RAG2** | 论文块级 reader 检索 | **默认关闭**；代码保留，作 A/B 对照/回退 | r2dev 证据召回 / reader 判定 |
| **MultiAnswer** | 论文集合级多答案：哪几篇满足问题 → **回答**（LLM 汇总） | ★ **独立页 `/ma`**（工作台左下角进入）；语料 = 本工作区那批论文，规模**固定 ≤10 篇** | **QAMPARI**（对外）+ 三簇论文集（内部） |
| **Fullctx** | 报告层不够时全文直读 | 论文问答默认 reader | 三列公平对比与端到端 QA |

> **QAMPARI 不是独立模块**，它属于 MultiAnswer，用来检查多答案检索能力；
> 关系等同于 **LitSearch 是 RAG1 的评测**。
> **RAG2 也不是 MultiAnswer**：前者是块级 reader 检索，后者是集合级多答案检索。

---

## 2. 现行文档

| 文档 | 看什么 | 注意 |
|---|---|---|
| **`README.md`** | 能力地图、快速开始、接口、CLI、架构、评测、开发约定 | **第一入口**；行为/接口/默认值优先看它 |
| ★ **`docs/RESULTS_SUMMARY.md`** | **成果汇总报告**：做了什么 / **上线了什么** / 上线效果 / **没上线的为什么** / **大 N 重采样外推** / 可信度支撑（显著性·消融·A/B·泄露·污染） | ★ **对外汇报的第一入口**；每个数字**带出处**，引用前请点回原文件 |
| **`docs/RAG2_CORPUS_WIRING.md`** | **MultiAnswer 语料接线与能力边界**：★ 前端形态（独立页 `/ma`，2026-10-06）、语料口径、升级路径、指标不可比 | 文件名里的 **RAG2 是历史遗留**；正文已按 MultiAnswer 组织，QAMPARI 归入其中 |
| `docs/RAG_COMPONENT_NOTES.md` | 组件默认值与实测依据（Splitter / Router / Retriever / Reranker / Generator / Validator 等） | 数字必须带日期与口径读 |
| `docs/RETRIEVAL_LOG.md` | RAG1 的篇级检索设计、语料与消融（LitSearch 口径） | 研究区指标，不与线上业务数字混算 |
| `qa/COMPARE_DESIGN.md` | B0/B1/B2 三列公平对比设计 | 数字看最新完整报告 |
| `retrieval/scripts/` 与脚本 docstring | 出题、评测、导出契约 | `_qa_groups.py` 是分组唯一真源 |

---

## 3. 评测层级与模块归属

| 层 | 评什么 | 主要入口 | 对应能力 |
|---|---|---|---|
| **L0** | 仓库不变量、路径存活、gold 与导出同步 | CI / `evals/checks/` | 全项目 |
| **L1** | 篇级检索：Recall@k / nDCG | `cli/eval/run_retrieval_eval.py` | **RAG1**（LitSearch） |
| **L2** | 证据召回 / reader 判定 | `retrieval/tmp/_r2_*.py` | **RAG2**（r2dev） |
| **L3** | 端到端问答与报告链路 | `cli/eval/run_group_qa.py` | Fullctx / 产品链路 |
| **L4** | 对外可比基准 | `evals/runners/l4_loft.py` | **MultiAnswer（当前主用 QAMPARI）** |

**MultiAnswer 的关键文件（2026-10-06）**

| 文件 | 作用 |
|---|---|
| `web/multianswer.html` + `GET /ma` | ★ **多答案展示独立页**（本工作区那批论文 → 回答 + 依据）；语料由 URL `?pdfs=` 唯一确定 |
| `src/paperpilot/multianswer/synth.py` | ★ **LLM 汇总**：把逐篇判定塞给 LLM 生成**回答**（失败回退到「依据」清单） |
| `src/paperpilot/multianswer/pipeline.py` / `cli/run_rag2_flow.py` | 离线编排：方向 → RAG1 → 取料 → 切块 → MultiAnswer |

> `evals/reports/*.jsonl` 不入库；入库摘要是 **`evals/RESULTS.md`** 与 **`evals/baselines/metrics.json`**。
> 干净 clone 不能引用本机路径或未入库原始记录。

---

## 4. 历史报告（只在追溯结论时看）

| 位置 | 当前规模（2026-10-06） | 口径 |
|---|---:|---|
| `qa/recall/` | 91 份 | 2026-09-06 ~ 09-12；表格/检索 A-B 与负结果 |
| `qa/**/*.md` | 101 份 | 带日期的阶段报告，不能当现状 |
| `retrieval/results/*.md` | 86 份 | 篇级检索与 MultiAnswer 实验记录 |
| `retrieval/results/R2_PROD_FINAL.csv` | 1 份 | MultiAnswer 三簇论文域汇总（已入库） |
| `retrieval/results/R2_QAMPARI_*.md` | 多份 | QAMPARI：MultiAnswer 的对外评测 |

> `R2_*` 多数指“第二轮实验”，**不等于 RAG2 模块**；看文件名不能直接推断能力归属。

---

## 5. 已归档 / 不入库

| 位置 | 性质 |
|---|---|
| `archive/qa_funnel_v2/` | v2 四层漏斗快照，只作对照 |
| `archive/design.md` | 早期设计稿，只作对照 |
| `assets/**` | 论文、产物、运行数据；本机可再生成 |
| `qa/snapshots/`、`qa/_archive/`、`qa/_scratch/` | 本机快照/清理区 |
| `retrieval/data/` 多数文件 | 语料/索引；由 `.gitignore` 管理，仅保留可复现真值与说明 |
| `evals/reports/` | 跑批原始 JSONL；入库摘要见 `evals/RESULTS.md` |

---

## 6. 维护约定

1. 改行为先改 **README** 的接口/默认值/数据流；再同步本索引。
2. 新增报告：文件名带日期，写清**口径 + 语料 + 样本量**。
3. 旧结论被推翻：保留历史报告并在顶部标注“已被 X 取代”。
4. **不要**把历史清理流水账当现状；一次性修复过程统一进 `DEVELOPMENT_LOG.md`。
5. 凡是“检索能力”表述，必须明确写 **RAG1 / RAG2 / MultiAnswer / QAMPARI** 中的哪个。
