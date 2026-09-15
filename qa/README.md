# `qa/` 目录说明（评测 · 实验记录）

> ⚠️ **这里不是测试套件**。测试在 `tests/`（一条命令 `uv run pytest -q`，178 用例、无需 key）。
> `qa/` 放的是**评测（质量）与实验记录**：判"高低"，有噪声、要 key/模型，**按需手动跑**。
> 两者的分工见 README 的「三层测试」小节。

## 先看这三份

| 文档 | 内容 |
|---|---|
| [`QASPER_EVAL_LOG.md`](QASPER_EVAL_LOG.md) | QASPER 第三方基准的逐次记录（**84.3%** 那条线的来龙去脉、每次改动的归因） |
| [`RETRIEVAL_EVAL_FRAMEWORK.md`](RETRIEVAL_EVAL_FRAMEWORK.md) | **评测脚本清单**：每个脚本的定位 / 样本量 / 得分 / 是否可复用 |
| [`CAMPAIGN_20260906-07.md`](CAMPAIGN_20260906-07.md) | v2→v3 的**决策史**（为什么砍 L2、为什么表池并集设为默认、什么是负结果） |

## 目录一览

| 目录 / 文件 | 是什么 | 入库 |
|---|---|---|
| `recall/` | 检索与表格线的主战场：报告（`.md`）、评测口径模块（`_tgt.py`）、保留的 runner | ✅ 141 个 |
| `questions/` | 中文 QA 题集（31 个 json） | ✅ |
| `compare/` | B0 / B1 / B2 三列对照（设计 + 报告 + 配对 McNemar / 检验功效脚本） | ✅ 12 个 |
| `reader/` `negqa/` `robust/` `stress/` | 四个专项：外行问答 / 防幻觉 / 同义改写稳定性 / 对抗诱导 | ✅ |
| `review/` | 代码审查的复现脚本与应答台账 | ✅ |
| `snapshots/` | 场景快照（**本机生成、未入库**；生成脚本在 `cli/`） | ❌ |
| `_archive/` | **本机留档**：136 个一次性诊断脚本（2026-09-14 从 `recall/` 移入；清单见下） | ❌ 被 `.gitignore` 忽略 |
| `_scratch/` | 本机临时脚本（不入库） | ❌ 被忽略 |
| `_overnight_*.py`（顶层 4 个）+ `_launch_*.ps1` | 夜间批跑的编排脚本（campaign 期间用） | ✅ |

## 两件必须知道的事（避免把"正常"当成"坏了"）

1. **报告里引用的 `qa/*.json` / `*.jsonl` 运行记录，多数没入库**。
   `qa/` 下入库 238 个文件（136 `.md` / 40 `.json` / 35 `.py` …），本机另有 ~250 个运行记录与日志。
   所以看到"某个 `qa/qasper_run_*.json` 不存在"，通常不是文档坏了，而是**那份记录只在本机**
   （README 里也写了「本地运行记录，未入库」）。
2. **2026-09-14 归档了 136 个一次性脚本**（`_diag_*` / `_ab_*` / `_probe_*` / `_scan_*` / `_verify_*` …）：
   它们现在在 `qa/_archive/recall/`（**本机留档、不入库**，git 历史仍在），
   **清单 + 每个脚本的一句话用途见 [`recall/ARCHIVED.md`](recall/ARCHIVED.md)**。
   `recall/` 只保留 23 个仍被引用的（例如 `_tgt.py` 是测试用的评测口径模块）。
   因此：**老报告里指向 `qa/recall/_xxx.py` 的路径可能已经不在**，去 `ARCHIVED.md` 里找。

## 怎么跑（都需要真 key / 本地模型，CI 不跑）

| 目的 | 命令 | 需要 |
|---|---|---|
| QASPER 基准（第三方题集 + 异源裁判 1~5 分，≥4 pass） | `uv run python cli/run_qasper_eval.py --papers N` | LLM + 裁判 key |
| 中文 QA 回归（手写题集 + 基线 diff） | `uv run python cli/run_qa_v2.py [--limit N] [--save-baseline]` | LLM key |
| 产物层回归护栏（**不调 LLM**，与 `bench/baseline.json` 勾叉 diff） | `uv run python cli/run_bench.py [--save-baseline]` | 本机 `assets/` 论文与产物 |
| 三列对照（B0 全文 / B1 摘要 / B2 检索增强） | `uv run python cli/run_compare.py` | LLM + 向量模型 |
| 检索/分块评测 | `uv run python cli/run_retrieval_eval.py`、`cli/run_chunk_eval.py` | 向量模型 |
| 四个专项 | `qa/reader/_run_reader.py`、`qa/negqa/_run_neg.py`、`qa/robust/_run_robust.py`、`qa/stress/` | LLM key |

> 这些脚本给的是**分数**（单次 churn 8~25%，见 README），不是 pass/fail；
> "对错"那一层在 `tests/`，由 CI 每次 push 自动跑。
> 想要**真模型能不能跑**的 pass/fail → `uv run pytest -m local`（4 条冒烟：真向量检索位次 /
> 真 LLM 作答带引用 / 真 MinerU 解析 / 真实裁判体检；缺环境会 skip 而不是红）。
