# MultiAnswer · 多答案开放域检索：语料接线与能力边界

> **文件名说明**：本文原名 `RAG2_CORPUS_WIRING.md`，是历史命名。
> **RAG2** 在本项目里指论文**块级 reader 检索**（`PAPERPILOT_QA_READER=retrieval`）；
> 本文讲的是 **MultiAnswer：论文集合级多答案检索**。两者不是同一条链路。
>
> **QAMPARI 属于 MultiAnswer 的评测体系**，用于检查多答案检索/覆盖能力；
> 对应关系是 **QAMPARI : MultiAnswer = LitSearch : RAG1**。
> QAMPARI 不是独立模块。
>
> 日期：2026-10-06 ｜ 现状一律以代码为准。

---

## 0. 一句话定位

MultiAnswer 回答的是：

> **在一组论文里，哪几篇满足问题，并给出逐篇证据。**

它不是单篇论文问答，也不是 RAG2 块级 reader。系统里三条检索能力必须分开：

| 名称 | 作用 | 当前状态 |
|---|---|---|
| **RAG1** | 方向问题 → 候选论文；篇级召回与排序 | 现行；稳定交付 **top-5**，LLM 精排上限 **top-10** |
| **RAG2** | 论文块级召回 → reader 生成答案 | 默认关闭；只作对照/回退 |
| **MultiAnswer** | 在论文集合里做多答案检索：哪几篇满足问题 → **回答** | 独立页 `/ma`（工作台左下角进入） |
| **QAMPARI** | MultiAnswer 的对外 passage 级基准 | 属于 MultiAnswer 评测，不是独立模块 |

> `R2_*` 文件大多是“第二轮实验”的历史命名，**不等于 RAG2 模块**。

---

## 0.2 前端形态：独立页 `GET /ma`（2026-10-06 简化）

**入口**：工作台**左栏左下角**「🎯 多答案展示」→ 跳**独立页**。

| | |
|---|---|
| 页面 | `web/multianswer.html`（路由 `GET /ma`） |
| 语料 | URL 的 `?pdfs=a.pdf,b.pdf,…`（**裸文件名**，相对 `assets/papers/`）—— 这就是它的全部语料；后端**没有**「当前批」的会话状态，参数丢了语料就没了 |
| N | **没有 N 选择器**：后端**强制 `n = len(pdfs)`**，且 `len(pdfs) ≤ _MA_LIVE_MAX`（10） |
| 产物 | **回答**（`multianswer.synth.synthesize()` 把逐篇判定**塞进 LLM 汇总**）＋「依据」（`set_judge.render()` 的逐篇清单，页面折叠区） |

入口这样收拢，是因为「方向检索」和多答案本来就是同一件事（**主题类查询 → 10 篇论文**），
下游可以复用同一批语料。

job 的字段：`answer`（回答）／`answer_ok`／`answer_error`／`evidence_text`（依据）。

§1–§4 讲的是**语料与离线评测口径**，与前端形态无关。

---

## 1. MultiAnswer 的两种**语料档**（仅用于离线评测 / CLI）

独立页 `/ma` **只有一条路**：URL 的 `pdfs` 决定语料（固定 ≤10 篇）。
下面两档**只用于离线评测 / CLI**（`cli/run_rag2_flow.py`、`cli/eval/run_multianswer.py`）：

| 语料档 | 语料来源 | 规模 | 用途 |
|---|---|---|---|
| **实时档** | RAG1 现场检索 → arXiv 取料 → 切块 | **≤10 篇** | 验证实时编排（CLI `--direction`） |
| **离线簇** | 预生成的固定论文簇 | **50 篇/簇** | 稳定评测（CLI `--cluster` / `--chunks`） |

### 1.1 三步编排（离线 CLI 的链路）

| 步骤 | 动作 | 接口 / 实现 |
|---|---|---|
| ① 方向检索 | 方向问题 → RAG1 → 候选 arXiv id | `POST /api/direction/search` |
| ② 取料下载 | arXiv id → PDF → MinerU 摄取 | `POST /api/multianswer/fetch` |
| ③ 多答案检索 | 问题 + 语料 → 逐篇判定 → **LLM 汇总成回答** | `POST /api/multianswer/search` |

### 1.2 后端契约

`POST /api/multianswer/search` 用 `pdfs` 是否存在来决定**语料来源**：

| 请求 | 语料来源 | 规则 |
|---|---|---|
| 带非空 `pdfs` | **就这批 PDF**（独立页 `/ma` 走这条） | `n` 强制等于 `len(pdfs)`，且 `len(pdfs) ≤ 10` |
| 不带 `pdfs` | 服务端语料（`PAPERPILOT_MA_CORPUS`，默认 `prodchunk50/mineru/c0.parquet`） | `n` 生效 —— **离线评测路径**，前端不用 |

完成后 job 带 `answer` / `answer_ok` / `answer_error` / `evidence_text` / `candidates`。

`GET /api/meta` 里：**独立页只用** `ma_live_max` / `ma_stages`；
`ma_n_options` / `ma_n_max` / `ma_corpus_exists` **仍返回，但只服务离线评测** —— 前端不再渲染 N。

---

## 2. 规模口径：5 / 10 / 50 / 150 分别是什么意思

五个数的含义与用法：

| 数字 | 含义 | 能不能当效果结论 |
|---|---|---|
| **top-5** | RAG1 原本稳定的交付工作区 | 可以；这是 RAG1 设计目标 |
| **top-10** | LLM listwise 精排上限 | 只能表示“排到了第 10 名”，不等于质量已验证 |
| **≤10** | MultiAnswer **实时档**（离线编排）的工程上限 | 只证明**链路可跑通**；当前**没有 N=10 独立成绩表** |
| **50** | **离线簇**一次加载的论文规模 | 用于评测；属于研究区语料 |
| **150** | 3 簇 × 50 的离线评测总量 | 不是在线一次输入 150 篇 |

一句话：

> **RAG1 稳定交付是 top-5；MultiAnswer 实时档最多 10 篇（离线编排上限，也是独立页 `/ma` 的在线上限）；
> 50 是离线簇的规模；150 是离线三簇总量。**

---

## 3. QAMPARI 在 MultiAnswer 下的位置

QAMPARI 是 MultiAnswer 的**对外可比评测**，不是独立功能模块。
它和 LitSearch 的关系类似：

| 模块 | 对应评测 | 测什么 |
|---|---|---|
| RAG1 | **LitSearch** | 篇级 Recall@k / nDCG |
| RAG2 | r2dev 证据召回 / reader 判定 | 块级召回与 reader 质量 |
| **MultiAnswer** | **QAMPARI** | 多答案检索、覆盖与子串匹配 |
| MultiAnswer · 论文域 | 三簇 × 50 篇 | 论文产品场景的集合级 P/R/F1 |

### 3.1 QAMPARI 外部口径

- 数据：LoFT RAG / QAMPARI，128k passage 档，100 题。
- 指标：`em`、`coverage`、`subspan_em`。
- 注意：`coverage` 只有 recall；官方 `f1` 在 multi-value 分支恒为 0，不能引用。
- 该档保证 gold 存在，**测不了空答案场景**；其中有一部分来自模型闭卷知识。
- QAMPARI 是 passage 语料基准，**不能替代论文语料的端到端结论**。

摘录的运行结果：

| 运行 | `em` | `coverage` | `subspan_em` |
|---|---:|---:|---:|
| `direct128k_k20` | **0.490** | 0.799 | **0.700** |
| `audit_k40_k40` | 0.470 | 0.815 | 0.690 |
| `exh_k40_k40` | 0.440 | 0.821 | 0.700 |
| `retr1m_k40_k40` | 0.230 | 0.769 | 0.570 |
| `exh_k40_ce_k40` | **0.150** | **0.860** | 0.180 |

> `exh_k40_ce_k40` 的 coverage 最高、em 最低，说明**不能挑单一好看指标宣传**。
> 汇总看 `evals/RESULTS.md`；原始 `evals/reports/*.jsonl` 不入库。

### 3.2 论文域三簇评测

这是更接近 MultiAnswer 产品场景的内部评测：

- 语料：3 簇 × 50 篇 = 150 篇论文。
- 真值：28 个组合，三判官多数票。
- 指标：篇级 `setP` / `setR` / `setF1`。

| k | 交付/题 | P | R | F1 |
|---:|---:|---:|---:|---:|
| 20 | 8.3 | 0.706 | 0.535 | 0.572 |
| 30 | 10.2 | 0.718 | 0.641 | 0.647 |
| 40 | 11.8 | 0.710 | 0.706 | 0.683 |
| 50 / ∞ | 13.1 | 0.709 | 0.771 | **0.717** |

> 这是**离线簇**指标：语料是预生成的簇，不是 RAG1 现场检索来的。
> **实时档（≤10 篇）目前没有独立成绩表**，不能把这组 50 篇指标搬去宣称它在 ≤10 篇下的效果。

---

## 4. 语料接线与复现

### 4.1 离线簇语料

| 项 | 值 |
|---|---|
| 路径 | `retrieval/data/r2dev/prodchunk50/mineru/c{0,1,2}.parquet` |
| 规模 | 3 簇 × 50 篇 |
| 切块 | `document_cache.ordered_chunks` |
| 真值 | `retrieval/data/r2dev/gold_final3.csv` |
| 题集 | `retrieval/data/r2dev/facets_v2_selected.json` |
| 环境变量 | `PAPERPILOT_MA_CORPUS=<切块 parquet>` |

⚠️ 这些 parquet 与部分真值是本地/研究区产物，干净 clone 可能不存在。
可复现报告与摘要优先看已入库的：

- `evals/RESULTS.md`
- `evals/baselines/metrics.json`
- `retrieval/results/R2_PROD_FINAL.csv`
- `retrieval/results/R2_PROD_FINAL_20261001.md`

### 4.2 复现命令

```powershell
# 离线簇：固定 50 篇簇上跑 MultiAnswer
./.venv/Scripts/python.exe cli/eval/run_multianswer.py `
    --question "哪些论文报告了多次运行的波动或显著性检验" `
    --chunks retrieval/data/r2dev/prodchunk50/mineru/c0.parquet `
    --corpus-n 50

# 换到其它簇
$env:PAPERPILOT_MA_CORPUS="retrieval/data/r2dev/prodchunk50/mineru/c1.parquet"
```

```powershell
# 实时档：方向 → RAG1 → 取料 → MultiAnswer
# CLI 文件名里的 rag2 是历史遗留，实际走的是 MultiAnswer
./.venv/Scripts/python.exe cli/run_rag2_flow.py `
    --direction "如何用检索增强生成做知识密集问答" `
    --n 10 `
    --question "这些论文里，哪些报告了消融实验？"
```

---

## 5. 为什么不现场做 150 篇

RAG1 的设计是“池大、输出小”：

```text
DEPTH = 1000    POOL = 200    K_CE = 50    N_OUT = 10
全量 arXiv → dense + BM25 → RRF → CE 前 50 → LLM 精排前 10 → 交付 top-5
```

实测问题：

| 试验 | 结果 |
|---|---|
| 用论文自己的标题检索 | 4/4 都进不了 top-50 |
| RAG1 top-50 ∩ 既有 50 篇语料 | 10% ~ 29% |
| 放大 pool 到 500 / 2000 / 5000 | 无改善 |
| 中文方向直连 arXiv API | 0 篇（元数据是英文） |
| 英文短语 + `cat:cs.CL` | 约 100% 贴合 |

结论：**RAG1 适合自己的 top-5 交付区；不适合直接拿来现场生成 150 篇语料。**
要扩大 MultiAnswer，必须换检索源或增加 RAG1 的检索/精排预算。

---

## 6. 升级路径

### 6.1 换更合适的论文检索源

硬约束：后续取料工具只能吃 **arXiv id**，所以检索源必须能稳定返回 arXiv id。

可选：

- Semantic Scholar / OpenAlex：语义更合适，但要设相关性阈值。
- arXiv 官方 API：零依赖，但需要把中文方向转成英文短语 + 分类。
- 不把“抽 arXiv id”交给 LLM，避免幻觉 id；让 LLM 只生成检索参数。

### 6.2 增加 RAG1 预算

| 常量 | 现值 | 提高后的影响 |
|---|---:|---|
| `POOL` | 200 | CE 候选更多，召回上限提高，但更慢更贵 |
| `N_OUT` | 10 | LLM 精排队列更长，但成本随 N 增长 |

必须先量再调；盲目放大池深已经被证明无效。

---

## 7. 指标可比性纪律

1. 换语料、换检索源、换 N 档后，旧的 P/R/F1 **不能直接比较**。
2. QAMPARI 是 passage 评测，论文域结论必须看三簇自建集。
3. **实时档（≤10 篇）目前没有独立成绩表**，不能借用离线簇的 50 篇指标。
4. 原始跑批 JSONL 不入库；汇报引用 `evals/RESULTS.md` 与 `retrieval/results/` 报告。
5. 文档必须写清 **模块 / 语料 / N / 日期 / 指标**，否则数字视为不可比。
