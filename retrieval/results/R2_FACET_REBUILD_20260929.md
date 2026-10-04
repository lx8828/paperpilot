# 扩大抽检 → 重建 facet → 接线（RAG-2 · 20 篇论文线）

日期：2026-09-29
脚本：`_r2_gold_recalib2.py`（可换 facet 表）、`_r2_facets_v2.py`、`_r2_gold_adjudicate.py`、
`_r2_gold_consensus.py`、`_r2_facet_audit.py`、`_r2_dispute_view.py`、`_r2_reader.py`（已接线）

---

## 1. 为什么要重建 facet

旧题集是**语料构建时用词面锚点筛的**（要求锚点正例率 15%~75%）
→ **假阳性也能"撑起"一道题**。实测（`_r2_facet_audit.py`，全 19 facet × 3 簇 = 57 组合）：

| 例 | 锚点正例 | 真值正例 | 说明 |
|---|---|---|---|
| `簇1/2/3 deployment` | 4 / 6 / 2 | **0 / 0 / 0** | 12 个锚点正例**全是假阳性**（`real-world applications` 指问题领域、课程讲师页…） |
| `簇1 multilingual` | 1 | 0 | 同上 |
| `簇2 context_length` | 1 | 0 | 同上 |
| `簇3 safety_bias` | 7 | 1 | 7 个里 6 个是"只提 harmless" |

---

## 2. 扩大抽检（三步，逐层收紧）

### 2.1 第一步：旧口径的第三判官复核（n=172）

用**独立第三判官** `deepseek-reasoner`（A/B 是 `deepseek-chat`/`glm-4-flash`）复核
**全部 dispute（删去 87 + 新增 38）+ 每层抽样 25**：

| 分层 | 条数 | 一致率 |
|---|---|---|
| `removed`（锚点正→真值负） | 87 | 86.2% |
| `added`（锚点负→真值正） | 37 | 86.5% |
| `agree_pos` | 23 | 100% |
| `agree_neg` | 25 | 96% |
| **总体** | **172** | **89.5%** |

**18 条分歧逐条人工判读**（`_r2_dispute_view.py`）后的结论：
· **新真值误删 7**（如 `efficiency` 在 **Limitations** 里报了 `2 examples/s`；
  `significance` 表注有 `* denotes significant differences (p<0.05)`；
  `multilingual` 本文在 TyDiQA/MGSM 上评测了；`context_length` 是 Long-Range Transformer 工作）
· **新真值误收 3**（`safety_bias` 只讲 harmful vs helpful；`case_study` 只有定量分析）
· **新真值对 5**（第三判官过宽：`context_length` 只是"被 1024 限制"、`deployment` 只是动机…）
· 边界 3

**关键观察：分歧 100% 集中在"定义有歧义"的 facet 上；
定义最清晰的 `ablation` / `code_release` / `fine_tuning` 一条分歧都没有。**

### 2.2 第二步：把 facet 定义改成"可判定"（`_r2_facets_v2.py::F2`）

把题面直接写成「**算什么 ∧ 不算什么**」（例：`efficiency` → 「报告**模型推理本身**的时间/延迟/
吞吐/FLOPs，**无论出现在哪一节**（含 Limitations）；不算人工标注速度、训练成本」）。
锚点同步收紧（`ablation` → `\bablat`，修 v1 漏动词 `ablate`）。

**用新定义重标 893 对**（`gold_recalib3.csv`）：锚点 P 0.546 / R 0.799，真值 149 条。

**再抽检 → 一致率 89.1%（vs v2 的 89.5%）——没有改善。**

| 分层 | v1 定义 | v2 定义 |
|---|---|---|
| `removed` | 86.2% | **90.8%** ↑ |
| `added` | 86.5% | **74.1%** ↓ |
| 正例错误率 | 8.3% | **19.2%** ↑ |

→ **结论：收紧定义只是把误差从"漏"挪到"误收"，净效果持平。**
  且细读分歧发现**很多是第三判官自己漏了**（`We train a Transformer decoder…` 被判非微调、
  `The average of 3 runs with different seeds` 被判非多次运行）。
→ **边界分歧是任务固有的主观性，靠换定义/换判官消不掉。**

### 2.3 第三步：三判官多数票（最终方案）

对**全部 893 对**跑三个异源判官，取多数票（票型即置信度）：

| 票型 | 条数 | 占比 |
|---|---|---|
| 3:0 全票正 | 66 | 7.4% |
| 2:1 多数正 | 75 | 8.4% |
| 1:2 多数负 | 45 | 5.0% |
| 0:3 全票负 | 707 | 79.2% |

**正例 141 ｜ 全票一致 86.6% ｜ 锚点 P 0.541 / R 0.837**

---

## 3. 重建后的题集（★ 最终）

判据：正例率 ∈ [10%, 90%] ∧ 正例 ≥2 ∧ 负例 ≥2；**核心集 = 同一 facet 在 ≥2 簇可用**。

### 3.1 核心集（**10 个 facet / 28 组合**，跨簇复现）

| facet | ✅簇 | 总正例 | 各簇（正例/篇 = 率） | **分歧率**（2:1 占比） |
|---|---|---|---|---|
| `code_release` | 3 | 27 | 9/16=56% ｜ 8/18=44% ｜ 10/13=77% | **12.8%** ← 最稳 |
| `error_analysis` | 2 | 6 | 3/16=19% ｜ 3/18=17% | **11.8%** ← 最稳 |
| `human_eval` | 2 | 7 | 3/16=19% ｜ 3/18=17% | 14.7% |
| `annotation_cost` | 2 | 4 | 2/16=12% ｜ 2/18=11% | 20.6% |
| `case_study` | 2 | 6 | 4/16=25% ｜ 2/13=15% | 24.1% |
| `ablation` | 3 | 18 | 7/16=44% ｜ 6/18=33% ｜ 5/13=38% | 25.5% |
| `efficiency` | 2 | 7 | 2/18=11% ｜ 4/13=31% | 25.8% |
| `zero_few_shot` | 2 | 9 | 5/16=31% ｜ 4/18=22% | 26.5% |
| `significance` | 3 | 12 | 3/16=19% ｜ 7/18=39% ｜ 2/13=15% | 29.8% |
| `fine_tuning` | 3 | 29 | 12/16=75% ｜ 10/18=56% ｜ 7/13=54% | **34.0%** ← 最不稳 |

### 3.2 扩展集（仅 1 簇）

`knowledge_distill`、`multilingual`、`prompt_eng`、`proof_theory`

### 3.3 剔除（5 个 facet）

`context_length`、`deployment`、`new_dataset`、`rl_training`、`safety_bias`
（全簇正例率 <10% 或正例 <2 —— 在本文语料上**无分辨率**）

### 3.4 与旧题集对比

| | 旧 | 新 |
|---|---|---|
| 组合数 | 35 | **28** |
| facet 数 | 19 | **14**（核心 10 + 扩展 4） |
| 真值来源 | 词面锚点（P 0.895/R 0.567 的旧标定） | **三判官多数票**（锚点 P 0.541/R 0.837） |
| 正例/题 | 5.87 | **4.71** |
| 置信度 | 无 | **有**（票型 + 逐题分歧率） |

产物：`data/r2dev/gold_final2.csv`（451 条 / 132 正例）、
`data/r2dev/facets_v2_selected.json`（核心/扩展/剔除 + 逐题票型）。

---

## 4. 接线（`_r2_reader.py`）

| 环节 | 旧 | **新（默认）** | 回退 |
|---|---|---|---|
| 真值 | `gold_recalib.csv`（v1，锚点校准） | **`gold_final2.csv`**（三判官多数票） | `--legacy` |
| facet 题面 | `_r2_std_metrics.F`（v1 简述） | **`_r2_facets_v2.F2`（操作化）** | `--legacy` |
| 题集 | `clusters/meta.json` 的 `usable`（35） | **`facets_v2_selected.json`（28）** | `--legacy` |
| 切块 | `chunks_of(full_paper)` 固定窗 1000/900 | **`prodchunk/mineru/`（生产章节段落 · MinerU）** | `--legacy` |
| `max_seq_length` | 512（截断坑） | **8192**（生产值） | `--legacy` |

**修复的一个真 bug**：`load_gold2()` 起初忘了按 `gold` 过滤 → 每题 gold 变成"全部篇"
→ 全部题被 `len(g)==len(docs)` 跳过（跑出 0 题）。已修 + 注释说明。

---

## 5. 接线后结果（28 题 / 451 次逐篇判定 / **29 秒**）

| 指标 | **新口径** | 检索基线 top-10 |
|---|---|---|
| 集合 P | **0.764** | 0.389 |
| 集合 R | **0.791** | 0.866 |
| **集合 F1** | **0.741** | 0.498 |
| 交付篇数 | 4.50（gold 4.71） | 10 |

### 与旧口径同指标对照（⚠️ **非 apples-to-apples**：题集与真值都变了）

| 文件 | 题数 | gold/题 | readerP | readerR | **readerF1** | retF1 | 增益 |
|---|---|---|---|---|---|---|---|
| `R2_r2reader_b12`（旧） | 30 | 5.87 | 0.839 | 0.891 | **0.846** | 0.539 | +0.307 |
| `R2_r2reader_b6`（旧） | 30 | 5.87 | 0.879 | 0.801 | **0.817** | 0.539 | +0.278 |
| **`R2_r2reader2_b6`（新）** | **28** | **4.71** | 0.764 | 0.791 | **0.741** | 0.498 | **+0.243** |

→ 绝对值下降（0.817→0.741）主要来自**真值更严**（gold 5.87→4.71/题、检索基线 F1 0.539→0.498）；
→ **判定层相对检索的增益基本保持**（+0.278 → **+0.243**）。

### 短板诊断（`_r2_reader_cmp.py`）

| 方向 | 位置 | gold vs 判 yes | F1 |
|---|---|---|---|
| ➖ **偏严** | `簇1 fine_tuning` | **12 vs 5** | 0.471 |
| ➖ 偏严 | `簇1/2/3 code_release` | 9/8/10 vs 6/4/5 | 0.667 ×3 |
| ➕ 偏松 | `簇1/2 annotation_cost` | 2 vs 4/6 | 0.667 / 0.500 |
| ➕ 偏松 | `簇1/2 human_eval` | 3 vs 5 | 0.750 |

**根因（重要）**：**reader 与真值用的判官协议不一致**——
· 真值：**四档**（`entail/partial/neutral/contradict`）+ 双 LLM + 第三轮（`_r2_gold_recalib.SYS`）
· reader：**三档**（`yes/no/unclear`，`_r2_reader.SYS`，旧提示）
→ `code_release`/`fine_tuning` 上 reader 系统性偏严（大量假阴性）。
→ **下一步应让 reader 复用真值那套四档协议**，而不是另立一套。

---

## 6. 边界

1. **多数票不是金标**：三个判官都是 LLM，共模偏差（都偏保守/都受同一提示影响）无法排除；
   `1:2`/`2:1` 的配对（共 120 条，13.4%）本质是**不确定**的，已在 `facets_v2_selected.json` 里记 `disagree_rate`。
2. **判官 B（glm-4-flash）偏保守**且 `contradict` 恒为 0；有 16 条调用错误（已剔除该条）。
3. **`fine_tuning` 分歧率 34%** 最高 → 它的指标应视为**低置信**。
4. **换定义没有提高一致率**（89.5% → 89.1%）——这是本次的**负面结果**，说明"更细的定义"不必然更好。
5. 语料仍是 **47/60 篇**（4 篇无 PDF + 7 篇 acm/doi 主动跳过）；1 篇 MinerU `degraded`。
6. 接线后的对照**不是 apples-to-apples**（题集/真值同时变了），只能说明量级与相对增益。

---

## 7. 复现

```powershell
cd f:/paperpilot
$env:HF_HOME='F:\hf_cache'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
# 重标（v1 定义 / v2 定义）
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_recalib2.py --all-facets --workers 10
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_recalib2.py --all-facets --facets-file _r2_facets_v2.py --out gold_recalib3.csv --workers 10
# 第三判官（全量，供多数票）
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_adjudicate.py --recalib gold_recalib3.csv --all-pairs --workers 8
# 多数票 → 最终真值 + 题集
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_consensus.py
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_facet_audit.py --recalib gold_recalib3.csv --min-pos 2 --min-neg 2
# reader（新口径）与对照
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_reader.py --workers 8
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_reader.py --legacy --tag r2reader_b6   # 复现旧口径
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_reader_cmp.py
```
