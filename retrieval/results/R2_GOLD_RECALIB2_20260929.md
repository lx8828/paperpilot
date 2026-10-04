# A1｜真值重标（生产切块口径）+ 人工抽检

日期：2026-09-29 ｜ 脚本：`retrieval/tmp/_r2_gold_recalib2.py`、`_r2_gold_spotcheck.py`、`_r2_gold_diag.py`、`_r2_gold_finalize.py`

## 0. 为什么要重标

切块已换成**生产章节段落切块（MinerU 口径）**（`R2_CHUNK_SWITCH_20260929.md`），
而旧真值 `gold_recalib.csv` 的双 LLM 校准是**在"固定窗 1000/900 over LitSearch 纯文本"**上做的
→ 证据池不同、锚点命中不同 → **真值必须同批重标**，否则 reader / 指标全建在错真值上。

## 1. 方法：协议**一字不改**，只换证据块

从 `_r2_gold_recalib.py` **import** 提示词与采信逻辑（保证与旧口径可比）：

| 项 | 值 |
|---|---|
| 证据池 | （facet 正则命中的块）∪（**干净查询** `zh` + 净化子查询的语义 top-3/4） |
| 判官 A | `deepseek-chat`（`PAPERPILOT_LLM`） |
| 判官 B | `glm-4-flash`（`PAPERPILOT_JUDGE`，异源） |
| 档位 | `entail / partial / neutral / contradict` + 置信度 + **逐字 quote** |
| 采信 | 双方 `entail` → **yes**；双方非 `entail` → **no**；分歧 → **第三轮严格对抗**（找到"这不算本文做的"的理由就 NO） |
| 四纪律 | 只看片段／区分"本文做的 vs 引用他人做的"／列出≠做了／中英同义改写 |
| **唯一改动** | 证据块 = `prodchunk/mineru/`（生产切块）；`max_seq_length` 512 → **8192**（生产值） |

**规模**：541 对（47 篇 × 35 题）｜**444 秒**｜可续跑（CSV 缓存）

## 2. 结果：锚点真值 vs 新真值

| | 旧口径（LitSearch + 固定窗） | **新口径（PDF + 生产切块）** |
|---|---|---|
| 锚点正例 | — | **190** |
| 新真值正例 | — | **137** |
| **锚点 precision** | 0.895 | **0.621**（38% 假阳性） |
| **锚点 recall** | 0.567（漏 43%） | **0.861**（漏 14%） |
| 新真值/题 | — | **3.91**（锚点 5.43） |

→ **方向与旧口径相反**：旧的是"锚点假阳性少、漏报多"，新的是"**假阳性多、漏报少**"。
→ 机制：新块是**完整的章节/段落**（1,493~1,848 字符）→ 上下文完整 → 正则命中更多，
   但也因此命中大量"**提及/讨论/引用**"（非本文自己做的）。

## 3. 人工抽检（9 例，逐条判读）★

抽检命令：`_r2_gold_spotcheck.py --kind removed|added|err --facet ... --evchars 380`

### 3.1 "删去"（锚点正例 → 判官判负）：6 例

| # | 位置 | 判官 | 证据里的关键句 | 我的判读 |
|---|---|---|---|---|
| 1 | 簇1 `deployment`·C1P1 | A/B neutral 0.95 | `has numerous real-world applications` | ✅ **正确**：这是说 **OpenQA 这个领域**有实际应用，不是本文面向部署 |
| 2 | 簇2 `deployment`·C2P11 | A/B neutral 0.95 | （LLM 领域综述与引用） | ✅ **正确**：无本文自己的部署动作 |
| 3 | 簇2 `deployment`·C2P12 | A **partial** 0.7／B neutral | `when using such models in real-world applications, efficiency considerations are paramount` | ⚠️ **边界偏严**：论文**讨论**了真实应用场景的效率，但**没有**部署/线上系统 → 按 facet 定义判 no 可辩护 |
| 4 | 簇2 `deployment`·C2P13 | A/B neutral 0.95 | `6 Instructors • Aida Nematzadeh, DeepMind …` | ✅ **正确**：证据是**课程讲师介绍页**，完全无关（纯假阳性） |
| 5 | 簇3 `safety_bias`·C3P1 | A/B neutral 0.98 | `Training CT … rankers … back-propagate` | ✅ **正确**：与 safety/fairness/bias **无关** |
| 6 | 簇3 `safety_bias`·C3P11 | A/B neutral 0.95 | `3 Empirical Evaluation Robustness to Document Length` | ✅ **正确**：同上 |

→ **5/6 明显正确，1 例边界偏严**。**"锚点 P 0.621"站得住**：正则确实在命中"讨论/引用/无关页面"里的词。

### 3.2 "新增"（锚点漏报 → 判官捞回）：3 例

| # | 位置 | 判官 | 证据里的关键句 | 我的判读 |
|---|---|---|---|---|
| 7 | 簇1 `ablation`·C1P12 | A/B entail | **`We also ablate the system showing the effectiveness of each feature.`** | ✅ **正确**：锚点正则 `\bablation` **漏了动词 `ablate`** |
| 8 | 簇1 `annotation_cost`·C1P14 | A entail／B neutral → **第三轮 YES** | `We create supervised data by prompting GPT-4 to generate reflection tokens …` | ✅ **正确**：用 GPT-4 自动生成监督数据 = 降标注成本（SELF-RAG） |
| 9 | 簇1 `annotation_cost`·C1P15 | A entail／B neutral → **第三轮 YES** | `… create counterfactual … data with **minimal human supervision**` | ✅ **正确**：同义表达，锚点正则抓不到 |

→ **3/3 正确**。锚点漏报的两个来源：**词形变化**（`ablation` vs `ablate`）与**同义表达**（`minimal human supervision`）。

### 3.3 抽检结论

**新真值更准，不是判官太严。** 锚点在**两个方向**都错，且错因相反：

| 方向 | 数量 | 机制 |
|---|---|---|
| **假阳性**（P 0.621） | 53 条 | 正则匹配到"讨论 / 引用 / 无关页面"里的词 |
| **漏报**（漏 14%） | 19 条 | **词形变化**（ablation/ablate）、**同义表达**（minimal human supervision） |

## 4. 判官行为诊断（`_r2_gold_diag.py`）

| 指标 | A `deepseek-chat` | B `glm-4-flash` |
|---|---|---|
| `entail` | 24.9% | 21.0% |
| `partial` | 1.5% | 3.5% |
| `neutral` | 72.6% | 74.6% |
| `contradict` | 1.1% | **0.0%** |
| 调用错误 | 0.0% | **0.9%（5 条）** |

| 采信路径 | 占比 |
|---|---|
| 双方 entail → yes | 18.8%（103） |
| 双方非 entail → no | 72.4%（396） |
| 分歧 → 第三轮 | 8.2%（45） |

**第三轮结果：45 条里 76% 判 YES**；其中 **「A=entail 且 B=neutral」的 26 条，第三轮 96.2% 判 YES**。

→ **判官 B（glm-4-flash）偏保守**：它的 `neutral` 把 26 条真阳性推进第三轮（幸好第三轮救回 25/26）。
→ 净效果：现规则 ≈「任一方 `entail` **且通过对抗复核**」→ **结论可信**，但 **B 的独立信号价值低**。
→ 另：B 的 `contradict` 恒为 0（A 有 1.1%）→ B 在"说反了"这一档上**不提供信息**。

## 5. 必须剔除 6 个 facet（题数 35 → **29**）

`_r2_gold_finalize.py` 剔除"正例 = 0 或 = 全部"的题：

| 簇 | facet | 篇数 | **新真值正例** | 锚点正例 | 原因 |
|---|---|---|---|---|---|
| 1 | `deployment` | 16 | **0** | 4 | 无正例 |
| 1 | `multilingual` | 16 | **0** | 1 | 无正例 |
| 2 | `deployment` | 18 | **0** | 6 | 无正例 |
| 3 | `deployment` | 13 | **0** | 2 | 无正例 |
| 3 | `multilingual` | 13 | **0** | 2 | 无正例 |
| 3 | `zero_few_shot` | 13 | **0** | 1 | 无正例 |

→ **根因**：语料构建时用**词面锚点**筛 facet（要求正例率 15%~75%）
→ **假阳性也能"撑起"一道题**（如 `deployment` 三簇全靠 12 个假阳性入选）。
→ **这是"语料构建依赖词面锚点"的连带后果**，需在下一轮语料构建里改成"用重标后的真值筛"。

## 6. 产物

| 文件 | 内容 |
|---|---|
| `data/r2dev/gold_recalib2.csv` | 541 对：锚点 vs 新真值 + 两判官标签/置信度/quote + 第三轮 + 证据块号 |
| `data/r2dev/gold_recalib2_evidence.json` | 逐对的**证据全文**（抽检用） |
| **`data/r2dev/gold_final.csv`** | **最终真值**（集中格式：cluster, facet, docid, gold, anchor_gold）｜**458 条 / 29 题** |
| **`data/r2dev/gold_final_meta.json`** | `usable` 29 个 + `dropped` 6 个 + 逐题正例数 |
| `results/R2_GOLD_RECALIB2_SUMMARY.csv` | 逐 facet 的锚点 P/R/新增/删去 |

## 7. 边界

1. **判官不是人**：9 例抽检是**我人工判读**，第 3 例（C2P12）**边界偏严**已被我标出
2. **B 判官偏保守**且 5 条调用错误 → 依赖第三轮兜底（76% 判 YES）
3. **6 个 facet 被剔除** → 与旧的 35 题**不可直接比**（题集本身变了）
4. 只覆盖**有 PDF 的 47 篇**；`degraded` 那 1 篇（pymupdf 骨架、丢表值）仍在真值里
5. **下游接线未改**：`_r2_reader.load_gold()` 仍读 `gold_recalib.csv` → 需切到 `gold_final.csv`（待办 A1-接线）

## 8. 复现

```powershell
cd f:/paperpilot
$env:HF_HOME='F:\hf_cache'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_recalib2.py --workers 8   # 重标（~7.5 min）
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_diag.py                  # 判官诊断
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_spotcheck.py --kind removed --n 3 --evchars 380
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_finalize.py             # 出最终真值
```
