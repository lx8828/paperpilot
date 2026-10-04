# Task 1｜集合问答接进生产链路 ｜ Task 2｜QAMPARI 抽取粒度（负面）

日期：2026-09-29

---

# Task 1｜「哪几篇做了 X」接进生产链路（完成）

## 1. 形态：**第三种问答读取器**（沿用仓库既成的 reader 开关模式）

```bash
PAPERPILOT_QA_READER=set      # fullctx（默认）| retrieval（RAG-2）| set（集合问答）
```

| 读取器 | 做什么 | 产出 |
|---|---|---|
| `fullctx`（默认） | 语料一次塞进单次调用 | 成稿答案 |
| `retrieval` | RAG-2 挑块 → 组上下文 → 作答 | 成稿答案 |
| **`set`（新）** | **逐篇判定（map）→ 聚合** | **篇集合 + 逐篇证据** |

→ 前两者**产出一段答案文本**，**不产出"哪些篇"这个集合**，也没有篇粒度判定这一步。
→ `set` 补的正是这一环（实测：篇级集合 F1 **0.539 → 0.846**，见 `R2_READER_20260929.md`）。

## 2. 改动清单（6 个文件）

| 文件 | 改动 |
|---|---|
| **`src/paperpilot/components/set_judge.py`**（新） | 核心：`per_paper_hits` / `judge_one` / `run` / `render` + 判定提示（四纪律） |
| **`src/paperpilot/agents/nodes/set_answer.py`**（新） | 节点 `answer_set(state)`：写 `set_papers` + `cites` + `route`/`debug` |
| `agents/state.py` | **登记 `set_papers`**（⚠️ 未登记的键会被 LangGraph 静默丢弃） |
| `agents/nodes/read_full.py` | `reader()` 新增 `set`（容错 `sets`） |
| `graph/qa_graph_v3.py` | 两个 builder 都加节点 `answer_set` + 条件边 `"set": "answer_set"` + `answer_set → answer` |
| `agents/nodes/answer.py` | `_answer_from_set` 透传分支（`level="SET"`），**排在 fullctx 之前** |
| `agents/nodes/__init__.py` | 导出 `answer_set` |

## 3. 输出契约

```python
set_papers = {
  "answer": str,                      # 渲染好的答案文本
  "papers": [{"pdf","evidence":[片段号],"why","score","snippet","chunk_id","page","section"}],
  "n_papers": int, "n_yes": int, "unclear": [pdf...], "b": int, "seconds": float,
  "labels": {pdf: "yes|no|unclear"},
}
cites = [{"pdf","chunk_id","page","section","evidence"}]   # 与 fullctx 同形状，闸门可直接用
```

⚠️ **不做完备性宣称**（ASReview 的教训）：输出是「候选 + 逐篇证据 + 可提高 `PAPERPILOT_SET_B` 的提示」，
不是「这个方向我都看过了」。

## 4. 验收（真实 5 篇论文，端到端）

```
reader()='set'
① 路由分支： L0 判 够 → 'answer' ｜ L0 判 不够 → 'set'          ✅
② 逐篇判定（5 篇真论文 / 47.0s / 5 次调用 / 22,614 prompt token）
   n_papers=5 ｜ 判 yes 4 篇 ｜ unclear 0 ｜ b=12
   ✅ 2609.01316v1 证据 [1,10,12] 本文4.4节及表6明确做了组消融、留一法字段消融
   ✅ 2608.29179v1 证据 [1,10]    片段1说Ablations隔离联合前缀训练和PMI校准，片段10给消融表
   ✅ 2609.01456v1 证据 [1,9]     片段1有Ablation Analysis表，片段9有门控敏感性分析
   ✅ 2608.29290v1 证据 [6,7]     表2列出配置变体、表3比较修复率，属消融实验
③ generate_answer 透传： level=SET ｜ cites 4 条                  ✅
④ 成本：5 次调用 / 22.6k prompt token ≈ **¥0.03/题**              ✅
```

## 5. 测试

- **`tests/test_set_judge.py`（新，12 项全过）**：分组取块 / 判定解析 / 失败降级 `unclear` /
  `run` 只留 yes 且降序 / `render` **不含完备性措辞** / 节点输出契约（`set_papers`+`cites`）/
  单篇跳过 / `generate_answer` 透传 / `reader()` 开关 / 图里有 `answer_set` 节点 / 状态键已登记 /
  **提示词四纪律未丢**（防误删）
- **全量 `pytest -q` exit=0（无回归）**，lint 干净（0 诊断）

## 6. 用法

```bash
# 集合问答（走完整图，含 Router / 闸门）
PAPERPILOT_QA_READER=set uv run python cli/run_group_qa.py --group group1 --qid G1-M1-1
# 调证据块数（默认 12；实测越大 F1 越高）
PAPERPILOT_SET_B=6 PAPERPILOT_QA_READER=set ...
```

## 7. 已知边界

1. **Router 短路**：若 `judge_l0` 判「够」，会直接走 `answer`，**不到 `set` 节点**（这是既有设计）。
   → 对"哪几篇做了 X"这类**必须横向枚举**的题，L0 概览答不全；建议对这类题形**强制 `set`**
   （例如按题形前缀判断，或给 Router 加一个"集合型"意图）。
2. `set_judge` 目前**只判"篇"**，不判块级证据质量（`cite_recall/precision` 未算）。
3. 判定依赖**检索到的片段**；`PAPERPILOT_SET_B` 太小会漏（实测 b=3 F1 0.736 < b=12 的 0.846）。

---

# Task 2｜QAMPARI 抽取粒度（改提示要"最具体形式"）——**基本无效**

按上一轮的推断（`R2_QAMPARI_MAPREDUCE_20260929.md §3.5`：多列的条目多为"缺限定词"），
把 map 提示改成**要求抽取片段中最具体/最完整的写法**：

| 臂 | 提示 | `subspan_em` | `em` | `coverage` | 预测数 |
|---|---|---|---|---|---|
| `passage_union` | plain | 0.440 | 0.140 | 0.6859 | 7.02 |
| **`passage_union`** | **specific** | **0.460** | 0.100 | **0.6973** | 7.60 |
| `grp10` | plain | 0.390 | **0.250** | 0.6742 | 4.63 |
| **`grp10`** | **specific** | 0.380 | 0.230 | 0.6537 | 4.84 |

**读数**：
- `passage_union`：`subspan_em` **+0.02**、`coverage` **+0.011**（微弱）
- `grp10`：`subspan_em` **−0.01**、`coverage` **−0.021**（反降）
- **两臂 `em` 都下降**（0.140→0.100、0.250→0.230）——答案变长后**精确匹配更难过**

→ **结论：改提示粒度基本无效**，与"归并无效"同源：**gold 是数据集侧的最具体写法，
我们无法从片段里"猜出"官方粒度**。这条线到此为止，不再投入。

## 复现

```powershell
cd f:/paperpilot
$env:PYTHONIOENCODING='utf-8'
./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_mapreduce.py --k 20 --workers 10 `
    --groups 10 --arms passage_union --map-style specific --tag tier1m_mr20_spec
```

产物：`results/official_tier1m_mr20_spec/`、`R2_QAMPARI_tier1m_mr20_spec_detail.csv`
