# v2 四层漏斗归档（2026-09-10 下线）

## 这是什么

PaperPilot 问答层**旧版检索架构**的完整快照：`L0 总览 → L1 claims 检索 → L2 圆心扩窗（自环）
→ L3 全局检索 → unknown`，每层配一个 judge 判"够不够"决定是否下钻。

它**已从活代码中彻底移除**，不再提供 `ask_v2` 回退；现行唯一检索链是
`src/paperpilot/graph/qa_graph_v3.py`（两级：L0 → L3 + Validator 输出闸门）。

## 为什么下线（数据，不是感觉）

| 证据 | 结论 |
|---|---|
| `qa/recall/ab_v3regress_result.json`（250 题同裁判 A/B）| v3 两级 **203** ≥ v2 四层 **201**；calls 3.7 vs 5.6（**−34%**）|
| L2 目标题群 67 题五连 A/B | L2 可赢空间**≈1 题**（comb 45 vs 直 L3 44，噪声内），却要付"圆心+半径+预算自环"的整块复杂度 |
| L1 claims 分支 A/B | V1=62 < V0=64 —— **claims 分支捞不回 L2 那 8 题**（那 8 题实为 L3 检索召回短板）|
| judge 精度 | L2 曾出现 **61%** 判够准确率 —— LLM 预测当路由**危险**，判"够"的对象必须简单 |
| 根因诊断 | 判断"够不够"的中间层收益被"证据已到、只是排位靠后"掩盖：L1 池含 gold 70%，问题在排序而非层数 |

一句话：**四层漏斗买到的准度≈0，复杂度却真实存在**；单篇场景下瓶颈在
"全局限检索召回 + 答案层"，不在"多一层判够"。

## 目录内容

| 文件 | 原位置 | 说明 |
|---|---|---|
| `qa_graph.py` | `src/paperpilot/graph/qa_graph.py` | v2 漏斗图（L0→L1→L2→L3 接线）|
| `retrieve.py` | `src/paperpilot/agents/nodes/retrieve.py` | L1 `retrieve_claims`（claim embedding top12）|
| `judge_snapshot_full.py` | `src/paperpilot/agents/nodes/judge.py` | 含 `judge_l1` / `judge_l2` / `judge_l2_strict` / `_COMPLETENESS`（L1/L2 专属，已从活文件删除）|
| `pull_chunk_snapshot_full.py` | `src/paperpilot/agents/nodes/pull_chunk.py` | 含 L2 `expand_l2`（圆心 + 半径 + 预算自环，已从活文件删除）；L3 `search_l3` 仍现役 |
| `qa_graph_v3_snapshot_full.py` | `src/paperpilot/graph/qa_graph_v3.py` | 含被证否的 V1 分支 `build_qa_graph_v3_l1`（已从活文件删除）|
| `QA_FUNNEL_DESIGN.md` | 仓库根 | v2 漏斗设计文档 |

## 可复用的教训

1. **中间层要用数据赎买**：L2 的每一分收益都必须能对着"直 L3"算出来；算不出就别加。
2. **judge 是 LLM 预测，必须配降级链**：判"不够"只是多花钱，判"够"才会答错——危险的是后者。
3. **诊断要分"没送到"和"送到了没用好"**：把失败样本的 gold 定位回检索块，才知道该修检索还是修答案。
4. **融合平坦即有害**：多路 RRF 等权融合会稀释最强那一路（同一结论在查询改写 A/B 上再次复现）。

## 相关档案

- `qa/RETRIEVAL_EXPLORATION_20260908.md`（阶段 Q：L2 存废之争全过程）
- `qa/RETRIEVAL_EVAL_FRAMEWORK.md`（C/T/E/A 评测框架）
- `docs/RAG_COMPONENT_NOTES.md`（组件默认配置与铁律）
- `DEVELOPMENT_LOG.md` 阶段 Q/R
