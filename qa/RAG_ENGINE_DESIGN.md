# 检索层系统化工程：设计与尺子规格（2026-09-08 立项）

> 目标：把检索层从"散装函数 + 一次性实验"升级成**可评估、可迭代、可回归**的 RAG 检索工程。
> 核心纪律：**先立尺子，再动刀**。任何检索改动（混合/重排/query 理解/压缩/路由）必须先能在离线 Recall@k 上看到方向性变化，再谈接管线与端到端。

---

## 0. 现状结论（只读盘点，2026-09-08）

**已有**：向量 + BM25 混合（RRF）、多查询改写（默认关、曾负收益）、L3 top12、L2 确定性窗口；`judge_l3`/`search_l3` 已可脱离 graph 独立调用；评测有 QASPER harness / 三列对比 / 关键词 QA。

**真窟窿（本工程要补）**：
1. **离线检索评估缺失**：唯一的 `qa/_scratch/_bm25_ab.py` 是一次性的；README 引用"top12 ≈85%"**仓库内不可追溯**，口径还打架。
2. **检索原语不齐**：无重排、无 min-score 阈值、BM25 每查询重建、hit 的 `score` 语义不统一（hybrid 的 score 是余弦、不能当阈值）。
3. **配置散落**：`L3_TOP_K`/`AUDIT_TOP_K`/`judge max_show`/`B1_TOP_K` 手工同步，文档已漂移。
4. **judge 不可切模型**：漏斗内 judge 与主链路同模型；评测裁判才是异源。

---

## 1. 第一把尺子：离线 Recall@k（本阶段先立它）

### 1.1 为什么是它
Recall@k 直接回答"**gold evidence 能不能被检索进我们真正会喂给 judge/answer 的上下文单元**"；全离线、零 LLM、分钟级 → 是迭代护栏。

### 1.2 chunk 口径（铁律）
**复用线上 chunk，不另建**。普通 PDF = `parse_pdf → chunk_document(4000) → analyzer.extractable`；QASPER = `qasper_source.build_chunks`（chunk_id 恒定）。检索单元 = `document_cache.ordered_chunks` 同源 = 线上问答单元。**Recall@k 不评估分块质量本身**（那是另一把尺子，先不混）。

### 1.3 数据集与分层
- 数据源：QASPER train 有答案题（自带人工 gold evidence）。
- 规模：~250 题 / 每篇 ≤4，seed 固定可复现，json 固化入库。
- 分层（报告分组合计）：**normal**（普通有答案，~160）＋ **hard**（final_lock 中仍 fail/unknown 的已知硬题，~90，对应残差 1a/1c）。
- 排除：无答案题（无 evidence）。

### 1.4 gold → chunk 定位规则（确定性）
1. 规范化（空白折叠 + 剥离 BIBREF/FIGREF 占位，与 chunk 清洗一致）；
2. 整段 evidence 在单个 chunk 内 → `C_gold = {该 chunk}`；
3. 否则按句切分，收集"含任一句"的 chunk → `C_gold`；
4. 仍为空 → **mapping_fail**，不计入分母，但报告定位成功率（尺子自身健康度）。
5. **诊断列（喂给 sub-chunk 决策，不建索引）**：gold 是否跨块、命中 chunk 内的相对位置（start_frac）。

### 1.5 检索器与指标
- 列：`vec`（纯向量=search）、`hybrid`（向量+BM25 RRF=search_hybrid，系统默认）、`bm25`（纯 BM25，归因用）。将来追加：多查询、改写、rerank。
- 主指标 **Recall@k**：`|top_k ∩ C_gold|>0` 占比，k ∈ {8,12,16}；附 **MRR@16**。
- 每列对**同一篇复用一份 BM25**（eval 侧一次构建；结果与线上公式一致，仅更快）。

### 1.6 产出
`qa/recall/recall_set_v1.json`（样本固化）· `cli/run_retrieval_eval.py` · `qa/recall/RECALL_BASELINE_*.md`（基准快照，含分组合计 + mapping 成功率 + 诊断列）。

---

## 2. 延迟决策观察项（记着，不做）

### sub_chunk 两层粒度 + 二级 id
- **现状不做**：单篇 ~100–200 块、L3 top12 块级 recall 已 ~85–96%，瓶颈不在"块太大要再切细"，而在表格/图未进块、块命中后未读对。两层 id 会扩散到 state/缓存/gold 映射，属提前付债。
- **触发信号（未来该做的依据）**：
  1. 块级 Recall@k 已高、端到端 1a 残差仍高 → 料送到没读对，才可能需 sub-chunk；
  2. 引入多文档/论文库检索（候选 >10³）→ 需段落级返回单元；
  3. gold 句子级定位诊断需要二级 id。
- **种数据**：Recall@k 的"跨块/块内位置"诊断列即为此攒样本，不另建索引。

---

## 3. 后续工程蓝图（按序，每步都过第 1 把尺子）

1. **地基**：检索统一 hit schema、预算/阈值单点配置、BM25 缓存、judge 提纯为可注入 prefix 的服务、L3 consult 抽共享服务（graph/审计/B1 三处合一）。
2. **评估扩展**：层间送达审计（1a 纯召回 / 1c 判太严 口径量化）、端到端回归沿用 run_compare + QASPER 子集。
3. **增量实验**：query 理解/改写（旧负收益在新尺子上重验）、rerank（先离线看 hit@k 增益，16G 选小体量模型）、上下文压缩（只许"给 judge/answer 的摘要精简"，**禁证据截断**，配忠实度校验）、min-score/路由（做成取料/预算参数，**不重蹈旧 router 决策债**，见 QA_FUNNEL_DESIGN §9）。

## 4. 铁律
- **评测先于改动**；改动后重跑同一样本 diff。
- 一切数字须**可复现**（样本 json + seed + 报告落盘），不再有口头 85%。
- 不破坏：`parse_pdf → blocks`、chunk 文本不二次截断、证据溯源闭环。
