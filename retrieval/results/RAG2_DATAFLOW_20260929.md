# RAG-2 实现整理：数据流 · 操作步骤 · 切块口径

日期：2026-09-29 ｜ 依据：逐文件读源码（非记忆）
核实脚本：`retrieval/tmp/_rag2_corpus_stat.py`（语料/切块规模实测）

---

## 0. 结论先行 —— **两种切块，各有其归属**

### 0.1 章节段落切块（`chunk_document`）用在哪 —— **生产全链路，无一处例外**

`document_cache.retrieval_chunks` 的实现（`tools/chunker.py::chunk_document`）：
```
find_headings(blocks)      → 标题树（章节切块）
_assemble(ordered, headings) → 按标题聚块
若 块长 > max_len(4000) → _split_by_paragraph()  → 段落为原子单元累加切分（段落切块）
子块带 part（"1/3"）并共享父 title_path
```

**调用方清单（`grep 'ChunkIndex\(|MultiChunkIndex\(' src/paperpilot/`，15 处）**：

| 模块 | 用途 |
|---|---|
| `agents/nodes/pull_chunk.py:216` | **RAG-2（`search_l3`）** 块级检索 |
| `agents/nodes/read_full.py` → `components/fullctx.py:58,82` | **全文直读**（逐个 chunk 编号 `[P·§·¶]`） |
| `agents/nodes/set_answer.py:54` → `components/set_judge.py` | **集合问答**（逐篇判定） |
| `agents/nodes/answer.py:495` | L3 作答的组上下文 |
| `agents/nodes/report.py` | 报告链 |
| `components/repairer.py:122` | CRAG 修复（独立重检索） |
| `components/retriever.py:24,41` | 单篇 chunk / claim 检索 |
| `workflow.py:263` / `worker.py:173` | 预建索引 |

→ **生产问答（`fullctx` / RAG-2 / `set`）、报告链、修复链，全部走章节段落切块**（≤4000 字符 + MinerU 表格/公式注入）。**没有用固定窗。**

### 0.2 固定窗（`chunks_of` 1000 字符 / 步长 900）用在哪 —— **只有 `retrieval/tmp/` 实验脚本**

`grep -rn chunks_of` 命中 **14 个文件，全在 `retrieval/tmp/`**，`src/` 里**零命中**：
`_r2_std_metrics.py`（定义处）/ `_r2_probe.py` / `_r2_score.py` / `_r2_score2.py` / `_r2_setsel.py` /
`_r2_phase1.py` / `_r2_phase1_clean.py` / `_r2_reader.py` / `_r2_fullread.py` /
`_r2_gold_metrics.py` / `_r2_gold_recalib.py` / `_r2_gold_dispute.py` / `_r2_calib2.py` /
`_r2_l3_nli.py` / `_nli_route.py` / `_leak_audit.py` / `_two_track_summary.py`

⚠️ **一个关键澄清**：这批脚本自称「**新 RAG-2 · 集合型检索**」（`_r2_build_corpus.py:1`、
`_r2_probe.py:1`），**不是 RAG-1**。
- **RAG-1** = **论文级排序检索**（LitSearch：query → top-k 篇），`_r2_probe.py:10` 明写
  「臂 A = 全局 chunk top-k → **RAG-1 式**」；其自身切块/向量来自 LitSearch 预计算产物，
  代码在 `retrieval/scripts/` + `tools/corpus_search.py`（`max_seq_length=512`、CE 512）。
- **固定窗 `chunks_of`** = **"新 RAG-2 集合型检索"的实验脚手架** —— 为了在 **LitSearch 纯文本**
  上快速迭代而写，**从未进 `src/`**。

### 0.3 所以两边到底差什么

| 维度 | **实验脚手架**（`retrieval/tmp/_r2_*.py`，自称"新 RAG-2 集合型检索"） | **生产链路**（`src/paperpilot/`，`set` 节点 = 同一套提示的生产版） |
|---|---|---|
| 语料 | **LitSearch `corpus_clean.full_paper`**（外部数据集全文） | **`assets/papers/*.pdf`**（用户给的 5 篇） |
| **切块** | **`chunks_of()` 1000 字符 / 步长 900**（纯字符串窗口，无结构） | **`retrieval_chunks()`：章节段落切块（≤4000）+ MinerU 注入** |
| `max_seq_length` | **512**（截断） | **8192**（默认，`embedder.py` 明写"不要改"） |
| 词法 | 自带 `BM25`（仅对照臂，**未融合**） | `BM25Index` + **`rrf_order` RRF 融合** |
| 查询 | `[中文题面] + LLM 子查询`（按 facet 预生成） | `search_hybrid(单条自然问句)` |
| 判定 | `_r2_reader.py` 自带提示 | `set_judge`（同一套四纪律 + `extract`） |
| 语料 | **LitSearch `corpus_clean.full_paper`**（外部数据集全文） | **`assets/papers/*.pdf`**（用户给的 5 篇） |
| **切块** | **`chunks_of()`：朴素固定窗 1000 字符 / 步长 900**，切的是**纯文本字符串** | **`retrieval_chunks()`：我们的语义切块**（标题树 `chunk_document(max_len=4000)` → extractable）**+ MinerU 表格/公式注入** |
| 块/篇 | 54~64 块（每块 1000 字符） | **24 块**（均值块长 1488~1882 字符） |
| 向量 | bge-m3 **现场 encode**，无缓存 | 各篇 **`.cvec.npy` 缓存**（`assets/artifacts/out_views/`） |
| 词法 | 自带 `BM25`（**仅作对照臂，未融合**） | `BM25Index` + **`rrf_order` RRF 融合**（dense+BM25） |
| 查询 | `[中文题面] + LLM 子查询`（`subqueries.json`，**按 facet 预生成**） | `search_hybrid(单条自然问句)` |
| 判定 | `_r2_reader.py` 自带提示 | `set_judge`（同一套四纪律，**+ `extract`**） |
| 产物 | `R2_*.csv` 实验台账 | `set_papers` / `cites` 状态字段 |

→ **结论：实验脚手架的全部数字（篇级 F1 0.846 / 0.817 / 0.539 …）都建立在"1000 字符固定窗 + LitSearch 纯文本 + 512 截断"上；而生产 `set` 节点用的是"章节段落切块 + 我们的 PDF + 8192 不截断"。搬过去时切块整条换了，这一步没有被验证过。** 这是当前最大的方法学缺口（§4-①）。

**注意区分两件事**（别混）：
- ✅ **"我们问答切块用章节段落切块"** —— **成立**，生产全链路都是（§0.1）。
- ⚠️ **"所以实验脚手架的数字就能代表生产问答"** —— **不成立**，因为脚手架**没走**章节段落切块（§0.2）。

---

## 1. 实验脚手架的数据流与操作步骤（`retrieval/tmp/_r2_*.py`）

语料：**LitSearch 全文**（不是我们的 PDF）｜ 切块：**固定窗 1000/900**（**非生产切块**）

```
[0] 语料构建      _r2_score.py :: build_or_load()          → clusters/c{0,1,2}.parquet + meta.json
[1] 词面真值      _r2_std_metrics.py :: judge(chunk, facet)（facet 正则命中）
[2] 真值校准      _r2_gold_recalib.py / _r2_gold_dispute.py → gold_recalib{,_rich}.csv
[3] 切块          _r2_std_metrics.py :: chunks_of()        → 1000 字符 / 步长 900
[4] 索引          enc.encode(chunks)（bge-m3, normalize, max_seq_length=512）
[5] 查询构造      _r2_phase1.py :: get_subqueries()        → subqueries.json（LLM，输入只有中文题面）
[6] 打分          S = Q @ C.T；篇级分 = 篇内 max
[7] 排序臂        A base / B mq_max / C mq_rr(AMER 轮询) / D,E MMR(λ 扫描)
[8] 评测（3 层）  _r2_std_metrics.py → R2_STD_L1/L2/L3；干净版 _r2_phase1_clean.py
[9] 判定(reader)  _r2_reader.py：每篇取自己 top-b 块 → 1 次 LLM → yes/no/unclear → 篇集合
[10] 整档对照     _r2_fullread.py（2026-09-29 新增，20 篇全文一次塞）
```

### 逐步说明

**[0] 语料构建** `_r2_score.py::build_or_load`
- 用 LitSearch **已算好的向量**（`data/litsearch/derived/emb/part_*.npy` + `corpusid.npy`）
- 3 个主题串 → dense top-240 候选 → 只保留"正例率 15%~75%"的 facet（有分辨率）
- 每簇按 `Σ p(1−p)` **选 20 篇**（让 facet 在这 20 篇里有分歧）
- 落 `{docid, corpusid, title, full_paper, n_chars}`；`meta.json = [{topic, usable}]`
- 注：早期单簇版是 `_r2_build_corpus.py` → `corpus20.parquet` + `facet_matrix.json`

**[1] 词面真值** `judge(chunk, facet)` = `re.search(F[facet][0], chunk, re.I)`
- 篇级 gold = **该篇任一块命中正则** → 这是 ALCE NLI 判定器的**本地替代**（明示为检索侧代理）
- `F` = 20 个 facet × (锚点正则, 中文题面, 英文锚点词, 英文改写句)

**[2] 真值校准**（词面真值 ≠ 可用真值）
- `_r2_gold_recalib.py` → `gold_recalib.csv`；`_r2_gold_dispute.py` → `gold_recalib_rich.csv`（富证据复判）
- `_r2_reader.load_gold()` 合并为 `final_gold`：**锚点 7.6 → 5.0 篇/题**，可用题 30~35

**[3] 切块** `chunks_of(t) = [t[i:i+1000] for i in range(0, len(t), 900)]`
- ⚠️ **纯字符串窗口**：不管段落、标题、表格 → 与生产切块无任何共同点

**[4] 索引** 每簇一次 `enc.encode(chunks)`（bge-m3，`batch_size=32`，`normalize`）
- 无落盘缓存 → 每次跑都要重编码（3 簇耗时约 1~3 分钟）

**[5] 查询构造** `get_subqueries()`：LLM 输入 **只有 `论断：{中文题面}`**
- 产出按 facet 缓存的 `subqueries.json`（**固定条数，所有簇共用同一组**）
- 泄露审计后确定：**只有 `[zh] + subqueries` 是干净的**；`[zh+anchor]` 有 63% 词面重合

**[6] 打分** `S = Q @ C.T`（Q = 查询集向量）；**篇级分 = 该篇所有块的 max**
- 这是"多值检索"的关键：**篇级分由最相关的那一块决定**，不是平均

**[7] 排序臂**（`_r2_phase1_clean.py`）
- `A base`：单查询（中文题面）
- `B mq_max`：多查询 → 逐块取**跨查询 max**
- `C mq_rr`：多查询 → **round-robin 交错**（AMER 轮询）
- `D/E MMR`：λ ∈ {1.0, 0.7, 0.5, 0.3} 覆盖导向重排（pool=40）
- 对照：`random`（200 次排列均值）、`oracle`（用 gold，仅上界）

**[8] 评测（3 层）** `_r2_std_metrics.py`
- L1 篇级交付 k 篇：`MRecall@k` / `StRecall@k` / 集合 P / 集合 F1 / `α-nDCG`（k=5/10/20）
- L2 等块预算 B：`global_topB` vs `round_robin`（b=1/3/6/12 → B=b×20），含**证据块召回**
- L3 ALCE 式引用指标（`cite_recall` / `cite_precision`，`b`=每篇块数）
- ⚠️ `k=20` **无信息量**（送全部 20 篇 → 随机 StRecall 也 1.000）

**[9] 判定（reader）** `_r2_reader.py`
- 检索：`S = enc.encode([zh] + subqueries) @ C.T`，`Smax = S.max(axis=0)`
- **每篇取自己 top-b 块**（b=3/6/12）→ **1 次 LLM 调用/篇**（并发 8）
- 提示四纪律：只看片段 / 区分"本文做的 vs 引用他人做的" / 列出≠做了 / 中英同义改写
- 输出 `{label: yes|no|unclear, evidence:[片段号], why}` → 判 yes 的篇 = 交付集合
- 指标：篇级**集合 P/R/F1**，并与"检索排序 top-k"基线对照

**[10] 整档直读对照** `_r2_fullread.py`（新增）
- 20 篇全文一次塞（266k token），要求逐篇结论 + 末行 `结论篇：P3,P7`
- 记 `n_touched`（答案里提到 `【Pn】` 的篇数）= 长上下文注意力代理

---

## 2. 生产链路的数据流与操作步骤（`src/paperpilot/`）

语料：**用户给的 5 篇 PDF**（`assets/papers/`）｜ 切块：**章节段落切块**（`retrieval_chunks`，§0.1）

```
[P1] 路由        graph/qa_graph_v3.py 条件边 {"set": "answer_set"}（由 nodes/read_full.reader() 决定）
[P2] 建库        MultiChunkIndex(pdfs)  → 各篇 retrieval_chunks 拼扁平空间 + _owner
[P3] 向量        vectors() = vstack(ChunkIndex(p).vectors())  ← assets/artifacts/out_views/<stem>.cvec.npy
[P4] 混合检索    search_hybrid(question, top_k=n_all)
                 = encode_query → cosine  ⊕  BM25Index.score → rrf_order(k=60)
[P5] 逐篇取块    set_judge.per_paper_hits：沿全局 RRF 序走，每篇保留前 b 块（b 默认 12）
[P6] 逐篇判定    set_judge.judge_one × N 篇（并发 workers 默认 6，1 次 LLM/篇）
[P7] 聚合        label=="yes" → papers（按检索 score 降序）
[P8] 渲染        render() → 答案文本；cites → validator.gate
[P9] 透传        answer.generate_answer 见 set_papers → _answer_from_set（level="SET"）
```

### 逐步说明

**[P1] 路由** `graph/qa_graph_v3.py:80,88`
- 开关 `PAPERPILOT_QA_READER = fullctx（默认）| retrieval（RAG-2）| set（集合问答）`
- `reader()` 在 `agents/nodes/read_full.py`；`answer_set` 节点已登记在两个 builder
- ⚠️ **已知短板**：`judge_l0` 判"够"时会直接走 `answer`，**不到 `set` 节点**

**[P2] 建库** `MultiChunkIndex(pdfs)`
- `_doc_chunks()` = 各篇 `retrieval_chunks(pdf)` 顺序拼接（扁平 chunk 空间）+ `_owner`（下标→来源篇）
- `retrieval_chunks` 链路（**这就是我们的切块逻辑**）：
  ```
  assets/papers/<stem>.pdf
    → MinerU 产物（默认，PAPERPILOT_USE_MINERU=1）/ pymupdf（容灾）
    → chunk_document(blocks, max_len=MAX_CHUNK_LEN=4000)   ← 标题树 + 段落原子切分
    → analyzer.extractable()
    → ordered_chunks(pdf)                                   ← 基础块
    → + MinerU 表格/公式注入（按页匹配）                      ← retrieval_chunks = 检索视图
  ```

**[P3] 向量** `MultiChunkIndex.vectors()` = 各篇 `ChunkIndex(p).vectors()` 的 vstack
- 各篇向量缓存在 `assets/artifacts/out_views/<stem>.cvec.npy`（+ `.cidx.json` 记 chunk_id 顺序）
- 失效条件：`chunk_id` 列表与索引时不一致 → 重建；**多篇不落新缓存**

**[P4] 混合检索** `ChunkIndex.search_hybrid`
- 向量：`encode_query(query)`（bge 指令前缀）→ `q @ vecs.T`
- 词法：`BM25Index([c.text]).score(query)`
- **闸门** `_bm_or_none`：BM25 全 0（纯中文查询占 13%）→ **返回 None，只用向量路**
  （否则 `argsort` 对全 0 数组返回索引序 → 拼接序第一篇被系统性加分）
- **融合** `rrf_order(vec, bm, k=60, w_vec, w_bm)`：`score = Σ weight/(k+rank)`
  - `_ext_weights(chunks, α)`：**外部块（MinerU `xtbl-*`）权重 (2α, 2(1−α))，文本块恒 (1,1)**
  - α 由 env `PAPERPILOT_EXT_RRF_ALPHA` 控制，**默认 0.5 = 生产原样（不构建掩码）**

**[P5] 逐篇取块** `set_judge.per_paper_hits(idx, question, b)`
- 用 `search_hybrid(question, top_k=n_all)` 拿**全局序**再分组 → 每篇的"前 b 块"是**同一把尺子**下的
- 只编码一次查询；b 由 env `PAPERPILOT_SET_B`（默认 `DEFAULT_B=12`）

**[P6] 逐篇判定** `set_judge.judge_one(claim, hits, extract=)`
- 提示二选一：
  - `SYS_SET`（默认）：只判 `label`
  - `SYS_SET_EXTRACT`（env `PAPERPILOT_SET_EXTRACT=1`）：**+ `extract` = 该篇在本题上的具体内容**
- 四纪律：只看片段 / 区分"本文做的 vs 引用他人做的" / 列出≠做了 / 中英同义改写
- 输出 `{label: yes|no|unclear, extract, evidence:[片段号], why}`；调用失败 → 降级 `unclear`

**[P7] 聚合** `run()`：只留 `label=="yes"`，按检索 `score` 降序

**[P8] 渲染** `render(question, res)` → 答案文本；格式 `{i}. <pdf>：<why>；<extract>（证据：片段 …）`
- ⚠️ **不做完备性宣称**（ASReview 教训）：输出是"候选 + 逐篇证据 + 可提高 `PAPERPILOT_SET_B` 的提示"
- `cites` 形状与 `fullctx` 一致 → `validator.gate` 可直接用

**[P9] 透传** `agents/nodes/answer.py::_answer_from_set`
- `generate_answer` 见到 `set_papers` 即原样返回（**排在 fullctx 分支之前**）

---

## 3. 规模实测（`_rag2_corpus_stat.py`）

### 开发线（LitSearch 全文 · 朴素窗）

| 簇 | 篇数 | 字符/篇(中位) | 总字符 | ≈token | 块数 | 块/篇 |
|---|---|---|---|---|---|---|
| 1 | 20 | 53,740 | 1,096,058 | 221,874 | 1,228 | 61.4 |
| 2 | 20 | 56,527 | 1,136,716 | 230,104 | 1,270 | 63.5 |
| 3 | 20 | 47,598 | 964,301 | 195,202 | 1,081 | 54.0 |
| **合计** | 60 | — | **3,197,075** | **647,181** | **3,579** | — |

### 生产线（我们的 PDF · 语义切块）

| 组 | 篇数 | 基础块 | 检索块 | **注入块** | 字符(检索视图) | 均值块长 |
|---|---|---|---|---|---|---|
| group1 | 5 | 121 | 121 | **0** | 180,067 | 1,488 |
| group2 | 5 | 113 | 113 | **0** | 190,438 | 1,685 |
| group3 | 5 | 103 | 103 | **0** | 186,980 | 1,815 |
| group4 | 5 | 102 | 102 | **0** | 192,011 | 1,882 |
| group5 | 5 | 100 | 100 | **0** | 170,128 | 1,701 |

**读出的两件事**：
1. **`注入块 = 0`** —— MinerU 表格/公式注入在这 5 组上**没有生效** → 生产线切块**等于基础块**
   → 凡答案出自表格的题会退化（这正是"降级"要标记的原因）
2. 生产 5 篇实测 **170k~192k 字符 ≈ 34k~39k token**
   → `nodes/read_full.py` docstring 写的"≈48k token"是**早期估计**，实测偏小约 20%

---

## 4. 缺口清单（按风险排序）

| # | 缺口 | 后果 |
|---|---|---|
| **①** | **切块口径不一致**：脚手架固定窗（1000/900）vs 生产章节段落切块（≤4000 + 注入） | 脚手架的 F1 0.846/0.817 **不能直接搬到生产 `set` 节点**；块粒度差 2.5 倍（54~64 vs 24 块/篇） |
| **①b** | **`max_seq_length` 不一致**：脚手架 **512**（截断）vs 生产 **8192**（不截断，`embedder.py` 明写"不要改"） | **移植硬坑**：512 对脚手架的 1000 字符块无害（≈200 token），但生产块最长 **2806 token** → 512 会砍掉 82%，正是仓库记录的"Run1 真漏检根因"。**换切块必须同时确认 `max_seq_length`** |
| ② | 开发线是 **dense-only**，生产线是 **RRF（dense+BM25）** | "多查询/轮询/MMR"的结论（含 **MMR 显著有害 −0.059**）**未在 RRF 下复验** |
| ③ | 开发线语料是**外部数据集纯文本**，生产线是**我们解析的 PDF** | 生产线多了页眉/页脚/参考文献/表格碎裂等噪声；开发线没有 |
| ④ | 生产 5 篇**无篇级真值** | `set` 的判定质量在生产题形上**不可测**（只有 M1 的 `must_all` 锚点间接度） |
| ⑤ | **注入块 = 0** | 表格类问题结构性退化（未量化） |
| ⑥ | 开发线查询是**按 facet 预生成**的固定子查询；生产线是单条自然问句 | 多查询增益（+0.043 不显著）在生产形态下**不存在** |
| ⑦ | M1 是**复合题**（筛选 ∧ 内容枚举），`set` 只答筛选 | 补 `extract` 后 14% → **34%**，仍 < `fullctx` 60%（见 `R2_SC_VERDICT_20260929.md` §3.3） |

---

## 5. 复现命令

```powershell
cd f:/paperpilot
$env:HF_HOME='F:\hf_cache'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'

# 规模实测（本文 §3 的数字）
./.venv/Scripts/python.exe -u retrieval/tmp/_rag2_corpus_stat.py

# ── 开发线 ──
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_std_metrics.py     # L1/L2/L3
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_phase1_clean.py    # 多查询/轮询/MMR（干净口径）
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_reader.py          # 逐篇判定
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fullread.py        # 整档直读对照

# ── 生产线（5 篇 PDF）──
PAPERPILOT_QA_READER=set PAPERPILOT_SET_B=12 python cli/run_group_qa.py --group group1
PAPERPILOT_QA_READER=set PAPERPILOT_SET_EXTRACT=1 ...                # 复合题开 extract
```

**关键源码**：`src/paperpilot/components/set_judge.py`（判定核心）、
`src/paperpilot/agents/nodes/set_answer.py`（节点）、`src/paperpilot/agents/embedder.py`
（`ChunkIndex` / `MultiChunkIndex` / `rrf_order`）、`src/paperpilot/agents/document_cache.py`
（`ordered_chunks` / `retrieval_chunks` / `MAX_CHUNK_LEN=4000`）。

---

## 6. 一句话现状

**切块只有一种进了生产**：章节段落切块（`chunk_document` ≤4000 + MinerU 注入）—— `fullctx` / RAG-2(`search_l3`) / `set` / 报告链 / 修复链**全部**用它，**没有例外**。

**固定窗（1000/900）只活在 `retrieval/tmp/` 的实验脚手架里**（`_r2_*.py` 等 14 个文件，`src/` 零命中），那批脚本自称"新 RAG-2 · 集合型检索"——**它是"新 RAG-2"的实验版，不是 RAG-1，也不是生产切块**。

→ 所以：**"我们问答用章节段落切块"是成立的**；但**"脚手架的数字代表生产问答"不成立**，因为脚手架根本没走章节段落切块。
→ 要打通：把脚手架**换成 `retrieval_chunks`（在 PDF 上跑）+ `max_seq_length=8192`** 再复跑一次（§4-①/①b）。
