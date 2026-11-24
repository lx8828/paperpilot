# 目录地图 · 按业务线画链路

> **建立**：2026-10-13 ｜ **怎么用**：先看下面六条线里哪条是你的，再顺着**编号**走一遍链路 —— 每一步都写了
> **做什么 / 哪个文件 / 什么产物**。只想定位单个文件时，直接搜本文档。
> 结论与数字**一律回原报告引用**，本文件只写"链路 + 落点 + 依据"。

**六条线**：一、论文检索 ｜ 二、论文取料 ｜ 三、论文问答直读 ｜ 四、论文问答（下线的 RAG） ｜ 五、多答案开放域问答 ｜ 六、不进这五条的

---

# 一、论文检索（RAG1）

> 一句话：**给一个问题 → 在 54 万篇 arXiv 里找出最该读的 top-5**。全程**篇级**，不切块。

| # | 做什么 | 实现 / 入口 | 产物与关键参数 |
|---|---|---|---|
| 1 | **抓元数据**（含 abstract） | [`retrieval/scripts/fetch_arxiv.py`](../retrieval/scripts/fetch_arxiv.py)（OAI-PMH，`set=cs:cs`） | `retrieval/data/arxiv/raw/<YYYY-MM>.xml.gz`；**按月切片**、断点续抓；★ 按 `<created>` 过滤（OAI datestamp 是**更新日**，会把老论文的更新混进来） |
| 2 | **建语料表** | 同上汇总 | `retrieval/data/arxiv/meta/corpus_text.parquet`（docid/title/abstract/text/n_chars/arxiv_id/created/categories） |
| 3 | **语料粒度**（★ **不切块**） | — | **篇级 `title + abstract`**：中位 **1,374** / 均值 **1,370** 字符（CV `0.25`、p99 **2,008**、min 59 / max 4,114）；实测 **540,052 篇、`created` 2023-01-01 ~ 2026-09-17、100% 含 `cs.`**（`cs.LG 167,536` / `cs.AI 147,673` / `cs.CV 122,675` / cs.CL 仅 80,193） |
| 4 | **编码 + 建索引**（★ **没有向量数据库**） | [`retrieval/scripts/build_arxiv_index.py`](../retrieval/scripts/build_arxiv_index.py)（arXiv 侧）、[`encode_corpus.py`](../retrieval/scripts/encode_corpus.py)（LitSearch 侧） | 向量 `bge-m3` / max_len 512 / **归一化**（点积=cosine）/ fp16 / 分片落盘 → `data/arxiv/emb/part_XXXXX.npy` + `docid.npy`；BM25 = **直接 import** `eval_retrieval.SparseBM25`（scipy 稀疏）。★ 索引本体就是**一个 `(540052,1024)` fp32 矩阵（2.21 GB）**，一次 matmul 算完全部相似度 |
| 5 | **检索五步**（线上） | [`src/paperpilot/tools/corpus_search.py`](../src/paperpilot/tools/corpus_search.py) | ① **打分** dense + BM25 ② **融合** RRF，α=0.5，各组件先取 `DEPTH=1000` ③ **粗排** `bge-reranker-v2-m3`(pointwise) `POOL=200 → K_CE=50` ④ **精排** `deepseek-chat` listwise，**只输出前 10** ⑤ **交付 top-5**。常驻：稠密 **2.21 GB(RAM)** · BM25 ~0.6 GB · 两模型 ~2.4 GB(显存) |
| 6 | **入口** | [`cli/run_search.py`](../cli/run_search.py) ｜ Web `POST /api/direction/search` | 线上**工具①** |
| 7 | **评测（L1）** | [`cli/eval/run_retrieval_eval.py`](../cli/eval/run_retrieval_eval.py) ← [`retrieval/scripts/eval_retrieval.py`](../retrieval/scripts/eval_retrieval.py) | LitSearch **64,183 篇**（本地实跑 63,269）/ **597 英文查询**；链（R@1）：**BM25 0.2588 → hyb(α=0.5) 0.3548 → +CE 0.4154 → +listwise 0.6092** |
| 8 | **评测（线上口径）** | [`corpus_search.py`](../src/paperpilot/tools/corpus_search.py) docstring | arXiv-only 200 题：R@1 0.875 / R@5 0.885 ★ **该题集已判"饱和无分辨率"，不可当效果证据** |

**报告落点**：[`LITSEARCH_METRICS_20260927.md`](../retrieval/results/LITSEARCH_METRICS_20260927.md)（指标 / 消融）、[`LITSEARCH_SIGNIFICANCE.md`](../retrieval/results/LITSEARCH_SIGNIFICANCE.md)（显著性）、[`LITSEARCH_DELIVER_N_20260927.md`](../retrieval/results/LITSEARCH_DELIVER_N_20260927.md)（交付数与池深）—— 一组共 **24 份**，都在 `retrieval/results/` 下（前缀 `LITSEARCH_`）。

> **为什么这样选（依据）**
> - **语料不切块（篇级 `title+abstract`）** —— 判据有两条：① **`title + abstract` 本身就是天然的切块** —— 论文自己已经切好了（一句标题 + 一段摘要），是**语义自足**的单元，不需要我们再切一遍（对照 §五 MultiAnswer：拿到的是**全文**，那才必须切块）；② **实测长度收敛、无长尾 → 切块没有收益空间** —— 中位 **1,374** / 均值 **1,370** 字符、**CV 0.25**、p99 **2,008**、**>3,000 只 19 篇**、max 4,114，且**编码 512 token 下只截 1.0%**（见 [`R2_TRUNC_LAYERS_20261013.md`](../retrieval/results/R2_TRUNC_LAYERS_20261013.md)）。
> - **这一层为什么必须检索（`S / C` 判据）** —— **`S / C` 不等式**：单篇正文中位 **6,244 token ≤ 上下文预算** → **单篇直读即可**；而本层 6.4 万篇 ≈ **4 亿 token，任何窗口都放不下** → 检索是物理必需（[`docs/RETRIEVAL_LOG.md`](RETRIEVAL_LOG.md)）。★ 它是「**要不要检索 / 要不要全文直读**」的尺子（§三 的「为什么不用检索」用的是**同一条**；S/C 阶梯见 [`R2_SC_VERDICT_20260929.md`](../retrieval/results/R2_SC_VERDICT_20260929.md)），**与「要不要切块」是两件事**。
> - ★ **不用向量数据库、也不用 ANN（精确暴力扫）** —— 理由写在 [`corpus_search.py:24-29`](../src/paperpilot/tools/corpus_search.py)，共三条：① **评测要"精确"召回**（池内丢一篇，后面的 CE/LLM 位次全受影响 → **指标会被污染且难以归因**；上近似 = 把评测的尺子做成可变的）；② **这个规模下 ANN 买不到什么** —— 一记矩阵乘实测 **221 ms p50**，而真正的瓶颈是 **CE 重排 2,879 ms（占总计 94%）** → **优化 dense 扫的收益上限 <7%**（[`R2_RETR_LATENCY_540K_20261013.md`](../retrieval/results/R2_RETR_LATENCY_540K_20261013.md)）；③ **向量库的核心价值我们一条都用不上** —— 语料**离线建、只读、全量重载**（无增量 upsert），过滤只要**时间窗**（**前置掩码**一次整数比较搞定，[`corpus_search.py:197-224`](../src/paperpilot/tools/corpus_search.py)），单机单租户、无水平扩展需求。于是索引本体就是**一个本地矩阵 + scipy 稀疏 BM25**，`np.concatenate` 全量常驻（**非 memmap、无外部服务**），首次加载 **28.1 s**、之后常驻。★ **上线条件也预写好了**：多并发要上 HNSW 时，**上线前必须先验证 `ANN top-100 vs 精确 top-100` 重合率 ≥ 99.5%**。检索本体**零 token**（bge-m3 与 CE 都是本地模型），整次成本只有 LLM 精排 1 次 ≈ **¥0.01**。
> - ★ **三层截断各吃掉多少：①编码 1.0% ｜ ②CE 4.0% ｜ ③LLM 23.8%**（2026-10-13 实测；结论是**三层全维持现状**）—— 上限依次是 编码 `max_seq_length=512`(token) / CE `DOC_ABS_CHARS=1500`(字符) / LLM `LLM_ABS_CHARS=1000`(字符)（[`corpus_search.py:71-73,142`](../src/paperpilot/tools/corpus_search.py)），实测被截篇数 **396(1.0%) / 148,599(27.5%) / 432,665(80.1%)**。**①基本可划掉**（真 tokenizer 下字符/token ≈ **4.31**，摘要中位 **316 token ≪ 512**）；⚠️ [`encode_corpus.py:95`](../retrieval/scripts/encode_corpus.py) 自报的「> 1,638 字符可能被截」按 3.2 字符/token 粗估，**把 1.0% 说成 22.8%，别引用那行**。**③是唯一值得动的**（**80%** 候选在"决定最终 top-10"那步只看摘要前 1,000 字符、均丢 **407** 字符，且 **30.4%** 被丢尾段含结论句），**但先不改**：要证明收益得有**有分辨率**的题集，而线上 arXiv200 已判「饱和无分辨率」→ **没有尺子，改了也测不出来**。全文见 [`R2_TRUNC_LAYERS_20261013.md`](../retrieval/results/R2_TRUNC_LAYERS_20261013.md)。
> - **融合用 RRF、α=0.5** —— ★ **扫过，不是默认值**：[`LITSEARCH_METRICS_20260927.md`](../retrieval/results/LITSEARCH_METRICS_20260927.md) §1 有 α=0.3/0.5/0.7/0.9 四档表。选 0.5 的真正理由是 **`α` 与候选深度耦合** —— **`k≤50 → hyb0.7 最优；k≥100 → hyb0.5 最优`**，而下游 CE 是在 100 深度上取的；且 `hyb0.7 vs hyb0.5` **不显著**（CI 跨 0，p≈0.24，[`LITSEARCH_SIGNIFICANCE.md`](../retrieval/results/LITSEARCH_SIGNIFICANCE.md)）。
> - **组件深度 `DEPTH=1000`** —— 加深**召回**侧是"白捡的召回"：top-100 之外还压着 **10.3pt** gold（池上界 **83.5% → 93.8%**），而这一步**零 LLM**、成本只是矩阵运算。
> - **粗排池深 `POOL=200` / `K_CE=50`** —— ★ **测出来的**：「**扩池对 pointwise 精排零收益**（K≥200 时 R@1 完全相同 **0.4137**，成本却 ×10）」（[`LITSEARCH_METRICS_20260927.md`](../retrieval/results/LITSEARCH_METRICS_20260927.md) §2）。★ `POOL` **就是**那次实验里的 `K_cand`（**同一个旋钮** = 进 CE 的候选数）；`K_CE` 是**另一个旋钮**（CE 输出给 LLM 的篇数，50 已在收益拐点）。实测对照（[`docs/RETRIEVAL_LOG.md`](RETRIEVAL_LOG.md):446-452）：`K_cand=100` → R@1 **0.4154** / R@5 **0.6168** / 成本 **6.0 万对**；`=200` → **0.4137** / **0.6135** / **11.9 万对** —— 即 **100 略优且成本减半**，而**线上取 200 是 2026-09-20 的定稿值、未被这次实验回改**。⚠️ 另注意「**池深**」在本仓指两个量：本条的"池深"= CE **输入**；`LITSEARCH_DELIVER_N` 里的"池深 K"其实是 **`K_CE`**（CE **输出**）。
> - **精排只输出前 10、交付 top-5** —— 代价**全在输出（decode）长度**，而交付只要 top-5 → 于是「**把池子深度与输出长度解耦**」：只让 LLM 给前 10 名。实测 `k50p10`（池 50、只输出 10）**813 ms ≈ K=20 的 825 ms**，R@5 反而更高。线上四个常量（`POOL 200 / K_CE 50 / N_OUT 10 / 交付 5`）经复核**"不用改"**。
> - **线上题集"饱和判死"** —— [`LITSEARCH_DELIVER_N_20260927.md`](../retrieval/results/LITSEARCH_DELIVER_N_20260927.md) §6：200 题是"**从论文反写 query**"（gold = 那 1 篇）→ CE 交付 5 篇就 **99.5%**，**饱和无分辨率**。
> - **论文对标口径** —— [`retrieval/scripts/paper_compare.py`](../retrieval/scripts/paper_compare.py)（换算成论文的 Avg-Broad R@20 / Avg-Specific R@5）；基线数字出自论文 **arXiv 2407.18940 Table 3**（已逐行核对）。

---

# 二、论文取料

> 一句话：**arXiv id → 本地 PDF**。只负责下载，**不触发 MinerU、不生成报告**。

| # | 做什么 | 实现 | 关键点 |
|---|---|---|---|
| 1 | **输入** | 上游（论文检索）给的 arXiv id 列表 | — |
| 2 | **下载 PDF** | [`src/paperpilot/tools/arxiv_fetch.py`](../src/paperpilot/tools/arxiv_fetch.py)（线上**工具②**） | 落盘 **`assets/papers/<id>.pdf`**（**与用户上传同一个库**）→ 与 `ingest(pdf_name)` 输入口径一致，**抓完直接喂、无需胶水**；版本号 `vN` 一律去掉（取最新版）；★ **校验 `%PDF-` 魔数** |
| 3 | **可选顺手摄取** | `ingest(pdf_name)` | 默认 MinerU / 失败降级 pymupdf（见 §三） |
| 4 | **入口** | [`cli/run_fetch.py`](../cli/run_fetch.py) ｜ Web `POST /api/multianswer/fetch` | ★ Web 侧上限 **10 篇** |
| 5 | **编排（离线）** | [`multianswer/pipeline.py`](../src/paperpilot/multianswer/pipeline.py) `fetch_papers` | 离线三步编排的第 ② 步：方向检索 → **取料** → MultiAnswer |

> **为什么这样选（依据）**
> - **只下载、不管别的** —— [`arxiv_fetch.py`](../src/paperpilot/tools/arxiv_fetch.py) docstring：不触发 MinerU、不生成报告，保持职责单一（那些是 `ingest()` 的事）。
> - **落盘到 `assets/papers/`（与用户上传同一个库）** —— 这样与 `ingest(pdf_name)` 的输入口径**完全一致**，抓完直接喂、**不需要任何胶水**。
> - **校验 `%PDF-` 魔数** —— **arXiv 出错时会返回 HTML 错误页且 HTTP 200**（不是 404），不查魔数就会把 HTML 喂进解析链，**报错点还会推迟到很远的地方**。
> - **版本号 `vN` 一律去掉** —— `https://arxiv.org/pdf/<id>` 不带版本即取**最新版**。

---

# 三、论文问答直读（fullctx · **默认 reader**）

> 一句话：**报告层答不了 → 直接把全文塞进上下文让模型读**。单篇默认走这条。

| # | 做什么 | 实现 | 关键点 |
|---|---|---|---|
| 1 | **路由** | [`graph/qa_graph_v3.py`](../src/paperpilot/graph/qa_graph_v3.py) | Router(**L0 报告层**) 够 → **直答**；不够 → **fullctx 全文直读** |
| 2 | **装配上下文** | [`components/fullctx.py`](../src/paperpilot/components/fullctx.py) `build_context(pdfs)` | ★ **不检索块**，读全文；预算 `ctx_budget_chars()` —— 单位**字符**、判据 **fail-safe**（非法配置只能收紧）、★ **超预算绝不去调 API** |
| 3 | **作答** | 同上 `answer(pdfs, question, temperature=0.0, …)` | 引用形如 **`[P{n}·§…·¶cid]`**；风格开关 `PAPERPILOT_FULLCTX_STYLE`（默认 `base`） |
| 4 | **输出前闸门** | [`validator.py`](../src/paperpilot/components/validator.py) + [`numbers.py`](../src/paperpilot/components/numbers.py) → [`repairer.py`](../src/paperpilot/components/repairer.py) | 机器件（引用越界 / 编数漏数）**硬拦** → 对症修复 → 修不动才 LLM「补充说明」→ 兜底拒答 |
| 5 | **入口** | Web `POST /api/ask` ｜ 三栏工作台右栏追问（[`web/index.html`](../web/index.html)） | [`cli/main.py`](../cli/main.py) 只管"一步处理单篇 → report.json"，**不做问答** |
| 6 | **评测** | [`cli/eval/run_fullctx_qa.py`](../cli/eval/run_fullctx_qa.py)、[`run_group_qa.py`](../cli/eval/run_group_qa.py)（L3）、[`run_qa_eval.py`](../cli/eval/run_qa_eval.py) | — |

> **为什么这样选（依据）**
> - **默认直读而不是检索** —— 判据是 **`S / C` 不等式**（单篇正文中位 6,244 token ≤ 预算 → 直读即可；实测 B0 ≈ B2）。[`fullctx.py`](../src/paperpilot/components/fullctx.py) 顶部注释还留着当时的实测：**篇内 top1 命中只有 20%**，所以单篇上"检索块"反而不划算。
> - ★★ **分层（先看摘要答简单题）实测成立** —— **该拦的拦住 92%、该下沉的下沉 98%、因路由错误而答错的题数 = 0**；每 100 题**省 41% 输入 token + 27% 耗时**（2026-10-13，自造题集 + 配对两臂）。题集：本仓 45 篇论文造 **53 题「只需标题+摘要」+ 53 题「必须读全文」**，标签经**客观校验 + 跨模型复核（一致 96%）**；臂1 = 现状（L0 判够 → 只吃报告层直答）、臂2 = 不分层（运行时把 `judge_l0` 恒判不够 → 每题都读全文）。**混淆矩阵**：A 组 **49/53 停在 L0**（4 例误杀但**全部答对**）｜ B 组 **52/53 下沉**（唯一 1 例"该下沉没下沉"**也答对了**）。**集合级（106 题）**：分层 **1.65M** vs 不分层 **2.33M** token（省 41%）、**616s vs 450s**；通过 **105/106（99.1%） vs 102/106（96.2%）** —— **分层不仅更省，还更准**（全文塞进去反而稀释关键信息）。全文 [`qa/LAYERED_VS_FLAT_20261013.md`](../qa/LAYERED_VS_FLAT_20261013.md)。
> - **预算单位是字符、判据 fail-safe、超预算绝不调 API** —— [`tests/test_ctx_budget.py`](../tests/test_ctx_budget.py) 明写它守的就是这三件事（详见 [`docs/RAG_COMPONENT_NOTES.md`](RAG_COMPONENT_NOTES.md) 组件默认值）。
> - **默认值锁成结构性安全** —— [`tests/test_reader_switch.py`](../tests/test_reader_switch.py) 断言默认必须是 `fullctx`，否则"**下线 RAG-2 这件事根本没生效**"。
> - **输出闸门（硬拦 → 修复 → 拒答）** —— [`docs/RAG_COMPONENT_NOTES.md`](RAG_COMPONENT_NOTES.md) §6.5「输出闸门：零引用答案不再**静默放行**」；误报归因与修复验证见 `qa/recall/GATE_*` 一组（[`GATE_NUM_DIAG_20260911.md`](../qa/recall/GATE_NUM_DIAG_20260911.md) / [`GATE_FIX_VERIFY_20260911.md`](../qa/recall/GATE_FIX_VERIFY_20260911.md) / [`GATE_PROBE_QASPER_20260911.md`](../qa/recall/GATE_PROBE_QASPER_20260911.md)）。
> - ★ **直读路径：两条「须知」+ 一个真缺口** —— ① **共用底座**：`fullctx` 的正文与 L0 的 `core_points` **都建在「解析 → 切块 → claims」之上** → **任何 reader 都省不掉，这不是耦合**；② **默认链路仍会调一次块级检索**（`answer._audit_absence`，[`answer.py`](../src/paperpilot/agents/nodes/answer.py):650-662）—— **复用同一份块**、向量**惰性现建**；③ ★ **真缺口**：直读锚点 `[P·§·¶]` 是**开放字符串键** → 越界"天然无信号"，且**零引用会被跳过**（详见 §四「共用底座」）。

---

# 四、论文问答（**下线的 RAG** / 代号 RAG2）

> 一句话：**块级 reader 检索**（论文块级召回 → reader 生成答案）。**默认关闭**，不在前端暴露。

| # | 做什么 | 实现 | 现状 |
|---|---|---|---|
| 1 | **块级召回** | [`components/retriever.py`](../src/paperpilot/components/retriever.py)：`ChunkRetriever` / `ClaimRetriever` / `search_chunks(pdf, question, top_k=12, hybrid=True)` | 向量 + BM25 **RRF 融合**（单查询），底层 [`agents/embedder.py`](../src/paperpilot/agents/embedder.py) `ChunkIndex.search_hybrid` |
| 2 | **开关** | `PAPERPILOT_QA_READER=retrieval` | 默认 `fullctx`；本模式**只作 A/B 对照与回退** |
| 3 | **归档** | `archive/qa_funnel_v2/` | v2 四层漏斗快照，只作对照 |
| 4 | **评测（L2）** | [`_r2_prod_eval.py`](../retrieval/tmp/_r2_prod_eval.py)、[`_r2_std_metrics.py`](../retrieval/tmp/_r2_std_metrics.py)（一组 `retrieval/tmp/_r2_*.py`） | r2dev 28 组合：块级证据召回 / reader 判定 |

★ **别和 MultiAnswer 混**：RAG2 是**块级 reader 检索**（单篇内找块）；MultiAnswer 是**集合级多答案**（一批论文里挑篇）。

> **为什么这样选（依据）**
> - ★ **为什么下线（端到端实测，不是"技术不行"）** —— 生产口径 = **帮读链路 reader 的语料恒 5 篇**（`S/C ≈ 0.37`），同题同裁判下 **fullctx 60% > RAG2 块级 40% > 逐篇判定 34% > 旧清单 14%**（n=50，[`R2_SC_VERDICT_20260929.md`](../retrieval/results/R2_SC_VERDICT_20260929.md):107,151）。★ **而且不是因为"塞不下"** —— 实测单次调用可接受 **810k token**，5 篇 ≈ **44k token** 完全塞得下；检索线的**价值起点是 `S/C ≈ 0.8~2.0`** → 生产 5 篇（0.37）**落在价值区之外**。★ 所以准确表述是：**下线的是"在这个档位用检索"，不是"检索方法本身差"** —— 20 篇档（`S/C≈2.03`）检索相对整档直读**确有净收益**（F1 +4.4pt 不显著、**省 ≈8.7× 输入 token**、精确率 +17pt）。另有 [`README.md`](../README.md) 专节「归档：v2 四层漏斗（为什么下线）」。
> - ★ **那"检索方法本身"到底多好？（QASPER 单篇块级 · `k=4` 口径）** —— **208 篇 / 3,247 块 / 250 题**，**零 LLM**，gold = 全部人工 evidence 段并集。★ **为什么只用 `k=4`** —— 每篇**块数中位仅 15**（均值 15.6）→ **k=12 ≈ 喂进 77% 的块**，且 **61/208 篇在 k=12 时就是「整篇全返回」（R 恒 1.000）** → **那种深度等于整篇直读，不能拿来论证检索效果**。**`k=4` = 25.6% 的块（≈20.9% 的字符）**，实测 **R@4 = 0.704 ｜ NDCG@4 = 0.488 ｜ P@4 = 0.197**；逐槽累计 **#1 31.6% → #2 48.4% → #3 62.0% → #4 70.4%**（`#4` 单槽仍贡献 21 题 → **未到饱和**）→ **用约 1/4 的篇幅拿到 ~70% 的召回**。与 k 无关的体检：**AUC 0.747** ｜ **跨块率仅 21.6%** ｜ margin>0 28.9% ｜ gold 块 ↔ 最相似非 gold 块余弦 **0.511** ｜ 5 种切分策略 **AUC 全落 0.66–0.71**。★ **瓶颈在排序精度、不在召回**（P@4 仅 0.197）→ **块级召回（1/4 篇幅下）过关，输的是 5 篇档的端到端**。全文 [`RAG2_RETRIEVAL_EFFECT_20261013.md`](../retrieval/results/RAG2_RETRIEVAL_EFFECT_20261013.md)。
> - ★ **每篇 `b=12` 块 + 窗口 4,200 字符：维持不改** —— 2026-10-13 已按"新块"重扫两次（[`R2_SCAN_B_20261013.md`](../retrieval/results/R2_SCAN_B_20261013.md) → [`R2_SCAN_BWIN_20261013.md`](../retrieval/results/R2_SCAN_BWIN_20261013.md)，1,400 篇-题 + 真调 90 次实测成本/延迟）。**① `b` 与 `WIN` 是正交的两个旋钮** —— `b` 只决定「**能不能拿到**」（锚点全丢篇数 **121/54/4/0** @b=6/12/24/32，**与 WIN 无关**）；`WIN` 只决定「**拿到多少**」（锚点覆盖 **39.1%→69.9%**、入窗块数 **2→5** @WIN 2000→8000）。**② 不改的理由** —— `b` 12→24 只触及 **3.9%** 的篇-题，预期 setF1 位移**很可能 < ±0.04**（判官噪声本身 ±0.04）→ 要证明它真涨了必须**同题配对 + 多轮**（约 ¥8~10），而 `b` **零成本**，即"**改了也未必测得出来**"。**③ 成本/延迟已实测** —— WIN 2000→8000：token **852→2,594**（`≈0.29×WIN+272`）、成本 **¥0.0009→¥0.0027**，而**延迟只 0.76→0.86 s/次**（**不吃窗口**；8 并发 → **0.095~0.107 s/篇**）。**④ 若将来真要改 `b`，必须一并重算这三样**：端到端 **setF1/P/R（0.717/0.709/0.771 只在 `b=12` 下成立）** → 由它标定的 [`R2_BUDGET_MODEL_20261001.md`](../retrieval/results/R2_BUDGET_MODEL_20261001.md) **外推曲线**（`ρ` 与 `P=0.71` 是标定值，拐点 497 会动）→ **单判官协议等价**（0.717 vs 0.718 基于同批记录）；**不受影响**：检索臂指标（0.392 / oracle 0.706 / 2.02×）、切块审计、gold、CI。
> - **为什么"切分"不值得再投入** —— [`qa/recall/CHUNK_METRICS_20260910.md`](../qa/recall/CHUNK_METRICS_20260910.md) §C：**5 种切分策略 AUC 全落 0.66–0.71** → 排序质量对切分粒度**不敏感** → **瓶颈不在切分**；复盘见 [`docs/RAG_COMPONENT_NOTES.md`](RAG_COMPONENT_NOTES.md) §①补充。★ 同文件另提醒**块数口径本身不公平**（同样"每篇 12 块"，语料 285 块时覆盖 **62%**、1,037 块时只覆盖 **18%**）→ 须用**等字符预算**衡量。
> - **为什么保留代码不删** —— 作 **A/B 对照与回退**（[`docs/DOC_INDEX.md`](DOC_INDEX.md) 能力地图明确写"默认关闭；代码保留，作 A/B 对照/回退"）。

### ★ 与检索链的**共用底座**（**不是耦合，也省不掉**）

**结论**：**解析 → 切块 → claims 是整条产品线的底座**（报告层 / 检索 / 直读**全都要**）→ 换 reader **省不掉，也无需解耦** ✓
产物链 = **PDF → MinerU 解析 → 切块 → 逐块 claims(LLM) → `report.json`**（[`report.py`](../src/paperpilot/agents/nodes/report.py):1-7、[`pipeline.py`](../src/paperpilot/pipeline.py):13-18）；L0 的 `core_points` **直接来自 `claims`**，而 `claims` 锚在 `chunk_id` 上。

| # | 事实（**须知，不是缺陷**） | 说明 |
|---|---|---|
| **①** | **默认链路仍会调一次块级检索** | `answer._audit_absence`：L0 浅层答案带「论文未提及 / 找不到」口吻 → **强制一次 L3 块级复核**（[`answer.py`](../src/paperpilot/agents/nodes/answer.py):650-662；[`read_full.py`](../src/paperpilot/agents/nodes/read_full.py):28 明写「仍走 RAG-2」）。★ 它**复用同一份块**、向量**惰性现建** → **按需、一次性，不重复付解析/切块的钱** |
| **②** | **"读全文"的正文 = 切块视图，不是 PDF 原文** | `fullctx` 正文 = `_doc_chunks()` **遍历全部块**（不筛选/不排序/无 top-k → "全文"语义成立）；块由 `chunk_document()` 从 MinerU 产物**现切**（[`document_cache.py`](../src/paperpilot/agents/document_cache.py):218）。★ 实测"切块字符 / MinerU 原文字符"保留率 **81%~135%**（>100% = 表格/公式转文本**注入**；<100% = 丢页脚/页码/纯图片）→ **口径事实，不是缺陷** |
| **③** | **`chunk_id` 就是引用锚点的坐标** | `[P·§·¶c3]` 的 `c3` 即块 id → ★ **动切块策略 = 引用体系失效** |

✅ **真正被修掉的那半**（不是耦合，是**白花**）：摄取期原本「不管哪个读取器都 eager 建 `<stem>.cvec.npy`」，而默认 `fullctx` **一个向量都不会被读到** → 实测 **5 篇 21.8s 全白花**。现改为**惰性**（`index_needed()`；`test_fullctx_does_not_need_index` / `test_lane_index_skips_without_touching_encoder`，8 条测试绿），逃生开关 `PAPERPILOT_EAGER_INDEX=1`。★ 代价只是**成本挪位置** —— [`read_full.py`](../src/paperpilot/agents/nodes/read_full.py):66-68 明写 **返回 False ≠ "用不了/没有向量"**。

### ★★ 不能丢的设计资产（**下线的是 reader，不是架构与工程思想**）

- **★ 输出检查器的原设计**（[`validator.py`](../src/paperpilot/components/validator.py) + [`repairer.py`](../src/paperpilot/components/repairer.py)）：**机器件硬闸**（引用越界 / 编数，**零 LLM** → HIGH 拦）＋ **LLM 体检分层**（unsupported / off_topic / contradiction / vague → MID **只标注不拦**）＋ **修复回路**（生成型 → Self-Refine；证据型 → CRAG 重检索）＋ **salvage**（修不动 → 带「非系统作答」**强制标注**）＋ **拒答兜底**。→ ★ **"哪些问题硬拦、哪些只标注"这条分界线是 RAG2 时代定的，现在 fullctx 路径照用。**
- **块级证据与多路检索**（向量 + BM25 + **RRF 融合**、`ChunkRetriever`）→ **MultiAnswer（§五）整条线在复用**。
- **引用可机器校验的设计**（`[n]` 闭区间 + 越界基准 = 上下文条目数）→ 这是"能对答案做引用审计"的前提。
- **切分与语料资产**（MinerU 骨架、切块质量档案）→ 现在是**每条线的底座**。

⚠️ **用这套资产时必须知道的边界**：fullctx 的锚点 `[P·§·¶]` 是**开放字符串键** → 越界"**天然无信号**"，且**零引用在直读路径上会被跳过**（[`validator.py:166-174`](../src/paperpilot/components/validator.py)）→ ★ **"引用可机器校验"这个能力，目前只在检索路径上完整成立**。

---

# 五、多答案开放域问答（MultiAnswer）

> 一句话：**一批论文 + 一个问题 → 哪几篇做了这件事 → 汇成一段回答**。
> 核心只有一个 `set_judge.run()`，上层（在线 / CLI / 编排）都是**薄编排**。

| # | 做什么 | 实现 | 关键点 |
|---|---|---|---|
| 1 | **定语料** | [`web/app.py`](../web/app.py)（`GET /ma`） | URL `?pdfs=a.pdf,b.pdf`（**裸文件名**，相对 `assets/papers/`）→ 后端**强制 `n = len(pdfs)`**，**≤10**（`_MA_LIVE_MAX=10`）；★ 后端**没有"当前批"会话状态** |
| 2 | **切块** | [`agents/document_cache.py`](../src/paperpilot/agents/document_cache.py) `ordered_chunks` | ★ **块级**（与 §一 的"篇级"相反）；**MinerU 作骨架**，失败降级 pymupdf；中位块长 ~1.9k 字符 |
| 3 | **建索引** | `CorpusIndex.from_parquet` + `encode_texts`（bge-m3） | — |
| 4 | **取块（多路）** | [`multianswer/corpus.py`](../src/paperpilot/multianswer/corpus.py) → `ChunkIndex.search_multiroute` | ★ **原样借生产方法，零分叉**；每篇留前 `b=12` 块（`per_paper_hits`） |
| 5 | **构证据窗口** | [`components/set_judge.py`](../src/paperpilot/components/set_judge.py) `build_user` | **判据锚点支 ∪ 语义支**，上限 **4,200 字符**；锚点块**优先入窗** → 片段编号**乱序**（有意为之） |
| 6 | **逐篇判定** | 同上 `judge_one` | 每篇 **1 次 LLM**（`deepseek-chat`），线程池（默认 6；CLI/在线传 8）→ `yes/no/unclear` + 证据片段号；默认**单判官**（`SET_DUAL=0`） |
| 7 | **汇总成回答** | [`multianswer/synth.py`](../src/paperpilot/multianswer/synth.py) `synthesize()` | ★ **只有在线 `/ma` 走这步**；逐篇清单作「依据」（`set_judge.render()`） |
| 8 | **入口** | Web `GET /ma` + `POST /api/multianswer/search` ｜ [`cli/run_set.py`](../cli/run_set.py) ｜ [`cli/run_rag2_flow.py`](../cli/run_rag2_flow.py) | [`cli/run_set.py`](../cli/run_set.py) → `MultiChunkIndex` + `components.set_judge` |
| 9 | **离线两档语料**（仅 CLI/评测） | [`multianswer/pipeline.py`](../src/paperpilot/multianswer/pipeline.py) | 实时档 **≤10 篇**（RAG1 → 取料 → 切块）｜ **离线簇 50 篇/簇**（`retrieval/data/r2dev/prodchunk50/`） |
| 10 | **评测（L4）** | [`evals/runners/l4_loft.py`](../evals/runners/l4_loft.py)（QAMPARI）＋ [`cli/eval/run_multianswer.py`](../cli/eval/run_multianswer.py)（离线 3×50） | QAMPARI **是它的对外基准，不是独立模块** |

★ **两个口径陷阱**：① 离线评测是 **3 簇 × 50 篇**，在线 `/ma` **最多 10 篇** —— 不能把 50 篇的 F1 当在线效果；② 离线 [`run_multianswer.py`](../cli/eval/run_multianswer.py) **只出答案、不算指标**（指标脚本在 `retrieval/tmp/`）。

> **为什么这样选（依据）**
> - **切块用 MinerU 作骨架** —— [`R2_CHUNK_SWITCH_20260929.md`](../retrieval/results/R2_CHUNK_SWITCH_20260929.md)：固定窗 → 生产章节切块，**12k 字符预算下证据召回 0.459 → 0.575（+11.6pt）**、成本降 **3.7×**；旁证 [`CHUNK_AUDIT_20261006.md`](../retrieval/results/CHUNK_AUDIT_20261006.md)（MinerU vs 同篇 pymupdf：B 自包含性 / C 主题一致性两组全面偏向 MinerU）；决策时点见 [`docs/RAG_COMPONENT_NOTES.md`](RAG_COMPONENT_NOTES.md) §6（**2026-09-22 翻转**）。
> - **多路检索（7 路）** —— [`R2_BM25_AND_MULTIQ_20260930.md`](../retrieval/results/R2_BM25_AND_MULTIQ_20260930.md)。
> - ★ **每篇 `b=12` 块 + 窗口 4,200 字符：维持不改** —— 理由同 §四（`b` 只关联 3.9%、位移在判官噪声 ±0.04 内）。报告：[`R2_SCAN_BWIN_20261013.md`](../retrieval/results/R2_SCAN_BWIN_20261013.md)。
> - **证据窗口 4,200 字符** —— [`R2_BUDGET_MODEL_20261001.md`](../retrieval/results/R2_BUDGET_MODEL_20261001.md)：**窗口封顶**带来一个关键性质 ——「**成本侧与论文长短无关**」（论文再长，判官也只吃 4,200 字符）→ 于是可判篇数可解析：`m = floor((X − 2c)/c)`。常量本体记在 [`R2_C3_WINDOW_20260930.md`](../retrieval/results/R2_C3_WINDOW_20260930.md)（`EV_MAX_CHARS=4200`）。
> - **窗口 = 判据锚点支 ∪ 语义支** —— [`R2_PROD_FINAL_20261001.md`](../retrieval/results/R2_PROD_FINAL_20261001.md)：**判据锚点支 R +0.117、F1 +0.061、精确率持平**（机制：锚点块优先入窗 → 判官看到证据就敢多判 yes，但**不引入假阳**）。
> - **单判官（`SET_DUAL=0`）** —— [`R2_JUDGE_PROTO_20261001.md`](../retrieval/results/R2_JUDGE_PROTO_20261001.md)：拿 **1,400 条已有记录零新调用离线复算**，`onlyA` ≈ 双判官（F1 **0.718 vs 0.717**），调用 **2.10 → 1.00 次/篇（省 52%）**；机理：第二判官系统性更保守，第三轮只是把它的假阴修回来。
> - **产物必须是"回答"而不是"清单"** —— [`R2_SC_VERDICT_20260929.md`](../retrieval/results/R2_SC_VERDICT_20260929.md)：旧形态（篇名+理由）准确率仅 **12~14%**；改成"判定当路由 + 把命中篇内容交给模型" → **68%**（因为题问的是"**各自是什么**"，清单答不了）。
> - **在线 ≤10 篇** —— [`docs/RAG2_CORPUS_WIRING.md`](RAG2_CORPUS_WIRING.md) §0.2（**2026-10-06 简化**：删掉 A/B 两条链路 + N 下拉框，语言由 URL 唯一确定）。
> - **离线 3 簇 × 50 篇** —— [`R2_CORPUS_50_20260930.md`](../retrieval/results/R2_CORPUS_50_20260930.md)：从 20 扩到 50，题内正例 **4.71 → 13.00**（是**难度变化**，不是"灌噪声"）。
> - **大 N 外推的四个假设** —— [`R2_BUDGET_MODEL_20261001.md`](../retrieval/results/R2_BUDGET_MODEL_20261001.md)（含"精确率取常数 0.71 是最弱环节"）。

---

# 六、不能放进上面五条的

| 类别 | 位置 | 说明 |
|---|---|---|
| **对外汇报 / 总入口** | [`RESULTS_SUMMARY.md`](../RESULTS_SUMMARY.md)、[`README.md`](../README.md) | 成果汇总与能力地图 |
| **文档索引与规范** | [`docs/DOC_INDEX.md`](DOC_INDEX.md)（信谁）、**本文件**、[`docs/RAG_COMPONENT_NOTES.md`](RAG_COMPONENT_NOTES.md)（组件默认值+依据）、[`docs/RETRIEVAL_LOG.md`](RETRIEVAL_LOG.md)（研究区检索史） | 冲突时优先级：**代码 > README > docs/\* > 历史报告** |
| **回归护栏** | `evals/`：`checks/`(3 闸门) · `baselines/` · [`runners/l4_loft.py`](../evals/runners/l4_loft.py) · [`RESULTS.md`](../evals/RESULTS.md) | L0 零 LLM 零网络；CI = `compileall + pytest` |
| **测试** | `tests/` | **29 个文件 / 379 用例**，离线（不装模型不联网），**13 秒**跑完 |
| **前端（横跨多条线）** | [`web/app.py`](../web/app.py)、[`web/index.html`](../web/index.html)、[`web/multianswer.html`](../web/multianswer.html) | 三个页面共用同一个后端 |
| **历史区** | `qa/`（262 份）、`archive/` | ★ 很多结论**已被取代**，只在追溯时看 |
| **报告（混装）** | `retrieval/results/`（86 份 md + 产物） | ★ 按**文件名前缀**找：`LITSEARCH_`=检索、`R2_`=第二轮实验、`CHUNK_`=切块、`QAMPARI_`=多答案对外（**前缀是标签，不是文件**） |
| **分析脚本** | `retrieval/tmp/` | 多为一次性分析；保留与否见 [`evals/baselines/tmp_scripts.json`](../evals/baselines/tmp_scripts.json) |
| **数据与产物（不入库）** | `assets/`、`retrieval/data/`、`evals/reports/` | 由 [`.gitignore`](../.gitignore) 管理，可再生成 |

### ★ 评测分层（L0–L4）：每层一个触发条件

| 层 | 判什么 | 实现 | 触发 | 成本 |
|---|---|---|---|---|
| **L0 · 不变量** | 仓库状态与契约（**不跑模型**）：路径引用是否悬空、gold 与导出是否同步、评测记录格式合规 | [`evals/checks/`](../evals/checks/path_liveness.py) 三闸门 + `tests/`（**379 例**） | **每次改动**（CI 默认） | **秒级 / 免费** |
| **L1 · 篇级检索（RAG1）** | LitSearch 64,183 篇：Recall@k / nDCG | [`cli/eval/run_retrieval_eval.py`](../cli/eval/run_retrieval_eval.py) | 发版前 / 手动 | 分钟级 |
| **L2 · 块级 reader（RAG2）** | r2dev 28 facet：证据召回 / reader 判定 | [`_r2_prod_eval.py`](../retrieval/tmp/_r2_prod_eval.py)、[`_r2_std_metrics.py`](../retrieval/tmp/_r2_std_metrics.py) | 发版前 / 手动 | 分钟~小时 |
| **L3 · 端到端** | 产品链路（`graph.ask`）+ 5 组 164 题 | [`cli/eval/run_group_qa.py`](../cli/eval/run_group_qa.py) | 发版前 / 手动 | 小时级 |
| **L4 · 对外可比** | MultiAnswer 外部评测（**主用 QAMPARI**）；runner 同时支持 LoFT 五任务 | [`evals/runners/l4_loft.py`](../evals/runners/l4_loft.py) | **按需** | 小时级 |

★ **为什么这么分层**：把「**免费且必须每次跑**」（L0）与「**贵且只需发版前跑**」（L1–L3）分开 —— 否则一套评测**要么贵到没人跑、要么便宜到抓不住问题**。L4 单独一档，因为它的价值在**对外可比**（与公开榜同口径），触发节奏与内部回归不同。
★ **`tests/` 的 379 例就是 L0 的主体**（秒级、零 LLM、零网络）→ 所以"一条命令"演示到的**就是 L0**；L1–L4 要真模型 / 真花钱，**发版前才跑**。
⚠️ 别与**系统漏斗**的 L0–L3 混（那是**运行期路由**：L0 报告层 → L3 全局检索）；v2 的 `L0 → L1 claims → L2 圆心扩窗 → L3` 四层漏斗**已归档**（[`README.md`](../README.md):414 明写"已从活代码彻底移除"）。

**三条最容易搞混的名字**：

1. **RAG2 ≠ MultiAnswer** —— RAG2 是块级 reader 检索（§四），MultiAnswer 是集合级多答案（§五）
2. **`R2_*` 文件名 ≠ RAG2 模块** —— 多数是"**第二轮实验**"的意思
3. [`docs/RAG2_CORPUS_WIRING.md`](RAG2_CORPUS_WIRING.md) **这个文件名是历史遗留** —— 正文讲的是 **MultiAnswer**

> **为什么这样设计（依据）**
> - **三道闸门都是"棘轮"（只拦**新增**）而不是"清零"** —— 理由写在闸门自己的 docstring 里（[`path_liveness.py`](../evals/checks/path_liveness.py)、[`tmp_scripts_audit.py`](../evals/checks/tmp_scripts_audit.py)）：**要求历史债清零会让闸门天天红、被"狼来了"淹没，比没有闸门更糟**；[`tests/test_tmp_scripts_audit.py`](../tests/test_tmp_scripts_audit.py) 也明写它"**只断言不变量、不比快照**"。
> - **测试能"无 key 全绿"** —— [`tests/conftest.py`](../tests/conftest.py) 把 LLM / embedding / MinerU **三样都换成替身**，并把 `urllib.request.urlopen` 换成"**一用就炸**"（漏了替身会**明确失败**，而不是偶发联网成功）；并**清掉一批 `PAPERPILOT_*` env**（否则"本机 `PAPERPILOT_QA_READER=retrieval`"会让测试测到的是**开发者的机器**而不是代码默认值）。[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) 注释说明为何**故意不装 sentence-transformers**（它连带 torch 几个 GB，而测试套件**不 import 它**）。
> - **报告为什么"混装"却仍能找到** —— 靠文件名前缀约定（见上表）。

## 维护约定

1. **新增/移动/改名任何文件 → 回本文件改那一行**，否则这张图立刻开始骗人。
2. **本文件只写"链路 + 落点 + 依据"，不写结论与数字** —— 数字回原报告引用（口径 + 日期是硬要求）。
3. 有新业务线 → 新开一节，按同样的"编号 + 实现 + 产物"三列写。
