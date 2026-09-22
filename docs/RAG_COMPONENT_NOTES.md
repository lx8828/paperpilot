# RAG 组件化设计笔记（2026-09-09）

> ✅ **现状核对（2026-09-22，以代码为准）**：本文档的**组件默认配置**已逐条对过源码，结论仍成立。
> 两处数字已更新：题集规模 **31 篇 / 304 题**（下文 §⑤ 里"276 题"是 09-21 口径）；
> 切分骨架默认已翻转为 **MinerU**（§① 与 §6 已注明）。**权威现状入口仍是根 `README.md`**，
> 本文档是"为什么这么配"的证据账，冲突时以 README + 代码为准。

> 背景：PaperPilot 检索层从"按漏斗层组织"（v2 L0-L3 / v3 两级）向**大厂模块化 RAG 管线**
> 过渡。本文档把此前全部探索结论固化为**每个组件的默认配置与证据**，目标是：
> ① 新架构每个组件的取舍都有据可查，不重蹈"反复调试无提升"；
> ② 单篇(短论文) → 多篇/主题级扩展时，知道哪些组件那时才生效；
> ③ 面试/协作有一套清晰、对得上业界名词的组件叙事。
>
> 历史决策链：`qa/RETRIEVAL_EXPLORATION_20260908.md`（做了什么，为什么收敛到 v3）；
> 评测规范：`qa/RETRIEVAL_EVAL_FRAMEWORK.md`（C/T/E/A 分层）。本笔记 = 目标形态与组件配置。

---

## 1. 目标管线（业界 Modular RAG 参考顺序）

```
论文 PDF ─► [DocumentSplitter] ─► [Router] ─► [QueryOptimizer] ─► [Retriever(混合)]
     ─► [Reranker] ─► [ContextBuilder] ─► [Generator] ─► [Validator] ─► 答案
                                    ▲___________________________│
                              不够/不合格则回灌（Router/检索升级）
```

> 这是经典 **Modular RAG** 形态（Naive → Advanced → Modular），大厂生产管线普遍是其子/全集。
> 注意：组件是**装配灵活性**，不是"越多越准"——单篇短论文下多个组件已被我们实测无益（见下）。

---

## 2. 组件清单：职责 / 现状 / 探索证据 / 默认配置

### ① DocumentSplitter（预处理·切分）—— ✅ 已有，且是独特资产
- 职责：把 PDF 切成带结构（标题层级/版面）的文本块，供检索与抽取。
- 现状：`pdf_parser → heading → chunker`（标题树切块，段落原子 ~4000 字符）+ QASPER `build_chunks`。
- 独特资产：切块之上还有**主张抽取层**（LLM 逐块抽 claim + 原文 evidence，锚 chunk/title_path）→
  这是"帮读/溯源"产品差异的来源，不要退回纯分块。
- **MinerU 默认（2026-09-22 翻转）**：MinerU 作 **chunk 骨架**，pymupdf 退为**容灾**。
  关键收益：**表格文本直接进骨架** → claims / 报告 / L0 都能看到表值
  （旧默认表格只进检索视图 → L0 对表格类问题系统性无能）。细节与开关见 §6。
  **同篇双源实测**（`1701.00185v1`）：pymupdf **漏判 `4. Experiments`** → 4.1~4.6
  六个子节**错挂到 `3. Methodology` 下**；还把 2 条**参考文献条目误判成 APPENDIX**
  （`heading.py` 的"逗号≥2"防护没兜住）；claims 154 vs 140、骨架边 86 vs 77、
  图表 16 vs 12、**溯源回核 ✓154 ✗0 vs ✓135 ✗5**；MinerU 还保住了 `STC²` 上标
  （pymupdf 退化成 `STC2`）。

#### ①补充 切分粒度实测档案（2026-09-10 两份资产；2026-09-21 重新发现并复核）

**为什么单独立档**：这两份实验当时都做完并写在 `qa/recall/`，但**结论没有被复用**——
`MERGE_AB_t900.md` 测出正收益却**既没落地、也没记原因**；`CHUNK_METRICS` 的等预算表
**已经量化了"固定 k 偏向大块"**。2026-09-21 做多篇检索时**把同一个坑重踩了一遍**
（绕了 quote/rrf/znorm/zmean/BM25 五种方案才发现根因是粒度）→ 故上提到本文档。

**A. `qa/recall/MERGE_AB_t900.md` —— 只并 <900 超小块（贪心）**
- 算法原文（归档 `qa/_archive/recall/_merge_plan.py`）：块序贪心；`len < TH` 并入下一块；
  接收块超 `HARD_CAP = 4500` 则**放弃并入**；TH 扫 700/900/1100 → 取 **900**。
- 结果（QASPER 208 篇 / 250 题）：块 **3247→2377**；R@8 0.864→0.916、R@12 0.960→**0.976**、
  R@16 0.972→**0.996**、NDCG@12 0.576→0.585、MRR@16 0.458→0.464；gold@1 0.264→0.256；
  逐题 变好 85 / 变差 49 / 持平 116。
- ⚠️ **口径警告（未复核）**：该实验用**固定 k** 口径，而 `CHUNK_METRICS` 明确指出固定 k 下
  **大块虚高**；merge 的作用正是"把块并大" → **+2.4pt 里有多少真实、多少虚高，至今没有复核**。

**B. `qa/recall/CHUNK_METRICS_20260910.md` §C —— 切分策略对照（含口径修正）**
- **固定 k=12（文档自标"偏向大块"）**：现状 R@12 **0.912** ＞ 窗口512 0.706；窗口2048 0.971 ＞ 现状。
- **等预算表（按 rank 序累计字符达 B 前命中 gold；消除"大块更占便宜"）**：

  | 策略 | R@8k | R@16k | R@24k |
  |---|---|---|---|
  | 现状（标题+段落打包≤4000） | 0.765 | 0.853 | **1.000** |
  | 固定窗口 512 | **0.853** | 0.941 | 1.000 |
  | 固定窗口 1024 | 0.765 | 0.912 | 0.971 |
  | 固定窗口 2048 | **0.853** | **0.971** | 0.971 |
  | 段落打包≤1024（无标题） | 0.735 | 0.853 | 0.941 |

  > 文档原话：**"固定 k 下大块因「每 slot 装更多字/覆盖更大比例正文」而虚高，判断切分优劣看等预算表。"**

**C. 2026-09-21 多篇检索复核（5 篇同主题 / 146 chunk）**
- **现象**：5 篇**总字符数相近**（57k~72k），但 chunk 数 22~52、**中位块长 676~3956（差 5.8×）**
  → 全局 cosine 实际在比"**谁切得细**"：泛化查询 top12 被单篇 **100% 霸占**、top48 仍占 71%。
- **根因**：`chunker` **只有上界、没有下界**（`if len(text) <= max_len: 整块输出`），
  切分单位是**标题段** → **块长 ≡ 论文作者划的小节粒度**。实测 ConVerSum 的 `title_path`
  深度 `{1:6, 2:26, 3:20}`（小节分到三层 `4.11.1`）→ 中位 659；另 4 篇最多两层 → 中位 2181~3956。
  **不是 heading 误判**（最短块标签复核均为真标题）、**不是 MinerU 解析源差异**（5 篇全走 pymupdf）。
- **已证否的手段（别再走）**：全局 vstack ❌；加大 top_k（12→48 无效）❌；
  **BM25 并不免疫**（其长度归一化是"惩罚长文档"，在 5.8× 的块长差下**反而过度补偿短块**：
  块长相关 r —— 向量路 −0.496 / BM25 **−0.563**）❌；
  rrf / znorm / zmean 融合均治标（**znorm 最差且引入新偏好**：偏爱"篇内方差大"的篇
  = 块长跨度大的那篇，于是查询里写着专名时首名反而给错）。
- **有效方向**：把检索单元切成**均匀窗口** → 5 篇窗口数 125/117/112/108/97（差 1.29× vs chunk 2.36×）；
  平均篇数 **2.2 → 4.0**，指向性目标命中 7.2 → 6.6（仅降 0.6），长度相关 r −0.496 → −0.288。
- **落点选择**：走**检索侧**（在 `document_cache` 再加一套"检索窗口"视图，与既有
  `ordered_chunks`(报告视图) / `retrieval_chunks`(检索视图) 同构），**不动 `chunk_id` 空间**
  → 报告链 / claims / cites / 526 个已有 `.cvec.npy` **全部不动**。
  数据层重切（合并 + 重切）需**全库重建**（claims/report/cites 全失效），暂不做。

### ② Router（路由/难度分类）—— 🟡 有隐式逻辑，需显式化
- 职责：判断问题类型 → 决定路径（全貌直答 / 局部检索 / 多维分解 / 拒答）。
- 现状：v3 用 `judge_l0` 的"够/不够"二分类当隐式 Router（全貌 vs 需检索），表现达标。
- **证据/警告**：LLM 预测作路由是危险做法——L2 时代 judge 判够精度仅 ≈61%（
  `JUDGE_DIAGNOSIS_20260908.md`）。**L0 的二分所以可靠**：输入完整(overview+core_points)、
  判断"有没有提到"而非"证据够不够拼答案"、判错可降级不致命。
- 默认配置：Router 输出带**降级链**（每条路判不够可升下级检索），绝不做"分类定生死"的硬路由。
- 多篇时代：主题识别/多论文归并需要更强的 Router。

### ③ QueryOptimizer（查询侧优化）—— 🟡 五级骨架已建，**仅 L3 实现，默认全关**
- 现状：`components/query_optimizer.py`。**L1 是路由器，L2~L5 是并列变换器 —— 不是串行五步**
  （串行跑满 = 5 次 LLM 调用 + 互相矛盾的重写，成本翻倍且互相稀释）：
  L1 意图识别与问题分类、L2 问题重述、**L3 改写与扩展（唯一已实现）**、L4 HyDE、L5 查询分解。
  **未实现的级别显式抛 `NotImplementedError`**（刻意：宁可响亮失败，也不要"开了却没发生"），
  设计约束与风险写在各自函数 docstring 里。
- **开关**：`PAPERPILOT_QUERY_LEVELS=l3`（等价旧的 `PAPERPILOT_QUERY_REWRITE=1`）；未设置 = 全关。
  （旧的 `components/query_rewriter.py` 与 `tools/query_expand.py` 是**死代码**，已删除。）
- **证据**：2026-09-07（40 题同裁判）ON pass 9 vs OFF 12；2026-09-10（100 题配对）
  ON 68% vs OFF 71%（救 3 / 坏 6），calls/题 +30%。
- **机制诊断（最关键的一条）**：`search_multi_hybrid` 把变体与原问题做**等权 RRF 融合**，
  偏题变体会稀释原问题的正确排序。**不是"改写无用"，是"RRF 平权融合有害"** ——
  已把三种融合原语写进组件：`equal_rrf`（旧，仅作对照）/ **`quota_union`（已实现：
  原问题取 k1 + 各变体各取 k2 的并集，变体只做召回补充）** / `rerank_orig`（待实现）。
- 默认配置：**全关**。多篇/语料库场景重开，**届时必须重新 A/B，且建议先测 `quota_union`**。
- 参照：混合检索在单篇上"无增益"、在 63k 语料上变成 **+6.7pt 显著**（`retrieval/README.md`）——
  **规模会改变结论，但不能预设改写一定会翻盘。**

### ④ Retriever（混合检索）—— ✅ 已有（真正的混合检索在这）
- 现状：
  - `ChunkIndex.search_hybrid` = 向量 + BM25 **RRF 融合**（单查询；专名/术语精确匹配互补）→ L3 主检索，top12。
  - `ClaimIndex` = claims 主张向量检索（v3 已不再单列一层，多篇/主题时可作检索源复用）。
  - top12 覆盖证据 ~85%（调过 top8→top12）。
- **证据**：朴素 RAG(B1)≈我们(B2)（`compare_20260907_191606.md`：49 vs 51）→ 检索件不需要花哨；
  真正的检索短板在**数值/表格块**（L3 捞不到表值），靠 MinerU/输入质量解，不是加检索复杂度。
- **默认配置**：hybrid + top12；改检索先跑 C 层 Recall@k（多篇再建真值集）。
- 🐛 **2026-09-21 修复：BM25 无信号时会产生「索引序伪位次」污染 RRF**（重要，别再踩）
  - **现象**：多篇语料下"第一篇霸占"——泛化查询 top12 100% 来自拼接序第一篇。
  - **机制**：`BM25Index` 分词是「英文词 + **单个汉字**」，而语料是**英文论文**
    → **纯中文查询的 token 全部无匹配 → `score()` 全 0**（实测 5 篇：中文查询 max 0.000、
    非零 0/146；掺英文的查询 max 0.911~3.212、非零 19%~30%）。
    此时 `np.argsort(-bm_scores)` 对全 0 数组返回**索引序** → 每块按"它在拼接语料里的
    先后"拿到 `1/(60+i)` 的**伪分** → **拼接序第一位的那篇被系统性加分**。
  - **影响面**：**产品是中文提问 → 必现**；QASPER 评测全英文查询 → **从未测到**
    （又一个"评测口径覆盖不到真实用法"的例子）。
  - **修复**：新增 `_bm_or_none()` —— 分数全 0（或空）时返回 `None`，
    `rrf_order` / `search_multi_hybrid` 随即**跳过 BM25 路**（`rrf_order` 本就支持
    `bm_scores=None`）；`hits["bm_rank"]` 在无信号时记 **0**（不再假装"第 1 名"）。
  - **实测收益（多篇）**（5 篇 / 10 题）：平均篇数 2.0→**3.7**、top1 覆盖 1.8→**3.5**/5、
    等预算覆盖 R@24k 21.3→**30.8**（上限 60）；泛化查询从"篇数 1"变为"篇数 4~5"。
  - **中文回归验收（单篇）**：`cli/run_qa_v2.py`，**31 篇 / 304 题**（改写于 2026-09-22：第一组 5 篇已重建为 gold 引文口径 70 题；当时口径为 276 题），与 09-09 基线**同题集逐题对比** →
    **✅ 268→270、⚠️ 8→6、❌ 0→0（净 +2，零倒退）**。4 题 ⚠️→✅，其中
    **2 题从 `answer_unknown` 变 `answer_L3`**（`03335-07` / `03035-07`）—— 正是
    "伪位次 → 该答的漏检/误拒答"这一症状被修掉；2 题 ✅→⚠️（路线未变，LLM 波动量级）。
    台账：`qa/qa_summary_20260921_221659.md`（基线 `qa/qa_summary_20260909_003404.md`）。
  - **注**：修复只影响"BM25 无有效信号"的查询 → **纯中文提问必现**，QASPER（全英文）
    **永远不会测到** → 再次印证"评测口径必须覆盖真实用法"。

### ⑤ Reranker —— ✅ **大召回面下已验证显著有效**（单篇仍不做）
- **2026-09-18 实测**（`retrieval/`，LitSearch 63,269 篇篇级检索，597 题）：
  用 `BAAI/bge-reranker-v2-m3` 对 hybrid α=0.5 的 **top-100** 重排 →
  **R@1 0.3498 → 0.4154（+6.56pt）**、R@10 0.6059 → 0.6712、MRR@10 0.4459 → 0.5072，
  **全部统计显著**（McNemar hit@10 净 +39，p=9.8e-06；bootstrap recall@1 p=0.0002）。
  详见 `retrieval/results/LITSEARCH_RERANK.md`。
- **本文档当时的预判成立**：单篇候选面小（top12 内几乎全相关）→ 理论上限低，不做；
  **大召回面（top-100）时才值得上**。
- **约束**：精排上限 = 基线的 R@100（重排无法召回新文档）→ **先保召回、再修排序**
  （有精排时 RRF 的 α 应选保 R@100 的那个）。
- 单篇默认配置不变：接口预留（`components/reranker.py`），不启用。

### ⑥ ContextBuilder（上下文组装）—— 🟡 有雏形，重点加压缩
- 现状：`generate_answer._build_context/_compose` 按证据形态组上下文（claims/窗口/L3 全文块）
  + **两步法**（L2/L3 先穷举 facts 清单再作答，防"踩点漏细节"）+ 缺失断言复核闸门（浅层答
  "论文没写"时强制 L3 复核，防误拒）。
- **证据**：缺失复核闸门对拒答可靠性有贡献（防幻觉 20/20 依赖它）；两步法 facts 与 pass 正相关。
- 待办：上下文**压缩**（多块去冗余）单篇收益未验；与表格块接入配合。维持 cites 引用编排不破坏。

### ⑦ Generator —— ✅ 已有
- 现状：`generate_answer`（LLM，主链路 `deepseek-chat`）；按档位组织上下文并引用编号。
- 默认配置：temperature=0；保持 cites 输出（前端可跳原文）——溯源是产品核心卖点，任何重构不得丢。

### ⑧ Validator（检查答案）—— 🟡 有零散逻辑，需独立 + 加数值校验
- 现状零散：缺失断言复核闸门、cites 规则校验（引用越界丢弃）、claims evidence 提取期验证。
- **证据**：诚实拒答/不编造是差异点（负向 20/20、无答案桶 6/8>B1/B0）——Validator 是最该保留并
  独立的组件；单篇若有 pass 增量，最可能来自这里（数值/事实校验）。
- 待办：数值类答案强制"列出从原文拿到的具体数字再下结论"（多方法对比题 50% 短板就在缺数）；
  引用支撑度检测（claim→evidence 是否真支持 answer 观点）。

---

## 3. 单篇 → 多篇/主题级 路线（哪些组件那时才兑现）

| 时机 | 启用/新增 | 依据 |
|---|---|---|
| 多篇/主题级 | QueryOptimizer 开级别（**重新 A/B，建议先测 `quota_union`**） | 单篇负收益是"检索面小 + 等权 RRF 稀释"所致 |
| 多篇/主题级 | Reranker（候选大后排序才有意义） | 单篇 top12 内无需精排 |
| 多篇/主题级 | C 层 Recall@k 真值集 + 检索评测 | RETRIEVAL_EVAL_FRAMEWORK 缺口① |
| 多篇/主题级 | Router 主题归并/跨论文指代 | 现 Router 只服务单篇 |
| 先于多篇 | MinerU（解析/表格/扫描） | 影响 DocumentSplitter 输入质量 |
| 单篇即可做 | Validator 数值校验、ContextBuilder 压缩实验 | 成本低、有明确失败样本 |

---

## 4. 代码组织过渡（门面渐进，不重写）

```
src/paperpilot/components/      # 目标：薄门面命名归位（import 现有实现，先不搬逻辑）
    splitter.py                 #   pdf_parser+chunker+claims 抽取 门面
    router.py                   #   judge_l0 门面（够/不够→路径）+ 降级链约定
    query_optimizer.py          #   查询侧优化（五级骨架；仅 L3 实现，默认全关）
    retriever.py                #   ChunkIndex/ClaimIndex 统一检索入口
    reranker.py                 #   空壳（单篇不做；大召回面已实测有效，见 §⑤）
    context_builder.py          #   _build_context/_extract_facts/缺失复核 收敛
    generator.py                #   generate_answer 门面
    validator.py                #   cites 校验/缺失复核/将来数值校验 收敛
src/paperpilot/{agents,graph}/  # 现行代码保留（v2 回退 + v3 默认），过渡期并存
qa/RETRIEVAL_EVAL_FRAMEWORK.md  # 组件评测沿用 C/T/E/A
```
原则：**门面先建、逻辑后搬**；任何组件上线前按框架跑 T/A（同口径 + 显著性），带成本 diff。

---

## 5. 铁律（踩坑换来的，别违反）
1. **单篇场景下：检索复杂度不保值**（朴素 RAG ≈ 我们）——改动先问"失败样本是什么"，别为架构先进而投。
2. **LLM 预测当 Router/Judge 危险**（61% 精度）——必须配降级链/兜底；判"够/不够"的对象要简单。
3. **溯源 cite 与诚实拒答是差异点**——Generator/Validator 重构永远保留，评测带负向/拒答专项。
4. **每个组件默认关/开都有 A/B 背书**，且评测带成本 diff（llm.usage 计量已就绪）。
5. 多篇再启用的组件（Rewriter/Reranker）现在只留接口，避免在无效场景上反复调试（教训：L2）。
6. **判断切分/召回优劣必须用「等预算」口径，不能用固定 k** —— 固定 k 下"大块因每 slot 装更多字"
   系统性虚高（§①补充 B 已定量；`CHUNK_METRICS` 原文）。凡改切分或改召回面，先问一句
   **"这个口径会不会奖励大块"**。（2026-09-21 重踩：多篇下块更细的篇被系统性偏袒。）
7. **实验做完必须写「结论 + 去留」，哪怕结论是"不做"** —— `MERGE_AB_t900` 测出正收益却
   既没落地也没记原因，变成"记忆 vs 代码"的悬案，同一个坑被重踩一遍。**没有记录的结论等于没做。**

---

## 5.5 数字补充复核（supplement，2026-09-09）

**问题**：Validator 机器数字层原把"被引证据缺失、但全篇某 chunk 有该数"直接放行
（防跨块综合误报）。洞：**数值存在 ≠ 归因正确**——论文可有多处 6000，此 6000
未必是答案所指的那个（cf63a4 的 7.5 天 vs 别处章节 7.5）。

**设计**（存在性检查 ≠ 归因判定，分层铁律的延伸）：
- Validator 只做确定性的存在性检查，**不引入 LLM 判断**（检查层拒绝不确定性）；
- "别块有数"不再是 pass 依据，只是**触发补充检索的信号**；
- 把命中该数的 chunk 作 Supplementary Context，连同原问题/初稿喂回 **Generator**
  复核一次——由它（看得到完整上下文）裁决"此数是否彼数"：
  - 支撑 → 保留并修正引用到真正含该数的条目；
  - 同名不同义（章节 7.5 vs 7.5 天）→ 删除/改写为"无法确认"，宁缺毋编。
- 复核仅一次，Generator 判定无需改 → pass；改 → repaired；结果再过机器复检。

**代码落点**（默认 GATE=1 才生效，主链路零改动）：
| 模块 | 改动 |
|---|---|
| `components/numbers.py` | Normalizer 独立组件（parse_number/scan_numbers，覆盖中英单位/科学计数/中文数词，round 8 位防精度抹零，w 词界防 "500 words" 误判） |
| `components/validator.py` | `validate_numbers → (issues, supplements)` 溯源命中 chunk；`check/gate` 透传；gate 增 supplement 分支（无 LLM 判断） |
| `components/repairer.py` | `supplement()`：初稿含数子句摘取 + 命中条目定位 → Generator 裁决 → 重对齐引用 + 机器复检 |
| `graph/__init__.py` | gate 接 `supplement` 回调、`extra_chunks` 传结构化 chunks |

**A/B 验证（100 题 ab_v3base 同题单，同主图输出分臂，同外部裁判）**：
- 真实触发率 **1/100**（cf63a4 的 6000）——L3 硬引用已让绝大多数数字落在被引证据，
  跨块漏引是少数；成本增量可忽略。
- 触发样本（cf63a4 单题 A/B）：旧=静默放行 pass(4 分)，新=复核 repaired(5 分)——
  Generator 把引用从错块 [1] 修正到真数所在块 [2]，裁判 +1。
- 强场景构造验证（模拟同名不同义）：7.5 天 vs 章节 7.5 → Generator 识别并删除断言；
  3600万 vs 36 million 同义 → 引用修正到真块；不误伤"被引确有该数"的答案。
- 未触发 99 题 A/B 输出逐字一致（changed=0）——补复核对默认行为零扰动。

**结论**：这是"罕见但严重"的兜底——触发率 ~1%、成本近零、触发时能修好机器
放行的归因错误；Validator 保持确定性（信号）与 LLM 裁决（Generator）分层干净。

### 5.6 修复回路（Self-Refine / CRAG）A/B（2026-09-09，10 道历史 V fail 题）

**背景**：Validator gate 检出 high 后走 `repairer.repair`：generation 型
（off_topic/contradiction/vague）→ Self-Refine（带质检意见重写）；
evidence 型（number/citation/unsupported）→ CRAG（缺口定向检索重答）。
此前从未量化过"修得怎么样"，这次补了对照。

**A/B（10 道 V fail 题：GATE off vs on(REPAIR_MID=1)，同裁判双打分）**：
- judge pass：off 1/10 → on **2/10**（Δ+1）；平均分 2.80 → 2.70。
- on action：repaired 5 / pass 4 / fallback 1。
- 逐题：2 题 citation 越界 repaired 后 +1（其一 3→4 翻绿）；1 题 fallback 3→1（砸）；
  其余持平。**修好一半、砸一小半，净 +1**。

**结论**：
1. **修复是"兜底保底"，不是"主力提升"**——对证据根本性缺失的 V fail 题，
   gate 只能把"能修的引用越界"捞回一部分，平均分不涨反微降。
2. **这批样本全走 CRAG**（issues 全为 citation/number 机器件）——
   **Self-Refine 仍无独立实测数据**（需专门构造 generation 型失败样本才能验）。
3. 与 5.5 一致：Validator/修复器在默认链上低频兜底；单篇 pass 大头在检索
   召回与输入质量（MinerU），不在检查器。**此步到此收敛，转向检索优化。**

### 5.7 闸门误杀修复：原文自带引用 + 派生数（2026-09-10，MinerU hard 4 题定位）

**背景**：MinerU 端到端 hard 自测（10 篇 × 3 题）26/30，4 道非 pass **全部 `action=fallback`**
（"未通过内部事实校验"）。`_gate_probe.py` 打印闸门判定后确认：4 题**答案本身正确**
（与 gold 一致），是 Validator 机器层两条**有界假阳性**：

| 模式 | 题 | 闸门判定 | 真相 |
|---|---|---|---|
| 引用越界 | mh-29290 q1/q2/q3 | `citation`：`[29]` 越界（共 12 条）→ HIGH | `[29]` 是**论文自带参考文献序号**（原文 `existing research [29]`），MinerU 原样解析进 chunk，模型忠实照抄——不是悬空的上下文引用 |
| 派生数 | mh-28447-q1 | `number`：`30.2` 无证据支撑 → HIGH | `30.2 = 66.0 − 35.8`（Table 1 两值均有据），且 SYSTEM 准则 8 明确要求算出差量——闸门与自己的 prompt 打架 |

**修复（`components/validator.py`，仅放宽、均有界）**：
1. **原文自带引用豁免**：越界编号若在被引证据原文中以 `[n]` 原样出现 → 判为论文参考文献序号，
   不报越界。只对越界号生效（越界号不可能是我们自己的合法引用），不会掩盖真悬空。
2. **派生数豁免**：答案同时给出操作数与计算结果、且两个操作数**均有证据支撑**时，
   差/和（`|a−b|`、`a+b`，相对容差 1e-3）视为合规推算，不判编数。

**验证**：
- MinerU hard 30 题：26/30 → **30/30**（均分 4.13 → 4.63）。4 题全回收（5/5/5/4）；
  其余 26 题本就无 HIGH（`action=pass`），逻辑上不受影响。
- 防幻觉负向组 20 题：**20/20**（0 编造 / 0 误断言），零回归。其中 03718-N1 首次被自动判据
  漏判（约答"论文中**未出现**…"，而 `_run_neg.ABSENCE_RE` 词表缺"未出现"）→ 补词表后
  离线重判为 OK-REFUSE——属**测试侧判据缺口，非系统缺陷**。

**关联**：§5.6 曾发现"gate 只能把'能修的引用越界'捞回一部分"——本次揭示其中一类"越界"
本就不是缺陷（原文自带引用），闸门对它的 fallback 是纯损失，解释了修复回路收益有限的一部分。

**待办**：QASPER 233 全量重跑（含 gate）量化该修复在外部基准上的净收益。


## 6. MinerU 解析后端接入记录（2026-09-09）

### 6.1 形态与启动
- 独立环境 `.venv-mineru`（Python 3.12，MinerU 3.4.5，`mineru[all]`，torch cu128），
  **4050 GPU**（6GB），backend `-b pipeline`，pipeline 模型 <1GB。
- 产物：`mineru -p <pdf> -o out_mineru/<stem> -b pipeline` → `auto/<stem>_content_list.json`
  （3.x schema：`type/text/text_level/bbox/page_idx`；table 带 `table_body(html)`/caption；
  chart/image 带 caption；list `sub_type=ref_text`=参考文献）。
- **判定**（`agents/document_cache.mineru_status`）—— 四态**必须分清**，
  **别把"没跑过"叫成"降级"**（2026-09-22 修过一次口径错误）：

  | 状态 | 含义 | 处置 |
  |---|---|---|
  | `ok` | 产物可用 → MinerU 作骨架 | 正常态 |
  | **`pending`** | **尚未摄取**（没跑过 ingest）| **待办**：跑 `cli/run_ingest.py <pdf>` 即可。正常流程（`worker._lane_mineru`）本来就会**先跑 MinerU** → 不该出现。**不是故障、不是降级** |
  | **`degraded`** | MinerU **试过但失败**（执行报错/超时、产物损坏、产不出 chunk）| **降级 pymupdf 容灾** + **必须告警** + 写进 `PaperReport.degraded` |
  | `disabled` | `PAPERPILOT_USE_MINERU=0`（骨架）或 `PAPERPILOT_MINERU=0`（执行）| 配置选择 |

  **"降级与容灾"的定义**：MinerU **试过、但失败了** → 为不让全链垮掉，用 pymupdf 顶上。
  `process_pdf` 对两者给**不同措辞**（`ⓘ` 提示该摄取 / `⚠️` 告警真降级）。
  `PAPERPILOT_USE_MINERU=0` → 骨架固定 pymupdf（容灾 / A-B 对照）。
  与 `PAPERPILOT_MINERU`（**跑不跑** MinerU，默认也开）是**两件事**，别混。
- **两条独立的路**（2026-09-22；切块组件重建为 MinerU 之后接上的容灾）：

      pdf ──► 路由 ──┬── **默认路** `_chunks_via_mineru`  ──► Chunk[]   （只读 out_mineru/）
                     └── **备用路** `_chunks_via_pymupdf` ──► Chunk[]   （只读 assets/papers/）
                                     ↑ 默认路任何环节失败 → 自动走它

  **"不相交"是硬要求**：默认路只读 `out_mineru/`、备用路只读 PDF 且**不做表格注入** →
  不会出现"半条 MinerU、半条 pymupdf"的混合产物。实测 5 种情形：

  | 情形 | 走哪条 | 状态 | chunk 数 |
  |---|---|---|---|
  | 真产物 | 默认路 MinerU | `ok` | 27 |
  | 产物损坏（非法 JSON）| 备用路 pymupdf | `degraded` | 24 |
  | **产不出有效 chunk**（空 content_list）| 备用路 pymupdf | `degraded` | 24 |
  | 无产物目录（没跑过）| 备用路 pymupdf | `pending` | 24 |
  | 恢复真产物 | 默认路 MinerU | `ok` | 27（**可自愈**）|

  ⚠️ "产不出有效 chunk"这一档**别漏**：旧实现返回空 list，而调用方用 `is not None` 判断
  → **返回空 chunk 而不降级**，与 `mineru_status` 判 `degraded` 自相矛盾
  （那篇会"有 chunk 但一条都搜不到"）。已修。
- **`degraded` 的两种来源都要认**（2026-09-22 补）：① **执行失败**
  （`ingest.json` 的 `mineru.status=="failed"`，**此时根本没有产物目录** —— 只看目录会把
  "跑了但失败"误报成 `pending`"还没跑"，降级提示永远不出现）；② 产物层问题（损坏/不可读/产不出块）。
- **问答闸门撤了**（2026-09-22 契约变更）：`ingest.qa_blocked_reason` 以前
  `mineru.status=="failed"` → **整篇问答不可用**。问题：报告链**早就在降级**（走 pymupdf），
  只有问答被拦 → **"默认路失败 → 降级到备用路"这条设计永远走不到**。现在失败 → **降级放行**
  （表值会缺，由 `mineru_status` / `PaperReport.degraded` / `QAState.parse_degraded` 标记）。
  仍拦的只剩**摄取进行中**（产物马上就好，不是错误）。
- **worker 索引 lane**：`failed` **也照建索引**（旧写法 failed 就跳过，理由是"问答会被闸门
  拦住、建了没人用" —— 闸撤了，那条理由不再成立；不建就等于"降级了却没有向量可用"）。
- **代价**：worker 由"MinerU ∥ 报告链"变成**串行**（报告链必须等 MinerU 跑完）
  → 单篇墙钟 ≈100s → **≈157s**。
- **已知未修**：MinerU 会**误标**标题 —— `_is_heading()` 只看 `type==text and text_level`，
  实测 1 例正文句被标 `lv=2` → 多一个假章节。建议加**轻校验**：`lv=2` 但
  **无编号 + 不在特殊词表（Abstract/References/…）+ 以逗号结尾** → 判为可疑。
- **不做的事**：不再要求 MinerU 的 `title_path` label 格式"仿 pymupdf"
  （同构约束是历史包袱；但**层级推断本身必须保留** —— MinerU `text_level` 只有 1/2 两档，
  不靠标题编号推 depth 的话树会塌成一层平表）。

### 6.2 代码落点（**MinerU 为默认骨架**；`PAPERPILOT_USE_MINERU=0` 退回 pymupdf）
| 模块 | 职责 |
|---|---|
| `tools/mineru_bridge.py` | content_list → 同构 `Chunk[]`（表格 HTML→markdown 表、公式 LaTeX、标题按编号重建层级、label 与 chunker `"L{d} {no} · text"` 格式对齐）；`title_from_mineru_dir`；`extract_figures_from_mineru`（report 图表一览） |
| `agents/document_cache.py` | `ordered_chunks` MinerU 分支 + `current_source(pdf)` → `qasper/mineru/pymupdf` |
| `pipeline.py` | `_source_chunks` 委托 `document_cache.ordered_chunks`（报告/QA 同源同 chunk_id）；标题/figures 按源分流；claims payload 打 `source`；`process_pdf` 源不一致 → 自动整链重建（`skip_llm` 时拒绝并报错） |
| `tools/analyzer.py` | `mask_table_rows` 增挡 markdown 管道表行（防 MinerU 表值拼伪 claim） |

### 6.3 实证结论（含边界，重要）
- **QASPER 评测（qasper-train-v0.3.json）无 PDF/无表格**：`full_text` 只有正文文本，
  `figures_and_tables` 仅 `{file: png, caption}`，png 本体需官方 images 包另下。
  故"答案只在 Table/Figure"类题（1000 题残差 ~80-110）对**任何只用 json 正文的系统**都无解
  ——属输入通道边界，不是系统缺陷，MinerU 对纯 JSON 语料无用武之地。
- 本地**文本层表格 PDF**：pymupdf 能抽出碎片，MinerU 增量="碎片→结构化"
  （可抓到列错位/编值：HyGRAIL Table1 pymupdf 把 ρ/P/C 张冠李戴、编 C=3.40），但净增益需批量统计，单篇不稳。
- 本地**复杂版面**（图内文字/双栏/页眉）：MinerU 稳定优势——图内标注文本不进正文
  （HyGRAIL p4 pymupdf 吐出流程图标签 "0/100/Final Prediction"，MinerU 归 chart/image）、
  双栏阅读序正确、页码/arXiv 边栏结构化剥离。**这是 MinerU 相对 pymupdf 的主价值轴**。
- 验证：env off 下 3 篇既有 claims 与新 `_source_chunks` 逐 chunk 一致（零回归）；
  env on 全链（08696）claims/figures(6→11)/title/report 正常，source 打标与 stale 保护生效。

### 6.4 遗留（后续按需）
- `cli/run_claims/run_figures/...` 等命令行入口仍直连 pymupdf（默认模式不受影响）。
- 前端坐标化高亮（bbox 进 cites/report）独立课题，暂缓——前端现用 pdf.js 视觉行+文本模糊匹配，普通正文已够准，表格/公式块是该方案的盲区。
- MinerU 表格 chunk 文本含 markdown/LaTeX → evidence 回核 hit 率可能轻微波动（mask 已防伪 claim），量级未测。

### 6.5 输出闸门：零引用答案不再"静默放行"（2026-09-13，外部审查修复 A 档）

**问题**：`gate()` 用 `cites` 构造 `entries`，故**零引用 ⇒ entries 空 ⇒ 原语义体检被跳过**，
机器层只剩一条 LOW，而 `gate` 只对 HIGH 拦 → 一条"纯文字编造 + 零引用"的答案会 `pass`。
（实测边界：带**显著数字**的编造会被数字层兜住 → `fallback`；漏的只是无数字的纯文字编造。）
**真实分布**：503 题 + 72 题里零引用共 6 条，**全部是闸门自己的兜底话术**，无一由模型产出；
`level=L0` 的 70 条零引用 0 条 → 属**防御缺口**而非正在漏水，但"prompt 准则 14 要求硬引用、
闸门却把零引用当可忽略"的口径不一致必须修。

**改法（只加可观测性，不改 gate 动作）**：
1. 零引用独立成 `no_citation` 类型；
2. 机器判据 `_substantive_claim`（剔除拒答话术后看数值/长度）→ 含实质断言 **MID**、纯拒答/过短 **LOW**；
3. 零引用时**不再跳过检查**：改跑 `_llm_uncited` **格式体检**（无原文 → 只判"该不该有引用"，
   不判真伪；自带守卫，非实质断言直接返回空、不花调用）；
4. `no_citation` **不进** MID 触发名单 → 行为不变（仍只对 HIGH 拦），`REPAIR_MID=1` 下也不会变成拒答；
5. 运行记录新增 `validator_action` / `issues` / `no_citation` / `no_citation_substantive`；
   `/api/ask` 回传 validator 摘要，前端新增"答案自检（AI 复核）"提示条（此前 **issues 根本没送前端**，
   等于"mid → 前端标注"这条设计从未落地）。

**验收**：`qa/recall/_selftest_validator_20260913.py`（机器侧 12/12；`PP_LLM=1` 加测真实格式体检）。
（2026-09-13 迁移进 pytest 统一套件：`tests/test_validator_offline.py`；`-m local` 跑真实体检。）
**台账**：`qa/review/RESPONSES_20260913.md`。**未做（B 档）**：按 `route` 分流
（`global_retrieve` + 零引用 + 实质断言 → 一次 repair 强制补引用）。

### 6.6 论文身份 = **内容指纹**，不是文件名（2026-09-13，外部审查修复 A+C 档）

**问题**：全链路产物（claims/summary/skeleton/figures/overview/guide/report、gvec/cvec、
`out_mineru/<stem>/`、`ingest.json`）都以 `Path(pdf).stem` 为键，**没有任何一层记内容指纹**。
复现（`qa/review/_dupname_repro_20260913.py`）：
- 把 `A.pdf` 换成 B 的字节（文件名不变）→ **解析层读 B、产物层给 A** → 状态混合、无告警；
- web `/api/report` 上传"文件名=A、内容=B" → **返回 A 的旧报告且不写盘**（用户拿到别篇论文）。
- **评测测不到**：QASPER 走虚拟名 `qasper_<pid>.qpdf`，身份由数据集给定，不存在"同名不同内容"。

**改法（A + C）**：
1. 新增 `paperpilot/paper_identity.py`：`fingerprint()`（sha256+size+mtime_ns，带缓存）、
   `check()` → `ok / stale / adopt / unknown`、`strict()`（`PAPERPILOT_PDF_ID_STRICT=1`）。
2. `pipeline.process_pdf` 与 `ingest` 各加**身份门**：stale → 整链重建；`--skip-llm` 下**直接报错**
   （不允许用不属于这份 PDF 的产物装配）；`run_mineru` 同样校验，同名换内容时 MinerU 也会重跑。
3. **历史产物迁移**（关键，避免一次性废库）：无指纹记录时按 **mtime 关系**判定——
   产物比 PDF 新 → **收编**并**补写指纹**（之后严格 sha 比对）；PDF 比产物新 → 判 stale。
   全库实测（66 篇）：32 篇收编 / 34 篇无产物 / **0 篇被迫重建** → 零爆炸半径。
4. `document_cache` 的进程内 `lru_cache` 键加入**文件版本**（`size:mtime_ns`）：
   同进程内换文件不再返回旧解析；同时保留 `ordered_chunks.cache_clear()` 等兼容属性。
5. **web 上传按内容判定落点**（`web.plan_upload`）：同名同内容 → 幂等复用；
   同名**不同内容** → 另存为 `<stem>__<sha8>.pdf` 当**新论文**（原文件一个字节不动），
   并在响应里带 `upload_note` 明确告知。

**验收**：`qa/review/_selftest_identity_20260913.py`（19 项全绿）+ 复现脚本转为验收脚本
（2026-09-13 迁移进 pytest：`tests/test_identity_cache.py`，并补了缓存命中/失效与向量指纹用例）
（两条路径均"被拒/另存"，不再静默复用）。**未做（B 档）**：产物**内容寻址**目录
（`by_hash/<sha12>/…`）——需迁移 6900+ 产物并改所有路径构造，留待确需多版本共存时再做。

