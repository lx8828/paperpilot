# RAG 组件化设计笔记（2026-09-09）

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
论文 PDF ─► [DocumentSplitter] ─► [Router] ─► [QueryRewriter] ─► [Retriever(混合)]
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
- 待办：扫描件/复杂版面/表格 → **MinerU 接入**（解析层对照实验，改前改后各跑一次评测）。

### ② Router（路由/难度分类）—— 🟡 有隐式逻辑，需显式化
- 职责：判断问题类型 → 决定路径（全貌直答 / 局部检索 / 多维分解 / 拒答）。
- 现状：v3 用 `judge_l0` 的"够/不够"二分类当隐式 Router（全貌 vs 需检索），表现达标。
- **证据/警告**：LLM 预测作路由是危险做法——L2 时代 judge 判够精度仅 ≈61%（
  `JUDGE_DIAGNOSIS_20260908.md`）。**L0 的二分所以可靠**：输入完整(overview+core_points)、
  判断"有没有提到"而非"证据够不够拼答案"、判错可降级不致命。
- 默认配置：Router 输出带**降级链**（每条路判不够可升下级检索），绝不做"分类定生死"的硬路由。
- 多篇时代：主题识别/多论文归并需要更强的 Router。

### ③ QueryRewriter（查询改写）—— ✅ 代码有，**默认关闭（实测负收益）**
- 现状：`_rewrite_queries`（生成 2-3 变体；向量+BM25 多查询 RRF 融合），`PAPERPILOT_QUERY_REWRITE=1` 开启。
- **证据**：2026-09-07 A/B（40 题同裁判）：改写 ON pass 9 vs OFF pass 12 —— 单篇负收益
  （救回 2 条但弄坏 5 条），故默认关闭。
- 默认配置：**OFF**。多篇/语料库召回面大、单查询带偏风险高时再开，届时重新 A/B。
- 若开启：重写须保留关键实体为锚（现规则已有）。

### ④ Retriever（混合检索）—— ✅ 已有（真正的混合检索在这）
- 现状：
  - `ChunkIndex.search_hybrid` = 向量 + BM25 **RRF 融合**（单查询；专名/术语精确匹配互补）→ L3 主检索，top12。
  - `ClaimIndex` = claims 主张向量检索（v3 已不再单列一层，多篇/主题时可作检索源复用）。
  - top12 覆盖证据 ~85%（调过 top8→top12）。
- **证据**：朴素 RAG(B1)≈我们(B2)（`compare_20260907_191606.md`：49 vs 51）→ 检索件不需要花哨；
  真正的检索短板在**数值/表格块**（L3 捞不到表值），靠 MinerU/输入质量解，不是加检索复杂度。
- 默认配置：hybrid + top12；改检索先跑 C 层 Recall@k（多篇再建真值集）。

### ⑤ Reranker —— ❌ 未做（单篇无收益预期，空壳预留）
- **证据/判断**：RRF 融合已是弱重排；交叉编码/LLM 重排在"候选面小（top12 内几乎全相关）"的
  单篇场景理论上限低（朴素 RAG≈我们）。QASPER 净亏题的根因是"答案块没进 top12"而非"排序不精"。
- 默认配置：接口预留（`components/reranker.py`），**多篇/大召回面时再实现与 A/B**。

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
| 多篇/主题级 | QueryRewriter ON（重新 A/B） | 单篇负收益是"检索面小"所致 |
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
    query_rewriter.py           #   _rewrite_queries（默认关）
    retriever.py                #   ChunkIndex/ClaimIndex 统一检索入口
    reranker.py                 #   空壳（接口预留，多篇实现）
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
