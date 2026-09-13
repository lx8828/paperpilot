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
- MinerU（2026-09-09 接入，细节见 §6）：扫描件/复杂版面/表格解析能力就绪；开启
  `PAPERPILOT_USE_MINERU=1` + `out_mineru/<stem>/` 产物存在即生效，默认关（pymupdf 不变）。

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
- **开启条件**：`PAPERPILOT_USE_MINERU=1` **且** `out_mineru/<stem>/` 有 content_list；
  否则自动回退 pymupdf（`agents/document_cache.current_source` 统一判定）。

### 6.2 代码落点（全部加法适配，默认 pymupdf 行为零改动）
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

