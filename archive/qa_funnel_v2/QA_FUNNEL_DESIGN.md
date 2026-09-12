# QA 检索漏斗设计（L0→L3 逐级加料）

> 2026-09-04 评审修订 v1.1。本文是问答层检索架构的唯一权威描述。
> 与 [DEVELOPMENT_LOG.md](./DEVELOPMENT_LOG.md) 的"阶段记录"互补：那里记"为什么这么做"，这里记"现在该怎么做"。

---

## 1. 现状问题（为什么要改）

当前图（`graph/qa_graph.py`）是"检索一次 + 判断一次"：

```
retrieve_claims（top12 claims + overview 常驻）
  → judge：够 → answer / 不够 → pull_chunks（≤2 个孤 chunk）→ answer
```

七个已知缺陷：

| # | 缺陷 | 现象 |
|---|---|---|
| D1 | **L1(overview) 名存实亡** | overview 只是 retrieve 顺手返回的字段，所有问题都先付 embedding 检索；"论文讲了什么"这类问题本不该唤醒模型 |
| D2 | **候选集逐层收缩** | judge 只能从 top12 claims 携带的 ≤12 个 chunk_id 里挑；一个 claim 只绑 1 个 chunk；Appendix 等"无 claim 区域"直接失明 |
| D3 | **升级粒度太小且不复查** | pull 最多 2 个孤 chunk，无邻域扩展、无 section 概念；拉完不再判够不够 |
| D4 | **无诚实收尾** | 拉不到相关内容时 judge 仍会硬答/编造，没有"我不知道"的出口 |
| D5 | **单圆心脆弱** | L2 若只以 score top1 chunk 为圆心，圆心选错或答案分散两节即漏 |
| D6 | **Judge 一套模板** | L0 的"够"（概述覆盖全貌）与 L2 的"够"（能引原文作答）标准不同；混用导致浅层误判不够（白检索）或深层误判够（虚答） |
| D7 | **gap 不驱动** | gap 只存不读，未用于引导下一层取料方向，白白浪费一次判断的产物 |

---

## 2. 目标架构总览

**核心转变：从"预分类 router"回归"证据驱动的逐级加料"。**

- ❌ 不引入问题分类器（router）：它是**额外的推理任务**，省下的只是毫秒级检索，还要担分类错误风险。
- ✅ 每层 **Judge 只回答它有资格回答的问题**："凭手上已有的证据，能不能答？" 不能，就按固定顺序加料。

```
question
  │
  ▼
[L0] Report：title + overview + core_points      ← 零检索、零 embedding
  │
  ▼
Judge0（证据：L0）
  │ 够 → Answer ✔
  │ 不够 ↓
[L1] Claims：embedding 检索 top12               ← 此刻才唤醒 bge-m3
  │
  ▼
Judge1（证据：L0 + L1）
  │ 够 → Answer ✔
  │ 不够 ↓
[L2] Section：命中 chunk 为圆心，同节增量扩展    ← 半径 1→2→3…
  │     每一轮扩完 → Judge
  │ 够 → Answer ✔
  │ 不够且已达阈值/扩到节边界 ↓
[L3] Global Chunk：独立全文检索 topK             ← 不继承 L0~L2 任何证据
  │
  ▼
Judge3（证据：L3 检索结果本身）
  │ 够 → Answer ✔（只用本层干净 chunks）
  │ 不够 → "我不知道，缺 XX"（诚实收尾）
```

**原则（本次设计的六条铁律）：**

1. **取料是确定性动作，判断才用 LLM。** 加料顺序固定、机械执行；Judge 每层都握着实证据说"够不够"，不做它做不了的微观定位。
2. **每层判定基于本层干净证据。** 下钻不清空旧料，但旧料只作为背景；绝不让已证伪的错误线索（如 L2 放弃的 section）污染下一层判断。
3. **L3 是独立保底，不是 L2 的延续。** 忘掉前面一切，用**原始问题**重新全文检索，防止错误传导。
4. **诚实收尾优先于编造。** L3 仍不满足 → 明说缺什么，不硬答。
5. **Judge 按层分模板。** 每层"够"的标准不同（见 §3.5），模板必须各自说清"证据是什么 / 什么算够 / gap 怎么描述 / 节选项从哪来"。
6. **gap 必须驱动取料。** Judge 判不够时产出的 `target_sections` 是下一层取料的输入（L1→定向 L2 圆心；L3→诚实收尾措辞），不是只存不读的死字段。

---

## 3. 各层详细定义

### L0 · Report 层（零检索）

| 项 | 内容 |
|---|---|
| 取料 | `report.overview` + `report.core_points`（g1/g2…，自带 gid 可溯源） |
| 检索 | **无**（不加载 embedding 模型） |
| 覆盖 | "论文讲了什么 / 核心贡献 / 主要结论"类全貌问题 |
| 退出 | Judge0 判够 → Answer（cites 锚在 core_points 的 gid） |
| 成本 | 1 次 LLM（Judge0） |

> 实现要点：embedding 模型必须**懒加载**——L0 能答的问题，进程里永远不出现 bge-m3。

### L1 · Claims 层（embedding top12）

| 项 | 内容 |
|---|---|
| 取料 | `ClaimIndex.search(question, top_k=12)` → 12 条 ClaimHit（含 chunk_id、evidence、page） |
| 检索 | 1 次 embedding（首次唤醒模型，之后缓存秒级） |
| 覆盖 | 主题/概括级问题，claim 浓缩够用、不必看原文 |
| 退出 | Judge1 判够 → Answer（cites 锚在 claim 的 evidence_quote/page） |
| 成本 | embedding 一次 + 1 次 LLM（Judge1） |

### L2 · Section 层（定向增量扩展，非整节拉取）

| 项 | 内容 |
|---|---|
| 圆心 | **不是 score top1，而是 Judge1 的 `target_sections` 有序候选**：L1 命中按 section 分组 → Judge1 从"命中 claims 出现的 section 名单"里按"最可能藏答案"降序输出（最多列 4 个）→ 代码截断 `MAX_CENTERS=2` 并按序逐个尝试 |
| 去重 | **预算账本制**：L2 全程维护 `pulled: set[chunk_id]` + `budget_used`（按新增唯一 chunk 计）。每轮扩展先查 pulled，拉过即跳过不占预算；圆心列表先按 section 去重（同节只留一个，防 LLM 重复），下一圆心 chunk 已全在 pulled 则直接跳过不空转 |
| 多圆心 | 预算共享（累计 6 chunk / ~12k 字符）：顺序 = 优先级，先扩主圆心，判够即停；不够才换次圆心；圆心间结果**分窗隔离**，避免跨节混料 |
| 扩展 | 同 section 内前后各 `+1 → +2 → +3…`（每轮扩完 → Judge2） |
| 覆盖 | 答案在某一节、且与命中 chunk 连续/相邻（定义、公式推导、实验细节）；答案跨两节时两个圆心分头试 |
| 出口 | ① 某轮 Judge2 判够 → Answer；② 当前圆心半径达阈值（±3）或扩到 section 边界仍不够 → 换下一圆心；③ 候选圆心全试完 / 预算耗尽（`budget_used ≥ 6`）→ 放弃 L2；④ Judge1 给 0 个候选 → 直接跳 L3 |
| 成本 | 每轮 1 次 LLM（Judge2）。**多数问题在 L0/L1 已退出，L2 是少数派** |

> 大章节为什么不能整节拉：20+ chunk 的一节整拉 = 预算爆炸 ≈ 退化全文。
> 增量扩展让"答案就在这一段"的小问题用最小上下文解决，试到阈值不对就放弃——**节不对，拉再多也没用**。
> 单圆心为什么不够：score 高的是"最像的 claim"不是"答案所在地"；且答案分散两节时单圆心必漏。用 LLM 读 claim 语义定向选节，比 embedding score 猜可靠。
> **取几个/去重为什么归代码不归 LLM**：LLM 只输出有序候选（排序是它擅长的）；个数（截断 MAX_CENTERS）、同节去重、重叠去重、预算计数全是确定性逻辑，可回归可调参。

### L3 · Global Chunk 层（独立保底）

| 项 | 内容 |
|---|---|
| 输入 | **只有原始 question**（不用 L2 的 section 判定、不用 L1 的 claim 候选、不用 L0 的 overview） |
| 检索 | `ChunkIndex` 对全文 20~44 chunk 语义检索 topK（如 8） |
| 覆盖 | L2 指错节的兜底、跨节综合问题、claim 从未覆盖的角落（Appendix/表格/公式） |
| Answer 上下文 | 只用本层检索到的 chunks，每块前拼自带 section 标签（title_path），让模型知道"在答哪一节" |
| 内部不做邻域 | 邻域是 L2 的职责；L3 满图找分散相关块，靠语义 topK 本身 |
| 退出 | Judge3 判够 → Answer；不够 → **"我不知道，缺 XX"** |
| 成本 | 1 次全文 encode（首次 ~10s，缓存后毫秒）+ 1~2 次 LLM。**到达率低，是质量保险不是日常路径** |

### 3.5 Judge 分模板（每层"够"的标准不同）

一个 prompt 工厂，四层各配四要素。不能共用一套模板（D6）。

| 层 | 证据清单 | "够"的标准 | gap 该写什么 | target_sections 从哪选 |
|---|---|---|---|---|
| L0 | title + overview + core_points | 全貌类问题（讲了什么/核心贡献/主要结论）已被概述覆盖 | 缺哪类信息：方法细节 / 结果数值 / 背景对比… | 无（L0 不选节，判不够直接进 L1） |
| L1 | L0 + top12 claims | 主张文本已能支撑**主题级**回答 | 需正文**精确表述**（数字/公式/表格/附录细节） | 命中 claims 出现的 section 名单（选 1~2 个最可能藏答案的节） |
| L2 | L0 + claims + 已拉正文窗口 | 能从 chunk **原句直接作答并引用**（cites 有原文锚） | 哪部分仍没覆盖（答案在别处/窗口边缘被截断…） | 窗口外邻近 section（多圆心分头试的下一目标） |
| L3 | 全文检索 chunks（无任何上层料） | 检索结果能作答，**或可判定论文确无此内容** | 最终缺什么（措辞直接喂给"我不知道"） | 无（L3 之后没有下一层取料） |

> L0 误判"不够" → 白唤醒模型白走 L1（浪费）；L2 误判"够" → 无原文支撑的虚答（更糟）。分层模板是质量关键，不是可选项。

沿用 `QAState`（`agents/state.py`）的数据契约，字段微调：

```python
class Verdict(TypedDict):
    enough: bool               # 本层证据够不够答
    target_sections: list[str] # 不够时：下一层该去哪找（由本层从证据中选，驱动取料）
    gap: str                   # 不够时缺什么（人话，供下钻/诚实收尾措辞）

class QAState(TypedDict, total=False):
    question: str
    pdf: str
    # L0
    overview: str
    core_points: list[dict]    # L0 answer 的 cites 锚
    # L1
    retrieved: list[ClaimHit]  # L1 命中（按 section 分组，供 target_sections 选）
    # L2
    chunks: list[ChunkText]    # L2 定向增量扩展累计的正文（分窗隔离，作背景）
    # L3
    l3_chunks: list[ChunkText] # L3 独立全文检索结果（与 L2 隔离，不混合）
    verdict: Verdict
    answer: str
    cites: list[Cite]
    debug: dict[str, Any]      # 记录走到哪层：route 履历 + 每层 target_sections
```

**关键隔离点**：
- L3 的 answer 上下文只含 `l3_chunks`；`chunks`（L2 废料）不得混入 L3 的 Judge/Answer 输入。
- L2 多圆心之间**分窗隔离**：跨节的 chunks 不拼在一起当背景，避免"两节废料互相稀释"。

---

## 5. 图拓扑（LangGraph）

```
retrieve_l0 ─→ judge_l0 ──enough──→ answer_l0
                 │ not_enough
                 ▼
              retrieve_l1 ─→ judge_l1 ──enough──→ answer_l1
                                │ not_enough
                                ▼
                    expand_l2 (循环: judge_l2)
                                │ enough → answer_l2
                                │ 阈值/到边且不够
                                ▼
                        search_l3 ─→ judge_l3 ──enough──→ answer_l3
                                              │ not_enough
                                              ▼
                                         answer_unknown（诚实收尾）
```

L2 的内部循环用 LangGraph 自环（self-edge）+ 条件边实现：
- `expand` 按 target_sections 顺序取圆心，每次半径 +1（或由 L2 出口条件判断是否已到边/多圆心试完）
- Judge2 判 enough → answer_l2；判不够且未达阈值 → 再 expand；达阈值/到边/多圆心试完 → search_l3
- Judge1 若给不出 target_sections（无把握）→ 直接 search_l3（图里加一条 bypass 边）

---

## 6. 阈值与预算（首版默认值，可调）

| 参数 | 默认 | 说明 |
|---|---|---|
| L1 topK | 12 | 不变 |
| MAX_CENTERS | 2 | Judge1 有序候选截断数；给 1 个即退化单圆心 |
| L2 最大半径 | ±3 | 单圆心第 4 轮起判放弃/换圆心 |
| L2 累计 chunk 上限 | 6 | 按 **pulled 新增唯一 chunk** 计，与半径互为保险 |
| L3 topK | 8 | 全文独立检索 |

**QA 层禁止二次截断 chunk 文本（铁律，教训源自 Run1）**：
- chunk 长度在**分块期**已由 `chunk_document(max_len≈4000)` 段落原子切分控制，单块 ~4000+。
- QA 取料（L2/L3）与 judge/answer 展示**一律给全文**，不得再 `text[:N]`。
- 曾出现两处拍脑袋截断（judge 展示 `[:900]`、取料 `[:3000]`）导致答案落在被砍掉的 chunk 尾部 → judge 误判 unknown（Run1 真漏检根因），均已移除。
- "省预算"只能通过**控制块数**（topK/圆心/半径/预算账本），不能通过**砍单块内容**——块数是设计参数，砍内容等于直接丢证据。

---

## 7. 验收测试问题集（改完必须全过）

| 问题 | 期望路径 | 验证点 |
|---|---|---|
| 这篇论文讲了什么？ | **L0 直答** | embedding 未被唤醒（debug 记录） |
| 核心方法/贡献是什么？ | L0 或 L1 | cites 可溯源 |
| Planner 用什么模型？（定义类） | L2 命中点 ±1~±2 | 相邻 chunk 上下文被带出 |
| 第 4 节那个衰减系数是多少？ | L2→L3 | 数字级细节能找到原文 |
| Appendix 里评测用了哪些指标？ | L2(若 claim 见附录) 或 L3 | section-level 拉齐，不再 top1 |
| 这篇用了哪些数据集？（跨节汇总） | **直达 L3** | 分布式答案能聚合 |
| 论文里根本没提的东西 | L3 → "我不知道" | 诚实收尾，不编造 |

debug 中必须可见：`route: ["L0","L1","L2","L3"]` 履历 + 每层 `n_retrieved/n_chunks`。

---

## 8. 代码落点

| 文件 | 动作 |
|---|---|
| `src/paperpilot/agents/embedder.py` | 新增 `ChunkIndex`（与 ClaimIndex 同构，`.cvec.npy` 缓存）；确认模型懒加载 |
| `src/paperpilot/agents/nodes/judge.py` | 改为 prompt 工厂：四层分模板（§3.5）；输出 `{enough, target_sections, gap}` |
| `src/paperpilot/agents/nodes/pull_chunk.py` | 改造为 `expand_l2`（target_sections 定向圆心 + 半径 + 分窗 + 预算出口）与 `search_l3`（独立 ChunkIndex 检索） |
| `src/paperpilot/agents/state.py` | Verdict 去 `need_chunk_ids`、增 `target_sections`；State 增 `core_points`、`l3_chunks` |
| `src/paperpilot/graph/qa_graph.py` | 重构为 L0→L1→L2(自环)→L3→unknown 五节点图 |
| `cli/run_qa_eval.py` | 按验收问题集加回归用例 |

---

## 9. 参考：与旧方案的取舍记录

| 方案 | 结论 | 原因 |
|---|---|---|
| 预分类 router（一次定档） | ❌ 弃 | 额外推理任务换毫秒检索，不值；先射箭后画靶，错则无兜底 |
| L2 整节拉取 | ❌ 弃 | 大章节预算爆炸；节不对拉再多也没用 |
| L2 增量扩展 ±1→±2… | ✅ 用 | 最小上下文解决局部问题，阈值外及时止损 |
| L3 独立全文检索 | ✅ 用 | 错误不传导；用原始问题重新找，是最干净的质量保险 |
| L2 单圆心（score top1） | ❌ 弃 | score 高≠答案所在地；答案跨两节必漏（D5）。改 Judge1 定向多圆心 |
| Judge 一套模板 | ❌ 弃 | 各层"够"的标准不同，混用致误判（D6）。改 prompt 工厂分层模板（§3.5） |
| gap 只存不读 | ❌ 弃 | 判断产物未消费（D7）。改 gap + `target_sections` 驱动下一层取料 |

**v1.1 修订摘要**（2026-09-04 评审）：
1. L2 圆心从 score top1 → Judge1 输出的 `target_sections` 定向（可多圆心、预算共享、分窗隔离）；Judge1 无把握则 bypass 直接 L3。
2. Judge 改为 prompt 工厂四层分模板（§3.5），各配证据/够的标准/gap 写法/节选项。
3. Verdict 增 `target_sections`，gap 从展示字段升级为取料驱动。
