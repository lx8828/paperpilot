# 对比测试设计（朴素 RAG / 直接 LLM vs PaperPilot）

> 目标：① 用**公平基线**把 PaperPilot 的架构差异点量化出来（不只"自评分多高"，而是"比最简可做的系统强多少、强在哪"）；② 顺带测出能力边界（长文、无答案、枚举/数值等）。
> 状态：**首轮已执行（2026-09-07，20 篇/64 题）**，结果与解读见 `qa/compare/compare_20260907_191606.md`（run 记录 `run_*_B0/B1/B2.json`）。
> 首轮定案：规模 20 篇；B0 超长不截断、如实记 overflow；首轮固定 deepseek-chat；表格子集不做。
> 关联：`CAMPAIGN_20260906-07.md`（残差画像/收口）、`run_qasper_eval.py`（评测协议与裁判口径）、`.env`（裁判现状 glm-4-flash）。

---

## 0. 一句话设计

三列 **B0 直接 LLM / B1 朴素 RAG / B2 PaperPilot**，同一批题、同一生成模型、同一裁判、同一判分 prompt（盲判），只在"上下文怎么组装"上不同；每列每题记录 **正确性 / 诚实拒答 / 耗时 / LLM 调用数 / token / 引用**，得到"准度 × 成本 × 溯源"三角画像。

- 不是"谁全能赢"的比赛，是**各列画自己的坐标**：预期 B2 准度接近最好、成本居中、溯源与拒答领先——这就是差异点故事。

## 1. 对比对象与控制变量（公平性核心）

| | 列 | 是什么 | 说明 |
|---|---|---|---|
| B0 | 直接 LLM | 把论文全文喂给 LLM 直接答 | 参照系：检索是否有价值；超长文放不下/贵本身是边界发现 |
| B1 | 朴素 RAG | 复用 `ChunkIndex` 向量检索 top-k，拼块直接答；**无 claims、无漏斗、无强制拒答、无引用** | 关键一刀：B1→B2 差值 = claims 证据链 + 漏斗 + 收尾的价值 |
| B2 | PaperPilot | 现完整链路（`graph.ask`，含缺失断言闸门、answer_unknown 提炼） | 被测对象 |

**不变（控制变量）——任何一列输赢必须能归因到架构，而非别的：**

1. **生成模型同一**：三列都用 `deepseek-chat`（v4-flash），temperature=0；只用 `PAPERPILOT_LLM_*` 一套配置。
2. **题目同一**：锁定同一批论文（pid 集合 + 各自全部 QASPER 题），执行前固化为 `qa/compare/papers_<ts>.json`；B1/B2 用同一 `out_views` report 产物缓存（不重复跑 pipeline）。
3. **检索底料同一**：B1 直接复用 B2 的 `ChunkIndex`（同 chunk、同 embedder）——差值只来自"上层怎么用"。
4. **裁判同一且盲判**：同一裁判 prompt（沿用 `run_qasper_eval.py` 的 `_JUDGE_SYS/_JUDGE_USER`，含"引用必须真支撑否则降分"条款），裁判不知道答案来自哪列。
5. **计分口径同一**：score 1~5、pass≥4、状态分桶（pass/fail/unknown_ok/honest_refuse/error）与 QASPER 定版完全一致。

**只变**：上下文组装（全文 vs top-k 块 vs claims 证据链）、是否走 L0→L3、是否输出引用、是否做缺失断言复核/诚实拒答。

> 注意：B2 的 answer 模板带 PaperPilot 自己的"缺什么说明什么"约束；为公平 B0/B1 用**中立的答案指令**（"基于给定材料作答"），不带我们模板的口吻，避免把 prompt 差异混进架构差异。

## 2. 评测维度（每列每题都记录）

| 维度 | 度量 | 数据来源 / 口径 |
|---|---|---|
| 精准（主指标） | pass≥4 通过率 + 均分，**按题类切片** | QASPER 自带 question_type（extractive/free_form/yes_no）+ 无答案题单列；数值/枚举/表格类可另打标子集 |
| 真不编造 | 无答案题"诚实拒答且被裁判认可"率；有答案题的 citation 支撑性 | 裁判 1~5 本身含"编造/引用不实降分"；另抽查 judge reason |
| 可溯源（仅 B2 有） | cites_n>0 题占比、引用是否真实支撑 | B0/B1 无引用概念——这是**结构性差异点**，单独报告 |
| 时间 | 每题端到端耗时 time_s；B2 附 route 分档 | 记录见 §3 |
| 成本 | LLM 调用次数、输入/输出 token、估算金额 | 记录见 §3 |
| 稳定性（第二轮） | 同题改写 ×3 的一致率 | 独立小样本，不做首轮 |

### 结果呈现
- 主表：三列 ×（pass%、均分、by-type pass%、拒答率、cites_n、avg time_s、avg token、avg 调用、est ¥/题）。
- 散点：每题（time_s × pass）与（token × pass）；延迟与准度的取舍一眼可见。
- 差值摘要：`Δ(B1→B2)` 与 `Δ(B0→B2)` 逐题型一句话结论。

## 3. 时间与成本记录（本轮明确要求）

### 3.1 每题必记（机器自动）
| 字段 | 含义 | 现状 |
|---|---|---|
| `time_s` | 端到端 wall-clock（发起→答案出） | `run_qasper_eval.py` 已记 |
| `llm_calls` | 该题 LLM 请求次数 | 需加：monkeypatch `llm._chat` 计数（`run_qa_v2.py` 已这么做，照抄） |
| `prompt_tokens` / `completion_tokens` | DeepSeek 返回的 usage | **当前 `llm.py` 丢弃 usage** → 落地时在 `_chat` 内累加模块级计数器（向后兼容），或 harness 侧包一层读响应 |
| `est_cost` | 按 deepseek-chat 单价（¥/1M tokens）× usage 估算 | 单价写入 run 元数据，便于事后重算 |

### 3.2 时间口径注意（防误导）
- **冷启动不算**：embedder 权重加载是"每进程一次性"成本，会影响 first-call 延迟。每列开跑前各 warm-up 一次（跑 1 题丢弃），time_s 全部取 warm 后。
- **B2 分档参考**：B2 把 `route` 与 `answer_level` 写进记录，可事后按 L0/L1/L2/L3/unknown 分档看耗时；B0/B1 无档位，耗时即单次调用。
- **网络抖动**：同机同批、温度 0、judge 重试上限一致（现有 `_chat` 已 3 次重试 4xx 不重试）；单点异常题单独标出，不进均值主导结论。

### 3.3 run 元数据（复现性，每次 run 顶部一段）
```
run_id / datetime / git commit / 生成模型(deepseek-chat) / 裁判模型(glm-4-flash) /
题集版本(papers_*.json 的 pid 数 + 题数) / 是否 warm-up / 单价(¥/1M in, out)
```
> 裁判已从 glm-5.3（无余额）→ **glm-4-flash（免费，异源保持）**。glm-4-flash 判分严格度可能与 glm-5.3 不同：**本次三列同裁判同模板，列间可比；但与历史 75%/81.6% 等数字不可直接比**，报告须注明。

## 4. 题目与规模（默认值，可调）

- 数据源：QASPER train（`qasper_source.load_papers`），只取已跑过 pipeline 的论文（避免重复烧 pipeline），用 `--from-run` 复用旧 run 的 pid 列表锁样本。
- 默认规模：**~20–40 篇 ≈ 60–120 题**（QASPER 篇均 ~2.9 题）。包含天然的无答案题（gold 空）用于拒答维度。
- 三列共跑一遍；B0 遇超长论文（估算超上下文预算）自动用"摘要+引言+结论"截断版并在记录里打 `truncated=true`——这条本身是边界发现，不是偷跑。
- 执行顺序建议 B1 → B0 → B2（先便宜后贵，中途可随时停；每列结果独立成文件，不互相依赖）。

## 5. 产物与文件约定（均在 `qa/compare/`）

```
qa/compare/
  papers_<ts>.json          # 锁定的 pid 集（含题数统计）
  run_<ts>_B0.json / _B1.json / _B2.json   # 每列逐题记录（schema 见 §3，含 judge score/status）
  compare_<ts>.md           # 汇总报告：元数据 + 主表 + 分题型表 + 差值摘要 + 时间/成本表
```

记录逐题字段 = QASPER 基础通过协议字段（qid/paper/question/gold/unanswerable/answer/cites/route/level/score/reason/status/time_s）**+ 本设计新增**（llm_calls/prompt_tokens/completion_tokens/est_cost/truncated）。

## 6. 风险与限制（先写死，别等跑完再补）

1. 裁判同为低成本模型，判分偏松/偏紧都可能——列间公平，绝对值谨慎。
2. LLM 单次波动（±2~3pt 见 CAMPAIGN §6）：首轮只给方向性结论；要给精确差值需同题多次或换问法小样本复核。
3. B0 全文输入 token 量大，单题成本可能数十倍于 B1/B2——预算先按 100 题粗估，超了就缩小样本。
4. 表格/图数据层是当前已知短板（CAMPAIGN §6），B2 在数值/表格类上**预期输**——如实呈现即为边界，不遮掩。

## 7. 待拍板项（执行前确认）

- [ ] 样本规模：~20 篇（60 题）/ ~40 篇（120 题）？
- [ ] B0 超长文截断方案是否接受（截断版 + 打标）？
- [ ] 首轮生成模型固定 deepseek-chat 是否确认？第二轮的"换 glm-4-flash 或 deepseek-v4-pro × 30 题"稳健性检查留到首轮出结论后再开。
- [ ] 表格/数值类是否单独抽 ~20 题打标子集做切片（可选，先不做也行）。
