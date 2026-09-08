# 对比测试报告（直接 LLM / 朴素 RAG vs PaperPilot）

> 生成: 2026-09-07 21:00:45 ｜ 题数: 30（与 B0/B2 同题）
> 裁判: glm-4-flash（异源，三列同 prompt 盲判）｜ 生成: deepseek-chat(v4-flash) ｜ 成本估算单价: 入 2.0 / 出 8.0 元/M（假设，见 COMPARE_DESIGN §3.3）

## 1. 主指标（三列同一批题）

| 指标 | B0 直接 LLM | B1 朴素 RAG | B2 PaperPilot | 说明 |
|---|---|---|---|---|
| pass 通过数(总) | 10/30 | 8/30 | 9/30 | — |
| 有答案题 pass 率 | 33.3% | 25.9% | 22.2% | — |
| 无答案题 pass(诚实拒答被认可) | 1/3 | 1/3 | 3/3 | — |
| overflow(放不下) | 0 | 0 | 0 | — |
| error | 0 | 0 | 0 | — |
| 均分(1-5) | 3.33 | 3.03 | 3.27 | — |
| 平均耗时/题(s, 含判分) | 4.2 | 7.1 | 10.8 | — |
| 平均耗时/题(s, 仅作答) | 2.0 | 4.6 | 7.9 | — |
| 生成 token 总量 | 177,762 | 118,881 | 359,139 | — |
| 生成调用总数 | 30 | 30 | 173 | — |
| 裁判 token 总量 | 15,561 | 14,806 | 36,478 | — |
| 平均 cites/题 | 0.0 | 0.0 | 1.97 | — |
| 估算生成成本(元) | 0.382 | 0.259 | 0.793 | — |

## 2. 分题型 pass 率

| qtype | B0 | B1 | B2 |
|---|---|---|---|
| free_form | 3/14| 3/14| 4/14 |
| yes_no | 0/1| 0/1| 1/1 |
| extractive | 6/12| 4/12| 1/12 |
| unanswerable | 1/3| 1/3| 3/3 |

## 3. 关键差异题（B2 与 B1/B0 判分不一致且 B2 更高/更低）

| qid | 题型 | B0 | B1 | B2 | 问题(截断) |
|---|---|---|---|---|---|
| 790ed4458a | yes_no | 3 | 3 | 5 | Do all the instances contain code-switching? |
| 6407dae0c0 | unanswerable | 3 | 2 | 5 | How long is the dataset used for training? |
| 81e8d42dad | free_form | 3 | 2 | 4 | What is the state-of-the art? |
| 102a043973 | extractive | 4 | 4 | 3 | What datasets were used? |
| ec91b87c3f | unanswerable | 3 | 3 | 4 | What is the baseline used? |
| 38c74ab829 | free_form | 3 | 3 | 2 | How large is the corpus? |
| 96526a1482 | extractive | 4 | 4 | 3 | How was the corpus obtained? |
| 7081b6909c | extractive | 3 | 2 | 1 | How many twitter users are surveyed using th |


---

## 4. 深水区结论（人工）

**先声明方法论**：这 30 题取自 `qasper_final_lock` 中当前代码**仍未 pass** 的硬题（fail/unknown）——
即**按 B2 的已知薄弱点选的池**。B0/B1 没有这种选择偏差，所以"哪列强"要谨慎；本测试更接近
"在我们最差的题上，别人能不能轻松赢"。

**结果**：
1. **没有整体翻盘**：B0 10 / B2 9 / B1 8（有答案题 33%/22%/26%），均分 B0 3.33 > B2 3.27 > B1 3.03。
   → 深水区并未给 B2 带来整体优势，"难题上会拉开差距"的假设**被证伪**（至少在此题池）。
2. **我们的"诚实拒答"优势在硬题上依然稳**：无答案 3 题 B2 **3/3** vs B0/B1 各 1/3。
3. **硬 extractive 定位题是 B2 的真伤**：B2 1/12，而 B0（全文直读、能引原文）6/12、B1 4/12。
   这些正是 final_lock 里我们 fail 的题——说明我们这些 extractive 失败**不是题本身难到无解**，
   是漏斗在"精确摘取指定位置/表内数值/词句"上系统性输给全文直读。与既有边界（表格/数值、枚举细值）同源。
4. **成本/引用**：B2 仍 ~3× token、单题 7.9s、cites 1.97/题——深水区同样成立。

**一句话定位修正**：PaperPilot 的整体主故事不该是"难题也全面更强"，而应是
"**同分段的准度 + 别家没有的诚实拒答与证据引用**"；在"精确摘取 extractive 细值"上反而是短板（标准版如此，MinerU/表格数据层正是为此留的口子）。
