# RAG-2（多答案开放域问答）· 两条测试集全组件清单

日期：2026-09-29 ｜ 依据：逐文件读源码 + 实测（`retrieval/tmp/_two_sets_stat.py`、`_token_ratio.py`）

我们要做的 **RAG-2 = 多答案开放域问答**：「这 N 篇/这堆语料里，**哪几篇/哪些**满足 X？」
回答的是 **集合**（篇集合 / 答案列表），而不是一段成稿。

**两条测试集**：
- **A · QAMPARI**（LoFT 官方，维基语料）—— 可与外界对比
- **B · 20 篇论文**（我们自己出题，LitSearch 全文）—— 无公开基准，看相对随机/上界

---

## 0. 一页总表：9 个环节 × 2 条测试集

| # | 环节 | **A · QAMPARI** | **B · 20 篇论文（我们出题）** |
|---|---|---|---|
| 1 | **数据源** | LoFT 官方 `data/loft/qampari/{32k,128k,1m}/{corpus,test_queries}.jsonl` | LitSearch `corpus_clean.full_paper` → `data/r2dev/clusters/c{0,1,2}.parquet` |
| 2 | **语料规模** | 755 段 / 421,289 字符（128k 档）≈ **103k token** | 20 篇 × 3 簇 / 3,197,075 字符 ≈ **780k token** |
| 3 | **切块** | **不切块** —— passage 本身就是检索单元（中位 **581 字符**/段） | **固定窗 1000 字符 / 步长 900**（`chunks_of`）→ **54~64 块/篇** |
| 4 | **编码** | bge-m3，`max_seq_length=512`，**无落盘缓存** | 同（每簇一次） |
| 5 | **查询** | **单条自然问句**（官方 `query_text`） | **`[中文题面] + LLM 子查询`**（16 个 facet，输入只有中文题面） |
| 6 | **打分** | dense(cosine) ⊕ **BM25**(k1=1.2, b=0.75) | dense 为主；BM25 作**独立对照臂**（`bm25_para` 非同源 / `bm25_anchor` 同源⚠️） |
| 7 | **排序** | **RRF k=60** 融合名次（`--retriever rrf` 默认）｜另可 `dense` / `bm25` | A `base`(单查询) / B `mq_max`(跨查询逐块 max) / C `mq_rr`(**轮询交错**, AMER) |
| 8 | **重排** | **CE `BAAI/bge-reranker-v2-m3`**（`max_length=512`）：pool → top-k | **MMR**（λ=1.0/0.7/0.5/0.3，pool=40）｜**无 CE** |
| 9 | **作答（reader）** | **4 臂**：`single`(一次看 K 段) / `passage_union`(**逐段抽取**并集) / `passage_verified`(**+逐候选判定**) / `grp_union`(5 段一组) | `_r2_reader`：**逐篇判定** `yes/no/unclear` + 证据块号 |
| 10 | **检查器** | `--verify`(LLM 二轮复核) / `--ground`(**确定性接地过滤**) / `_qampari_norm_diag`(归一化诊断) | **真值双 LLM 三轮校准** / **ALCE NLI 引文检查** / **判官可靠性**（三方、四方） |
| 11 | **指标** | 官方 `subspan_em` / `em` / `coverage` + ALCE `P/R/F1` + 段落 `recall@k` | 篇级 `MRecall` / `StRecall` / 集合 `P/F1` / **α-nDCG**(Clarke 2008) |
| 12 | **对照基线** | 论文 Table 2 外部线（专用 RAG 0.55 / Gemini 直读 0.44…）+ 闭卷对照（污染） | `random`（200 次排列）+ `oracle`（用 gold，仅上界） |
| — | 题量/真值 | 100 题，gold 均 **5.02** | **35 个真值**（30 可用），gold 均 **5.4/20** |

---

## 1. 测试集 A：QAMPARI（`retrieval/tmp/_qampari_*.py`）

### 1.1 语料与切块 —— **没有"切块"这一步**

LoFT 官方语料本身就是**段落级**（`corpus.jsonl` 每行 = 一个 wiki passage）：

| 档位 | passage 数 | 字符/段(中位) | 总字符 | ≈token |
|---|---|---|---|---|
| 32k | 186 | 594 | 107,468 | 26k |
| **128k** | **755** | **581** | **421,289** | **103k** |
| 1m | 5,878 | 592 | 3,377,938 | **825k** |

语料文本 = `full = title_text + " \n " + passage_text`（`_qampari_run.py:221`）。
**检索单元 = 段落**，`D = enc.encode(corpus["full"])` 一段一向量（`:235`）。
→ **QAMPARI 线上不存在"章节/段落切块"这个环节**，因为官方已经把语料切到段落。

### 1.2 打分（`_qampari_run.py:240-266`）

- **dense**：`D @ qv`（bge-m3，normalized，`max_seq_length=512`）
- **BM25**：自实现，`idf * f*(k1+1) / (f + k1*(1-b+b*dl/avgdl))`，**k1=1.2, b=0.75**，分词 `[a-z0-9]+`
- 可切换：`--retriever rrf`（默认）/ `dense` / `bm25`

### 1.3 排序（`:283-284`）
**RRF k=60**：`score = 1/(60+rank_dense) + 1/(60+rank_sparse)` → `argsort(-score)[:pool]`

### 1.4 重排（`:287-294`）
`--rerank` → **CrossEncoder `BAAI/bge-reranker-v2-m3`**（`max_length=512`）对 **pool** 内逐段打分
→ 重排后取 top-k。另：`_qampari_rerank.py` 专门做"重排诊断 + 导出"。

### 1.5 作答（4 个臂，`_qampari_mapreduce.py` 的 docstring 表）

| 臂 | 做法 | 调用/题 | 针对 |
|---|---|---|---|
| `single` | 一次看 K 段直接列（旧做法，基线） | 1 | — |
| `passage_union` | **逐段抽取**（每段 1 次）→ 并集 | K | 提前收手（系统性少列） |
| `passage_verified` | 上一臂 + **逐候选判定**剔除假阳性 | K + 候选数 | 近邻干扰 |
| `grp_union`（`grp10`） | 把 K 段按 **5 段一组**枚举后并集 | K/5 | 提前收手（更省） |

提示词严格照抄官方常量（`CORPUS_INSTRUCTION` + `FORMATTING_INSTRUCTION` + `QUERY_FORMAT_WITH_COT`），
另有 `--exhaustive` 版（"列出**全部**答案，通常多于五个"）。
reader 直连 API（**可显式关思考模式** `--no-think`）；`--mode direct` = 整档全塞（直读对照）。

### 1.6 检查器（5 类）

| 检查器 | 机制 | 检查什么 |
|---|---|---|
| `--verify`（`_qampari_run.py:310-324`） | **LLM 二轮**严格复核首轮候选（去假阳性 + 补漏） | 回答对不对 |
| `--ground`（`:327-329`） | **确定性**：只保留能在检索段落原文里出现的答案（零 LLM） | 幻觉/不接地 |
| `passage_verified` 臂（`SYS_VERIFY`） | **逐候选二元裁决**："这个候选确实回答了该查询吗" | 回答对不对 |
| `_qampari_norm_diag.py` | **零 LLM**：strict 相等 vs 互为子串，量化"没做归并"导致的 em 损失 | 指标口径 |
| `_contam_check.py` | **4 路探针**：无上下文直答 / 半题面补全 / 半答案补全 / 配对判别 | **预训练是否见过 QAMPARI** |
| 官方复核 | `--dump` 导出官方格式 → 官方 CLI / `_loft_official_eval.py` | 指标是否与官方逐位一致 |

### 1.7 指标（严格照抄官方）
- `em` = 集合相等（`compute_em_multi`）
- `coverage` = |pred ∩ gold| / |gold|（**只有 recall**）
- **`subspan_em`** = 双向子串匹配矩阵 → `linear_sum_assignment` → 全对齐才 1（`:87-99`）
- 自算补充：set P / R(=coverage) / F1、段落级 `recall@k`
- ⚠️ 官方 `multi_value_rag` **不输出 precision**（`f1` 恒 0.0）→ precision 用 **ALCE `eval_alce.py` 算术**移植

### 1.8 脚本清单
`_qampari_run.py`（主）、`_qampari_mapreduce.py`（4 臂）、`_qampari_rerank.py`（重排诊断）、
`_qampari_tier1m.py` / `_qampari_tier_retrieval.py`（档位）、`_qampari_merge.py`、
`_qampari_norm_diag.py`（归一化诊断）、`_qampari_probe.py`、
`_loft_dl.py` / `_loft_fetch_code.py` / `_loft_official_eval.py` / `_loft_probe.py`（官方代码与复核）、
`_contam_check.py`（污染）、`_audit_official_runs.py` / `_audit_precision_all.py` / `_audit_final_v2.py`（指标审计）

---

## 2. 测试集 B：20 篇论文（`retrieval/tmp/_r2_*.py`）

> ⚠️ 这批脚本自称「**新 RAG-2 · 集合型检索**」，是 **RAG-2 的实验脚手架**；
> **不是 RAG-1**（RAG-1 = 论文级排序检索，`_r2_probe.py:10` 明写"臂 A = RAG-1 式"）。

### 2.1 语料与切块

| 簇 | 篇数 | 字符/篇(中位) | 块数 | 块/篇 |
|---|---|---|---|---|
| 1 | 20 | 53,740 | 1,228 | 61.4 |
| 2 | 20 | 56,527 | 1,270 | 63.5 |
| 3 | 20 | 47,598 | 1,081 | 54.0 |
| **合计** | **60** | — | **3,579** | — |

- 语料：LitSearch `corpus_clean.full_paper`（**外部数据集的纯文本全文**）
- 切块：`chunks_of(t) = [t[i:i+1000] for i in range(0, len(t), 900)]` —— **固定窗，不管段落/标题/表格**
- 编码：bge-m3，`max_seq_length=512`

### 2.2 真值（三层，逐层收紧）
1. **词面锚点**：`judge(chunk, facet) = re.search(F[facet][0], chunk, re.I)` → 篇 gold = 任一块命中
2. **双 LLM 三轮校准**（`_r2_gold_recalib.py`）→ `gold_recalib.csv`（**700 行 / 35 个真值**）
   - 判官 A `PAPERPILOT_LLM`、判官 B `PAPERPILOT_JUDGE`；都 entail → 正例；分歧走**第三轮严格对抗提示**
   - 纪律：区分"本文做的 vs 引用他人做的"、"列出≠做了"、中英同义改写
3. **富证据复判**（`_r2_gold_dispute.py`，12 块证据）→ `gold_recalib_rich.csv`（**173 行，争议子集**）
   - `_r2_reader.load_gold()`：`final_gold` 默认 = `new_gold`，**rich 里有记录的对被 `new_gold_rich` 覆盖**

### 2.3 打分（`_r2_std_metrics.py:115-121`）

| 打分 | 查询 | 性质 |
|---|---|---|
| `dense_zh` | 中文题面 | ✅ 干净（**主口径**） |
| `dense_zh_en` | 中文 + 英文锚点词 | ⚠️ **泄露**（锚点词 63% 与 gold 正则同词） |
| `dense_en` | 纯英文锚点 | ⚠️ 同源 |
| `bm25_para` | 英文**改写句**（刻意不含锚点原词） | ✅ 非同源 |
| `bm25_anchor` | 英文锚点原词 | ⚠️ 同源 |

- 篇级分 = **该篇所有块的 max**（`:255`）—— 多值检索的关键：由最相关那块决定
- `_r2_score2.py` 补了**融合对照**：加权 z-norm（w 扫描）+ **RRF k=60**
- BM25 参数同 A：**k1=1.2, b=0.75**

### 2.4 排序（`_r2_phase1_clean.py`，干净口径）

| 臂 | 做法 |
|---|---|
| `A base` | 单查询（中文题面） |
| `B mq_max` | 多查询 → 逐块取**跨查询 max** |
| `C mq_rr` | 多查询 → **轮询交错**（AMER） |
| `D/E MMR` | λ ∈ {1.0, 0.7, 0.5, 0.3}，pool=40 |

### 2.5 重排
**只有 MMR**（覆盖导向），**没有 CE**。（`_r2_reader.py:125` import 了 `CrossEncoder` 但**未使用** = 遗留。）

### 2.6 作答（reader）
`_r2_reader.py`：每篇取**它自己**的 top-b 块（b=3/6/12）→ **1 次 LLM/篇**（并发 8）
→ `{label: yes|no|unclear, evidence:[块号], why}` → 判 yes 的篇 = 交付集合
提示四纪律与生产 `set_judge` **同源**。

### 2.7 检查器（4 类，14 个脚本）

| 类别 | 脚本 | 检查什么 | 机制 |
|---|---|---|---|
| **真值校准** | `_r2_gold_recalib.py` | 真值对不对 | 双 LLM 异源 + 第三轮严格对抗 |
| | `_r2_gold_dispute.py` | 新真值是否偏严 | 富证据（12 块）复判 |
| | `_r2_gold_metrics.py` | 臂排序对真值敏感吗 | 三种真值口径对照（零 LLM） |
| **引文检查（ALCE）** | `_r2_l3_nli.py` | **引文有没有支持论断** | NLI `mDeBERTa-v3-xnli-2mil7` + 词面代理并排 |
| | `_nli_route.py` | 中文论断 × 英文证据能否判定 | 3 个 NLI judge × 论断三路线（zh / 译en / 改写en） |
| **判官可靠性** | `_r2_calib2.py` / `_r2_calib_llm.py` / `_r2_calib_inspect.py` | NLI 阈值 / 锚点精度 | NLI + LLM 看**同一段**文本校准 |
| | `_r2_judge2.py`(三方) / `_r2_judge3.py`(四方, reasoner) | 判定器一致率 | NLI vs glm-4-flash vs deepseek-chat vs reasoner |
| | `_r2_adjudge.py` | 人工裁定样本 | 导出 5 组样本，**无自动判定** |
| | `_r2_nli_audit.py` / `_r2_nli_lang_final.py` / `_nli_smoke.py` | "NLI 出错"是结论还是用法 bug | 换 tokenizer / 换语言 / 换模型 |
| **泄露** | `_leak_audit.py` | 查询是否复用 gold 词汇 | 静态重合率 + **净化锚点实验**（AUC 对照） |

> 只有 `_qampari_norm_diag.py` 与 `_contam_check.py` 同时服务 A 线；其余 14 个只服务 B 线。

### 2.8 指标
- L1 篇级：`MRecall@k`（JPR）/ `StRecall@k`（Zhai/CoverageBench）/ 集合 `P` / 集合 `F1` / **`α-nDCG`**（Clarke 2008，α=0.9）
- L2 等块预算：`global_topB` vs `round_robin`，含**证据块召回** `ev_recall`
- L3 ALCE 式（检索侧）：`cite_recall`（至少一块支持）/ `cite_precision`（引用块里支持比例）

### 2.9 脚本清单
**建库/真值**：`_r2_build_corpus.py`、`_r2_score.py`、`_r2_score2.py`、`_r2_setsel.py`、`_r2_probe.py`、`_r2_gold_*.py`
**检索/评测**：`_r2_std_metrics.py`（L1/L2/L3）、`_r2_phase1.py`、`_r2_phase1_clean.py`
**reader**：`_r2_reader.py`；**整档对照**：`_r2_fullread.py`
**本次新增**：`_set_extract_eval.py`、`_regress_cmp.py`、`_two_sets_stat.py`

---

## 3. 两者能共用什么 / 不能共用什么

| 能共用 | 不能共用 |
|---|---|
| 编码器（bge-m3）与 `max_seq_length` | **语料单位**：段落(A) vs 固定窗块(B) |
| BM25 参数（k1=1.2, b=0.75） | **查询构造**：单问句(A) vs 中文题面+子查询(B) |
| 重排模型（`bge-reranker-v2-m3`）—— 但 **B 线没用** | **题目形态**：「列全部答案」(A) vs 「哪几篇」(B) |
| **判定提示四纪律**（生产 `set_judge` 从这里继承） | **指标**：官方 `subspan_em`(A) vs 篇级集合 F1(B) |
| 接地过滤思路 | 真值来源：官方 qrels(A) vs 双 LLM 校准(B) |

→ **两条线只有"编码器 + BM25 参数 + 判定四纪律"三样共用**，其余全不同。

---

## 4. 与生产链路的关系（一句话）

| | 语料 | 切块 | 用途 |
|---|---|---|---|
| **A/B 两条测试集** | 外部数据集（LoFT 维基 / LitSearch 论文） | **A 不切块**、**B 固定窗 1000/900** | 实验脚手架，**非生产切块** |
| **生产 `set` 节点** | 用户给的 5 篇 PDF | **章节段落切块**（`chunk_document` ≤4000 + MinerU 注入） | 真正的产品路径 |

→ 生产问答（`fullctx`/RAG-2/`set`/报告链/修复链）**全部**用章节段落切块；
→ 但 **A/B 两条测试集的数字都不是在生产切块上测的**（详见 `RAG2_DATAFLOW_20260929.md` §0、§4）。

---

## 5. 本次核实中发现的 **3 处我自己的计量问题**（必须更正）

### 5.1 token 换算率用错了（影响所有"token/S-C"数字）
实测（`_token_ratio.py`，真实 API `usage.prompt_tokens`）：

| 文本类型 | 实测比率 |
|---|---|
| 论文正文（PDF 检索视图） | **4.05 字符/token** |
| LitSearch 论文全文 | **4.28** |
| QAMPARI 维基 passage | **4.08** |
| 英文填充句（我此前用的口径） | 5.08 |

我此前混用了 **/4** 与 **/4.94** → 同一份语料报出过两个 token 数。**更正后**：

| 设定 | 字符 | **≈token（÷4.1）** | S/C(÷128k) |
|---|---|---|---|
| 生产 5 篇/组 | 170k~192k | **42k~47k** | **0.32~0.36** |
| QAMPARI 128k | 421,289 | **103k** | **0.79** |
| R2 20 篇/簇 | 965k~1,137k | **235k~277k** | **1.79~2.12** |
| QAMPARI 1m | 3,377,938 | **825k** | **6.29** |

→ **S/C 阶梯的排序与全部结论不变**（价值起点仍在 0.8~2.0 之间）。
→ ⚠️ **撤回一条**：我上轮说"生产 5 篇实测 34k~39k token → docstring 的 ≈48k 偏大约 20%"是**错的**
（那是用 4.94 算的）。按实测 4.05，**5 篇 ≈ 44k token，docstring 的 ≈48k 是对的**。

### 5.2 `llm.usage_stats()` 是**全局非线程安全**的
```35:36:src/paperpilot/tools/llm.py
_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
          "cache_hit_tokens": 0, "cache_miss_tokens": 0}
```
`reset_usage()/usage_stats()` 直接读写这个模块级字典，**无锁**（docstring 也只在"顺序执行"下成立）。
→ 凡**并发**脚本里"每题 reset 后读"的数字**都被其他线程污染**：
`_r2_fullread.py`（workers=4）、`_m1_set_ab.py` / `_set_extract_eval.py`（`set_judge` 内层 workers=6）、
`_r2_reader.py`（内层 workers=8，但只报总量 → **总量可信**）。
→ **影响范围：只有"每题 token/成本"这类表述；所有 P/R/F1 指标不受影响（不依赖用量计数）。**

### 5.3 "省 11× token" → 应为 **≈8.7×**
按结构算：整档直读 ≈**249k** token/题 vs 逐篇判定 20 篇 × 6 块 × 1000 字符 = 120k 字符 ÷ 4.28 ≈ **28.6k** token/题
→ **249 / 28.6 ≈ 8.7×**（此前 11× 是用法中位数 266k 与 /4.94 混算的）。

---

## 6. 复现

```powershell
cd f:/paperpilot
$env:HF_HOME='F:\hf_cache'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'

# 本清单的规模数字
./.venv/Scripts/python.exe -u retrieval/tmp/_two_sets_stat.py    # A/B 两套语料规模
./.venv/Scripts/python.exe -u retrieval/tmp/_token_ratio.py      # 字符/token 实测比率
./.venv/Scripts/python.exe -u retrieval/tmp/_rag2_corpus_stat.py # 生产切块 vs 实验固定窗

# A · QAMPARI
./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_run.py --k 40 --exhaustive --no-think --dump
./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_mapreduce.py --k 20 --workers 8
./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_rerank.py

# B · 20 篇论文
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_std_metrics.py
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_phase1_clean.py
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_reader.py
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fullread.py
```
