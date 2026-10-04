# 用 QAMPARI（LoFT-RAG）当标准考场：数据实测与定位（2026-09-29）

> **结论**：该用，而且要用**比 QAMPARI 原版更对味的形态**——
> **LoFT-RAG-QAMPARI**：三个**语料规模档位**（27k / 105k / 844k token）+ 标准三段式 + **已知合格线**。
> 它一次解决"真值规范 / 可对标 / 知道上限在哪"三件事，还额外给了我们最缺的**语料规模扫描**。

## 1. 取数方式（权威）

- **GCS 直链**（LoFT 官方）：`https://storage.googleapis.com/loft-bench/rag/qampari.zip`（**1.8 MB**，已下载）
- 论文：Lee et al., *Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?*（Google DeepMind，arXiv:2406.13121）
- 仓库：`github.com/google-deepmind/loft`（另有 `download.sh` 与 prompts 包）
- ⚠️ 官方记录里 **`mteb/rag-qampari-32k` 只是占位样例**（语料 186 段、问题 10 条），真正数据在 LoFT 的 GCS 包。

## 2. 数据实测（本地已解剖）

| 档位 | 语料段落 | 语料 token（粗估） | dev | **test** | 每题 gold 答案 | 每题 qrels 段落 |
|---|---|---|---|---|---|---|
| **32k** | 186 | **27k** | 10 题 | **缺失** | 5.00 | 5.00 |
| **128k** | 755 | **105k** | 10 题 | **100 题** | 5.02 | 5.00 |
| **1m** | 5,878 | **844k** | 10 题 | **100 题** | 5.02 | 5.00 |

**字段**：corpus = `pid / title_text / passage_text`；queries = `qid / query_text / answers / metadata.qrels`
**关键性质**：
- **每题 5.02 个答案、0% 单答案题** → **纯多答案任务**（正是"覆盖"问题的标准形态）
- **1m 档 844k token → 任何窗口都装不下 → 检索物理必需** ✓✓
- 128k 档 105k token → 接近窗口边界（直读勉强，论文实测直读 0.44）

> ⚠️ 更正我此前的说法：LoFT 的 `1m` 指**上下文 token 档位**，不是"100 万段落"（1m 档实际只有 5,878 段）。

## 3. 已知合格线（论文 Table 2，**128k test**，指标 `subspan_em`）

| 方法 | subspan_em |
|---|---|
| **专用 RAG pipeline**（Gecko 检索 top-40 + Gemini 1.5 Pro reader） | **0.55 ← 合格线** |
| **Gemini 1.5 Pro 直读 128k**（即"不检索"） | **0.44 ← 直读天花板** |
| GPT-4o 直读 | 0.27 |
| Claude 3 Opus 直读 | 0.25 |

**三条可直接用的判据**：
1. **合格线 = 0.55**；**只要超过 0.44，检索就有价值**（否则不如直读）。
2. **128k 上 RAG(0.55) > 直读(0.44)** → 检索确有增益（+11pt）。
3. **1M 档没有公开数字**（论文只给总体趋势：128k 可与 pipeline 相当，到 1M 下降）→ **1M 是空白区**，也是我们最能证明价值的地方。

**指标定义**：`subspan_em` = 把预测答案与 gold 按 overlap 做 **linear sum assignment 匹配**，
**每个 gold 都要被完美匹配**才给满分 → **全有或全无**（与 `MRecall@k` 同源，比 set-F1 严格）。

**论文的 prompt 消融**（128k，Gemini 1.5 Pro）——很有参考价值：

| 设置 | 分 | 设置 | 分 |
|---|---|---|---|
| Best prompt | 0.44 | Corpus in each few-shot | 0.36 |
| Generic instruction | 0.42 | Without ID echo | 0.36 |
| Alphanumeric IDs | 0.42 | **Query at beginning** | **0.35** |
| Without CoT | 0.33 | **Titles only（不给正文）** | **0.09** |

→ **prompt 设计的影响（0.09~0.44）比很多检索优化还大**；我们做对比实验时**必须固定 prompt**。

## 4. 对我们的定位：双轨

| 轨道 | 语料 | 作用 |
|---|---|---|
| **A 标准考场**（LoFT-RAG-QAMPARI） | 维基百科段落，1m 档 844k token / 128k 档 105k token，100 题 test | **对外可对标**（0.55 / 0.44）；验证检索与覆盖方法的增益 |
| **B 产品场景**（自有 20 篇论文 + facet） | 论文全文，240~280k token | **我们的差异化**（跨语言、facet 判定、用户自带语料） |

两轨**共用同一套检索/判定组件** → 标准考场上的增益可直接迁移。

## 5. 诚实边界

1. **样本规模**：128k/1m 各 100 题 test（可用）；**32k 只有 10 题 dev、没有 test** → 32k 档统计意义弱。
2. **语料不同域**：维基百科段落 ≠ 学术论文；**答案空间是实体**（比赛名/电影名）≠ "论文有没有做 X"。
   → 迁移性必须验证，不能直接假定。
3. **1M 档无公开基线** → 我们要自己定对照（可用"直读不可行"论证必要性 + 在 128k 上做对照）。
4. `subspan_em` 是 all-or-nothing，**对部分正确不给分** → 数值会低于 set-F1。
5. LoFT 的 RAG 任务假设**语料整体塞进上下文**（Corpus-in-Context）；我们做的是**检索**路线，
   两者在这个基准上正好是**对照关系**（论文已给出 0.44 vs 0.55）。

## 6. 关于"NLI 在 QAMPARI 上会不会变好"

- **值得测**：QAMPARI 的证据是**维基百科正文**、答案是**实体**，
  比"判断论文里有没有做 X"更接近 NLI 的训练分布（MNLI/XNLI 也是维基/新闻语体）。
- **成本**：NLI 模型已按你要求删除 → **需重新下载**（英文最优候选 `DeBERTa-v3-base-mnli-fever-anli` 约 **380MB / ~1.5 分钟**；
  多语言候选 551MB）。
- **预期先说明**：**ALCE 在 QAMPARI 上的引用评测用的就是 11B 的 `TRUE`**，且它自报与人工一致率仅 85.1%/77.6%
  → **279M 的模型大概率仍不够**。所以这是"低成本验证一个假设"，不是"换回 NLI"。

## 7. 下一步（建议）

| 序 | 做什么 | 为什么 |
|---|---|---|
| **1** | **在 128k 档跑通我们的 pipeline**（检索 top-k → LLM reader 出答案列表），报 `subspan_em` + `Recall@5/MRecall@5` | 有 100 题 + **公开数字 0.44/0.55 可直接对标** |
| **2** | 上 **1M 档（844k token）** | 检索物理必需的空白区，最能证明价值 |
| 3 | 用 **128k vs 1M 的落差**画"语料规模 → 该不该检索"的拐点 | 我们缺的核心判据 |
| 4 | 可选：重下英文 NLI，在 QAMPARI 上复测（验证"域不同"假设） | 低成本、能关门 |

## 8. 复现

```bash
./.venv/Scripts/python.exe -u retrieval/tmp/_loft_dl.py    # 下载 + 解剖（含各档语料/题数/gold 数）
```
产物：`retrieval/data/loft/qampari/{32k,128k,1m}/{corpus,dev_queries,test_queries}.jsonl`
