# R2｜泄露审计（leakage audit）

日期：2026-09-29 ｜ 范围：R2 开发实验全链路（`_qampari_run.py` / `_r2_std_metrics.py` / `_r2_phase1.py` /
`src/paperpilot/components/query_optimizer.py` / `src/paperpilot/agents/embedder.py` / `pull_chunk.py`）

---

## 0. 审计项（用户提出）

1. 索引和排序全程没有用到 gold evidence 的段落 ID
2. 「每篇保底轮询」只在检索排序结果上做、不碰 gold 标注
3. 多查询生成的 prompt 里只有问题本身

---

## 1. 逐项结论

| # | 审计项 | 结论 | 关键证据 |
|---|---|---|---|
| 1 | 索引不用 gold | ✅ **干净** | `C = enc.encode(chunks)`，`chunks ← full_paper` 切块；BM25 语料 = `chunks`。**gold 从未进入编码/索引** |
| 1 | 排序不用 gold | ✅ **干净** | `order = np.argsort(-score)`；`per[d] = idx[d][np.argsort(-s[idx[d]])]`；`s` 由查询向量算。**gold 未参与任何打分** |
| 1 | 唯一 gold 进排序处 | ⚠️ **已显式标注为上界** | `add_l1("oracle", sorted(docs, key=lambda d: (d not in gold, d)), "完美排序（上界）")` — 不可部署，只作参照线 |
| 2 | 轮询只走排序结果 | ✅ **干净** | `per = {d: idx[d][np.argsort(-s[idx[d]])] for d in docs}` → 只按分数；`rr` 遍历**全部 20 篇**（不筛 gold）；`docs` 来自聚类语料，非 gold 派生 |
| 3 | 多查询 prompt 只有问题 | ✅ **干净** | `chat_json(SUBQ_SYS, f"论断：{F[f][1]}")` ← `F[f][1]` = **中文题面**；`expand/crosslingual/optimize/search` 的**签名里没有 gold**（结构上不可能泄露） |
| **4** | **★ 新发现：查询侧泄露** | ❌ **有泄露** | `dense_zh_en` 的查询 = `f"{zh} {anchor}"`，而 `anchor` 与 **gold 正则 63% 同词汇** |

### 1.1 第 1 项补充：`_qampari_run.py` 的 `qrels` 是"指标专用"

```python
276:  dense = D @ qv[qi]            # ← 打分
277:  sparse = bm25(q["query_text"])
286:  order = np.argsort(-score)[:pool]   # ← 排序完成
...
297:  qrels = [str(x[0]) for x in q["metadata"]["qrels"]]   # ← 之后才读 gold 段落 ID
298:  gold_pass = set(qrels)
299:  hit = len(set(pid_of[i] for i in order) & gold_pass)
341:  recall_pass=hit / len(gold_pass)      # ← 只喂给指标
```
**读 gold 的行（297）严格晚于排序（286），且只流向 `recall_pass` 这一个诊断指标，不回流任何决策。** ✅

### 1.2 第 3 项补充：签名级保证

| 函数 | 签名 | 能否传入 gold |
|---|---|---|
| `query_optimizer.optimize(question, ...)` | 只有 `question` | ❌ 结构上不可能 |
| `query_optimizer.expand(question)` / `crosslingual(question)` | 只有 `question` | ❌ |
| `MultiChunkIndex.search(query)` / `search_multi_hybrid(queries)` / `search_layered(query)` | 只有 query | ❌ |
| `_r2_phase1.get_subqueries(facets)` → `f"论断：{F[f][1]}"` | 只传中文题面 | ❌ |

→ `embedder.py` 里出现 "gold" 的地方**全在 docstring**（引用历史离线扫描结论），**无运行时读取**。

---

## 2. ★ 新发现：`dense_zh_en` 查询被 gold 词汇污染（已量化）

### 2.1 静态证据：锚点词 63% 就是 gold 正则词

```python
# _r2_std_metrics.py L53 起
F = { "ablation": (r"\bablation", "做了消融实验…", "ablation study component analysis removing parts", …), … }
#                                                                 ↑ 英文锚点词
qvec = {"zh": zh, "zh_en": f"{zh} {anchor}", "en": anchor}    # L238-239
```

| facet | 锚点词数 | 正则词数 | 重合 | 重合/锚点 | 重合的词 |
|---|---|---|---|---|---|
| `new_dataset` | 5 | 9 | 5 | **100%** | benchmark, corpus, dataset, introduce, new |
| `prompt_eng` | 6 | 7 | 6 | **100%** | design, engineering, few, prompt, shot, template |
| `context_length` | 5 | 4 | 4 | 80% | context, length, long, window |
| `significance` | 6 | 8 | 5 | 83% | deviation, multiple, runs, significance, standard |
| `efficiency` | 6 | 7 | 5 | 83% | cost, inference, latency, throughput, time |
| `deployment` | 6 | 8 | 5 | 83% | online, production, real, serving, world |
| `annotation_cost` | 5 | 6 | 4 | 80% | annotation, cheap, cost, labeling |
| `code_release` | 11 | 9 | 7 | 64% | available, code, github, open, publicly, release, source |
| `ablation` | 6 | 1 | 1 | 17% | ablation |
| **合计** | **107** | — | **67** | **63%** | — |

- 对照 **`para`（英文改写句，设计上排除锚点原词）**：重合 **6 个**（不是 0，但低一个量级）→ **大体干净**
- **`zh`（中文题面）**：全汉字，与英文正则**零重合** → **干净**

### 2.2 动态实验（决定性）：把与正则重合的词删掉再测

对每个 facet 的 `anchor` 删去与 gold 正则重合的词（"净化锚点"），重新编码检索，算**篇级 AUC**：

| 查询变体 | 篇级 AUC | 相对 `zh` | 判定 |
|---|---|---|---|
| `zh`（**干净基线**） | **0.702** | — | — |
| `zh+anchor`（**现行配置**） | 0.739 | **+0.037** | **← 这个 "+3.7pt" 是泄露** |
| `anchor`（最污染） | 0.776 | **+0.074** | ← 优势 |
| **`anchor` 净化后** | **0.694** | **−0.009** | ← **优势归零，甚至略低于基线** |
| `bm25_para`（干净） | 0.644 | −0.059 | 干净但弱 |
| `bm25_anchor`（污染） | **0.894** | **+0.192** | 纯泄露（它就是 gold 正则词本身） |

**结论**：
- `anchor` 的 **+0.074 AUC 优势，在删掉与 gold 正则重合的词后变成 −0.009 → 泄露解释了 100% 的优势。**
- `zh+anchor`（**我们一直在用的 "dense_zh_en"**）的 **+0.037** 同理 → **"加英文锚点词 +4pt" 是泄露产物**。

### 2.3 因此必须撤回的一条旧结论

`R2_SCORING_20260927.md` 里那条：

> 「中文题面 0.665 → 中文+英文锚点词 0.704 = **+4pt**，说明 **XLING（跨语言检索式）在这里值 +4pt**」

→ **撤回**。该 +4pt 来自"查询里塞了 gold 正则词"，不是跨语言能力。
（XLING 在**生产链路**上仍有独立价值，但**不能用这套实验的数字作证**。）

### 2.4 第 2 层（轮询 vs 全局 top-B）在干净打分下**结论不变**

| 打分 | B | 全局 top-B 证据召回 | 轮询 证据召回 | 结论 |
|---|---|---|---|---|
| `zh` **干净** | 3 | 0.257 | 0.204 | 全局更好（+0.052） |
| `zh` **干净** | 12 | 0.534 | 0.456 | 全局更好（+0.077） |
| `zh_en` 污染 | 3 | 0.304 | 0.236 | 全局更好（+0.068） |
| `zh_en` 污染 | 12 | 0.588 | 0.477 | 全局更好（+0.111） |

→ **定性与方向不变**（"全局 top-B 证据召回更好、轮询不是答案"），**只是绝对水平被污染抬高**（B=12: 0.588 → 真实 0.534）。
→ 轮询的 **100% 覆盖率依然是平凡的**（涉及 20.0/20 篇）。

---

## 3. 结论汇总

| 项 | 状态 |
|---|---|
| ① 索引不用 gold | ✅ 干净 |
| ① 排序不用 gold | ✅ 干净（唯一例外 = 显式标注的 oracle 上界） |
| ① `qrels` 的使用 | ✅ 仅指标，且严格晚于排序 |
| ② 轮询只走排序结果 | ✅ 干净（遍历全 20 篇，不筛 gold） |
| ③ 多查询 prompt 只有问题 | ✅ 干净（签名级保证） |
| **④ 查询侧 `dense_zh_en`** | ❌ **污染，63% 锚点词 = gold 正则词；+0.074 AUC 全部来自泄露** |

**必须行动**：
1. **弃用 `dense_zh_en` / `dense_en` / `bm25_anchor` 作为"方法性能"**；改用 `dense_zh`（干净）为准。
2. 报告里**明确标注**：任何含 `anchor` 的打分都是**同源上界**。
3. 第 2/3 层若要用英文查询，需**重建锚点词表**：只允许"论文作者会写的自然表述"，并**逐词剔除与 gold 正则重合的词**（脚本已做，见 `_leak_audit.py`）。

## 4. 复现命令

```powershell
cd f:/paperpilot
$env:HF_HOME='F:\hf_cache'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
./.venv/Scripts/python.exe -u retrieval/tmp/_leak_audit.py
```

产物：`R2_LEAK_STATIC.csv`（逐 facet 词汇重合）、`R2_LEAK_AUC.csv`（净化实验）、`R2_LEAK_L2.csv`（第 2 层复算）。

## 5. 边界

1. **"净化锚点"只删与正则重合的单词**，未做同义替换 → 净化后可能过弱（−0.009 是**下限**，不是精确的真实增益）。
2. `para` 与正则仍有 6 个词重合 → "改写句"通道**不是零泄露**，只是低一个量级。
3. 真值本身是**词面锚点正则**（未人工抽检）→ 泄露的定义依赖该正则；换人工真值需重跑本审计。
4. 本审计只覆盖 **R2 开发实验**链路；生产链路（`agents/` / `components/`）未做同类"查询词是否复用真值词汇"检查。
