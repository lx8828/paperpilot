# facet / 子查询体检（50 篇口径，2026-09-30）

零 LLM ｜ 静态：`_r2_subq_static.py` ｜ 动态：`_r2_subq_diag.py` ｜
产物 `results/R2_SUBQ_{STATIC,COMPLEMENT,CONTRIB,FACET}.csv`、`R2_FACET_TYPE.csv`

**口径**：语料 `corpus50/`（50 篇/簇）｜切块 `prodchunk50/mineru`｜真值 `gold_final3.csv`（28 组合）｜
k=13，篇分 = **篇内块分 max** ｜路 = `zh-dense` ×1 ｜ `para-{dense,bm25}` ｜ `sq{1,2,3}-{dense,bm25}`

---

## ① 子查询静态质量

| facet | 条数 | 含中文 | token 中位 | ↔anchor Jaccard | **↔gold 正则重合** | 标记 |
|---|---|---|---|---|---|---|
| `ablation` | 3 | 0 | 6 | 0.178 | 6.1% | · 干净 |
| `annotation_cost` | 3 | 0 | 7 | 0.169 | 13.9% | · 干净 |
| `case_study` | 3 | 0 | 3 | 0.340 | 33.3% | · 干净 |
| `code_release` | 3 | 0 | 4 | 0.278 | 22.2% | · 干净 |
| `efficiency` | 3 | 0 | 5 | 0.274 | 16.7% | · 干净 |
| `error_analysis` | 3 | 0 | 4 | 0.306 | 40.0% | · 干净 |
| `fine_tuning` | 3 | 0 | 5 | 0.145 | **0.0%** | · 零重合 |
| `human_eval` | 3 | 0 | 2 | 0.289 | 29.2% | · 干净 |
| `knowledge_distill` | 3 | 0 | 8 | 0.368 | **45.8%（最大 62.5%）** | ⚠️ **泄露偏高** |
| `multilingual` | 3 | 0 | 8 | 0.226 | 25.0% | · 干净 |
| **`prompt_eng`** | **0** | — | — | — | — | ❌ **无子查询** |
| `proof_theory` | 3 | 0 | 4 | 0.238 | **0.0%** | · 零重合 |
| `significance` | 3 | 0 | 6 | 0.159 | 19.0% | · 干净 |
| `zero_few_shot` | 3 | 0 | 3 | 0.254 | 14.3% | · 干净 |

**泄露基线**：第一版用 `anchor` 做查询时该值为 **63%**（判泄露）。
全体子查询：**均 20.4% ｜ 中位 25.0% ｜ 最大 62.5%** → 多数干净，但 **`knowledge_distill` 已达锚点量级 → 该 facet 召回数字不可信**。

**语言纯度**：含中文子查询 **0 条** ✓（>0 会让该路 BM25 恒 0）。

**互补性**（同 facet 内两两 Jaccard，>0.4 视为冗余）：

| facet | 中位 | 最大 | |
|---|---|---|---|
| **`case_study`** | 0.200 | **0.667** | ⚠️ **两条几乎同义（冗余）** |
| `human_eval` | 0.250 | 0.333 | · 互补 |
| 其余 11 个 | 0.000~0.250 | ≤0.250 | · 互补 |

---

## ② ★★ 逐路贡献（本节**推翻** `R2_SCORE_VS_RANK` 的权重结论）

k=13，篇分 = 篇内 max；`独占gold` = 本路 top-13 里**其余所有路**都没召回到的 gold

| 路 | 种类 | **StRecall** | 纯度 | 超随机 | 独占 gold 合计 | 占总 gold |
|---|---|---|---|---|---|---|
| `zh-dense` | dense | 0.448 | 0.409 | +0.188 | 7 | 1.9% |
| `para-dense` | dense | 0.445 | 0.412 | +0.185 | 4 | 1.1% |
| **`para-bm25`** | bm25 | **0.381** | 0.352 | +0.121 | 11 | 3.0% |
| `sq1-dense` | dense | **0.475** | 0.433 | +0.215 | 5 | 1.4% |
| `sq2-dense` | dense | 0.444 | 0.393 | +0.184 | 1 | 0.3% |
| `sq3-dense` | dense | 0.443 | 0.416 | +0.183 | 2 | 0.5% |
| **`sq1-bm25`** | bm25 | **0.529** ★ | **0.496** ★ | **+0.269** ★ | 7 | 1.9% |
| `sq2-bm25` | bm25 | 0.482 | 0.422 | +0.222 | 6 | 1.6% |
| **`sq3-bm25`** | bm25 | 0.473 | 0.405 | +0.213 | **8** | 2.2% |

### ★ 结论 ②-1：**BM25 不弱 —— 是 `para` 这条查询弱**

| | StRecall@13 |
|---|---|
| `para-bm25`（用英文**检索式**做 BM25 查询） | **0.381**（全 9 路最弱） |
| `sq{1,2,3}-bm25`（用**子查询**做 BM25 查询） | **0.473 ~ 0.529**（**全 9 路最强**） |

**`sq1-bm25` 是全部 9 条路里最强的单路**（StRecall 0.529、纯度 0.496）。

**机制**：`para`（`F2[facet][3]`）是**面向 dense 的语义式**（例 `our implementation can be downloaded so others can reproduce the results`），
它的**词面不一定出现在论文里**；而子查询是**照"论文里会怎么写"造的陈述句**
（例 `we release our source code and datasets`）→ **词面匹配度高** → BM25 才发挥出来。

### ★ 结论 ②-2：因此 `ρ = 4:1`（dense 主导）**建立在一个错误前提上**

`R2_SCORE_VS_RANK` 里得出"给 BM25 权重越大越差"（0.408 → 0.371 → 0.348），
**当时 BM25 的唯一查询是 `para`** → 给它权重 = **给一个弱通道权重**。
换成子查询后 BM25 变成**最强通道之一** → **权重必须重标**（很可能不再是 4:1）。

### 结论 ②-3：子查询的**独占贡献**同样以 BM25 路最大

| 路 | 独占 gold | 有贡献的题数 |
|---|---|---|
| `sq3-bm25` | **16** | 12 |
| `sq2-bm25` | **14** | 11 |
| `sq1-bm25` | 13 | 8 |
| `sq1-dense` | 8 | 8 |
| `sq3-dense` | 6 | 6 |
| `sq2-dense` | **3** | 3 |

→ `sq2-dense` 近乎冗余（独占 3）；**BM25 路是主要的额外召回来源**。

---

## ③ 逐 facet 分型（`bm25最强 vs dense最强`）

**汇总：`bm25` 赢 16 ｜ `平` 7 ｜ `dense` 赢 4** → **绝大多数 facet 上 BM25 最强**。

| 谁赢 | facet（簇） |
|---|---|
| **bm25** | 1 annotation_cost / 1 case_study / 1 code_release / 1 error_analysis / 1 human_eval / 2 ablation / 2 code_release / 2 efficiency / 2 knowledge_distill / 2 significance / 2 zero_few_shot / 3 ablation / 3 code_release / 3 efficiency / 3 fine_tuning / 3 proof_theory |
| 平 | 1 ablation / 1 significance / 1 zero_few_shot / 2 error_analysis / 2 fine_tuning / 2 multilingual / 3 case_study |
| **dense** | 1 fine_tuning / 2 annotation_cost / 2 human_eval / 3 significance |

→ **"不该全局一个 ρ"得到证实**：至少 4~7 个 facet 该给 dense 更大权重，16 个该给 bm25 更大。
（逐 facet 建议权重见 `results/R2_FACET_TYPE.csv`。）

---

## ④ 需要修的子查询问题（3 条）

| # | 问题 | 影响 | 修法 |
|---|---|---|---|
| 1 | **`prompt_eng` 无子查询** | 该题（簇2）在多查询里**直接缺 3 路** | 补生成（连同下文 per-cluster 一起做） |
| 2 | **`knowledge_distill` 泄露 62.5%**（≈ 锚点基线 63%） | 该 facet 召回数字**不可信** | 走 `desens` 净化（删掉与 gold 正则重合的词）后复测 |
| 3 | **`case_study` 两条子查询 Jaccard 0.667** | 一条冗余（3 路实际只有 2 种措辞） | 重新生成第 3 条（换侧面） |

---

## ⑤ 对后续实验的影响（顺序调整）

1. **权重 ρ 必须用"子查询做 BM25 查询"重标**（旧结论作废）；
2. **`para` 的角色要重新定位**：它在 dense 路（0.445）与 `zh`（0.448）几乎相同 →
   可考虑**去掉 `para`**，只留 `zh` + 3 条子查询（省 1/4 的列表）；
3. **`sq2` 值得重生成**（dense 路独占仅 3；互补性虽好但贡献低）；
4. 补 `prompt_eng` 子查询 + 净化 `knowledge_distill` + 修 `case_study` 冗余；
5. 然后再跑 **`d`（每路截断深度）× ρ × 段级聚合** 网格 —— 此时 ρ 的搜索范围要按分型放宽（**不再假设 dense 主导**）。

---

## 边界

1. 单次运行、28 题；**本节的单路差值（如 0.381 vs 0.529）远超 σ≈0.01**，但逐 facet 的"谁赢"（差 1~3pt 的那些）**需 n≥3 复跑**。
2. `独占gold` 是**相对固定 top-13** 算的；k 变大时独占数会变（k=20 时各路的池更大，独占会减少）。
3. 篇分用 **max**；`R2_RECALL_AGG` 那套（top-n 均值/求和）尚未跑。
4. `↔gold正则重合` 用的是**词面 token 重合**，不等于"语义泄露"；语言必然的那部分无法完全剔除，
   故 `desens` 臂只作**下界**。
5. 子查询是**按 facet 存、三簇共用**（`subqueries.json` 无簇维度）→ 本节未检验"per-cluster 措辞是否更好"。
