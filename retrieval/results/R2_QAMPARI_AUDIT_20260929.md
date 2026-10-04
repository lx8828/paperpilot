# R2｜指标审计（precision / 答案数 / 官方脚本核查）

日期：2026-09-29 ｜ 基准：LoFT `qampari/128k`（100 题）｜ 全部数字由**官方 LoFT CLI** 产出

---

## 0. 审计问题（用户提出）

1. **补报 precision**：统计每题平均预测答案数；若远超 gold 数（如 gold 8 / 列 40），则 recall 是虚的。
2. **有没有用官方仓库的 eval 脚本直接跑我们的输出文件**，而不是自己的重实现（`subspan_em` 等）。

---

## 1. 结论摘要

| # | 结论 | 证据 |
|---|---|---|
| 1 | **主方法没有 recall 灌水** | 预测 5.10 vs gold 5.02（1.016×）；P 0.830 ≥ R 0.813 |
| 2 | **但 `retr1m_*` 三个运行有 8~10pt 水分** | 预测 6.50 vs 5.02（1.30×）；P 0.637 < R 0.738，Δ **+0.101** |
| 3 | **我们用的是 LoFT 官方脚本，不是 HELMET** | `data/loft/official/`（google-deepmind/loft）；**LoFT 的 `multi_value_rag` 不产出 precision** |
| 4 | **HELMET 官方有 precision 尺子**：`eval_alce.py::compute_qampari_f1` | 含 `num_preds` + prec + rec + rec@5 + f1；已逐字移植 |
| 5 | **我的重实现与官方口径一致** | ALCE P/R 与集合式 P/R **全部 12 个运行差 0.000**；历史数字被官方 CLI 逐位复现 |
| 6 | **历史审计链有断裂（已修复）** | `official_audit_k40_ascii/` 缺 `preds.jsonl`；且目录命名与实际运行不匹配 |
| 7 | **★ 官方 coverage 跨运行不可比** | 空预测 80/100 的运行 coverage=**0.86（全表最高）**，但 em/subspan 最差 |

---

## 2. ① 答案数与 precision 审计

**关键事实：本数据集 gold 恰好 5~6 个/题（均值 5.02），不是 8 个、更没有 40 个。**

| 运行 | 空预测题 | gold 均 | 预测均 | **比值** | P | R | Δ(R−P) | 判定 |
|---|---|---|---|---|---|---|---|---|
| `exh_k40_k40`（主） | 1 | 5.02 | 5.10 | **1.016** | 0.830 | 0.813 | −0.018 | ✅ 1:1 |
| `audit_k40_k40` | 1 | 5.02 | 5.19 | 1.034 | 0.817 | 0.807 | −0.010 | ✅ 1:1 |
| `direct128k_k20` | 0 | 5.02 | 4.94 | 0.984 | 0.811 | 0.799 | −0.012 | ✅ 1:1 |
| `exh_k20_k20` | 0 | 5.02 | 4.81 | 0.958 | 0.834 | 0.787 | −0.047 | ✅ 1:1 |
| **`retr1m_k40_k40`** | 4 | 5.02 | **6.50** | **1.295** | 0.637 | 0.738 | **+0.101** | ⚠️ 多列换召回 |
| **`retr1m_ground_k40`** | 4 | 5.02 | 6.13 | 1.221 | 0.651 | 0.735 | **+0.084** | ⚠️ 同上 |
| `retr1m_verify_k40` | 5 | 5.02 | 5.84 | 1.163 | 0.620 | 0.675 | +0.055 | ⚠️ 略 |
| `retr_k20` | 1 | 5.02 | 3.50 | 0.697 | 0.820 | 0.591 | −0.229 | · R 被**低估** |
| `k40_noexh_k40` | 1 | 5.02 | 3.58 | 0.713 | 0.783 | 0.595 | −0.188 | · R 被**低估** |
| `exh_k40_ce_k40` | **80** | 5.02 | 0.97 | 0.193 | 0.180 | 0.172 | −0.008 | 坏运行 |
| `exh_k80_k80` | **74** | 5.02 | 1.19 | 0.237 | 0.237 | 0.222 | −0.015 | 坏运行 |

**读数**：
- **主结果（`exh_k40_k40`）的 recall 0.813 是实的** —— 预测条数与 gold 基本 1:1，precision 0.830 甚至略高于 recall。
- **`retr1m_*`（检索增强）确实靠"多列"换了召回**：多列 30%，precision 掉到 0.64，recall 里约 **+0.10 是水分**。→ 这三个运行的数字**不能直接与主结果并列**。
- **另两个方向相反**：`retr_k20` / `k40_noexh` 是**列不够**（0.70×）→ recall 被自身压住，属**低估**而非虚高。
- 之前 `_f1_selfcheck` 报的 F1（0.798~0.888）**建立在这份主预测上，无灌水问题**。

---

## 3. ② 官方脚本核查

### 3.1 我们用的是 **LoFT 官方**，且它**不输出 precision**

官方 `evaluation/rag.py::MultiValueRagEvaluation.evaluate()`（逐行核对）：

```python
instance_metrics['em']         = compute_em_multi_value(gold, pred)      # set 相等
instance_metrics['coverage']   = compute_coverage(gold, pred)            # |pred∩gold| / |gold|  ← 只有 recall
instance_metrics['subspan_em'] = compute_multi_value_subspan_em(...)     # 双向子串 + 匈牙利
# ← 多值分支**没有 f1 赋值** → aggregate 时 defaultdict 取 0.0
```

- 空预测分支只写 `em/subspan_em/f1`，**连 `coverage` 都不写** → 见 §4 的陷阱。
- 预处理：pred = `convert_to_str` → `normalize_answers`；gold = `normalize_answer`（NFD→lower→去标点→去冠词→空白归一）。**与我的重实现同源。**

**⇒ "补报 precision" 官方脚本做不到，必须自己算。**

### 3.2 HELMET 官方**有** precision 尺子

`princeton-nlp/HELMET::eval_alce.py::compute_qampari_f1`：

```python
prec  = sum([p in flat_answers for p in preds]) / len(preds)       # ← precision
rec   = sum([any(x in preds for x in a) for a in answers]) / len(answers)
rec5  = min(5, hits) / min(5, len(answers))
f1    = 2PR/(P+R)
num_preds = len(preds)                                            # ← 平均预测答案数
```

- 输出字段：`num_preds / qampari_prec / qampari_rec / qampari_rec_top5 / qampari_f1 / qampari_f1_top5`
- **`num_preds` 正是"每题平均预测答案数"**，官方自带该审计字段。
- 匹配是**归一化后的等值匹配**（列表 `in`），**不是子串**。
- ⚠️ **HELMET 的 QAMPARI 是 ALCE 版**（带引用、预测按逗号切分），与 LoFT 版**数据形态不同** → 脚本不能直接 drop-in，但 **P/R 算术是 QAMPARI 的官方精度定义**，已逐字移植。

### 3.3 交叉验证：我的重实现 = 官方

| 检验 | 结果 |
|---|---|
| ALCE 式 P/R vs 我的集合式 P/R（12 个运行） | **P、R 全部差 0.000** ✅ |
| 我的 LoFT em / coverage / subspan_em vs 官方 CLI | 逐位一致 ✅ |
| 官方 CLI 重跑 `audit/preds.jsonl` | 与存档**逐题 0 差异** ✅ |

### 3.4 审计链断裂（已修复）

| 存档位置 | 实际对应运行 | 问题 |
|---|---|---|
| `data/loft/official/audit/` | **`exh_k40_k40`**（em 0.44 / cov 0.8209 / subspan 0.70） | 目录名 `audit` 误导；实测是 exh 运行 |
| `results/official_audit_k40_ascii/` | **`audit_k40_k40`**（em 0.47 / cov 0.8148 / subspan 0.69） | **缺 `preds.jsonl`** → 无法回溯 |

**修复**：已把全部 12 个运行写成官方格式并用**官方 CLI** 跑一遍，产物落在
`results/official_runs/<run>/{queries.jsonl, preds.jsonl, preds_metrics.json, preds_metrics_per_line.jsonl}` → **每个数字都可一键复跑**。

**Windows 坑（记下）**：官方 CLI 用 `open(path)`（平台默认编码 GBK）读文件 → 预测必须 **ASCII 转义**（`json.dumps(..., ensure_ascii=True)`）写入，否则 `UnicodeDecodeError`。这就是目录名带 "ascii" 的由来。

---

## 4. ★ 顺带炸出的严重陷阱：官方 `coverage` 跨运行不可比

`coverage` 只在**非空预测行**上求均（空预测分支不写该字段）→ **空预测越多，coverage 越"好看"**：

| 运行 | 空预测 | coverage（官方报道） | coverage（全行口径） | em | subspan_em |
|---|---|---|---|---|---|
| **`exh_k40_ce_k40`** | **80** | **0.8600** ← **全表最高** | 0.1720 | **0.150** | **0.180** |
| **`exh_k80_k80`** | 74 | 0.8538 | 0.2220 | 0.180 | 0.180 |
| `exh_k40_k40`（主） | 1 | 0.8209 | 0.8127 | 0.440 | 0.700 |

→ **两次坏运行的 coverage 比主结果还高**，而 em/subspan 是最差的。
→ **coverage 单独报是危险的**；必须同时报 **空预测题数 + em + subspan_em**。
→ **ALCE 口径更安全**：空预测记 0、分母恒定 100。

---

## 5. 报数纪律（本次审计后确定）

1. **precision 一律用 ALCE 官方算术**（`compute_qampari_f1`），并**同时报 `num_preds`**。
2. **coverage 必须与空预测题数、em、subspan_em 同时出现**，禁止单独引用。
3. **凡"检索增强"类运行，先看 `预测数/gold 数` 比值**；> 1.2 的要在表里标注"含多列水分"。
4. **LoFT 指标与 ALCE 指标双报**：前者是我们跟的基准（可比文献），后者提供 precision（官方定义）。
5. 每个运行必须保留 `preds.jsonl` 才能回溯。

## 6. 复现命令

```powershell
cd f:/paperpilot
./.venv/Scripts/python.exe -u retrieval/tmp/_audit_precision_all.py   # 答案数 + 双口径 P/R
./.venv/Scripts/python.exe -u retrieval/tmp/_audit_official_runs.py   # 官方 CLI 跑全部运行
./.venv/Scripts/python.exe -u retrieval/tmp/_audit_final_v2.py        # 最终表 + 陷阱 + 对账
```

产物：`R2_QAMPARI_PRECISION_ALLRUNS.csv`、`R2_QAMPARI_OFFICIAL_ALLRUNS.csv`、`R2_QAMPARI_AUDIT_FINAL.csv`、`results/official_runs/`。

## 7. 边界

1. ALCE 的 QAMPARI **不是**我们用的 LoFT QAMPARI（前者带引用、按逗号切分）→ 只沿用其 **P/R 算术**，非其数据。
2. `smoke_k20` 只有 5 题，不参与结论。
3. `retr1m_*` 的"水分"是**相对于其自身 precision**而言；若产品允许"多列候选"，这不必然是缺陷，但**报数时必须声明**。
4. 真值仍是数据集官方 gold（非我们构造），但**未人工抽检**。
