# R2-dev 评测真值（**自建 · 不可再生**）

论文线（refined R2）的**评测真值**与题集，由人工 + 多判官标注得到。
**不能靠重跑脚本再生**（既定原则：**gold 不重标**）→ 因此**必须入库**。

> ⚠️ 读这份文件的人，通常是来回答一个问题：**"我现在该读哪个 gold？"**
> 答案在第 1 节，**别按文件名的新旧猜**。

---

## 1. 现役真值：按「语料口径」选

| 语料口径 | **真值** | 题集 | PDF 映射 | 语料 / 切块 |
|---|---|---|---|---|
| **20 篇**（3 簇 × 20 池 = 60，47 篇有 PDF） | **`gold_final2.csv`** | `facets_v2_selected.json`（28 组合） | `pdf_map.json` | `clusters/` + `prodchunk/mineru/` |
| **50 篇**（3 簇 × 50 = 150，含同领域干扰项） | **`gold_final3.csv`** | 同上 | `corpus50/pdf_map_all.json` | `corpus50/` + `prodchunk50/mineru/` |

选法：`cli/eval/run_*` 的 `--corpus 20|50`（默认 `20`）。
**两个都是现役**，只是口径不同 —— `gold_final2.csv` **不是**"旧版本"（它比 `gold_final.csv` 新）。

---

## 2. 三代 gold 的谱系（为什么有三个 `gold_final*`）

```
gold_final.csv ──(题集重选 29→28 组合 + 补三判官票数)──▶ gold_final2.csv ──(换语料 47→150 篇)──▶ gold_final3.csv
   第一代                                                    第二代·现役(20篇)                 第三代·现役(50篇)
```

| 文件 | 行 | 组合 | 正例 | 定位 |
|---|---|---|---|---|
| `gold_final.csv` | 458 | 29 | 137 | **第一代**（源 `gold_recalib2.csv`，35→29 题）。**已被取代**：题集换成 28 组合、且不带票数 → 仅作历史 |
| **`gold_final2.csv`** | 451 | 28 | 132 | **现役 · 20 篇口径**：三判官多数票（列 `votes/A_yes/B_yes/C_yes`） |
| **`gold_final3.csv`** | 1400 | 28 | 364 | **现役 · 50 篇口径**：同协议、换语料 |

---

## 3. 推导链（复核「为什么这题算正例」用）

```
锚点(正则命中)
  → gold_recalib.csv         v1 真值（35 组合）              ← 仅 --legacy 路径读
  → gold_recalib2.csv        锚点 vs 新真值（57 组合）
  → gold_recalib3.csv        A/B 判官标签
  → gold_adjudicate_gold_recalib3.csv   第三判官 C
  → calib2 / _3judge / _4judge.csv      NLI / LLM / DS / ST 判官校准（28 组合）
  → gold_final2.csv          多数票裁决 ← **默认路径读这里**
```

★ `gold_recalib*.csv` 是**中间产物**。只有 `cli/eval/run_*` 的 **`--legacy`** 会读
`gold_recalib.csv`（`_r2_reader.load_gold()`）；**默认走 `load_gold2()` → `gold_final2.csv`**。

---

## 4. 入库策略（本目录 27.9MB，只有 1.3MB 入库）

**入库**：三代 gold、推导链、题集、子查询、映射、校准表（24 个文件，约 1.3MB）。

**不入库**（可再生 / 体积大 / 缓存）：

| 内容 | 体积 | 重建方式 |
|---|---|---|
| `*_evidence.json` ×3 | 11.6MB | 按 gold 的 `ev_idx` 从 `prodchunk*/` 重抽 |
| `corpus20.parquet` | 0.5MB | 建索引时生成 |
| `clusters/` `corpus50/` `pdftext*/` `prodchunk*/` | 14.3MB | 语料切块与文本，从 PDF 可重生成 |
| `mt_cache.json` `mineru_progress*.json` `nli_calibration*.csv` | 0.5MB | 运行缓存 |

---

## 5. ⚠️ 踩过的坑（别再踩）

1. **本目录曾整体被 ignore**（`.gitignore` 的 `retrieval/data/`）→ 干净 clone 拿不到真值，
   **论文线评测完全无法复现**。2026-10-05 改为「挡可再生大件、放行手工标注真值」。
2. **`git check-ignore` 不能用来验证 `!` 反选** —— 反选命中时它**仍会打印路径**。
   要验证请用：`git ls-files --others --exclude-standard retrieval/data`。
3. `retrieval/results/R2_GOLD_RECALIB2_20260929.md` 里写的
   「`_r2_reader.load_gold()` 仍读 `gold_recalib.csv`，待办 A1 接线」——
   **已闭合**：默认走 `load_gold2()`，旧口径退到 `--legacy`。**该行已过时。**
