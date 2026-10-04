# `retrieval/tmp/` —— 出题真源 + 评测/实验脚本

> 目录名看着像"临时"，其实装着**两类不可再生的东西**。读这份文件能少踩两个坑：
> ① 哪些**必须入库** ② 为什么这些脚本**没有搬到 `evals/`**。

## 一、这里有什么（四类，处置各不同）

| 内容 | 是什么 | 入库？ |
|---|---|---|
| `group*/<stem>.questions.json`、`group*/_group.questions.json` | **出题 gold（唯一真源）** —— 每题带逐字原文引文 | ✅ **必须入库** |
| `_*.py`（135 个） | 评测/实验脚本：**库 14 个 + 有出处的入口 ~120 个** | ✅ **入库**（2026-10-05 补） |
| `_archive/`（161 个 `.py`） | **一次性实验**的归档（无出处、无人引用） | ❌ gitignore |
| `_gold_history/` | 出题的草稿/备份（`*.bak` / `draft*` / `new*`） | ❌ gitignore |

**分类依据不是猜的**，是 `evals/checks/tmp_scripts_audit.py` 算出来的，结果落在
`evals/baselines/tmp_scripts.json`（含每个脚本被谁引用、被哪份报告提到）。可重跑复核。

## 二、⚠️ 哪些脚本是**库**（动之前先看这里）

这 14 个被别的脚本 import，**删/改会连带一大片**：

| 脚本 | 被引用 | 干什么 |
|---|---|---|
| `_r2_std_metrics.py` | **30** | R2 标准指标复算（MRecall / StRecall / α-nDCG / ALCE 引用） |
| `_r2_facets_v2.py` | **22** | R2 题集 v2（facet 定义与题面） |
| `_sandbox.py` | **15** | 检索侧**可复算沙盘**（把 encode+打分冻结到磁盘，之后全在冻结数据上复算） |
| `_qampari_run.py` | 8 | LoFT-RAG-QAMPARI 128k 检索 pipeline |
| `_r2_gold_recalib.py` | 8 | 真值校准（判官协议 `SYS` / `SYS_STRICT`） |
| `_scan_quota.py` | 5 | 跨篇 quota 扫描 |
| `_sweep.py` / `_xling_ab.py` | 各 4 | 参数扫描 / 跨语言 A-B |
| `_r2_calib_llm.py` | 3 | NLI/LLM 判官校准 |
| `_bge_m3` / `_r2_budget_model` / `_r2_prod_eval` / `_r2_reader` | 各 2 | 编码器 / 预算模型 / 生产评测 / reader 评测 |

其余 ~120 个是**入口**（有 `argparse`），其中**被报告引用过**的才算"有出处"——
报告会写"由 `_xxx.py` 产出"，那是"真跑过、有结论"的记录。

## 三、⚠️ 为什么这些脚本**留在 `retrieval/tmp/` 而没搬去 `evals/`**

**因为它们按「相对 `retrieval/`」定位**：

```python
HERE = Path(__file__).resolve().parents[1]        # = retrieval/
sys.path.insert(0, str(HERE / "scripts"))         # → retrieval/scripts（出题脚本）
sys.path.insert(0, str(ROOT / "retrieval" / "tmp"))  # → 本目录（库互相 import）
spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")  # 动态加载
```

实测（`_archive/_measure_locators.py`）：**133/135 个脚本用 `parents[N]`**、
88 个改过 `sys.path`、50 个用 `spec_from_file_location` 动态加载兄弟脚本。

**搬到 `evals/` 会让 `HERE / "scripts"` 指向 `evals/scripts` —— 语义变了，
不是换个目录名能修的**：得在 100+ 个文件里逐处判断"这个相对路径想要的是谁"。
零功能收益、全是风险，所以**不做**。

> 真要拆成 `evals/lib/` + `evals/runners/`，前提是**先改成包导入**
> （`import evals.lib.r2_std_metrics`），而不是字符串替换。那是独立的一次重构。

## 四、常用入口

```bash
# 分类器（只读，可重跑；确认"谁是库、谁是入口、谁该归档"）
python evals/checks/tmp_scripts_audit.py

# 出题流水线（见 retrieval/scripts/_qa_groups.py 的范式说明）
python retrieval/scripts/_validate_group_questions.py --group group2
python retrieval/scripts/_export_questions.py --group group2
python retrieval/scripts/_check_export_sync.py --group group2   # ⚠️ 跑批前必跑
```
