# 换切块：从"固定窗 1000/900"换成**生产章节段落切块**（MinerU 口径）

日期：2026-09-29 ｜ 脚本：`retrieval/tmp/_r2_arxiv_map3.py`、`_r2_fetch_finish.py`、`_r2_mineru_run.py`、`_r2_parse_pdfs.py`、`_r2_retr_eval.py`

## 0. 背景：为什么原来的语料用不了生产切块

R2 语料原先直接取 **LitSearch 语料的 `full_paper` 字段**（`_r2_build_corpus.py:41`）。
实测（`_litsearch_struct_probe.py`）该字段是 **PDF 抽取的行转储**：
· 疑似章节标题仅 **2.8 个/篇**（真实论文 8~15）→ **章节树不可靠**
· 没有版面（无行高/坐标）→ `chunk_document` 的标题识别（依赖字号）失效

→ 所以旧脚手架只能用 `chunks_of()`（固定窗 1000/900），**与生产切块无关**。
→ 正路（本次做的）：**下载 60 篇的真 PDF → 跑生产解析（MinerU）→ 生产切块**。

## 1. Step 0：把语料从"LitSearch 文本"改锚到"真 PDF"

| 步骤 | 做法 | 结果 |
|---|---|---|
| ① corpusid → arXiv id | **S2 batch 端点**（`POST /paper/batch`，1 次请求最多 500 id） | **38/60** 有 arXiv（36/38 标题相似度 1.00） |
| ② 下载 | `cli/run_fetch.py --ids-file`（37 篇，间隔 3s） | **37/37 成功** |
| ③ 补非 arXiv | `aclanthology.org` 直链（**跳过 acm/doi**：其 socket 会**挂住数分钟**，`urllib` timeout 管不住） | +10 篇 |
| ④ 无来源 | 无 arXiv/开放 PDF | **4 篇**（`C3P9/C3P16/C3P18/C3P19`，1990s 老 NLP） |
| **合计可解析** | | **47/60（78%）**｜簇 16/18/13 篇 |

⚠️ **度量陷阱（踩过）**：`dl.acm.org` / `doi.org` 会挂住 → 必须**按主机白名单跳过** + **硬墙钟超时**。

## 2. Step 1~2：MinerU 解析（**生产口径**）+ 生产切块

**为什么必须 MinerU**：生产默认 `PAPERPILOT_MINERU=1` / `PAPERPILOT_USE_MINERU=1` →
`ordered_chunks` = MinerU 骨架（**表格结构化、公式 LaTeX、图内文字不污染正文**）。
pymupdf 单独用会**丢表格数值**（仓库记录：pymupdf 把表撕碎）。

| 项 | 实测 |
|---|---|
| MinerU 版本 / 设备 | 3.4.5 ｜ `.venv-mineru` torch 2.8.0+cu128，`get_device() → cuda` |
| GPU 占用（单进程） | **1~20%（脉冲式）** ← `pipeline` backend 多阶段串行 + CPU 密集型前后处理 |
| **GPU 占用（2 并发）** | **76%** ← **开 2 进程才喂得饱**（6GB 显存，每进程 ≈1.9GB） |
| 47 篇耗时 | **20.6 分钟**（2 并发，均 ~26s/篇）｜**零失败** |
| 幂等 | `ingest.run_mineru()` 复用判定（产物 + 版本 + PDF 指纹）+ `mineru_progress.json` → **可续跑** |

**切块结果（生产口径）**：

| 臂 | 块数 | 块/篇 | 字符/块(中位) | 标题路径 |
|---|---|---|---|---|
| 固定窗 1000/900 | **1,068** | ~57 | 1,000 | **0 种**（无标题） |
| **生产章节段落（MinerU）** | **285** | **17~20** | **1,493~1,848** | **565 种** |
| pymupdf 骨架（对照） | 1,070 | ~22 | 1,850~2,383 | 630 种 |

· MinerU 状态：**`ok` 46 篇 / `degraded` 1 篇**（1 篇退回 pymupdf → 丢表值）
· **"注入块=0"是正确行为**（`document_cache.py:377`）：`USE_MINERU=1` 时 `ordered_chunks` **本身就是 MinerU chunks**（表格文本已在内），不再二次注入。已实测 `chunk c10` 含 `Table 1: Dataset statistics …` ✓

## 3. Step 3：三臂检索评测（隔离"文本源"与"切块"两个变量）

同 47 篇语料、**同真值**（PDF 全文上的 facet 正则命中）、同查询集（`[中文题面] + 子查询`）、
`max_seq_length=8192`（生产值）、35 个真值 / 均 gold **6.57 篇**。

| 臂 | 文本源 | 切块 | 块数 |
|---|---|---|---|
| `A_ls_fix` | LitSearch 纯文本 | 固定窗 | 1,037 |
| `B_pdf_fix` | **PDF** | 固定窗 | 1,068 |
| **`C_pdf_prod`** | **PDF** | **生产章节段落（MinerU）** | **285** |

### 3.1 篇级指标（多查询 `B mq_max`）

| 臂 | MRecall@5 | MRecall@10 | StRecall@5 | **StRecall@10** | 集合P@10 | **集合F1@10** | α-nDCG@10 |
|---|---|---|---|---|---|---|---|
| A_ls_fix | 0.200 | 0.257 | 0.482 | **0.784** | **0.506** | **0.581** | 0.828 |
| B_pdf_fix | 0.229 | 0.257 | 0.450 | 0.746 | 0.486 | 0.556 | **0.852** |
| **C_pdf_prod** | 0.171 | 0.229 | 0.414 | **0.748** | 0.480 | 0.552 | 0.826 |

**变量分解**：

| 变量 | StRecall@10 | 集合F1@10 |
|---|---|---|
| **文本源**（A→B：LitSearch → PDF） | **−0.037** | −0.026 |
| **切块**（B→C：固定窗 → 生产，**文本源不变**） | **+0.002** | −0.004 |

→ **换切块对篇级检索是"持平"**（StRecall@10 +0.2pt、F1 −0.4pt，均在噪声内）。

### 3.2 证据召回（★ 本次核心）

⚠️ **必须用"等字符预算"**：块数口径不公平（C 臂 285 块时"每篇 12 块"= 覆盖 62% 语料，
A 臂 1,037 块时同样 12 块只覆盖 18%）→ 块数口径下 C 的 +36pt 是**度量假象**。

| 臂 | @6块 | @12块 | **@6k字符** | **@12k字符** | 实选字符/篇@12k |
|---|---|---|---|---|---|
| A_ls_fix | 0.367 | 0.546 | 0.320 | 0.479 | 12,036 |
| B_pdf_fix | 0.384 | 0.540 | 0.295 | 0.459 | 12,028 |
| **C_pdf_prod** | 0.633 | 0.907 | **0.398** | **0.575** | 13,245 |

**等字符预算下（公平）**：

| 变量 | 证据召回@6k | 证据召回@12k |
|---|---|---|
| 文本源（A→B） | −0.025 | −0.020 |
| **切块（B→C，文本源不变）** | **+0.103** | **+0.116** |

→ **★ 换生产切块的真正收益：同样 12k 字符预算，证据召回 0.459 → 0.575（+11.6pt）。**
→ 原因：块 = **语义单元**（章节/段落），证据不会被固定窗从中间切断；
   固定窗下"命中块"常被切碎成 2~3 块、每块只有半句 → 覆盖不全。

### 3.3 成本

块数 **1,068 → 285（3.7× 更少）** → 向量编码 / 索引 / 检索候选数同步下降。

## 4. 结论

| 指标 | 换切块的效果 |
|---|---|
| StRecall@10 / 集合 F1@10 | **持平**（+0.2pt / −0.4pt） |
| **证据召回@12k字符** | **+11.6pt**（0.459 → 0.575） |
| 块数（成本） | **−73%**（1,068 → 285） |
| 标题路径（可解释性） | 0 种 → **565 种**（cites 能定位到章节） |

→ **换切块：篇级不亏、证据召回显著变好、成本降 3.7×。收益主要落在"证据质量"这一环。**

## 5. 诚实边界

1. **语料从 60 缩到 47 篇**：4 篇无 PDF；7 篇（acm/doi）被我**主动跳过**（会挂住）→ 可人工补
2. **可用 facet 数不变（35），但 gold 集合变了**：4 个 facet 正例率变化 >15pt
   （`簇1 fine_tuning` 70%→50%、`簇3 code_release` 55%→84.6%、`簇3 efficiency` 25%→46%、`簇3 safety_bias` 50%→69%）
   → **新数字不能与旧的 60 篇数字直接比**；三臂之间可比（同语料同真值）
3. **真值是"词面锚点"**，未重跑双 LLM 校准（`gold_recalib.csv` 基于旧块/旧文本 → **待重标**）
4. **1 篇 MinerU `degraded`**（退回 pymupdf，丢表值）
5. **只测了 dense 一路**；BM25/RRF 融合未在新块上复跑
6. `max_seq_length=8192` 是生产值；pymupdf 骨架下曾出现最长块 2,806 token（见 `embedder.py:227`）
7. **QAMPARI 线不受影响**（它是段落级、不切块）

## 6. 复现

```powershell
cd f:/paperpilot
$env:HF_HOME='F:\hf_cache'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'

# Step 0：语料改锚 + 下载
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_arxiv_map3.py      # S2 batch，1 次请求
./.venv/Scripts/python.exe -u cli/run_fetch.py --ids-file retrieval/data/r2dev/arxiv_map.ids.txt --interval 3
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fetch_finish.py    # ACL 白名单（跳过 acm/doi）

# Step 1~2：MinerU（可续跑）+ 生产切块
$env:PAPERPILOT_MINERU='1'; $env:PAPERPILOT_USE_MINERU='1'
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_mineru_run.py --workers 2    # ~21 min，可中断续跑
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_mineru_run.py --status
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_parse_pdfs.py                # → prodchunk/mineru/

# Step 3：三臂评测
./.venv/Scripts/python.exe -u retrieval/tmp/_r2_retr_eval.py
```

产物：`retrieval/data/r2dev/{arxiv_map,pdf_map,mineru_progress}.json`、
`pdftext/c{0,1,2}.parquet`、`prodchunk/mineru/c{0,1,2}.parquet`、
`results/R2_RETR_CHUNK_AB.csv`、`results/R2_FACET_60vs47.csv`。
