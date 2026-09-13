# 代码审查回应台账（2026-09-13）

> 外部审查逐条核验与处置。原则：**先复现、再定性、最后动手**；能实测的都实测，
> 不把"看起来像问题"直接当问题，也不把"设计如此"当挡箭牌。

## 条目 1：无引用、无依据的答案会被放行

**审查意见**：`validator.py:121` 把"答案完全没有 [n] 引用"记为 LOW；`gate()`（`:523`）只对
HIGH 强制修复或拒答；且 `entries` 为空时（`:445`）LLM 语义检查根本不执行。
实测一段完全虚构、零引用的答案调 `gate()` 返回 `{'action': 'pass', 'issues': [{'sev':'low',...}]}`。

### 复现与边界（我自己跑的）

用真实全篇 chunks（34 块）调 `gate()`：

| 情形 | 结果 |
|---|---|
| **纯文字编造 + 零引用**（有无裁决通道都一样） | **`pass`**，只有 `low/citation` ← 审查所指的洞 |
| 编造 + **显著数字** + 零引用 | **`fallback`**（`high/number`）← 数字层兜住了 |

**根因链**：`gate()` 用 **`cites` 构造 `entries`** → 零引用 ⇒ `entries` 空 ⇒
① LLM 语义体检整个不跑（`and entries`）② 数字层没有"被引证据"可比 ⇒ 只剩一条 LOW ⇒
`trigger` 只认 HIGH ⇒ `pass`。

更深的两个点：
1. **prompt 与闸门口径不一致**：`answer.py` 准则 14 是「**硬引用**……绝不让无引用支撑的断言留在答案里」，
   而闸门把零引用当"可忽略"（注释原文：*若为概述直答可忽略*）；
2. **闸门手上有 `route`（`l0_answer` / `global_retrieve`）却没用它**区分"合法的概述直答"与"证据型作答违规"。

### 真实分布：**没发生过**（575 题实查）

| 运行 | 零引用 | 备注 |
|---|---|---|
| 503 题全量（`qasper_run_20260911_072510`） | 4 条（0.8%） | **全部是闸门自己的兜底话术**（"未能通过内部事实校验"），全部 <4 分；`level=L0` 的 70 条**零引用 0 条** |
| 72 题 A 桶（并集两臂） | 2 条（2.8%） | 同上 |

→ 定性：**真·防御缺口（robustness hole），不是正在漏水的地方**。优先级中低，但"口径不一致"值得修。

### 处置：**已修（A 档，不改 gate 动作）**

| 改动 | 位置 |
|---|---|
| ① 零引用**独立成类型** `no_citation` | `validator.TYPES` |
| ② 机器判"含实质断言"（`_substantive_claim`，0 LLM 成本）→ **MID**；纯拒答/过短 → **LOW**；并带 `substantive` 标记 | `_machine_checks` |
| ③ 零引用**不再跳过检查** → 跑一次**格式体检** `_llm_uncited`（只判"该不该有引用"，无原文故不判真伪；自带守卫：非实质断言直接返回空、不花调用） | `check()` |
| ④ 落进运行记录：`validator_action` / `issues` / `no_citation` / `no_citation_substantive` | `cli/run_qasper_eval.py`、`qa/recall/_qasper_tbl_exp.py` |
| ⑤ **前端渲染** `issues`（"答案自检（AI 复核）"提示条 + flow 行自检计数）；`/api/ask` 回传瘦身后的 validator | `web/app.py`、`web/index.html` |

**硬约束（已回归验证）**：`gate()` 动作不变（仍只对 HIGH 拦）→ 不触碰拒答率；
`no_citation` 不进 MID 触发名单，故 `PAPERPILOT_VALIDATOR_REPAIR_MID=1` 下也不会把它变成拒答。

**验证**：`qa/recall/_selftest_validator_20260913.py` → 机器侧 12/12 全绿；
`PP_LLM=1` 时额外跑真实格式体检（识别出"1 句含具体断言但未标来源"，对拒答话术返回空）。

**未做（B 档，留待定）**：按 `route` 分流——`global_retrieve` + 零引用 + 含实质断言 →
一次 repair **强制补引用**（把准则 14 从"prompt 期望"变成"代码强制"）；`l0_answer` 仍允许零引用。

---

## 条目 2：文件名被当作论文身份 → 同名不同内容会**系统性**错误

**审查意见**：论文文件名被当成论文身份，缓存会产生错误结果，同名 pdf 会系统性错误。

### 复现（`qa/review/_dupname_repro_20260913.py`，两条路径都跑通）

身份来源：`Path(pdf).stem` **贯穿全部产物**——claims / summary / skeleton / figures / overview /
guide / report、`gvec`/`cvec`（+ `gidx`/`cidx`）、`out_mineru/<stem>/`、`ingest.json`。
**没有任何一层记录 PDF 的内容指纹**（grep `sha256|md5|st_size|st_mtime` 只匹配到 `lru_cache(maxsize=…)`）。

| 路径 | 操作 | 结果 |
|---|---|---|
| **流水线** | 把 `A.pdf` 换上 **B 的字节**（文件名不变） | **解析层读 B**（`chunk[0]` 变成 B 的文本）而 **claims/报告仍是 A**（`n_claims=116`、`n_groups=97` 未变）→ **状态混合**，全程无告警 |
| **用户可见** | `/api/report` 上传"文件名=A、内容=B" | **HTTP 200 返回 A 的报告**，且**磁盘上一个字节都没写**（sha 未变）→ 用户拿到**另一篇论文**的报告 |

**为什么评测没抓到**：QASPER 走虚拟名 `qasper_<pid>.qpdf`（身份由数据集给定，不存在"同名不同内容"）
→ **只有真实上传路径受影响**，而用户文件名恰恰多是 `paper.pdf` 这类通用名。

### 影响面（迁移成本估算）

`out_views` **6414** 文件、`out_claims` **479** 文件、`out_mineru` **46** 目录，全部按名寻址。

### 修法（三档）

| 档 | 做法 | 成本 / 风险 |
|---|---|---|
| **A（已做）** | **内容指纹 + 读取校验**：`paper_identity` 记 `sha256/size/mtime_ns`；`pipeline` / `ingest` 加身份门 → 不一致即**失效重建**（`--skip-llm` 下直接报错，不许装配别篇产物）；`run_mineru` 同样校验。历史产物按 **mtime 关系**迁移（产物比 PDF 新 → 收编并补写指纹；PDF 更新 → 判 stale），避免一次性废库。`document_cache` 的 `lru_cache` 键加入**文件版本**（`size:mtime_ns`），防同进程内改文件读旧解析 | 已落地；全库 66 篇：32 收编 / 34 无产物 / **0 被迫重建**（零爆炸半径） |
| **B（未做）** | 产物**内容寻址**（`by_hash/<sha12>/…`），文件名只用于展示；维护 name→hash 映射 | 大：迁移 6900+ 产物并改所有路径构造 |
| **C（已做）** | 上传按内容判定落点：**同内容 → 幂等复用**；**不同内容 → 另存 `<stem>__<sha8>.pdf` 当新论文**（原文件不动）+ `upload_note` 明确告知 | 小 |

### 处置：**A + C 已实现并验收**

- 新增 `src/paperpilot/paper_identity.py`（`fingerprint` / `check` / `strict`）；
- `pipeline._pdf_identity_stale` + `process_pdf` 身份门、`ingest` 入口校验与指纹写回、
  `run_mineru` 校验、`document_cache` 三处缓存按文件版本失效（保留 `cache_clear` 兼容）；
- `web.plan_upload` + `/api/report` 新落点逻辑；
- 自测 `qa/review/_selftest_identity_20260913.py`（**19 项全绿**）；
- 原复现脚本已转为**验收**脚本 `qa/review/_dupname_repro_20260913.py`：
  · 流水线：`[identity] … → 整链重建产物` 且 `--skip-llm` **被拒**（不再状态混合）；
  · web：**未返回旧报告**、另存 `2608.27843v1__89adb392.pdf`、**原文件未改动**、带提示文案。
- 回归：QASPER 虚拟名路径不受影响（真实问答冒烟通过）；三个自测全绿；`compileall` 0 错。
- 开关：`PAPERPILOT_PDF_ID_STRICT=1`（默认关）→ 把历史产物（`adopt`）也强制重建，用于存量库排雷。

**B 档（内容寻址目录）不做**，理由如上（成本/收益比）。
