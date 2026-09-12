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
