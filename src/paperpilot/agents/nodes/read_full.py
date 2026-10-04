"""**全文直读节点**（`PAPERPILOT_QA_READER=fullctx`，2026-09-27 起为**默认**问答读取器）。

## 为什么换掉 L3 检索（判据：S/C 不等式）

    S/C = 语料规模（token） / 上下文预算（token）
        · **S/C ≤ 1** → 全文放得下 → **直读优于"块级检索 + top-K 挑块"**（本文，5 篇 ≈ 48k token）
        · **S/C ≫ 1** → 检索是**物理必需**（线上篇级检索工具：54 万篇 ≈ 7.7 亿 token）

本层语料 = 用户给的那批论文，**S/C ≤ 1** → 不该检索。

⚠️ **预算按「字符」算，不按「篇数」算**（2026-10-02 实测修正）：
本文这批论文平均 **≈46k 字符/篇**，而**篇与篇差 1.6 倍**（38,060 ~ 60,606 字符/篇）。
旧注释写的「≤5 篇 ≈ 180k 字符 ≈ 48k token」**已过期** —— 实测 **180k 字符其实对应 4 篇**，
按现在的语料 **5 篇 = 230,691 字符 ≈ 61.5k token**（换算口径 3.75 字符/token）。
`fullctx.build_context` 与 `llm.chat_text` **都没有截断保护** → 超预算就是 API 报错。
所以现在有**两道**拦：`fullctx` 自己的 `ctx_budget_chars()`（默认 1.2M 字符，
`graph.ask` 与 `fullctx.answer` 各一层），以及上游 `direction.SELECT_MAX`（=10 篇）。
**都不是"模型窗口不够"** —— 窗口 1M token，10 篇 ≈ 58~62k，只占 ~6%。
实测（5 组 164 题 / M1 50 题，`qa/multi/_runs/fullctx_*`）：
    · M1：直读**仅答案 30/50** vs 检索 20/50；耗时 **4.3s vs 22.1s**；成本 **¥0.006 vs ¥0.026**
    · 可核性：答案里 99.5% 的数字能在原文找到、引用段号归属 100% 正确

## 与 RAG-2（`pull_chunk.search_l3`）的关系 —— **RAG-2 不删**

| 用途 | 走谁 |
|---|---|
| 「读论文」问答（本节点） | **`fullctx`（默认）**；`PAPERPILOT_QA_READER=retrieval` 一键切回 RAG-2（面试展示 / A/B 对照） |
| 问答链内部两处**独立重检索**（`answer._audit_absence` 缺失断言复核、`repairer._crag_regenerate` CRAG 修复） | **仍走 RAG-2** —— 那里要的是"证据块"而非整篇，且只在少数失败样本上触发 |
| 报告链（`pipeline.process_pdf` = claims / 骨架 / 报告） | **与检索无关**（只用切块 + claims 抽取）→ 完全不受影响 |

## 输出契约

本节点**不写 `l3_chunks`**（那是 RAG-2 的产物），只写 `fullctx`；`generate_answer` 见到
`fullctx` 即**直接返回**该答案（见 `answer._answer_from_fullctx`），不再走 `_build_context`。
"""
from __future__ import annotations

import os
import time
from typing import Any

from paperpilot.agents.state import QAState
from paperpilot.components import fullctx

# 问答读取器开关：`fullctx`（默认）| `retrieval`（= RAG-2）| `set`（集合问答）
ENV_READER = "PAPERPILOT_QA_READER"

# 摄取期"强制预建索引"的逃生开关（默认关 = 跟随读取器）。
# 需要"任何问题都不等现建"时置 1，退回 2026-10-03 之前的行为。
ENV_EAGER_INDEX = "PAPERPILOT_EAGER_INDEX"

# **需要向量索引（`.cvec.npy`）的读取器** —— 只有它们会做**块级检索**。
# ⚠️ `fullctx` **不在里面**：它读全文走的是 `ChunkIndex._doc_chunks()`（读**切块**产物），
#    一个向量都不用。见 `index_needed`。
_VECTOR_READERS = {"retrieval", "set"}


def index_needed() -> bool:
    """当前配置下，**摄取时要不要预先建向量索引**（`<stem>.cvec.npy`）。

    判据只有一个：**当前读取器会不会做块级检索**。
      · `fullctx`（默认）→ **False**：读全文只调 `_doc_chunks()`（切块产物），不需要向量；
      · `retrieval` / `set` → True：`search_l3` / `set_judge` 都是块级检索。
    置 `PAPERPILOT_EAGER_INDEX=1` 可强制预建（退回旧行为）。

    ⚠️ **返回 False ≠ "用不了/没有向量"** —— 索引是**惰性**的：任何真需要它的地方
    （`search_l3`、**L0 的缺失断言复核 `_audit_absence`**、`set_judge`、CRAG）
    调 `ChunkIndex(pdf).vectors()` 时会**现建现用**，只是那个问题会多等几秒。
    所以这里决定的是"**要不要提前替所有篇付这笔钱**"，而不是"能不能用"。

    （2026-10-03 改：此前无论什么读取器都在摄取期 eager 建，实测 5 篇 **21.8s**，
      而默认 fullctx 下这些向量**一次都不会被读到**。）
    """
    import os
    if (os.environ.get(ENV_EAGER_INDEX) or "").strip() not in ("", "0", "false"):
        return True
    return reader() in _VECTOR_READERS


def reader() -> str:
    """当前问答读取器名。容错：`rag2` / `l3` → `retrieval`；`set` / `sets` → `set`。

    · `fullctx`   全文直读（默认）：语料一次塞进单次调用 → 成稿答案
    · `retrieval` RAG-2 块级检索：挑块 + 组上下文 → 作答
    · `set`       **集合问答**：逐篇判定 → 「哪几篇做了 X」+ 逐篇证据
                  （见 `nodes/set_answer.py`；适用题形 = `哪几篇/哪些篇…做了 X`）
    """
    v = (os.environ.get(ENV_READER) or "fullctx").strip().lower()
    if v in ("retrieval", "rag2", "l3"):
        return "retrieval"
    if v in ("set", "sets"):
        return "set"
    return "fullctx"


def answer_fullctx(state: QAState) -> dict[str, Any]:
    """把语料（`state["pdfs"]` 全篇）一次性交给 LLM 作答（**单次调用**）。

    与 `search_l3` 的差别：不返回 `l3_chunks`，返回 `fullctx`（成稿答案 + 段号 cites）。

    ⚠️ 函数名**故意不叫 `read_full`**：那会与模块名同名，`from ... import read_full`
    会把**模块**（`nodes.read_full`）遮蔽成函数 —— 仓库惯例是"模块名 ≠ 函数名"
    （`pull_chunk.py`→`search_l3`、`answer.py`→`generate_answer`）。
    """
    question = str(state.get("question") or "")
    pdfs = [str(p) for p in (state.get("pdfs") or []) if p]
    debug = dict(state.get("debug") or {})
    route = list(state.get("route") or [])
    if "FULLCTX" not in route:
        route.append("FULLCTX")

    if not pdfs:                                   # 无语料：不当成错误，交给上层话术
        debug["reader"] = {"path": "fullctx", "n_papers": 0, "ctx_chars": 0,
                           "style": "base", "seconds": 0.0, "skipped": "no_corpus"}
        return {"fullctx": {"answer": "（没有指定任何论文，无法作答）", "cites": [],
                            "ctx_chars": 0, "n_papers": 0, "style": "base", "seconds": 0.0},
                "route": route, "debug": debug}

    t0 = time.time()
    out = fullctx.answer(pdfs, question, history=list(state.get("history") or []))
    out = {**out, "seconds": round(time.time() - t0, 1)}
    debug["reader"] = {"path": "fullctx", "n_papers": out.get("n_papers"),
                       "ctx_chars": out.get("ctx_chars"), "style": out.get("style"),
                       "seconds": out["seconds"]}
    return {"fullctx": out, "route": route, "debug": debug}
