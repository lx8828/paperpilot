"""**集合问答节点**（`PAPERPILOT_QA_READER=set`）：「这 N 篇里，哪几篇做了 X」。

## 为什么单开一个 reader（而不是塞进 fullctx / retrieval）
`fullctx` 是"把语料一次性喂给模型，让它自己作答"；`retrieval` 是"挑块 → 组上下文 → 作答"。
两者都**产出一段答案文本**，但**不产出"哪些篇"这个集合**，也没有"逐篇判定"这一步。

实测（`retrieval/results/R2_READER_20260929.md`）：
    「一次看 K 篇」→ 篇级 集合 F1 **0.539**（precision 只有 0.457：把大量不该列的篇也列了）
    「逐篇判定」  → 篇级 集合 F1 **0.846**（precision 0.839，交付篇数还自适应）
→ 这类题的核心难点在**判定**，必须在**篇粒度**上单独过一次。

## ⚠️ 复合题必须开 `extract`（2026-09-29 实测，`R2_SC_VERDICT_20260929.md` §3.1）
生产的 M1 是「**集合筛选 ∧ 内容枚举**」的复合题：
    「这五篇里，哪些篇做了组件消融？**各自消融掉的是什么**？」
只判"哪几篇" → 生产 50 题上 **6~7/50（12~14%）**，而 `fullctx` 是 **30/50**；
诊断显示 **篇选对了（常 5/5）但答案里没有数字/专名** → 失败点是**内容缺失，不是判定**。
→ 对策 = `SYS_SET_EXTRACT`：判定之外多要一个 `extract`（该篇的具体内容，数字照抄）。
→ 开关：环境变量 **`PAPERPILOT_SET_EXTRACT=1`**（默认关，保持旧行为）。

## 输出契约
写 `set_papers`（`{answer, papers, n_papers, n_yes, unclear, b}`），**不写** `l3_chunks` / `fullctx`；
`generate_answer` 见到 `set_papers` 时**原样透传**（与 fullctx 分支同构，见 answer.py）。

`cites` 的形状与 fullctx 一致（`{pdf, chunk_id, page, section, evidence}`）—— 输出闸门
`validator.gate` 自己从 `cites[].evidence` 重建 entries（见 answer.py:535 的注释）。
"""
from __future__ import annotations

import time
from typing import Any

from paperpilot.agents.state import QAState
from paperpilot.components import set_judge


def answer_set(state: QAState) -> dict[str, Any]:
    """逐篇判定 → 篇集合 + 逐篇证据（见模块 docstring）。"""
    question = str(state.get("question") or "")
    pdfs = [str(p) for p in (state.get("pdfs") or []) if p]
    debug = dict(state.get("debug") or {})
    route = list(state.get("route") or [])
    if "SET" not in route:
        route.append("SET")

    if len(pdfs) < 2:
        debug["reader"] = {"path": "set", "n_papers": len(pdfs), "skipped": "need>=2_papers"}
        return {"set_papers": {"answer": "（集合问答需要至少 2 篇语料）", "papers": [],
                               "n_papers": len(pdfs), "n_yes": 0, "unclear": [], "b": 0},
                "route": route, "debug": debug}

    from paperpilot.agents.embedder import MultiChunkIndex

    t0 = time.time()
    idx = MultiChunkIndex(pdfs)
    # `extract` 由环境变量 `PAPERPILOT_SET_EXTRACT`（默认关）控制，见 `set_judge.run`
    res = set_judge.run(question, idx)
    answer = set_judge.render(question, res)
    cites = [{"pdf": p["pdf"], "chunk_id": p["chunk_id"], "page": p["page"],
              "section": p["section"], "evidence": p["snippet"]} for p in res["papers"]]
    out = {"answer": answer, "papers": res["papers"], "n_papers": res["n_papers"],
           "n_yes": res["n_yes"], "unclear": res["unclear"], "b": res["b"],
           "extract": res.get("extract", False),
           "seconds": round(time.time() - t0, 1), "labels": res["labels"]}
    debug["reader"] = {"path": "set", "n_papers": res["n_papers"], "n_yes": res["n_yes"],
                       "n_unclear": len(res["unclear"]), "b": res["b"],
                       "extract": res.get("extract", False),
                       "seconds": out["seconds"]}
    return {"set_papers": out, "cites": cites, "route": route, "debug": debug}
