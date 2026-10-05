"""**多答案开放域检索**（multi-answer open-domain retrieval）—— 独立模块。

## 这个模块是什么

输入 **一个问题**（如"哪些论文用了对比学习做跨语言摘要"），输出 **一份回答**：
先逐篇判「是/否做了这件事」（带证据片段 + 理由），再把判定结果**塞进 LLM 汇总**
（`synth.synthesize()`）产出**回答**；逐篇清单作为「依据」保留。

★ 它**不在**现有的"方向问题 → 取料 → 建索引 → 单篇精读"那条链路上 ——
  那条链路的产物是"一篇论文的深度报告"；本模块的产物是"**一批论文里哪几篇符合，并汇成一段回答**"。
  ★ 2026-10-06 起入口是工作台左下角的「🎯 多答案展示」→ **独立页 `GET /ma`**
  （**不再**与方向检索在起始页并列；那条并列入口已删除）。

## ★ 它不是新写的链路（重要）

生产实现**早已存在**，只是被藏在环境变量开关后、且没有独立入口：

| 环节 | 生产实现 | 在哪 |
|---|---|---|
| 多路检索 | `ChunkIndex.search_multiroute` | `paperpilot.agents.embedder` |
| 逐篇判定（多答案） | `set_judge.run()` | `paperpilot.components.set_judge` |

本模块做的是**薄适配**：把"任意一批语料"包成 `set_judge` 要的鸭子契约，
再给一个**独立入口**（不经方向流程、不依赖全局开关）。

★ **检索与判定的逻辑零分叉**：`corpus.py` 里 `search_multiroute` / `search_hybrid`
  是**直接赋值**在生产方法上的（不是复制实现的）。

## 用

    from paperpilot.multianswer import CorpusIndex, answer, sweep_n

    idx = CorpusIndex.from_parquet("retrieval/data/r2dev/prodchunk50/mineru/c0.parquet", n=50)
    res = answer("哪些论文用了对比学习", idx, n=30)
    print(res["n_yes"], [p["pdf"] for p in res["papers"]])
"""
from __future__ import annotations

from paperpilot.multianswer import synth
from paperpilot.multianswer.corpus import CorpusIndex, load_chunks_from_parquet
from paperpilot.multianswer.runner import answer, render, sweep_n

__all__ = [
    "CorpusIndex",
    "load_chunks_from_parquet",
    "answer",
    "render",
    "sweep_n",
    # ★ 2026-10-06 新增：把逐篇判定**汇成回答**（LLM）。`render` 仍是「依据」的产物。
    "synth",
]
