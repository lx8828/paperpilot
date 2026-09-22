"""components：Modular RAG 组件门面层（过渡期薄命名，逻辑仍委派 agents/、tools/ 现有实现）。

背景与组件默认配置：docs/RAG_COMPONENT_NOTES.md。
目的：① 让代码组织对上业界组件名词（面试/协作/多篇扩展）；
      ② 每个组件的证据与默认配置写在其模块 docstring，避免重蹈"反复调试无提升"；
      ③ 门面先建、逻辑后搬——不重写、不破坏现行 agents/graph 入口。

组件顺序（业界参考管线）：
    DocumentSplitter → Router → QueryOptimizer → Retriever → Reranker
    → ContextBuilder → Generator → Validator

**2026-09-18：QueryRewriter → QueryOptimizer（查询侧优化）**
原 `query_rewriter.py`（门面）与 `tools/query_expand.py` 均为**死代码**（无任何 import），
已删除；真正生效的 `pull_chunk._rewrite_queries` 已抽出为 `query_optimizer.py` 的 L3。
新组件按"**L1 路由 + L2~L5 并列变换器**"组织（不是串行五步），目前仅 L3 实现，
其余四级的**设计约束与风险写在各自函数 docstring 里**（未实现会显式抛错，不做沉默失败）。
"""
