"""components：Modular RAG 组件门面层（过渡期薄命名，逻辑仍委派 agents/、tools/ 现有实现）。

背景与组件默认配置：docs/RAG_COMPONENT_NOTES.md。
目的：① 让代码组织对上业界组件名词（面试/协作/多篇扩展）；
      ② 每个组件的证据与默认配置写在其模块 docstring，避免重蹈"反复调试无提升"；
      ③ 门面先建、逻辑后搬——不重写、不破坏现行 agents/graph 入口。

组件顺序（业界参考管线）：
    DocumentSplitter → Router → QueryRewriter → Retriever → Reranker
    → ContextBuilder → Generator → Validator
"""
