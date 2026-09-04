Paperpilot 论文深度阅读与综述生成
简介：输入一篇或多篇论文，agent自动完成解析提取理解对比生成结构化报告
单篇精读，方法拆解，实验复现指南，核心公式提取
主题追踪，输入某主题，如LLM Agent Memory,自动爬取Arxiv，找到最近的论文，近半年十篇或二十篇，半年内数量不足再扩大时间范围，生成趋势综述，甚至发展情况等
多篇对比，输入三篇同主题论文，表格对比
<!--技术架构：
输入层：
  本地pdf，word上传pymuPDF（难点解析：pymu对于目前很多的双栏格式论文处理不优秀，甚至出现乱码，有要求更高，实现难点更大的minerU可以考虑）
  自动爬取Arxiv（针对爬取数据重复，论文v1v2v3的问题，直接用最新数据覆盖旧数据来达到数据库去重）
  已有论文库（自动摘要：论文自带的summary以及喂给llm生成的摘要，合并起来生成该论文摘要存入摘要库，论文全文存入论文库。做到冷热存储和两级查询）
LangGraph编排层：
  Parser Agent:PDF----结构化
  Chunker Agent:语义分块
  Claim Verifier:支持用户倒查论文原处，任何数据，结论，实验总结来自哪儿，并且可以回到原文
  Report generator:输出报告

状态定义 (State Schema)：
系统采用 LangGraph 的共享状态（Shared State）机制，各 Agent 节点通过统一的 State 字典进行数据传递与交换。核心状态字段设计如下：

| 字段名        | 类型      | 产生节点 | 消费节点          | 说明 |
| raw_text     | 文本       | Parser | Chunker           | PDF/Word 解析出的原始全文（含参考文献） |
| cleaned_text | 文本       | Parser | Chunker           | 截断参考文献后的纯净正文 |
| metadata     | 键值对     | Parser | Report            |标题、作者、发表时间、来源等元信息 |
| chunks       | 文本列表   | Chunker | analyzer, Verifier | 按语义切分后的文本块集合 |
| claims       | JSON 列表  | analyzer |  Verifier       |结构化提取结果 |
| verification | 键值对     | Verifier | Report           | 溯源结果，包含结论对应的原文位置引用 |
| final_report | 文本       | Report | 输出层             | 最终生成的综述报告或精读报告 |

MCP工具层：
  PDF解析：paperpilot
  Arxiv API
  代码执行（预留，概念设计）
  LateX渲染
基础设施：
  向量库（chroma）
  sqlite
 Langgraph Langsmith追踪
  图数据库Neo4j方法-数据集-指标关系（预留，概念设计）
  PostgreSQL 
  
目前mvp:
单篇精读，用户上传可解析的，排版合理的pdf，主要做的是Claim Verifier -->
## 技术架构

### 当前 MVP 范围
单篇精读：用户上传 PDF → 结构化提取 → 每个结论可溯源至原文页码。

&gt; 主题追踪、多篇对比、arXiv 自动爬取、论文库管理为后续扩展项，详见文末。

---

### 输入层
- **本地 PDF 上传**：基于 pymuPDF 解析文本与布局信息
  - 单栏：直接提取全文
  - 双栏：基于文本块 x 坐标分布的启发式分栏策略，按列合并后输出顺序文本
  - 复杂版式（图文混排、公式环绕）作为后续优化项

---

### LangGraph 编排层（5 节点）

| 节点 | 职责 | 输入 | 输出 |
|------|------|------|------|
| Parser | PDF 解析 + 元信息提取 | 上传的 PDF 文件 | raw_text, metadata, page_map |
| Chunker | 语义分块 + 页码锚定 | raw_text | chunks（含 chunk_id, page, bbox） |
| Analyzer | 结构化提取（方法/实验/结论/公式） | chunks | claims（每项绑定 chunk_id） |
| Verifier | 溯源验证：claim ↔ 原文比对 + 置信度打分 | claims + chunks | verification |
| Reporter | 报告渲染 | verification + metadata | final_report（Markdown） |

&gt; **核心设计**：Analyzer 负责"提取"，Verifier 负责"验真"。每个 claim 必须携带 chunk_id，Verifier 根据 chunk_id 精确定位原文片段，解决 LLM 在学术场景下的幻觉问题。

---

### 状态定义（State Schema）

| 字段名 | 类型 | 产生节点 | 消费节点 | 说明 |
|--------|------|----------|----------|------|
| raw_text | 文本 | Parser | Chunker | PDF 解析出的原始全文（已去除参考文献） |
| metadata | 键值对 | Parser | Reporter | 标题、作者、页码数、文件来源 |
| page_map | 列表 | Parser | Verifier | 页码与文本块的映射表，支撑精确溯源 |
| chunks | 列表 | Chunker | Analyzer, Verifier | 语义分块，每项含 {chunk_id, text, page, bbox} |
| claims | JSON 列表 | Analyzer | Verifier | 结构化提取结果，每项含 {claim, type, chunk_id} |
| verification | 键值对 | Verifier | Reporter | 溯源结果：{claim, source_text, page, confidence, reason} |
| final_report | 文本 | Reporter | 输出层 | Markdown 格式的精读报告 |

---

### 工具层
- **PDF 解析**：pymuPDF（文本提取 + 布局分析）
- **LLM 调用**：Ollama 本地模型（开发/测试）/ 兼容 OpenAI 格式 API（演示）

---

### 基础设施
- **SQLite**：论文元数据、解析状态、历史记录持久化

---

### 未来扩展（预留接口，当前未实现）

| 功能 | 当前状态 | 实现思路 |
|------|----------|----------|
| 主题追踪 | 预留 | 接入 arXiv API + 定时任务，按主题自动拉取近半年论文 |
| 多篇对比 | 预留 | 基于单篇 Analyzer 输出的 claims，做跨论文维度对齐 |
| 论文库 & 冷热存储 | 预留 | 摘要库（SQLite）+ 全文库（文件系统），两级查询 |
| 向量语义检索 | 预留 | 论文量大时接入 Chroma，支持跨论文语义搜索 |
| 方法-数据集-指标图谱 | 预留 | Neo4j 关系图谱，当前用 JSON 文件模拟 schema |
| 复杂版式解析 | 预留 | 评估 minerU 替代 pymuPDF 的 ROI |