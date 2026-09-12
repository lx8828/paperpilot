# 闸门兜底题诊断（2026-09-11 01:35）

> 来源 `qa/qasper_run_20260911_005347.json`｜闸门兜底 10 道｜逐题复跑并打印判定

================================================================================================
## 1. 1912.01673｜How many sentence transformations on average are available per unique sentence in dataset?
- gold: `27.41 transformation on average of single seed sentence is available in dataset.`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=10 cites=0 t=21s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 0.069, 0.07
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（232 字）: 你问的“每个唯一句子平均有多少个句子变换”对应论文中 COSTRA 1.0 数据集的规模统计。论文给出：第二轮共收集 293 条标注，经 Korektor 处理后形成 4262 个唯一句子（含 150 个种子句）[1]。据此可推算平均每个唯一句子对应的变换数约为 293 ÷ 4262 ≈ 0.069，即约 0.07 个变换/唯一句子。不过论文本身并未直接给出“平均每句变换数”这一指标，上述数值是由 293 条标注与 4262 个唯一句子两个数字推算得出[1]。
- 草稿里 validator 提取到的数字: `['0.069', '0.07', '1', '150', '293', '4262']`

================================================================================================
## 2. 1909.00338｜Which features do they use to model Twitter messages?
- gold: `word unigrams, bigrams, and trigrams`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=12 cites=0 t=9s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 15000
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（225 字）: 论文用于建模 Twitter 消息的特征是词级 n-gram：他们抽取了 word unigrams、bigrams 和 trigrams 作为特征 [4]。这些特征被编码为二元值，即某特征在消息中出现则设为 1，否则设为 0 [4]；训练时只保留出现频率最高的前 15,000 个特征，其余全部移除 [4]。此外，分词使用 Ucto 并针对 Twitter 做了配置，token 被小写化，同时保留标点、emoji 和 emoticons [4]。
- 草稿里 validator 提取到的数字: `['15000']`

================================================================================================
## 3. 1901.11117｜What is in the model search space?
- gold: `Our search space consists of two stackable cells, one for the model encoder and one for the decoder  Each cell`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=12 cites=0 t=8s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 72999999999999996761335927885055692026574247739994753929654390585025993173485953518437935065046617233336704457244672
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（643 字）: 你问的“模型搜索空间”对应论文中为进化搜索定义的编码空间。该搜索空间由两个可堆叠的 cell 组成，一个用于编码器、一个用于解码器，每个 cell 包含 NASNet 风格的 block，每个 block 接收两个隐藏状态输入并产生新的隐藏状态输出[1]。编码器含 6 个 block、解码器含 8 个 block，从而能精确表示 Transformer[1]。搜索空间包含五个 branch 级搜索字段（input、normalization、layer、output dimension、activation）、一个 block 级搜索字段（combiner function）和一个 cell 级搜索字段（number of cells）[1]。子模型的基因编码为 [left input, left normalization, left layer, left relative output dimension, left activation, right input, right normalization, right layer, right relative output dimension, right activation, combiner function] × 14 + [number of cells] × 2，前 6 个 block 分配给编码器、后 8 个
- 草稿里 validator 提取到的数字: `['72999999999999996761335927885055692026574247739994753929654390585025993173485953518437935065046617233336704457244672']`

================================================================================================
## 4. 2001.07263｜How much bigger is Switchboard-2000 than Switchboard-300 database?
- gold: `Switchboard-2000 contains 1700 more hours of speech data.`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=12 cites=0 t=7s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 6.67, 1700
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（278 字）: 你问的是 Switchboard-2000 相对 Switchboard-300 的规模差距：论文中 Switchboard-300 是 300 小时的英语会话语音任务 [6]，而 Switchboard-2000 是在 2000 小时的 Switchboard+Fisher 数据上训练的 [2]。因此两者相差 2000 − 300 = 1700 小时，即 Switchboard-2000 约为 Switchboard-300 的 6.67 倍（2000 ÷ 300）。论文还指出，增加这些训练数据使系统大幅提升，某些情况下相对提升约 30% [2]。
- 草稿里 validator 提取到的数字: `['6.67', '30', '300', '1700']`

================================================================================================
## 5. 2001.07209｜Which datasets are used in the paper?
- gold: `Google N-grams
COHA
Moral Foundations Dictionary (MFD)`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=12 cites=0 t=10s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 4.1e+08, 850000000000
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（402 字）: 论文使用的数据集/语料如下：历史词嵌入方面用了两个英语历时语料——Google N-grams（8.5×10¹¹ tokens，1800–1999）和 COHA（4.1×10⁸ tokens，1810–2009）[4]；词汇资源方面使用了 Moral Foundations Dictionary（MFD）作为道德情感种子词来源[7]，并采用约 14,000 个英语词的效价评分来选取道德无关种子词[7][3]。人类判断验证方面使用了 Pew Research Center 2013 Global Attitudes survey（40 个国家、8 个话题）[1]，以及约 14,000 个英语名词的效价评分[3]。此外，大规模检索分析选取了 1800–1999 年间词频最高的 10,000 个名词（以 WordNet 名词标签校验）[9]，并使用具体性（concreteness）评分规范[11]。
- 草稿里 validator 提取到的数字: `['1800', '1810', '10000', '14000', '4.1e+08', '850000000000']`

================================================================================================
## 6. 2001.07209｜Which dataset sources to they use to demonstrate moral sentiment through history?
- gold: `（空 / 无答案）`
- 原判: score=1 status=fail unanswerable=True
- 复跑: action=**fallback** level=L3 n_entries=12 cites=0 t=14s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 4.1e+08, 850000000000
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（305 字）: 你问的“用于展示历史道德情感的数据集来源”，对应论文中训练历时词嵌入所用的两个历史英语语料库：Google N-grams 和 COHA [7]。Google N-grams 是取自英语文学（Google Books，全类型）的约 8.5 × 10^11 词次语料，时间跨度为 1800–1999 年 [7]；COHA 是较小的约 4.1 × 10^8 词次语料，选自体裁均衡、代表 1810–2009 年美国英语的作品 [7]。论文将历史时间划分为十年一档，并分别在这两个语料上训练词嵌入 [7]。此外，在后续应用中作者使用的是在 Google N-grams 上训练的历时嵌入，因为其评测表现更优 [2]。
- 草稿里 validator 提取到的数字: `['1800', '1810', '4.1e+08', '850000000000']`

================================================================================================
## 7. 2004.03925｜Do the authors give examples of positive and negative sentiment with regard to the virus?
- gold: `no`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=9 cites=0 t=8s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 15, 24, 29, 54, 60
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（150 字）: 否。论文只报告了情感极性的统计比例（如普通公众推文中约 54% 中性、29% 正面、15% 负面 [1]；WHO 推文中约 60% 正面、24% 中性、15% 负面 [1][2]），并说明极性值大于 0 为正面、小于 0 为负面、等于 0 为中性 [3]，但并未给出任何具体的正面或负面情感文本示例。
- 草稿里 validator 提取到的数字: `['15', '24', '29', '54', '60']`

================================================================================================
## 8. 1906.03338｜How do they demonstrate the robustness of their results?
- gold: `performances of a purely content-based model naturally stays stable`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=11 cites=0 t=24s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 100
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（346 字）: 你问的“他们如何证明结果的稳健性”，对应论文中通过改变测试实例的上下文来检验模型表现的两组实验。具体做法是：在 randomized-context 设置中，系统性地随机交换测试实例的上下文；在 no-context 设置中，直接删除待分类 ADU 周围的上下文 [5]。结果显示，在标准设置下，能访问上下文（含仅访问上下文）的模型总是优于基于内容的模型；但一旦随机翻转或完全抑制上下文，情况反转，基于内容的模型总是优于其他模型，某些类别（support）的差距可超过 50 个 F1 百分点 [5]。此外，ANOVA 特征 F 分数也显示，来自 EAU 周围上下文的特征得分远高于来自 EAU span 的特征，进一步证明模型在有选择时会强烈依赖上下文线索而忽视 EAU 本身内容 [5]。
- 草稿里 validator 提取到的数字: `['100']`

================================================================================================
## 9. 2003.04967｜What experimental evaluation is used?
- gold: `root mean square error between the actual and the predicted price of Bitcoin for every minute`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=12 cites=0 t=15s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 2.7
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（402 字）: 你问的“experimental evaluation”对应论文中 KryptoOracle 的实验评估部分。实验用 PySpark v2.3、Jupyter notebooks 和 Python 2.7 编写，在 SOSCIP 云上 32 GB RAM、8 CPUs、120 GB HDD、Ubuntu 18.04 的服务器实例上运行了 30 天 [4]。数据方面，Twitter 数据用 #Bitcoin 和 #BTC 标签采集 14 天，Bitcoin 实时价格用 Cryptocompare API 同步采集 [12]；项目整体采集数据约 30 天，Bitcoin 每天约 25000 条推文、约 10 MB 数据 [7]。评估指标是实际价格与预测价格之间的 RMS 误差，运行一天后整体 RMS 误差为 10 美元 [12]，并在实验末尾展示了 5 小时内的 RMS 误差（以美元计）[11]。
- 草稿里 validator 提取到的数字: `['2.3', '2.7', '18.04', '120', '25000', '10000000']`

================================================================================================
## 10. 1809.01060｜What were the results of the first experiment?
- gold: `Best performance achieved is 0.72 F1 score`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**fallback** level=L3 n_entries=9 cites=0 t=11s
- 触发: `sev=high type=number`
  - 句: 
  - 由: 疑似无证据支撑的数字: 0.81
- 闸门后答案: 抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

- **闸门前草稿**（348 字）: 你问的“第一个实验”对应论文中的 AMT 人类标注实验（对 200 个含上下文句子对进行隐喻转述贴切度评分）[1][2]。其核心结果是：上下文对贴切度评分产生“压缩效应”，即把评分拉向量表中间——低分被抬高、高分被压低[4][5]。具体地，71.1% 的脱离上下文时被评为 4 的隐喻在上下文中降为 3，而反向提升很罕见（仅 5% 的 3 分对升为 4）[5]；68.2% 的 1 分对在上下文中升为 2，仅 3.9% 的 2 分对降为 1[5]。中间档位基本稳定：9.8% 的 2 分对变为 3，10% 的 3 分对变为 2[5]。此外，上下文内与上下文外的人类平均评分之间 Pearson 相关系数为 0.81[5]，且转述与非转述的分界线（2 与 3 之间）未受上下文强烈影响[4][5]。
- 草稿里 validator 提取到的数字: `['0.81', '3.9', '5', '9.8', '10', '68.2', '71.1', '200']`


## 汇总

- 判定类型统计: `{'number': 10}`
- 复跑 action 分布: `{'fallback': 10}`
- 本轮成本: calls=52 prompt=0 completion=0 ≈ 0.00 元（高峰档）
