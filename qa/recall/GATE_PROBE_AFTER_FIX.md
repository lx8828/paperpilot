# 闸门兜底题诊断（2026-09-11 01:59）

> 来源 `qa/qasper_run_20260911_005347.json`｜闸门兜底 10 道｜逐题复跑并打印判定

================================================================================================
## 1. 1912.01673｜How many sentence transformations on average are available per unique sentence in dataset?
- gold: `27.41 transformation on average of single seed sentence is available in dataset.`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=10 cites=1 t=17s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 根据证据，无法计算“每个唯一句子平均有多少个句子变换”这一数值。论文只给出了：COSTRA 1.0 共含 4262 个唯一句子（含 150 个种子句）[1]，以及每条标注需提供 1 个种子句及其全部 15 种变换[1]，但并未给出“变换总数”或“每个唯一句子的平均变换数”这类统计。因此缺少计算该平均值所需的分子（变换总数）或直接给出的平均值，无法作答。

- **闸门前草稿**（177 字）: 根据证据，无法计算“每个唯一句子平均有多少个句子变换”这一数值。论文只给出了：COSTRA 1.0 共含 4262 个唯一句子（含 150 个种子句）[1]，以及每条标注需提供 1 个种子句及其全部 15 种变换[1]，但并未给出“变换总数”或“每个唯一句子的平均变换数”这类统计。因此缺少计算该平均值所需的分子（变换总数）或直接给出的平均值，无法作答。
- 草稿里 validator 提取到的数字: `['1', '150', '4262']`

================================================================================================
## 2. 1909.00338｜Which features do they use to model Twitter messages?
- gold: `word unigrams, bigrams, and trigrams`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=2 t=4s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 论文用于建模 Twitter 消息的特征是词级 n-gram：他们提取了 word unigrams、bigrams 和 trigrams 作为特征 [4]。这些特征被编码为二元值，即某特征在消息中出现则设为 1，否则设为 0 [4]。训练时，除最高频的 15,000 个特征外，其余特征都被移除 [4]。此外，分词使用 Ucto，token 被小写化，标点、emoji 和 emoticons 被保留 [4]。论文也指出当前研究未尝试其他类型特征，词嵌入等留作未来工作 [2]。

- **闸门前草稿**（240 字）: 论文用于建模 Twitter 消息的特征是词级 n-gram：他们提取了 word unigrams、bigrams 和 trigrams 作为特征 [4]。这些特征被编码为二元值，即某特征在消息中出现则设为 1，否则设为 0 [4]。训练时，除最高频的 15,000 个特征外，其余特征都被移除 [4]。此外，分词使用 Ucto，token 被小写化，标点、emoji 和 emoticons 被保留 [4]。论文也指出当前研究未尝试其他类型特征，词嵌入等留作未来工作 [2]。
- 草稿里 validator 提取到的数字: `['15000']`

================================================================================================
## 3. 1901.11117｜What is in the model search space?
- gold: `Our search space consists of two stackable cells, one for the model encoder and one for the decoder  Each cell`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=2 t=5s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“模型搜索空间”对应论文中为进化搜索定义的编码搜索空间。该搜索空间由两个可堆叠的 cell 组成，一个用于编码器、一个用于解码器，每个 cell 包含 NASNet 风格的 block，每个 block 接收两个隐藏状态输入并产生新的隐藏状态输出[1]。编码器含 6 个 block、解码器含 8 个 block，从而能精确表示 Transformer[1]。搜索空间包含五个 branch 级搜索字段（input、normalization、layer、output dimension、activation）、

- **闸门前草稿**（645 字）: 你问的“模型搜索空间”对应论文中为进化搜索定义的编码搜索空间。该搜索空间由两个可堆叠的 cell 组成，一个用于编码器、一个用于解码器，每个 cell 包含 NASNet 风格的 block，每个 block 接收两个隐藏状态输入并产生新的隐藏状态输出[1]。编码器含 6 个 block、解码器含 8 个 block，从而能精确表示 Transformer[1]。搜索空间包含五个 branch 级搜索字段（input、normalization、layer、output dimension、activation）、一个 block 级搜索字段（combiner function）和一个 cell 级搜索字段（number of cells）[1]。子模型的基因编码为 [left input, left normalization, left layer, left relative output dimension, left activation, right input, right normalization, right layer, right relative output dimension, right activation, combiner function] × 14 + [number of cells] × 2，前 6 个 block 分配给编码器、后 8
- 草稿里 validator 提取到的数字: `['72999999999999996761335927885055692026574247739994753929654390585025993173485953518437935065046617233336704457244672']`

================================================================================================
## 4. 2001.07263｜How much bigger is Switchboard-2000 than Switchboard-300 database?
- gold: `Switchboard-2000 contains 1700 more hours of speech data.`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=2 t=4s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的是 Switchboard-2000 相对 Switchboard-300 的规模差距：论文中 Switchboard-300 是标准的 300 小时英语会话语音识别任务 [6]，而 Switchboard-2000 是在 2000 小时的 Switchboard+Fisher 数据上训练的 [2]。因此两者相差 2000 − 300 = 1700 小时，即 Switchboard-2000 约为 Switchboard-300 的 6.67 倍（2000 ÷ 300）。论文还指出，增加更多训练数据能大幅提升

- **闸门前草稿**（282 字）: 你问的是 Switchboard-2000 相对 Switchboard-300 的规模差距：论文中 Switchboard-300 是标准的 300 小时英语会话语音识别任务 [6]，而 Switchboard-2000 是在 2000 小时的 Switchboard+Fisher 数据上训练的 [2]。因此两者相差 2000 − 300 = 1700 小时，即 Switchboard-2000 约为 Switchboard-300 的 6.67 倍（2000 ÷ 300）。论文还指出，增加更多训练数据能大幅提升系统，某些情况下相对提升约 30% [2]。
- 草稿里 validator 提取到的数字: `['6.67', '30', '300', '1700', '2000']`

================================================================================================
## 5. 2001.07209｜Which datasets are used in the paper?
- gold: `Google N-grams
COHA
Moral Foundations Dictionary (MFD)`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=6 t=4s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 论文使用的数据集/语料包括以下几项：

1. **Pew Research Center 2013 Global Attitudes survey**：用于将模型预测与人类道德判断对比，来自 40 个国家的参与者对 8 个话题（如堕胎、同性恋）进行评分 [1]。
2. **Google N-grams**：$8.5 \times 10^{11}$ tokens 的英语文献语料（Google Books，全类型），覆盖 1800–1999 年 [4]。
3. **COHA**：较小的语料，$4.1 \times 10

- **闸门前草稿**（602 字）: 论文使用的数据集/语料包括以下几项：  1. **Pew Research Center 2013 Global Attitudes survey**：用于将模型预测与人类道德判断对比，来自 40 个国家的参与者对 8 个话题（如堕胎、同性恋）进行评分 [1]。 2. **Google N-grams**：$8.5 \times 10^{11}$ tokens 的英语文献语料（Google Books，全类型），覆盖 1800–1999 年 [4]。 3. **COHA**：较小的语料，$4.1 \times 10^8$ tokens，体裁均衡、代表 1810–2009 年的美式英语 [4]。 4. **Moral Foundations Dictionary (MFD)**：提供各 MFT 类别的种子词，用于道德极性和细粒度分类 [7]。 5. **约 14,000 个英语单词的情绪效价评分**：用于选取道德无关种子词及效价相关性分析 [3][7]。 6. **约 14,000 个英语名词的效价评分**：用于与正向道德极性预测做相关分析 [3]。 7. **具体性（concreteness）评分规范**：用于大规模道德变化分析 [11]。  此外，检索道德变化概念时选取了 1800–1999 年间总频率最高的 10,000 个名词，并限定为 WordNet 中标注为名词的词 [9
- 草稿里 validator 提取到的数字: `['1800', '1810', '2013', '10000', '14000', '4.1e+08', '850000000000']`

================================================================================================
## 6. 2001.07209｜Which dataset sources to they use to demonstrate moral sentiment through history?
- gold: `（空 / 无答案）`
- 原判: score=1 status=fail unanswerable=True
- 复跑: action=**pass** level=L3 n_entries=12 cites=3 t=8s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“用于展示历史道德情感的数据集来源”对应论文中训练历时词嵌入的两个历史语料库：Google N-grams 和 COHA [7]。Google N-grams 是取自英语文学（Google Books，全类型）的 8.5 × 10^11 词符语料，时间跨度为 1800–1999 年 [7]；COHA 是规模较小的 4.1 × 10^8 词符语料，选自体裁均衡、代表 1810–2009 年美国英语的作品 [7]。论文将历史时间划分为十年一档，并分别在这两个语料上训练词嵌入 [7]。在后续应用中，作者主要使用在 

- **闸门前草稿**（303 字）: 你问的“用于展示历史道德情感的数据集来源”对应论文中训练历时词嵌入的两个历史语料库：Google N-grams 和 COHA [7]。Google N-grams 是取自英语文学（Google Books，全类型）的 8.5 × 10^11 词符语料，时间跨度为 1800–1999 年 [7]；COHA 是规模较小的 4.1 × 10^8 词符语料，选自体裁均衡、代表 1810–2009 年美国英语的作品 [7]。论文将历史时间划分为十年一档，并分别在这两个语料上训练词嵌入 [7]。在后续应用中，作者主要使用在 Google N-grams 上训练的历时嵌入，因为它在评测中表现更优 [2][9]。
- 草稿里 validator 提取到的数字: `['1800', '1810', '4.1e+08', '850000000000']`

================================================================================================
## 7. 2004.03925｜Do the authors give examples of positive and negative sentiment with regard to the virus?
- gold: `no`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=9 cites=3 t=4s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 否。论文只给出了正面、负面、中性推文的比例（如普通公众推文中约 54% 为中性、29% 为正面、15% 为负面 [1]；WHO 推文中约 60% 为正面、24% 为中性、15% 为负面 [1][2]），并说明极性大于 0 为正面、小于 0 为负面、等于 0 为中性 [3]，但并未给出任何具体的正面或负面推文示例文本。

- **闸门前草稿**（159 字）: 否。论文只给出了正面、负面、中性推文的比例（如普通公众推文中约 54% 为中性、29% 为正面、15% 为负面 [1]；WHO 推文中约 60% 为正面、24% 为中性、15% 为负面 [1][2]），并说明极性大于 0 为正面、小于 0 为负面、等于 0 为中性 [3]，但并未给出任何具体的正面或负面推文示例文本。
- 草稿里 validator 提取到的数字: `['15', '24', '29', '54', '60']`

================================================================================================
## 8. 1906.03338｜How do they demonstrate the robustness of their results?
- gold: `performances of a purely content-based model naturally stays stable`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=11 cites=1 t=16s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“他们如何证明结果的稳健性”，对应论文中通过改变测试实例的上下文来检验模型表现的两项实验。具体做法是：在 randomized-context 设置中，他们以随机方式交换测试实例的上下文；在 no-context 设置中，他们直接删除待分类 ADU 周围的上下文 [5]。结果显示，在标准设置下，能访问上下文（含仅访问上下文）的模型总是优于基于内容的模型 [5]；但一旦随机翻转或完全抑制上下文，情况反转，基于内容的模型总是优于其他模型，某些类别（support）的差距可超过 50 个 F1 百分点 [5]。此外

- **闸门前草稿**（340 字）: 你问的“他们如何证明结果的稳健性”，对应论文中通过改变测试实例的上下文来检验模型表现的两项实验。具体做法是：在 randomized-context 设置中，他们以随机方式交换测试实例的上下文；在 no-context 设置中，他们直接删除待分类 ADU 周围的上下文 [5]。结果显示，在标准设置下，能访问上下文（含仅访问上下文）的模型总是优于基于内容的模型 [5]；但一旦随机翻转或完全抑制上下文，情况反转，基于内容的模型总是优于其他模型，某些类别（support）的差距可超过 50 个 F1 百分点 [5]。此外，ANOVA 特征 F 分数也显示，来自 EAU 周围上下文的特征得分远高于来自 EAU span 的特征，进一步说明模型在可选时会强烈依赖上下文线索 [5]。
- 草稿里 validator 提取到的数字: `[]`

================================================================================================
## 9. 2003.04967｜What experimental evaluation is used?
- gold: `root mean square error between the actual and the predicted price of Bitcoin for every minute`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=4 t=10s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“实验评估”对应论文中 KryptoOracle 的实验设置与结果。实验用 PySpark v2.3、Jupyter notebooks 和 Python 2.7 编写，在 SOSCIP 云服务器（32 GB RAM、8 CPU、120 GB HDD、Ubuntu 18.04）上运行 30 天，数据针对 Bitcoin [4]。数据来自 Twitter API（#BTC、#Bitcoin，约 25000 条/天、约 10 MB/天）和 Cryptocompare API，按一分钟时间窗对齐后存入训练 RDD 

- **闸门前草稿**（380 字）: 你问的“实验评估”对应论文中 KryptoOracle 的实验设置与结果。实验用 PySpark v2.3、Jupyter notebooks 和 Python 2.7 编写，在 SOSCIP 云服务器（32 GB RAM、8 CPU、120 GB HDD、Ubuntu 18.04）上运行 30 天，数据针对 Bitcoin [4]。数据来自 Twitter API（#BTC、#Bitcoin，约 25000 条/天、约 10 MB/天）和 Cryptocompare API，按一分钟时间窗对齐后存入训练 RDD [7]。评估指标为实际价格与预测价格之间的 RMS 误差，运行一天后整体 RMS 误差为 10 美元 [12]；论文还展示了实验末尾 5 小时的 RMS 误差（以美元计）及实际与预测价格对比图，显示能提前 1 分钟预测比特币价格 [11]。
- 草稿里 validator 提取到的数字: `['2.3', '2.7', '18.04', '120', '25000']`

================================================================================================
## 10. 1809.01060｜What were the results of the first experiment?
- gold: `Best performance achieved is 0.72 F1 score`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=9 cites=4 t=7s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“第一个实验”对应论文中的 AMT 人类标注实验（对 200 个带上下文句子对进行隐喻转述贴切度评分）[1][2]。其核心结果是：上下文对贴切度评分产生“压缩效应”，即把评分向量表中间拉平——低分被抬高、高分被压低[4][5]。具体地，71.1% 的原本高分（4）配对在上下文中降为 3，而反向提升很少，仅 5%（2 对）从 3 升到 4[5]；低端则有 68.2% 的 1 分对在上下文中升到 2，仅 3.9% 的 2 分对降到 1[5]。中间类别基本稳定：9.8% 的 2 分对变为 3，10% 的 3 分对变

- **闸门前草稿**（343 字）: 你问的“第一个实验”对应论文中的 AMT 人类标注实验（对 200 个带上下文句子对进行隐喻转述贴切度评分）[1][2]。其核心结果是：上下文对贴切度评分产生“压缩效应”，即把评分向量表中间拉平——低分被抬高、高分被压低[4][5]。具体地，71.1% 的原本高分（4）配对在上下文中降为 3，而反向提升很少，仅 5%（2 对）从 3 升到 4[5]；低端则有 68.2% 的 1 分对在上下文中升到 2，仅 3.9% 的 2 分对降到 1[5]。中间类别基本稳定：9.8% 的 2 分对变为 3，10% 的 3 分对变为 2[5]。此外，上下文内与上下文外的人类平均评分之间 Pearson 相关为 0.81[5]，且转述与非转述的分界线（2 与 3 之间）未受上下文强烈影响[5]。
- 草稿里 validator 提取到的数字: `['0.81', '3.9', '5', '9.8', '10', '68.2', '71.1', '200']`


## 汇总

- 判定类型统计: `{}`
- 复跑 action 分布: `{'pass': 10}`
- 本轮成本: calls=31 prompt=0 completion=0 ≈ 0.00 元（高峰档）
