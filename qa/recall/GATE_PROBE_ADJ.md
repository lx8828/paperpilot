# 闸门兜底题诊断（2026-09-11 02:26）

> 来源 `qa/qasper_run_20260911_005347.json`｜闸门兜底 10 道｜逐题复跑并打印判定

================================================================================================
## 1. 1912.01673｜How many sentence transformations on average are available per unique sentence in dataset?
- gold: `27.41 transformation on average of single seed sentence is available in dataset.`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=10 cites=1 t=16s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“每个唯一句子平均有多少个句子变换”对应论文中 COSTRA 1.0 数据集的统计。证据中并未直接给出“平均变换数”这一数值，但可据已有数字推算：数据集共有 4262 个唯一句子（含 150 个种子句）[1]，而第二轮收集了 293 条标注，每条标注即为一个种子句提供全部 15 种变换[1]。因此平均每个种子句对应约 293÷150 ≈ 1.95 条标注，即约 1.95×15 ≈ 29.3 个变换；但论文本身未直接报告该平均值，上述为基于 [1] 的推算。

- **闸门前草稿**（233 字）: 你问的“每个唯一句子平均有多少个句子变换”对应论文中 COSTRA 1.0 数据集的统计。证据中并未直接给出“平均变换数”这一数值，但可据已有数字推算：数据集共有 4262 个唯一句子（含 150 个种子句）[1]，而第二轮收集了 293 条标注，每条标注即为一个种子句提供全部 15 种变换[1]。因此平均每个种子句对应约 293÷150 ≈ 1.95 条标注，即约 1.95×15 ≈ 29.3 个变换；但论文本身未直接报告该平均值，上述为基于 [1] 的推算。
- 草稿里 validator 提取到的数字: `['1', '1.95', '29.3', '150', '293', '4262']`

================================================================================================
## 2. 1909.00338｜Which features do they use to model Twitter messages?
- gold: `word unigrams, bigrams, and trigrams`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=1 t=4s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“features”对应论文中用于建模推文的特征。论文提取的是词级 n-gram 特征：word unigrams、bigrams 和 trigrams [4]。这些特征被编码为二元值（出现设为 1，否则为 0）[4]，训练时只保留最高频的 15,000 个特征，其余全部移除 [4]。此外，分词使用 Ucto，token 被小写化，标点、emoji 和 emoticon 被保留 [4]。

- **闸门前草稿**（198 字）: 你问的“features”对应论文中用于建模推文的特征。论文提取的是词级 n-gram 特征：word unigrams、bigrams 和 trigrams [4]。这些特征被编码为二元值（出现设为 1，否则为 0）[4]，训练时只保留最高频的 15,000 个特征，其余全部移除 [4]。此外，分词使用 Ucto，token 被小写化，标点、emoji 和 emoticon 被保留 [4]。
- 草稿里 validator 提取到的数字: `['15000']`

================================================================================================
## 3. 1901.11117｜What is in the model search space?
- gold: `Our search space consists of two stackable cells, one for the model encoder and one for the decoder  Each cell`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=2 t=5s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“model search space”对应论文中为进化搜索定义的编码搜索空间。该搜索空间由两个可堆叠的 cell 组成，一个用于编码器、一个用于解码器；每个 cell 包含 NASNet 风格的 block，每个 block 接收两个隐藏状态输入并产生新的隐藏状态输出[1]。编码器含 6 个 block、解码器含 8 个 block，从而能精确表示 Transformer[1]。搜索空间包含五个 branch 级搜索字段（input、normalization、layer、output dimension、

- **闸门前草稿**（650 字）: 你问的“model search space”对应论文中为进化搜索定义的编码搜索空间。该搜索空间由两个可堆叠的 cell 组成，一个用于编码器、一个用于解码器；每个 cell 包含 NASNet 风格的 block，每个 block 接收两个隐藏状态输入并产生新的隐藏状态输出[1]。编码器含 6 个 block、解码器含 8 个 block，从而能精确表示 Transformer[1]。搜索空间包含五个 branch 级搜索字段（input、normalization、layer、output dimension、activation）、一个 block 级字段（combiner function）和一个 cell 级字段（number of cells）[1]。其基因编码为 [left input, left normalization, left layer, left relative output dimension, left activation, right input, right normalization, right layer, right relative output dimension, right activation, combiner function] × 14 + [number of cells] × 2，前 6 个 block 分配给编码
- 草稿里 validator 提取到的数字: `['72999999999999996761335927885055692026574247739994753929654390585025993173485953518437935065046617233336704457244672']`

================================================================================================
## 4. 2001.07263｜How much bigger is Switchboard-2000 than Switchboard-300 database?
- gold: `Switchboard-2000 contains 1700 more hours of speech data.`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=2 t=4s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的 Switchboard-2000 与 Switchboard-300 的规模差异，对应论文中的 2000 小时与 300 小时两个训练集。Switchboard-300 是标准的 300 小时英语会话语音识别任务 [6]，而 Switchboard-2000 是在 2000 小时的 Switchboard+Fisher 数据上训练的 [2]。因此前者约为后者的 2000 ÷ 300 ≈ 6.67 倍，即大约大 6.7 倍（多出约 1700 小时）。论文也指出增加更多训练数据能大幅提升系统，某些情况下相对提升

- **闸门前草稿**（270 字）: 你问的 Switchboard-2000 与 Switchboard-300 的规模差异，对应论文中的 2000 小时与 300 小时两个训练集。Switchboard-300 是标准的 300 小时英语会话语音识别任务 [6]，而 Switchboard-2000 是在 2000 小时的 Switchboard+Fisher 数据上训练的 [2]。因此前者约为后者的 2000 ÷ 300 ≈ 6.67 倍，即大约大 6.7 倍（多出约 1700 小时）。论文也指出增加更多训练数据能大幅提升系统，某些情况下相对提升约 30% [2]。
- 草稿里 validator 提取到的数字: `['6.67', '6.7', '30', '300', '1700', '2000']`

================================================================================================
## 5. 2001.07209｜Which datasets are used in the paper?
- gold: `Google N-grams
COHA
Moral Foundations Dictionary (MFD)`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=4 t=4s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“论文使用的数据集”对应论文中实际用于训练/分析的历史语料与外部标注数据。论文使用的两个历史英语语料是 Google N-grams（约 8.5×10¹¹ tokens，1800–1999）和 COHA（约 4.1×10⁸ tokens，1810–2009）[4]。此外，论文还使用了 Pew Research Center 2013 Global Attitudes survey 的人类判断数据（40 个国家、8 个话题）[1]，以及约 14,000 个英语单词的 valence ratings 用于道德不相

- **闸门前草稿**（276 字）: 你问的“论文使用的数据集”对应论文中实际用于训练/分析的历史语料与外部标注数据。论文使用的两个历史英语语料是 Google N-grams（约 8.5×10¹¹ tokens，1800–1999）和 COHA（约 4.1×10⁸ tokens，1810–2009）[4]。此外，论文还使用了 Pew Research Center 2013 Global Attitudes survey 的人类判断数据（40 个国家、8 个话题）[1]，以及约 14,000 个英语单词的 valence ratings 用于道德不相关种子词和效价验证[3][7]。
- 草稿里 validator 提取到的数字: `['1800', '1810', '1999', '2009', '2013', '14000', '4.1e+08', '850000000000']`

================================================================================================
## 6. 2001.07209｜Which dataset sources to they use to demonstrate moral sentiment through history?
- gold: `（空 / 无答案）`
- 原判: score=1 status=fail unanswerable=True
- 复跑: action=**pass** level=L3 n_entries=12 cites=3 t=7s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“用于展示历史道德情感的数据集来源”，对应论文中训练历时词嵌入的两个历史语料库：Google N-grams 和 COHA [7]。Google N-grams 是取自英语文学（Google Books，全类型）的约 8.5×10¹¹ 词符语料，时间跨度为 1800–1999 年 [7]；COHA 是较小的约 4.1×10⁸ 词符语料，选自体裁均衡、代表 1810–2009 年美国英语的作品 [7]。论文将历史时间划分为十年一档，并分别在这两个语料上训练词嵌入 [7]。在后续应用中，作者使用的是在 Googl

- **闸门前草稿**（299 字）: 你问的“用于展示历史道德情感的数据集来源”，对应论文中训练历时词嵌入的两个历史语料库：Google N-grams 和 COHA [7]。Google N-grams 是取自英语文学（Google Books，全类型）的约 8.5×10¹¹ 词符语料，时间跨度为 1800–1999 年 [7]；COHA 是较小的约 4.1×10⁸ 词符语料，选自体裁均衡、代表 1810–2009 年美国英语的作品 [7]。论文将历史时间划分为十年一档，并分别在这两个语料上训练词嵌入 [7]。在后续应用中，作者使用的是在 Google N-grams 上训练的历时词嵌入，因为它在评估中表现更优 [2][9]。
- 草稿里 validator 提取到的数字: `['1800', '1810', '4.1e+08', '850000000000']`

================================================================================================
## 7. 2004.03925｜Do the authors give examples of positive and negative sentiment with regard to the virus?
- gold: `no`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=9 cites=3 t=4s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 否。论文只报告了情感极性的判定规则（polarity > 0 为正面、< 0 为负面、= 0 为中性）[3]，以及各数据集正面/中性/负面推文的百分比（如公众推文约 54% 中性、29% 正面、15% 负面；WHO 推文约 60% 正面、24% 中性、15% 负面）[1][2]，但并未给出任何具体的正面或负面情感推文示例文本。

- **闸门前草稿**（164 字）: 否。论文只报告了情感极性的判定规则（polarity > 0 为正面、< 0 为负面、= 0 为中性）[3]，以及各数据集正面/中性/负面推文的百分比（如公众推文约 54% 中性、29% 正面、15% 负面；WHO 推文约 60% 正面、24% 中性、15% 负面）[1][2]，但并未给出任何具体的正面或负面情感推文示例文本。
- 草稿里 validator 提取到的数字: `['15', '24', '29', '54', '60']`

================================================================================================
## 8. 1906.03338｜How do they demonstrate the robustness of their results?
- gold: `performances of a purely content-based model naturally stays stable`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=11 cites=1 t=15s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“他们如何证明结果的稳健性”，对应论文中通过**改变测试实例的上下文**来模拟跨文档场景的两组实验。具体做法是：在 randomized-context 设置中，系统性地随机交换测试实例的上下文；在 no-context 设置中，直接删除待分类 ADU 周围的上下文 [5]。结果显示，在标准设置下，能访问上下文（或仅访问上下文）的模型总是优于基于内容的模型；但一旦随机翻转或完全抑制上下文，情况反转，基于内容的模型始终优于其他模型，某些类别（support）的差距可超过 50 个 F1 百分点 [5]。此外，A

- **闸门前草稿**（351 字）: 你问的“他们如何证明结果的稳健性”，对应论文中通过**改变测试实例的上下文**来模拟跨文档场景的两组实验。具体做法是：在 randomized-context 设置中，系统性地随机交换测试实例的上下文；在 no-context 设置中，直接删除待分类 ADU 周围的上下文 [5]。结果显示，在标准设置下，能访问上下文（或仅访问上下文）的模型总是优于基于内容的模型；但一旦随机翻转或完全抑制上下文，情况反转，基于内容的模型始终优于其他模型，某些类别（support）的差距可超过 50 个 F1 百分点 [5]。此外，ANOVA 特征 F 分数也显示，来自 EAU 周围上下文的特征得分远高于来自 EAU span 的特征，进一步证明模型在可选时会强烈依赖上下文线索而忽视 EAU 本身的信息 [5]。
- 草稿里 validator 提取到的数字: `[]`

================================================================================================
## 9. 2003.04967｜What experimental evaluation is used?
- gold: `root mean square error between the actual and the predicted price of Bitcoin for every minute`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=12 cites=3 t=9s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“experimental evaluation”对应论文中 KryptoOracle 的实验评估部分。该实验用 PySpark v2.3、Jupyter notebooks 和 Python 2.7 编写代码，并在 SOSCIP 云上 32 GB RAM、8 CPUs、120 GB HDD、Ubuntu 18.04 的服务器实例上运行了 30 天 [4]。数据方面，Twitter 数据用 #Bitcoin 和 #BTC 标签采集 14 天，Bitcoin 实时价格用 Cryptocompare API 同步

- **闸门前草稿**（383 字）: 你问的“experimental evaluation”对应论文中 KryptoOracle 的实验评估部分。该实验用 PySpark v2.3、Jupyter notebooks 和 Python 2.7 编写代码，并在 SOSCIP 云上 32 GB RAM、8 CPUs、120 GB HDD、Ubuntu 18.04 的服务器实例上运行了 30 天 [4]。数据方面，Twitter 数据用 #Bitcoin 和 #BTC 标签采集 14 天，Bitcoin 实时价格用 Cryptocompare API 同步采集 [12]；实验收集的是 Bitcoin 加密货币数据 [4]。评估指标是实际价格与预测价格之间的均方根（RMS）误差，运行一天后整体 RMS 误差为 10 美元 [12]，并在实验末尾 5 小时期间展示了 RMS 误差（以美元计）[11]。
- 草稿里 validator 提取到的数字: `['2.3', '2.7', '18.04', '120']`

================================================================================================
## 10. 1809.01060｜What were the results of the first experiment?
- gold: `Best performance achieved is 0.72 F1 score`
- 原判: score=1 status=fail unanswerable=False
- 复跑: action=**pass** level=L3 n_entries=9 cites=4 t=6s
- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）
- 闸门后答案: 你问的“第一个实验”对应论文中的 AMT 人类标注实验（对 200 个含上下文隐喻-释义对进行适切度评分）[1][2]。其核心结果是：上下文使适切度评分向量表中间“压缩”——原本脱离上下文被评为 4 的隐喻有 71.1% 在上下文中降为 3，而反向提升很少，仅 5%（2 对）从 3 升到 4[5]；低端则有 68.2% 的 1 分对在上下文中升到 2，仅 3.9% 的 2 分对降到 1[5]。中间档基本稳定：9.8% 的 2 分对变为 3，10% 的 3 分对变为 2[5]。此外，上下文内外平均人类评分之间的 Pe

- **闸门前草稿**（318 字）: 你问的“第一个实验”对应论文中的 AMT 人类标注实验（对 200 个含上下文隐喻-释义对进行适切度评分）[1][2]。其核心结果是：上下文使适切度评分向量表中间“压缩”——原本脱离上下文被评为 4 的隐喻有 71.1% 在上下文中降为 3，而反向提升很少，仅 5%（2 对）从 3 升到 4[5]；低端则有 68.2% 的 1 分对在上下文中升到 2，仅 3.9% 的 2 分对降到 1[5]。中间档基本稳定：9.8% 的 2 分对变为 3，10% 的 3 分对变为 2[5]。此外，上下文内外平均人类评分之间的 Pearson 相关系数为 0.81[5]，而释义与非释义的分界线（2 与 3 之间）并未被上下文强烈改变[4][5]。
- 草稿里 validator 提取到的数字: `['0.81', '3.9', '5', '9.8', '10', '68.2', '71.1', '200']`


## 汇总

- 判定类型统计: `{}`
- 复跑 action 分布: `{'pass': 10}`
- 本轮成本: calls=31 prompt=0 completion=0 ≈ 0.00 元（高峰档）
