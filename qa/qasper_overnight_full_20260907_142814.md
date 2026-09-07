# QASPER 基础通过测试（2026-09-07 14:28）

- 总题数: 1000 ｜ 通过(pass≥4): 750 ｜ 失败: 190 ｜ unknown_ok: 60 ｜ 无答案拒答未认可: 0 ｜ error: 0
- 有分题均分: 3.94/5 ｜ 平均耗时: 13.86s

| paper | qid | 问题 | gold | 系统 | route | score | 状态 |
|---|---|---|---|---|---|---|---|
| qasper_1904.05584 | 7fe48939ce34 | Where do they employ feature-wise sigm | gating mechanism acts upon eac | 你问的“feature-wise sigmoid gating”对应论文中的 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1904.05584 | 65ad17f614b7 | Which model architecture do they use t | BiLSTM with max pooling | 你问的“用于获得句子表示”的模型架构，对应论文中的 **BiLSTM wit | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1904.05584 | 323e100a6c92 | Which downstream sentence-level tasks  | BIBREF13 , BIBREF18 | 你问的下游句子级评测任务对应论文中 SentEval 评测基准所包含的 11 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1904.05584 | 9f89bff89cea | Which similarity datasets do they use? | MEN MTurk287 MTurk771 RG RW Si | 你问的相似度数据集对应论文中用于词级评测的多个数据集。证据中列出了以下数据集 | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_1909.04181 | e9cfe3f15735 | Does the paper report F1-scores for th | no | 你问的是论文是否报告了年龄（age）和语言变体（dialect）任务的 F1 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.04181 | 52ed2eb6f4d1 | Are the models compared to some baseli | yes | 是的，论文将所提出的模型与基线模型进行了比较。你问的基线对应论文中的 GRU | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.04181 | 2c576072e494 | What are the in-house data employed? | we manually label an in-house  | 你问的 in-house data 对应论文中用于数据增强的自有标注数据集， | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.04181 | 8602160e98e4 | What are the three datasets used in th | Data released for APDA shared  | 你问的“三个数据集”对应论文中用于训练/扩展的三个数据来源：共享任务官方训练 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2002.11402 | 1771a5523682 | What is the difference in recall score | Between the model and Stanford | 你问的“systems”对应论文中对比的模型，包括 Flair-NER、St | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2002.11402 | 1d74fd1d38a5 | What is their f1 score and recall? | F1 score and Recall are 68.66, | 你问的 F1 分数和召回率对应论文中模型在训练和测试集上的表现。论文指出，两 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2002.11402 | da8bda963f17 | How many layers does their system have | 4 layers | 你问的系统层数对应论文中最终采用的 GRU 架构。根据正文，他们尝试了多种  | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_2002.11402 | 5c059a13d599 | Which news corpus is used? | — | 你问的新闻语料对应论文中的英文新闻文章数据集。论文使用了过去 4 年间发布的 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_2002.11402 | a1885f807753 | How large is the dataset they used? | English wikipedia dataset has  | 你问的数据集规模对应论文中用于创建并行语料库的新闻文章数据。最终用于训练的并 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1709.08299 | 7a7e279170e7 | What two types the Chinese reading com | cloze-style reading comprehens | 你问的“Chinese reading comprehension data | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1709.08299 | e3981a11d3d6 | For which languages most of the existi | English | 你问的“大多数现有 MRC 数据集的语言”对应论文中提到的既有 cloze- | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1701.00185 | 62a6382157d5 | What were the evaluation metrics used? | accuracy normalized mutual inf | 你问的评估指标对应论文中用于衡量短文本聚类性能的两个指标：准确率（ACC）和 | L0→L3→answer_L3 | 5 | pass |
| qasper_1701.00185 | 9e04730907ad | What were their performance results? | On SearchSnippets dataset ACC  | 你问的“their performance results”对应论文中提出的 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1701.00185 | 5a0841cc0628 | By how much did they outperform the ot | on SearchSnippets dataset by 6 | 你问的“they”对应论文中提出的 STC 系列方法（STC-LE、STC- | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1701.00185 | a5dd569e6d64 | Which popular clustering methods did t | K-means, Skip-thought Vectors, | 你问的“popular clustering methods”对应论文中与  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1701.00185 | 785c054f6ea0 | What datasets did they use? | SearchSnippets StackOverflow B | 你问的数据集对应论文中的三个公开短文本数据集，分别是：SearchSnipp | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1707.01890 | b10632eaa0ca | What are the categories being extracte | — | 抱歉，我无法从该论文中找到答案，因为检索到的内容未提及任何被提取的类别（如具 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.08191 | b124137e6217 | What new interesting tasks can be solv | analogy query analogy browsing | 你问的“基于嵌入空间语义结构可解决的新任务”对应论文中提出的**类比查询（a | L0→answer_L0 | 4 | pass |
| qasper_1909.08191 | c6aa8a02597f | What are the uncanny semantic structur | Semantic similarity structure  | 你问的“uncanny semantic structures”对应论文中嵌 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.08191 | bfad30f51ce3 | What is the general framework for data | three main components, namely  | 你问的“通过语义查询进行数据探索的通用框架”对应论文中提出的**通用语义查询 | L0→answer_L0 | 2 | fail |
| qasper_1909.08191 | dd9883f4adf7 | What data exploration is supported by  | Task processing: converting da | 你问的“这些语义结构所支持的数据探索”对应论文中源于知识图谱嵌入空间语义结构 | L0→answer_L0 | 3 | fail |
| qasper_1705.03487 | 91c81807374f | What is barycentric Newton diagram? | The basic idea of the visualiz | 你问的“barycentric Newton diagram”对应论文中用于 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2003.04973 | 3ee976add83e | What were the model's results on flood | Queensland flood which provide | 你问的“模型在洪水检测上的结果”对应论文中 ULMFiT 在洪水相关推文分类 | L0→answer_L0 | 5 | pass |
| qasper_2003.04973 | ef04182b6ae7 | What dataset did they use? | disaster data from BIBREF5 Que | 你问的数据集对应论文中的 **CrisiLexT6 数据集** [1]。该数 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1606.06361 | 75773ee868c0 | Do they evaluate the syntactic parses? | no | 否。论文中未对句法分析结果（syntactic parses）本身进行评测； | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1606.06361 | 11ed8c4d98a4 | What knowledge bases do they use? | NELL | 你问的知识库对应论文中的 **NELL**（Never-Ending Lan | L0→answer_L0 | 5 | pass |
| qasper_1804.08139 | 0fd678d24c86 | What evaluation metrics are used? | Accuracy on each dataset and t | 抱歉，我无法从该论文中找到所用评估指标的相关信息，因为检索内容中未提及任何评 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1804.08139 | b556fd3a9e0c | What dataset did they use? | 16 different datasets from sev | 你问的数据集对应论文中的 16 个情感分类数据集，它们来自多个常用评论语料库 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1804.08139 | 0db1ba66a7e7 | What tasks did they experiment with? | Sentiment Classification Trans | 你问的实验任务对应论文中的情感分类任务（sentiment classifi | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1912.11602 | f95097cf4a0d | What were the baselines? | $\textsc {Lead-X}$ $\textsc {P | 你问的“baselines”对应论文中用于对比的基线摘要模型。论文列出的基线 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1912.11602 | bb8a0035b767 | What metric was used in the evaluation | ROUGE-1, ROUGE-2 and ROUGE-L F | 你问的评估指标对应论文中的标准 ROUGE 指标，具体包括 ROUGE-1、 | L0→L1→answer_L1 | 5 | pass |
| qasper_1912.11602 | 2a0a44f169ad | What did they pretrain the model on? | hree years of online news arti | 你问的预训练数据对应论文中的“新闻文章（news articles）”。具体 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1912.11602 | fd6c19463223 | What does the data cleaning and filter | many news articles begin with  | 你问的数据清洗与过滤流程对应论文中的预训练数据预处理步骤，具体包括以下内容： | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1912.11602 | 487dc65bf8a8 | What unlabeled corpus did they use? | three years of online news art | 你问的“unlabeled corpus”对应论文中用于预训练的无标注新闻语 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1708.04557 | 5d9b088bb066 | what processing was done on the speech | Remove numbers and interjectio | 你问的“speeches 在解析前做了哪些处理”，对应论文中用于文本分析的语 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1803.03786 | 3d34a02ceebc | what classifiers were used in this pap | Support Vector Machines (SVM)  | 你问的分类器对应论文中使用的 **SVM** 和 **神经网络（neural | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_1803.03786 | 96992460cfc5 | what are their evaluation metrics? | F1 accuracy | 你问的评估指标对应论文中用于评测模型性能的指标，包括 F1 分数和准确率。多 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1803.03786 | 363ddc06db57 | what types of features were used? | stylometric, lexical, grammati | 你问的特征类型对应论文中的手工特征与词表示两大类。手工特征包括：停用词使用频 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1803.03786 | f12a282571f8 | what lexical features did they experim | TF.IDF-based features | 你问的“lexical features”对应论文中的手工特征（hand-c | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_1803.03786 | 5b1cd21936ae | what is the size of the dataset? | The number of fake news that h | 你问的数据集规模对应论文中的保加利亚语新闻语料库。训练数据集包含 2,815 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1803.03786 | 964705a100e5 | what datasets were used? | training dataset contains 2,81 | 你问的数据集对应论文中的保加利亚语新闻语料库（Bulgarian news  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1803.03786 | f08a66665f01 | what are the three reasons everybody h | — | 抱歉，我无法从该论文中找到答案。检索到的内容未提及“they”的具体指代对象 | L0→L1→L3→answer_unknown | 5 | pass |
| qasper_1909.04625 | 946676f1a836 | What is the performance achieved by th | — | 抱歉，我无法从该论文中找到关于模型性能的具体数据（如准确率、困惑度等），因为 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.04625 | 3b090b416c4a | What is the best performance achieved  | — | 抱歉，我无法从这篇论文中找到关于监督模型最佳性能的具体信息，因为文中未提供任 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.04625 | a1e07c7563ad | What is the size of the datasets emplo | (about 4 million sentences, 13 | 你问的数据集规模，对应论文中训练各语言模型所用的语料。其中，LSTM (fr | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.04625 | a1c4f9e8661d | What are the baseline models? | Recurrent Neural Network (RNN) | 你问的基线模型对应论文中用于句法监督的两种神经语言模型：ActionLSTM | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2003.10816 | f63519bb5e11 | What low-resource languages were used  | — | 你问的低资源语言对应论文中的波斯语（Farsi）和法语。证据显示，实验在英语 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_2003.10816 | ffa4d4bfb226 | What classification task was used to e | Paraphrase Identification | 你问的跨语言适应方法对应的评测任务是论文中的 **PI（Paraphrase | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.07816 | c034f38a570d | How better is proposed method than bas | Perplexity of proposed MEED mo | 你问的“proposed method”对应论文中的 MEED 模型。在困惑 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1908.07816 | 9cbea686732b | How does the multi-turn dialog system  | we extract the emotion informa | 你问的“多轮对话系统如何学习”对应论文中提出的端到端数据驱动模型 MEED。 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1908.07816 | 6aee16c4f319 | How is human evaluation performed? | (1) grammatical correctness (2 | 你问的人工评估流程对应论文中的具体实施步骤。评估由4名与作者实验室无关的英语 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.07816 | 4d4b9ff2da51 | Is some other metrics other then perpl | no | 是的，除困惑度外还评测了其他指标。论文在人工评估中衡量了回复的情感适切性、语 | L0→answer_L0 | 2 | fail |
| qasper_1908.07816 | 180047e1ccfc | What two baseline models are used? | sequence-to-sequence model (de | 你问的两个基线模型对应论文中的 S2S（vanilla sequence-t | L0→L1→answer_L1 | 5 | pass |
| qasper_2003.08132 | 20eb673b01d2 | What representations are presented by  | the number of speakers of each | 你问的“representations”对应论文中关于**性别表征（gend | L0→L3→answer_L3 | 5 | pass |
| qasper_2003.08132 | e8e6719d531e | What corpus characteristics correlate  | — | 你问的“与更均衡性别平衡相关的语料库特征”对应论文中关于语料库类型（elic | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2003.08132 | f6e5febf2ea5 | What natural languages are represented | — | 你问的语言对应论文中 OpenSLR 平台上被调查的语音语料库所涵盖的自然语 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.11491 | be7b375b22d9 | What evaluation metrics do they use? | ROUGE F1 METEOR | 你问的评估指标对应论文中的主要评估指标。论文采用完整长度ROUGE F1和M | L0→L1→answer_L1 | 5 | pass |
| qasper_1910.11491 | c4b5cc2988a2 | How do they define local variance? | The reciprocal of the variance | 你问的 local variance 对应论文中的 **local vari | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.02731 | dc5ff2adbe1a | How do they evaluate the sentence repr | The unsupervised tasks include | 你问的“如何评估句子表示”对应论文中的无监督评测（Unsupervised  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1809.02731 | 04b43deab0fd | What are the two decoding functions? | a linear projection and a bije | 你问的两种解码函数对应论文中考虑的两种类型 [1]：

1. **线性投影（ | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1910.08772 | ee27e5b56e43 | Do they beat current state-of-the-art  | no | 你问的“they”对应论文中的 MonaLog 系统。否，MonaLog 并 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.08772 | 4688534a07a3 | How do they combine MonaLog with BERT? | They use Monalog for data-augm | 你问的“MonaLog 与 BERT 的结合方式”对应论文中的混合系统（hy | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1910.08772 | 45893f31ef07 | How do they select monotonicity facts? | They derive it from Wordnet | 抱歉，我无法从该论文中找到关于“如何选择单调性事实”的具体说明；现有内容仅涉 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1708.01776 | 2ec97cf890b5 | How do they measure correlation betwee | They look at the performance a | 你问的“预测与解释质量之间的相关性如何度量”对应论文中通过**分别分析查询（ | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1708.01776 | 41174d8b176c | Does the Agent ask for a value of a va | yes | 你问的 Agent 是否用自然语言生成的文本询问变量值，对应论文中的交互流程 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2003.05377 | 52b113e66fd6 | what is the source of the song lyrics? | Vagalume website | 你问的歌词来源对应论文中的数据采集部分。歌词是从 Vagalume 网站获取 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.05377 | 163a21c0701d | what genre was the most difficult to c | bossa-nova and jovem-guarda ge | 你问的“最难分类的流派”对应论文实验中的多个困难类别。根据证据，bossa- | L0→L1→answer_L1 | 4 | pass |
| qasper_2003.05377 | 36b5f0f62ee9 | what word embedding techniques did the | Word2Vec, Wang2Vec, and FastTe | 你问的词嵌入技术对应论文中实验使用的预训练词向量模型。论文明确提到使用了 W | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.05377 | 6b91fe29175b | what genres do they songs fall under? | Gospel, Sertanejo, MPB, Forró, | 你问的歌曲体裁对应论文中构建的巴西歌曲数据集所覆盖的 14 个音乐体裁。论文 | L0→L1→L2→L3→answer_L3 | 3 | fail |
| qasper_1909.00088 | dccc3b182861 | Does the model proposed beat the basel | no | 你问的“模型”对应论文中提出的 SMERTI 流水线（含 SMERTI-Tr | L0→L1→L2→L3→answer_L3 | 3 | fail |
| qasper_1909.00088 | 98ba7a7aae38 | Has STES been previously used in the l | no | 否。你问的 STES（Semantic Text Exchange Scor | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.00088 | 3da9a861dfa2 | What are the baseline models mentioned | Noun WordNet Semantic Text Exc | 你问的基线模型对应论文中实现的三个基线模型，分别是：NWN-STEM（名词W | L0→L1→answer_L1 | 5 | pass |
| qasper_1608.03902 | 8fcbae7c3bd8 | what was their baseline comparison? | Support Vector Machine (SVM) L | 你问的“baseline comparison”对应论文中与 CNN 模型对 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1707.00995 | f0317e48dafe | What misbehavior is identified? | if the attention loose track o | 你问的“misbehavior”对应论文中识别的注意力机制缺陷。论文指出以下 | L0→L1→answer_L1 | 4 | pass |
| qasper_1707.00995 | ec91b87c3f45 | What is the baseline used? | — | 你问的基线对应论文中的 **CalixtoLC17b 多模态基线**。该基线 | L0→answer_L0 | 1 | fail |
| qasper_1707.00995 | f129c97a81d8 | Which attention mechanisms do they com | Soft attention Hard Stochastic | 你问的“they”对应论文中作者提出的图像注意力机制。论文比较了三种图像注意 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1707.06945 | 83f24e4bbf9d | What baseline is used for the verb cla | — | 你问的“verb classification experiments 的  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1707.06945 | 6b8a3100895f | What clustering algorithm is used on t | MNCut spectral clustering algo | 你问的聚类算法对应论文中的 **MNCut spectral cluster | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1707.06945 | daf624f7d162 | How many words are translated between  | — | 抱歉，我无法从该论文中找到答案，因为文中未提供跨语言翻译对的具体词数或相关统 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1707.06945 | 74261f410882 | What are the six target languages? | Answer with content missing: ( | 你问的六个目标语言对应论文中用于跨语言动词分类实验的六种语言，分别是：意大利 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.03270 | 1835f6569469 | Does the paper report the performance  | no | 该论文未报告神经机器翻译（NMT）模型的性能，因为检索到的内容仅涉及hash | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1911.03270 | a61732774faf | What are the predefined morpho-syntact | — | 抱歉，我无法从该论文中找到您所问的“用于过滤训练数据的预定义形态句法模式”的 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1911.03270 | 994ac7aa662d | Is the RNN model evaluated against any | yes | 是的，你问的 RNN 模型对应论文中提出的 char-RNN 模型，其评估结 | L0→answer_L0 | 5 | pass |
| qasper_1911.03270 | 9282cf80265a | Which languages are used in the paper? | English Russian | 你问的语言对应论文中的英语和俄语。论文在两个数据集上测试了话题标签分割方法， | L0→answer_L0 | 5 | pass |
| qasper_1908.01060 | e6c163f80a11 | Do they test their approach on large-r | yes | 否。你问的“是否在大资源任务上测试”对应论文中的实验设置：该实验以**低资源 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.01060 | 6adfa9eee76b | By how much do they, on average, outpe | 1.6% lower phone error rate on | 你问的“平均提升幅度”对应论文中的平均音素错误率降低幅度。论文指出，所提方法 | L0→answer_L0 | 5 | pass |
| qasper_1908.01060 | 450a359d117b | How do they compute corpus-level embed | First, the embedding matrix IN | 你问的“如何计算语料库级嵌入”对应论文中第 7 节“Corpus Embed | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.00170 | a999761aa976 | What is their model? | cross-lingual NE recognition | 你问的“their model”对应论文中提出的**基于嵌入分布的超球面模型 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.00170 | f229069bcb05 | Do they evaluate on NER data sets? | yes | 是的，论文在命名实体识别（NER）数据集上进行了评估。具体而言，实验使用了  | L0→answer_L0 | 4 | pass |
| qasper_1911.11933 | 046ff04d1018 | Which dataset do they use? | small_parallel_enja Asian Scie | 你问的数据集对应论文实验中使用的两个语料库：**small_parallel | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.11933 | 5a06f11aa75a | Do they trim the search space of possi | no | 抱歉，我无法从该论文中找到直接支撑这一问题的内容。检索到的段落均未提及对可能 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1911.11933 | ffbd6f583692 | Which model architecture do they use t | model is composed of an encode | 你问的模型架构对应论文中的“Wait-k”同传模型，其编码器和解码器均采用* | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.11933 | 74fe054f5243 | Do they compare simultaneous translati | no | 你问的“是否将同传翻译性能与常规机器翻译对比”对应论文中的“full sen | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1911.11933 | cc2b98b46497 | Which metrics do they use to evaluate  | BLEU BIBREF8 RIBES BIBREF9 tok | 你问的评估指标对应论文中用于衡量翻译准确度和延迟的指标。论文使用 **BLE | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.13695 | 3241f90a0385 | What standard large speaker verificati | non-native speech from the BUL | 你问的“用于评测的标准大型说话人验证语料库”对应论文中的 **VoxCele | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1909.13695 | 52e8f7981473 | What systems are tested? | BULATS i-vector/PLDA
BULATS x- | 你问的“测试的系统”对应论文中在 BULATS 和 Linguaskill  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.00542 | b6a4ab009e6f | How did the author's work rank among o | — | 抱歉，我无法从该论文中找到关于作者工作在挑战中与其他提交系统相比的具体排名或 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.00542 | cfffc94518d6 | What approaches without reinforcement  | classification, regression, ne | 你问的“未使用强化学习的方法”对应论文中的分类（classification | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.00542 | f60629c01f99 | What classification approaches were ex | NNC SU4 F1 NNC top 5 Support V | 你问的分类方法对应论文中用于摘要生成的分类框架，具体为：训练一个分类器来预测 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.00542 | a7cb4f8e29fd | Did classification models perform bett | yes | 是的，分类模型的表现优于回归模型。论文明确指出，分类方法产生的结果优于回归方 | L0→answer_L0 | 5 | pass |
| qasper_1611.03382 | 9555aa8de322 | By how much does their model outperfor | w.r.t Rouge-1 their model outp | 你问的“their model”对应论文中提出的 **Read-Again  | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1611.03382 | 81e8d42dad08 | What is the state-of-the art? | neural attention model with a  | 你问的“state-of-the-art”对应论文中提出的 Read-Aga | L0→answer_L0 | 2 | fail |
| qasper_1905.00563 | bc9c31b3ce81 | What datasets are used to evaluate thi | Kinship and Nations knowledge  | 你问的评估数据集对应论文中的 WN18 和 YAGO3-10 两个知识图谱数 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1905.00563 | 185841e97937 | How is this approach used to detect in | if there is an error in the gr | 你问的“用该方法检测错误事实”对应论文中的“Finding Errors i | L0→answer_L0 | 2 | fail |
| qasper_1905.00563 | d427e3d41c4c | Can this adversarial approach be used  | yes | 抱歉，我无法从该论文中找到直接回答这一问题的内容。检索到的段落仅涉及对抗攻击 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1906.07234 | 64ab2b92e986 | What is the performance difference in  | — | 你问的“对抗训练与 FHVAE 解缠表示学习在无监督特征学习中的性能差异”， | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1708.05873 | a2103e7fe613 | What are the country-specific drivers  | wealth  democracy  population  | 你问的“国际发展话语的国家特定驱动因素”对应论文正文中分析的影响两个发展话题 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1708.05873 | 13b366443578 | Is the dataset multilingual? | no | 抱歉，我无法从该论文中找到关于数据集是否多语言的明确信息。检索内容未涉及数据 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1708.05873 | e4a19b91b57c | How are the main international develop | They focus on exclusivity and  | 你问的“各国提出的主要国际发展议题如何被识别”对应论文中通过结构化主题模型（ | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1909.00161 | e1f5531ed04d | Do they use pretrained models? | yes | 是的，论文使用了预训练模型。你问的“预训练模型”对应论文中的蕴含模型，即基于 | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.00161 | 4a4b7c0d3e73 | What are their baseline models? | Majority ESA Word2Vec  Binary- | 你问的基线模型对应论文中在零样本文本分类评测里对比的基线方法，共 5 个：M | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1703.09684 | cf93a209c800 | From when are many VQA datasets collec | late 2014 | 你问的“many VQA datasets collected”对应论文中提 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1606.00189 | 66f0dee89f08 | Do they use pretrained word representa | no | 否。论文中的两个神经网络模型（NNGLM 和 NNJM）均未使用预训练词向量 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1606.00189 | 8f882f414d7e | How do they combine the two proposed n | ncorporating NNGLM and NNJM bo | 你问的“两个神经网络模型的结合方式”对应论文中 NNGLM 与 NNJM 的 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1606.00189 | a49832c89a2d | Which dataset do they evaluate grammat | CoNLL 2014 | 你问的评测数据集对应论文中的 **CoNLL 2014 测试集**（CoNL | L0→answer_L0 | 5 | pass |
| qasper_2001.07615 | 8e5e03a5f35f | What model do they use a baseline to e | a BiLSTM without attention (Bi | 你问的“用于估计满意度的基线模型”对应论文中的 **SVM（支持向量机）** | L0→L1→L2→L3→answer_L3 | 3 | fail |
| qasper_2004.04228 | dfd9302615b2 | What models are evaluated with QAGS? | bert-large-wwm bert-base bert- | 你问的“用 QAGS 评估的模型”对应论文中作为评测对象的摘要生成模型。根据 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2004.04228 | e09dcb6fc163 | Do they use crowdsourcing to collect h | yes | 是的，论文使用了众包方式收集人工判断。具体而言，他们通过 Amazon Me | L0→L1→answer_L1 | 5 | pass |
| qasper_1707.03764 | 157b9f6f8fb5 | How do their results compare against o | They achieved best result in t | 你问的“他们的结果与其他参赛者对比”对应论文中 N-GrAM 系统在 PAN | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1707.03764 | 9bcc1df7ad10 | On which task does do model do worst? | Gender prediction task | 你问的“模型表现最差的任务”对应论文中的 **gender（性别）分类任务* | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1707.03764 | 8427988488b5 | On which task does do model do best? | Variety prediction task | 你问的“模型表现最好的任务”对应论文中的 **language variet | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1909.05360 | e3dc8689d8db | What kind of events do they extract? | — | 你问的“提取的事件”对应论文中事件抽取模块所识别的事件。论文提取的是文本中的 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1909.05360 | cfb5ab893ed7 | Is this the first paper to propose a j | yes | 是的，你问的“首次提出联合模型”对应论文中的核心主张 [1][3]。论文明确 | L0→answer_L0 | 5 | pass |
| qasper_1909.05360 | a5abd4dd91e6 | What datasets were used for this work? | TB-Dense  MATRES | 你问的数据集对应论文实验所用的两个时序关系数据集，即 **TB-Dense* | L0→L1→answer_L1 | 5 | pass |
| qasper_1804.00079 | 63a77d2640df | Which language(s) do they work with? | — | 抱歉，我无法从该论文中找到答案，因为检索内容未提及任何具体语言，缺少关于“他 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1804.00079 | 50be9e6203c4 | How do they evaluate their sentence re | standard benchmarks BIBREF36 , | 你问的“如何评估句子表示”对应论文中的迁移学习评估策略。他们将自己学到的表示 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1804.00079 | 36a9230fadf9 | Which model architecture do they for s | Answer with content missing: ( | 你问的句子编码模型架构对应论文中多任务消融实验里采用的 GRU 编码器。该模 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1804.00079 | 496304f63006 | How many tokens can sentences in their | — | 抱歉，我无法从该论文中找到答案，因为检索内容未提及模型可处理的最大句子长度或 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1804.00079 | 00e9f088291f | Which training objectives do they comb | multi-lingual NMT natural lang | 你问的训练目标对应论文中多任务框架所组合的多种目标。论文明确说明，该框架结合 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1804.00079 | e2f269997f5a | Which data sources do they use? | - En-Fr (WMT14)
- En-De (WMT15 | 你问的“they”对应论文中用于评测句子表示的数据集来源。论文使用了以下数据 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1910.08418 | 557d1874f736 | Which language family does Mboshi belo | Bantu | 你问的 Mboshi 对应论文中实验所用的濒危语言 Mboshi。根据论文， | L0→L1→L3→answer_L3 | 5 | pass |
| qasper_1910.08418 | f41c401a4c6e | Does the paper report any alignment-on | yes | 你问的“alignment-only baseline”对应论文中的 Pis | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1910.08418 | 09cd7ae01fe9 | What is the dataset used in the paper? | French-Mboshi 5K corpus | 你问的数据集对应论文中的 **bilingual French-Mboshi | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.08418 | be3e020ba84b | How is the word segmentation task eval | precision, recall, and F-measu | 你问的“分词任务如何评测”对应论文中的分割性能评估方式。论文使用精确率、召回 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1904.04055 | 5ea87432b916 | What conclusions are drawn from these  | best results were obtained usi | 你问的实验结论对应论文中关于词向量在时间表达式识别任务上的评估结果。主要结论 | L0→answer_L0 | 4 | pass |
| qasper_1904.04055 | 3af9156b95a4 | What experiments are presented? | identify the boundaries of tim | 你问的“experiments”对应论文中的 Task A 实验，其目的是识 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1904.04055 | 7e328cc3cffa | What is specific about the specific em | predicting the word given its  | 你问的“specific embeddings”对应论文中新建的 KGR10 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1904.04055 | 80f19be1cbe1 | What embedding algorithm is used to bu | CBOW and Skip-gram methods in  | 你问的嵌入算法对应论文中用于构建 KGR10 词向量的方法。论文明确说明使用 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1904.04055 | b32381583926 | How was the KGR10 corpus created? | most relevant content of the w | 你问的 KGR10 语料的创建过程对应论文中的 plWordNet Corp | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1906.00378 | 591231d75ff4 | Which vision-based approaches does thi | CNN-mean CNN-avgmax | 你问的“该方法的视觉基线对比对象”对应论文中的 **CNN-mean** 和 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1906.00378 | 9e805020132d | What baseline is used for the experime | CNN-mean CNN-avgmax | 你问的基线对应论文中的两个视觉基线方法：CNN-mean 和 CNN-avg | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1906.00378 | 95abda842c4d | Which languages are used in the multi- | German-English, French-English | 你问的多语言图像描述模型涉及的语言，对应论文中的 German、French | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2002.06053 | a45edc04277a | Are datasets publicly available? | yes | 是的，论文指出数据集和源代码通常通过 GitHub 或 Zenodo 等平台 | L0→L1→L2→L3→answer_L3 | 3 | fail |
| qasper_2002.06053 | 8c8a32592184 | Are this models usually semi/supervise | Both supervised and unsupervis | 你问的“these models”对应论文中综述的各类 NLP 文本表示与机 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2002.06053 | 16646ee77975 | Is there any concrete example in the p | yes | 否。论文中未提供任何具体实例表明 NLP 方法在药物发现中产生了“巨大影响” | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1804.09301 | c2553166463b | Which coreference resolution systems a | the BIBREF5 sieve system from  | 你问的“被测试的共指消解系统”对应论文中评估的三个公开现成系统，分别代表三种 | L0→L1→answer_L1 | 5 | pass |
| qasper_2002.10361 | 38c74ab8292a | How large is the corpus? | It contains 106,350 documents | 抱歉，我无法从该论文中找到关于语料库规模的具体信息（如文档数、用户数等），因 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2002.10361 | ff307b10e56f | Which document classifiers do they exp | logistic regression (LR), recu | 你问的文档分类器对应论文实验部分采用的四种分类模型，即逻辑回归（LR）、循环 | L0→L1→answer_L1 | 5 | pass |
| qasper_2002.10361 | 16af38f7c477 | How large is the dataset? | over 104k documents | 你问的数据集规模对应论文中构建的多语言仇恨言论语料库。论文提到，他们请求了  | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1908.10275 | 75043c17a2cd | What trends are found in musical prefe | audiences wanted products more | 你问的音乐偏好趋势对应论文中的听众偏好发现。研究表明，听众偏好更当代、强烈、 | L0→answer_L0 | 5 | pass |
| qasper_1908.10275 | 95bb3ea4ebc3 | Which decades did they look at? | between 1900s and 2010s | 你问的“they”对应论文中标注流派年代的时间维度。论文考察的年代范围是 1 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.10275 | 3ebdc1548025 | How many genres did they collect from? | 77 genres | 你问的“收集了多少种流派”对应论文中从 Wikipedia 选取的流派数量。 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2001.00582 | 827b5bd21559 | What large corpus is used for experime | The De7 database | 你问的实验用大型语料库对应论文中的 **De7 database**。该数据 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.11546 | 6f2118a0c64d | What datasets are used for training/te | Microsoft Research dataset con | 你问的训练/测试所用数据集对应论文中的 Microsoft Research | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.11546 | 8a0a51382d18 | How better is gCAS approach compared t | For entity  F1 in the movie, t | 你问的 gCAS 相比其他方法的优势，对应论文中提出的 gated CAS  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.11546 | b8dea4a98b4d | What is specific to gCAS cell? | It has three sequentially conn | 你问的 gCAS cell 对应论文中提出的新型循环单元（recurrent | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1906.02715 | 2ba0c7576eb5 | How were the feature representations e | attention probes using visuali | 你问的“特征表示如何被评估”对应论文中两类实验：一是用可视化工具定性评估，二 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1906.02715 | c58e60b99a65 | What linguistic features were probed f | dependency relation between tw | 你问的“被探测的语言特征”对应论文中的**依存关系（dependency r | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.02339 | 9c4a4dfa7b0b | How does TP-N2F compare to LSTM-based  | Full Testing Set accuracy: 84. | 抱歉，论文中未提供TP-N2F与LSTM-Seq2Seq在训练或推理速度上的 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1910.02339 | 4c7ac51a66c1 | What is the performance proposed model | Full Testing Set Accuracy: 84. | 你问的“AlgoList benchmark”对应论文中的 **AlgoLi | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1910.02339 | 05671d068679 | What is the performance proposed model | Operation accuracy: 71.89
Exec | 你问的“proposed model”对应论文中的 TP-N2F 模型。在  | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2002.02224 | ac706631f2b3 | Did they experiment on this dataset? | no | 你问的“this dataset”对应论文中提出的捷克最高法院引用数据集（即 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2002.02224 | 8b71ede81701 | How is quality of the citation measure | it is necessary to evaluate th | 抱歉，我无法从该论文中找到关于“引文质量如何度量”的答案。检索到的内容仅涉及 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2002.02224 | fa2a384a23f5 | How big is the dataset? | 903019 references | 你问的数据集规模对应论文中从三个捷克最高法院判决提取的引用总数。该数据集共包 | L0→answer_L0 | 5 | pass |
| qasper_1809.07629 | c5171daf8210 | What evaluation metrics are used? | the evaluation metrics include | 你问的评估指标对应论文中使用的 BLEU 和 ROUGE（1, 2, L）分 | L0→L1→answer_L1 | 5 | pass |
| qasper_1809.07629 | baeb67850779 | What datasets did they use? | The E2E NLG challenge dataset  | 你问的数据集对应论文中的 **E2E NLG 挑战数据集**（E2E NLG | L0→L1→answer_L1 | 5 | pass |
| qasper_2002.08307 | dc4096b8bab0 | How they observe that fine-tuning BERT | we compressed English BERT usi | 你问的“fine-tuning 不改善可剪枝性”对应论文中“Downstre | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2002.08307 | c4c9c7900a04 | How much is pre-training loss increase | The increase is linearly from  | 你问的“Low/Medium/Hard 级别剪枝”对应论文中的“低（30-4 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1905.13497 | d6d29040e7fa | Which datasets do they evaluate on? | PDP-60 WSC-273 | 你问的评测数据集对应论文中的 PDP-60 和 WSC-273 两个数据集。 | L0→answer_L0 | 5 | pass |
| qasper_1905.13497 | 21663d2744a2 | How does their model differ from BERT? | Their model does not differ fr | 你问的“their model”对应论文中提出的方法——即直接利用开箱即用（ | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2003.05522 | 86a93a2d1c19 | How does Frege's holistic and function | interpretation of Frege's work | 你问的 Frege 整体论与功能主义意义观，对应论文中 Tugendhat  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2003.05522 | 6090d3187c41 | What does Frege's holistic and functio | Only in the context of a sente | 你问的 Frege 的整体论与功能主义意义观，对应论文中 Frege 提出的 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1707.03904 | dcb18516369c | Which retrieval system was used for ba | The dataset comes with a ranke | 你问的“用于基线的检索系统”对应论文中的 Lucene 索引系统。对于 Qu | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1911.11951 | 79620a2b4b12 | What are the state-of-the-art models f | To the best of our knowledge,  | 你问的“state-of-the-art models”对应论文中用于立场检 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1910.07134 | aaed6e30cf16 | What is WNGT 2019 shared task? | efficiency task aimed  at redu | 你问的 WNGT 2019 shared task 对应论文中的“2019  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.02402 | f1bd66bb354e | How do slot binary classifiers improve | by adding extra supervision to | 你问的“slot binary classifiers”对应论文中的可请求槽 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1908.02402 | 25fd61bb20f7 | What baselines have been used in this  | NDM, LIDM, KVRN, and TSCP/RL | 你问的 baselines 对应论文中的四个基线方法，均出自条目[4]：ND | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.00876 | 1d9b953a324f | Does API provide ability to connect to | yes | 你问的 API 对应论文中的 Torch-Struct 库的分布 API。该 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2002.00876 | 093039f97480 | Is this library implemented into Torch | It uses deep learning framewor | 你问的“library”对应论文中的 Torch-Struct。该库并非框架 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.00876 | 8df89988adff | What baselines are used in experiments | Typical implementations of dyn | 抱歉，我无法从该论文中找到答案，因为检索到的内容未提及实验中所用的具体基线方 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2002.00876 | 94edac71eea1 | What general-purpose optimizations are | Parallel Scan Inference Vector | 你问的“通用优化”对应论文中第 1 节描述的三种通用 GPU 优化技术。具体 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1703.04617 | a891039441e0 | What MC abbreviate for? | machine comprehension | 很抱歉，我无法从这篇论文中找到“MC”的缩写含义，因为文中并未提供相关定义或 | L0→L1→L3→answer_unknown | 1 | unknown_ok |
| qasper_1703.04617 | 73738e42d488 | how much of improvement the adaptation | 69.10%/78.38% | 你问的“adaptation model 的提升幅度”对应论文中的 soft | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1703.04617 | 6c8bd7fa1cfb | what is the architecture of the baseli | word embedding, input encoder, | 你问的 baseline 模型架构对应论文中的“The Baseline M | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1703.04617 | fa218b297d9c | What is the exact performance on SQUAD | Our model achieves a 68.73% EM | 你问的“SQuAD 上的确切性能”对应论文中模型在 SQuAD 测试集与开发 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.05058 | 5d85d7d4d013 | How they add human prefference annotat | human preference annotation is | 你问的“如何将人类偏好标注加入微调过程”对应论文中 comparative  | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2002.05058 | 6dc9960f046e | What previous automated evalution appr | Text Overlap Metrics, includin | 你问的“先前自动评估方法”对应论文中总结的几类方法。作者提到的具体方法包括： | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2002.05058 | 75b69eef4a38 | How much better peformance is achieved | Pearson correlation to human j | 你问的“考虑所提指标训练模型后，在人类评估中性能提升多少”对应论文中“将评估 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2002.05058 | 7488855f09b9 | Do the authors suggest that proposed m | no | 否。论文并未建议用所提指标完全取代人工评估；相反，作者将人工评估视为衡量指标 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1805.07966 | 536e4a39b654 | Is the affect of a word affected by co | — | 否。你问的“单词的情感是否受语境影响”对应论文中关于**一般用途情感词典（如 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.05467 | 4b8a0e99bf3f | To what other competitive baselines is | LSTMs with and without attenti | 你问的“this approach”对应论文中提出的模型（MinAvgOut | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_2001.05467 | a09633584df1 | How is human evaluation performed, wha | Through Amazon MTurk annotator | 你问的人工评估方式对应论文中的 Amazon MTurk 人类研究，采用成对 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2001.05467 | 5e9732ff8595 | How much better were results of the pr | on diversity 6.87 and on relev | 你问的“proposed models 相对 base LSTM-RNN 模 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2001.05467 | 58edc6ed7d69 | Which one of the four proposed models  | the hybrid model MinAvgOut + R | 你问的“四个模型中表现最佳者”对应论文中的 MinAvgOut+RL 混合模 | L0→answer_L0 | 5 | pass |
| qasper_1911.03681 | 83f14af3ccca | Which of the two ensembles yields the  | Answer with content missing: ( | 你问的“两个集成”对应论文中的 **CONCAT ensemble** 与  | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1911.03681 | 0154d8be7721 | What are the two ways of ensembling BE | mean-pooling their outputs (AV | 你问的“集成 BERT 和 E-BERT 的两种方式”对应论文中的 E-BE | L0→L1→answer_L1 | 5 | pass |
| qasper_1911.03681 | e737cfe0f6cf | How is it determined that a fact is ea | filter deletes all KB triples  | 你问的“如何判定一个事实容易被猜出”对应论文中 LAMA-UHN 的构建过程 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1905.06906 | dfbab3cd991f | For the purposes of this paper, how is | reviews under distinct product | 你问的“如何判定领域特有知识”对应论文中门控机制对 n-gram 权重的学习 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1905.06906 | df510c85c277 | Does the fact that GCNs can perform we | no | 抱歉，我无法从该论文中找到支撑这一问题的内容。检索到的段落均未讨论任务难度或 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1905.06906 | d95180d72d32 | Are there conceptual benefits to using | yes | 你问的“GCN”对应论文中的“gated architectures”（门控 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1808.02022 | 5dc2f79cd807 | Which news organisations are the headl | BBC and CNN | 你问的新闻机构来源对应论文流水线第 1 步中的数据采集对象。证据明确指出，标 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1903.03467 | 2fea3c955ff7 | What conclusions are drawn from the sy | our method enables to control  | 你问的“syntactic analysis 得出的结论”对应论文中基于细粒 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1903.03467 | faa4f28a2f29 | What type of syntactic analysis is per | Speaker's Gender Effects Inter | 你问的句法分析对应论文中基于希伯来语依存解析器的细粒度分析。该分析使用希伯来 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1903.03467 | da068b209888 | How is it demonstrated that the correc | correct information substantia | 你问的“如何证明正确性别与数信息被注入”对应论文中的实验设计部分。论文通过对 | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_1903.03467 | 0d6d5b6c0055 | Which neural machine translation syste | Google's machine translation s | 你问的 NMT 系统对应论文中的谷歌机器翻译系统（GMT），即 Google | L0→L1→answer_L1 | 5 | pass |
| qasper_1903.03467 | edcde2b675cf | What are the components of the black-b | supply an NMT system with know | 你问的“black-box context injection system | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.12673 | ff9495982b88 | What is proof that proposed functional | estimated test accuracy is hig | 你问的“泛化误差近似良好的实证证据”对应论文中的外推实验。在ImageNet | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.12673 | dcd8138f0cba | How is proposed functional form constr | — | 你问的“proposed functional form”对应论文中提出的泛 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1907.11499 | e1f559da7fa5 | What is the performance of their model | — | 你问的“their model”对应论文中提出的 DetNet 模型。在句子 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.11499 | a96a1a354cb3 | Which text genres did they experiment  | — | 你问的“text genres”对应论文中实验所用的文本类型。论文实验涉及两 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.11499 | 427252648173 | What domains are detected in this pape | Answer with content missing: ( | 你问的“检测的领域”对应论文中 DetNet 模型在 Wikipedia 数 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1606.08495 | 5efed109940b | Do they use skipgram version of word2v | yes | 是的，论文使用的是 skipgram 版本的 word2vec。文中明确说明 | L0→L1→answer_L1 | 5 | pass |
| qasper_1606.08495 | b8137eb0fa0b | What domains are considered that have  | relational entities general te | 你问的“具有如此大规模词表的领域”对应论文中列举的 word2vec 非常规 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1606.08495 | 38b527783330 | Do they perform any morphological toke | no | 抱歉，我无法从该论文中找到关于形态学分词（morphological tok | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1606.08495 | 6b2fbc1c0834 | How many nodes does the cluster have? | — | 抱歉，我无法从该论文中找到集群节点数量的信息，因为文中未提供相关数据或描述。 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1904.04019 | a88f8cae1f59 | Do they evaluate only on English datas | — | 否。你问的“是否只在英语数据集上评测”对应论文中使用的四个语料库，证据显示这 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1904.04019 | bea60603d78b | What baseline models are used? | — | 你问的基线模型对应论文中用于对比的已有方法。论文明确提及与 **Poira  | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1904.04019 | 37db7ba2c155 | What classical machine learning algori | Support Vector Machine (SVM) L | 你问的经典机器学习算法对应论文中比较的四种分类器，分别是：逻辑回归、支持向量 | L0→L1→answer_L1 | 5 | pass |
| qasper_1904.04019 | c2cbc2637761 | What are the different methods used fo | Support Vector Machine (SVM),  | 你问的“不同语料库所用的不同方法”对应论文中的四组实验设置。论文使用了四种机 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1904.04019 | 774ead7c642f | In which domains is sarcasm conveyed i | Amazon reviews | 你问的“sarcasm conveyed in different ways | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.00324 | 1763a029daca | How big is the improvement over the st | AGDT improves the performance  | 你问的“state-of-the-art 提升幅度”对应论文中 AGDT 与 | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_1909.00324 | f9de9ddea0c7 | Is the model evaluated against other A | yes | 是的，论文将 AGDT 与多个基于方面的情感分析（ABSA）基线模型进行了对 | L0→answer_L0 | 5 | pass |
| qasper_1909.01515 | 4226a1830266 | What meta-information is being transfe | high-order representation of a | 你问的“被传递的元信息”对应论文 MetaR 中的两种关系特定元信息：**关 | L0→L1→answer_L1 | 4 | pass |
| qasper_1909.01515 | 5fb348b2d7b0 | What datasets are used to evaluate the | NELL-One, Wiki-One | 你问的评估数据集对应论文中的 NELL-One 和 Wiki-One 两个少 | L0→answer_L0 | 4 | pass |
| qasper_1804.08186 | 626873982852 | what evaluation methods are discussed? | document-level accuracy precis | 你问的“evaluation methods”对应论文中“Empirical | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1804.08186 | b3a09d2e3156 | what are the off-the-shelf systems dis | Answer with content missing: ( | 你问的 off-the-shelf 系统对应论文中第 1 段列举的现成语言识 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1602.01595 | b0fd686183b0 | How does the model work if no treebank | train the parser on six other  | 你问的“无树库时模型如何工作”对应论文中“Target Languages  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1602.01595 | 7065e6140dba | How many languages have this parser be | seven | 你问的“parser”对应论文中提出的 MaLOPa 多语言解析器。该解析器 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1910.08210 | 8816333fbed2 | How is data for RTFM collected? | — | 抱歉，我无法从该论文中找到关于RTFM数据收集方式的具体描述；检索到的内容未 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1910.08210 | 37e8f5851133 | How better is performance of proposed  | Proposed model achive 66+-22 w | 你问的“proposed model”对应论文中提出的模型（文中以占位符表示 | L0→L1→L3→answer_L3 | 2 | fail |
| qasper_1910.08210 | c9e9c5f44364 | How does propose model model that capt | We first encode text inputs us | 你问的“propose model”对应论文中提出的模型，其核心是 () 层 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2001.06888 | 0106bd9d54e2 | Which social media platform is explore | twitter | 你问的社交媒体平台对应论文中的 Twitter。论文明确指出其研究对象是 T | L0→L3→answer_L3 | 5 | pass |
| qasper_2001.06888 | e015d033d4ee | What datasets did they use? | BIBREF8 a refined collection o | 你问的数据集对应论文中的 Twitter 数据集。该数据集在 [3] 中被描 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.06888 | 8a871b136cce | What are the baseline state of the art | Stanford NER, BiLSTM+CRF, LSTM | 抱歉，我无法从该论文中找到关于基线模型的具体信息；文中仅笼统提及与state | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2001.10179 | 3604c4fba0a8 | Is their implementation on CNN-DSA com | no | 抱歉，我无法从该论文中找到您问题的答案。检索到的内容未涉及CNN-DSA，也 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2001.10179 | 38e2f07ba965 | Does this implementation on CNN-DSA le | — | 抱歉，我无法从该论文中找到与“CNN-DSA”相关的任何内容，也未涉及该实现 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2001.10179 | 931a2a13a1f6 | How is Super Character method modified | simply split the image into tw | 你问的“Super Character 方法如何修改以处理表格数据”对应论文 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2003.07996 | 6baf5d773975 | Which four languages do they experimen | German, English, Italian, Chin | 你问的“四种语言”对应论文中实验涉及的语言。根据证据，论文实验使用的语言包括 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.06675 | 526ae24fa861 | How big are improvements with multilin | relative WER improvement of 10 | 你问的多语言训练相对单语言训练的改进幅度，对应论文中“speaker-ope | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.06675 | 8a5254ca726a | How much transcribed data is available | Transcribed data is available  | 你问的“可用的已转写阿伊努语数据量”对应论文中阿伊努语语音语料库的构建情况。 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2002.06675 | 3c0d66f9e55a | What is the difference between speaker | In the speaker-closed conditio | 你问的说话人开放与封闭设定的差异，对应论文中的 ASR 实验条件。在说话人封 | L0→L1→answer_L1 | 5 | pass |
| qasper_1803.07771 | 0b9021cefca7 | What are the other models they compare | CNN-C CNN-W CNN-Lex-C CNN-Lex- | 你问的“其他对比模型”对应论文中与所提 Tl-LSTM 进行比较的三个竞争算 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1803.07771 | 6ad92aad46d2 | What is the agreement value for each d | — | 抱歉，我无法从该论文中找到每个数据集的agreement值，因为文中未提供任 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1803.07771 | 4fdc707fae57 | How many annotators participated? | — | 你问的“有多少标注者参与”对应论文中的两阶段标注策略，两阶段人数不同：第一阶 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1803.07771 | 2d307b43746b | How long are the datasets? | Travel dataset contains 4100 r | 抱歉，我无法从该论文中找到关于数据集长度的具体信息，因为文中未提及数据集的规 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1803.07771 | fe90eec1e3cd | What are the sources of the data? | User reviews written in Chines | 你问的数据来源对应论文中编译的三个中文情感语料库。这些数据来自在线数据，涵盖 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1803.07771 | 9d5df9022cc9 | What is the new labeling strategy? | They use a two-stage labeling  | 你问的“new labeling strategy”对应论文中的“two-s | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1910.02789 | d79d897f94e6 | What result from experiments suggest t | Average reward across 5 seeds  | 你问的“自然语言代理更鲁棒”的实验结果，对应论文中 NLP 表示在状态空间膨 | L0→answer_L0 | 4 | pass |
| qasper_1910.02789 | 599d9ca21bbe | How better is performance of natural l | — | 你问的“natural language based agents 的性能提 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.02789 | 827464c79f33 | How much faster natural language agent | — | 抱歉，我无法从该论文中找到答案。检索到的内容未提供自然语言智能体在实验中收敛 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1910.02789 | 8e857e44e423 | What experiments authors perform? | a basic scenario, a health gat | 你问的实验对应论文中在 ViZDoom 环境下对自然语言表示与视觉、特征等表 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1910.02789 | 084fb7c80a24 | How is state to learn and complete tas | represent the state using natu | 你问的“如何用自然语言表示状态以学习和完成任务”，对应论文中提出的“自然语言 | L0→L3→answer_L3 | 4 | pass |
| qasper_1706.01450 | 0fa81adf0066 | Which components of QA and QG models a | parameter sharing | 你问的“QA 和 QG 模型共享的组件”对应论文中 JointQA 模型的参 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1706.01450 | 4ade72bfa28b | How much improvement does jointly lear | We see that A-gen performance  | 你问的“联合学习问答（QA）与问题生成（QG）相比仅训练 QA 的提升幅度” | L0→answer_L0 | 4 | pass |
| qasper_1611.01400 | d32b6ac003cf | what were the baselines? | Rank by the number of times a  | 你问的 baselines 对应论文中系统对比的多种基线方法。论文将系统与以 | L0→L1→answer_L1 | 5 | pass |
| qasper_1611.01400 | c10f38ee97ed | what is the supervised model they deve | SVMRank | 你问的“他们开发的监督模型”对应论文中测试的三种监督模型，而非单一模型。论文 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1611.01400 | 340501f23ddc | what is the size of this built corpus? | 90 annotated documents with 5  | 你问的“built corpus”对应论文中创建的基准数据集。该数据集由 9 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1611.01400 | fbb85cbd41de | what crowdsourcing platform is used? | asked the authors to rank by c | 你问的众包平台对应论文中提到的 **Amazon Mechanical Tu | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1901.00570 | 524abe0ab77d | Do the authors suggest any future exte | yes | 是的，作者提出了未来扩展方向。他们指出这项工作可以扩展为提前一天预测未来事件 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.00570 | 858c51842fc3 | Which of the classifiers showed the be | Logistic regression | 你问的“表现最好的分类器”对应论文中实际选用的分类器是 **Naive Ba | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1901.00570 | 7c9c73508da6 | Were any other word similar metrics, b | yes | 是的，除了 Jaccard 指标外，论文还测试了其他词语相似度指标。在特征选 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1901.00570 | 7b2bf0c1a24a | How are the keywords associated with e | By using a Bayesian approach   | 你问的关键词（与抗议等事件关联）选择方法，对应论文中的“尖峰匹配方法”（Sp | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2003.03612 | 62736ad71c76 | How is order of binomials tracked acro | draw our data from news public | 你问的“二项式顺序如何随时间追踪”对应论文中的 **movement（运动性 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.03612 | aaf50a6a9f44 | What types of various community texts  | news publications, wine review | 你问的“各类社区文本”对应论文中用于研究二项式全局结构的数据来源。论文主要调 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2003.03612 | a19172324418 | Are there any new finding in analasys  | Trinomials are likely to appea | 你问的三项式分析新发现对应论文中关于三项式排序模式的内容。是的，有三项式独有 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.03612 | 574f17134e4d | What new model is proposed for binomia | null model | 你问的新模型对应论文中提出的**近因原则（Proximity Princip | L0→answer_L0 | 1 | fail |
| qasper_2003.03612 | 41fd359b8c14 | How was performance of previously prop | close to random, | 你问的“previously proposed rules”对应论文中的传统 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2003.03612 | d216d715ec27 | What previously proposed rules for pre | word length, number of phoneme | 你问的“先前提出的预测二项式排序的规则”对应论文中的“Basic Featu | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.03612 | ba973b13f26c | What online text resources are used to | news publications, wine review | 你问的“用于测试二项式列表的在线文本资源”对应论文中的数据来源。论文使用的数 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1911.06747 | 65e2f97f2fe8 | How did they measure effectiveness? | number of dialogs that resulte | 你问的“effectiveness”对应论文中的“success rate” | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.13375 | 9ab43f941c11 | What approach did previous models use  | Only MTMSM specifically tried  | 你问的“previous models”对应论文中的 NAQANET 和 N | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.13375 | 5a02a3dd2648 | How they use sequence tagging to answe | To model an answer which is a  | 你问的“如何用序列标注回答多跨度问题”对应论文中的多跨度头训练目标与正确标签 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1909.13375 | 579941de2838 | What is difference in peformance betwe | For single-span questions, the | 你问的“proposed model 与 state-of-the-art  | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1909.13375 | 9a65cfff4d99 | What is the performance of proposed mo | The proposed model achieves  E | 你问的“proposed model 在整个 DROP 数据集上的表现”对应 | L0→answer_L0 | 2 | fail |
| qasper_1909.13375 | a9def7958eac | What is the previous model that attemp | MTMSN BIBREF4 | 你问的“先前尝试处理 multi-span 问题的模型”对应论文中的 **M | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1711.10124 | 280f863cfd63 | Are their corpus and software public? | yes | 是的，论文中的语料库和软件都是公开的。正文明确指出“This system, | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1602.01208 | 23d0637f8ae7 | How do they show that acquiring names  | unsupervised morphological ana | 你问的“获取地名如何帮助自定位”对应论文中通过空间概念（SpCoA）修正自定 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1602.01208 | 21c104d14ba3 | How do they evaluate how their model a | PAR score | 你问的“如何评估模型习得词汇”对应论文中的空间概念估计精度实验。他们通过比较 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1602.01208 | d557752c4706 | Which method do they use for word segm | unsupervised word segmentation | 你问的词分割方法对应论文中的 **latticelm**，这是一种无监督词分 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1602.01208 | 1bdf7e9f3f80 | Does their model start with any prior  | no | 否。你问的“模型是否具备先验词汇知识”对应论文中提出的 SpCoA 方法，其 | L0→L1→answer_L1 | 5 | pass |
| qasper_1902.00821 | 3213529b6405 | What is the baseline model used? | The baseline models used are D | 你问的“baseline model”对应论文中用于对比的多个基线模型。证据 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2001.06286 | 589be705a5cc | What data did they use? | the Dutch section of the OSCAR | 你问的训练数据对应论文中的 RobBERT 预训练语料。RobBERT 是在 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2001.06286 | 6e962f1f2306 | What is the state of the art? | BERTje BIBREF8 an ULMFiT model | 抱歉，我无法从这篇论文中找到关于“state of the art”的具体内 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2001.06286 | 594a6bf37eab | What language tasks did they experimen | sentiment analysis the disambi | 你问的实验任务对应论文中的两个任务：情感分析（Sentiment Analy | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1708.09157 | a43c400ae37a | How are character representations from | shared character embeddings fo | 你问的“各语言字符表示如何联合”对应论文中提出的跨语言多任务迁移机制。论文通 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1708.09157 | 4056ee2fd7a0 | On which dataset is the experiment con | We use the morphological taggi | 你问的实验数据集对应论文中的 **Universal Dependencie | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1602.06291 | 43ee69902a5f | Does CLSTM have any benefits over BERT | — | 抱歉，根据现有论文内容，无法回答该问题。文中仅对比了CLSTM与LSTM，未 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.01638 | c65b6470b7ed | How are seed dictionaries obtained by  | the latest CLWE developments a | 你问的“完全无监督方法如何获得种子词典”对应论文中的 C1 组件（Seed  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.01638 | 6e2899c444ba | How does BLI measure alignment quality | we use mean average precision  | 你问的 BLI 对齐质量度量对应论文中的双语词典归纳（BLI）评测任务，其采 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.01638 | 896e99d7f8f9 | What methods were used for unsupervise | Unsupervised CLWEs. These meth | 你问的无监督 CLWE 方法对应论文中的 **vecmap** 模型 [1] | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1610.08597 | f8d32088d17b | Do the authors report on English datas | yes | 你问的“English datasets”对应论文中的 Twitter 用户 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1610.08597 | 4f0f446bf451 | Which supervised learning algorithms a | Logistic Regression (LR) Rando | 你问的实验中所用的监督学习算法，对应论文中的三种学习算法：逻辑回归（Logi | L0→L1→answer_L1 | 5 | pass |
| qasper_1610.08597 | 663b36f99ad2 | How in YouTube content translated into | words extracted from YouTube v | 你问的“YouTube 内容如何转化为向量格式”对应论文中的处理流程：You | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1610.08597 | be595b201754 | How is the ground truth of gang member | text in their tweets and profi | 抱歉，我无法从该论文中找到关于gang成员ground truth如何建立（ | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2003.02639 | 639c145f0bcb | What are three possible phases for lan | Phase I: $\langle cc \rangle $ | 你问的三个语言形成相区对应论文中的相区 I、II、III，分别由歧义参数 \ | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1907.00854 | 4d5e2a83b517 | how many domains did they experiment w | 2 | 你问的实验领域对应论文中的两个主题（topics）。论文在示例使用中为问答系 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1907.00854 | 2c3b2c3bab6d | what pretrained models were used? | BiDAF BERT | 你问的预训练模型对应论文中用于两个主题的 NER 模型。针对 Medical | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1707.06878 | 2a1069ae3629 | Do they use a neural model for their t | no | 否。你问的“是否使用神经模型”对应论文中的 WSD 系统，其模型并非基于神经 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.04453 | 479d334b79c1 | Does the performance necessarily drop  | yes | 你问的“更多控制是否必然导致性能下降”对应论文中调节 INLINEFORM0 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1909.04453 | b02d2d351bd2 | How does the model perform in comparis | — | 你问的“模型与端到端标题生成模型的对比表现”对应论文中的 VRS 模型与 S | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.04453 | a035472a5c6c | How is the model trained to do only co | target some heuristically extr | 你问的“模型如何训练以仅做内容选择”对应论文中的 VRS 方法。VRS 通过 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.09986 | 56a3c9bd74c6 | How is fluency of generated text evalu | manually reviewed | 你问的“生成文本流畅度如何评估”对应论文中的自动指标与人工分析两种方式。

 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.09986 | 73cd785d474b | How is faithfulness of the resulting t | manually inspect | 你问的“faithfulness 评估”对应论文中的输出验证（output  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.09986 | 65461516098e | How are typing hints suggested? | concatenating to the embedding | 你问的“typing hints”对应论文中的“typing informa | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.09986 | 92d09654011d | What is the effectiveness plan generat | clear mapping between plans an | 你问的“plan generation 的有效性”对应论文中**新提出的神经 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1909.09986 | 8a94766f8251 | How is neural planning component train | plan-to-DFS mapping to perform | 你问的神经规划组件训练方式对应论文中的“planner”训练过程。该组件通过 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.00194 | 496e81769a8d | Which language(s) are found in the WSD | WSD is predominantly evaluated | 你问的 WSD 数据集涉及的语言对应论文中的英语和中文两种。英语任务使用 S | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.00194 | f103789b85b0 | What datasets are used for testing? | Senseval-2 (SE2), Senseval-3 ( | 你问的“测试数据集”对应论文中不同语言任务的评测集。对于英语全词WSD，测试 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1709.01256 | 3bfb8c12f151 | What metrics are used to evaluation re | precision recall F-measure | 你问的评估指标对应论文中的 precision、recall 和 F-mea | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1709.01256 | 3f85cc5be844 | How large is the Wikipedia revision du | eight GB | 你问的 Wikipedia revision dump 数据集对应论文中的  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1709.01256 | 126e8112e26e | What are simulated datasets collected? | There are 6 simulated datasets | 你问的“模拟数据集如何收集”对应论文中的 simulated data se | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1709.01256 | be08ef81c3cf | Which are the state-of-the-art models? | WMD VSM PV-DTW PV-TED | 你问的“state-of-the-art models”对应论文中提出的语义 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1805.07513 | 5c4a2a3d6e02 | Do they compare with the MAML algorith | no | 抱歉，根据检索到的论文内容，我无法回答您的问题——文中并未提及或比较MAML | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.13362 | ba56afe42690 | What are the datasets used for the tas | Datasets used are Celex (Engli | 你问的“任务所用数据集”对应论文中用于评估语言无关音节切分器的六种语言数据集 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.13362 | 14634943d96e | What is the accuracy of the model for  | Authors report their best mode | 你问的“六个语言的准确率”对应论文中 Base 模型在六个评测语言上的表现。 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1909.13362 | d71cb7f3aa58 | Which models achieve state-of-the-art  | CELEX (Dutch and English) - SV | 你问的“达到最先进性能的模型”对应论文中提出的 **Base 模型（BiLS | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1909.13362 | f6556d2a8b42 | Is the LSTM bidirectional? | yes | 是的，你问的 LSTM 对应论文中的 BiLSTM，即双向 LSTM。该模型 | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.08167 | 37016cc987d3 | How are different domains weighted in  | To achieve this purpose, we in | 你问的“不同域如何加权”对应论文中的 WDIRL 框架，其加权策略仅作用于* | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.08167 | b3dc6d95d157 | How is DIRL evaluated? | Through the experiments, we em | 你问的 DIRL 评估对应论文中的实验设计部分。论文通过对比多个模型在跨领域 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.08167 | cc5d3903913f | Which sentiment analysis tasks are add | 12 binary-class classification | 你问的情感分析任务对应论文中的两类跨域任务：一是12个二分类跨域情感分析任务 | L0→L1→answer_L1 | 5 | pass |
| qasper_2004.03034 | ca26cfcc755f | How better are results compared to bas | F1 score of best authors' mode | 你问的“与基线模型相比结果提升多少”对应论文中多个模型与基线的对比。结合父论 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2004.03034 | 6cdd61ebf84a | What models that rely only on claim-sp | SVM with RBF kernel | 你问的“仅依赖 claim 自身语言特征的基线模型”对应论文中的 SVM（含 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2004.03034 | 8e8097cada29 | How is pargmative and discourse contex | While evaluating the impact of | 你问的“pragmatic and discourse context 如何 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2004.03034 | 951098f0b716 | What annotations are available in the  | 5 possible impact labels for a | 你问的数据集中的标注对应论文中的“impact votes”（影响力投票）。 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.04899 | c6e63e3b8074 | What are the supported natural command | Set/Change Destination Set/Cha | 你问的“支持的 natural commands”对应论文中标注的 10 种 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.04899 | 4ef2fd79d598 | What is the size of their collected da | 3347 unique utterances | 你问的数据集规模对应论文中收集的自动驾驶舱内多模态多轮对话数据集。该数据集包 | L0→L1→answer_L1 | 5 | pass |
| qasper_1901.04899 | 40e3639b79e2 | Did they compare against other systems | yes | 是的，论文将所提方法与多个竞争基线进行了对比。实验结果显示，其意图识别和槽位 | L0→answer_L0 | 4 | pass |
| qasper_1901.04899 | 8383e52b2adb | What intents does the paper explore? | Set/Change Destination Set/Cha | 你问的“论文探索的意图”对应论文中标注的10种乘客意图。论文明确列出了全部1 | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.00279 | 6b9310b577c6 | What are some guidelines in writing in | if a vernacular paragraph cont | 抱歉，我无法从这篇论文中找到关于“如何撰写输入白话文以引导模型生成更好内容” | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1909.00279 | d484a71e23d1 | How much is proposed model better in p | Perplexity of the best model i | 你问的“proposed model”对应论文中的 **Transforme | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1909.00279 | 5787ac3e8084 | What dataset is used for training? | We collected a corpus of poems | 你问的训练数据集对应论文中的训练集与验证集，由两个未对齐的语料库构成：诗歌语 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1603.00968 | f7d67d6c6fbc | How much faster is training time for M | It is an order of magnitude mo | 你问的 MGNC-CNN 的训练时间对比，对应论文中与更复杂模型（即 Yin | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1603.00968 | 085147cd3215 | What are the baseline models? | MC-CNN
MVCNN
CNN | 你问的基线模型对应论文中的两个对比对象：一是仅利用单一词嵌入集合的标准CNN | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1603.00968 | c0035fb1c2b3 | By how much of MGNC-CNN out perform th | In terms of Subj the Average M | 你问的“MGNC-CNN 相比基线的提升幅度”对应论文中的结果部分。证据显示 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1603.00968 | a8e4a67dd67a | What dataset/corpus is this evaluated  | SST-1 SST-2 Subj  TREC  Irony | 你问的评测数据集对应论文中的 **Irony** 数据集，因为该数据集在正文 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1603.00968 | 34dd0ee1374a | What are the comparable alternative ar | standard CNN C-CNN MVCNN | 你问的“可比较的替代架构”对应论文中与 MGNC-CNN 进行对比的基线模型 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.09491 | d28260b5565d | How they measure robustness in experim | We empirically provide a formu | 抱歉，我无法从该论文中找到关于“如何在实验中度量鲁棒性”的具体内容，因为检索 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1909.09491 | 8670989ca392 | Is new method inferior in terms of rob | — | 你问的“新方法在鲁棒性上是否不如MIRA”对应论文中的PL(k)方法。基于实 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.09491 | 923b12c0a50b | What experiments with large-scale feat | Plackett-Luce Model for SMT Re | 你问的大规模特征实验对应论文中的两个任务。第一个任务使用8个基本特征加220 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1910.04519 | 13149342ccbb | How big is dataset used for fine-tunin | a total of 2,560 pseudo-tweets | 你问的“用于微调模型以检测个人陈述中红旗医疗症状的数据集”对应论文中的 Me | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.04519 | fbe149bd7686 | Is there any explanation why some choi | translations that were reasona | 你问的语言对表现差异的解释，对应论文中关于零样本迁移实验的分析。是的，论文给 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_2003.01472 | 90f80a94faba | Did they experiment with the tool? | yes | 是的，论文中确实用 Seshat 工具进行了实验。具体而言，Seshat 被 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.01472 | 5872279c5165 | Can it be used for any language? | — | 你问的“it”对应论文中的 Seshat 工具。根据证据，Seshat 支持 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2003.01472 | da55878d048e | Is this software available to the publ | yes | 你问的“this software”对应论文中提出的工具 Seshat。证据 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1808.06834 | e09e89b3945b | How many speeches are in the dataset? | 5575 speeches | 你问的数据集中的演讲数量对应论文中标注的演讲总数。论文明确指出，他们用上述四 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1808.06834 | 0cf5132ac790 | What classification models were used? | fastText and SVM BIBREF16 | 你问的分类模型对应论文中用于两项分类任务的模型，即 **fastText** | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1808.06834 | 1d860d7f615b | Do any speeches not fall in these cate | — | 否。你问的“这些类别”对应论文中提出的四类互斥演讲语义分类法。论文明确指出， | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1701.02877 | 94e0cf443458 | What web and user-generated NER datase | MUC, CoNLL, ACE, OntoNotes, MS | 你问的“web 和用户生成内容的 NER 数据集”对应论文中的社交媒体（so | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1808.09633 | 9a4aa0e4096c | What text sequences are associated wit | abstracts sentences | 你问的“与每个顶点关联的文本序列”对应论文中的集合 \(T\)，即分配给每个 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1808.09633 | 1d1ab5d8a24d | How long does it take for the model to | — | 抱歉，论文中未提供模型运行时间或训练/推理耗时的数据，因此无法回答该问题。缺 | L0→L1→L3→answer_unknown | 5 | pass |
| qasper_1902.07285 | a458c649a793 | Which strategies show the most promise | At the moment defender can dra | 你问的“最有前景的防御策略”对应论文中介绍的文本对抗防御方法。论文指出，** | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1902.00330 | 9aca4b89e18c | How fast is the model compared to base | — | 抱歉，我无法从该论文中找到模型与基线在运行速度上的直接比较数据；现有内容仅涉 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1902.00330 | b0376a7f67f1 | How big is the performance difference  | Comparing with the highest per | 你问的“this method”对应论文中提出的 RLEL 模型，其与基线（ | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1902.00330 | dad8cc543a87 | What datasets used for evaluation? | AIDA-B ACE2004 MSNBC AQUAINT W | 你问的评测数据集对应论文中的 AIDA-B 及其他公开数据集。正文明确列出用 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1902.00330 | 0481a8edf795 | what are the mentioned cues? | output of global LSTM network  | 你问的“mentioned cues”在论文中对应的是**用于实体消歧的各类 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1709.10445 | ae60079da9d3 | Does this approach perform better than | yes | 是的，你问的“this approach”对应论文中提出的基于词源学词根的新 | L0→answer_L0 | 4 | pass |
| qasper_1709.10445 | 83f567489da4 | Have the authors tried this approach o | no | 你问的“其他语言”对应论文中作者对方法可移植性的讨论。作者明确表示，该方法可 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1912.01046 | 5bcc12680cf2 | What evaluation metrics were used in t | For sentence-level prediction  | 你问的实验评估指标对应论文中的 **tolerance accuracy（容 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1912.01046 | 7a53668cf2da | What kind of instructional videos are  | tutorial videos for a photo-ed | 你问的数据集中的教学视频类型，对应论文中的“76 tutorial vide | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1912.01046 | 8051927f914d | What baseline algorithms were presente | a sentence-level prediction al | 你问的基线算法对应论文中的 Baseline1 和 Baseline2 两个 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1912.01046 | 09621c9cd762 | What is the source of the triples? | a tutorial website about an im | 你问的 triples 来源对应论文中的 TutorialVQA 数据集。这 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1706.07206 | 2555ca85ff6b | Which datasets are used for evaluation | Stanford Sentiment Treebank | 你问的评测数据集对应论文中的 Stanford Sentiment Tree | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1605.07683 | a02696d4ab72 | How large is the Dialog State Tracking | 1,618 training dialogs, 500 va | 抱歉，我无法从该论文中找到关于Dialog State Tracking D | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1911.00133 | 4e8233826f9e | Is the dataset balanced across categor | yes | 你问的数据集类别平衡性，对应论文中的 Dreaddit 语料库。**是**， | L0→L1→answer_L1 | 5 | pass |
| qasper_1911.00133 | adae0c32a699 | What supervised methods are used? | Support Vector Machines (SVMs) | 你问的监督方法对应论文中实验的监督模型，包括非神经与神经两类。非神经模型有支 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.00133 | d0f831c97d34 | What labels are in the dataset? | binary label of stress or not  | 你问的数据集标签对应论文中标注任务里给每个文本片段设置的选项。标注员可为每个 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1911.00133 | 1ccfd288f746 | What categories does the dataset come  | abuse, social, anxiety, PTSD,  | 你问的数据集类别对应论文中的五个领域（domains）。根据正文，Dread | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1805.12032 | b54fc86dc2cc | How is speed measured? | time elapsed between the momen | 你问的“speed”对应论文中的“reaction delay”（反应延迟） | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1805.12032 | b43a8a0f4b84 | What is the architecture of their mode | we combine a text sequence sub | 你问的模型架构对应论文中提出的神经网络模型，它采用“late fusion” | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1805.12032 | b161febf86cd | What are the nine types? | agreement answer appreciation  | 你问的“nine types”对应论文中反应类型分类的最终口径。论文在结论部 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1709.02271 | cfbccb51f0f8 | How are discourse embeddings analyzed? | They perform t-SNE clustering  | 你问的“话语嵌入如何被分析”对应论文中的 t-SNE 聚类分析。研究者使用最 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1709.02271 | feb4e92ff160 | What was the previous state-of-the-art | character bigram CNN classifie | 你问的“previous state-of-the-art”对应论文中在作者 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1709.02271 | f10325d022e3 | How are discourse features incorporate | They derive entity grid with g | 你问的“话语特征如何融入模型”对应论文中的 CNN2-DE 与 CNN2-P | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1709.02271 | 5e65bb0481f3 | What discourse features are used? | Entity grid with grammatical r | 你问的话语特征对应论文中的 GR 与 RST 两种变体，均从实体网格（ent | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1811.11365 | cf15c4652e23 | Why is this work different from text-o | the image can play the role of | 你问的“this work”对应论文提出的带多模态注意力的无监督NMT模型。 | L0→answer_L0 | 3 | fail |
| qasper_1912.05238 | 8de64483ae96 | Do they report results only on English | yes | 抱歉，我无法从该论文中找到明确答案，因为检索内容未涉及实验数据的语言范围，缺 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1912.05238 | 4d062673b714 | What is the Moral Choice Machine? | Moral Choice Machine computes  | 你问的 Moral Choice Machine（MCM）对应论文中提出的一 | L0→answer_L0 | 2 | fail |
| qasper_1912.05238 | f4238f558d6d | How is moral bias measured? | Answer with content missing: ( | 你问的“道德偏差如何测量”对应论文中通过**道德子空间投影**来量化道德偏差 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1912.05238 | 3582fac4b270 | What sentence embeddings were used in  | — | 你问的“previous Jentzsch paper”中使用的句子嵌入，对 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1912.05238 | 96dcabaa8b6b | How do the authors define deontologica | These ask which choices are mo | 你问的“deontological ethical reasoning”对应 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.03243 | 04f72eddb1fc | How much more coverage is in the new d | 278 more annotations | 你问的“新数据集覆盖率提升”对应论文中“我们的金标准集”与2015年数据集的 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.03243 | f74eaee72cbd | How was coverage measured? | QA pairs per predicate | 你问的“coverage”对应论文中 QA-SRL 金标准相对于 PropB | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1911.03243 | 068dbcc117c9 | How was quality measured? | Inter-annotator agreement, com | 你问的质量评估对应论文中的标注一致性（Inter-Annotator Agr | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1911.03243 | 96526a14820b | How was the corpus obtained? | trained annotators BIBREF4 cro | 你问的语料获取方式对应论文中的标注样本采集过程。该语料是从Dense集中抽取 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1911.03243 | 32ba4d2d1519 | How are workers trained? | extensive personal feedback | 你问的“workers trained”对应论文中的标注人员筛选与训练流程。 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.03243 | 78c010db6413 | What is different in the improved anno | a trained worker consolidates  | 你问的“改进标注协议的不同之处”对应论文中提出的新众包标注协议。其核心差异在 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1911.03243 | a69af5937cab | How was the previous dataset annotated | the annotation machinery of BI | 你问的“previous dataset”对应论文中的 2015 年原始 Q | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1911.03243 | 8847f2c67619 | How big is the dataset? | 1593 annotations | 你问的数据集规模对应论文中从 Dense 集抽取的标注样本。该样本覆盖 Wi | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_2001.06354 | 68edb6a483cd | What metrics are used in challenge? | NDCG MRR recall@k mean rank | 你问的挑战赛评估指标对应论文中的视觉对话任务评估指标。该任务采用四个评估指标 | L0→L1→answer_L1 | 5 | pass |
| qasper_2001.06354 | f64531e460e0 | What model was winner of the Visual Di | — | 抱歉，我无法从提供的论文中找到2019年Visual Dialog挑战赛冠军 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2001.06354 | cee29acec4da | What model was winner of the Visual Di | DL-61 | 你问的 2018 年 Visual Dialog 挑战赛冠军对应论文中的 D | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2001.06354 | 7e54c7751dbd | Which method for integration peforms b | ensemble model | 你问的“哪种集成方法更好”对应论文中提出的两种模型组合方式：consensu | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2001.06354 | d3bcfcea00de | How big is dataset for this challenge? | 133,287 images | 你问的数据集规模对应论文中的 VisDial v1.0 数据集。该数据集按图 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.11879 | 80b9600e51a8 | Does the dataset contain non-English r | yes | 是的，该数据集包含非英语评论。你问的数据集对应论文中用于方面和观点词提取任务 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.11879 | 8aeaa3ccb8e0 | Does the paper report the performance  | no | 你问的“训练超过 8 个 epoch 的性能”对应论文中的 BERT/BER | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.12108 | b1ced2d6dcd1 | What is the performance of their syste | — | 你问的“their system”对应论文中提出的端到端共情聊天机器人 ** | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.12108 | f3be1a27df2e | What evaluation metrics are used? | — | 抱歉，我无法从该论文中找到关于评估指标的具体信息，因为检索到的内容未提及任何 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1907.12108 | a45a86b6a02a | What is the source of the dialogues? | — | 你问的“对话来源”对应论文中 CAiRE 训练所用的对话数据集。CAiRE  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.12108 | 1f1a9f2dd8c4 | What pretrained LM is used? | Generative Pre-trained Transfo | 你问的预训练语言模型对应论文中的 GPT（Generative Pre-tr | L0→L1→answer_L1 | 5 | pass |
| qasper_2003.09520 | cf63a4f9fe0f | How does the semi-automatic constructi | Automatic transcription of 500 | 你问的半自动构建过程对应论文中的“Incremental and Semi- | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.09520 | 8829f738bcdf | Does the paper report translation accu | yes | 抱歉，该论文未报告突尼斯语到Arabish自动翻译模型的翻译准确率；检索内容 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2002.02758 | 544b68f6f729 | What is their baseline? | Google's Neural Machine Transl | 抱歉，我无法从该论文中找到“基线”的具体定义或对比基准，因为检索内容中未提及 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1707.04662 | 92cfac12d958 | How do they evaluate interpretability  | we suggest to estimate the int | 你问的“如何评估可解释性”对应论文中提出的基于词向量矩阵的代数度量方法。论文 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2004.03061 | cbbcafffda71 | Was any variation in results observed  | It is observed some variabilit | 你问的语言类型学差异对应论文中在六种类型多样语言（Basque、Czech、 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2004.03061 | 1e59263f7aa7 | Does the work explicitly study the rel | no | 你问的“模型复杂度与语言结构编码的关系”对应论文中关于**探针（probe） | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1810.07091 | d13efa7dee28 | Do they differentiate insights where t | yes | 是的。论文明确区分了“手工特征”（hand-crafted features | L0→answer_L0 | 4 | pass |
| qasper_1810.07091 | 6cd01609c8af | Do they show an example of usage for I | yes | 是的，论文展示了 INFODENS 的使用示例。文中通过一个图例演示了一个简 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1810.07091 | 7a70fb11cb34 | What kind of representation exploratio | — | 你问的“INFODENS 提供的表示探索”对应论文中对手工特征与学习表示的组 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2003.00576 | 2ed02be0c183 | By how much they improve over the prev | 1.08 points in ROUGE-L over ou | 你问的“they”对应论文中的 **StructSum 组合模型（combi | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2003.00576 | be73a88d5b69 | Is there any evidence that encoders wi | yes | 是的，有证据。论文在引言中指出，带有潜在结构的编码器已被证明有益于多项任务， | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1705.10586 | 2df2f6e4efd1 | What other non-neural baselines do the | bag of words, tf-idf, bag-of-m | 你问的非神经基线对应论文中的传统模型，共三类：标准词计数方法 BoW（词频特 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1906.01946 | 1a7d2ade1614 | how many speeches are in the dataset? | 7,507 | 你问的数据集中的演讲数量对应论文中的 UNGA speeches datas | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.08845 | fc65f19a3015 | Is is known whether Sina Weibo posts a | no | 是，论文对此有明确判断。你问的“微博帖子由人工还是自动分类器审查”对应论文中 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2004.02083 | 99760276cfd6 | How does morphological analysis differ | Morphological analysis is the  | 你问的“morphological analysis”与“morpholog | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2004.02083 | 247e1fe05223 | What was the criterion used for select | — | 抱歉，我无法从该论文中找到答案。检索到的内容仅涉及词元数量、形态标注和实验设 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2004.02083 | 79cfd1b82c72 | What are the architectures used for th | DyNet | 你问的三个任务对应论文中的形态屈折（Inflectional Realiza | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2004.02083 | 9e1bf306658e | Which language family does Chatino bel | the Otomanguean language famil | Chatino 属于 Otomanguean 语系（Otomanguean  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2004.02083 | 25b24ab1248f | What system is used as baseline? | DyNet | 你问的基线系统对应论文中的**字符级编码器-解码器模型（character- | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_2004.02083 | 8486e06c03f8 | How was annotation done? | hand-curated collection of com | 抱歉，我无法从该论文中找到关于标注过程的具体信息；文中未提及标注人员、工具或 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2004.02083 | 27f575e90487 | How was the data collected? | — | 抱歉，我无法从该论文中找到关于数据收集方法的具体信息。现有内容未涉及数据采集 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1703.05260 | 352c081c9380 | What are the key points in the role of | — | 你问的“script knowledge 可被研究的关键作用”对应论文中关于 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1703.05260 | 18fbf9c08075 | Did the annotators agreed and how much | For event types and participan | 是的，标注者之间达成了一致。你问的“标注者一致性”对应论文中的“Inter- | L0→L1→answer_L1 | 4 | pass |
| qasper_1703.05260 | a37ef83ab6bc | How many subjects have been used to cr | four different annotators | 抱歉，我无法从该论文中找到答案。检索到的内容仅涉及标注流程、标注者数量和工作 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1905.10238 | 4146e1d8f799 | What dataset do they evaluate their mo | CoNLL-2012 shared task BIBREF2 | 你问的评估数据集对应论文中的 CoNLL-2012 共享任务语料，该语料选自 | L0→L1→answer_L1 | 5 | pass |
| qasper_1905.10238 | 42394c54a950 | What is the source of external knowled | counts of predicate-argument t | 你问的外部知识来源对应论文中的选择偏好（SP）知识库。该知识库通过统计语料中 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.02222 | 5067e5eb2cdd | Which matching features do they employ | Matching features from matchin | 你问的匹配特征对应论文匹配层中通过三种匹配函数生成的三个特征。它们分别是：联 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.09419 | 6852217163ea | What benchmark datasets are used for t | WN18RR FB15k-237 YAGO3-10 | 你问的基准数据集对应论文中用于链接预测评测的三个数据集：WN18RR、YAG | L0→answer_L0 | 3 | fail |
| qasper_1911.09419 | cd1ad7e18d8e | What are state-of-the art models for t | TransE DistMult ComplEx ConvE  | 你问的“state-of-the-art models”对应论文中作为对比基 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.09419 | 9c9e90ceaba3 | How better does HAKE model peform than | 0.021 higher MRR, a 2.4% highe | 你问的 HAKE 相对 state-of-the-art 方法的性能提升，对 | L0→answer_L0 | 5 | pass |
| qasper_1911.09419 | 2a058f8f6bd6 | How are entities mapped onto polar coo | radial coordinate and the angu | 你问的实体映射方式对应论文中 HAKE 模型的核心机制。HAKE 将实体映射 | L0→L1→answer_L1 | 5 | pass |
| qasper_1609.00081 | f01aa192d97f | What are the baselines model? | (i) Uniform (ii) SVR+W (iii) S | 你问的 baselines 对应论文中用于与 GraLap 对比的五个基线模 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1905.13413 | ca7e71131219 | How does this compare to traditional c | No reliability diagrams are pr | 抱歉，我无法从该论文中找到关于其与传统校准方法（如Platt Scaling | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1905.13413 | d77c9ede2727 | What's the input representation of Ope | word embeddings | 你问的“OpenIE 元组的输入表示”对应论文中模型的具体输入形式。根据正文 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1903.12495 | e86d381322c8 | how well this method is compared to ot | — | 抱歉，该论文中未提供该方法与其他方法的对比评估，缺少性能比较或基准测试的实质 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1904.08386 | 508580af5148 | How do they model a city description u | We experiment with three diffe | 你问的“如何用嵌入建模城市描述”对应论文中的方法：先取每个词的 token  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1904.08386 | 89d168727065 | How do they obtain human judgements? | Using crowdsourcing | 你问的“如何获取人类判断”对应论文中的“odd-one-out”任务实验。论 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1904.08386 | fc6cfac99636 | Which clustering method do they use to | We devise a simple clustering  | 你问的聚类方法对应论文中作者自行设计的一种简单聚类算法。该算法先以随机聚类分 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2002.00750 | 26c64edbc5fa | How much in-domain data is enough for  | — | 抱歉，我无法从该论文中找到直接回答此问题的内容。检索到的段落未涉及训练数据量 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2002.00750 | e06e1b103483 | How many parameters does their propose | — | 抱歉，我无法从该论文中找到答案，因为检索到的内容未提及该联合模型的具体参数量 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1604.00125 | 03b939ad7059 | What models do they compare to? | LEAD QUERY_SIM MultiMR SVR Doc | 你问的对比模型对应论文中与 AttSum 进行比较的基线系统。论文对比了以下 | L0→answer_L0 | 5 | pass |
| qasper_1608.06378 | a53683d1a064 | What approach does this work propose f | We propose a listening compreh | 你问的“this work 提出的方法”对应论文中的“proposed mo | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1608.06378 | 0fd7d12711df | What is the new task proposed in this  | listening comprehension task | 你问的“新任务”对应论文中提出的“机器理解口语内容并回答相关问题”这一任务。 | L0→answer_L0 | 4 | pass |
| qasper_1804.05868 | df2839dbd68e | How big is the provided treebank? | 1448 sentences more than the d | 你问的“provided treebank”对应论文中作者扩展并标注的 Hi | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1804.05868 | 3996438cef34 | What is LAS metric? | — | 抱歉，我无法从该论文中找到关于“LAS metric”的定义或解释；检索内容 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1704.05572 | 2fffff59e57b | What corpus was the source of the Open | domain-targeted $~$ 80K senten | 你问的语料来源对应论文中的文本语料库 S，即来自 aristo2016:co | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1704.05572 | eb95af36347e | What is the accuracy of the proposed t | 51.7 and 51.6 on 4th and 8th g | 你问的“proposed technique”对应论文中提出的系统 **Tu | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1704.05572 | cd1792929b9f | Is an entity linking process used? | no | 抱歉，我无法从该论文中找到关于是否使用实体链接（entity linking | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1704.05572 | 65d34041ffa4 | Are the OpenIE extractions all triples | no | 否。你问的“OpenIE extractions”对应论文中从 Open I | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1704.05572 | e215fa142102 | What method was used to generate the O | for each multiple-choice quest | 你问的“生成 OpenIE 抽取所用的方法”对应论文中的 **Open IE | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1704.05572 | a8545f145d5e | Can the method answer multi-hop questi | yes | 你问的“method”对应论文中提出的 TupleInf 系统。**否**， | L0→L3→answer_L3 | 5 | pass |
| qasper_1704.05572 | 417dabd43d62 | What was the textual source to which O | domain-targeted $~$ 80K senten | 你问的文本来源对应论文中的语料库 S，即来自 aristo2016:comb | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1704.05572 | fed230cef7c1 | What OpenIE method was used to generat | for each multiple-choice quest | 你问的 OpenIE 方法对应论文中的 Open IE v4。在构建 tup | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1704.05572 | 7917d44e952b | Is their method capable of multi-hop r | yes | 你问的“their method”对应论文中的 TupleInf。**否** | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1904.05527 | 43d8057ff0d3 | What do the models that they compare p | national dialects of English | 你问的“他们比较的模型预测什么”对应论文中的方言识别（dialect ide | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1704.00177 | b512ab8de268 | What kernels are used in the support v | — | 抱歉，我无法从这篇论文中找到关于支持向量机所用核函数的信息，因为检索内容中未 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1704.00177 | 4e4d377b140c | What dataset is used? | ACL Anthology Reference Corpus | 你问的数据集对应论文中的三个评测数据集：dataset-basic、data | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1704.00177 | 828ce5faed77 | What metrics are considered? | F-score micro-F macro-F weight | 你问的评估指标对应论文中用于比较分类性能的 **宏F值（macro-F）、微 | L0→answer_L0 | 4 | pass |
| qasper_1908.09137 | b236b9827253 | How much better performance of propose | significant MAP performance im | 你问的“proposed model”对应论文中的 Propagate-se | L0→L1→answer_L1 | 5 | pass |
| qasper_1908.09137 | b53efdbb9e53 | How are some nodes initially connected | we fully connect nodes that re | 你问的“节点初始连接方式”对应论文中 Propagate-selector（ | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.00439 | fac052c4ad6b | How do they evaluate their method? | Calinski-Harabasz score t-SNE  | 你问的“他们如何评估方法”对应论文中的聚类实验评估流程。他们固定所有方法的特 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.00439 | aa54e12ff71c | What is an example of a health-related | The health benefits of alcohol | 抱歉，我无法从该论文中找到具体的健康相关推文示例，因为检索到的内容仅涉及数据 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1812.03593 | 40a45d59a2ef | Is the model evaluated on other datase | no | 否。你问的“模型”对应论文中提出的 SDNet 模型，其仅在 CoQA 数据 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1812.03593 | b29b5c395754 | Does the model incorporate coreference | As the question has integrated | 你问的“模型是否融入指代（coreference）与蕴含（entailmen | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1812.03593 | 4040f5c9f365 | Is the incorporation of context separa | no | 是的，论文对上下文信息的引入进行了单独评估。在消融研究中，作者通过设置对话历 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1710.02772 | 0747cecb3c72 | What other solutions do they compare t | strong competitive methods on  | 你问的“其他对比方案”对应论文中与 Smarnet 对比的基线方法。在 SQ | L0→L1→L2→L3→answer_L3 | 3 | fail |
| qasper_1710.02772 | 0e9c08b635c1 | How does the gatint mechanism combine  | when the gate has high value,  | 你问的门控机制对应论文中的 fine-grained gating mech | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1706.04815 | c348a8c06e20 | What two components are included in th | evidence extraction and answer | 你问的框架对应论文中提出的“抽取后合成”（extraction-then-s | L0→answer_L0 | 5 | pass |
| qasper_1706.04815 | 0300cf768996 | Which framework they propose in this p | extraction-then-synthesis fram | 你问的框架对应论文中提出的 **S-Net**（抽取后合成框架，extrac | L0→answer_L0 | 5 | pass |
| qasper_1706.04815 | dd8f72cb3c09 | Why MS-MARCO is different from SQuAD? | there are several related pass | 你问的 MS-MARCO 与 SQuAD 的差异，对应论文中描述的两个数据集 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.00652 | cc9f0ac8ead5 | How big is improvement in performances | Compared with the previous SOT | 你问的“proposed model 相对 state of the art | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.00652 | 69e678666d11 | What two large datasets are used for e | SParC BIBREF2 and CoSQL BIBREF | 你问的两个用于评测的大型数据集对应论文中的 **SParC** 和 **Co | L0→answer_L0 | 5 | pass |
| qasper_2002.00652 | 471d624498ab | What context modelling methods are eva | Concat
Turn
Gate
Action Copy
T | 你问的“被评估的上下文建模方法”对应论文中在同一解析器上评测的 13 种方法 | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_2002.00652 | f858031ebe57 | What are two datasets models are teste | SParC BIBREF2 and CoSQL BIBREF | 你问的两个数据集对应论文中的 **SParC** 和 **CoSQL**，均 | L0→answer_L0 | 5 | pass |
| qasper_1907.04380 | 2fd8688c8f47 | Is such bias caused by bad annotation? | no | 否。你问的“bias 是否由糟糕的标注导致”，对应论文中讨论的“annota | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.00137 | 637aa32a34b2 | Do they report results only on English | yes | 你问的“是否只报告英文数据集结果”对应论文中的两个评测任务：Twitter  | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2001.00137 | 4b8257cdd9a6 | How do the authors define or exemplify | typos in spellings or ungramma | 你问的“incorrect words”对应论文中“不完整数据（incomp | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2001.00137 | 7e161d9facd1 | How many vanilla transformers do they  | — | 你问的“embedding layer 之后使用的 vanilla tran | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2001.00137 | abc5836c54fc | Do they test their approach on a datas | no | 是的，论文在无缺失数据的干净数据集上测试了其方法。在 Twitter 情感分 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2001.00137 | 4debd7926941 | Should their approach be applied only  | no | 你问的“their approach”对应论文中的 Stacked DeBE | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.00137 | 3b745f086fb5 | By how much do they outperform other m | In the sentiment classificatio | 你问的“sentiment in intent classification | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1905.07471 | e7329c403af2 | How much better does this baseline neu | The model outperforms at every | 你问的“baseline neural model”对应论文中提出的隐式 O | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2001.09899 | bf25a202ac71 | What are the state of the art measures | Randomwalk Walktrap Louvain cl | 你问的“state of the art measures”对应论文中用于对 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2001.09899 | abebf9c8c9cf | What controversial topics are experime | political events such as elect | 你问的“controversial topics”对应论文中用于测试的争议性 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_2001.09899 | 2df910c9806f | What datasets did they use? | BIBREF32, BIBREF23, BIBREF33 d | 你问的数据集对应论文中用于测试争议度量方法的讨论集合。论文使用了三十个不同的 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2001.09899 | a2a3af59f3f1 | What social media platform is observed | Twitter | 你问的观察平台对应论文中的 Twitter。论文明确报告了在 Twitter | L0→answer_L0 | 5 | pass |
| qasper_2001.09899 | d92f1c15537b | How many languages do they experiment  | four different languages: Engl | 你问的语言数量对应论文中实验所覆盖的语言。论文明确说明实验考虑了四种不同语言 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1905.12260 | c33d0bc5484c | Do the images have multilingual annota | monolingual | 你问的图像标注对应论文中的查询文本语言标签。论文中每个查询的每个词都基于用户 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1905.12260 | 93b1b94b301a | Could you learn such embedding simply  | yes | 是的，可以。你问的“仅从图像标注、不使用视觉信息来学习嵌入”对应论文中的“C | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1905.12260 | e8029ec69b0b | How much important is the visual groun | performance is significantly d | 你问的“视觉基础（visual grounding）的重要性”对应论文中“利 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1810.08732 | 7f2fd7ab968d | What type and size of word embeddings  | word2vec 200 as the dimension  | 你问的词嵌入类型与维度对应论文中用于土耳其语NER的连续Skip-gram模 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1810.08732 | 369b0a481a4b | What data was used to build the word e | Turkish news-web corpus  TS Tw | 你问的词向量构建数据对应论文中的无监督阶段所用两类未标注语料 [1]：一是土 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1810.00663 | aa800b424db7 | Did the collection process use a WoZ m | no | 抱歉，我无法从该论文中找到关于数据收集是否采用Wizard-of-Oz（Wo | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1810.00663 | fbd47705262b | By how much did their model outperform | increasing accuracy by 35% and | 你问的“their model”对应论文中的“Ours with Mask  | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1810.00663 | 51aaec4c511d | What baselines did they compare their  | the baseline where path genera | 你问的基线对应论文中的 baseline 模型。该基线将行为导航指令解析分为 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1810.00663 | 3aee5c856e0e | What was the performance of their mode | For test-repeated set, EM scor | 你问的“their model”对应论文中提出的最终模型“Ours with | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1810.00663 | f42d470384ca | What evaluation metrics are used? | exact match, f1 score, edit di | 你问的评估指标对应论文中的定量评估部分。使用的指标包括：精确匹配（Exact | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1904.06941 | 405964517f37 | How is performance measured? | they use ROC curves and cross- | 你问的性能度量方式对应论文中的两种指标：对于二分类问题使用 ROC 曲线及  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1908.07218 | 3b995a7358ce | What types of word representations are | GloVE; SGNS | 你问的“评估的词表示类型”对应论文中通过 retrofitting 注入 E | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_2004.04696 | 3f5f74c39a56 | How are the synthetic examples generat | Random perturbation of Wikiped | 你问的“synthetic examples”的生成方式对应论文中的三种数据 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1907.00758 | 73d657d6faed | Do they compare their neural network a | no | 否，论文没有将 UltraSync 与任何其他模型进行对比实验。文中仅提及  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.00758 | 9ef182b61461 | Do they annotate their own dataset or  | Use an existing one | 你问的数据集来源对应论文中的 UltraSuite 语料库。论文明确说明使用 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1907.00758 | f6f8054f327a | Does their neural network predict a si | yes | 你问的“他们的神经网络是否预测录音中的单一偏移量”对应论文中的 UltraS | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.00758 | b8f711179a46 | What kind of neural network architectu | CNN | 你问的神经网络架构对应论文中的 **UltraSync 模型**，它采用** | L0→L1→answer_L1 | 4 | pass |
| qasper_1910.04887 | dfb0351e8fa6 | How better does auto-completion perfor | — | 你问的“同时使用语言与视觉的自动补全”对应论文中提出的方法。论文明确指出，基 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.04887 | a130aa735de3 | How big is data provided by this resea | 16k images and 740k correspond | 你问的“data provided by this research”对应论 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.04887 | 0c1663a7f775 | How they complete a user query prefix  | we replace user embeddings wit | 你问的“如何基于图像完成用户查询前缀的补全”对应论文中的 FactorCel | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1704.08424 | b686e10a7252 | How does this compare to contextual em | represent each word with an ex | 抱歉，我无法从该论文中找到答案，因为检索到的内容未涉及任何上下文嵌入方法（如 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1708.00214 | 56836afc57ca | What NLP tasks do the authors evaluate | language identification, part- | 你问的 NLP 任务对应论文中实验覆盖的四个任务。作者在四个 NLP 任务上 | L0→L1→answer_L1 | 5 | pass |
| qasper_1601.06068 | 117aa7811ed6 | Do they evaluate the quality of the pa | no | 是的，论文对释义生成模型的质量进行了评估。作者构建了一个二元分类器来过滤生成 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1601.06068 | c359ab8ebef6 | How many paraphrases are generated per | 10*n paraphrases, where n depe | 你问的“每个问题生成的释义数量”对应论文中 WebQuestions 的实体 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1601.06068 | ad362365656b | What latent variables are modeled in t | syntactic information semantic | 你问的 PCFG 中的潜在变量对应论文中的 bi-layered L-PCF | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1601.06068 | 423bb905e404 | What are the baselines? | GraphParser without paraphrase | 你问的 baselines 对应论文中的前两个系统：**original** | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2002.10832 | ed7985e73306 | What is different in BERT-gen from sta | They use a left-to-right atten | 你问的 BERT-gen 与标准 BERT 的差异，对应论文中提出的生成方法 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_2002.10832 | cd8de03eac49 | How are multimodal representations com | The image feature vectors are  | 你问的多模态表示融合方式对应论文中“将图像表示为文本序列”的处理方法。具体而 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1708.04120 | 094ce2f912aa | What is the source of the tables? | The Online Retail Data Set con | 抱歉，我无法从该论文中找到表格数据的来源信息。检索到的内容未提及任何数据集名 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1906.01840 | e9d882775a13 | Which of their proposed attention meth | attention parsing | 你问的“提出的注意力方法中哪个整体表现更好”，对应论文中的 GANE-OT  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1906.01840 | 6367877c05be | Which dataset of texts do they use? | Cora Hepth Zhihu | 你问的“文本数据集”对应论文中的三个基准数据集：Cora（带文本信息的论文引 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1906.01840 | d151327c93b6 | Do they measure how well they perform  | yes | 抱歉，我无法从该论文中找到直接回答这一问题的内容。检索到的段落未涉及对更长序 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1906.01840 | 70f9358dc01f | Which other embeddings do they compare | MMB DeepWalk LINE  Node2vec TA | 你问的“其他嵌入方法”对应论文中用于对比的基线模型。论文将基线分为两类：拓扑 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.02262 | 6ead576ee581 | How do they measure the quality of sum | Rouge-L Bleu-1 | 你问的“如何衡量摘要质量”对应论文中在 MS MARCO 2.1 任务上的评 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1901.02262 | 0117aa1266a3 | Does their model also take the expecte | yes | 是的，你问的“模型是否将期望的答案风格作为输入”对应论文中 Masque 模 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.02262 | 5455b3cdcf42 | What do they mean by answer styles? | well-formed sentences vs conci | 你问的“answer styles”对应论文中提出的两种互补的答案风格：一是 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.02262 | 6c80bc3ed6df | Is there exactly one "answer style" pe | yes | 否。你问的“每个数据集是否恰好对应一种答案风格”，对应论文中的 MS MAR | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.02262 | 2d274c93901c | What are the baselines that Masque is  | BiDAF, Deep Cascade QA, S-Net+ | 抱歉，我无法从该论文中找到关于Masque对比基线的具体信息；检索到的内容未 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1901.02262 | e63bde5c7b15 | What is the performance achieved on Na | Bleu-1: 54.11, Bleu-4: 30.43,  | 抱歉，我无法从该论文中找到关于NarrativeQA性能的信息，因为检索到的 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1901.02262 | cb8a6f5c2971 | What is an "answer style"? | well-formed sentences vs conci | 你问的“answer style”对应论文中提出的两种互补的答案风格：一种是 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1603.04513 | 67a28fe78f07 | Where is MVCNN pertained? | on the unlabeled data of each  | 你问的“MVCNN 在哪里预训练”对应论文中的 **pretraining  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1603.04513 | d8de12f5eff6 | How much gain does the model achieve w | 0.8 points on Binary; 0.7 poin | 你问的“pretraining 带来的增益”对应论文中“tricks”模块对 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1603.04513 | 9cba2ee1f8e1 | What are the effects of extracting fea | The system benefits from filte | 你问的“提取多粒度短语特征的效果”对应论文中MVCNN采用可变大小卷积滤波器 | L0→answer_L0 | 4 | pass |
| qasper_1603.04513 | 7975c3e1f613 | What are the effects of diverse versio | each embedding version is cruc | 你问的“多种预训练词嵌入版本的效果”对应论文中“versions”实验块（行 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1603.04513 | eddb18109495 | How is MVCNN compared to CNN? | MVCNN, a novel CNN architectur | 你问的 MVCNN 与 CNN 的对比，对应论文中 MVCNN 与采用固定尺 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.03484 | 9508e9ec675b | Do they use attention? | yes | 是，论文使用了注意力机制。文中明确指出，注意力分数基于点积注意力机制计算 [ | L0→L1→answer_L1 | 5 | pass |
| qasper_1910.03484 | a65e5c97ade6 | What non-annotated datasets are consid | E2E NLG challenge Dataset The  | 你问的非标注数据集对应论文中的 Wikipedia Company Data | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1909.12231 | 1a7d28c25bb7 | How big is dataset domain-specific emb | — | 抱歉，我无法从该论文中找到答案，因为检索内容未提及领域特定嵌入训练所用数据集 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.12231 | 6bc45d4f9086 | How big is unrelated corpus universal  | — | 抱歉，论文中未提供该无关语料库的具体规模（如词数或句子数），仅提及基于英文维 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.12231 | 48cc41c372d4 | How better are state-of-the-art result | we achieve better results than | 你问的“state-of-the-art 结果”对应论文中与 SemSent | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.02480 | b14f13f2a3a3 | Does this model train faster than stat | — | 抱歉，论文中未提供该模型与state-of-the-art模型在训练速度上的 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.02480 | ba6422e22297 | What is the performance difference bet | Difference is around 1 BLEU sc | 你问的“proposed method”对应论文中的 FlowSeq，“st | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.02480 | 65e72ad72a9c | What non autoregressive NMT models are | NAT w/ Fertility NAT-IR NAT-RE | 你问的对比非自回归NMT模型对应论文[3]中列出的基线模型，包括：NAT w | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.02480 | cf8edc6e8c4d | What are three neural machine translat | WMT2014, WMT2016 and IWSLT-201 | 你问的三个评测数据集对应论文中的 WMT2014、WMT2016 和 IWS | L0→answer_L0 | 5 | pass |
| qasper_1808.03894 | aceac4ad16ff | Did they use the state-of-the-art mode | we provide an extensive analys | 是的，论文使用了当时最先进的模型来分析注意力。你问的“state-of-th | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1808.03894 | f7070b2e258b | What is the performance of their model | — | 抱歉，我无法从该论文中找到关于模型性能（如准确率等具体指标）的答案，因为检索 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1808.03894 | 2efdcebebeb9 | How many layers are there in their mod | two LSTM layers | 你问的“their model”对应论文中分析的 ESIM 模型。ESIM  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1808.03894 | 4fa851d91388 | Did they compare with gradient-based m | — | 否。论文提出的解释策略是**注意力显著性（attention salienc | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1612.04675 | ba1da61db264 | What does recurrent deep stacking netw | Stacks and joins outputs of pr | 你问的“recurrent deep stacking network”对应 | L0→answer_L0 | 2 | fail |
| qasper_1704.00939 | 6dfad97356b6 | How do they incorporate lexicon into t | concatenation of GloVe pre-tra | 你问的“如何将情感词典融入神经网络”对应论文中的词表示构建与网络输入设计。具 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1704.00939 | 59a5959a6abf | What is the source of their lexicon? | DepecheMood | 你问的“lexicon”对应论文中的 **DepecheMood** 情感词 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1704.00939 | e2e31ab279d3 | What was their performance? | beneficial impact of word-repr | 你问的“performance”对应论文中 Fortia-FBK 系统在 S | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1704.00939 | 6407dae0c095 | How long is the dataset used for train | — | 你问的训练数据集长度对应论文中的训练集，其包含 **1142** 条带标注的 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1704.00939 | 5a2f7e27efde | What embeddings do they use? | GloVe | 你问的嵌入对应论文中的词表示方法，由两部分拼接而成：GloVe 预训练词嵌入 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1611.04642 | b13d0e463d5e | What datasets are used to evaluate the | WN18 and FB15k | 抱歉，我无法从该论文中找到答案，因为检索到的内容仅包含致谢和推理步骤的标题， | L0→L1→L3→answer_unknown | 1 | unknown_ok |
| qasper_1908.10422 | 702e2d02c25a | How do they obtain human generated pol | derive rewards from human-huma | 你问的“如何获得人类生成的策略”对应论文中通过强化学习从人类对话中学习对话策 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.10422 | a83a351539fb | How many agents do they ensemble over? | 100 | 你问的“集成多少个智能体”对应论文中的 ChatDQN 智能体集成。证据显示 | L0→answer_L0 | 4 | pass |
| qasper_1811.02076 | 4bae74eb707e | Will these findings be robust through  | yes | 你问的“这些发现是否在不同数据集和不同问答算法上稳健”，对应论文中在 **S | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1811.02076 | c30c3e0f8450 | What is the underlying question answer | The system extends BiDAF BIBRE | 你问的“底层问答算法”对应论文中提出的 **Posterior Distil | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1811.02076 | 21656039994c | What datasets have this method been ev | document-level variants of the | 你问的方法在论文中对应的评估数据集是 **document-level va | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1707.06939 | 84d36bca0678 | What was the task given to workers? | conceptualization task | 你问的“workers 的任务”对应论文中的“conceptualizati | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1707.06939 | 7af01e2580c3 | How was lexical diversity measured? | By computing number of unique  | 你问的“词汇多样性如何测量”对应论文中的三项分析之一。词汇多样性通过计算每个 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1707.06939 | c78f18606524 | How many responses did they obtain? | 1001 | 你问的“responses”对应论文中工人完成的任务数。共获得 1001 个 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1707.06939 | 0cf6d52d7eaf | What crowdsourcing platform was used? | AMT | 你问的平台对应论文中的 AMT（Amazon Mechanical Turk | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1707.00110 | 2d3bf170c164 | Which baseline methods are used? | standard parametrized attentio | 你问的“baseline methods”对应论文中用于对比的基线模型。论文 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1707.00110 | 6e8c587b6562 | How much is the BLEU score? | Ranges from 44.22 to 100.00 de | 抱歉，我无法从该论文中找到具体的BLEU分数数值，因为文中仅涉及计算方法或相 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1707.00110 | ab9453fa2b92 | Which datasets are used in experiments | Sequence Copy Task and WMT'17 | 你问的实验数据集对应论文中的两类：玩具复制实验数据集和 WMT'17 机器翻 | L0→L1→answer_L1 | 4 | pass |
| qasper_1811.00266 | 013a8525dbf7 | Do they use pretrained word embeddings | yes | 是的，论文使用了预训练词嵌入。你问的“预训练词嵌入”对应论文中生成器所利用的 | L0→L1→answer_L1 | 5 | pass |
| qasper_1702.06378 | aecb485ea7d5 | Can SCRF be used to pretrain the model | no | 抱歉，根据现有论文内容，无法回答该问题。文中仅提及使用CTC预训练RNN编码 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2001.09332 | 9447ec36e397 | What is the dataset used as input to t | Italian Wikipedia and Google N | 你问的“Word2Vec 算法的输入数据集”对应论文中训练 W2V 所用的数 | L0→L1→answer_L1 | 4 | pass |
| qasper_2001.09332 | 12c6ca435f4f | Are the word embeddings tested on a NL | yes | 抱歉，我无法从该论文中找到答案。检索到的内容仅涉及词向量的训练和类比测试，未 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2001.09332 | 32c149574edf | Are the word embeddings evaluated? | yes | 是的，词嵌入得到了评估。论文将第 50 个 epoch 得到的 Skip-g | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2001.09332 | 3de27c81af30 | How big is dataset used to train Word2 | $421\,829\,960$ words divided  | 你问的训练意大利语 Word2Vec 的数据集大小为 26 亿字节原始文本  | L0→L1→answer_L1 | 2 | fail |
| qasper_2001.09332 | cc680cb8f45a | How does different parameter settings  | number of epochs is an importa | 你问的不同参数设置对模型性能与语义能力的影响，对应论文中针对 Skip-gr | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.09332 | fab4ec639a0e | Are the semantic analysis findings for | — | 抱歉，我无法从该论文中找到答案。现有内容仅涉及意大利语Word2Vec模型的 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2001.09332 | 9190c56006ba | What dataset is used for training Word | extracted from a dump of the I | 你问的意大利语 Word2Vec 训练数据集对应论文中的 W2V 训练数据。 | L0→L1→answer_L1 | 5 | pass |
| qasper_2004.00139 | 4d30c2223939 | How many words are coded in the dictio | 11'248 | 你问的词典词条数对应论文中的“标准德语词条数”。该词典共包含 **11,24 | L0→L1→answer_L1 | 5 | pass |
| qasper_2004.00139 | 7b47aa6ba247 | Is the model evaluated on the grapheme | yes | 是的，该模型在grapheme-to-phoneme（g2p）任务上进行了评 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1703.03097 | fb3687ea05d3 | Do they evaluate on relation extractio | no | 该论文未涉及关系抽取（relation extraction）的评估，检索到 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1704.02686 | fb381a597324 | Do they test their word embeddings on  | yes | 是的，论文在多个下游NLP任务上测试了其词嵌入。具体包括监督语义类比任务、句 | L0→answer_L0 | 2 | fail |
| qasper_1704.02686 | a9b10e3db590 | What are the main disadvantages of the | — | 抱歉，我无法从该论文中找到关于其词嵌入方法主要缺点的明确讨论；现有内容仅涉及 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1704.02686 | 54415efa9156 | What dimensions of word embeddings do  | 300-dimensional vectors | 你问的“通过因子分解产生的词嵌入维度”对应论文中的 CP 嵌入。论文明确说明 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1704.02686 | dcd22abfc9e7 | On which dataset(s) do they compute th | 10 million sentences gathered  | 你问的“计算词嵌入所用的数据集”对应论文中的 Wikipedia 语料。论文 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1704.02686 | 05238d1fad21 | Do they measure computation time of th | yes | 抱歉，我无法从该论文中找到相关支撑内容。检索到的段落未提及任何关于因子分解的 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1607.03542 | 33957fde72f9 | What knowledge base do they use? | Freebase | 你问的知识库对应论文中的 **Freebase** [1]。论文明确指出，语 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1607.03542 | 1c4cd22d6eae | How big is their dataset? | 3 million webpages processed w | 你问的“their dataset”对应论文中用于训练和评测的数据集。训练数 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1607.03542 | 2122bd05c03d | What task do they evaluate on? | Fill-in-the-blank natural lang | 你问的评测任务对应论文中的**开放域填空问答任务**（open-domain | L0→answer_L0 | 4 | pass |
| qasper_1909.05017 | 2255c36c8c7e | What is the motivation behind the work | Such a system would benefit ed | 你问的“动机”对应论文中作者开展基于 Transformer 的自动问题生成 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.05017 | 9e391c8325b4 | Why did they choose WER as evaluation  | WER can reflect our model's ef | 你问的“选择 WER 作为评估指标的原因”对应论文中的 WER 度量方法。论 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1708.00481 | 4c1847f0f3e6 | What programming language is the tool  | JavaScript | 你问的工具对应论文中的 LUWAK。LUWAK 采用纯 JavaScript | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.00154 | c45a160d31ca | What datasets are used for evaluation? | Swissmetro dataset | 你问的“用于评估的数据集”对应论文中的 Swissmetro dataset | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.00154 | 7358a1ce2eae | How do their train their embeddings? | The embeddings are learned sev | 你问的“their embeddings”对应论文中提出的 Travel B | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1909.00154 | 1165fb0b400e | How do they model travel behavior? | The data from collected travel | 你问的“如何建模出行行为”对应论文中提出的 **Travel Behavio | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.00154 | f2c5da398e60 | How do their interpret the coefficient | The coefficients are projected | 你问的“他们如何解释系数”对应论文中关于 embeddings 编码与 du | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1810.13024 | 521a7042b630 | What is a confusion network or lattice | graph-like structures where ar | 你问的“confusion network 或 lattice”对应论文中的 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1611.02550 | d40662236eed | How do they represent input features o | a vector of frame-level acoust | 你问的输入特征表示对应论文中声学词嵌入模型的帧级输入。模型输入是每个语音帧的 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1611.02550 | 1d791713d1aa | Which dimensionality do they use for t | 1061 | 你问的嵌入维度对应论文中探索的维度范围与最终模型输出维度。论文探索了 8 到 | L0→L1→answer_L1 | 2 | fail |
| qasper_1611.02550 | 6b6360fab2ed | Which dataset do they use? | Switchboard conversational Eng | 你问的数据集对应论文中的 **Switchboard 语料库**（Switc | L0→L1→answer_L1 | 5 | pass |
| qasper_1611.02550 | b6b5f92a1d9f | By how much do they outpeform previous | Their best average precision t | 你问的“they”对应论文中提出的 RNN 声学词嵌入模型。论文指出，与在同 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1712.03547 | 9c0cf1630804 | Do the authors analyze what kinds of c | no | 抱歉，我无法从该论文中找到您所询问的内容。检索到的段落未涉及对新嵌入方法失败 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1712.03547 | a4d8fdcaa8ad | When they say "comparable performance" | Performance was comparable, wi | 你问的“comparable performance”对应的性能下降，论文中 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1712.03547 | 9ac923be6ada | How do they evaluate interpretability? | For evaluating the interpretab | 你问的可解释性评估对应论文中的方法，其评估方式如下：使用 $Coherenc | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.04686 | 05196588320d | Do the other multilingual baselines ma | — | 抱歉，我无法从该论文中找到答案。检索到的内容未提及任何其他多语言基线方法及其 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1809.04686 | e930f153c89d | How big is the impact of training data | — | 抱歉，我无法从该论文中找到关于训练数据规模对多语言编码器性能影响的具体内容， | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1809.04686 | 545ff2f76913 | What data were they used to train the  | WMT 2014 En-Fr parallel corpus | 你问的“用于训练多语言编码器的数据”对应论文中的多语言NMT系统训练数据。该 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1906.05963 | 9a5d02062fa7 | What are the common captioning metrics | the CIDEr-D BIBREF22 , SPICE B | 你问的“常见描述生成指标”对应论文中用于评测图像描述模型的指标，包括：CID | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.10644 | 50690b72dc61 | Do they report results only on English | yes | 你问的“是否只报告英语数据结果”对应论文中使用的三个数据集（SR、HATE、 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1809.10644 | 8266642303fb | Which publicly available datasets are  | BIBREF3 BIBREF4 BIBREF9 | 你问的公开数据集对应论文中用于训练和评估分类器的三个数据集，分别是：Sexi | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1809.10644 | 3685bf2409b2 | What embedding algorithm and dimension | 300 Dimensional Glove | 你问的嵌入算法与维度对应论文中的 **GloVe Common Crawl  | L0→L1→answer_L1 | 5 | pass |
| qasper_1809.10644 | 19225e460fff | What data are the embeddings trained o | Common Crawl | 你问的嵌入训练数据对应论文中的预训练词嵌入来源。论文使用的是 **300 维 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1809.10644 | f37026f518ab | how much was the parameter difference  | our model requires 100k parame | 你问的参数差异对应论文中“我们的模型”与对比方法（即“requires 25 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1809.10644 | 1231934db6ad | how many parameters did their model us | Excluding the embedding weight | 你问的“their model”对应论文中提出的方法。该模型在排除词嵌入权重 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.10644 | 81303f605da5 | which datasets were used? | Sexist/Racist (SR) data set HA | 你问的数据集对应论文中的 SR、HATE 和 HAR 三个数据集。其中 SR | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1809.10644 | a3f108f60143 | what was their system's f1 performance | Proposed model achieves 0.86,  | 你问的系统 F1 性能对应论文中提出的方法（基于 SWEM 的模型）。论文指 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1809.10644 | 118ff1d7000e | what was the baseline? | logistic regression | 你问的 baseline 对应论文中的 logistic regressio | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1912.06813 | 6ee27ab55b1f | What datasets are experimented with? | the CMU ARCTIC database BIBREF | 你问的实验数据集对应论文中的 **CMU ARCTIC 数据库** [1]。 | L0→L1→answer_L1 | 3 | fail |
| qasper_1912.06813 | bb4de896c0fa | What is the baseline model? | a RNN-based seq2seq VC model c | 你问的基线模型对应论文中的 ATTS2S，这是一个基于 RNN 的 seq2 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.07433 | 53712f0ce764 | Do they evaluate only on English datas | — | 你问的“是否只在英语数据集上评估”对应论文中的 LAXARY 模型评估。** | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2003.07433 | 0bffc3d82d02 | Do the authors mention any possible co | no | 抱歉，我无法从该论文中找到关于潜在混淆因素（confounds）的讨论。检索 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2003.07433 | bdd8368debcb | How is the intensity of the PTSD estab | Given we have four intensity,  | 你问的“PTSD强度如何确立”对应论文中Dryhootch对PTSD强度的四 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_2003.07433 | 3334f50fe179 | How is LIWC incorporated into this sys | For each user, we calculate th | 你问的 LIWC 在系统中的应用对应论文中的 PTSD 检测基线模型部分。L | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.07433 | 7081b6909cb8 | How many twitter users are surveyed us | 210 | 你问的“接受临床验证调查问卷的推特用户数量”对应论文中“分发 Dryhoot | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_2003.07433 | 1870f871a5bc | Which clinically validated survey tool | DOSPERT, BSSS and VIAS | 你问的“clinically validated survey tools” | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2003.04748 | f8f135761159 | What languages do they look at? | — | 你问的“they look at”对应论文中建模的竞争语言对象。论文研究的是 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1812.08879 | e9209ebf38c4 | What evaluation metrics are used to me | — | 抱歉，我无法从该论文中找到关于多样性评估指标的具体内容，因为文中未涉及任何用 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1812.08879 | 4319a13a6c4a | How is some information lost in the RN | the generated sentences often  | 你问的“RNN-based generation models”对应论文中的 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1711.11221 | 9d016eb3913b | Did the authors evaluate their system  | yes | 是的，作者确实评估了其系统输出的连贯性。你问的“系统输出连贯性评估”对应论文 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1711.11221 | c1c611409b56 | What evaluations did the authors use o | BLEU scores, exact matches of  | 你问的“系统评估”对应论文中的实验设置部分。作者使用了 NIST05 数据集 | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1905.01962 | f887d5b7cf2b | Do they use the cased or uncased BERT  | — | 抱歉，我无法从该论文中找到关于BERT模型使用cased还是uncased版 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1905.01962 | ace60950ccd6 | How are the two different models train | They pre-train the models usin | 你问的“两个不同模型”对应论文中的 BERT-BASE 和 BERT-LAR | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1905.01962 | 2e1660405bde | How long is the dataset? | 645, 600000 | 你问的“dataset”对应论文中的训练与验证所用的 645 篇人工标注文章 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1809.08652 | 790ed4458a23 | Do all the instances contain code-swit | no | 你问的“所有实例是否都包含语码转换”对应论文中用于训练和评测的 Hingli | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1809.08652 | 562a2feeba95 | What embeddings do they use? | Glove Twitter word2vec | 你问的嵌入方式对应论文中训练词向量时尝试的两种嵌入类型。论文使用了 **Gl | L0→L1→answer_L1 | 5 | pass |
| qasper_1809.08652 | d4e78d3205e9 | Do they perform some annotation? | no | 是的，论文中进行了标注工作。HEOT 数据集中的推文被标注为三个类别：off | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1809.08652 | 8b51db1f7a62 | Do they use dropout? | yes | 是，模型使用了 dropout。LSTM 层设置了 dropout 值为 0 | L0→L1→answer_L1 | 5 | pass |
| qasper_1809.08652 | f37128d3533d | What definition of hate speech do they | — | 抱歉，我无法从该论文中找到对“仇恨言论”的具体定义。检索到的内容仅将其分类为 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1706.03610 | f704d182c9e0 | Among various transfer learning techni | — | 你问的“最佳迁移学习技术”对应论文中的**纯微调（fine-tuning）结 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.02562 | df25dd9004a3 | Does model uses pretrained Transformer | no | 否。你问的“模型是否使用预训练 Transformer 编码器”对应论文中提 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.02562 | 5328cc2588b2 | What was previous state of the art mod | LSTM-based RNN-T | 你问的“previous state of the art model”对应 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2002.02562 | 2ebd7a59baad | What was previous state of the art acc | — | 你问的“LibriSpeech 基准上的先前 SOTA 准确率”对应论文中  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.02562 | 766e2e35968e | How big is LibriSpeech dataset? | 970 hours of audio data with c | 你问的 LibriSpeech 数据集规模对应论文中的评估语料库。该数据集包 | L0→L1→answer_L1 | 4 | pass |
| qasper_1901.01911 | 04a4b0c6c8bd | Is this an English-language dataset? | yes | 是的，该数据集是英语数据集。论文明确指出任务数据为“English twee | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1901.01911 | dbfce07613e6 | What affective-based features are used | affective features provided by | 你问的情感特征对应论文中使用的五种情感资源，具体如下：Emolex（含14, | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.01911 | b7e419d2c4e2 | What conversation-based features are u | Text Similarity to Source Twee | 你问的对话特征对应论文中的“Conversation Based Featu | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.03385 | 9213159f874b | Is this style generator compared to so | yes | 是的，你问的“style generator”对应论文中的 StyleEQ  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1911.03385 | 5f4e6ce4a811 | How they perform manual evaluation, wh | accuracy | 你问的人工评估方式对应论文中的流畅度评估（Fluency Evaluatio | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1911.03385 | a234bcbf2e41 | What metrics are used for automatic ev | classification accuracy BLEU s | 你问的自动评估指标对应论文中的 BLEU 分数、模型困惑度（perplexi | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.03385 | c383fa9170ae | How they know what are content words? | words found in the control wor | 你问的“如何确定内容词”对应论文中的处理方式：内容词是**未被控制词表收录、 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.03385 | 83251fd4a641 | How they model style as a suite of low | style of a sentence is represe | 你问的“如何将风格建模为一套低层语言控制”对应论文中的 **StyleEQ  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1707.07922 | 182c7919329b | How does the model recognize entities  | gating function Dynamic Memory | 你问的“模型在推理时如何识别实体及其与答案的关系”对应论文中的 **QDRE | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1809.09795 | e196e2ce72eb | Do they evaluate only on English? | yes | 你问的“是否只在英语上评测”对应论文中的评测数据集来源。**是**，论文仅在 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.09795 | 46570c8faaee | What are the 7 different datasets? | SemEval 2018 Task 3 BIBREF20 B | 你问的 7 个数据集对应论文中用于讽刺/反讽检测评测的七个基准数据集，具体如 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.09795 | 982d37537823 | What are the three different sources o | Twitter Reddit Online Dialogue | 你问的“三个不同数据来源”对应论文中的 Twitter、Reddit 和 O | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.09795 | bbdb2942dc6d | What type of model are the ELMo repres | A bi-LSTM with max-pooling on  | 你问的 ELMo 表示所用于的模型，对应论文中提出的纯字符级架构（purel | L0→L1→answer_L1 | 5 | pass |
| qasper_1809.09795 | 4ec538e11435 | Which morphosyntactic features are tho | all caps quotation marks emoti | 你问的“指示反讽或讽刺的形态句法特征”对应论文中列举的词汇与形态句法线索。论 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1611.00514 | 21a96b328b43 | Do they single out a validation set fr | no | 抱歉，我无法从该论文中找到直接支撑这一问题的内容。现有资料仅涉及训练集组成和 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1611.00514 | 30803eefd7cd | How well does their system perform on  | EER 16.04, Cmindet 0.6012, Cde | 你问的系统在 SRE 2016 development set 上的表现，对 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1611.00514 | 442f8da2c988 | Which are the novel languages on which | Cebuano and Mandarin Tagalog a | 你问的“SRE 重点强调的新语言”对应论文中的 2016 NIST SRE  | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1910.07181 | d6e2b276390b | What models other than standalone BERT | Only Bert base and Bert large  | 你问的“新模型”对应论文中的 Bertram。除独立 BERT 外，Bert | L0→answer_L0 | 2 | fail |
| qasper_1910.07181 | 32537fdf0d4f | How much is representaton improved for | improving the score for WNLaMP | 你问的“rare/medium 频率词上的表示提升幅度”对应论文中 WNLa | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1910.07181 | ef081d78be17 | What are three downstream task dataset | MNLI BIBREF21 AG's News BIBREF | 你问的三个下游任务数据集对应论文中的 MNLI、AG's News 和 DB | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.07181 | 537b2d779912 | What is dataset for word probing task? | WNLaMPro dataset | 你问的“word probing task”数据集对应论文中的 **WNLa | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.08051 | 2159062595f2 | Are any of the utterances ungrammatica | yes | 是的，语料中包含不合语法的语句。你问的“utterances”对应论文中的书 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.08051 | 9ebb2adf92a0 | How is the proficiency score calculate | They used 6 indicators for pro | 你问的“proficiency score”对应论文中由人类专家给出的总分。 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.08051 | 973f62846646 | What proficiency indicators are used t | 6 indicators:
- lexical richne | 你问的“用于给话语评分的熟练度指标”对应论文中专家用来给书面句子和口语话语打 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2001.08051 | 0a3a8d1b0cba | What accuracy is achieved by the speec | Accuracy not available: WER re | 抱歉，我无法从该论文中找到关于语音识别系统具体准确率（如词错误率或准确率数值 | L0→L1→L2→L3→answer_unknown | 2 | unknown_ok |
| qasper_2001.08051 | ec2b8c43f142 | How is the speech recognition system e | Speech recognition system is e | 你问的“语音识别系统评估”对应论文中基于强自适应 ASR 在评测数据集上进行 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2001.08051 | 5e5460ea955d | How many of the utterances are transcr | Total number of transcribed ut | 你问的“有多少话语被转写”对应论文中的手动转写数据量。论文指出，5 位研究人 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_2001.08051 | d7d611f62255 | How many utterances are in the corpus? | Total number of utterances ava | 抱歉，我无法从该论文中找到语料库中话语总数的信息，因为检索到的内容未提供任何 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1911.02086 | 7f5ab9a53aef | Do they compare executionttime of thei | no | 否。论文中未对模型执行时间（execution time）与其他模型进行对比 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.02086 | 7fbbe191f4d8 | What is the memory footprint decrease  | — | 抱歉，我无法从该论文中找到答案，因为检索内容未提供其模型与其他模型的内存占用 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1702.06589 | 07c9863e1e86 | Does a neural scoring function take bo | yes | 你问的“神经评分函数是否同时以问题和逻辑形式为输入”，对应论文中的“联合嵌入 | L0→L1→answer_L1 | 5 | pass |
| qasper_1702.06589 | bf7cb53f4105 | What is the source of the paraphrases  | WikiTableQuestions | 你问的“paraphrases of the questions”对应论文中 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1702.06589 | a6419207d229 | Does the dataset they use differ from  | no | 你问的数据集对应论文中的 WikiTableQuestions 数据集。论文 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1612.06897 | 2a46db1b91de | How many examples do they have in the  | Around 388k examples, 194k fro | 你问的“target domain”对应论文中的“in-domain”数据。 | L0→L1→L3→answer_L3 | 2 | fail |
| qasper_1705.01991 | 4e2e19a58e1f | Do they only test on one dataset? | yes | 你问的“是否只在一个数据集上测试”对应论文中的评测数据设置。是的，论文只在  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1705.01991 | 69ca609e8688 | What baseline decoder do they use? | a standard beam search decoder | 你问的基线解码器对应论文中的“标准束搜索解码器”。论文明确说明基线解码器采用 | L0→L1→answer_L1 | 5 | pass |
| qasper_1911.03562 | c95fd189985d | Which NLP area have the highest averag | sentiment analysis, informatio | 你问的“女性作者平均引用最高的 NLP 领域”对应论文中“女性第一作者平均引 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1911.03562 | 4a61260d6edf | Which 3 NLP areas are cited the most? | machine translation, statistic | 你问的“被引用最多的 3 个 NLP 领域”对应论文中按标题 bigram  | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1911.03562 | 5c95808cd3ee | Which journal and conference are cited | CL Journal and EMNLP conferenc | 你问的“近年被引最多的期刊和会议”对应论文中的 CL journal（期刊） | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1911.03562 | b6f5860fc4a9 | Which 5 languages appear most frequent | English, Chinese, French, Japa | 你问的“AA 论文标题中出现频率最高的 5 种语言”对应论文中基于 122  | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1911.03562 | 7955dbd79ded | What aspect of NLP research is examine | size, demographics, areas of r | 你问的“NLP 研究的方面”对应论文中 NLP Scholar 项目所考察的 | L0→answer_L0 | 2 | fail |
| qasper_1911.03562 | 6bff681f1f67 | Are the academically younger authors c | yes | 你问的“学术年龄更小的作者是否被引更少”对应论文中“第一作者学术年龄与被引量 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1911.03562 | 205163715f34 | How many papers are used in experiment | 44,896 articles | 抱歉，我无法从该论文中找到关于实验所用论文数量的信息；现有内容仅涉及引用统计 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1811.04604 | 43878a6a8fc3 | What datasets did they use? | the personalized bAbI dialog d | 你问的数据集对应论文中的 **personalized bAbI dialo | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.06083 | 3f326c003be2 | What evaluation metric is used? | F1 and Weighted-F1 | 你问的评估指标对应论文中用于报告结果的指标。论文采用**加权F1**（按类别 | L0→L1→answer_L1 | 4 | pass |
| qasper_1908.06083 | c84590ba32df | What datasets are used? | The Wikipedia Toxic Comments d | 你问的数据集对应论文中的 **Wikipedia Toxic Comment | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.00578 | ff28d34d1aaa | What are their correlation results? | High correlation results range | 你问的“their correlation results”对应论文中 Su | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_1909.00578 | ae8354e67978 | What dataset do they use? | datasets from the NIST DUC-05, | 你问的数据集对应论文中的 NIST DUC-05、DUC-06 和 DUC- | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.00578 | 02348ab62957 | What simpler models do they look at? | BiGRU s with attention ROUGE L | 你问的“simpler models”对应论文中与 Sum-QE 对比的基线 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.00578 | 3748787379b3 | What linguistic quality aspects are ad | Grammaticality, non-redundancy | 你问的语言学质量方面对应论文中的五个语言学质量标准（$\mathcal{Q} | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1811.08048 | ce14b87dacfd | How does the QuaSP+Zero model work? | does not just consider the que | 你问的 QuaSP+Zero 对应论文中提出的零样本解析模型。其工作机制是： | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1811.08048 | 709a49939271 | Which off-the-shelf tools do they use  | information retrieval system w | 抱歉，我无法从该论文中找到关于QuaRel使用哪些现成工具（off-the- | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1811.08048 | a3c6acf90012 | How do they obtain the logical forms o | workers were given a seed qual | 你问的逻辑形式获取方式对应论文中 QuaRel 数据集的构建过程。他们采用一 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1811.08048 | 31b631a8634f | Do all questions in the dataset allow  | yes | 否。你问的“数据集中所有问题是否都提供 2 个选项”对应论文中的 QuaRe | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1709.05563 | 57fdb0f6cd91 | What are the potentials risks of this  | — | 抱歉，我无法从该论文中找到关于此方法潜在风险的任何内容，因为检索到的段落仅概 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1709.05563 | 3aa43a0d543b | What elements of natural language proc | translated the responses in mu | 你问的“用于分析定性数据的自然语言处理要素”对应论文中介绍的多种 NLP 方 | L0→L1→L2→L3→answer_L3 | 3 | fail |
| qasper_2003.09586 | e75f5bd7cc71 | How much is decoding speed increased b | the Transformer with 10 encode | 你问的“增加编码器层数、减少解码器层数带来的解码速度提升”对应论文中“10  | L0→answer_L0 | 5 | pass |
| qasper_1707.08559 | 762f2527f85c | What was the baseline? | — | 抱歉，我无法从该论文中找到关于“baseline”的具体定义或设置，因为检索 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1707.08559 | e414d819f10c | What is the average length of the reco | 40 minutes | 你问的“recordings”对应论文中的“each game's vide | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1707.08559 | 2e73006e5d00 | How big was the dataset presented? | 321 videos | 你问的数据集规模对应论文中的视频数据集。该数据集共包含 321 个视频，其中 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1707.08559 | 1a8b7d3d1269 | What were their results? | Best model achieved F-score 74 | 你问的“their results”对应论文中提出的多模态高光预测模型的实验 | L0→answer_L0 | 2 | fail |
| qasper_1802.05322 | a7adb63db506 | what was the baseline? | There is no baseline. | 抱歉，我无法从该论文中找到关于“基线”的具体内容，因为检索到的段落中未提及任 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1802.05322 | 980568848cc8 | how many movie genres do they explore? | 27 | 你问的“they explore”的电影类型数量，对应论文中从 IMDb 抓 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1802.05322 | f1b738a7f118 | what evaluation metrics are discussed? | precision  recall  Hamming los | 你问的评估指标对应论文中用于多标签分类器评测的指标，共讨论了以下四种：准确率 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.03023 | 65ebed1971dc | what experiments are conducted? | a reliability study for the pr | 你问的“实验”对应论文中的信度与效度研究。论文开展了标注方案的信度实验：由两 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.03023 | b24b56ccc5d4 | what opportunities are highlighted? | Our annotation scheme introduc | 你问的“opportunities”对应论文中提出的标注方案所带来的两类机会 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.03023 | 3bfdbf2d4d68 | how do they measure discussion quality | Measuring three aspects: argum | 你问的“讨论质量”对应论文中与英语教育专家评价对比的效度检验。他们通过将学生 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.03023 | 9378b41f7e88 | do they use a crowdsourcing platform? | no | 否，论文未提及使用众包平台。文中标注工作由专家标注员完成，如“two pai | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.09886 | 735f58e28d84 | Are there datasets with relation tuple | yes | 你问的数据集对应论文中的 **NYT（New York Times）语料库* | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1911.09886 | 710fa8b3e74e | Which one of two proposed approaches p | WordDecoding (WDec) model | 你问的两个方法中表现更好的，对应论文中的 **WDec（WordDecodi | L0→L1→answer_L1 | 5 | pass |
| qasper_1911.09886 | 56123dd42cf5 | What is previous work authors reffer t | SPTree Tagging CopyR HRL Graph | 你问的“previous work”对应论文中作者回顾并对比的**已有联合实 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.09886 | 1898f999626f | How higher are F1 scores compared to p | WordDecoding (WDec) model achi | 你问的“与先前工作的 F1 对比”对应论文中与 HRL 基线的比较。在 NY | L0→answer_L0 | 5 | pass |
| qasper_1907.01413 | 1fdcc650c65c | Do they report results only on English | — | 否。论文使用的数据集是 Ultrax Typically Developin | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1907.01413 | abad9beb7295 | Do they propose any further additions  | yes | 是的，论文在“Future Work”部分提出了若干可提升对未见说话者泛化能 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.01413 | 265c9b733e4d | What are the characteristics of the da | synchronized acoustic and ultr | 你问的数据集特征对应论文中的 UXTD（Ultrax Typically D | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1907.01413 | 0f928732f226 | What type of models are used for class | feedforward neural networks (D | 你问的分类模型对应论文中评估的两类分类器：第一类是**前馈神经网络（DNNs | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1907.01413 | 11c5b12e675c | Do they compare to previous work? | no | 是的，论文与先前工作进行了比较。文中指出，先前关于超声数据语音识别的研究大多 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1907.01413 | d24acc567eba | How many instances does their dataset  | 10700 | 你问的数据集实例数量对应论文中的训练示例总数。该数据集共有 **10700  | L0→L1→answer_L1 | 5 | pass |
| qasper_1907.01413 | 2d62a75af409 | What model do they use to classify pho | feedforward neural networks co | 你问的分类模型对应论文中评估的两类分类器：**前馈神经网络（DNN）** 和 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.01413 | fffbd6cafef9 | How many speakers do they have in the  | 58 | 你问的数据集中的说话人数量对应论文中的 UXTD 数据集，共包含 **58  | L0→L1→answer_L1 | 5 | pass |
| qasper_1903.09588 | e9d9bb87a5c4 | How much do they outperform previous s | On subtask 3 best proposed mod | 你问的“they”对应论文中的 BERT-pair 模型。在 SentiHo | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1903.09588 | 3554ac92d4f2 | How do they generate the auxiliary sen | The sentence we want to genera | 你问的“如何生成辅助句”对应论文中提出的四种将TABSA转换为句子对分类任务 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.00153 | 936878cff0e6 | Do they analyze what type of content A | no | 你问的“阿拉伯机器人传播的内容类型是否与英文机器人对比分析”对应论文中的主题 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.00153 | 58c1b162a449 | Do they propose a new model to better  | yes | 是的，论文提出了一种新的回归模型，专门用于检测阿拉伯语推特机器人 [1]。该 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1901.10826 | b1457feb6cdb | Do they compare computational time of  | no | 抱歉，我无法从该论文中找到关于AM-softmax与Softmax计算时间对 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1901.10826 | 46bca122a872 | Do they visualize the difference betwe | yes | 是的，论文中确实对 AM-Softmax 与传统 Softmax 的差异进行 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1912.09152 | 601f96770726 | What does their system consist of? | rule-based and dictionary-base | 你问的系统构成对应论文中提出的资源型NER系统，它由两大核心资源组成：一是S | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1912.09152 | 1c68d18b4b65 | What are the two PharmaCoNER subtasks? | Entity identification with off | 你问的两个 PharmaCoNER 子任务对应论文中的 subtask 1  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.00547 | acd05f31e258 | What is the size of the dataset? | 9,892 stories of sexual harass | 你问的数据集规模对应论文中标注的 Safecity 个人故事数据集。该数据集 | L0→answer_L0 | 4 | pass |
| qasper_1911.00547 | 8c78b21ec966 | What model did they use? | joint learning NLP models that | 你问的“模型”对应论文中提出的联合关键元素提取与分类模型，共包含三种变体：J | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1911.00547 | af60462881b2 | What patterns were discovered from the | we demonstrate that harassment | 你问的“从故事中发现的模式”对应论文中通过分类与关键要素提取所揭示的性骚扰发 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1911.00547 | 879bec20c0fd | Did they use a crowdsourcing platform? | — | 抱歉，我无法从该论文中找到关于是否使用众包平台的明确信息；检索内容中未涉及任 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1901.03859 | 10f560fe8e1c | How does nextsum work? | selects the next summary sente | 你问的 NextSum 的工作方式对应论文中的“下一句预测”框架。NextS | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.08041 | 13d92cbc2c77 | What baseline approaches do they compa | HotspotQA: Yang, Ding, Muppet
 | 你问的“baseline approaches”对应论文中对比的“previ | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.08041 | 9df4a7bd0abb | How do they train the retrieval module | We treated the neural semantic | 你问的检索模块训练对应论文中的语义检索（semantic retrieval | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.08041 | b7291845ccf0 | How do they model the neural retrieval | BERT-Base BIBREF2 to provide t | 你问的神经检索模块对应论文中的“Semantic Retrieval”部分。 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.08041 | ac54a9c30c96 | Retrieval at what level performs bette | This seems to indicate that th | 你问的“哪个层级检索表现更好”对应论文中的消融实验结论，但论文并未直接比较两 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.02030 | e854edcc5e91 | Do the authors report results only on  | — | 否。论文未明确限定或声明其数据仅涉及英语，也未报告任何非英语数据的结果。文中 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1907.02030 | bd6cec2ab620 | How is the accuracy of the system meas | F1 score of 0.71 for this task | 你问的“系统精度如何衡量”对应论文中的两个测试：Quora 重复问题数据集测 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1907.02030 | 4b0ba460ae3b | How is an incoming claim used to retri | text clustering on the embeddi | 抱歉，该论文内容未涉及“如何用新claim检索已核查claim”的具体机制或 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1907.02030 | 63b0c93f0452 | What existing corpus is used for compa | Quora duplicate question datas | 你问的“用于比较的现有语料库”对应论文中的 **Quora duplicat | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.02030 | d27f23bcd80b | What are the components in the factche | — | 你问的“factchecking 算法”对应论文中提出的自动化事实核查流程， | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1711.00106 | 1f63ccc379f0 | How much is the gap between using the  | The mixed objective improves E | 抱歉，我无法从该论文中找到关于“使用所提目标与仅使用交叉熵目标之间性能差距” | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1807.00868 | d20d6c8ecd7c | What normalization techniques are ment | FBanks with cepstral mean norm | 你问的归一化技术对应论文中探索的语音特征归一化方法，共提及两种：cepstr | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1807.00868 | 11e6b79f1f48 | What features do they experiment with? | 40 mel-scaled log filterbank e | 你问的特征实验对应论文中的输入特征与归一化方式。他们主要使用 40 维 me | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1807.00868 | 2677b88c2def | Which architecture is their best model | 6-layer bLSTM with 1024 hidden | 你问的最佳模型对应论文中的6层bLSTM架构。该模型采用6层双向LSTM、1 | L0→L1→answer_L1 | 5 | pass |
| qasper_1807.00868 | 8ca31caa34cc | What kind of spontaneous speech is use | — | 你问的“spontaneous speech”对应论文中的“conversa | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.05752 | c82e945b43b2 | What additional features are proposed  | distinguishing between clinica | 你问的未来工作拟新增特征，对应论文中计划在现有特征集（段落领域得分、词袋、住 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1809.05752 | fbee81a9d90f | What are their initial results on this | Achieved the highest per-domai | 你问的“initial results”对应论文中模型在风险因子域分类任务上 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1809.05752 | 39cf0b3974e8 | What datasets did the authors use? | a corpus of discharge summarie | 你问的数据集对应论文中的目标数据集与训练数据集两类。作者使用的目标数据集来自 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1803.09745 | b5bfa6effdea | Which regions of the United States do  | all regions except those that  | 你问的美国区域对应论文中按县级（county）划分的考察范围。论文纳入了美国 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1803.09745 | bf00808353ee | Why did they only consider six years o | — | 抱歉，我无法从该论文中找到答案，因为检索内容未说明为何只考虑六年（2003- | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1701.06538 | a85698f19a91 | Approximately how much computational c | — | 你问的“this model”对应论文中的 MoE 模型。证据显示，最快的  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1701.06538 | af073d84b8a7 | What improvement does the MOE model ma | 1.34 and 1.12 BLEU score on to | 你问的 MoE 模型在机器翻译上相对 SOTA 的提升，对应论文中的“Mac | L0→answer_L0 | 4 | pass |
| qasper_1701.06538 | e8fcfb1412c3 | What improvement does the MOE model ma | Perpexity is improved from 34. | 你问的 MoE 模型在语言建模上相对 SOTA 的提升，对应论文中的多个结果 | L0→answer_L0 | 2 | fail |
| qasper_1701.06538 | 0cd90e5b79ea | How is the correct number of experts t | varied the number of experts b | 你问的“如何决定正确的专家数量”对应论文中通过**实验对比不同专家数量并观察 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1701.06538 | f01a88e15ef5 | What equations are used for the traina | DISPLAYFORM0 DISPLAYFORM0 DISP | 你问的“trainable gating network”对应论文中的 So | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1701.08118 | 7486c9d9e6c4 | What languages are were included in th | German | 抱歉，我无法从该论文中找到关于数据集包含哪些语言的明确信息；现有内容仅提及德 | L0→L1→L2→L3→answer_unknown | 2 | unknown_ok |
| qasper_1701.08118 | 0f2403fa7773 | How was reliability measured? | level of agreement (Krippendor | 你问的“reliability（可靠性）”对应论文中标注者间一致性（inte | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1701.08118 | 21df76462c76 | How did the authors demonstrate that s | participants in group one very | 你问的“作者如何证明定义使标注者部分对齐自身观点”对应论文中的第一组（gro | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1701.08118 | 45be26c01e82 | What definition was one of the groups  | Twitter definition of hateful  | 你问的“其中一个组看到的定义”对应论文中的 Twitter 仇恨行为定义（T | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1701.08118 | e28019afcb55 | Was the degree of offensiveness taken  | Personal thought of the annota | 你问的“offensiveness 的判定标准”对应论文中调查问卷的第三个问 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1701.08118 | 551cc0401674 | How were potentially hateful messages  | 10 hashtags that can be used i | 你问的“如何识别潜在仇恨言论消息”对应论文中语料库构建的筛选流程。研究者先用 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1905.11901 | 4547818a3bbb | what amounts of size were used on germ | Training data with 159000, 800 | 你问的德英（German–English）数据规模对应论文中的 IWSLT  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1905.11901 | 07d7652ad4a0 | what were their experimental results i | 10.37 BLEU | 你问的低资源数据集实验结果对应论文中的“超低数据设置”及韩英翻译等结果。主要 | L0→answer_L0 | 4 | pass |
| qasper_1905.11901 | 9f3444c9fb2e | what are the methods they compare with | gu-EtAl:2018:EMNLP1 | 你问的“Korean–English 数据集上比较的方法”对应论文中的配置  | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1905.11901 | 2348d68e0654 | what pitfalls are mentioned in the pap | highly data-inefficient underp | 你问的“pitfalls”对应论文中指出的**先前研究在低资源NMT上的缺陷 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1910.03814 | 6976296126e4 | What models do they propose? | Feature Concatenation Model (F | 你问的“they propose”对应论文中提出的多模态检测模型。论文提出了 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.03814 | 53640834d68c | Are all tweets in English? | — | 抱歉，我无法从该论文中找到答案，因为检索内容未涉及推文的语言筛选或语言分布信 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1910.03814 | b2b0321b0aaf | How large is the dataset? | $150,000$ tweets | 你问的数据集规模对应论文中收集并标注的多模态仇恨言论数据集。该数据集共包含  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.03814 | 4e9684fd68a2 | What is the results of multimodal comp | Unimodal LSTM vs Best Multimod | 你问的多模态模型与单模态模型的对比结果，对应论文中的核心结论：**多模态模型 | L0→answer_L0 | 3 | fail |
| qasper_1910.03814 | 2e632eb5ad61 | What is author's opinion on why curren | Noisy data Complexity and dive | 你问的多模态模型为何无法超越纯文本模型，对应论文中作者给出的三点原因分析。作 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.03814 | d1ff6cba8c37 | What metrics are used to benchmark the | F-score Area Under the ROC Cur | 你问的评测指标对应论文中的 F-score、Area Under the R | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1910.03814 | 24c0f3d61706 | How is data collected, manual collecti | Twitter API | 你问的数据收集方式对应论文中的 MMHS150K 数据集构建过程。数据是通过 | L0→L1→answer_L1 | 5 | pass |
| qasper_1910.03814 | 21a9f1cddd7c | How many tweats does MMHS150k contains | $150,000$ tweets | 是的，你问的 MMHS150k 对应论文中标注的 150,000 条推文。条 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.03814 | a0ef0633d8b4 | What unimodal detection models were us | single layer LSTM with a 150-d | 你问的“unimodal detection models”对应论文中仅使用 | L0→L1→L2→L3→answer_L3 | 3 | fail |
| qasper_1910.03814 | b0799e261521 | What different models for multimodal d | Feature Concatenation Model (F | 你问的多模态检测模型对应论文中提出的三种架构，分别是：特征拼接模型（Feat | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.03814 | 4ce4db7f277a | What annotations are available in the  | No attacks to any community  r | 你问的“数据集中的标注”对应论文中的 MMHS150K 数据集标注。该数据集 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.06138 | 12391aab31c8 | Which algorithm is used in the UDS-DFK | Our transference model extends | 你问的 UDS-DFKI 系统所用算法，对应论文中的“transferenc | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.06138 | 8b43201e7e64 | Does the use of out-of-domain data imp | no | 你问的“使用域外数据是否提升性能”对应论文中“微调系统相对通用模型的性能提升 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1907.08937 | 10ddc5caf36f | Which competitive relational classific | For relation prediction they t | 你问的“competitive relational classificat | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1907.08937 | 29571867fe00 | Which tasks do they apply their method | relation prediction relation e | 你问的应用任务对应论文中第 [5] 条所述的五个实验问题。该方法被应用于以下 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1907.08937 | 1a678d081f97 | Which knowledge bases do they use? | Wikidata ReVerb FB15K TACRED | 你问的知识库对应论文中的 Wikidata [1]。论文在 Wikidata | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1907.08937 | b9f2a30f5ef6 | How do they gather human judgements fo | By assessing similarity of 360 | 你问的人类判断收集方式对应论文中的“Human Judgments”实验。他 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1907.08937 | 3513682d4ee2 | Which sampling method do they use to a | monte-carlo sequential samplin | 你问的采样方法对应论文中的“sequential sampling”（顺序采 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.05672 | bee74e96f244 | Is there a machine learning approach t | — | 抱歉，我无法从该论文中找到答案。论文中仅介绍了基于Prolog和DCG的规则 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2001.05672 | a56fbe90d5d3 | What DCGs are used? | Author's own DCG rules are def | 你问的“使用的 DCG”对应论文中用于处理主动句与被动句转换的 Defini | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2001.05672 | b1f2db88a6f8 | What else is tried to be solved other  | cases of singular/plural, subj | 你问的“除12种时态、情态动词和否定形式外还尝试解决什么”，对应论文中主动句 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_2001.05672 | cf3af2b68648 | What is used for evaluation of this ap | — | 抱歉，根据检索到的论文内容，我无法回答该问题——文中未涉及该方法的评估方式， | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_2001.05672 | 7883a52f008f | Is there information about performance | no | 否，论文中未提供这些转换方法的性能信息。全文仅描述了转换流程的实现细节（如三 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2001.05672 | cd9776d03fe4 | Are there some experiments performed i | no | 是的，论文中进行了实验。正文段落 [1] 提到该工作已通过三个文件完成，并给 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1810.01570 | 864b5c1fe8c7 | What evaluation metrics do they use? | Precision, Recall and INLINEFO | 你问的评估指标对应论文中用于评估去标识化架构性能的指标。论文采用精确率（Pr | L0→L1→answer_L1 | 5 | pass |
| qasper_1810.01570 | d469c7de5c9e | What performance is achieved? | — | 你问的“性能”对应论文中系统在护理语料与 i2b2 两个黄金标准数据集上的表 | L0→answer_L0 | 4 | pass |
| qasper_1810.01570 | 0a050658d09f | Do they use BERT? | no | 抱歉，我无法从该论文中找到关于是否使用BERT的明确信息。检索到的内容未提及 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1810.01570 | fd80a7162fde | What is their baseline? | Burckhardt et al. BIBREF22 Liu | 抱歉，我无法从该论文中找到关于“baseline”的定义、设置或对比对象的相 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1810.01570 | 4d4739682d54 | Which two datasets is the system teste | 2014 i2b2 de-identification ch | 你问的系统测试数据集对应论文中的两个黄金标准数据集：2014 i2b2去标识 | L0→L1→answer_L1 | 5 | pass |
| qasper_1606.02006 | 102a04397394 | What datasets were used? | KFTT BIBREF12 and BTEC BIBREF1 | 你问的数据集对应论文中的两个英日翻译语料：KFTT 和 ASPEC [3]。 | L0→answer_L0 | 2 | fail |
| qasper_1606.02006 | d9c26c1bfb38 | What language pairs did they experimen | English-Japanese | 你问的语言对对应论文中的英日（English-Japanese）翻译语料 [ | L0→answer_L0 | 4 | pass |
| qasper_1902.09393 | af75ad21dda2 | How much does this system outperform p | The system outperforms by 27.7 | 抱歉，我无法从该论文中找到关于系统性能或与先前工作对比的具体数据，因为检索到 | L0→L1→L3→answer_unknown | 1 | unknown_ok |
| qasper_1902.09393 | de12e059088e | What are the baseline systems that are | The system is compared to base | 抱歉，我无法从该论文中找到关于基线系统的信息，因为检索到的内容仅包含格式说明 | L0→L1→L3→answer_unknown | 1 | unknown_ok |
| qasper_1701.03051 | 6b55b558ed58 | What previously proposed methods is th | Naive Bayes SVM Maximum Entrop | 你问的“本方法对比的先前方法”对应论文中的基线模型（baseline mod | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1701.03051 | 3e3f5254b729 | How is effective word score calculated | We define the Effective Word S | 你问的“有效词分（Effective Word Score）”对应论文中的  | L0→L1→answer_L1 | 5 | pass |
| qasper_1701.03051 | 5bb96b255dab | How is tweet subjectivity measured? | — | 你问的推文主观性度量方式对应论文中的主观性过滤步骤。论文使用 TextBlo | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1601.00901 | 74b0d3ee0cc9 | How did they induce the CFG? | the parser first learns to par | 你问的“如何归纳 CFG”对应论文中的半自动引导式语法归纳流程（semi-a | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1601.00901 | 9eb5b336b3dc | How big is their dataset? | 1.1 million sentences 119 diff | 你问的“their dataset”对应论文中用于实验的文本语料库。该数据集 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.06522 | 9e2e5918608a | How much training data is required for | — | 你问的“每种低资源语言所需的训练数据量”对应论文中 7 种语言各自的训练数据 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.06522 | 0ec4143a4f1a | What are the best within-language data | Frequency masking Time masking | 你问的“最佳语内数据增强方法”对应论文中在单语ASR（Kannada和Hin | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.06522 | 90159e143487 | How much of the ASR grapheme set is sh | Little overlap except common b | 你问的“ASR grapheme set 中语言间共享程度”对应论文中的 7 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1608.04917 | 938688871913 | Do the authors mention any possible co | yes | 你问的“可能的混淆因素”对应论文中关于方法论局限性的讨论。作者确实提到了一个 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1608.04917 | 4170ed011b02 | What is the relationship between the c | we observe a positive correlat | 你问的“共同投票与转发模式的关系”对应论文中关于投票与转发行为关联性的研究。 | L0→answer_L0 | 4 | pass |
| qasper_1608.04917 | fd08dc218eff | Does the analysis find that coalitions | no | 你问的“不同政策领域是否以相同方式形成联盟”对应论文中对两个政策领域（Sta | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1608.04917 | a85c2510f25c | What insights does the analysis give a | Greens-EFA, S&D, and EPP exhib | 你问的“政治团体凝聚力”对应论文中的“Cohesion of politic | L0→answer_L0 | 2 | fail |
| qasper_1608.04917 | fa572f1f3f3c | Do they authors account for difference | no | 抱歉，我无法从该论文中找到直接支撑内容来回答您的问题。检索到的段落未提及作者 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1608.04917 | 5e057e115f89 | Did the authors examine if any of the  | no | 抱歉，我无法从该论文中找到相关信息。检索到的内容未涉及MEPs在Twitte | L0→L1→L3→answer_unknown | 4 | pass |
| qasper_1809.01341 | 29e5e055e01f | What other multimodal knowledge base e | merging, concatenating, or ave | 你问的“其他多模态知识库嵌入方法”对应论文中与 MKBE 对比的基线方法。论 | L0→L1→L2→L3→answer_L3 | 3 | fail |
| qasper_2001.02943 | 010e3793eb13 | What are the sizes of both datasets? | The Dutch section consists of  | 你问的两个数据集对应论文中的 Europarl 和 SoNaR 语料。预处理 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1908.09156 | 7b3d207ed47a | What is the dataset that is used in th | — | 抱歉，我无法从该论文中找到关于所用数据集的信息，因为检索到的内容中并未提及任 | L0→L1→L3→answer_unknown | 5 | pass |
| qasper_1908.09156 | d58c264068d8 | What is the performance of the models  | — | 抱歉，我无法从该论文中找到关于模型性能的具体答案，因为文中未提供任何模型的量 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1908.09156 | f80d89fb905b | Does the paper consider the use of per | no | 抱歉，我无法从该论文中找到关于使用困惑度（perplexity）识别文本异常 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1908.09156 | 5f6fac08c97c | Does the paper report a baseline for t | no | 抱歉，我无法从该论文中找到关于是否报告基线的信息，因为检索到的内容未提及任何 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1906.06442 | c431c142f5b8 | What datasets was the method evaluated | WMT18 EnDe bitext WMT16 EnRo b | 你问的评估数据集对应论文中的 WMT18 EnDe、WMT16 EnRo 和 | L0→L1→answer_L1 | 4 | pass |
| qasper_1909.08090 | c80669cb444a | On average, by how much do they reduce | — | 抱歉，我无法从该论文中找到答案。检索到的内容仅定性说明该方法优于平均融合并接 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.08090 | 10045d7dac06 | Do they compare their algorithm to vot | no | 你问的“they”对应论文中提出的 DOVER 算法。论文在描述加权投票版本 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1909.08090 | 4e4946c02321 | How do they assign weights between vot | — | 你问的“投票间如何分配权重”对应论文中 DOVER 算法的加权投票扩展。论文 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1907.00455 | e98d331faacd | Do they compare results against state- | yes | 是的，论文将结果与现有最先进的语言模型进行了对比。文中明确提到，mGRU 模 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1907.00455 | fbe22e133fa9 | Do they integrate the second-order ter | — | 你问的“they”对应论文中提出的 mLSTM 模型。根据正文，mLSTM  | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1907.00455 | f319f2c3f933 | Which dataset do they train their mode | Penn Treebank Text8 | 你问的训练数据集对应论文中的 **Text8** 语料库 [1]。该语料库包 | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_1909.08103 | 18d8b52b4409 | How long are dialogue recordings used  | average 12.8 min per recording | 你问的“用于评估的对话录音时长”对应论文中的 CSJ 对话录音数据。该评估数 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1705.07830 | 33d2919f3400 | how are multiple answers from multiple | The selection model selects th | 你问的“多个改写问题产生的多个答案如何聚合”，对应论文中的 **candid | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.12579 | 5b6aec1b88c9 | How does proposed word embeddings comp | Proposed SG model vs SINDHI FA | 你问的“proposed word embeddings”对应论文中提出的嵌 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.12579 | a6717e334c53 | Are trained word embeddings used for a | no | 否。你问的“训练好的词嵌入是否用于其他 NLP 任务”，对应论文中提出的 S | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.12579 | a1064307a19c | How many uniue words are in the datase | 908456 unique words are availa | 你问的数据集中的唯一词数对应论文中构建的 Sindhi 语料库词汇量。该数据 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.12579 | 8cb9006bcbd2 | How is the data collected, which web r | daily Kawish and Awami Awaz Si | 你问的数据收集方式对应论文中的语料库构建过程。语料库是通过使用 web-sc | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1705.03151 | 7b89515d731d | Which is the baseline model? | The three baseline models are  | 你问的“baseline model”对应论文中的三个基线系统：基于 i-v | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1705.03151 | 1db37e98768f | How big is the Babel database? | — | 你问的 Babel 数据库规模对应论文中的 IARPA Babel 项目语料 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1705.03151 | 79a28839fee7 | What is the main contribution of the p | Proposing an improved RNN mode | 你问的“论文主要贡献”对应论文提出的 **PTN LID 方法**。其核心贡 | L0→answer_L0 | 5 | pass |
| qasper_1909.06762 | ee31c8a94e07 | What were the evaluation metrics? | BLEU Micro Entity F1 quality o | 你问的评估指标对应论文中用于衡量模型性能的指标。论文采用 **BLEU**  | L0→L1→answer_L1 | 4 | pass |
| qasper_1909.06762 | 66d743b735ba | What were the baseline systems? | Attn seq2seq Ptr-UNK KV Net Me | 你问的基线系统对应论文中对比的多个基线模型，包括：Attn seq2seq、 | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.06762 | b9f852256113 | Which dialog datasets did they experim | Camrest InCar Assistant | 你问的实验数据集对应论文中的 **Camrest** 和 **InCar A | L0→answer_L0 | 5 | pass |
| qasper_1909.06762 | 88f8ab2a417e | What KB is used? | — | 你问的 KB 对应论文中使用的两个公开数据集：CamRest 和 InCar | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1708.00111 | 0a92352839b5 | By how much do they outperform basic g | 2 accuracy points | 你问的“they”对应论文提出的连续松弛束搜索训练方法（INLINEFORM | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1708.00111 | 242f96142116 | Do they provide a framework for buildi | yes | 是的。论文提出了一种连续松弛的束搜索（continuous relaxati | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1708.00111 | fcd0bd2db398 | Do they compare partially complete seq | yes | 否。论文中并未比较束搜索过程中产生的部分完成序列与金标准/目标序列。论文提出 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1708.00111 | 5cc937c2dcb8 | Which loss metrics do they try in thei | continuous relaxation to top-k | 你问的“新训练过程中在 beam search 输出上评估的损失指标”对应论 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1908.07822 | ce6f6cd55ada | How efective is MCDN for ambiguous and | — | 你问的 MCDN 在模糊与隐式因果关系推断上的有效性，对应论文中提出的 Mu | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.07822 | d4ce220dcdab | What performance did proposed method a | increases F1-score by 10.2% an | 你问的“proposed method”对应论文中的 MCDN 模型。在公开 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.07822 | c806e891324a | What was previous state-of-the-art app | TextCNN, TextRNN, SASE, DPCNN, | 抱歉，我无法从该论文中找到关于“先前最先进方法”的具体内容，因为检索段落未提 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1908.07822 | e1132564b0dd | How is Relation network used to infer  | we integrate object-pairs with | 你问的“Relation network 在段级如何推断因果关系”对应论文中 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1912.01679 | 04cab3325e20 | What are baseline models on WSJ eval92 | Wav2vec BIBREF22 a fully-super | 你问的 WSJ eval92 和 LibriSpeech test-clea | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1612.07843 | defc17986d3c | Are the document vectors that the auth | yes | 你问的“document vectors”对应论文中作者提出的 **docu | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1612.07843 | 03895bc75e4d | According to the authors, why does the | CNN model which additionally l | 你问的 CNN 模型对应论文中的 CNN2 模型。作者认为其解释力更高，是因 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1612.07843 | 6bcff3ef61aa | Does the LRP method work in settings t | yes | 你问的“LRP 方法是否能在将词语彼此语境化的设置中工作”，对应论文中 LR | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.11189 | bfa3776c30cb | What is the algorithm used for the cla | Random Forest Ensemble classif | 你问的分类任务所用算法对应论文中的**随机森林集成分类器（Random Fo | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.11189 | a2a66726a5dc | Is the outcome of the LDA analysis eva | yes | 是的，LDA分析的结果得到了评估。论文通过监督分类任务来验证LDA特征的有效 | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.11189 | ee87608419e4 | What is the corpus used in the study? | TextGrid Repository | 你问的语料库对应论文中用于诗歌历时研究的德语诗歌语料库。该语料库来自 zen | L0→L1→answer_L1 | 3 | fail |
| qasper_1609.01962 | fe6bb55b28f1 | Why is a Gaussian process an especiall | avoids the need for expensive  | 你问的“高斯过程为何特别适合该分类问题”，对应论文中 GP 作为推文立场分类 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.02776 | 49ea25af6f75 | What dataset is used for this task? | the Pasokh dataset BIBREF42 | 你问的数据集对应论文中的 **Pasokh 数据集**，包含 100 篇波斯 | L0→L1→answer_L1 | 5 | pass |
| qasper_1909.02776 | aecd09a817c3 | What features of the document are inte | Ordinal position Length of sen | 你问的“整合进每个句子向量的文档特征”对应论文中的“document-awa | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.02776 | 81064bbd0a0d | By how much is precission increased? | ROUGE-1 increases by 0.05, ROU | 抱歉，我无法从该论文中找到关于精确率提升幅度的具体数值或对比数据，因此无法回 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1909.02776 | 7d841b98bcee | Is new approach tested against state o | no | 否。论文中并未将新方法（document-aware features）与“ | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.07481 | c1f4d632da78 | Which language-pair had the better per | French-English | 你问的“哪个语言对表现更好”对应论文中的三个翻译方向（En→De、En→Fr | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.07481 | 749a307c3736 | Which datasets were used in the experi | WMT 2019 parallel dataset a re | 你问的实验数据集对应论文中的训练与测试语料。训练数据方面，英德语言对使用完整 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.07481 | 102de97c123b | What evaluation metrics did they use? | BLEU and TER scores | 你问的评估指标对应论文中用于衡量翻译质量的指标。论文使用了 **BLEU** | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1912.00864 | 572458399a45 | How much more accurate is the model th | For the Oshiete-goo dataset, t | 你问的“模型”对应论文中的 NAGM（完整模型），“基线”对应 Seq2se | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1906.10519 | 9c4ed8ca59ba | what baseline do they compare to? | VecMap Muse Barista | 你问的基线对应论文中与 Blse 对比的投影方法及资源需求更高的上界方法。论 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.08402 | f5cf8738e8d2 | By how much do they outperform standar | up to four percentage points i | 你问的“they”对应论文中提出的带元数据特征和作者嵌入的 BERT-Ger | L0→answer_L0 | 5 | pass |
| qasper_1909.08402 | bed527bcb0dd | What dataset do they use? | 2019 GermEval shared task on h | 你问的数据集对应论文中基于 GermEval 2019 共享任务构建的书籍分 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1909.08402 | aeab5797b541 | How do they combine text representatio | all three representations are  | 你问的“文本表示与知识图谱嵌入的结合方式”对应论文中模型架构部分的处理流程。 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1807.08204 | 848ab388703c | How are proof scores calculated? | '= ( , { ll k(h:, g:) if hV, g | 你问的 proof scores 的计算方式对应论文中的最终成功分数（fin | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1807.08204 | 68794289ed60 | What are proof paths? | A sequence of logical statemen | 你问的“proof paths”对应论文中 NTP（神经定理证明器）在反向链 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2003.12218 | ce6201435cc1 | Did they experiment with the dataset? | yes | 是的，论文对 CORD-19 语料库（2020-03-13 版本，含 29, | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2003.12218 | 928828544e38 | What is the size of this dataset? | 29,500 documents | 你问的“this dataset”对应论文中的 CORD-19-NER 数据 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2003.12218 | 4f243056e63a | Do they list all the named entity type | no | 你问的“是否列出全部命名实体类型”对应论文中的实体类型层级（entity t | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1612.09113 | 85e45b37408b | What is the baseline? | The baseline is a multi-task a | 你问的 baseline 对应论文中的基线多任务架构，其灵感来源于 yang | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1612.09113 | f4e1d2276d3f | What is the unsupervised task in the f | Language Modeling | 你问的“最终层的无监督任务”对应论文中的 Language Modeling | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1612.09113 | bf2ebc9bbd4c | How many supervised tasks are used? | two | 你问的监督任务数量对应论文中的错误信号来源数量。论文明确指出模型有 3 个错 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1612.09113 | c13fe4064df0 | What is the network architecture? | The network architecture has a | 你问的网络架构对应论文中提出的多任务双向循环神经网络（multi-task  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2004.01878 | 5a23f436a7e0 | How big is dataset used? | 553,451 documents | 你问的数据集规模对应论文中用于实验的金融新闻数据集。该数据集是从 Reute | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_2004.01878 | 2f4acd34eb2d | What is dataset used for news-driven s | the public financial news data | 你问的数据集对应论文中用于新闻驱动股票走势预测的公开金融新闻数据集。该数据集 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1606.01404 | f4e17b14318b | How is the generative model evaluated? | Comparing BLEU score of model  | 你问的生成模型评估对应论文中的序列到序列蕴含生成模型。该模型通过两种方式评估 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1604.06076 | cb1126992a39 | How is the semi-structured knowledge b | using a mixture of manual and  | 你问的知识库构建方式对应论文中的表格知识库（semi-structured  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.14254 | 08cbc9b8a8df | Which dataset do they use? | the annotated dataset reported | 你问的数据集对应论文中的 **degen2015investigating  | L0→L1→answer_L1 | 5 | pass |
| qasper_1601.06738 | 320d72a9cd19 | Does this paper address the variation  | no | 该论文未涉及英语方言间在这些模糊限制语使用上的差异，检索内容中缺乏相关讨论或 | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1909.09524 | 191d4fe8a376 | Are experiments performed with any oth | no | 你问的实验语言对对应论文中的两个非英语语言对：法语→德语（French→Ge | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.09524 | 6e76f114209f | Is pivot language used in experiments  | yes | 你问的实验中使用的枢轴语言对应论文中的英语（English）。[1] 明确指 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.09524 | 6583e8bfa7bc | What are multilingual models that were | Direct source$\rightarrow $tar | 你问的“被超越的多语言模型”对应论文中的 **Multilingual**  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1811.01001 | da2b43d7d048 | What training settings did they try? | Training and testing are done  | 你问的训练设置对应论文中的三种训练长度窗口。作者实验了三个不同的长度窗口：$ | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1811.01001 | b7708cbb5008 | How do they get the formal languages? | These are well-known formal la | 你问的“如何获得这些形式语言”对应论文中的**语言定义与输入-目标（inpu | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1811.01001 | 17988d65e46f | Are the unobserved samples from the sa | no | 否。你问的“未观测样本是否与训练数据同分布”对应论文中的测试方式——论文明确 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2004.01694 | c1429f7fed5a | What empricial investigations do they  | empirically test to what exten | 你问的“empirical investigations”对应论文中作者自己 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2004.01694 | a93d4aa89ac3 | What languages do they investigate for | English  Chinese | 你问的“机器翻译涉及的语言”对应论文中评估实验所用的语言对。论文主要调查的是 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2004.01694 | bc473c5bd0e1 | What recommendations do they offer? | Choose professional translator | 你问的“they”对应论文作者。作者提出的评估建议包括：不要为流畅性而激进编 | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_2004.01694 | cc5d8e12f6ae | What percentage fewer errors did profe | 36% | 抱歉，我无法从该论文中找到答案。检索到的内容未提供专业翻译与机器翻译错误数量 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2004.01694 | 9299fe72f19c | What was the weakness in Hassan et al' | MT developers to which crowd w | 你问的 Hassan et al. 评估设计的弱点，对应论文中指出的其宣称“ | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1711.03438 | a81941f93390 | Can the model add new relations to the | The model does not add new rel | 你问的“模型能否向知识图谱添加新关系”对应论文中的 ConMask 模型。* | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.05763 | 8ea664a72e6d | Are recurrent neural networks trained  | no | 否。你问的“循环神经网络是否在扰动数据上训练”对应论文中的 LSTM 基线模 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.05763 | 5e41516a27c5 | How does their perturbation algorihm w | same sentences after applying  | 你问的扰动算法对应论文中的字符级与词级两类扰动操作。字符级扰动中，每个词以固 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1809.08298 | 7633be56ae46 | Which machine learning models do they  | conditional random field model | 你问的用于纠正流水句的机器学习模型，对应论文中的 **roCRF** 和 * | L0→answer_L0 | 5 | pass |
| qasper_1809.08298 | dafa760e1466 | How large is the dataset they generate | 4.756 million sentences | 抱歉，我无法从该论文中找到关于生成数据集规模（如句子数量或样本量）的具体信息 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1910.02754 | 04aff4add28e | What is result of their attention dist | visual attention is very spars | 你问的注意力分布分析结果对应论文中的视觉注意力稀疏性分析。分析发现视觉注意力 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.02754 | a8e4522ce2ce | What is result of their Principal Comp | existing visual features aren' | 你问的 PCA 分析结果对应论文中对 How2 与 Multi30k 数据集 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.02754 | f6202100cfb8 | What are 3 novel fusion techniques tha | Step-Wise Decoder Fusion Multi | 你问的 3 种新融合技术对应论文中提出的三种方法，分别是：Step-Wise | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1807.09000 | 4b6240643320 | Did participants behave unexpectedly? | no | 是，参与者行为符合预期。脚本条件下听者在大约50%的关键试次中错误移动了隐藏 | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1807.09000 | 65ba7304838e | Was this experiment done in a lab? | no | 抱歉，我无法从该论文中找到关于实验场所（如是否在实验室进行）的明确信息，因为 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1705.03261 | b42323d60827 | How big is the evaluated dataset? | contains thousands of XML file | 你问的“被评估的数据集”对应论文中的 DDI 语料库（2013 DDIExt | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1705.03261 | 1a69696034f7 | By how much does their model outperfor | Answer with content missing: ( | 你问的“their model”对应论文中提出的“recurrent neu | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1705.03261 | 9a596bd3a1b5 | What is the performance of their model | Answer with content missing: ( | 你问的“their model”对应论文中提出的 RNN + 动态词嵌入 + | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1705.03261 | 1ba28338d3f9 | What are the existing methods mentione | Chowdhury BIBREF14 and Thomas  | 你问的“现有方法”对应论文中 DDI 抽取与关系抽取领域被提及的各类方法。文 | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1708.06022 | e91692136033 | It looks like learning to paraphrase q | using multiple pivot sentences | 你问的“端到端训练”对应论文中提出的 **Para4QA** 框架。该框架中 | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1912.07025 | 79bb1a1b71a1 | What accuracy does CNN model achieve? | Combined per-pixel accuracy fo | 抱歉，我无法从该论文中找到CNN模型准确率的具体数值或评估结果，因为检索内容 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1912.07025 | 26faad6f42b6 | How many documents are in the Indiscap | 508 | 你问的 Indiscapes 数据集规模对应论文中的“508 annotat | L0→L1→answer_L1 | 5 | pass |
| qasper_1912.07025 | 20be7a776dfd | What language(s) are the manuscripts w | — | 你问的手稿语言对应论文中 Indiscapes 数据集的书写语言。论文指出这 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1903.07398 | c8f11561fc4d | Which dataset(s) do they evaluate on? | LJSpeech | 你问的评测数据集对应论文中的 LJSpeech Dataset [1]。该数 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1903.07398 | 51de39c8bad6 | Which modifications do they make to we | Replacing attention mechanism  | 你问的“对成熟 Seq2Seq 架构的修改”对应论文中提出的三项改进。第一， | L0→answer_L0 | 4 | pass |
| qasper_1903.07398 | d9cbcaf8f045 | How do they measure the size of models | Direct comparison of model par | 你问的“模型大小”对应论文中直接比较模型参数数量的做法。论文通过统计模型的总 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1903.07398 | fc69f5d9464c | Do they reduce the number of parameter | yes | 是，论文提出的模型确实减少了参数量。你问的“其他直接文本转语音模型”对应论文 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.03154 | 317a6f211ecf | Has there been previous work on SNMT? | yes | 是的，论文中明确提到此前已有大量关于 SNMT（同传神经机器翻译）的研究工作 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.03154 | a726046eec1e | Which languages do they experiment on? | German English Chinese | 你问的实验语言对应论文中的三个翻译数据集。实验涉及的语言对包括：WMT15  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.03154 | 6d9fbd42b543 | What corpora is used? | IWSLT16 WMT15 NIST | 你问的语料对应论文中用于评测的多个翻译数据集，具体包括：WMT15 德英（D | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2001.06785 | 7835d8f57838 | Is the model evaluated against a basel | no | 是的，论文中的模型是相对于基线进行评估的。你问的“基线”对应论文中的系统 A | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_2001.06785 | 32e78ca99ba8 | How many people are employed for the s | 14 volunteers | 你问的主观评价参与人数对应论文中的志愿者人数。共有 14 名志愿者参与了评价 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1603.04553 | 80de3baf97a5 | Are resolution mode variables hand cra | no | 你问的“resolution mode variables”对应论文中的 $ | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1603.04553 | f5707610dc8a | What are resolution model variables? | Variables in the set {str, pre | 你问的“resolution model variables”对应论文中的  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1603.04553 | e76139c63da0 | Is the model presented in the paper st | No, supervised models perform  | 你问的“论文提出的模型是否达到 state of the art”对应正文中 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1709.07916 | e5ae8ac51946 | Do they evaluate only on English data? | yes | 是，论文仅在英语数据上进行评估。数据收集阶段明确说明，研究使用实时方法随机收 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1709.07916 | 18288c7b0f8b | How strong was the correlation between | weak correlation with p-value  | 你问的“exercise 与 diabetes 的相关性”对应论文中 DDE | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1709.07916 | b5e883b15e63 | How were topics of interest about DDEO | using topic modeling model Lat | 你问的“DDEO相关主题如何被识别”对应论文中的主题发现（Topic Dis | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1812.10860 | 1d6c42e3f545 | Do some pretraining objectives perform | yes | 是的。论文发现语言建模是最有效的单一预训练任务 [2]，而多任务预训练可进一 | L0→answer_L0 | 4 | pass |
| qasper_1704.04539 | 8fa7011e7bea | Do the authors test their annotation p | no | 抱歉，我无法从该论文中找到答案。检索到的内容仅围绕AMR解析展开，未涉及将标 | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_1704.04539 | e0b7acf4292b | How is annotation projection done when | Word alignments are generated  | 你问的“不同语序下的标注投影”对应论文中的 **Annotation Pro | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1806.01733 | ff0f77392abc | What features did they train on? | direct similarity over Concept | 你问的“训练特征”对应论文中用于区分性属性任务的特征提取，即从 Concep | L0→L1→L2→answer_L2 | 2 | fail |
| qasper_1906.11604 | 0bd864f83626 | How are sentence embeddings incorporat | BERT generates sentence embedd | 你问的“sentence embeddings 如何融入语音识别系统”对应论 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1910.04601 | b11ee27f3de7 | What is the baseline? | path ranking-based KGC (PRKGC) | 你问的 baseline 对应论文中的 RC-QED$^{\rm E}$ b | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1910.04601 | 7aba5e448329 | What dataset was used in the experimen | WikiHop | 你问的实验所用数据集对应论文中作者自行创建的RC-QED语料库，该语料库包含 | L0→answer_L0 | 1 | fail |
| qasper_1910.04601 | 565d668947ff | Did they use any crowdsourcing platfor | yes | 是的，论文使用了众包平台。文中明确提到通过亚马逊土耳其机器人（AMT）发布众 | L0→L1→answer_L1 | 4 | pass |
| qasper_1910.04601 | d83304c70fe6 | How was the dataset annotated? | True, Likely (i.e. Answerable) | 你问的数据集标注方式对应论文中的众包标注流程。标注任务部署在 Amazon  | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1910.04601 | e90ac9ee085d | What is the source of the proposed dat | — | 你问的数据集来源对应论文中的语料库构建方式。该数据集通过众包（crowdso | L0→L1→L2→answer_L2 | 1 | fail |
| qasper_1909.01093 | d82ec1003a3d | How does the method measure the impact | We collected the historical 52 | 你问的“方法如何衡量事件对市场价格的影响”对应论文中的案例研究（Case S | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.01093 | 58f08d38bbcf | How is sentiment polarity measured? | For each cluster, its overall  | 你问的“情感极性如何衡量”对应论文中争议识别阶段的**情感分类（Sentim | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1703.02507 | 09a993756d27 | Do they report results only on English | yes | 你问的“they”对应论文中提出的 Sent2Vec 模型及其对比实验。基于 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1703.02507 | 37eba8c3cfe2 | Which other unsupervised models are us | Sequential (Denoising) Autoenc | 你问的“其他无监督对比模型”对应论文中用于对比的无监督基线模型。证据中出现的 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1703.02507 | cdf1bf4b2025 | What metric is used to measure perform | Accuracy and F1 score for supe | 你问的性能度量指标对应论文中的两类评测：监督分类任务用**准确率（Accur | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1703.02507 | 03f4e5ac5a90 | How do the n-gram features incorporate | by also learning source embedd | 你问的 n-gram 特征如何体现组合性，对应论文中 Sent2Vec 的句 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1710.01507 | acc8d9918d19 | What are the differences with previous | This approach considers relate | 你问的“previous applications of neural ne | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1709.08858 | 238ec3c1e109 | How is the fluctuation in the sense of | Our method performs a statisti | 你问的“词与其邻近词在词义上的波动如何衡量”对应论文中的方法：通过计算邻近词 | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_1908.06606 | 71a7153e1287 | What data is the language model pretra | Chinese general corpus | 你问的语言模型预训练数据对应论文中的 BERT-base 模型，其参数直接采 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.06606 | 85d1831c28d3 | What baselines is the proposed model c | BERT-Base QANet | 你问的基线对应论文中的 QANet 和 BERT-Base 两个模型 [1] | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.06606 | 1959e0ebc21f | How is the clinical text structuring t | Clinical text structuring (CTS | 你问的“临床文本结构化任务”对应论文中首次提出的 **QA-CTS（ques | L0→answer_L0 | 3 | fail |
| qasper_1908.06606 | 77cf43791064 | What are the specific tasks being unif | three types of questions, name | 你问的“被统一的特定任务”对应论文中的三个具体临床文本结构化任务。根据证据， | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.06606 | 06095a4dee77 | Is all text in this dataset a question | the dataset consists of pathol | 你问的“数据集中的文本是否全是问题”对应论文中的 QA-CTS 数据集。否， | L0→L1→L2→answer_L2 | 4 | pass |
| qasper_1908.06606 | 19c9cfbc4f29 | How many questions are in the dataset? | 2,714 | 你问的数据集中的问题数量对应论文中的 QA-CTS 数据集。该数据集共含 * | L0→L1→answer_L1 | 5 | pass |
| qasper_1908.06606 | 6743c1dd7764 | What is the perWhat are the tasks eval | — | 你问的“评估的任务”对应论文中的 QA-CTS 任务所涵盖的三个具体临床文本 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1908.06606 | 14323046220b | Are there privacy concerns with clinic | — | 抱歉，该论文中未涉及临床数据的隐私问题，缺少相关讨论或结论，因此无法基于此文 | L0→L1→L3→answer_unknown | 5 | pass |
| qasper_1908.06606 | 08a5f8d36298 | How they introduce domain-specific fea | integrate clinical named entit | 你问的“如何将领域特定特征引入预训练语言模型”对应论文中的方法：作者将临床命 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1908.06606 | 975a4ac9773a | How big is QA-CTS task dataset? | 17,833 sentences, 826,987 char | 你问的 QA-CTS 任务数据集对应论文中基于瑞金医院胃肠外科中文病理报告构 | L0→L1→answer_L1 | 5 | pass |
| qasper_1908.06606 | 326e08a0f575 | How big is dataset of pathology report | 17,833 sentences, 826,987 char | 你问的“瑞金医院病理报告数据集规模”对应论文中基于瑞金医院胃肠外科中文病理报 | L0→L1→answer_L1 | 5 | pass |
| qasper_1908.06606 | bd78483a746f | What are strong baseline models in spe | state-of-the-art question answ | 你问的“strong baseline models”对应论文中用于对比的基 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.03242 | a74886d789a5 | What were the baselines? | a BiLSTM over all words in the | 你问的“baselines”对应论文中用于验证事实性预测（claim ver | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1909.03242 | e9ccc74b1f1b | What metadata is included? | besides claim, label and claim | 你问的元数据对应论文中从每个事实核查网站抓取的字段。论文概述和正文[1]中列 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.03242 | 2948015c2a5c | How many expert journalists were there | — | 抱歉，我无法从该论文中找到关于专家记者数量的任何信息，因为文中并未提供相关数 | L0→L1→L3→answer_unknown | 5 | pass |
| qasper_1905.12516 | f197e0f61f79 | Any other bias may be detected? | — | 你问的“其他偏见”对应论文中除种族偏见外、在相关工作里提到的各类偏见。论文指 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.03467 | b367b823c5db | Are synonymous relation taken into acc | yes | 是。在日语-越南语任务中，同义关系被纳入考量：由于越南语 WordNet 不 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1910.03467 | 84737d871bde | Is the supervised morphological learne | no | 否。你问的“监督式形态学习器”对应论文中的 SAA（separating a | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.03438 | 334f90bb715d | What semantic features help in detecti | No feature is given, only disc | 你问的“检测文本真伪所依赖的语义特征”对应论文中讨论的“语义侧信息（sema | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.03438 | 53c8416f2983 | Which language models generate text th | Outputs from models that are b | 你问的“更容易被判定为真实或生成文本的语言模型”对应论文中关于**交叉熵或困 | L0→answer_L0 | 4 | pass |
| qasper_2002.03438 | 5b2480c65336 | Is the assumption that natural languag | It is not completely valid for | 你问的“自然语言是平稳且遍历的假设是否成立”，对应论文中 Manning 和 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1906.06045 | 1083ec9a2a33 | What is the training objective of thei | is to minimize the negative li | 你问的 pair-to-sequence 模型的训练目标对应论文中的方法核心 | L0→L1→answer_L1 | 5 | pass |
| qasper_1906.06045 | 58a00ca123d6 | How do they ensure the generated quest | learn to ask unanswerable ques | 你问的“如何确保生成的问题不可回答”对应论文中提出的**生成约束与训练数据构 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1906.06045 | 199bdb3a6b1f | Does their approach require a dataset  | yes | 是的。论文提出的方法需要将不可回答问题与相似的可回答问题配对的数据集：他们使 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1911.03514 | e70236c876c9 | Did they experiment on the proposed ta | no | 抱歉，根据检索到的内容，我无法确认该论文是否针对所提出的任务进行了实验。现有 | L0→L1→L2→L3→answer_unknown | 3 | unknown_ok |
| qasper_1911.03514 | aa1f605619b2 | Is annotation done manually? | yes | 是，标注是人工完成的。论文在描述标注流程时指出，标注者需要“从开头到结尾阅读 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1911.03514 | 9f2634c142dc | How large is the proposed dataset? | we obtain 52,053 dialogues and | 你问的数据集规模对应论文中最终用于标注的对话数据。经预处理后，论文最终获得  | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1912.05066 | 5b029ad0d20b | How many label options are there in th | two labels | 你问的多标签任务中的标签选项数量，对应论文中每个 tweet 的标注结构。根 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1912.05066 | 79bd2ad4cb5c | What is the interannotator agreement o | — | 抱歉，我无法从该论文中找到关于众包用户标注者间一致性（interannota | L0→L1→L2→L3→answer_unknown | 5 | pass |
| qasper_1912.05066 | d3a1a53521f2 | Who are the experts? | political pundits of the Washi | 你问的“experts”对应论文中与“crowd wisdom”（微博用户群 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1912.05066 | 38e11663b03a | Who is the crowd in these experiments? | peoples' sentiments expressed  | 你问的“crowd”对应论文中的“microblog users”（微博用户 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1912.05066 | 14421b7ae445 | How do you establish the ground truth  | experts in Washington Post | 你问的“辩论获胜者的 ground truth”对应论文中由专家（Washi | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1710.10609 | 3f7a7e81908a | Do they study frequent user responses  | yes | 你问的“frequent user responses”对应论文中的“fre | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1710.10609 | 28e7711f94e0 | How do they divide text into utterance | — | 你问的“如何将文本划分为话语”对应论文中的数据预处理步骤。论文将同一说话者的 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1710.10609 | 49b38189b833 | Do they use the same distance metric f | yes | 你问的“距离度量”对应论文中 SimCluster 与 K-means 的聚 | L0→L1→L2→L3→answer_L3 | 2 | fail |
| qasper_1710.10609 | 40c2bab4a6bf | How do they generate the synthetic dat | using generative process | 你问的合成数据集生成方式对应论文正文 [1] 中的生成过程。他们通过一个生成 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.07758 | ead5dc1f3994 | How many category tags are considered? | 14 categories | 抱歉，我无法从该论文中找到关于“category tags”数量的任何信息， | L0→L1→L2→L3→answer_unknown | 1 | unknown_ok |
| qasper_2003.07758 | 86cd12283747 | What domain does the dataset fall into | YouTube videos | 你问的数据集领域对应论文中的 **dense video captionin | L0→L1→L2→answer_L2 | 3 | fail |
| qasper_2003.07758 | 7011b26ffc54 | What ASR system do they use? | YouTube ASR system | 你问的 ASR 系统对应论文中用于获取语音内容的 YouTube ASR 系 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_2003.07758 | 3a6559dc6eba | What is the state of the art? | — | 你问的“state of the art”对应论文中与现有方法对比的部分。论 | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_1707.05853 | 2210178facc0 | What type of recurrent layers does the | GRU | 你问的模型使用的循环层对应论文中的 **GRU（门控循环单元）**。论文明确 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1707.05853 | 7cf726db952c | What is a word confusion network? | It is a network used to encode | 你问的“word confusion network”对应论文中的“conf | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2002.06644 | 830de0bd007c | Do the authors report only on English? | — | 你问的“是否只报告英语”对应论文中的实验语料与语言设置。基于证据，答案是** | L0→L1→L2→L3→answer_L3 | 4 | pass |
| qasper_2002.06644 | 680dc3e56d1d | What is the baseline for the experimen | FastText BiLSTM BERT | 你问的“baseline”对应论文中的基线模型，共三个：FastText、B | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_2002.06644 | bd5379047c2c | Which experiments are perfomed? | They used BERT-based models to | 你问的“experiments”对应论文中在 WNC 语料上进行的**主观偏 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1909.00754 | ed7a3e7fc167 | Does this approach perform better in t | single-domain setting | 你问的“该方法的性能在单域还是多域设置下更好”，对应论文中 COMER 在  | L0→L1→L2→L3→answer_L3 | 1 | fail |
| qasper_1909.00754 | 72ceeb58e783 | What are the performance metrics used? | joint goal accuracy | 你问的性能指标对应论文中的 **joint goal accuracy**（ | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1909.00754 | 9bfa46ad5513 | Which datasets are used to evaluate pe | the single domain dataset, WoZ | 你问的评测数据集对应论文中的 WoZ2.0 和 MultiWoZ 两个数据集 | L0→answer_L0 | 4 | pass |
| qasper_1804.02233 | dc28ac845602 | How many tweets were manually labelled | 44,000 tweets | 你问的人工标注推文数量对应论文中用于构建 Twitter 立场模型的标注子集 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1906.04571 | f7817b949605 | Why does not the approach from English | Because, unlike other language | 你问的“English approach”对应论文中针对英语开发的性别刻板印 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1906.04571 | 8255f74cae13 | How do they measure grammaticality? | by calculating log ratio of gr | 你问的“如何衡量语法正确性”对应论文中的外在评估部分。他们通过计算“语法短语 | L0→L1→L2→answer_L2 | 5 | pass |
| qasper_1906.04571 | db62d5d83ec1 | Which model do they use to convert bet | Markov random field with an op | 你问的“用于转换阳性/阴性屈折句子的模型”对应论文中提出的方法，即一个**马 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1809.08731 | 7aa8375cdf46 | Is ROUGE their only baseline? | no | 否。你问的“baseline”对应论文中用于对比的基线指标，ROUGE 只是 | L0→L1→L2→L3→answer_L3 | 5 | pass |
| qasper_1809.08731 | 3ac30bd7476d | what language models do they use? | LSTM LMs | 你问的语言模型对应论文中的 LSTM 语言模型（LSTM LMs）[1]。这 | L0→L1→L2→answer_L2 | 5 | pass |

## 主通过率（仅计有答案题 pass / 全部题）：75.0%

## 附加统计（20260907_142814）
- status: {'pass': 750, 'fail': 190, 'unknown_ok': 60}
- 有答案题 status: {'pass': 660, 'fail': 176, 'unknown_ok': 60}
- error 题数: 0
