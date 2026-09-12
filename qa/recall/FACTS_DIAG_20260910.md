# 两步法失效归因（off 臂失败题）

> 判读三层：gold 内容在 (a)12块内 → (b)事实清单内 → (c)成稿内

==========================================================================================
## [1] de12e05908 [hard] off_score=1
Q: What are the baseline systems that are compared against?
gold: The system is compared to baseline models: LSTM, RL-SPINN and Gumbel Tree-LSTM
evidence: FLOAT SELECTED: Table 1: Accuracy on the ListOps dataset. All models have 128 dimensions. Results for models with * are taken from Nangia and Bowman (2018).

--- (a) 检索到 12 块 ---
  [1] 'Sections' len=3938 | Headings: Type and label section and subsection headings in the style shown on the present document. Use numbe
  [2] 'Acknowledgments' len=2647 | Adina Williams, Nikita Nangia, and Samuel Bowman. 2018b. A broad-coverage challenge corpus for sentence unders
  [3] 'Acknowledgments' len=3893 | The acknowledgments should go immediately before the references. Do not number the acknowledgments section. Do
  [4] 'Accessibility' len=400 | In an effort to accommodate people who are color-blind (as well as those printing to paper), grayscale readabi
  [5] 'Graphics' len=876 | Illustrations: Place figures, tables, and photographs in the paper near where they are first discussed, rather
  [6] 'The Ruler' len=1003 | The NAACL-HLT 2019 style defines a printed ruler which should be presented in the version submitted for review
  [7] 'Length of Submission' len=1919 | The NAACL-HLT 2019 main conference accepts submissions of long papers and short papers. Long papers may consis
  [8] 'Format of Electronic Manuscript' len=1735 | For the production of the electronic manuscript you must use Adobe's Portable Document Format (PDF). PDF files
  [9] 'Acknowledgments' len=3893 | Andriy Mnih and Karol Gregor. 2014. Neural variational inference and learning in belief networks. arXiv prepri
  [10] 'Introduction' len=341 | The following instructions are directed to authors of papers submitted to NAACL-HLT 2019 or accepted for publi
  [11] 'Credits' len=966 | This document has been adapted from the instructions for earlier ACL and NAACL proceedings, including those fo
  [12] 'Electronically-available resources' len=469 | NAACL-HLT provides this description in 2e (naaclhlt2019.tex) and PDF format (naaclhlt2019.pdf), along with the

--- (b) 事实清单（0 条）---

--- (c) 成稿 ---
提供的证据中没有任何关于该论文（Cooperative Learning of Disjoint Syntax and Semantics）所对比的基线系统的内容。所有条目均为 NAACL-HLT 2019 的格式说明、参考文献列表、致谢与排版指南等，未涉及该论文的实验设置或基线方法 [1][2][3][4][5][6][7][8][9][10][11][12]。因此无法列出任何基线系统名称。

==========================================================================================
## [2] 3de27c81af [hard] off_score=1
Q: How big is dataset used to train Word2Vec for the Italian Language?
gold: $421\,829\,960$ words divided into $17\,305\,401$ sentences
evidence: The dataset needed to train the W2V was obtained using the information extracted from a dump of the Italian Wikipedia (dated 2019.04.01), from the main categories of Italian Google News (WORLD, NATION, BUSINESS, TECHNOLOGY, ENTERTAINMENT, SPORTS, SCIENCE, HEALTH) and from some anonymized chats betwe

--- (a) 检索到 9 块 ---
  [1] 'Implementation details' len=1548 | The dataset needed to train the W2V was obtained using the information extracted from a dump of the Italian Wi
  [2] 'Conclusion' len=1107 | In this work we have analysed the Word2Vec model for Italian Language obtaining a substantial increase in perf
  [3] 'Introduction' len=3023 | In order to make human language comprehensible to a computer, it is obviously essential to provide some word e
  [4] 'Analysis of the various models' len=2220 | We first analysed 6 different implementations of the Skip-gram model each one trained for 20 epochs. Table sho
  [5] 'Word2Vec' len=1479 | The W2V structure consists of a simple two-level neural network (Figure ) with one-hot vectors representing wo
  [6] 'Comparison with other models' len=1018 | Finally, a comparison was made between the Skip-gram model W10N20 obtained at the 50th epoch and the other two
  [7] 'Negative sampling' len=1340 | Working with one-hot pairs of words means that the size of the network must be the same at input and output, a
  [8] 'Sampling rate' len=1155 | The common words (such as “the", “of", etc.) carry very little information on the target word with which they 
  [9] 'Results' len=1279 | To analyse the results we chose to use the test provided by , which consists of $19\,791$ analogies divided in

--- (b) 事实清单（0 条）---

--- (c) 成稿 ---
你问的训练意大利语 Word2Vec 所用数据集大小，对应论文中用于训练 W2V 的数据集。该数据集由意大利语维基百科、Google News 主要类别以及客服聊天机器人 Laila 的匿名聊天记录构成，原始文本为 2.6 GB [1]。它包含 421,829,960 个词，划分为 17,305,401 个句子 [1]。预处理后最终得到的词表规模为 618,224 个词 [1]。

==========================================================================================
## [3] af75ad21dd [hard] off_score=2
Q: How much does this system outperform prior work?
gold: The system outperforms by 27.7% the LSTM model, 38.5% the RL-SPINN model and 41.6% the Gumbel Tree-LSTM
evidence: FLOAT SELECTED: Table 1: Accuracy on the ListOps dataset. All models have 128 dimensions. Results for models with * are taken from Nangia and Bowman (2018).

--- (a) 检索到 12 块 ---
  [1] 'Acknowledgments' len=2647 | Adina Williams, Nikita Nangia, and Samuel Bowman. 2018b. A broad-coverage challenge corpus for sentence unders
  [2] 'Acknowledgments' len=3893 | The acknowledgments should go immediately before the references. Do not number the acknowledgments section. Do
  [3] 'Sections' len=3938 | Headings: Type and label section and subsection headings in the style shown on the present document. Use numbe
  [4] 'Sections' len=1019 | The and Bib style files provided roughly fit the American Psychological Association format, allowing regular c
  [5] 'Credits' len=966 | This document has been adapted from the instructions for earlier ACL and NAACL proceedings, including those fo
  [6] 'The Ruler' len=1003 | The NAACL-HLT 2019 style defines a printed ruler which should be presented in the version submitted for review
  [7] 'Fonts' len=312 | For reasons of uniformity, Adobe's Times Roman font should be used. In 2e this is accomplished by putting \use
  [8] 'Length of Submission' len=1919 | The NAACL-HLT 2019 main conference accepts submissions of long papers and short papers. Long papers may consis
  [9] 'Electronically-available resources' len=469 | NAACL-HLT provides this description in 2e (naaclhlt2019.tex) and PDF format (naaclhlt2019.pdf), along with the
  [10] 'General Instructions' len=1744 | Manuscripts must be in two-column format. Exceptions to the two-column format include the title, authors' name
  [11] 'Acknowledgments' len=3893 | Andriy Mnih and Karol Gregor. 2014. Neural variational inference and learning in belief networks. arXiv prepri
  [12] 'Translation of non-English Terms' len=302 | It is also advised to supplement non-English characters and terms with appropriate transliterations and/or tra

--- (b) 事实清单（0 条）---

--- (c) 成稿 ---
根据提供的全部条目，无法回答该系统比先前工作提升了多少。这些条目全部是 NAACL-HLT 2019 的格式说明、参考文献列表和致谢模板内容 [1][2][3][4][5][6][7][8][9][10][11][12]，其中没有任何关于 "Cooperative Learning of Disjoint Syntax and Semantics" 这一系统性能、对比基线或提升幅度的实验数据或数值。缺少的是：该系统的评测结果、所对比的先前工作（prior work）及其对应指标数值。

==========================================================================================
## [4] a3f108f601 [hard] off_score=3
Q: what was their system's f1 performance?
gold: Proposed model achieves 0.86, 0.924, 0.71 F1 score on SR, HATE, HAR datasets respectively.
evidence: FLOAT SELECTED: Table 2: F1 Results3

--- (a) 检索到 12 块 ---
  [1] 'Results and Discussion' len=1538 | The approach we have developed establishes a new state of the art for classifying hate speech, outperforming p
  [2] 'Preprocessing' len=1271 | Original text RT @AGuyNamed_Nick Now, I'm not sexist in any way shape or form but I think women are better at 
  [3] 'Experimental Setup' len=1048 | We tokenize the data using Spacy . We use 300 Dimensional Glove Common Crawl Embeddings (840B Token) and fine 
  [4] 'Related Work' len=2123 | Many efforts have been made to classify hate speech using data scraped from online message forums and popular 
  [5] 'Supplemental Material' len=301 | We experimented with several different preprocessing variants and were surprised to find that reducing preproc
  [6] 'Error Analysis' len=1600 | False negatives Many of the false negatives we see are specific references to characters in the TV show “My Ki
  [7] 'Transformed Word Embedding Model (TWEM)' len=438 | Our training set consists of INLINEFORM0 examples INLINEFORM1 where the input INLINEFORM2 is a sequence of tok
  [8] 'Introduction' len=1501 | The increasing popularity of social media platforms like Twitter for both personal and political communication
  [9] 'Data' len=929 | In this paper, we use three data sets from the literature to train and evaluate our own classifier. Although a
  [10] 'Pooling' len=626 | We make use of two pooling methods on the updated embedding space INLINEFORM0 . We employ a max pooling operat
  [11] 'Embedding Analysis' len=1845 | Since our method was a simple word embedding based model, we explored the learned embedding space to analyze r
  [12] 'Word Embeddings' len=586 | Each token in the input is mapped to an embedding. We used the 300 dimensional embeddings for all our experime

--- (b) 事实清单（5 条）---
  (1) The approach we have developed establishes a new state of the art for classifying hate speech, outperforming previous results by as much as 12 F1 points.
  (1) Table illustrates the robustness of our method, which often outperform previous results, measured by weighted F1.
  (1) On the SR dataset, we outperform 's text based model by 3 F1 points, while just falling short of the Text + Metadata Interleaved Training model.
  (1) Excluding the embedding weights, our model requires 100k parameters , while requires 250k parameters.
  (4) Both networks used in combination - and especially when trained using transfer learning - achieved higher F1 scores than either neural network classifier alone.

--- (c) 成稿 ---
你问的“their system's F1 performance”对应论文自己提出的方法（基于 SWEM 的模型）。论文没有给出该系统在三个数据集上的具体 F1 数值，只给出相对比较：其方法刷新了 hate speech 分类的 state of the art，比此前结果最多高出 12 F1 points [1]；在 SR 数据集上比某文本模型高 3 F1 points，但略低于 Text + Metadata Interleaved Training 模型 [1]。因此，证据中可确认的是“最多提升 12 F1 points”和“SR 上高 3 F1 points”这两个差量，而非该系统自身的绝对 F1 分数 [1]。

==========================================================================================
## [5] 8df89988ad [hard] off_score=3
Q: What baselines are used in experiments?
gold: Typical implementations of dynamic programming algorithms are serial in the length of the sequence Computational complexity is even more of an issue for parsing algorithms, which cannot be as easily p
evidence: Optimizations ::: a) Parallel Scan Inference

--- (a) 检索到 12 块 ---
  [1] 'Conditional Random Fields' len=2460 | We now describe the technical approach underlying the library. To establish notation first consider the implem
  [2] 'a) Parallel Scan Inference' len=881 | The commutative properties of semiring algorithms allow flexibility in the order in which we compute $A(\ell )
  [3] 'Introduction' len=2337 | Structured prediction is an area of machine learning focusing on representations of spaces with combinatorial 
  [4] 'Motivating Case Study' len=2144 | While structured prediction is traditionally presented at the output layer, recent applications have deployed 
  [5] 'Dynamic Programming and Semirings' len=2573 | Torch-Struct is a collection of generic algorithms for CRF inference. Each CRF distribution object, $\textsc {
  [6] 'Library Design' len=1736 | The library design of Torch-Struct follows the distributions API used by both TensorFlow and PyTorch . For eac
  [7] 'b) Vectorized Parsing' len=953 | Computational complexity is even more of an issue for parsing algorithms, which cannot be as easily paralleliz
  [8] 'Conclusion and Future Work' len=1071 | We present Torch-Struct, a library for deep structured prediction. The library achieves modularity through its
  [9] 'Acknowledgements' len=226 | We thank Yoon Kim, Xiang Lisa Li, Sebastian Gehrmann, Yuntian Deng, and Justin Chiu for discussion and feedbac
  [10] 'Optimizations' len=432 | Torch-Struct aims for computational and memory efficiency. Implemented naively, dynamic programming algorithms
  [11] 'c) Semiring Matrix Operations' len=887 | The two previous optimizations reduce most of the cost to semiring matrix multiplication. In the specific case
  [12] 'Related Work' len=974 | Several software libraries target structured prediction. Optimization tools, such as SVM-struct , focus on par

--- (b) 事实清单（0 条）---

--- (c) 成稿 ---
提供的证据中没有关于实验所用基线（baselines）的信息。各条目主要介绍 Torch-Struct 的技术方法（CRF、半环、并行扫描、向量化解析等）[1][2][5][7]、库设计[6]、动机案例[4]、优化[10][11]及相关工作[12]，均未列出实验中对比的基线模型。唯一提到"baseline"一词的是潜在树 RL 模型中的方差缩减基线（self-critical baseline）[4]，但那是方法内部的基线项，并非实验对比的基线。因此，证据不足以回答实验中使用了哪些基线。

==========================================================================================
## [6] 05671d0686 [hard] off_score=3
Q: What is the performance proposed model achieved on MathQA?
gold: Operation accuracy: 71.89
Execution accuracy: 55.95
evidence: Given a natural-language math problem, we need to generate a sequence of operations (operators and corresponding arguments) from a set of operators and arguments to solve the given problem. Each operation is regarded as a relational tuple by viewing the operator as relation, e.g., $(add, n1, n2)$. W

--- (a) 检索到 12 块 ---
  [1] 'EXPERIMENTS' len=461 | The proposed TP-N2F model is evaluated on two N2F tasks, generating operation sequences to solve math problems
  [2] 'Generating operation sequences to solve math problems' len=1749 | Given a natural-language math problem, we need to generate a sequence of operations (operators and correspondi
  [3] 'Implementations of TP-N2F for experiments' len=3830 | In this section, we present details of the experiments of TP-N2F on the two datasets. We present the implement
  [4] 'Generating program trees from natural-language descriptions' len=1654 | Generating Lisp programs requires sensitivity to structural information because Lisp code can be regarded as t
  [5] 'Interpretation of learned structure' len=1079 | To interpret the structure learned by the model, we extract the trained unbinding relation vectors from the TP
  [6] 'Unbinding relation vector clustering' len=603 | We run K-means clustering on both datasets with $k = 3,4,5,6$ clusters and the results are displayed in Figure
  [7] 'INTRODUCTION' len=2073 | When people perform explicit reasoning, they can typically describe the way to the conclusion step by step via
  [8] 'TP-N2F Model' len=1914 | We propose a general TP-N2F neural network architecture operating over TPRs to solve N2F tasks under a propose
  [9] 'Role-level description for relational representations' len=3208 | In this section, we propose a novel recursive role-level description for representing symbolic relational tupl
  [10] 'CONCLUSION AND FUTURE WORK' len=824 | In this paper we propose a new scheme for neural-symbolic relational representations and a new architecture, T
  [11] "The tensor that is input to the decoder's Unbinding Module is a TPR" len=2679 | Here we show that, if learning is successful, the order-3 tensor $$ that each iteration of the decoder's Tuple
  [12] 'Related work' len=1058 | N2F tasks include many different subtasks such as symbolic reasoning or semantic parsing , , , , , . These tas

--- (b) 事实清单（15 条）---
  (1) TP-N2F achieves state-of-the-art performance on two N2F tasks, generating operation sequences to solve math problems and generating Lisp programs.
  (2) TP-N2F is evaluated on the MathQA dataset, which consists of about 37k math word problems.
  (2) There are about 30% noisy data in MathQA (where the execution script returns the wrong answer when given the ground-truth program).
  (2) We report both execution accuracy (of the final multi-choice answer after running the execution engine) and operation sequence accuracy (where the generated operation sequence must match the ground truth sequence exactly).
  (2) TP-N2F is compared to a baseline provided by the seq2prog model, an LSTM-based seq2seq model with attention.
  (2) Our model outperforms both the original seq2prog, designated SEQ2PROG-orig, and the best reimplemented seq2prog after an extensive hyperparameter search, designated SEQ2PROG-best.
  (2) Ablation experiments replace either the encoder with a standard LSTM (denoted LSTM2TP) or the decoder with a standard attentional LSTM (denoted TP2LSTM).
  (3) The MathQA dataset consists of about 37k math word problems ((80/12/8)% training/dev/testing problems).
  (3) During our experiments we observed that there are about 30% noisy examples (on which the execution script fails to get the correct answer on the ground truth program).
  (3) We report both execution accuracy (the final multi-choice answer after running the execution engine) and operation sequence accuracy (where the generated operation sequence must match the ground truth sequence exactly).
  (3) In the experiment on the MathQA dataset, we use n_F= 150, n_R= 50, d_F= 30, d_R= 20, d_Rel = 20, d_Arg = 10, d_Pos = 5 and we train the model for 60 epochs with learning rate 0.00115.
  (3) For the baseline SEQ2PROG-orig, TP2LSTM and LSTM2TP, we use hidden size 100, single-direction, one-layer LSTM.
  (3) For the SEQ2PROG-best, we performed a hyperparameter search on the hidden size for both encoder and decoder; the best score is reported.
  (5) With 5 clusters in the MathQA dataset, arithmetic operators such as add, subtract, multiply, divide are clustered together, and operators related to square or volume of geometry are clustered together.
  (6) In the MathQA dataset, arithmetic operators such as add, subtract, multiply, divide are clustered together at middle, and operators related to geometry such as square or volume are clustered together at bottom left.

--- (c) 成稿 ---
你问的是 TP-N2F 在 MathQA 上的性能。论文明确说明 TP-N2F 在该任务上达到 state-of-the-art，并优于原始 seq2prog（SEQ2PROG-orig）以及经超参搜索后最佳重实现的 seq2prog（SEQ2PROG-best）[1][2]。评测报告两项指标：执行准确率（运行执行引擎后最终多选答案的准确率）和操作序列准确率（生成的操作序列须与真实序列完全一致）[2][3]。但具体数值在给出的正文条目中未出现（结果以 Table 形式呈现，表格内容未提供）[2]。

==========================================================================================
## [7] 9555aa8de3 [hard] off_score=3
Q: By how much does their model outperform both the state-of-the-art systems?
gold: w.r.t Rouge-1 their model outperforms by 0.98% and w.r.t Rouge-L their model outperforms by 0.45%
evidence: Evaluation on DUC2004: DUC 2004 ( BIBREF15 ) is a commonly used benchmark on summarization task consisting of 500 news articles. Each article is paired with 4 different human-generated reference summaries, capped at 75 characters. This dataset is evaluation-only. Similar to BIBREF2 , we train our ne

--- (a) 检索到 12 块 ---
  [1] 'Conclusion' len=731 | In this paper we have proposed two simple mechanisms to alleviate the problems of current encoder-decoder mode
  [2] 'Quantitative Evaluation' len=2250 | Results on Gigaword: We compare the performances of different architectures and report ROUGE scores in Tab. . 
  [3] 'Experimental Evalaluation' len=657 | In this section, we show results of abstractive summarization on Gigaword ( , ) and DUC2004 ( ) datasets. Our 
  [4] 'Summarization' len=2230 | In the past few years, there has been a lot of work on extractive summarization, where a summary is created by
  [5] 'Introduction' len=2894 | Encoder-decoder models have been widely used in sequence to sequence tasks such as machine translation ( , ). 
  [6] 'Encoder' len=1958 | Different from the GRU architecture, LSTM calculates the hidden state by applying a non-linear activation func
  [7] 'Decoder Vocabulary Size' len=1578 | Table shows the effect on our model of decreasing the decoder vocabulary size. We can see that when using the 
  [8] 'Encoder' len=3794 | We first review the typical encoder used in machine translation (e.g., , ). Let INLINEFORM0 be the input seque
  [9] 'The read again model' len=1160 | Text summarization can be formulated as a sequence to sequence prediction task, where the input is a longer te
  [10] 'Decoder with copy mechanism' len=2724 | In this paper we argue that only a small number of common words are needed for generating a summary in additio
  [11] 'Neural Machine Translation' len=364 | Our work is also closely related to recent work on neural machine translation, where neural encoder-decoder mo
  [12] 'Out-Of-Vocabulary and Copy Mechanism' len=1080 | Dealing with Out-Of-Vocabulary words (OOVs) is an important issue in sequence to sequence approaches as we can

--- (b) 事实清单（10 条）---
  (2) Our Read-Again models outperform the baselines on all ROUGE scores, when using both 15K and 69K sized vocabularies.
  (2) Even though the decoder vocabulary size of our approach with copy (15K) is much smaller than ABS (69K) and GRU (69K), it achieves a higher ROUGE score.
  (2) Our Multiple-Sentences model achieves the best performance.
  (2) our method outperforms all previous methods on Rouge-1 and Rouge-L, and is comparable on Rouge-2.
  (2) our model only uses 15k decoder vocabulary, while previous methods use 69k or 200k.
  (3) our copy mechanism can successfully reduce the typical vocabulary size by a factor 5 while achieving much better performance than the state-of-the-art, and by a factor of 30 while maintaining the same level of performance.
  (4) our experiments show state-of-the-art performance on the DUC competition.
  (5) We demonstrate the effectiveness of our approach in the challenging Gigaword dataset and DUC competition showing state-of-the-art performance.
  (7) when using the copy mechanism, we are able to reduce the decoder vocabulary size from 69K to 2K, with only 2-3 points drop on ROUGE score.
  (7) reducing vocabulary size dramatically reduces the decoding time from 0.38 second per sentence to 0.08 second.

--- (c) 成稿 ---
你问的“their model 相对 state-of-the-art 系统的提升幅度”，对应论文中带 copy 机制的 Read-Again 模型与 SOTA 基线的对比。论文给出的量化说法是：copy 机制可将典型词表规模缩小 5 倍，同时性能远优于 state-of-the-art；缩小 30 倍时仍能保持同等性能水平 [3]。在 Gigaword 上，其带 copy 的模型即使解码词表仅 15K（远小于 ABS 的 69K 和 GRU 的 69K），ROUGE 分数仍更高 [2]；在 DUC2004 上，该方法在 Rouge-1 和 Rouge-L 上优于所有先前方法，Rouge-2 上相当，且仅用 15k 解码词表，而先前方法用 69k 或 200k [2]。不过论文并未给出“具体高出多少 ROUGE 点”的数值，只报告了上述倍数与词表规模对比 [2][3]。

==========================================================================================
## [8] 1d74fd1d38 [hard] off_score=3
Q: What is their f1 score and recall?
gold: F1 score and Recall are 68.66, 80.08 with Traditional NERs as reference and 59.56, 69.76 with Wikipedia titles as reference.
evidence: FLOAT SELECTED: Table 2. Comparison with Traditional NERs as reference

--- (a) 检索到 7 块 ---
  [1] 'Results' len=1483 | Comparison with existing open-source NER libraries is not exactly fair as they are NOT trained for detecting t
  [2] 'Training' len=1032 | Trained the topic model on single 32gb NVidia-V100 and it took around 50 hours to train the model with sequenc
  [3] 'Discussions' len=3337 | Lets check some examples for detailed analysis of the models and their results. Following is the economy relat
  [4] 'Introduction & Related Work' len=2502 | Named-Entity-Recognition(NER) approaches can be categorised broadly in three types. Detecting NER with predefi
  [5] 'Data Preparation' len=2555 | We need good amount of data to try deep learning state-of-the-art algorithms. There are lot of open datasets a
  [6] 'Model Architecture' len=1646 | We tried multiple variations of LSTM and GRU layes, with/without CRF layer. There is a marginal gain in using 
  [7] 'Conclusion and Future Work' len=1268 | Through this exercise, we were able to test out the best suitable model architecture and data preparation step

--- (b) 事实清单（8 条）---
  (1) In terms of F1 and overall stability Spacy did better than Stanford NER, on our News Validation set.
  (1) Stanford did well in Precision but could not catch up with Spacy and our model in terms of Recall.
  (1) recall is great for our model but precision is not good
  (1) In capturing Wikipedia topics our model clearly surpasses other models in all scores.
  (2) We stopped the training for both the models when it crossed 70% precision, 90% recall on training and testing sets
  (2) Both the models reach this point at around 16 epochs.
  (5) in future we are planning to use whole dataset and hope to see gains in F1 and Recall further
  (4) Detecting caseless topics with higher recall and reasonable precision has been given a priority over f1 score.

--- (c) 成稿 ---
你问的“their f1 score and recall”对应论文中作者自己模型（our model）的 F1 与召回率。证据中并未给出该模型 F1 和召回率的具体数值：论文只说明训练在训练集和测试集上达到 70% 精确率、90% 召回率时停止 [2]，并强调其召回率高、精确率不佳 [1]，且在 Wikipedia 主题匹配任务上所有指标均超过其他模型 [1]。因此，除“召回率约 90%（训练/测试停止点）”外，F1 的具体分数在给定证据中未提供。

==========================================================================================
## [9] 5a23f436a7 [hard] off_score=3
Q: How big is dataset used?
gold: 553,451 documents
evidence: FLOAT SELECTED: Table 1: Statistics of the datasets.

--- (a) 检索到 12 块 ---
  [1] 'Experiments' len=507 | We use the public financial news dataset released by , which is crawled from Reuters and Bloomberg over the pe
  [2] 'Predicting Individual Stock Movements' len=1711 | Other than predicting the S&P 500 index, we also investigate the effectiveness of our approach on the problem 
  [3] 'Study on Trading Sequence Length' len=661 | We use the development set to find a suitable length $T$ for trading sequence, which is searched from $\lbrace
  [4] 'Method' len=1072 | The framework of our model is shown in Figure . We explicitly model both events and noise over a recurrent sto
  [5] 'Related Work' len=1912 | There has been a line of work predicting stock markets using text information from daily news. We compare this
  [6] 'Modeling News Events' len=1827 | For a trading day $t$ in a trading sequence, we model both long-term and short-term impact of news events. For
  [7] 'Training Objective' len=649 | For training, there are two main terms in our loss function. The first term is a cross entropy loss for the pr
  [8] 'LSTM-based Recurrent State Transition' len=734 | ANRES uses LSTM with peephole connections . The underlying stock value trends are represented as a recurrent s
  [9] 'Task Definition' len=381 | Following previous work , , the task is formalized as a binary classification task for each trading day. Forma
  [10] 'Settings' len=717 | The hyper-parameters of our ANRES model are shown in Table . We use mini-batches and stochastic gradient desce
  [11] 'Predicting S&P 500 Index' len=1497 | We compare our approach with the following strong baselines on predicting the S&P 500 index, which also only u
  [12] 'Case Study' len=1596 | To look into what news event contributes the most to our prediction result, we further analyze the test set re

--- (b) 事实清单（11 条）---
  (1) We use the public financial news dataset released by , which is crawled from Reuters and Bloomberg over the period from October 2006 to November 2013.
  (1) We conduct our experiments on predicting the Standard & Poor’s 500 stock (S&P 500) index and its selected individual stocks, obtaining indices and prices from Yahoo Finance.
  (1) Detailed statistics of the training, development and test sets are shown in Table .
  (2) We count the amounts of individual company related news events for each company by name matching, and select five well known companies with sufficient news, Apple, Citigroup, Boeing Company, Google and Wells Fargo from four different sectors, which is classified by the Global Industry Classification Standard.
  (2) The experimental results and news statistics are listed in Table .
  (3) We use the development set to find a suitable length $T$ for trading sequence, which is searched from $\lbrace 1, 3, 5, 7, 9, 11, 13, 15\rbrace $.
  (3) we choose the hyper-parameter $T=7$ and use it in the remaining experiments.
  (4) we extended the time span of one sample to $T$ previous continuous trading days (${t-T+1, t-T+2, ..., t-1, t}$), which we call a trading sequence, is used as the basic training element in this paper.
  (6) for long-term impact, we use the news published no more than thirty calendar days ago as the past news set.
  (12) Note that there are almost fifty different past news events in total for the trading sequence, and the news events listed on the left table are selected by ranking attention scores from the past news events, which are the most effective news according to the ablation study.
  (12) we take the example trading sequence from 07/15/2013 to 07/23/2013 for illustration.

--- (c) 成稿 ---
你问的数据集规模，论文中并未给出具体数值。证据只说明该数据集是公开的金融新闻数据集，抓取自 Reuters 和 Bloomberg，时间跨度为 2006 年 10 月至 2013 年 11 月 [1]；训练集、开发集和测试集的详细统计被放在 Table 中，但正文未列出具体数字 [1]。此外，论文提到对每家公司按名称匹配统计相关新闻事件数量，并选取了 Apple、Citigroup、Boeing、Google、Wells Fargo 五家公司，其新闻统计同样列在 Table 中，正文也未给出数值 [2]。因此，缺少的是 Table 中训练/开发/测试集以及各公司新闻数量的具体统计数字。

==========================================================================================
## [10] 81064bbd0a [hard] off_score=3
Q: By how much is precission increased?
gold: ROUGE-1 increases by 0.05, ROUGE-2 by 0.06 and ROUGE-L by 0.09
evidence: FLOAT SELECTED: Figure 3: ROUGE Quality of produced summaries in term of precision.

--- (a) 检索到 12 块 ---
  [1] 'Results and Discussion' len=1810 | In section () MSE, R2 and ROUGE scores are remarked as evaluation measures. The results of our experiments are
  [2] 'Evaluation Measures' len=1732 | In this section, some measures are described to evaluate the performance of both phases explained in the previ
  [3] 'Introduction' len=3063 | From the early days of artificial intelligence, automatically summarizing a text was an interesting task for m
  [4] 'Training Model' len=1394 | A vast collection of scikit-learn tools were used for the learning phase. K-fold cross-validation is applied w
  [5] 'Document-aware Features' len=2130 | Cosine position: As mentioned in () a good definition of position should take into account document length. A 
  [6] 'Document-unaware Features' len=3643 | Ordinal position: It is shown that inclusion of sentence, in summary, is relevant to its position in the docum
  [7] 'Target Assignment' len=796 | Every feature vector needs a target value from which the system should learn how to rank sentences. The value 
  [8] 'Target Assignment' len=828 | In assigning the target to a sentence, as mentioned in section (), the goal is to assign a number between 0 an
  [9] 'Extracting Features and Scaling' len=438 | All features introduced in section are calculated. Pre-processing, sentence and word tokenization, stop words 
  [10] 'Summarization Phase' len=415 | Having acquired a model that can precisely rank sentences, we can apply it to any new given text and use ranke
  [11] 'Sentence Selection' len=720 | By sorting sentences based on their ranks, the most appropriate sentences for being included in summary will b
  [12] 'Incorporating Document Features' len=928 | As a way to investigate the need for document features in sentence ranking (as explained in the introduction a

--- (b) 事实清单（14 条）---
  (1) ROUGE scores are displayed separately in terms of precision, recall and f-measure in Figures to respectively.
  (1) Figures and allow comparison of precision and recall scores.
  (1) The higher values gained in experiment 2, confirm that document-aware features perform better than unaware features.
  (1) Another insight gained from these charts is that a random summarizer resulted in scores more than 50% in all measures, and without using document-aware features, the model achieves a small improvement over a random summarizer.
  (2) ROUGE is proposed in as an evaluation metric for summaries. It matches n-grams in both system produced summaries and reference summaries and returns the percentage of matching in terms of precision, recall and f-measure.
  (2) There is a variety of ROUGE family metrics, namely ROUGE-1, ROUGE-2, and ROUGE-L.
  (2) In ROUGE-1 the overlap of 1-grams, each word, is calculated.
  (2) In ROUGE-2 the bigrams are considered as units of comparison.
  (2) The ROUGE-L uses the Longest Common Subsequence (LCS) to measure resemblance.
  (2) Nevertheless, we found that ROUGE assessments are always relatively high, even for a summary that is produced perfunctorily.
  (2) Hence, we also designed a random summarizer that selects random sentences for the summary, and evaluated it by ROUGE. This could be used as a baseline for comparison.
  (4) To compare with each standard summary, a summary with the same count of sentences was produced, and compared by ROUGE.
  (4) Averaging these ROUGE scores over each document and then over the dataset, the overall quality of summaries produced by the model can be obtained.
  (4) Apart from measuring the performance of this regressor on the test set, the quality of summaries produced is evaluated and reported as a baseline.

--- (c) 成稿 ---
证据中没有给出精确率（precision）提升的具体数值。论文只说明实验2中均方误差降低、R² 提高，且文档感知特征（document-aware features）比非感知特征表现更好，ROUGE 的精确率、召回率和 F 值分别以图的形式展示 [1][2]，但未在正文中报告精确率提升的幅度。因此无法给出具体数值，缺少的是精确率提升的量化数据。

==========================================================================================
## [11] 79bb1a1b71 [hard] off_score=3
Q: What accuracy does CNN model achieve?
gold: Combined per-pixel accuracy for character line segments is 74.79
evidence: FLOAT SELECTED: TABLE IV: Class-wise average IoUs and per-pixel accuracies on the test set. Refer to Table I for full names of abbreviated region types listed at top of the table.

--- (a) 检索到 12 块 ---
  [1] 'Indic Manuscript Layout Parsing' len=1067 | To succeed at layout parsing of manuscripts, we require a system which can accurately localize various types o
  [2] 'Training' len=2518 | The network is initialized with weights obtained from a Mask R-CNN trained on the MS-COCO dataset with a ResNe
  [3] 'Results' len=2074 | We report quantitative results using the measures described in Section . Table reports Average Precision and T
  [4] 'Evaluation' len=1542 | For quantitative evaluation, we compute Average Precision (AP) for a particular IoU threshold, a measure widel
  [5] 'Implementation Details' len=499 | The dataset splits used for training, validation and test phases can be seen in Table . All manuscript images 
  [6] 'Inference' len=803 | During inference, the images are rescaled and processed using the procedure described at the beginning of the 
  [7] 'Network Architecture' len=2269 | The Mask-RCNN architecture contains three stages as described below (see Figure ). Backbone: The first stage, 
  [8] 'Related Work' len=3366 | A number of research groups have invested significant efforts in the creation and maintenance of annotated, pu
  [9] 'Introduction' len=3152 | The collection and analysis of historical document images is a key component in the preservation of culture an
  [10] 'Annotation Tool' len=744 | Keeping the aforementioned challenges in mind, we introduce a new browser-based annotation tool (see Figure ).
  [11] 'Annotation Challenges' len=2069 | A variety of unique challenges exist in the context of annotating Indic manuscript layouts. The challenges ari
  [12] 'Conclusion' len=1507 | Via this paper, we propose Indiscapes, the first dataset with layout annotations for historical Indic manuscri

--- (b) 事实清单（22 条）---
  (3) Table reports Average Precision and Table reports class-wise average IOUs and per-pixel accuracies.
  (3) Holes, Character line segments, Page boundary and Pictures are parsed the best while Physical degradations are difficult to parse.
  (3) performance for Penn in Hand (PIH) documents is better compared to Bhoomi manuscripts.
  (4) we compute Average Precision (AP) for a particular IoU threshold
  (4) We specifically report $AP_{50}$ and $AP_{75}$, corresponding to AP at IoU thresholds 50 and 75 respectively
  (4) we report an overall score by averaging AP at different IoU thresholds ranging from $0.5$ to $0.95$ in steps of $0.05$.
  (4) we report two additional measures – average class-wise IoU (cwIoU) and average class-wise per-pixel accuracy (cwAcc).
  (4) The per-class IoU score for class $i$ and document $k$ is computed as ${cwIoU}^d_i = \frac{\sum _r {IoU}_r}{r_i}$.
  (4) The overall per-class IoU score for class $i$ is computed as ${cwIoU}_i = \frac{\sum _d {cwIoU}^d_i}{N_i}$.
  (4) we define class-wise pixel accuracy ${pwAcc}^d_i$ at document level and average it across all the documents containing class $i$, i.e. ${cwAcc}_i = \frac{\sum _d {pwAcc}^d_i}{N_i}$.
  (6) we choose the top 100 region detections with objectness score exceeding $0.5$
  (6) The generated masks are then binarized using an empirically chosen threshold of $0.4$
  (6) NMS with a threshold value of $0.5$ is applied to obtain the final set of predicted masks.
  (2) The weighting factors ($\lambda $s) are set to 1. However, to ensure priority for our task of interest namely mask prediction, we set $\lambda _{mask}=2$.
  (2) we use Stochastic Gradient Descent (SGD) optimizer with a gradient norm clipping value of $0.5$.
  (2) The batch size, momentum and weight decay are set to 1, $0.9$ and $10^{-3}$ respectively.
  (2) For the first stage (30 epochs), we train only the task branch sub-networks using a learning rate of $10^{-3}$
  (2) For the second stage (20 epochs), we additionally train stage-4 and up of the backbone ResNet-50.
  (2) For the final stage (15 epochs), we train the entire network using a learning rate of $10^{-4}$.
  (5) All manuscript images are adaptively resized to ensure the width does not exceed 1024 pixels.
  (5) the input to the deep network has spatial dimensions of $1024 \times 1024$.
  (5) they are downsized to $28 \times 28$ in order to match output dimensions of the mask sub-network.

--- (c) 成稿 ---
你问的“CNN model 的 accuracy”对应论文中的 Mask R-CNN 实例分割模型，但论文并未给出一个单一的“CNN 准确率”数值。论文报告的是实例分割指标：AP（含 AP₅₀、AP₇₅ 及 IoU 0.5–0.95 步长 0.05 的平均）[4]，以及按类别统计的平均 IoU（cwIoU）和平均逐像素准确率（cwAcc）[4]。定性上，Holes、Character line segments、Page boundary 和 Pictures 解析效果最好，Physical degradations 最难解析[3]。因此，证据中没有任何具体的准确率数字，缺的是论文正文未提供的具体 AP/cwAcc 数值（这些数值在正文引用的 Table 中，未在给定条目里给出）。

==========================================================================================
## [12] b42323d608 [hard] off_score=3
Q: How big is the evaluated dataset?
gold: contains thousands of XML files, each of which are constructed by several records
evidence: The DDI corpus contains thousands of XML files, each of which are constructed by several records. For a sentence containing INLINEFORM0 drugs, there are INLINEFORM1 drug pairs. We replace the interested two drugs with “drug1” and “drug2” while the other drugs are replaced by “durg0”, as in BIBREF9 d

--- (a) 检索到 12 块 ---
  [1] 'Datasets and Evaluation Metrics' len=972 | We use the DDI corpus of the 2013 DDIExtraction challenge to train and test our model. The DDIs in this corpus
  [2] 'Sentence Level Attention' len=1424 | The previous layers captures the features only from the given sentence. However, other sentences may contains 
  [3] 'Hyperparameter Settings and Training' len=2206 | We use TensorFlow r0.11 to implement the proposed model. The input of each word is an ordered triple (word, re
  [4] 'Embedding Layer' len=887 | Given an instance INLINEFORM0 which contains specified two drugs INLINEFORM1 , INLINEFORM2 , each word is embe
  [5] 'Acknowledgment' len=86 | This work is supported by the NSFC under Grant 61303191, 61303190, 61402504, 61103015.
  [6] 'Classification and Training' len=1054 | A given sentence INLINEFORM0 is finally represented by the feature vector INLINEFORM1 . Then we feed it to a s
  [7] 'Experimental Results' len=2877 | We save our model every 100 step and predict all the DDIs of the instances in the testing set. These predictio
  [8] 'Introduction' len=2329 | Drug-drug interaction (DDI) is a situation when one drug increases or decreases the effect of another drug . A
  [9] 'Word Level Attention' len=934 | The purpose of word level attention layer is to extract sentence representation (also known as feature vector)
  [10] 'Bidirectional RNN Encoding Layer' len=1470 | The words in the sequence are read by RNN's gated recurrent unit (GRU) one by one. The GRU takes the current w
  [11] 'DDI Prediction' len=554 | The model is trained for DDI classification. The parameters in list INLINEFORM0 are tuned during the training 
  [12] 'Preprocessing' len=990 | The DDI corpus contains thousands of XML files, each of which are constructed by several records. For a senten

--- (b) 事实清单（7 条）---
  (1) We use the DDI corpus of the 2013 DDIExtraction challenge to train and test our model.
  (1) The DDIs in this corpus are classified as five types.
  (1) This standard dataset is made up of training set and testing set.
  (12) The DDI corpus contains thousands of XML files, each of which are constructed by several records.
  (12) For a sentence containing INLINEFORM0 drugs, there are INLINEFORM1 drug pairs.
  (8) There are 3 drug pairs in this example thus we have 3 instances.
  (12) the sentence in figure generates 3 instances after drug blinding

--- (c) 成稿 ---
你问的“evaluated dataset”对应论文中用于训练和测试的 DDI corpus（2013 DDIExtraction challenge 数据集）[1]。证据中只说明该语料库由训练集和测试集构成[1]，并包含数千个 XML 文件、每个文件由若干记录组成[12]，但**没有给出该数据集的具体规模数值**（如句子数、实例数或文件总数）。因此，论文提供的证据无法回答“数据集有多大”这一具体数值问题，缺少的是数据集大小的明确数字。
