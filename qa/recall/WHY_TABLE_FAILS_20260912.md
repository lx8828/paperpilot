# 为什么「表进了上下文却答不出」— 5 题逐条诊断（2026-09-12）

## 9c4a4dfa（1910.02339）score 2 → 3

- **问题**：How does TP-N2F compare to LSTM-based Seq2Seq in terms of training and inference speed?
- **金标准**：Full Testing Set accuracy: 84.02 Cleaned Testing Set accuracy: 93.48
- **目标表块**：['xtbl-113'] | 在 q2 上下文里的位次：14（共 14 块）
- 表块 `xtbl-113` 内容（前 700 字）：

```
Table 2. Results of AlgoLisp dataset
| MODEL | Acc (%) | 50p-Acc (%) | M-Acc (%) | Acc (%) | 50p-Acc(%) | M-Acc (%) |
|---|---|---|---|---|---|---|
| Seq2Tree | 61.0 |  |  |  |  |  |
| LSTM2LSTM+atten | 67.54 | 70.89 | 75.12 | 76.83 | 78.86 | 75.42 |
| TP2LSTM (ours) | 72.28 | 77.62 | 79.92 | 77.67 | 80.51 | 76.75 |
| LSTM2TPR (ours) | 75.31 | 79.26 | 83.05 | 84.44 | 86.13 | 83.43 |
| SAPSpre-VH-Att-256 | 83.80 | 87.45 |  | 92.98 | 94.15 |  |
| TP-N2F (ours) | 84.02 | 88.01 | 93.06 | 93.48 | 94.64 | 92.78 |
```
- **q0（现状）**：score=2 cites=['c0014', 'c0015', 'c0019']（其中表块：无）
  - 答案：提供的证据中没有关于 TP-N2F 与 LSTM-based Seq2Seq 在训练速度或推理速度上的任何比较数据。各条目只报告了准确率类指标（如 MathQA 的 execution accuracy 与 operation sequence accuracy [1][5]、AlgoLisp 的 Acc/50p-Acc/M-Acc [4][5]），以及模型结构、超参数和训练轮数/学习率 [5]，并未给出任何速度、耗时或吞吐量数值。因此无法回答该问题，缺少的是训练/推理时间或速度的对比结果。
  - 裁判：系统回答正确地指出了提供的证据中没有关于训练和推理速度的比较数据，但未提及标准答案中提供的准确率信息，这表明对问题的理解有偏差。
- **q2（并集）**：score=3 cites=['c0012', 'c0019', 'xtbl-103', 'xtbl-113']（其中表块：['xtbl-103', 'xtbl-113']）
  - 答案：提供的证据中没有关于 TP-N2F 与 LSTM-based Seq2Seq 在训练速度或推理速度上的任何比较数据。各条目只报告了准确率类指标（如 MathQA 上的 Operation Accuracy 与 Execution Accuracy [13]，AlgoLisp 上的 Acc、50p-Acc、M-Acc [14]），以及模型结构与训练设置（如 Adam 优化器、teacher-forcing、学习率与 epoch 数 [3][5]），并未提及训练时长、推理耗时或速度对比。因此，关于训练与推理速度的比较，证据不足，无法作答。
  - 裁判：系统回答指出了证据中缺乏关于训练和推理速度的比较数据，这是正确的。然而，它没有提及TP-N2F和LSTM-based Seq2Seq在训练和推理速度上的具体比较，这是标准答案中提到的关键信息。

## 75b69eef（2002.05058）score 3 → 3

- **问题**：How much better peformance is achieved in human evaluation when model is trained considering proposed metric?
- **金标准**：Pearson correlation to human judgement - proposed vs next best metric Sample level comparison: - Story generation: 0.387 vs 0.148 - Dialogue: 0.472 vs 0.341 Model level comparison: - Story generation:  0.631 vs 0.302 - Dialogue: 0.783 vs 0.553
- **目标表块**：['xtbl-66'] | 在 q2 上下文里的位次：13（共 14 块）
- 表块 `xtbl-66` 内容（前 700 字）：

```
Table 1: Sample-level correlation between metrics and human judgments, with p-values shown in brackets. Table 2: Model-level correlation between metrics and human judgments, with p-values shown in brackets.
| Task Story Generation | Dialogue |
|---|---|
| Metric | Spearman | Pearson | Spearman | Pearson |
| our approach |  |  |  |  |
| - with skill rating | 0.612(< 0.001) | 0.631(< 0.001) | 0.764(< 0.001) | 0.783(< 0.001) |
| - averaged sample-level skill rating | 0.518 | 0.541 | 0.651 | 0.675 |
| - averaged reference-based score | 0.473 | 0.482 | 0.634 | 0.653 |
| ADEM | 0.291 | 0.302 | 0.541 | 0.553 |
| Adversarial Evaluator | 0.248 | 0.272 | 0.491 | 0.502 |
| BLEU | 0.096 | 0.103 | 0.217 
```
- **q0（现状）**：score=3 cites=['c0016', 'c0019']（其中表块：无）
  - 答案：你问的是“用所提指标指导训练后，人类评估表现提升多少”。论文并未给出一个可直接量化的“提升幅度”数值：它报告的是在 10 次运行中，各指标成功选出最佳超参数组合或早停检查点的次数（满分 10，候选 4 个）以及所选模型的人类评分，但正文未列出这些具体数字 [1]。论文只说明，把评估指标从 BLEU 或困惑度换成所提比较评估器，能在不改变模型架构或训练目标的情况下带来“不可忽视的改进”（non-neglectable improvements）[1]，并称所提模型在这些选择上“可靠得多” [2]。因此，具体提升数值需查 Table 3，而该表内容未在给定证据中给出 [1]。
  - 裁判：系统回答正确地指出了论文中未给出具体提升幅度的数值，并引用了相关内容。然而，它没有提及具体的数据对比，如0.387 vs 0.148等，这是标准答案中明确提到的关键信息。
- **q2（并集）**：score=3 cites=['c0016', 'c0019', 'c0013', 'xtbl-77']（其中表块：['xtbl-77']）
  - 答案：你问的是“用所提指标指导训练后，在人工评测上取得多大提升”，对应论文中关于用不同指标做超参搜索与早停选择、再以人工评测衡量所选模型表现的实验。论文指出，传统自动指标（BLEU、perplexity）在此类选择中表现很差、导致次优结果，而把评测指标换成所提比较式评估器可带来“不可忽视的改进”，且无需改变模型结构或训练目标 [1][2]。但证据中只给出了该实验的设定（10 次运行、每次从 4 个候选中选最佳、按成功次数和所选模型的人工评分均值衡量）[1][10]，以及表 3 的标题 [14]，**并未给出具体的提升数值**（如成功次数差或人工评分差）。因此无法给出具体数字，缺的是表 3 的实际数值内容。
  - 裁判：系统回答正确地指出了实验设计和结果，但遗漏了具体的提升数值，这是关键信息。

## 579941de（1909.13375）score 3 → 3

- **问题**：What is difference in peformance between proposed model and state-of-the art on other question types?
- **金标准**：For single-span questions, the proposed LARGE-SQUAD improve performance of the MTMSNlarge baseline for 2.1 EM and 1.55 F1. For number type question,  MTMSNlarge baseline  have improvement over LARGE-SQUAD for 3,11  EM and  2,98 F1.  For date question,  LARGE-SQUAD have improvements in 2,02 EM but MT
- **目标表块**：['xtbl-78'] | 在 q2 上下文里的位次：14（共 14 块）
- 表块 `xtbl-78` 内容（前 700 字）：

```
Table 2: Example failure cases of TASEBIO+SSE on DROP. The first answer exhibits a lack of numeric reasoning and ignores the expected number of spans stated in the question. The second splits a correct span into two spans.
| Question | Excerpt from Context | Gold Answer | Prediction |
|---|---|---|---|
| Which two nationalities have the same number of immigrants in Bahrain? | Indians, 125,000 Bangladeshis, 45,000 Pakistanis, 45,000 Filipinos, and 8,000 Indonesians | $\{ ^ { \ast } \mathrm { F i l i p i n o s } ^ { \ast } ,$ $\mathrm { \tilde { \ r p } a k i s i a n i s ^ { , } } \}$ | $\mathrm { \{ ^ { * } F i l i p i n o s ^ { * } , } $ $\mathrm { \tilde { \hbar } P a k i s t a n i s ^ { \p
```
- **q0（现状）**：score=3 cites=['c0025', 'c0022', 'c0023']（其中表块：无）
  - 答案：你问的“其他问题类型（other question types）”上的性能差异，对应论文中与当时 SOTA（MTMSN）在单跨度（single-span）问题上的比较。论文指出，其最佳模型在单跨度性能上略优于 MTMSN [8]；但为公平比较而移除单跨度头后，单跨度性能反而略降，落后 MTMSN 0.8 EM 和 0.6 F1 [8]。此外，论文称其模型在整体 DROP 数据集上略微超越当时的 SOTA [1][10]。至于“其他类型”中数值推理等具体类型的逐项差值，证据中未给出具体数字。
  - 裁判：系统回答正确地提到了模型在单跨度问题上的性能，但遗漏了关于数值类型和日期类型问题的具体性能差异，这是标准答案中明确提到的关键信息。
- **q2（并集）**：score=3 cites=['c0025', 'c0022', 'c0023']（其中表块：无）
  - 答案：你问的“其他问题类型（other question types）”上的性能差异，对应论文中与当时 state-of-the-art（MTMSN）在单跨度（single-span）问题上的对比。论文指出，其最佳模型在单跨度性能上略优于 MTMSN [8]；但为公平比较，作者去掉了单跨度头（no-single-span-heads 设置），此时单跨度性能反而略降，比 MTMSN 落后 0.8 EM 和 0.6 F1 [8]。此外，论文称其模型在整个 DROP 数据集上略微超越了当时的 state-of-the-art 结果 [1][10]。因此，在“其他问题类型”上，论文提出的模型整体略占优势，但在去掉单跨度头后单跨度指标略逊于 MTMSN [8]。
  - 裁判：系统回答正确地提到了模型在多跨度问题上的优势，但在比较不同问题类型时，遗漏了具体数值和不同类型问题的比较细节。同时，系统回答中提到的“其他问题类型”并未明确指出具体是哪些类型，与标准答案中的具体问题类型（单跨度、数字类型、日期类型）存在差异。

## 0fd678d2（1804.08139）score 3 → 3

- **问题**：What evaluation metrics are used?
- **金标准**：Accuracy on each dataset and the average accuracy on all datasets.
- **目标表块**：['xtbl-86'] | 在 q2 上下文里的位次：不在候选（共 14 块）
- 表块 `xtbl-86` 内容（前 700 字）：

```
Table 2: Performances on 16 tasks. The column of “Single Task” includes bidirectional LSTM (BiLSTM), bidirectional LSTM with attention (att-BiLSTM) and the average accuracy of the two models. The column of “Multiple Tasks” shows several multi-task models. \* is from [Liu et al., 2017] .
| Task | Single Task | Multiple Tasks |  |
|---|---|---|---|
| BiLSTM | att-BiLSTM | Avg. | FS-MTL | SSP-MTL | PSP-MTL | ASP-MTL* | SA-MTL | DA-MTL |
| Books | 81.0 | 82.0 | 81.5 | 84.0 | 85.5 | 85.5 | 87.0 |  | 86.8 | 88.5 |
| Electronics | 81.8 | 83.0 | 82.4 | 84.8 | 86.8 | 87.3 |  | 89.0 | 87.5 | 89.0 |
| DVD | 83.3 | 83.0 | 83.1 | 85.0 | 85.3 | 84.5 | 87.4 |  | 87.3 | 88.0 |
| Kitchen | 80.8 | 80.3 | 80.5
```
- **q0（现状）**：score=3 cites=['c0008']（其中表块：无）
  - 答案：提供的证据中没有提及任何评价指标（如准确率、F1 等）的名称。各实验仅报告了“performance”“accuracy”等笼统说法，例如 DA-MTL 相比 SA-MTL 提升了 INLINEFORM0 accuracy [1]，但未指明具体评测指标。因此，依据现有证据无法确认论文使用了哪些评价指标。
  - 裁判：系统回答指出了论文中未明确提及具体评价指标名称的问题，但未能准确引用原文中提到的准确率（accuracy）作为评价指标。
- **q2（并集）**：score=3 cites=['c0008']（其中表块：无）
  - 答案：根据提供的证据，论文中**没有明确说明**使用了哪些评价指标（如准确率、F1 等）。证据中只提到各任务被划分为训练/开发/测试集 [1]，并报告了不同方法的性能对比 [1]，但未给出具体指标名称。因此，评价指标这一信息在给定证据中缺失。
  - 裁判：系统回答指出了评价指标信息在给定证据中缺失，这是正确的。但是，它没有提及论文中提到的具体评价指标，如准确率。

## c4c9c790（2002.08307）score 4 → 3

- **问题**：How much is pre-training loss increased in Low/Medium/Hard level of pruning?
- **金标准**：The increase is linearly from lowest on average 2.0 , medium around 3.5, and the largest is 6.0
- **目标表块**：（该篇未产出/未匹配到目标表） | 在 q2 上下文里的位次：不在候选（共 13 块）
- **q0（现状）**：score=4 cites=['c0010', 'c0006']（其中表块：无）
  - 答案：你问的“Low/Medium/Hard 三档剪枝下预训练损失增加多少”，对应论文按剪枝比例（sparsity）给出的预训练损失数值。论文并未用“Low/Medium/Hard”命名档位，而是按 0%–90% 的稀疏度报告：低档（30–40%）不增加预训练损失，中档开始上升，高档大幅上升 [4]。  论文明确指出低水平剪枝（30–40%）完全不增加预训练损失，中等水平剪枝则使预训练损失上升 [4]；超过 40% 后性能开始退化、预训练损失随剪掉拟合预训练数据所需的权重而增加 [1]。
  - 裁判：系统回答正确地描述了预训练损失随剪枝比例的变化，但未直接给出 Low/Medium/Hard 档位对应的预训练损失数值，而是提供了论文中关于稀疏度的描述。
- **q2（并集）**：score=3 cites=['c0009', 'c0010']（其中表块：无）
  - 答案：你问的“Low/Medium/Hard level of pruning”对应论文中的低、中、高剪枝水平。论文明确给出：低水平剪枝（30–40%）完全不增加预训练损失 [3]；中等水平剪枝会提高预训练损失 [7]。论文未用“Hard”这一标签，但高剪枝（70% 以上）损失急剧上升。论文中无法确认 0%、60%、90% 等具体剪枝水平对应的预训练损失数值。
  - 裁判：系统回答正确地指出了不同剪枝水平对预训练损失的影响，但遗漏了具体数值，且未提及“Hard”剪枝水平的损失数值。

## 汇总

- q2 臂里**上下文含表块**的题：32/36
- 其中答案**引用了表块**的：17/32 = 53%

## 结论（2026-09-12，逐条人工核对后）

「召回变好但回答没变好」不是一个原因，是**五个独立障碍**：

1. **模型不会从表里读数值做对比（真瓶颈，铁证）** —— `75b69eef`：correlation 表含 gold 的 `0.631 vs 0.302`，它就在上下文中、**还被引用了** `xtbl-77`，模型仍写「**并未给出具体的提升数值**」。
   同类措辞在 `579941de`（「证据中未给出具体数字」）、`c4c9c790`（「无法确认具体数值」）、`0fd678d2`（「没有明确说明」）反复出现。
2. **误拒** —— `0fd678d2`：gold =「Accuracy on each dataset and the average accuracy」，而含 accuracy 的表（`xtbl-86`，16 tasks 性能表）**没进候选**、正文又没明说 → 系统答「论文没提评价指标」。
3. **表没被用** —— 汇总：**表块在上下文里时只有 17/32 = 53% 的答案引用了表块**，近一半白给。
4. **评测天花板** —— `9c4a4dfa`：问题问 **speed**，gold 给 **accuracy**（表里也只有 accuracy）
   → 系统诚实答「没有速度数据」反被判 3 分。**这题召回再好也答不对**，属 QASPER 标注与问题不匹配。
5. **MinerU 表体质量** —— `5a0841cc`(1701.00185) 的 Table 6 出现 `| bi-LSTM (last)bi-LSTM (mean)` 这种**行结构错乱**；
   另 `579941de` 确证 1 例**错表**（gold 证据指「Table 2. Performance ... EM and F1」，而该篇 MinerU 的 `xtbl-78` 是「Table 2: Example failure cases」→ 目标性能表根本没被产出）。

**自我纠错**：本轮一度用「表块是否含 gold 数字」估出假阳性 47%，**该指标不可用** ——
它把「gold 是聚合/推导值」（如 gold `over 104k documents` vs 表里 `83,077`）误判为「表错了」；
抽查 2 例均是对的表。真错表**至少 1 例**（579941de），**不是**系统性问题。

**指向**：瓶颈已从「召回」转移到「**读表**」——①表内容解析质量、②模型从表取数做对比的能力、③误拒倾向。与上下文压缩/重排无关。
