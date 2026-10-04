"""**facet 重建 v2**：把 19 个 facet 的题面改成**可判定的操作定义**，并收紧锚点。

## 为什么要重建（依据 `_r2_gold_adjudicate.py` 的 18 条分歧）
扩大抽检（独立第三判官 `deepseek-reasoner`，n=172）暴露：
· **分歧 100% 集中在「定义有歧义」的 facet 上**；
  定义最清晰的 `ablation` / `code_release` / `fine_tuning` **一条分歧都没有**。
· 典型歧义（新真值被误删的例子）：
    `efficiency`     —— 本文在 **Limitations** 里报了 2 examples/s，算不算「报告效率指标」？
    `significance`   —— 表注写 * denotes significant differences (p<0.05)，算不算？
    `multilingual`   —— 本文**自己在 TyDiQA/MGSM 上评测**，算不算「多语言实验」？
    `zero_few_shot`  —— 研究设置**就是** zero-shot cross-lingual，算不算？
    `context_length` —— 本文是 **Long-Range Transformer** 工作，算不算「处理长上下文」？
    `deployment`     —— 以「真实应用成本」为**动机**，算不算「面向真实部署」？
· 反向（误收）：`safety_bias` 只提 harmless answer；`case_study` 只有定量敏感性分析。

## 修法：题面 = 判定规则本身
每个 facet 的**中文题面**直接写成「算什么 ∧ 不算什么」，
判官只需按字面比照 → 边界不再靠自由裁量。
（提示词的四纪律**不改**，见 `_r2_gold_recalib.py::SYS`）

锚点正则同步收紧/放宽（只为**证据池召回**服务，**不再是真值来源**）：
· `ablation` → `\\bablat`（v1 漏了动词 ablate，实测误删过真阳性）
· `efficiency` → 明确「模型推理」
· `significance` → 加 std / p<0.0
· `context_length` → 加 long-range
"""
from __future__ import annotations

# facet → (锚点正则【仅用于证据池召回】, 中文题面=判定规则, 英文锚点词, 英文改写句)
F2: dict[str, tuple[str, str, str, str]] = {
    "ablation": (
        r"\bablat|remov(?:e|ing|ed) (?:each |every |one |the )?(?:component|module|part|feature)",
        "本文**报告了消融结果**：把**自己方法**的某个部分去掉/替换后，性能如何变化。"
        "（算：`we ablate` / `we remove X and observe` 并给出结果；"
        "不算：只引用他人的消融、只说「还需要消融」、只做超参敏感性）",
        "ablation study component analysis removing parts",
        "we remove or drop parts of our model and see how the results change"),

    "code_release": (
        r"github\.com|our code (?:is|will be) (?:publicly )?available|we (?:release|open[- ]source)",
        "本文提供了**作者自己**代码/模型/数据的可获取链接。"
        "（算：`our code is available at github.com/...`、`we release our data`；"
        "不算：引用他人的仓库（如 ALCE、FiD）、只列第三方工具（pytorch / OpenNMT））",
        "our code and data are publicly available github open source release",
        "our implementation can be downloaded so others can reproduce the results"),

    "fine_tuning": (
        r"fine[- ]tun|finetun",
        "本文**在自己数据上继续训练（微调）了模型**，并报告了该训练。"
        "（算：本文执行了 fine-tuning / finetune 并给出设置或结果；"
        "不算：只用现成模型、只引用他人微调）",
        "fine-tuning finetuning train the model",
        "we continue training the model on our own data"),

    "zero_few_shot": (
        r"zero[- ]shot|few[- ]shot|without (?:any )?training examples",
        "本文在**不给训练样本**（zero-shot）或**只给极少样本**（few-shot）的设置下**报告了结果**。"
        "（算：`zero-shot evaluation`、`few-shot setting`、`without training examples`、"
        "`a few examples given in the prompt`；"
        "不算：仅把 zero-shot 当作**对比基线**、仅提「零样本也能做」）",
        "zero-shot few-shot evaluation without training",
        "results when the model is given no examples or only a couple of them"),

    "human_eval": (
        r"human (?:evaluation|study|annotation|rating|assessment)|annotators|"
        r"manual (?:assessment|annotation|evaluation)",
        "本文**收集了人工判断/标注并报告其结果**（评估、标注、人工核对任一）。"
        "（算：`human evaluation`、`annotators were asked`、`manual assessment shows`；"
        "不算：引用他人的人工评估、只说「人工标注很贵」）",
        "human evaluation annotators human ratings",
        "people were asked to read the outputs and judge their quality"),

    "error_analysis": (
        r"error analysis|failure (?:case|analysis|mode)",
        "本文**检查了自己方法的失败/错误案例**，并给出**具体例子或分类统计**。"
        "（算：`error analysis`、`we show failure cases`；不算：只报总体分数）",
        "error analysis failure cases analysis",
        "we examine the situations where our method does not work"),

    "significance": (
        r"significan|p\s*[<=]\s*0\.0|\bstd\b|standard deviation|multiple (?:runs|seeds)",
        "本文报告了**多次运行的波动**或**显著性检验**："
        "标准差 / 方差（**来自重复运行或不同种子**），或 p 值 / 显著性星号标注。"
        "（算：`±std over 3 runs`、`p<0.05`、`* denotes significant differences`；"
        "不算：语义/方差**分解**、单次运行结果、只提「显著提升」而无检验）",
        "statistical significance multiple runs standard deviation",
        "we repeat each experiment several times and report the variation"),

    "case_study": (
        r"case stud|qualitative (?:analysis|example|study)",
        "本文对**少数具体输入/实例**做了**逐例展示与讨论**。"
        "（算：`case study`、`Table X shows examples` 并逐一分析；"
        "不算：只有定量的敏感性/归因/熵分析、只有总体 qualitative 声明）",
        "case study examples qualitative analysis",
        "we look closely at a few concrete examples from our system"),

    "multilingual": (
        r"multilingual|cross-lingual|non[- ]English",
        "本文在**非英语数据**上报告了实验结果。"
        "（算：在 TyDiQA / MGSM / XNLI 等非英数据上评测并给出数字；"
        "不算：只声称支持多语言、只引用他人的多语言结果）",
        "multilingual cross-lingual experiments in many languages",
        "we test the method in languages other than English"),

    "efficiency": (
        r"(?:inference|decoding|generation)[- ](?:time|latency|speed|cost)|throughput|FLOPs|"
        r"\bexamples?/s\b|\btokens?/s\b|GPU memory|peak memory",
        "本文报告了**模型推理本身**的效率数值：时间 / 延迟 / 吞吐 / FLOPs / 显存。"
        "（算：`inference time`、`latency`、`throughput`、`2 examples/s`、`FLOPs` —— "
        "**无论出现在哪一节**（含 Limitations / 附录）；"
        "不算：**人工标注速度**、训练成本、只定性说「更快」）",
        "inference latency throughput time cost efficiency",
        "how much time or memory it takes to run the model"),

    "context_length": (
        r"long[- ](?:context|range|sequence|input|document)|context (?:window|length)",
        "本文**以「让模型处理更长输入」为方法目标**，提出机制/改进，并在长输入上报告了效果。"
        "（算：Longformer / BigBird 类长上下文方法 + 长输入实验；"
        "不算：因长度限制而**砍掉任务**、只在数据描述里提到长序列、只泛泛说「上下文很长」）",
        "long context window length extension long-range transformers",
        "how the model behaves when the input becomes very large"),

    "deployment": (
        r"deploy(?:ed|ment|ing)?\b|production system|online serving|in production",
        "本文描述了**自己系统在真实产品/线上环境的实际部署**。"
        "（算：`we deployed our system in production`、`serving in production`；"
        "不算：以「真实应用成本」为**动机**、把实验设置改成「实际场景」、泛泛说「有实际应用价值」）",
        "deployment production real-world online serving",
        "we describe how the system is used in practice outside of research"),

    "new_dataset": (
        r"we (?:introduce|present|construct|release|collect|build) (?:a |the )?"
        r"(?:new )?(?:dataset|benchmark|corpus|data)",
        "本文**构建并发布了新的数据集/基准/语料**，且说明了其构造方式。"
        "（算：`we introduce X dataset`、`we collect and release`；"
        "不算：只用已有数据集、只提「未来可建」）",
        "we introduce a new dataset benchmark corpus",
        "a new collection of examples that others can use for training and testing"),

    "knowledge_distill": (
        r"distillation|distill(?:ing|ed)? (?:knowledge|the model)|teacher[- ]student",
        "本文**用「教师模型指导学生模型」作为自己的方法**，并报告了结果。"
        "（算：本文提出的蒸馏方法/蒸馏训练；不算：只引用他人的蒸馏工作）",
        "knowledge distillation teacher student",
        "a smaller model is taught by copying the behaviour of a bigger one"),

    "prompt_eng": (
        r"prompt (?:engineering|template|design|format)|few[- ]shot prompt",
        "本文**系统研究或设计提示模板**，并报告其对结果的影响（含模板对比/消融）。"
        "（算：prompt 模板的对比实验；不算：只使用了一个固定 prompt）",
        "prompt engineering template design few-shot",
        "how the wording and the format of the instruction affect the results"),

    "proof_theory": (
        r"\btheorem\b|\blemma\b|\bproof\b",
        "本文给出了**针对自己方法**的定理/引理/证明或理论保证。"
        "（算：本文提出并证明的命题；不算：引用他人定理、只在背景里提「有理论保证」）",
        "theoretical analysis theorem proof guarantee",
        "we provide a formal argument with mathematical statements"),

    "rl_training": (
        r"reinforcement learning|\bRLHF\b|\bPPO\b|policy (?:gradient|optimization)",
        "本文**用强化学习训练/对齐了自己的模型**（PPO / RLHF / 策略梯度），并报告了结果。"
        "（算：本文执行 RL 训练；不算：只引用他人 RL 工作、只提「可用 RL 改进」）",
        "reinforcement learning RLHF PPO policy optimization",
        "we improve the model using reward signals coming from feedback"),

    "safety_bias": (
        r"toxic|harmful|fairness|bias(?:ed)?\b|\bsafety\b",
        "本文**专门评估或讨论了输出的安全（有害内容）/ 公平 / 偏见问题**，并给出具体结果或分析。"
        "（算：专门一节或一组实验测 toxicity / fairness / bias；"
        "不算：只提「提供无害答案」这类泛泛表述、只在局限里提「可能有偏见」）",
        "safety toxicity fairness bias evaluation",
        "we examine whether the outputs are harmful or unfair to some groups"),

    "annotation_cost": (
        r"annotation cost|crowdsourc|cheap(?:er)? (?:label|annotation)|labeling cost|"
        r"minimal human (?:supervision|effort)|with(?:out)? human annotation",
        "本文提出或使用了**降低人工标注成本**的自动/廉价标注手段，并报告其效果。"
        "（算：用 GPT-4 造监督数据、`minimal human supervision`、众包廉价标注；"
        "不算：只抱怨标注贵、只引用他人做法）",
        "annotation cost crowdsourcing cheap labeling",
        "getting labels is expensive or hard so we look for a cheaper way"),
}

# 与 v1 的关键差异（供文档/审计引用）
CHANGED = {
    "ablation": "锚点 `\\bablation` → `\\bablat`（v1 漏动词 ablate，实测误删过真阳性）",
    "efficiency": "题面明确「模型推理本身」且「无论出现在哪一节」→ 修 v1 把 Limitations 里的数值误删",
    "significance": "题面明确「来自重复运行」+ 锚点加 std / p<0.0 → 区分「方差分解」",
    "multilingual": "题面明确「在非英语数据上报告结果」→ 修 v1 误删",
    "zero_few_shot": "题面明确「报告了结果」vs「当作对比基线」→ 修 v1 边界",
    "context_length": "题面明确「以长输入为方法目标」+ 锚点加 long-range → 修 v1 双向错误",
    "deployment": "题面明确「实际部署」vs「以部署为动机」→ 修 v1 误删",
    "case_study": "题面明确「逐例展示与讨论」→ 修 v1 误收（只有定量分析）",
    "safety_bias": "题面明确「专门评估/讨论」→ 修 v1 误收（只提 harmless）",
    "human_eval": "题面明确「收集人工判断并报告」→ 修 v1 边界",
}
