# 闸门「无证据支撑数字」溯源判定（离线，零 LLM）

> 来源 `qa/recall/GATE_PROBE_QASPER_20260911.md`｜对每个被报数字查：**在论文 chunk 里吗？能由其他数字推得吗？**

## 1912.01673｜How many sentence transformations on average are available per unique sentence in dataset?
- 论文 chunk 数: **10**｜共 15,185 字符
- `0.069` → **C 可由草稿数字推得**（`0.07 × 1`）→ **误杀**（豁免② 不覆盖）
- `0.07` → **C 可由草稿数字推得**（`0.069 × 1`）→ **误杀**（豁免② 不覆盖）

## 1909.00338｜Which features do they use to model Twitter messages?
- 论文 chunk 数: **29**｜共 42,632 字符
- `15000` → **B 原文确有 `15,000`，归一化后丢了**
  - 原文：` set to 0 otherwise. During training, all features apart from the top 15,000 most frequent ones were removed. ` → **误杀（解析把数字弄坏）**

## 1901.11117｜What is in the model search space?
- 论文 chunk 数: **20**｜共 33,664 字符
- `7.3e+115` → **B 原文以 LaTeX 科学计数法存在**（`$… \times 10^{…}$`）→ scan 拆成三个数合成不了 → **误杀**

## 2001.07263｜How much bigger is Switchboard-2000 than Switchboard-300 database?
- 论文 chunk 数: **12**｜共 18,394 字符
- `6.67` → **D 论文里找不到、也推不出** → **真拦（疑似编数）**
- `1700` → **D 论文里找不到、也推不出** → **真拦（疑似编数）**

## 2001.07209｜Which datasets are used in the paper?
- 论文 chunk 数: **16**｜共 20,790 字符
- `4.1e+08` → **B 原文以 LaTeX 科学计数法存在**（`$… \times 10^{…}$`）→ scan 拆成三个数合成不了 → **误杀**
- `8.5e+11` → **B 原文以 LaTeX 科学计数法存在**（`$… \times 10^{…}$`）→ scan 拆成三个数合成不了 → **误杀**

## 2001.07209｜Which dataset sources to they use to demonstrate moral sentiment through history?
- 论文 chunk 数: **16**｜共 20,790 字符
- `4.1e+08` → **B 原文以 LaTeX 科学计数法存在**（`$… \times 10^{…}$`）→ scan 拆成三个数合成不了 → **误杀**
- `8.5e+11` → **B 原文以 LaTeX 科学计数法存在**（`$… \times 10^{…}$`）→ scan 拆成三个数合成不了 → **误杀**

## 2004.03925｜Do the authors give examples of positive and negative sentiment with regard to the virus?
- 论文 chunk 数: **9**｜共 13,187 字符
- `15` → **B 原文确有 `15`，归一化后丢了**
  - 原文：`ows that around 54$\%$ tweets are neutral, 29$\%$ positive and a mere 15$\%$ is negative. Fig. corresponds to ` → **误杀（解析把数字弄坏）**
- `24` → **B 原文确有 `24`，归一化后丢了**
  - 原文：` retrieval data, the authors showed that query frequency and 5 out of 24 term frequency distributions could be` → **误杀（解析把数字弄坏）**
- `29` → **B 原文确有 `29`，归一化后丢了**
  - 原文：`inferred from Table that shows that around 54$\%$ tweets are neutral, 29$\%$ positive and a mere 15$\%$ is neg` → **误杀（解析把数字弄坏）**
- `54` → **B 原文确有 `54`，归一化后丢了**
  - 原文：` positive. The same can be inferred from Table that shows that around 54$\%$ tweets are neutral, 29$\%$ positi` → **误杀（解析把数字弄坏）**
- `60` → **B 原文确有 `60`，归一化后丢了**
  - 原文：`s have a neutral and positive sentiment. Table that shows that around 60$\%$ tweets are positive, 24$\%$ neutr` → **误杀（解析把数字弄坏）**

## 1906.03338｜How do they demonstrate the robustness of their results?
- 论文 chunk 数: **11**｜共 22,163 字符
- `100` → **D 论文里找不到、也推不出** → **真拦（疑似编数）**

## 2003.04967｜What experimental evaluation is used?
- 论文 chunk 数: **12**｜共 28,355 字符
- `2.7` → **B 原文确有 `2.7`，归一化后丢了**
  - 原文：` market trends. We used PySpark v2.3 in Jupyter notebooks with Python 2.7 kernels to code KryptoOracle. The en` → **误杀（解析把数字弄坏）**

## 1809.01060｜What were the results of the first experiment?
- 论文 chunk 数: **9**｜共 23,939 字符
- `0.81` → **B 原文确有 `0.81`，归一化后丢了**
  - 原文：` average of 14 annotators per pair. We found a Pearson correlation of 0.81 between the in-context and out-of-c` → **误杀（解析把数字弄坏）**

## 汇总

- {'A 归一化即匹配': 0, 'B 原文有·归一化坏': 13, 'C 可推导': 2, 'D 真找不到': 3}

> 误杀（B 归一化损坏 + C 豁免不覆盖）= **15**｜真拦（D）= **3**｜A（理论上不该报）= 0
