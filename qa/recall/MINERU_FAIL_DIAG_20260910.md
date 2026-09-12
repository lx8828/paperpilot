# MinerU 自测失败归因（4 题）

========================================================================================
## mh-28447-q1 (2608.28447v1) score=1 [L0/fallback]
Q: How much does Tool-DAPO improve pass@1 over Tool-SFT, and what are the two exact numbers?
gold: From 35.8% (Tool-SFT) to 66.0% (Tool-DAPO)
judge: 答非所问，未提供具体数值。

--- 系统回答 ---
抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

--- 检索到的 top12（chunk_id / 节 / 长度 / 首80字）---
  [c14] '' len=3548 | 5.3 Comparison of Different Reinforcement Learning Methods We also compare diffe
  [c1] '' len=1425 | Abstract Current large language models (LLMs) increasingly benefit from external
  [c10] '' len=1586 | 4 Experimental Setup We evaluate on the Countdown arithmetic reasoning task. Eac
  [c13] '' len=2239 | 5.2 Tool Integration As shown in Figure 3, we compare no-tool SFT, tool-augmente
  [c5] '' len=1831 | 3.1 Construction of Tool-Integrated SFT Dataset The base Countdown task asks the
  [c18] '' len=1060 | 7 Conclusion This project shows that reinforcement learning and calculator tool 
  [c17] '' len=1293 | 6.2 Reinforcement Learning Encourages Tool Use As shown in Figure 8, starting fr
  [c2] '' len=2823 | 1 Introduction Large language models (LLMs) have become increasingly capable in 
  [c9] '' len=2434 | 3.5 DAPO Tool-DAPO (Yu et al., 2026) modifies the policy update in two main ways
  [c12] '' len=1225 | 5.1 Evaluation of the SFT Model without Tools To diagnose the failure modes of t
  [c3] '' len=1991 | 2 Related Work Beyond supervised fine-tuning, preference optimization and online
  [c8] '' len=2028 | 3.4 GRPO Group Relative Policy Optimization (GRPO) (Shao et al., 2024) is a poli

========================================================================================
## mh-29290-q1 (2608.29290v1) score=1 [L3/fallback]
Q: How is the repair success rate defined and computed in their evaluation?
gold: The ratio of successful patch generations to the total number of execution runs; per case, the number of successful patches across five executions divided by the total number of executions
judge: 答非所问，未提供关于修复成功率定义和计算的具体信息。

--- 系统回答 ---
抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

--- 检索到的 top12（chunk_id / 节 / 长度 / 首80字）---
  [c12] '' len=1744 | 3.5 Evaluation The success rate was calculated as the ratio of successful patch 
  [c21] '' len=1958 | 6 Conclusion This study examined how database design in RAG afects LLM-based aut
  [c2] '' len=493 | – We demonstrate that separating retrieval contexts by API version and content t
  [c16] '' len=2415 | 4.2 Threats to Validity First, the evaluation covers only two API services, Swit
  [c6] '' len=1226 | 2.3 RAG We build on existing automated program repair methods by incorporating R
  [c13] '' len=409 | 4 Results We applied the proposed method to datasets containing misuse cases fro
  [c5] '' len=1985 | 2.2 Existing LLM-based Automated Program Repair Methods and Their Limitations In
  [c15] '' len=379 | Overall, increasing the number of databases in the RAG architecture improves the
  [c14] '' len=3899 | 4.1 Discussion The results indicate three main causes of repair failures. First,
  [c20] '' len=1015 | 5.3 Knowledge-Augmented Repair RAG has also been applied to APR and bug analysis
  [c1] '' len=3899 | 1 Introduction In recent years, Internet of Things (IoT) devices have become inc
  [c7] '' len=2027 | 3 Proposed Method Recent studies have applied various RAG configurations to auto

========================================================================================
## mh-29290-q2 (2608.29290v1) score=1 [L3/fallback]
Q: What conditions must a generated patch satisfy to be counted as a successful repair?
gold: It must satisfy the rules defined in existing research and match the corresponding GitHub repository; it must fix all misuse instances in the target function; when applied to the original repository it must compile successfully
judge: 答非所问，未提供任何与问题相关的信息。

--- 系统回答 ---
抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

--- 检索到的 top12（chunk_id / 节 / 长度 / 首80字）---
  [c12] '' len=1744 | 3.5 Evaluation The success rate was calculated as the ratio of successful patch 
  [c16] '' len=2415 | 4.2 Threats to Validity First, the evaluation covers only two API services, Swit
  [c19] '' len=2140 | 5.2 Automated Program Repair APR aims to generate patches for specific software 
  [c14] '' len=3899 | 4.1 Discussion The results indicate three main causes of repair failures. First,
  [c1] '' len=3899 | 1 Introduction In recent years, Internet of Things (IoT) devices have become inc
  [c20] '' len=1015 | 5.3 Knowledge-Augmented Repair RAG has also been applied to APR and bug analysis
  [c5] '' len=1985 | 2.2 Existing LLM-based Automated Program Repair Methods and Their Limitations In
  [c2] '' len=493 | – We demonstrate that separating retrieval contexts by API version and content t
  [c11] '' len=1365 | 3.4 Construction of Evaluation Dataset To construct the dataset, we collect sour
  [c4] '' len=2211 | 2.1 REST APIs Numerous IoT device development eforts adopt web APIs based on the
  [c9] '' len=2479 | 3.2 Standard RAG Method The baseline method has several limitations, such as inc
  [c18] '' len=1151 | 5.1 API Misuse Approaches that use graph structures and constraints to detect Ja

========================================================================================
## mh-29290-q3 (2608.29290v1) score=1 [L3/fallback]
Q: What limitations do they identify in existing LLM-based automated program repair methods, and which two representative approaches do they discuss?
gold: Xia et al. compared traditional repair tools (e.g. TBAR) with LLM-based repair over nine pretrained LLMs, five datasets and three languages, showing LLMs outperform traditional tools in language generality and repair accuracy, but repairs requiring external information such as REST API specifications fall outside their scope. Li et al. proposed a parameter-efficient fine-tuning (PEFT) method that updates only a subset of model parameters and showed PEFT can maintain or improve repair performance while using fewer computational resources.
judge: 答非所问，未提供任何与问题相关的信息。

--- 系统回答 ---
抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。

--- 检索到的 top12（chunk_id / 节 / 长度 / 首80字）---
  [c5] '' len=1985 | 2.2 Existing LLM-based Automated Program Repair Methods and Their Limitations In
  [c19] '' len=2140 | 5.2 Automated Program Repair APR aims to generate patches for specific software 
  [c21] '' len=1958 | 6 Conclusion This study examined how database design in RAG afects LLM-based aut
  [c9] '' len=2479 | 3.2 Standard RAG Method The baseline method has several limitations, such as inc
  [c17] '' len=176 | 5 Related Work In this section, we review existing research relevant to our stud
  [c7] '' len=2027 | 3 Proposed Method Recent studies have applied various RAG configurations to auto
  [c18] '' len=1151 | 5.1 API Misuse Approaches that use graph structures and constraints to detect Ja
  [c6] '' len=1226 | 2.3 RAG We build on existing automated program repair methods by incorporating R
  [c14] '' len=3899 | 4.1 Discussion The results indicate three main causes of repair failures. First,
  [c16] '' len=2415 | 4.2 Threats to Validity First, the evaluation covers only two API services, Swit
  [c1] '' len=3899 | 1 Introduction In recent years, Internet of Things (IoT) devices have become inc
  [c20] '' len=1015 | 5.3 Knowledge-Augmented Repair RAG has also been applied to APR and bug analysis
