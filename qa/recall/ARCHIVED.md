# qa/recall 已归档的一次性脚本（2026-09-14）

这里原本有 **159 个脚本**，其中 **136 个是一次性诊断脚本**（每次调查单独写、跑完即止）：`_diag_*` / `_ab_*` / `_probe_*` / `_verify_*` / `_scan_*` / `_chunk_*` / `_facts_*` …
它们把 `qa/recall/` 撑成 595 个文件，结果**「哪些是真测试、哪些是当时的一次性脚本」完全看不出来**。

已移到 `qa/_archive/recall/`：该目录被 `.gitignore` 忽略 → **不再进仓库**（本机文件仍在、git 历史也仍在）。保留在本目录的 **23 个**是仍被引用的：

| 为什么保留 | 文件 |
|---|---|
| 被 `tests/` 导入（评测口径模块） | `_tgt.py` |
| 被其它脚本 `import` | `_ab_rewrite.py`、`_table_policy_ab.py` |
| README / DEVELOPMENT_LOG / qa 下报告直接提到 | 其余 20 个（按前缀：`_ab_*` / `_diag_*` / `_gate_*` …） |

> ⚠️ 归档脚本按**当时的相对路径**写死（`assets/`、`qa/recall/` 等）。要复跑：先移回本目录，
> 或在调用处改 `sys.path` —— 它们本来就是「一次性」的，不建议直接复用。

## 已归档清单（136）

| 脚本 | 当时在做什么（docstring 首行） |
|---|---|
| `_ab_answer_unknown.py` | A/B：answer-before-unknown —— judge 判不够时是否该先试答一次。 |
| `_ab_gate_fail.py` | 修复回路考场：抽 10 道 QASPER 历史 fail 题（v3regress V 臂 fail 且有 gold evidence）， |
| `_ab_gate_qs.py` | 修复回路题集实测：HyGRAIL 需检索题 × GATE on/off 双对照。 |
| `_ab_l2center.py` | 实验策略④：L2 多候选圆心 A/B on L2 目标题群 67 题。 |
| `_ab_l2comb.py` | 实验：judge_l2_strict + 多候选圆心 组合 on L2 目标题群 67 题。 |
| `_ab_l2hard.py` | 实验策略③：judge_l2 硬指标 A/B（base vs HARD）on L2 目标题群 67 题。 |
| `_ab_l2strict.py` | 实验：judge_l2_strict（严判够，防首窗假阳性放行）on L2 目标题群 67 题。 |
| `_ab_l2target.py` | 实验 L2-target：在"L2 目标题群"上对比 A(现状扩窗) vs B(砍L2直L3)。 |
| `_ab_noL2.py` | 实验1 A/B：砍掉 L2（L1 不够→直接 L3）vs 现状完整漏斗。 |
| `_ab_nol3j.py` | A/B：v3 现图(judge_l3→unknown) vs 新图(nol3j 试答 + GATE)。 |
| `_ab_v3l1.py` | Step2：v3+L1(claims 分支) 单臂，与 Step1 已存的 A(现状60)/V0(纯两级64) 同 100 题对照。 |
| `_ab_v3regress.py` | v3 定稿回归：同裁判同进程 A(现状 v2 漏斗) vs V0(v3 纯两级)，权威规模样本。 |
| `_agg_v3.py` | 临时：聚合 v3 系列端到端对拍成绩（只读 json）。""" |
| `_aniso_ab.py` | 各向异性去除 A/B（离线打分解耦实验，零 LLM 生成、零文档重编码）。 |
| `_attr_target.py` | 把 target_sections 命中率拆成：claims 覆盖度 × judge 挑选准确率。 |
| `_audit_metrics.py` | 临时（唯一口径）：官方分母对齐的 Recall/MRR/NDCG/hit 多维对比。 |
| `_b13d_check.py` | 异常题 b13d0e46 归因：为什么 L3 检索只拿到 1 块 / 248 字符。""" |
| `_b2_ab.py` | B2 A/B：段落流均匀重打包（跨标题自由打包） vs 现状按标题切块。可断点续跑。 |
| `_b2_xcause.py` | 临时：因果检验——"碎块导致排序差"是真因果还是相关？ |
| `_big_probe.py` | 临时：大块干扰探针——查证三条假设（base 检索，官方口径，零新 encode）。 |
| `_bigsec.py` | 临时：量化"大节论文"——某单节块数很大的论文占比（决定 cap 是否需自适应）。""" |
| `_build_multiq.py` | 从全池扫"多方法/全局对比"题，抽 40（per_pid≤2）存 multiq_set.json。""" |
| `_build_set.py` | 建 Recall@k 样本集 recall_set_v1.json（普通 + hard 分层，各带 gold evidence 可用）。 |
| `_cap_ab.py` | 节级去重 cap=1 端到端 A/B：在 ab_v3base 同 100 题单上单臂跑 cap=1， |
| `_cap_fill.py` | 临时：cap=1 节级去重后，top_k 实际能填多少块（量化"尾巴空多少"）。 |
| `_cap_sum.py` | 临时：cap=1 A/B 汇总明细。""" |
| `_cap_sweep.py` | 临时：cap 上限对"接收块被推出甜点区"的影响扫描。 |
| `_chunk_audit.py` | 临时：切块质量审计——chunk 粒度是不是排序难的一个原因？ |
| `_chunk_gran.py` | 临时：chunk 粒度不均匀量化 + 与题目难度的混杂检查。 |
| `_chunk_sim_dist.py` | chunk 块间余弦相似度**分布**诊断（零 LLM，复用 cvec 缓存）。 |
| `_chunk_stats.py` | 临时：统计 recall_set 208 篇的 chunk 数分布 → top12 占全论文比例。""" |
| `_chunk_stats2.py` | 临时：准确统计 top12 覆盖全篇 chunk 比例（cap 到 100%）。""" |
| `_cost_probe.py` | 临时：估算评测成本——读历史 run 的实际 token/calls，算 250 题单臂/双臂的用量。""" |
| `_dbg_numfail.py` | 临时：查 545ff2/cf63a4 的 gate number 误报根因。 |
| `_diag_equation_value.py` | 公式块值不值得注入？三条实测（零 LLM）。 |
| `_diag_l1.py` | L1 judge 专项诊断：判"够"的假阳率 + target_sections 命中答案节的比例。 |
| `_diag_still_fail.py` | 诊断：补了表格通道后**仍然 ≤3 分的 22 题**到底卡在哪一段？ |
| `_diag_tbl_rank.py` | 诊断：为什么表格块进不了 top-12？（纯向量位次 + BM25 位次逐项拆开） |
| `_dump_for_questions.py` | 临时：导出论文关键材料（摘要 + 贡献句 + 表格块），供人工出 hard 题并写金标准。""" |
| `_dump_miss.py` | dump 覆盖但 judge 未选对 gold 节的 12 条，供 A/B/C/D 人工分类。""" |
| `_exit_stat.py` | Exit 判据量化：L2 值不值得修（决定"修 L2"还是"砍 L2"）。 |
| `_ext_variants.py` | 表块表征变体实测：去重 / 加 refs / 纯尺寸填充，谁在表池里排得更好？ |
| `_facts_diag.py` | 临时：两步法失效归因——失败题的答案缺失，是"第一步没捞到"还是"第二步没用"。 |
| `_facts_empty_check.py` | EMPTY（第一步返回空清单）归因：gold 证据到底在不在检索到的块里？（零 LLM） |
| `_facts_fix_regress.py` | 两步法修复的配对回归：新旧代码在**同一 100 题**上 diff。 |
| `_facts_probe.py` | 两步法第一步（_extract_facts）失效规模与根因探针。 |
| `_flip_attrib.py` | 并集 A/B 的翻转归因：q0(12 块) vs q2(12~14 块)。 |
| `_flow_diag.py` | 诊断 case1：L0 概述答案被 gate 兜底的具体 issues。""" |
| `_flow_smoke.py` | 全流程冒烟：Router 组件接入 v3 + 新 SYSTEM(硬引用) + gate on，一次跑通。""" |
| `_fusion_probe.py` | 临时：融合策略扫描——oracle(单路最好) 远超 hybrid，说明 RRF 融合在丢分。 |
| `_gold_in_chunks.py` | 临时：决定性检验——失败题的 gold 数值，是否出现在检索到的 12 块正文里。 |
| `_gold_mult.py` | 临时：gold 口径对比——只取 evidence[0]（现行） vs 取全部 evidence 段（QASPER 官方口径）。 |
| `_gold_rank.py` | 临时：解剖 hard 组 gold 的检索位次——gold 排第几、前面压着哪些块。 |
| `_judge_facts_pos.py` | 判据 2/3：facts 覆盖率 与 证据位次×通过率。 |
| `_k_scan.py` | 临时：K 值扫描 + 尾部价值诊断（新 gold 口径=全部 evidence 段）。 |
| `_l2center_report.py` | (无 docstring) |
| `_l2comb_report.py` | (无 docstring) |
| `_l2hard_dump_q.py` | (无 docstring) |
| `_l2hard_report.py` | (无 docstring) |
| `_l2strict_report.py` | (无 docstring) |
| `_l2target_miss.py` | (无 docstring) |
| `_l2target_report.py` | 从 ab_l2target_result.json 出分层报告（含 L2 目标题群明细）。""" |
| `_latex_norm_probe.py` | 原型：把 MinerU 的 LaTeX 公式文本规范化成**可检索文本**，并量化效果。 |
| `_layout_compare.py` | 临时：复杂版面对比——双栏阅读顺序 + 页眉/脚/页码剥离（pymupdf vs MinerU）。""" |
| `_merge_ab.py` | S_merge A/B：只并超小块、不切任何健康块（base 保留）vs 现状 base。 |
| `_merge_plan.py` | 临时：S_merge 方案量化——只并超小块、不切任何健康块。 |
| `_mineru_check.py` | 临时：验证 10 篇 MinerU 解析产物 → chunk 桥接是否正常（重点看表格是否文本化）。""" |
| `_mineru_cost.py` | 临时：MinerU hard 自测的 token 花费统计。""" |
| `_mineru_coverage.py` | MinerU 表格覆盖率自检 v2（QASPER 官方 caption 作 ground truth）。 |
| `_mineru_fail_diag.py` | 临时：MinerU 自测失败题归因——看答案、闸门动作、以及 gold 内容是否在检索块内。""" |
| `_multiq_report.py` | (无 docstring) |
| `_ndcg_diag.py` | 临时：NDCG 低的原因诊断——按题目宽泛度/锚点分组看 MRR/NDCG。 |
| `_ndcg_low.py` | 临时：列出 NDCG 最低 / gold 排位最差的题原文，人工判断是不是"题目宽泛导致难检索"。""" |
| `_nocap_probe.py` | 无 caption 表块：到底缺什么、有多少能补？ |
| `_numbers_verify.py` | 临时验证：numbers.py 抽取后 (1) 旧归一行为不丢 (2) 新形态增量正确。""" |
| `_only_table_retr.py` | 只检索表块（table-only pool）能比混合池好多少？—— 验证"分开检索"的检索层上限。 |
| `_order_ab.py` | 排序影响端到端对照（A/B）：gold 强制第1 vs 自然顺序。 |
| `_p1_retrieval_ab.py` | P1 检索层配对 A/B：**旧表文本 vs 新表文本**（同环境、确定性、0 LLM）。 |
| `_p1_verify.py` | P1 验证（可复算）：矩阵投影 + 多级表头折叠 + 竖线转义。 |
| `_p2_mech.py` | 验证 P2 双写变差的机制：摘要会不会让**同篇内的表块互相更像**（区分度下降）。 |
| `_p3_tblab.py` | 表专项抽取：**相关性过滤** vs **宁多勿少全列**，在全部含表的题上对照。 |
| `_p3_verify.py` | P3 验证：打印**改造后**实际喂给 LLM 的用户消息，并检查 facts 是否真去读表。 |
| `_para_len.py` | 临时：QASPER 段落长度分布——超长段(>1600/2400/4000)占比、落在哪些节、是否覆盖 gold。 |
| `_peek_ab.py` | (无 docstring) |
| `_peek_gate.py` | 临时：读 gate 修复回路考场结果（fail 10 道 + HyGRAIL qs）。""" |
| `_peek_nol2.py` | (无 docstring) |
| `_peek_v2.py` | (无 docstring) |
| `_prewarm.py` | 为 recall_set 中缺 cvec 的论文预建向量缓存（一次性；完成后后续尺子秒级）。""" |
| `_probe75.py` | 临时：查 cf63a4(2003.09520) 的 7.5 是否有推导素材（时间/倍率上下文）。""" |
| `_probe_num_origin.py` | 临时：查 545ff2 / cf63a4 原文里相关数字的真实形态（判断 A 型还是 B 型根因）。""" |
| `_probe_reject.py` | 诊断探针：judge 双拒(自然&gold-first 都 enough=false)的题，跳过 judge 直接 answer。 |
| `_probe_sec.py` | 临时：看真实 chunk 的 title_path / top_section 形态（定 intro/conclusion 匹配规则）。""" |
| `_probe_split.py` | 临时：验证二级切分产物——同 title_path 的多块是否共享同一节名（含 part）。""" |
| `_progress.py` | 临时：查看 MinerU hard 自测进度。""" |
| `_qa176_stat.py` | (无 docstring) |
| `_qasper_data_quality.py` | QASPER 数据质量扫描：recall_set 用到的论文里，有多少 full_text 正文是空的？ |
| `_quota_cost.py` | "分池 + 名额保底"的代价：给表池留 m 个名额，文本侧会丢多少？ |
| `_regress_gatefix.py` | 临时：数字归一修复后，重跑 ab_nol3j 中 gate 干预的 5 题（2 fallback + 3 repaired）。""" |
| `_repair_ab.py` | 修复回路 A/B：10 道历史 V fail 题，GATE off vs on(REPAIR_MID=1)，同裁判双打分。 |
| `_report_v2.py` | (无 docstring) |
| `_rerank_ab.py` | 离线排序 A/B（零 LLM、零下载）：能不能把"检索排序精度"提上来？ |
| `_scan_combo.py` | 临时：联合扫描——先 fullpath cap1 理顺候选，再叠加 M2 概述降权（decay）。 |
| `_scan_fullpath.py` | 临时：cap 按【完整 title_path】去重（而非顶层节 top_section）——修正版扫描。 |
| `_scan_metrics.py` | 临时：多维离线指标 + rank 位移诊断（不只 R@8）。 |
| `_scan_multiq.py` | 扫描全池"多方法/全局对比"类题（QASPER），供多维分支可行性判断采样。""" |
| `_scan_section.py` | 临时：单篇检索的节级优化扫描（Recall@k/MRR，零 LLM，复用 recall_set）。 |
| `_slice_l2.py` | (无 docstring) |
| `_spike_v2.py` | judge_v2 spike（最小闭环，不改生产代码）：两步法 judge vs 现状 L3 judge。 |
| `_split_sim.py` | 临时：静态模拟不同切块 target 会切掉多少 base 健康块（不 encode）。 |
| `_status.py` | A/B 状态速览（可反复调用）。""" |
| `_sum_nol3j.py` | 临时：nol3j A/B 明细（loss / gate 干预题）。""" |
| `_supp_ab.py` | A/B：补充复核是否有用（数字型题 16，同主图输出分两臂）。 |
| `_supplement_e2e.py` | 临时：真实 QASPER 题目 e2e 验证 supplement 链路（GATE=1）。 |
| `_supplement_test.py` | 临时验证：数字 supplement 机制（不烧 LLM，只验证机器层分流 + gate 分支）。""" |
| `_survey_papers.py` | 临时：盘点可用论文（报告缓存 / MinerU 产物 / 标题）。""" |
| `_tblfix_analysis.py` | P1+P3 同日配对 A/B 的归因分析。""" |
| `_tblfix_perq.py` | 逐题 Δ（跨轮均值）+ 输赢分布：看"new 更差"的题落在哪一组。""" |
| `_tblfix_pooled.py` | P1+P3 配对 A/B：多轮合并分析（自动发现轮次）。 |
| `_tblfix_stat.py` | P1+P3 配对 A/B 的统计检验（自动发现轮次；配对 t + 逐题符号检验）。""" |
| `_tgt_denom.py` | 算清两套口径各自的**分母**（旧口径定位不到的题不该算进旧口径的分母）。 |
| `_top_agg.py` | 临时：由 top-dump 引发的两个补充测量（零新 encode，仅查 encode_query）。 |
| `_top_dump.py` | 临时：查"排前面的是些啥"——把 gold 排位差的题，把它们 top12 的真实块内容打出来。 |
| `_union_prod_eval.py` | 生产链路下的并集候选评估（**口径 v2**：编号 ∪ 内容/数值）。 |
| `_v3base_report.py` | (无 docstring) |
| `_v3c14_regress.py` | 单臂回归：当前默认图（v3 Router 包装 + Generator SYSTEM 准则14 硬引用，gate 默认关） |
| `_v3l1_report.py` | (无 docstring) |
| `_v3regress_report.py` | (无 docstring) |
| `_valid_calib.py` | Validator 离线校准：在已有 run 的 answer 上跑"漏数/编数"检查，看能否抓出失败题。 |
| `_valid_calib20.py` | Validator 误报校准：negqa 20 道好答案跑 check（机器+LLM 体检）。 |
| `_valid_llm_smoke.py` | 非判断题的 LLM 体检仍工作：注入编造对象应被抓 unsupported。""" |
| `_valid_model_ab.py` | 模型 A/B：unsupported 体检 glm-4-flash(JUDGE) vs deepseek-chat(LLM=供应商 v4-flash)。 |
| `_valid_smoke.py` | Validator.check 冒烟：好答案应少误报；注入编造句/编数应被抓。""" |
| `_verify_cap.py` | 临时：验证 embedder 的 section cap 开关（env 开/关行为对照）。 |
| `_verify_facts_fix.py` | 验证两步法第一步两处修复（1.1 删负向提示 / 1.2 宽松解析兜底）。""" |
| `_weight_sweep.py` | 按块类型分权的 α 扫描：**文本块固定=生产权重（两路满权重）**，只调外部块（表格/公式）。 |
| `_why_fail.py` | 诊断：目标表已进上下文、但端到端仍失败的题，卡在哪一环？ |
