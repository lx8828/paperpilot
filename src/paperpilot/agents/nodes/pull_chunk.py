"""L3 · search_l3（独立全文检索）。

L3 是 v3 两级架构的检索层：L0 总览判不够后触发，独立 ChunkIndex 全文检索
（不继承 L0 证据），结果写 state["l3_chunks"]，供 judge_l3 / generate_answer 使用。

**查询侧优化已抽成独立组件** `components/query_optimizer.py`
（五级：L1 意图识别 / L2 问题重述 / L3 改写扩展 / L4 HyDE / L5 查询分解，
目前仅 L3 实现）。本节点只负责"取查询集 → 检索"，
默认全关（只回原问题）→ 与既有 A/B 基线行为完全一致。

**2026-09-10：v2 的 L2 expand_l2（定向增量扩展：圆心 + 半径 + 预算自环）已下线归档**
（archive/qa_funnel_v2/），本文件只保留 L3。原实现见归档快照。
"""
from __future__ import annotations

import os
from typing import Any

from paperpilot.agents.embedder import MultiChunkIndex
from paperpilot.agents.state import QAState
from paperpilot.components import query_optimizer

# ⚠️ 2026-09-24 重定：原依据（"top8 覆盖 gold ~73%、top12 ~85%、上下文可控"）是**单篇时代**的。
#    5 篇语料下重扫（`retrieval/tmp/_scan_quota.py`，2 组 × 60 道需检索题 = 120 题，
#    **零 LLM 成本**，量"必现锚点是否落进候选"，与 `_check_reachable --search` 同口径）：
#        layered/quota  N=12 floor=1 → 未命中 22/120 ｜ 候选均 21k 字符  ← 旧默认
#                       N=24 floor=1 → **11/120**（减半）｜ 42k          ← 新默认
#                       N=32 floor=1 →   9/120        ｜ 56k
#                       N=48 floor=1 →   5/120        ｜ 82k
#    为什么加大 N 是**唯一有效**的旋钮：产品是中文提问、语料是英文论文
#    → `_bm_or_none` 判定 BM25 无信号（见该函数 docstring）→ 整条**词法路被跳过**，
#    跨语言只能靠向量路，英文专名类锚点排名靠后时只能靠**拓宽候选**兜住。
#    为什么取 16（2026-09-26 取代 24；`retrieval/tmp/_inter_ab.py` 同口径实测）：
#      上面"加大 N 是唯一有效旋钮"成立于**补充路只能整块追加**的年代。现在补充路改走
#      **加权交错**（见 `_rrf_interleave_hits`）→ 16 块就够，且**同候选数**对照：
#        append 截到 16（旧合并语义） gold@12 69.3% ／ 加权交错 **79.4%**（+10.1pt）
#        字符 28535 → 27962（略省）／ M1 篇全 80% → **95%**／lift 5.2× → **6.2×**
#      → 旧值 24 的实测（N=12→22/120、N=24→11/120、N=48→5/120）是"**未命中**"口径，
#        与 gold@12 **不同轴**（集合口径 vs 位次口径）；在新合并下 24 只剩成本。
#
#    同批扫掉的三个"看起来该调"的旋钮（都**不该动**）：
#      · `floor`（quota 篇内保底）保持 1：floor≥2 在**每个 N 上都更差**——篇内保底是
#        **从尾部替换**，5 篇×floor 会吃掉 5*floor 个槽位，N=12/floor=3 直接退化成
#        "保底专属"（未命中 62/120）。
#      · **新增"每篇最多 cap 块"**（capfloor，防单篇霸占）：每个 N 上都比 quota 差
#        （N=24: 19 vs 11；N=32: 13 vs 9）—— 排序好的篇的**多块同源证据**被别的篇的弱块挤掉。
#      · 节级配额 `PAPERPILOT_RETRIEVE_SECTION_CAP` 保持关（现默认）：任何 cap 都负收益
#        （cap=3@N=24 → 13/120 vs 无 cap 11/120；cap=1 → 28~35/120）。⚠️ 且它的 key 是
#        `_section_of` 取的**篇内顶层节名、不含篇名** → 跨篇同名（Introduction/Method…）
#        会撞成一个节；一旦要开，必须改成 (篇, 节) 才算正确。
#    代价：候选字符数随 N 线性涨（单块长度由分块期 `MAX_CHUNK_LEN` 控制）。⚠️ 多路合并后
#    输出被 `_rrf_interleave_hits` 截到 `L3_TOP_K`，故**总数封顶**，不再随路数膨胀。
L3_TOP_K = int(os.environ.get("PAPERPILOT_L3_TOP_K", "16") or 16)

# ── L6 跨语言检索式（2026-09-24 接线；2026-09-26 数字按沙盒重校）───────────────
# 中文提问 + 英文语料：词法路（BM25）在**多数题**上其实仍开火 —— 实测 120 题中
#   87% 的题 BM25 有非零分（问句照抄了英文实体名/数字，见 `retrieval/tmp/_lang_effect.py`）；
#   真正缺的是**动词/领域表述**那部分英文措辞（这部分中文 token 在英文正文里零匹配）
#   → 英文专名/指标类锚点排名靠后，只能靠"硬加 top_k"兜。
# 给一条**英文检索式**并**配额并集追加**（数字口径：`retrieval/tmp/_sandbox.py --union/--ablate`）：
#     中文原问 11/120 → 加英文检索式 **7/120**（英文臂 K=8，旧默认）
#                      → **4/120**（英文臂**不截断** = 与主路同 K，**新默认**）
#   ⚠️ 早前注释里记的 "5/120" 是**非生产口径**（把英文臂算成"24 选的前 8"，绕开了 K=8 的
#      floor 替换）。四种口径实测：英文臂走 K=8（含 floor 替换）→ **7**；
#      24 选的前 8 → 5；纯全局 top8 → 5；**英文臂不截断（全 24）→ 4**（字符 56k）。
#      即：**小 K 下 floor 把英文臂 8 个槽位里的 5 个换成篇内保底项 → 反而变差**。
# ⚠️ "英文臂不截断 + 整块尾部追加"这条**已被取代**（2026-09-26 同日二次改定）：
#   当时的理由是"并集 ⊇ 主路 → 结构上不可能更差"。**实测发现那只在"未命中"口径成立**：
#   整块追加把英文臂放在主路 `L3_TOP_K` 名**之后**，而输出上限就是 `L3_TOP_K`
#   → **英文臂 100% 被截掉、等于没接**（`retrieval/tmp/_inter_ab.py`：n=1/2/3 条
#   补充路结果**完全一样**，正是这个原因）。
#   → 改用**加权交错**（`_rrf_interleave_hits`）+ `Kz = Ke = 16`，**同候选数**实测：
#        gold@12 69.3% → **79.4%** ／ M1 篇全 80% → **95%** ／ 字符 55773 → **27962（−50%）**
#        ／ lift 3.3× → **6.2×** ／ g1 73.6→78.0、g2 65.3→80.6（**两组同向**）
#   ⚠️ 旧口径的 Tier-2 结论（K=8→24：27/50 → 29/50，改善 2 / 回退 0）**仍有效**，但
#      噪声底 8/50（±16pt，同配置重跑翻盘）→ 只能当"**无回退**"旁证，不能宣称端到端收益。
#   ⚠️ 回滚开关：`PAPERPILOT_XLING_MERGE=append` 可退回旧"整块尾部追加"语义。
XLING_ENABLED = os.environ.get("PAPERPILOT_XLING", "1") != "0"
XLING_MERGE = os.environ.get("PAPERPILOT_XLING_MERGE", "inter").strip().lower()  # inter|append
XLING_W = float(os.environ.get("PAPERPILOT_XLING_W", "1.0") or 1.0)   # 补充路 RRF 权重 w_en
# 篇级保底（`_rrf_interleave_hits(cover=...)`）：交错是纯分数截断，会把整篇丢光；
# 而 `quota` 本来靠 `floor` 逐篇保底 → 实测不开时**篇覆盖 5.00 → 3.80**。默认开。
XLING_COVER = os.environ.get("PAPERPILOT_XLING_COVER", "1") != "0"
# 英文臂取多少条：与主路同（16）。实测 Ke 已到拐点（`_inter_ab.py` 里 Ke 再涨只增字符）。
L3_XLING_K = int(os.environ.get("PAPERPILOT_L3_XLING_K", str(L3_TOP_K)) or L3_TOP_K)


def _quota_union_hits(lists: list[list[dict[str, Any]]], quotas: list[int]
                      ) -> list[dict[str, Any]]:
    """多路命中 → 配额并集（**与 `query_optimizer.quota_union` 同语义**）。

    第 i 路只取前 `quotas[i]` 条，**按路顺序整块拼接**并去重（主路在最前）。

    ⚠️ **2026-09-26 修正**：原 docstring 写"**不能交错**：交错会稀释主查询排序"。这句
    对**这个场景**站不住 —— 逐条拆开那两处依据：
      · "被测试逮住" = `tests/test_query_optimizer.py::test_quota_union_puts_primary_first_and_never_drops_it`
        断言 `got[:2] == [1, 2]` → 钉的是**语义契约（主路优先）**，**不是质量**；
      · "同一机制" = 2026-09-07/09-10 的 A/B 测的是 **`equal_rrf`（等权）**，在**单篇漏斗**上。
    → **"加权交错 vs 整块拼接"从未被测过**。本函数保留只为**回退**用（`XLING_MERGE=append`）；
      实际致命的不是"交错稀释"，而是**整块拼接会让补充路被输出上限整段截掉**（见下一函数）。
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for hits, cap in zip(lists, quotas):
        for h in hits[: max(int(cap), 0)]:
            key = (str(h.get("pdf") or ""), str(h.get("chunk_id") or ""))
            if key not in seen:
                seen.add(key)
                out.append(h)
    return out


def _rrf_interleave_hits(lists: list[list[dict[str, Any]]], weights: list[float],
                         *, top_n: int, k: int = 60,
                         cover: bool = False) -> list[dict[str, Any]]:
    """多路命中 → **加权交错**（每路当一条 RRF 名次序列）→ 取前 `top_n`。← **当前默认**

    路由：第 i 路名次 r 的候选得分 `Σ weights[i] / (k + r)`；跨路同块**去重后分数相加**。

    为什么它取代"整块尾部追加"（2026-09-26 `retrieval/tmp/_inter_ab.py`，120 题，
    **同候选数**对照、零 LLM）：
      · **整块拼接结构上会让补充路失效**：主路配额 `L3_TOP_K` 与输出上限同为 16/24
        → 补充路排在主路之后 **100% 被截掉**。实测：n=1 / 2 / 3 条补充路**结果完全一样**。
      · 同候选数 16：append gold@12 **69.3%** → inter **79.4%**（+10.1pt）；字符略省
        （28535→27962）、M1 篇全 80%→**95%**、lift 5.2×→**6.2×**、g1/g2 **同向**（CV 通过）。
      · 交错还提高了**池上界**（84.7% vs 70.9%）→ 同样 16 个候选里 **gold 更多**，
        不只是"排得更好"。
      · `k=60` 保持不动：实测 k=10/20/60 在本预算下 gold@12 为 78.8/79.4/**79.4%**（打平），
        故不去动这个全局常量（**少改一处、少一处回归风险**）。
      · `w_en=1.0`（等权）最优；0.7/0.3 都更差 → 与"等权更有害"的旧印象相反（见上）。

    ⚠️ `cover=True`：**篇级保底**。交错是纯分数截断，会把"整篇没有任何块"的篇丢掉，
    而 `quota`（真实器）本来靠 `floor` 逐篇保底 → 实测 `cover=False` 时**篇覆盖 5.00 → 3.80**
    （`_sandbox.py --prod`）。开了它：先把每篇的最高分候选保住，再按分数填满 `top_n`
    （超额时淘汰"兄弟最多"的最低分项）。
    """
    sc: dict[tuple[str, str], float] = {}
    bykey: dict[tuple[str, str], dict[str, Any]] = {}
    for hits, wt in zip(lists, weights):
        for r, h in enumerate(hits):
            key = (str(h.get("pdf") or ""), str(h.get("chunk_id") or ""))
            sc[key] = sc.get(key, 0.0) + wt / (k + r + 1)
            bykey.setdefault(key, h)
    ranked = sorted(sc, key=lambda x: -sc[x])[: max(int(top_n), 0)]
    if not cover or not ranked:
        return [bykey[key] for key in ranked]
    best: dict[str, tuple[float, tuple[str, str]]] = {}
    for key, s in sc.items():
        p = key[0]
        if p not in best or s > best[p][0]:
            best[p] = (s, key)
    keep = list(ranked)
    for p, (s, key) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        if any(k[0] == p for k in keep):
            continue
        cnt: dict[str, int] = {}
        for k in keep:
            cnt[k[0]] = cnt.get(k[0], 0) + 1
        drop = next((k for k in reversed(keep) if cnt[k[0]] > 1), None)
        if drop is None:
            break
        keep.remove(drop)
        keep.append(key)
    return [bykey[key] for key in sorted(keep, key=lambda x: -sc[x])]


def _section_tail(paths: list[str]) -> str:
    for p in reversed(paths):
        if " · " in p:
            return p.split(" · ", 1)[1].strip()
    return paths[-1] if paths else ""


def _to_chunk_texts(chunks: list[Any]) -> list[dict[str, Any]]:
    """chunk → 可序列化文本切片。注意：**不再截断**。

    chunk 在分块期已按段落原子切到 ~4000 字符（document_cache.MAX_CHUNK_LEN），
    QA 层再截断就是纯丢信息（答案可能落在被砍掉的尾部）。
    """
    out = []
    for c in chunks:
        out.append({
            "chunk_id": c.chunk_id,
            "page": c.page_span[0],
            "section": _section_tail(list(c.title_path)),
            "text": c.text,
        })
    return out


def search_l3(state: QAState) -> dict[str, Any]:
    """L3 独立全文检索：ChunkIndex 对（优化后的）查询集检索 topK，写 l3_chunks。

    查询集来自 `components/query_optimizer.optimize()`：
      · 默认全关 → 只有原问题 → 走**多篇主路径** `search_layered`（篇内检索 + 跨篇 quota 融合）；
      · 开启任一级（如 `PAPERPILOT_QUERY_LEVELS=l3`）→ 多查询 `search_multi_hybrid`
        （向量+BM25 × 多查询 RRF）。**注意**：多查询的等权 RRF 已被两次 A/B 证伪
        （见 query_optimizer 模块 docstring），此路径仅作对照，新融合策略应在
        检索侧按 `query_optimizer.quota_union` 之类实现后再做 A/B。
      · **L6 跨语言（2026-09-24 接线，默认开；2026-09-26 改走加权交错）**：中文问题额外
        生成一条**英文检索式**，走同一条 `search_layered` 取 `L3_XLING_K` 条，与主路做
        **加权交错**（`_rrf_interleave_hits`）并把总数**封顶**到 `L3_TOP_K` 块
        → `path = "layered_quota+xling"`。
        实测（`retrieval/tmp/_sandbox.py --prod`，120 题，**走本文件的合并函数**）：
        gold@12 69.3% → **79.4%**、M1 篇全 16/20 → **19/20**、篇覆盖 5.00 → **4.99**、
        字符 55773 → **28030（−50%）**。
        ⚠️ **代价（必须知道）**：`未命中`（锚点**全在**候选里，集合口径）**4 → 7** —— 这是
        "候选 48 → 16"的必然代价（被砍掉的 3 道，其锚点原本落在第 17~48 名的块里）。
        两个口径不同轴，谁更重要由**端到端**裁决；若端到端显示集合口径更关键，把
        `PAPERPILOT_L3_TOP_K` 调回 24（未命中回 5、字符 42k）即可，无需改代码。
        `PAPERPILOT_XLING=0` 关掉；`PAPERPILOT_XLING_MERGE=append` 回退旧"整块追加"；
        `PAPERPILOT_XLING_COVER=0` 关篇保底（**不要**：篇覆盖会掉到 3.80）。
    """
    question = state.get("question") or ""
    # ⚠️ 2026-09-23：**语料只有多篇**（本系统没有单篇路径）→ 一律走 MultiChunkIndex。
    idx = MultiChunkIndex([str(p) for p in (state.get("pdfs") or []) if p])
    plan = query_optimizer.optimize(question)
    queries = plan.queries
    xling = ""
    if plan.multi:
        hits = idx.search_multi_hybrid(queries, top_k=L3_TOP_K)
        path = "multi_hybrid"
    else:
        # **多篇主路径（2026-09-23 接线）**：每篇内先检索（同粒度可比）→ 跨篇 quota 融合。
        # 为什么不用全局 `search_hybrid`：**全局混池会系统性漏篇**（N=24 实测：hybrid 平均
        # 命中 3.60/4.18 篇、覆盖满 5 篇仅 15/60、23/60；本方法 4.90/4.93、54/60、56/60）。
        # ⚠️ 别再引用"谁切得细/块长差 5.8×"——那是 2026-09-21 的结论，**粒度后来已抹平**
        # （现中位块长只差 1.70~2.09×）；现在的主因是"粒度残留 + 同主题分数集中"。
        # 完整实测与原因见 `MultiChunkIndex.search_layered` 的 docstring。
        hits = idx.search_layered(question, top_k=L3_TOP_K)
        path = "layered_quota"
        if XLING_ENABLED:                        # L6：跨语言补充（主路 + 英文检索式）
            xling = query_optimizer.crosslingual(question)
            if xling:
                extra = idx.search_layered(xling, top_k=L3_XLING_K)
                if XLING_MERGE == "append":      # 回退：整块尾部追加（补充路会被截掉，见 _quota_union_hits）
                    hits = _quota_union_hits([hits, extra], [L3_TOP_K, L3_XLING_K])
                else:                            # 默认：加权交错，并把总数**封顶**到 L3_TOP_K
                    hits = _rrf_interleave_hits([hits, extra], [1.0, XLING_W],
                                                top_n=L3_TOP_K, cover=XLING_COVER)
                # ⚠️ 路径标签**故意不变**：`cli/eval/run_group_qa.py` 有"生产路径白名单"
                # （只认 `layered_quota` / `layered_quota+xling`），改名会让守卫误报。
                path = "layered_quota+xling"
    l3_chunks: list[dict[str, Any]] = []
    for h in hits:
        l3_chunks.append({
            "chunk_id": h.get("chunk_id", ""),
            "page": h.get("page", 0),
            "section": _section_tail(list(h.get("title_path") or [])),
            "text": (h.get("text") or ""),  # 不截断：分块期已控制单块长度
            # ⚠️ 多篇语料：每条带**它属于哪一篇** —— cites 才能定位到具体论文（2026-09-23）
            "pdf": h.get("pdf", ""),
        })
    debug = dict(state.get("debug") or {})
    dbg_l3: dict[str, Any] = {"topk": len(l3_chunks), "path": path,
                              "n_papers_hit": len({h.get("pdf") for h in l3_chunks})}
    if plan.levels:
        dbg_l3["query_plan"] = plan.summary()     # 只在真的开了级别时记，避免污染基线
    if xling:
        dbg_l3["xling"] = xling                    # 记下这次的英文检索式（可追溯/可复现）
    debug["l3"] = dbg_l3
    route = list(state.get("route") or [])
    if "L3" not in route:
        route.append("L3")
    return {
        "l3_chunks": l3_chunks,
        "route": route,
        "debug": debug,
    }
