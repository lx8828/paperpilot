"""L2 expand_l2（定向增量扩展）与 L3 search_l3（独立全文检索）。

L2 · expand_l2
    Judge1 判不够后触发。按 verdict.target_sections（有序候选）→ 定向圆心：
      - 候选节 → 该节内命中 score 最高 claim 的 chunk 为圆心
      - 同 section 内按半径 0 → ±1 → ±2 → ±3… 逐步扩展
      - 每次 expand_l2 调用 = 确定性前进一步，返回当前圆心当前半径的窗口；
        由 router 决定是否回来自环（judge_l2 每步扩完判一次）
      - 圆心到节边界无新增 / 半径超阈值 → 换下一圆心；全部试完/预算耗尽 → done
    state["chunks"] 始终为**当前圆心当前窗口**（分窗隔离）。

L3 · search_l3
    L2 放弃后触发。独立 ChunkIndex 全文检索（不继承 L0~L2 证据），
    结果写 state["l3_chunks"]（与 L2 的 chunks 隔离），供 judge_l3 / answer 使用。

预算与常量（QA_FUNNEL_DESIGN.md §6，首版默认可调）：
    MAX_CENTERS=2       Judge1 有序候选截断数
    MAX_RADIUS=3        单圆心最大半径
    MAX_L2_CHUNKS=6     累计唯一 chunk 预算（预算账本 pulled 长度）
"""
from __future__ import annotations

import os
from typing import Any

from paperpilot.agents.document_cache import (ordered_chunks, section_chunks,
                                              section_from_path)
from paperpilot.agents.embedder import ChunkIndex
from paperpilot.agents.state import QAState
from paperpilot.tools import llm

MAX_CENTERS = 2
MAX_RADIUS = 3
MAX_L2_CHUNKS = 6
# 策略④（圆心诊断 2026-09-08：净亏题 L1 池 70% 含 gold 块、节 70% 命中，
# 但"每节 score 最高单 claim"圆心命中仅 10%——答案块躺在池里被浪费）：
# PAPERPILOT_L2_MULTI_CENTER=1 时，每个 target 节把该节**全部**命中 claim 的
# chunk（按 score 降序）都列入候选圆心（受下面总量上限约束），default 仍每节 1 个。
MAX_MULTI_CENTERS_TOTAL = 8
L3_TOP_K = 12   # 离线实测：top8 只覆盖 gold 证据的 ~73%，top12 在 ~85% 且上下文可控；改到 12

# 检索侧查询改写（只影响"找"，不改作答口径）。PAPERPILOT_QUERY_REWRITE=0 可关闭做 A/B。
_REWRITE_SYS = (
    "你是检索查询改写器。给定一个面向学术论文的问题，生成 2~3 条**仅供检索**的查询变体。\n"
    "规则：\n"
    "1. 抽取并保留关键实体（数据集/语料名、模型/方法名、指标、语言、数字等），这是检索的锚；\n"
    "2. 补同义/上位/常见论文表述（如 'how many utterances' → 'corpus size / number of utterances'；\n"
    "   'is the dataset multilingual?' → 'dataset language composition'）；\n"
    "3. 若问题是英文/面向英文论文，变体用英文；口语或长句改短、去虚词；每条 ≤12 词；\n"
    "4. 变体必须与问题同一语义方向，不要自创新问题。\n"
    '只输出 JSON：{"queries": ["...", "...", "..."]}'
)


def _rewrite_queries(question: str) -> list[str]:
    """返回 [原问题, 变体1, 变体2]；改写失败/被禁用时只返回原问题。

    2026-09-07 A/B（40 题，裁判一致）：改写 ON pass=9 vs OFF pass=12——负收益
    （救回 2 条但弄坏 5 条，含 4 条 OFF 已 pass 被改写带偏），故**默认关闭**；
    需要时用环境变量 PAPERPILOT_QUERY_REWRITE=1 开启做对照实验。
    """
    if os.environ.get("PAPERPILOT_QUERY_REWRITE", "0") != "1":
        return [question]
    try:
        obj = llm.chat_json(_REWRITE_SYS, f"问题：{question}", temperature=0.2)
        extra = []
        if isinstance(obj, dict):
            for q in (obj.get("queries") or []):
                s = str(q).strip()
                if s and len(s) <= 200 and s.lower() != question.lower():
                    extra.append(s)
        return ([question] + extra)[:3]
    except Exception:  # noqa: BLE001（改写失败不影响主链路）
        return [question]


def _section_tail(paths: list[str]) -> str:
    for p in reversed(paths):
        if " · " in p:
            return p.split(" · ", 1)[1].strip()
    return paths[-1] if paths else ""


def _to_chunk_texts(chunks: list[Any]) -> list[dict[str, Any]]:
    """chunk → 可序列化文本切片。注意：**不再截断**。

    chunk 在分块期已按段落原子切到 ~4000 字符（document_cache.MAX_CHUNK_LEN），
    QA 层再截断就是纯丢信息（答案可能落在被砍掉的尾部）。
    单块长度由分块期控制，这里给全文。
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


def _pick_centers(state: QAState,
                  ids_in_sec: dict[str, list[str]]) -> list[str]:
    """按 verdict.target_sections 顺序 + retrieved 命中，产出圆心 chunk_id 列表。

    默认：每个候选节取节内 score 最高命中的 chunk_id 为圆心（候选节无命中则跳过）。
    策略④（PAPERPILOT_L2_MULTI_CENTER=1）：每节取该节全部命中 claim 的 chunk_id
    （按 score 降序、总量 ≤ MAX_MULTI_CENTERS_TOTAL）作多候选圆心——圆心诊断显示
    答案块常在池内但非该节最高分，多候选能直接喂到正确窗口。
    """
    targets = (state.get("verdict") or {}).get("target_sections", [])[:MAX_CENTERS]
    retrieved = state.get("retrieved") or []
    multi = os.environ.get("PAPERPILOT_L2_MULTI_CENTER", "0") == "1"
    # home_section → [(score, chunk_id)]（保 retrieve 序内全部命中）
    hits_by_sec: dict[str, list[tuple[float, str]]] = {}
    for r in retrieved:
        sec = r.get("home_section") or ""
        cid = r.get("chunk_id") or ""
        if not sec or not cid:
            continue
        if cid not in ids_in_sec.get(sec, []):
            continue
        hits_by_sec.setdefault(sec, []).append((float(r.get("score", 0.0)), cid))

    centers: list[str] = []
    seen: set[str] = set()
    for sec in targets:
        if sec in seen:
            continue
        seen.add(sec)
        lst = hits_by_sec.get(sec, [])
        if not lst:
            continue
        lst.sort(key=lambda x: -x[0])
        if multi:
            for _s, cid in lst:
                if len(centers) >= MAX_MULTI_CENTERS_TOTAL:
                    break
                if cid not in centers:
                    centers.append(cid)
        else:
            cid = lst[0][1]
            if cid not in centers:
                centers.append(cid)
    return centers


def _window_at(group_ids: list[str], idx: int, radius: int) -> list[str]:
    lo, hi = max(0, idx - radius), min(len(group_ids), idx + radius + 1)
    return group_ids[lo:hi]


def _next_window(state: QAState, l2: dict[str, Any]) -> dict[str, Any]:
    """把 l2 游标前进一步，产出"该不该给 judge 判"的新窗口。

    步进语义（radius 从 -1 起，首次 advance → 圆心单块 radius=0）：
      当前圆心未判过(radius<0) 或 扩大 radius 有新增 → 推进并返回该窗口；
      当前圆心半径到 MAX_RADIUS 或窗口不再增长（节边界）→ 换下一圆心；
      全部圆心试完 / 预算耗尽 → done=True（不产窗口）。
    返回更新后的 l2（含新 window 或 done）。
    """
    chunks = ordered_chunks(state.get("pdf") or "")
    by_id = {c.chunk_id: c for c in chunks}
    ids_in_sec = {sec: [c.chunk_id for c in clist]
                  for sec, clist in section_chunks(chunks).items()}

    while not l2["done"]:
        if len(l2["pulled"]) >= MAX_L2_CHUNKS or l2["center_i"] >= len(l2["centers"]):
            l2["done"] = True
            break

        cid = l2["centers"][l2["center_i"]]
        c = by_id.get(cid)
        if c is None:
            l2["center_i"] += 1
            l2["radius"] = -1
            continue
        sec = section_from_path(list(c.title_path))
        group_ids = ids_in_sec.get(sec) or []
        try:
            idx = group_ids.index(cid)
        except ValueError:
            l2["center_i"] += 1
            l2["radius"] = -1
            continue

        nr = l2["radius"] + 1
        if nr > MAX_RADIUS:
            # 当前圆心已到最大半径仍不够 → 换下一圆心
            l2["center_i"] += 1
            l2["radius"] = -1
            continue

        nwin = _window_at(group_ids, idx, nr)
        old_win = _window_at(group_ids, idx, l2["radius"]) if l2["radius"] >= 0 else []
        if nr > 0 and set(nwin) <= set(old_win):
            # 无新增（到节边界）→ 换下一圆心
            l2["center_i"] += 1
            l2["radius"] = -1
            continue

        # 预算：新增唯一块不超 MAX_L2_CHUNKS
        new_unique = [x for x in nwin if x not in l2["pulled"]]
        if len(l2["pulled"]) + len(new_unique) > MAX_L2_CHUNKS:
            l2["done"] = True
            break

        l2["radius"] = nr
        l2["window"] = nwin
        for x in new_unique:
            l2["pulled"].append(x)
        return l2  # 产出了新窗口，交给 judge_l2 判
    return l2  # done，无新窗口


def expand_l2(state: QAState) -> dict[str, Any]:
    """L2 步进：前进一次圆心/半径，产出当前窗口 chunks（分窗隔离）。"""
    pdf = state.get("pdf") or ""
    l2_raw = state.get("l2")
    l2: dict[str, Any]
    if l2_raw is None:
        chunks = ordered_chunks(pdf)
        ids_in_sec = {sec: [c.chunk_id for c in clist]
                      for sec, clist in section_chunks(chunks).items()}
        centers = _pick_centers(state, ids_in_sec)
        l2 = {"centers": centers, "center_i": 0, "radius": -1,
              "window": [], "pulled": [], "done": False}
    else:
        l2 = dict(l2_raw)
        l2["pulled"] = list(l2_raw.get("pulled") or [])
        l2["centers"] = list(l2_raw.get("centers") or [])
        l2["window"] = list(l2_raw.get("window") or [])

    l2 = _next_window(state, l2)

    by_id = {c.chunk_id: c for c in ordered_chunks(pdf)}
    window_chunks = [by_id[x] for x in l2["window"] if x in by_id]
    debug = dict(state.get("debug") or {})
    debug["l2_step"] = {
        "n_centers": len(l2["centers"]),
        "center_i": l2["center_i"],
        "radius": l2["radius"],
        "done": l2["done"],
        "budget_used": len(l2["pulled"]),
        "window_n": len(l2["window"]),
        "center": l2["centers"][l2["center_i"]] if not l2["done"] and l2["center_i"] < len(l2["centers"]) else "",
    }

    route = list(state.get("route") or [])
    if l2["done"] and not l2["window"]:
        # 无窗口可判：不算真正进入 L2 判循环，避免 route 重复
        pass
    elif "L2" not in route:
        route.append("L2")

    return {
        "l2": l2,
        "chunks": _to_chunk_texts(window_chunks),
        "route": route,
        "debug": debug,
    }


def search_l3(state: QAState) -> dict[str, Any]:
    """L3 独立全文检索：ChunkIndex 对原始问题检索 topK，写 l3_chunks（干净隔离）。

    2026-09-07 改为多查询混合：查询改写（实体/同义变体）+ 原问题，search_multi_hybrid
    （向量+BM25 × 多查询 RRF）；PAPERPILOT_QUERY_REWRITE=0 可回退单查询 hybrid 做 A/B。
    """
    question = state.get("question") or ""
    idx = ChunkIndex(state.get("pdf") or "")
    queries = _rewrite_queries(question)
    if len(queries) > 1:
        hits = idx.search_multi_hybrid(queries, top_k=L3_TOP_K)
    else:
        hits = idx.search_hybrid(question, top_k=L3_TOP_K)
    l3_chunks: list[dict[str, Any]] = []
    for h in hits:
        l3_chunks.append({
            "chunk_id": h.get("chunk_id", ""),
            "page": h.get("page", 0),
            "section": _section_tail(list(h.get("title_path") or [])),
            "text": (h.get("text") or ""),  # 不截断：分块期已控制单块长度
        })
    debug = dict(state.get("debug") or {})
    debug["l3"] = {"topk": len(l3_chunks)}
    route = list(state.get("route") or [])
    if "L3" not in route:
        route.append("L3")
    return {
        "l3_chunks": l3_chunks,
        "route": route,
        "debug": debug,
    }
