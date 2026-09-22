"""多篇语料 QA 测试（**直连检索层，跳过 L0/路由**）。

为什么直连：L0 是"够不够"的路由，把它接进来只是多一个变量。
本脚本只测一件事 —— **语料 = 5 篇时，检索 + 作答的效果**。

两组问题（语料**都是** 5 篇全部 chunk）：
    A 组「单篇问题」：`qa/questions/*.json` 里属于语料中某篇的题（复用旧标注）
                      → 量「被另外 4 篇干扰后退化多少」
    B 组「跨篇问题」：`qa/multi/*.json`（新写，问题必须横跨多篇）
                      → 量「新能力效果」

判分：`must_have` 全命中 且 `must_not` 无命中（机器可判，不额外花 LLM）。

用法：
    # 语料 = 指定 5 篇；跑它们的单篇问题
    uv run python cli/run_multi_qa.py --pdfs 2408.09273.pdf 2305.14205.pdf \
        2403.13240.pdf 2112.08804.pdf 2305.09220.pdf --limit 12
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "qa"

TOPK = 12
MULTI_QA_DIR = QA_DIR / "multi"

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 同目录工具
from _anchors import hit_any  # noqa: E402


# ── 语料 ─────────────────────────────────────────────────────────────────────


def auto_corpus(paper: str, k: int = 5) -> list[str]:
    """用 `paper` 的标题检索同主题 top-k → **抓取** → 返回语料 PDF 文件名。

    ⚠️ 必须走 `fetch_arxiv`：arXiv 检索回的 id 无版本号（`2609.02094`），
    而论文库里是 `2609.02094v1.pdf` —— 直接拼名字会**全部匹配不上**
    （2026-09-21 首跑踩到：语料只剩主篇 → "不足 2 篇"）。
    """
    import re

    from paperpilot.tools.arxiv_fetch import fetch_arxiv
    from paperpilot.tools.corpus_search import search

    title = ""
    rp = ROOT / "assets/artifacts/out_views" / f"{Path(paper).stem}.report.json"
    if rp.exists():
        title = str(json.loads(rp.read_text(encoding="utf-8")).get("title") or "")
    q = title or Path(paper).stem
    sr = search(q, k=k + 2)

    main_id = re.sub(r"v\d+$", "", Path(paper).stem)
    out: list[str] = [paper] if (ROOT / "assets/papers" / paper).exists() else []
    for p in sr.papers:
        if str(p.arxiv_id) == main_id:
            continue                       # 主篇已在语料里，别重复收同一篇
        f = fetch_arxiv(str(p.arxiv_id))
        if not f.get("ok"):
            print(f"   [语料] ✗ {p.arxiv_id} 抓取失败：{f.get('reason')}")
            continue
        name = str(f["pdf_name"])
        if name not in out:
            out.append(name)
            print(f"   [语料] + {name}（{'缓存' if f['cached'] else '新下载'}）")
        if len(out) >= k:
            break
    return out[:k]


# ── 问题 ─────────────────────────────────────────────────────────────────────


def load_single_questions(corpus: list[str]) -> list[dict[str, Any]]:
    """A 组：**指向某一篇**的题（在整份语料里找答案；复用 `qa/` 下的旧标注）。

    ⚠️ **跳过"用「这篇」泛指、又解析不出是哪一篇"的题**（2026-09-23 加）：
    对着 5 篇语料提问时用户必然指明篇名或编号（「TAAL 那篇」「篇 3」），
    泛指"这篇论文…"**不是目标用法**。这类题在多篇跑批里只会诱使 judge 判
    "指代不明"、把答案变成五篇罗列 → 直接不收（计数打印，便于核对）。
    """
    from paperpilot.components.focus import resolve_focus

    stems = {Path(p).stem for p in corpus}
    qs: list[dict[str, Any]] = []
    for f in ["qa_set_v2.json"] if (QA_DIR / "qa_set_v2.json").exists() else []:
        d = json.loads((QA_DIR / f).read_text(encoding="utf-8"))
        qs += d.get("questions") or []
    for f in sorted(glob.glob(str(QA_DIR / "questions" / "*.json"))):
        qs += json.loads(Path(f).read_text(encoding="utf-8"))
    out = []
    skipped: list[str] = []
    for q in qs:
        pdf = str(q.get("pdf") or "")
        if Path(pdf).stem not in stems:
            continue
        qtext = str(q.get("question") or "")
        if any(k in qtext for k in ("这篇", "该论文", "本文")) and not resolve_focus(qtext, corpus):
            skipped.append(str(q.get("qid") or "?"))
            continue
        out.append({**q, "group": "A", "home": pdf})
    if skipped:
        print(f"  [A组] 跳过 {len(skipped)} 道「泛指且未指明篇」的题"
              f"（多篇语料下不是目标用法）：{', '.join(skipped[:6])}"
              f"{' …' if len(skipped) > 6 else ''}")
    return out


def load_cross_questions() -> list[dict[str, Any]]:
    """B 组：跨篇问题（`qa/multi/*.json`；**跳过 `_` 开头的文件**）。

    ⚠️ 为什么必须跳过：`--out` 若落在本目录，跑批产物同样含 `question` 字段 → 会被
    当成题目**再读回来**。实测 2026-09-23 冒烟把 20 题跑成 22 题（上一次的产物被读回）。
    约定：**产物写到 `qa/multi/_runs/`**（`_` 开头即自动忽略）。
    """
    out: list[dict[str, Any]] = []
    for f in sorted(glob.glob(str(MULTI_QA_DIR / "*.json"))):
        if Path(f).name.startswith("_"):
            continue                                   # 产物/临时文件，不是题库
        for q in json.loads(Path(f).read_text(encoding="utf-8")):
            if not isinstance(q, dict) or "question" not in q or "answer" in q:
                continue                               # 防御：跑批记录不做题
            out.append({**q, "group": "B", "home": ""})
    return out


# ── 判分 ─────────────────────────────────────────────────────────────────────


def keyword_ok(ans: str, must_all: list[str], must_any: list[str],
               must_not: list[str]) -> tuple[bool, list[str]]:
    """判定锚点（与 `cli/run_qa_v2.py` 同语义）：

        must_all  **全部**命中才算过（数字/专名等硬锚点）
        must_any  **任一**命中即可（备选措辞；语料级拒答题的"拒绝措辞"用这一档）
        must_not  出现即判错

    旧 schema 兼容：只有 `must_have`、没有 `must_all`/`must_any` 的题，`must_have`
    按「全部命中」处理（旧口径是"必现"）。

    归一在 `_anchors`（剥 markdown `**没有**报告` → `没有报告` + 折叠空白 + casefold）。
    """
    miss = [k for k in must_all if not hit_any([k], ans)]
    if must_any and not hit_any(must_any, ans):
        miss += [f"any(拒绝/备选措辞 {len(must_any)} 条)全未命中"]
    bad = bool(must_not) and hit_any(must_not, ans)
    return (not miss and not bad), miss


# ── L0 合并总览测试 ──────────────────────────────────────────────────────────


def run_l0(corpus: list[str], qs: list[dict[str, Any]]) -> int:
    """只跑 `report_l0`（合并总览）+ `judge_l0`（预测够不够）。

    要回答：**judge 的"够/不够"预测在 5 篇合并总览下还准不准。**
      · 若对「关于某一篇的泛化问题」（无专名）仍判 enough=True → L0 会直答，
        而那等于让 LLM 在 5 篇里**自己猜是哪篇** → 大概率错（**误判**）。
      · 判据：题面**指明了哪一篇** → `resolve_focus` 收窄 → L0 只喂那篇的总览 →
        判定行为与"语料只有那一篇"一致（这是定位，不是单篇路径）。
    """
    from paperpilot.agents.nodes.answer import generate_answer
    from paperpilot.agents.nodes.judge import judge_l0
    from paperpilot.agents.nodes.report import report_l0
    from paperpilot.components.focus import resolve_focus

    st0 = {"question": "", "pdfs": corpus, "route": [], "debug": {}}
    l0 = report_l0(st0)                                # type: ignore[arg-type]
    print(f"\n[L0] 合并总览：{len(l0['core_points'])} 条要点 ｜ "
          f"{len(l0['overview'])} 字符", flush=True)
    print(f"     标题：{l0['title'][:90]}")

    n_ok = 0
    n_focus = 0
    for q in qs:
        question = str(q.get("question") or "")
        st = {"question": question, "pdfs": corpus, "route": [], "debug": {}}
        st.update(l0)
        # 指代解析：问题里明确指了篇 → 覆盖面收窄（落到 1 篇即回单篇行为）
        focus = resolve_focus(question, corpus)
        n_focus += bool(focus)
        st["focus"] = focus
        st.update(report_l0(st))                       # type: ignore[arg-type]
        v = judge_l0(st)                               # type: ignore[arg-type]
        verdict = v.get("verdict") or {}
        enough = bool(verdict.get("enough"))
        n_ok += enough
        st.update(v)
        try:
            ans = str(generate_answer(st).get("answer") or "")   # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            ans = f"ERR {type(e).__name__}: {e}"
        fstr = ("→ " + ", ".join(Path(p).stem for p in focus)) if focus else "（无指代）"
        print(f"\n  {'够' if enough else '不够'}  {q['group']} "
              f"{str(q.get('qid') or '')[:12]:<12} 指代 {fstr}")
        print(f"      Q: {question[:88]}")
        print(f"      gap: {verdict.get('gap','')[:110]}")
        print(f"      A: {ans[:160]}")

    print(f"\n{'=' * 108}")
    print(f"【L0 汇总】{len(qs)} 题中 judge 判「够」{n_ok} 题 = {n_ok / max(len(qs),1):.0%}"
          f" ｜ 解析出明确指代 {n_focus}/{len(qs)} 题")
    print("  [判据] 指代明确的题 → 收窄到那篇（落到 1 篇即回单篇行为）；"
          "无指代（跨篇 / 「这篇论文」）→ 合并总览")
    return 0


def _apply_focus_floor(hits: list[dict[str, Any]], focus: list[str],
                       question: str) -> list[dict[str, Any]]:
    """**明确指代的篇保底**进候选（从尾部替换，保护全局排序头部）。

    为什么必须做：**编号类指代检索器完全无法理解**（「篇 2」在向量/BM25 里都没有意义）。
    篇名类本来靠 BM25 就能召回（实测主篇 12/12），保底对它无增益、也无害。
    """
    from paperpilot.agents.embedder import ChunkIndex

    have = {(str(h.get("pdf") or ""), str(h.get("chunk_id") or "")) for h in hits}
    slot = len(hits) - 1
    for p in focus:
        for h in ChunkIndex(p).search_hybrid(question, top_k=2):
            if slot < 0:
                break
            key = (p, str(h["chunk_id"]))
            if key in have:
                continue
            hits[slot] = {**h, "pdf": p}
            have.add(key)
            slot -= 1
    return hits


# ── 主流程 ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", nargs="*", default=[], help="语料（5 篇 PDF 文件名）")
    ap.add_argument("--auto", action="store_true", help="用 --paper 的标题检索 top5 当语料")
    ap.add_argument("--paper", default="2408.09273.pdf", help="--auto 时的主篇")
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--limit", type=int, default=0, help="每题组最多跑 N 题（0=不限）")
    ap.add_argument("--cross-only", action="store_true", help="只跑 B 组（跨篇题）")
    ap.add_argument("--l0", action="store_true",
                    help="只跑 L0 合并总览 + judge_l0（验证多篇下「够不够」预测是否可靠）")
    ap.add_argument("--no-focus-floor", action="store_true",
                    help="关掉「明确指代篇」的检索保底（用于 A/B）")
    ap.add_argument("--out", default="", help="落盘路径；⚠️ 别写进 qa/multi/ 根下，用 qa/multi/_runs/xxx.json")
    args = ap.parse_args()

    from paperpilot.agents.embedder import MultiChunkIndex
    from paperpilot.agents.nodes.answer import generate_answer
    from paperpilot.components.focus import resolve_focus   # ⚠️ 检索分支要用（run_l0 里另有局部 import）
    from paperpilot.workflow import _ensure_env

    _ensure_env()

    corpus = list(args.pdfs) or auto_corpus(args.paper)
    if not corpus:
        print("✗ 语料为空")
        return 1
    if not args.l0 and len(corpus) < 2:
        print("✗ 检索测试需要 ≥2 篇语料")
        return 1

    qs: list[dict[str, Any]] = [] if args.cross_only else load_single_questions(corpus)
    qs += load_cross_questions()
    if args.limit:
        a = [q for q in qs if q["group"] == "A"][: args.limit]
        b = [q for q in qs if q["group"] == "B"][: args.limit]
        qs = a + b

    print("=" * 108)
    print(f"[语料] {len(corpus)} 篇（多篇语料，语义切块）")
    for p in corpus:
        print(f"   · {p}")
    print(f"[问题] A组(指向某一篇) {sum(1 for q in qs if q['group'] == 'A')} 题 ｜ "
          f"B组(跨篇) {sum(1 for q in qs if q['group'] == 'B')} 题", flush=True)

    if args.l0:
        return run_l0(corpus, qs)

    idx = MultiChunkIndex(corpus)
    n_units = len(idx._doc_chunks())
    print(f"[语料] 单元数 {n_units}", flush=True)

    recs: list[dict[str, Any]] = []

    # ── M0（多篇·免检索）必须走 L0 路径 ──────────────────────────────────────────
    # 依据（2026-09-23 冒烟实测）：M0 的定义就是"合并总览即可答"，投进检索路径会被
    # **召回不足**污染 —— M0-1/M0-2/M0-4 都因某一篇没被召回而只答出一半。故按 kind 分流：
    #     M0        → report_l0(合并总览) → generate_answer（不走检索）
    #     M1 / X5   → 多语料检索 → generate_answer
    l0_state: dict[str, Any] = {}
    if any(str(q.get("kind") or "") == "M0" for q in qs):
        from paperpilot.agents.nodes.report import report_l0

        l0_state = dict(report_l0({"question": "", "pdfs": corpus,      # type: ignore[arg-type]
                                   "route": [], "debug": {}}))
        ov = str(l0_state.get("overview") or "")
        print(f"[L0] 合并总览：{len(l0_state.get('core_points') or [])} 条要点 ｜ {len(ov)} 字符",
              flush=True)

    for q in qs:
        question = str(q.get("question") or "")
        kind = str(q.get("kind") or "")
        if kind == "M0" and l0_state:
            hits: list[dict[str, Any]] = []
            state: dict[str, Any] = {**l0_state, "question": question, "pdfs": corpus,
                                     "route": ["L0"], "debug": {}}
            mode = "L0"
        else:
            focus = resolve_focus(question, corpus)
            hits = idx.search_hybrid(question, top_k=args.topk)
            if focus and not args.no_focus_floor:
                hits = _apply_focus_floor(hits, focus, question)
            l3 = [{"chunk_id": h.get("chunk_id", ""), "page": h.get("page", 0),
                   "section": (list(h.get("title_path") or [])[-1] if h.get("title_path") else ""),
                   "text": h.get("text") or "", "pdf": h.get("pdf", "")}
                  for h in hits]
            state = {
                "question": question,
                "pdfs": corpus,                 # **语料是多篇**（本系统没有单篇路径）
                "title": "",
                "overview": "",
                "l3_chunks": l3,
                "route": ["L3"],
                "debug": {},
            }
            mode = "retrieval"
        t0 = time.time()
        try:
            out = generate_answer(state)              # type: ignore[arg-type]
            answer = str(out.get("answer") or "")
        except Exception as e:  # noqa: BLE001
            answer = f"ERR {type(e).__name__}: {e}"
        all_ = [str(x) for x in (q.get("must_all") or [])]
        any_ = [str(x) for x in (q.get("must_any") or [])]
        if not all_ and not any_:                               # 旧 schema 兼容
            all_, any_ = [str(x) for x in (q.get("must_have") or [])], []
        ok, miss = keyword_ok(answer, all_, any_, q.get("must_not") or [])
        dist = Counter(Path(str(h.get("pdf") or "")).stem for h in hits)
        recs.append({**q, "answer": answer, "ok": ok, "miss": miss,
                     "mode": mode, "dist": dict(dist), "n_hits": len(hits),
                     "seconds": round(time.time() - t0, 1)})
        tag = "✅" if ok else "⚠️"
        if mode == "L0":
            print(f"  [{tag}] {q['group']} {str(q.get('qid') or '')[:12]:<12} "
                  f"[L0 免检索 · 合并总览]", flush=True)
        else:
            print(f"  [{tag}] {q['group']} {str(q.get('qid') or '')[:12]:<12} "
                  f"命中篇数 {len(dist)} ｜ {dict(dist)}", flush=True)
        print(f"        Q: {question[:88]}")
        print(f"        A: {answer[:150]}")

    # ── 汇总 ──
    print("\n" + "=" * 108)
    print(f"【汇总】语料={len(corpus)}篇 {n_units}单元")
    for g, name in (("A", "指向某一篇的题@多篇语料"), ("B", "跨篇问题（新能力）")):
        gs = [r for r in recs if r["group"] == g]
        if not gs:
            continue
        okn = sum(1 for r in gs if r["ok"])
        np_ = sum(len(r["dist"]) for r in gs) / len(gs)
        print(f"  {name:<18} ✅ {okn}/{len(gs)} = {okn / len(gs):.0%}"
              f"  ｜ 平均命中篇数 {np_:.1f}")
        if g == "B":                       # 组级题按类分开报（三类含义完全不同，别混着看）
            ks = Counter(str(r.get("kind") or "?") for r in gs)
            parts = [f"{k} {sum(1 for r in gs if str(r.get('kind')) == k and r['ok'])}/{ks[k]}"
                     for k in ("M0", "M1", "X5") if ks.get(k)]
            print(f"  {'':<18} B 组分类：{' ｜ '.join(parts)}"
                  f"   （M0 多篇免检索 / M1 跨篇需检索 / X5 语料级拒答）")
    if args.out:
        Path(args.out).write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"  → 落盘 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
