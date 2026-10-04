"""多篇语料 QA 测试（**直连检索层，跳过 L0/路由**）。

为什么直连：L0 是"够不够"的路由，把它接进来只是多一个变量。
本脚本只测一件事 —— **语料 = 5 篇时，检索 + 作答的效果**。

两组问题（语料**都是** 5 篇全部 chunk）：
    A 组「单篇问题」：`qa/questions/*.json` 里属于语料中某篇的题（复用旧标注）
                      → 量「被另外 4 篇干扰后退化多少」
    B 组「跨篇问题」：`qa/multi/*.json`（新写，问题必须横跨多篇）
                      → 量「新能力效果」

⚠️ **检索路径**（2026-09-23 接线）：生产 L3 走 `search_layered`（篇内检索+跨篇 quota
融合），本脚本此前恒走 `search_hybrid`（全局混池）＝**非生产路径**。
验收必须加 `--retrieval layered`；默认仍为 hybrid 只为保留旧基线可比。

⚠️ **B 组按语料过滤**（2026-09-23 修）：`qa/multi/*.json` 含**所有分组**的题集，
传给 `load_cross_questions` 的 corpus 会把不属于本语料的题剔除
（否则跑 group2 语料会混进 group1 的题 → 必然判错）。

判分：`must_have` 全命中 且 `must_not` 无命中（机器可判，不额外花 LLM）。

用法：
    # 语料 = 指定 5 篇；跑它们的单篇问题
    uv run python cli/eval/run_multi_qa.py --pdfs 2408.09273.pdf 2305.14205.pdf \
        2403.13240.pdf 2112.08804.pdf 2305.09220.pdf --limit 12
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

ROOT = Path(__file__).resolve().parents[2]
QA_DIR = ROOT / "qa"

TOPK = 12
MULTI_QA_DIR = QA_DIR / "multi"


# ── gold 隔离（结构性防泄露，2026-09-29）──────────────────────────────────────
# 为什么必须有：题集文件里 **`hint`（=出题者摘录的答案原文）与 `must_all` 同处**
# （`qa/multi/*.json` / `qa/questions/*.json`）。任何新 runner 只要写一句
# `q["hint"]`，就会把答案直接喂进 prompt → 瞬间满分，且**看不出是泄露**。
# 实测（`retrieval/tmp/_prod_leak_audit.py`）：运行时 0 处读 `hint`，但这是
# **偶然安全**，不是结构性安全。故把"pipeline 侧输入"收成函数 + 边界守卫。
GOLD_KEYS: tuple[str, ...] = (
    "hint", "evidence", "must_all", "must_any", "must_have", "must_not",
)


def gold_of(q: dict[str, Any]) -> dict[str, Any]:
    """题里的 **gold-only** 字段（只允许判分/诊断用；**绝不进 pipeline**）。"""
    return {k: q[k] for k in GOLD_KEYS if k in q}


def assert_no_gold(obj: Any, where: str) -> None:
    """边界守卫：pipeline 侧输入里出现任何 gold 字段 → **抛错**。

    宁可响亮失败，也不要"悄悄把答案喂进 prompt"（吞异常最贵，见
    `RAG_COMPONENT_NOTES` 的教训）。
    """
    if isinstance(obj, dict):
        bad = [k for k in obj if k in GOLD_KEYS]
        if bad:
            raise AssertionError(
                f"[gold-guard] {where}：pipeline 输入里出现 gold 字段 {bad}。"
                "这些字段只允许用于判分/诊断，绝不能进检索或 prompt。")
        for k, v in obj.items():
            assert_no_gold(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_gold(v, f"{where}[{i}]")


def pipeline_state(question: str, corpus: list[str], **extra: Any) -> dict[str, Any]:
    """构造**唯一许可**的 pipeline state（并过 guard）。

    新 runner 一律走这里，别自己拼 dict 再塞 `**q`（那就是泄露入口）。
    """
    st: dict[str, Any] = {**extra, "question": str(question), "pdfs": list(corpus)}
    assert_no_gold(st, "pipeline_state")
    return st


def _stem(name: str) -> str:
    """论文 stem 归一：**只剥 `.pdf` 后缀**。

    ⚠️ 不能用 `Path(...).stem`：论文 id 里带点（`2604.20087`），
    `Path('2604.20087').stem` 会剥掉 `.20087` → `'2604'`，
    与 `Path('2604.20087.pdf').stem`（=`'2604.20087'`）**对不上**
    （2026-09-23 实测：按语料过滤 B 组时把 40 题全滤成 0 道）。
    """
    s = str(name).strip()
    return s[:-4] if s.lower().endswith(".pdf") else s

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


def load_cross_questions(corpus: list[str] | None = None) -> list[dict[str, Any]]:
    """B 组：跨篇问题（`qa/multi/*.json`；**跳过 `_` 开头的文件**）。

    ⚠️ 为什么必须跳过：`--out` 若落在本目录，跑批产物同样含 `question` 字段 → 会被
    当成题目**再读回来**。实测 2026-09-23 冒烟把 20 题跑成 22 题（上一次的产物被读回）。
    约定：**产物写到 `qa/multi/_runs/`**（`_` 开头即自动忽略）。

    ⚠️⚠️ **2026-09-23 修：必须按语料过滤**。`qa/multi/*.json` 是**全部分组**的题集
    （group1.json / group2.json …），而本函数此前把它们**全部读进来** → 跑 group2 的
    5 篇语料时会混进 group1 的 20 题（问的是另外 5 篇论文，语料里根本没有 → 必然判错）
    → B 组通过率被稀释一半，且 M0/M1/X5 分类统计是两组的混合物（**假测试**）。
    实测：指定 group2 语料、`--limit 1` 跑出来的第一题是 `G1-M0-1`。
    过滤依据：导出时每题写入的 `corpus` 字段（`_export_questions.export_group`）。
    """
    want = {_stem(s) for s in (corpus or [])}
    out: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for f in sorted(glob.glob(str(MULTI_QA_DIR / "*.json"))):
        if Path(f).name.startswith("_"):
            continue                                   # 产物/临时文件，不是题库
        for q in json.loads(Path(f).read_text(encoding="utf-8")):
            if not isinstance(q, dict) or "question" not in q or "answer" in q:
                continue                               # 防御：跑批记录不做题
            qc = {_stem(x) for x in (q.get("corpus") or [])}
            if want and qc and qc != want:             # 为别的语料写的题 → 不收
                skipped[Path(f).name] = skipped.get(Path(f).name, 0) + 1
                continue
            out.append({**q, "group": "B", "home": "", "_src": Path(f).name})
            used[Path(f).name] = used.get(Path(f).name, 0) + 1
    print(f"  [B组] 收题 {len(out)} 道，来自 {used or '（无）'}"
          + (f"；⏭️ 跳过不属于本语料的 {sum(skipped.values())} 道 {skipped}"
             if skipped else ""), flush=True)
    return out


# ── 判分 ─────────────────────────────────────────────────────────────────────


_BRIDGE: dict[str, list[str]] | None = None


def anchor_bridge() -> dict[str, list[str]]:
    """**中英桥接表**：中文锚点 → 该锚点在论文原文里的英文表述（见 `retrieval/tmp/_anchor_bridge.py`）。

    为什么需要（2026-09-27 逐题复核实测）：`must_all`/`must_any` 里有些锚点是**中文**
    （如「技能门控」），而"全上下文"路径倾向**逐字照抄英文原词**（`skill gating mechanism`）
    → 字面不匹配 → 判错。而"照抄原文"恰恰是我们要的行为 → 所以**修判分侧、不动 gold**。
    表里的每条英文表述都由脚本**逐字校验过出现在论文原文里**（不是拍脑袋翻译）。
    """
    global _BRIDGE
    if _BRIDGE is None:
        p = Path(__file__).resolve().parents[2] / "qa" / "anchors_zh_en.json"
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            _BRIDGE = {str(k): [str(x) for x in v] for k, v in raw.items()}
        except Exception:  # noqa: BLE001  表不存在 → 退回严格匹配
            _BRIDGE = {}
    return _BRIDGE


def _alts(k: str) -> list[str]:
    """锚点的可接受写法 = 锚点本身 + 它的原文英文表述。"""
    return [k] + anchor_bridge().get(k, [])


_QUOTE_SPAN = re.compile(r"[「『“\"'‘][^」』”\"'’]{0,300}[」』”\"'’]")


def strip_quotes(text: str) -> str:
    """去掉**引述片段**（引号内）。

    用途：`must_not` 是"出现即判错"的陷阱锚点，但**引述原文来反驳**是合法行为
    （实测 `0087-N1`：答案先写"不对"，再引 `"scaling to stronger LLMs does not reliably help"`
    → 引述被当成命中陷阱）。去掉引号内内容后再查陷阱。
    """
    return _QUOTE_SPAN.sub(" ", text)


def keyword_ok(ans: str, must_all: list[str], must_any: list[str],
               must_not: list[str]) -> tuple[bool, list[str]]:
    """判定锚点（原与 `run_qa_v2.py` 同语义，该脚本已不在仓库；现由本模块定义、
    `run_group_qa` 复用）：

        must_all  **全部**命中才算过（数字/专名等硬锚点）
        must_any  **任一**命中即可（备选措辞；语料级拒答题的"拒绝措辞"用这一档）
        must_not  出现即判错

    旧 schema 兼容：只有 `must_have`、没有 `must_all`/`must_any` 的题，`must_have`
    按「全部命中」处理（旧口径是"必现"）。

    归一在 `_anchors`（剥 markdown `**没有**报告` → `没有报告` + 折叠空白 + casefold）。
    """
    miss = [k for k in must_all if not hit_any(_alts(k), ans)]
    if must_any and not hit_any([a for k in must_any for a in _alts(k)], ans):
        miss += [f"any(拒绝/备选措辞 {len(must_any)} 条)全未命中"]
    bad = bool(must_not) and hit_any(must_not, strip_quotes(ans))
    return (not miss and not bad), miss


def score_answer(q: dict[str, Any], answer: str, cites: list[dict[str, Any]] | None = None
                 ) -> tuple[bool, list[str], bool]:
    """**唯一判分口径**（`run_multi_qa` / `run_group_qa` 共用）。

    ⚠️ 为什么必须抽出来：判分此前散在两个 runner 里各写一份，加"引用通道"时只改了
    一处 → 同一份 gold 题在两个 runner 下**判分松紧不同**（假测试）。
    runner 只负责"拿到答案"，判分一律走这里。

    schema（`_export_questions.py` 产物）：`must_all` 全部命中 + `must_any` 任一命中
    + `must_not` 出现即错；旧 schema（只有 `must_have`）按"必现"处理。

    判分文本 = **答案 + 答案自己引用的证据原文**（`cites[].evidence`）：
    锚点多为英文专名/术语而答案是中文（`VAE`→"自编码器"、`topology`→"拓扑"），
    只匹配答案会必然落空（实测 M1 0/10、A 组约 24 题误判）。只拼"答案引用到的"
    证据 → 答错篇 / 没引用到证据仍判错，不放水。

    返回 `(ok, miss, ok_strict)`；`ok_strict` = 仅看答案（不含引用通道），用于量化通道贡献。
    """
    all_ = [str(x) for x in (q.get("must_all") or [])]
    any_ = [str(x) for x in (q.get("must_any") or [])]
    if "must_all" in q or "must_any" in q:
        # 新 schema：`must_have` = gold.must_any「任一命中」（组级导出还把 must_all 写进 must_have）
        # ⚠️ 旧写法 `if not all_ and not any_: all_, any_ = must_have, []` 把 must_have 当**必现**
        #    → 单篇层拒答题（must_all=[] + must_have=8 条拒答措辞）被要求"8 条全中" → 10/10 误判。
        any_ = any_ or [str(x) for x in (q.get("must_have") or [])]
    else:                                                       # 旧 schema：must_have = 必现
        all_ = [str(x) for x in (q.get("must_have") or [])]
    judge = answer
    ev = [str(c.get("evidence") or "") for c in (cites or []) if c.get("evidence")]
    if ev:
        judge = answer + "\n" + "\n".join(ev)
    ok_judge, miss = keyword_ok(judge, all_, any_, [])
    # `must_not` 是**答案层的陷阱**（出现即判错），只看答案：证据原文里有陷阱词是正常的
    # （如 2094-N1 的 "提升 8.3" —— 论文正文本来就有 8.3）。
    bad = bool(q.get("must_not")) and hit_any(q["must_not"], strip_quotes(answer))
    if bad:
        miss = miss + ["must_not 命中"]
    ok_strict = keyword_ok(answer, all_, any_, [])[0] and not bad
    return (ok_judge and not bad), miss, ok_strict


# ── L0 合并总览测试 ──────────────────────────────────────────────────────────


def run_l0(corpus: list[str], qs: list[dict[str, Any]]) -> int:
    """只跑 `report_l0`（合并总览）+ `judge_l0`（预测够不够）。

    要回答：**judge 的"够/不够"预测在 5 篇合并总览下还准不准。**
      · 若对「关于某一篇的泛化问题」（无专名）仍判 enough=True → L0 会直答，
        而那等于让 LLM 在 5 篇里**自己猜是哪篇** → 大概率错（**误判**）。
      · 判据：题面**指明了哪一篇** → `resolve_focus` 收窄 → L0 只喂那篇的总览 →
        判定行为与"语料只有那一篇"一致（这是定位，不是单篇路径）。

    ⚠️ 调用方**只传 B 组**（M0/M1/X5）：这三类才有"应判够/应判不够"的对照语义。
    混入 A 组会把统计稀释成无意义的平均数（见 `main` 里 `--l0` 分支的注释）。
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


def _retrieve(idx: Any, question: str, args: Any) -> tuple[list[dict[str, Any]], str]:
    """按 `--retrieval` 选检索路径，返回 `(hits, path_tag)`。

    ⚠️ **2026-09-23 接线**：生产 L3（`agents/nodes/pull_chunk.py`）走的是
    `search_layered`（篇内先检索 → 跨篇 quota 融合），而本脚本此前恒走
    `search_hybrid`（全局混池）→ **测的不是生产路径**（假测试）。
    为什么全局混池在多篇下不可用：5 篇总字符数相近但 chunk 数 22~52、中位块长
    差 5.8× → 全局 cosine 实际在比"**谁切得细**"而不是"谁相关"，泛化查询 top12
    被单篇 100% 霸占、把 k 从 12 加到 48 也没用（见
    `MultiChunkIndex.search_layered` 的 docstring 实测）。
    """
    if args.retrieval == "layered":
        hits = idx.search_layered(question, per_paper_k=args.per_paper_k,
                                  top_k=args.topk, mode=args.mode, floor=args.floor)
        return hits, f"layered/{args.mode}"
    return idx.search_hybrid(question, top_k=args.topk), "hybrid"


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
    # ── 检索路径（2026-09-23 接线）─────────────────────────────────────────────
    # ⚠️ 此前本脚本恒走 `search_hybrid`（全局混池），而生产 L3 走 `search_layered`
    #    → 测的不是生产路径。默认值保持 hybrid（不改既有基线口径），用 layered 做 A/B。
    ap.add_argument("--retrieval", choices=["hybrid", "layered"], default="hybrid",
                    help="检索路径：hybrid=全局混池（旧行为/默认）；"
                         "layered=篇内检索+跨篇 quota 融合（**生产 L3 主路径**）")
    ap.add_argument("--mode", choices=["quota", "rrf", "znorm", "zmean", "global"],
                    default="quota", help="--retrieval layered 的跨篇融合模式（默认 quota=实测最优）")
    ap.add_argument("--per-paper-k", type=int, default=4,
                    help="⚠️ **只在 --mode rrf 生效**：quota/znorm/zmean 下 `search_layered` "
                         "不读该参数（见 `_search_quota` 签名）—— 别把它当篇级配额调")
    ap.add_argument("--floor", type=int, default=1,
                    help="--retrieval layered + mode=quota 时每篇保底进候选的块数")
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
    qs += load_cross_questions(corpus)          # ⚠️ 必须传 corpus：按语料过滤掉别的组的题
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
        # ⚠️ 2026-09-23 修：**只喂 B 组**。此前把 A 组 70 题也丢进 judge_l0 统计，
        # 而 A 组是"指向某一篇"的题、没有 M0/M1 那样的负向对照意义
        # → 汇总的"judge 判够 x%"被 70 题稀释，没有任何解释力（口径污染）。
        b = [q for q in qs if q["group"] == "B"]
        print(f"[--l0] 只跑 B 组（M0 应判「够」/ M1 应判「不够」/ X5 拒答）：{len(b)} 题"
              f"（跳过 A 组 {len(qs) - len(b)} 题）", flush=True)
        return run_l0(corpus, b)

    idx = MultiChunkIndex(corpus)
    n_units = len(idx._doc_chunks())
    print(f"[语料] 单元数 {n_units}", flush=True)
    # ⚠️ 生产 L3（`agents/nodes/pull_chunk.py`）走 layered；hybrid 跑批 = 测非生产路径。
    _prod = "layered/quota"
    if args.retrieval == "layered":
        print(f"[检索] layered/{args.mode}（top_k={args.topk} per_paper_k={args.per_paper_k} "
              f"floor={args.floor}）｜ 生产 L3 = {_prod} ✓ 同路径", flush=True)
    else:
        print(f"[检索] ⚠️ hybrid（全局混池）＝ **非生产路径**（生产 L3 = {_prod}）"
              f"→ 只能当旧基线，**不能当验收**；加 `--retrieval layered` 才是生产口径。", flush=True)

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
            # ⚠️ 合并了 `l0_state` 后**必须再过一次 guard**（l0_state 来自 report_l0，
            #    不能假设它永远干净）。见文件头「gold 隔离」。
            assert_no_gold(state, f"{q.get('qid')}/state-L0")
            mode = "L0"
            rpath = "L0"
        else:
            focus = resolve_focus(question, corpus)
            hits, rpath = _retrieve(idx, question, args)
            if focus and not args.no_focus_floor:
                hits = _apply_focus_floor(hits, focus, question)
            l3 = [{"chunk_id": h.get("chunk_id", ""), "page": h.get("page", 0),
                   "section": (list(h.get("title_path") or [])[-1] if h.get("title_path") else ""),
                   "text": h.get("text") or "", "pdf": h.get("pdf", "")}
                  for h in hits]
            state = pipeline_state(
                question, corpus,               # **语料是多篇**（本系统没有单篇路径）
                title="", overview="", l3_chunks=l3, route=["L3"], debug={})
            mode = "retrieval"
        t0 = time.time()
        try:
            out = generate_answer(state)              # type: ignore[arg-type]
            answer = str(out.get("answer") or "")
        except Exception as e:  # noqa: BLE001
            answer = f"ERR {type(e).__name__}: {e}"
        # 判分一律走共享口径（`score_answer`），别在这里重写
        ok, miss, ok_strict = score_answer(q, answer, list(out.get("cites") or []))
        dist = Counter(Path(str(h.get("pdf") or "")).stem for h in hits)
        recs.append({**q, "answer": answer, "ok": ok, "ok_strict": ok_strict,
                     "miss": miss, "mode": mode, "dist": dict(dist),
                     "retrieval": rpath,
                     "n_hits": len(hits), "seconds": round(time.time() - t0, 1)})
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
    print(f"【汇总】语料={len(corpus)}篇 {n_units}单元 ｜ "
          f"检索={'layered/' + args.mode if args.retrieval == 'layered' else 'hybrid'}"
          f"{'' if args.retrieval == 'layered' else '（非生产路径）'}")
    for g, name in (("A", "指向某一篇的题@多篇语料"), ("B", "跨篇问题（新能力）")):
        gs = [r for r in recs if r["group"] == g]
        if not gs:
            continue
        okn = sum(1 for r in gs if r["ok"])
        stn = sum(1 for r in gs if r.get("ok_strict"))
        np_ = sum(len(r["dist"]) for r in gs) / len(gs)
        print(f"  {name:<18} ✅ {okn}/{len(gs)} = {okn / len(gs):.0%}"
              f"  ｜ 平均命中篇数 {np_:.1f}"
              f"  ｜ 仅答案判分 {stn}/{len(gs)} = {stn / len(gs):.0%}"
              + (f"（引用通道 +{okn - stn}）" if okn != stn else ""))
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
