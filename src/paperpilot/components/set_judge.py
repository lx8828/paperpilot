"""**集合问答核心**：「这 N 篇里，哪几篇做了 X」→ 篇集合 + 逐篇证据。

## 为什么需要它（R2 线的实证结论）
`retrieval/results/R2_READER_20260929.md`：把"一次看 K 篇"换成 **逐篇判定（map）**，

| 指标 | reader（逐篇判定） | 检索 top-10 基线 |
|---|---|---|
| 集合 P | **0.88** | 0.46 |
| 集合 R | 0.80 | 0.85 |
| **集合 F1** | **0.82** | **0.54** |

→ **瓶颈从来不是召回（`StRecall@10` 早 0.85），而是"这篇到底做没做"的判定**。
→ 交付篇数还能**自适应**（逐篇判定后 5.2~5.9 篇 vs gold 5.9），不再需要人为定 `top_k`。

## 形态（map → 聚合）
1. **检索**：`MultiChunkIndex.search_hybrid` 拿全局序 → 按 `pdf` 分组 → 每篇取 top-`b` 块
2. **map**：**逐篇 1 次调用**（并发）→ `{label: yes|no|unclear, evidence:[片段号], why}`
3. **聚合**：判 `yes` 的篇 = 交付集合；每篇附证据片段号 + 检索分数

## 判定提示的四条纪律（来自真值校准的实测，勿删）
1. 只看给出的片段，不用先验知识补充
2. **区分「本文做的」与「引用他人做的」**（`X et al. proposed…` → no）← 最常见的错法
3. **列出/提及 ≠ 做了**（任务清单、相关工作、未来工作里出现该词 → no）
4. 中文论断 ↔ 英文片段，注意同义改写

## 成本
**每题 N 次小调用**（N = 篇数，**单判官**；2026-10-01 起默认）—— 实测 b=12 时 30 真值
× 20 篇 = 600 次 ≈ ¥1.2。旧的双判官协议是 **2.1N** 次，与单判官**质量等价**
（`retrieval/results/R2_JUDGE_PROTO_20261001.md`）→ 已改默认为单判官。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from paperpilot.tools import llm

# 每篇给判官的证据块数（实测：3→0.736 / 6→0.817 / 12→0.846 F1，越大越好但有上限）
DEFAULT_B = 12
DEFAULT_WORKERS = 6

SYS_SET = """你要判断**某一篇论文**是否做了某件事。你会看到**该论文自己的**若干原文片段。

只输出三档之一：
- `yes`：该论文**自己做了**这件事（片段里有本文自己的做法/实验/设置）。
- `no`：该论文没有做这件事；或者这件事**只是本文引用的他人工作**；或者只是**任务清单/评价指标/
  相关工作/未来工作**里出现的一个词，本文自己并没有做。
- `unclear`：片段不足以判断（**不要**因为"没看到"就判 no，但也**不要**因为"看起来相关"就判 yes）。

判定纪律：
1. 只看给出的片段，**不要用先验知识补充**。
2. ⚠️ **区分「本文做的」与「引用他人做的」**：若动作的主语是被引用的他文（`X et al. proposed…`、
   `Prior work…`）且**没说明本文也做了** → `no`。这是最常见的错法。
3. **列出/提及 ≠ 做了**：只是清单里出现该词 → `no`。
4. 中文论断 ↔ 英文片段，注意同义改写（"做了消融"↔"we ablate each component"）。

只输出 JSON：{"label":"yes|no|unclear","evidence":[片段序号...],"why":"≤30 字中文理由"}"""


# ⚠️ **`extract` 模式**（2026-09-29 补）：`SYS_SET` 只答"哪几篇"，答不了"**各自是什么**"。
#    实测（`R2_SC_VERDICT_20260929.md` §3.1）：M1 是「集合筛选 ∧ 内容枚举」的复合题
#    （"哪些篇做了消融？**各自消融掉的是什么**？"）→ 仅判定只有 6~7/50，锚点全缺。
#    → 故在判定之外**多要一个 `extract` 字段**：该篇在本题上的具体内容（数字/专名逐字照抄）。
SYS_SET_EXTRACT = """你要判断**某一篇论文**是否做了某件事，**并给出该篇的具体内容**。你会看到**该论文自己的**若干原文片段。

**label** 三档：
- `yes`：该论文**自己做了**这件事（片段里有本文自己的做法/实验/设置）。
- `no`：该论文没有做；或者这件事**只是本文引用的他人工作**；或者只是**任务清单/评价指标/
  相关工作/未来工作**里出现的一个词，本文自己并没有做。
- `unclear`：片段不足以判断（**不要**因为"没看到"就判 no，但也**不要**因为"看起来相关"就判 yes）。

判定纪律：
1. 只看给出的片段，**不要用先验知识补充**。
2. ⚠️ **区分「本文做的」与「引用他人做的」**：若动作的主语是被引用的他文（`X et al. proposed…`、
   `Prior work…`）且**没说明本文也做了** → `no`。这是最常见的错法。
3. **列出/提及 ≠ 做了**：只是清单里出现该词 → `no`。
4. 中文论断 ↔ 英文片段，注意同义改写（"做了消融"↔"we ablate each component"）。

⚠️ **extract**（`label=="yes"` 时必填；否则写 `""`）：
用**中文一句话**回答题干里「**各自…是什么 / 怎么做 / 用了什么 / 数字是多少**」这类**具体内容**问句。
- **必须逐字照抄原文里的数字、指标名、方法名、数据集名、工具名**（原文是英文就保留英文原词，不要只给中文意译）。
- 片段里没有该具体信息就写「片段未给出」，**不要编**。
- ≤60 字，只写该篇的内容，不要罗列其他篇。

只输出 JSON：{"label":"yes|no|unclear","extract":"…","evidence":[片段序号...],"why":"≤30 字中文理由"}"""


# ── ★ 双判官第三轮（分歧/未决时启用）：对抗式复核 ─────────────────────
#    来自真值校准协议（`retrieval/tmp/_r2_gold_recalib.py::SYS_STRICT`），词表适配本模块的
#    `yes|no` 二值输出。**只在 A/B 分歧或出现 `unclear` 时才调用**（实测约 25~30%）。
#    ⚠️ 2026-10-01：**默认已关**（`PAPERPILOT_SET_DUAL=0`）—— 本环节只在
#    `PAPERPILOT_SET_DUAL=1` 时才会走到。保留代码是为了能复现旧协议的对照实验。
SYS_SET_STRICT = """你是**审稿人式的严格核查员**。任务：判断「某篇论文**自己**做了某件事」是否成立。

你会看到该论文的若干原文片段。请执行一次**对抗式复核**：
- 主动寻找"这**不算**本文做了这件事"的理由：是不是**引用的他人工作**？是不是只是**任务清单/
  评价指标/相关工作/未来工作**里的一个词？是不是只在**泛泛描述背景**而没有本文自己的做法？
- 只有在**排除上述全部可能**后，才判 YES。
- ⚠️ 中文论断 ↔ 英文片段，注意同义改写（"做了消融" ↔ "we ablate each component"）。

只输出 JSON：{"answer":"YES|NO","why":"≤30 字中文理由"}"""

# 证据窗口字符上限（定稿值 4,200；见 `R2_FINAL_SPEC_20260930.md`）
DEFAULT_WIN_CHARS = 4200


def env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


def per_paper_hits(idx: Any, question: str, *, b: int = DEFAULT_B,
                   top_k: int | None = None,
                   claim: str | None = None,
                   top_papers: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """一次检索 → 按篇分组 → 每篇保留检索序前 `b` 块。

    ⚠️ 用 `search_*(question, top_k=全部块)` 拿**全局序**再分组
    （与 `retrieval/tmp/_scan_quota.py` 同口径）：这样每篇的"前 b 块"是**同一把尺子**下的，
    与逐篇检索的排序一致，且只编码一次查询。

    ## 检索口径：`PAPERPILOT_SET_RETR`（默认 **`multiroute`** = R2 定稿）
    | 值 | 做法 |
    |---|---|
    | **`multiroute`**（默认） | **多路子查询 + 加权 RRF**：`zh` 走 1 路 dense；LLM 分解出的 ≤3 条英文子查询各走 dense + BM25；每路先截 top-d、按类加权（d=100/50/50，bm25 权重 4×）。见 `R2_FINAL_SPEC_20260930.md` |
    | `hybrid` | 旧行为：单查询 `search_hybrid`（向量+BM25 等权 RRF） |

    `claim`：用于**生成子查询**的论断（不传则用 `question`）。定稿是用**中文论断**生成
    （`query_optimizer.decompose`），而不是用问题原文。
    """
    n_all = len(idx._doc_chunks())
    tk = min(top_k or n_all, n_all)
    mode = (os.environ.get("PAPERPILOT_SET_RETR") or "multiroute").strip().lower()
    subs: list[str] = []
    if mode == "multiroute" and hasattr(idx, "search_multiroute"):
        from paperpilot.components import query_optimizer as _qo   # 惰性导入（避免环）
        subs = _qo.decompose(claim or question)
        hits = idx.search_multiroute(zh=claim or question, subs=subs, top_k=tk)
    else:
        hits = idx.search_hybrid(question, top_k=tk)
    out: dict[str, list[dict[str, Any]]] = {str(p): [] for p in idx.pdfs}
    for h in hits:
        p = str(h.get("pdf") or "")
        if p in out and len(out[p]) < b:
            out[p].append(h)
    if subs:
        print(f"[set_judge] 多路检索：1 中文 + {len(subs)} 子查询 × (dense+BM25) ｜ "
              f"命中 {len(hits)} 块 → {sum(1 for v in out.values() if v)} 篇")
    res = {p: v for p, v in out.items() if v}
    # ★ 候选篇上限（`PAPERPILOT_SET_TOPK`）：按**篇内最高块分**排序取前 N。
    #   定稿口径是"交付 top-k 篇"（`R2_FINAL_SPEC`，k=30）；不设上限 = 判定所有有命中的篇。
    if top_papers and len(res) > top_papers:
        keep = sorted(res, key=lambda p: -float(res[p][0].get("score") or 0.0))[:top_papers]
        res = {p: res[p] for p in keep}
    return res


def build_user(claim: str, hits: list[dict[str, Any]],
               *, max_chars: int | None = None,
               anchors: list[str] | None = None) -> str:
    """把该篇的片段拼成判官输入（**C3 证据窗口**）。

    `max_chars`：窗口字符上限。取块顺序 = **判据锚点支 ∪ 语义支**（与定稿同法）：
    1. **锚点支**：文中**词面命中任一 `anchors`**（判据锚点词，见
       `query_optimizer.criteria_terms`）的片段，按 `q_sim` 降序 → **优先入窗**；
    2. **语义支**：其余片段按 `q_sim`（该块在全部查询上的最大 cosine，由
       `ChunkIndex.search_multiroute` 给出）降序；
    3. 依此顺序累加，**最后一个片段按剩余预算截断**；至少保留 1 个片段。

    ⚠️ **编号保持"原 hits 里的位置 +1"**（不因筛选/重排而改号）—— 否则判官返回的
    `evidence` 片段号在 `run()` 里映射回 `hits` 时会错位。因此窗口里可能看到
    `[片段3] [片段1] …` 这种**编号非连续递增**的形态（= 锚点块被提前了），这是**预期**的。
    """
    if max_chars and max_chars > 0:
        def _qsim(i: int) -> float:
            v = hits[i].get("q_sim")
            return -1e9 if v is None else -float(v)

        low = [str(a).lower() for a in (anchors or []) if str(a).strip()]
        aset: set[int] = set()
        if low:
            for i in range(len(hits)):
                t = str(hits[i].get("text") or "").lower()
                if any(a in t for a in low):
                    aset.add(i)
        anch_i = sorted(aset, key=_qsim)
        rest_i = sorted((i for i in range(len(hits)) if i not in aset), key=_qsim)
        keep: list[tuple[int, str]] = []
        tot = 0
        for i in anch_i + rest_i:
            t = str(hits[i].get("text") or "")
            room = max_chars - tot
            if room <= 0:
                break
            tn = t if len(t) <= room else t[:room]
            keep.append((i, tn))
            tot += len(tn)
            if len(t) > room:
                break
    else:
        keep = [(i, str(hits[i].get("text") or "")) for i in range(len(hits))]
    blocks = "\n\n".join(f"[片段{i + 1}] {t}" for i, t in keep)
    return f"论断：**{claim}**\n\n该论文的片段：\n{blocks}"


def _norm_label(o: Any) -> str:
    """取 `yes|no|unclear`；非 dict / 非法值 → `"err"`（用于区分"模型说没看懂"与"调用挂了"）。"""
    if not isinstance(o, dict):
        return "err"
    lab = str(o.get("label") or "").strip().lower()
    return lab if lab in ("yes", "no", "unclear") else "err"


def judge_one(claim: str, hits: list[dict[str, Any]], *,
              extract: bool = False, dual: bool | None = None,
              max_chars: int | None = None,
              anchors: list[str] | None = None) -> dict[str, Any]:
    """单篇判定。

    ## 默认 = **单判官 A + 证据窗口**（2026-10-01 改；原为双判官）
    | 项 | 行为 | env |
    |---|---|---|
    | 证据窗口 | 按 `q_sim` 选片段拼到 ≤`max_chars` 字符 | `PAPERPILOT_SET_WIN_CHARS`（默认 **4200**）；`0` = 关（用全部 hits） |
    | 判官 | **A** = 主链路 `PAPERPILOT_LLM`（默认**只用它**）｜ **B** = 独立裁判 `PAPERPILOT_JUDGE`（可选，默认不启用） | `PAPERPILOT_SET_DUAL`（默认 **0** = 仅 A）；`1` = 恢复双判官 |
    | 采信（`dual=1` 时） | 双方 `yes`→yes ｜ 双方 `no`→no ｜ **其余（分歧 / `unclear` / 单边失败）→ 第三轮**（`SYS_SET_STRICT` 对抗式） | — |

    **为什么默认单判官**（`R2_JUDGE_PROTO_20261001.md`，1,400 篇-题离线复算）：
    双判官 + 第三轮与单判官 A **质量等价**（F1 0.717 vs 0.718，k=20/30/50 全部档位一致），
    但调用数 **2.10 → 1.00**。B 保守（单独用召回 0.461）⇒ 第三轮主要在补 B 的假阴，净收益为 0。

    ⚠️ 若裁判模型**未配置**（`llm.judge_configured()` 为假）→ 即使 `dual=1` 也退化为单判官 A，
    不会报错也不会多花调用。

    `extract=True`：额外要 `extract` 字段（该篇在本题上的具体内容）—— 服务复合题。
    `evidence`/`extract`/`why` 一律取 **A** 的输出（A 才带 `extract` 契约）。
    """
    if dual is None:
        # ★ 2026-10-01：默认改为 **单判官 A**（原默认双判官）。
        #   依据 `retrieval/results/R2_JUDGE_PROTO_20261001.md`（1,400 篇-题逐篇 A/B/third
        #   留档离线复算，零新调用）：单判官 A 与"双判官 + 第三轮"**质量等价**，
        #   而调用数 2.10 → 1.00（**省 52%**）。
        #   | 协议 | 调用/篇 | P | R | F1 |
        #   |---|---|---|---|---|
        #   | 双判官 + 第三轮（旧默认） | 2.10 | 0.709 | 0.771 | 0.717 |
        #   | **单判官 A（现默认）**     | **1.00** | 0.708 | **0.775** | **0.718** |
        #   | 双都须 yes | 2.00 | 0.777 | 0.454 | 0.541 |
        #   机理：B（`PAPERPILOT_JUDGE`）系统性保守（单独用召回只有 0.461），第三轮
        #   主要忙于把 B 压掉的判果恢复回来 ⇒ 一正一负抵消。
        #   置 `PAPERPILOT_SET_DUAL=1` 可恢复旧协议（对照用）。
        dual = (os.environ.get("PAPERPILOT_SET_DUAL") or "0").strip() not in ("", "0", "false")
    if max_chars is None:
        max_chars = env_int("PAPERPILOT_SET_WIN_CHARS", DEFAULT_WIN_CHARS)
    mt = 400 if extract else 300
    sysmsg = SYS_SET_EXTRACT if extract else SYS_SET
    user = build_user(claim, hits, max_chars=max_chars, anchors=anchors)
    try:
        o = llm.chat_json(sysmsg, user, temperature=0.0, max_tokens=mt)
    except llm.LLMError as e:
        return {"label": "unclear", "extract": "", "evidence": [], "why": f"调用失败: {e}"[:60]}
    la = _norm_label(o)
    lab = la if la != "err" else "unclear"
    dual_used, lb, third = False, "", ""
    if dual and llm.judge_configured():
        dual_used = True
        try:
            lb = _norm_label(llm.judge_json(sysmsg, user, temperature=0.0, max_tokens=mt))
        except Exception:  # noqa: BLE001（裁判挂了 → 走第三轮）
            lb = "err"
        if la == "yes" and lb == "yes":
            lab = "yes"
        elif la == "no" and lb == "no":
            lab = "no"
        elif la == "err" and lb == "err":
            lab = "unclear"
        else:
            try:
                t3 = llm.chat_json(SYS_SET_STRICT, user, temperature=0.0, max_tokens=200)
                third = str(t3.get("answer") or "") if isinstance(t3, dict) else ""
            except Exception:  # noqa: BLE001
                third = ""
            lab = "yes" if third.upper().startswith("Y") else "no"
    ev = o.get("evidence") if isinstance(o, dict) else None
    ev = ev or []
    if isinstance(ev, (int, str)):
        ev = [ev]
    ex = str((o.get("extract") if isinstance(o, dict) else "") or "").strip()[:160]
    return {"label": lab, "extract": ex if lab == "yes" else "",
            "evidence": [int(x) for x in ev if str(x).isdigit()],
            "why": str((o.get("why") if isinstance(o, dict) else "") or "")[:80],
            "A": la, "B": lb, "third": third, "dual": dual_used}


def run(question: str, idx: Any, *, claim: str | None = None,
        b: int | None = None, workers: int | None = None,
        extract: bool | None = None, dual: bool | None = None,
        max_chars: int | None = None, top_papers: int | None = None,
        anchors: list[str] | None = None) -> dict[str, Any]:
    """完整一次"哪几篇做了 X"。

    `extract`：是否额外抽取**逐篇内容**（复合题必需，见 `SYS_SET_EXTRACT`）。
              默认读环境变量 `PAPERPILOT_SET_EXTRACT`（未设 → 关，保持旧行为）。

    Returns: `{papers:[{pdf, extract, evidence:[片段号], why, score, snippet}],
              n_papers, n_yes}`；`papers` 只含判 `yes` 的篇，已按检索分数降序。
    """
    b = b or env_int("PAPERPILOT_SET_B", DEFAULT_B)
    workers = workers or env_int("PAPERPILOT_SET_WORKERS", DEFAULT_WORKERS)
    if extract is None:
        extract = (os.environ.get("PAPERPILOT_SET_EXTRACT") or "").strip() not in ("", "0", "false")
    cm = claim or question
    # `PAPERPILOT_SET_TOPK`：候选篇上限（定稿口径 = 交付 top-k 篇，k=30）。
    # ⚠️ 不能用 `env_int`（它把下限钳到 1，表达不了"0 = 不限"）→ 手工解析。
    if top_papers is None:
        _raw = (os.environ.get("PAPERPILOT_SET_TOPK") or "").strip()
        top_papers = int(_raw) if _raw.isdigit() else 0
    per = per_paper_hits(idx, question, b=b, claim=cm, top_papers=(top_papers or None))
    cand = sorted(per, key=lambda p: -float(per[p][0].get("score") or 0.0))
    # ★ 窗口的**判据锚点支**：从论断抽"能证明该论断成立的英文关键词/短语"，命中者优先入窗
    #   （定稿窗口的另一支；生产原先只有语义支 → 见 `R2_PROD_EVAL_20260930.md` 的归因）
    if max_chars is None:
        max_chars = env_int("PAPERPILOT_SET_WIN_CHARS", DEFAULT_WIN_CHARS)
    if anchors is None:
        anchors = []
        if max_chars and (os.environ.get("PAPERPILOT_SET_ANCHOR") or "1").strip() not in ("0", "false"):
            from paperpilot.components import query_optimizer as _qo2   # 惰性导入（避免环）
            anchors = _qo2.criteria_terms(cm)
    if anchors:
        print(f"[set_judge] 判据锚点 {len(anchors)} 条：{', '.join(anchors[:5])}"
              f"{' …' if len(anchors) > 5 else ''}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = list(ex.map(
            lambda kv: (kv[0], judge_one(cm, kv[1], extract=extract, dual=dual,
                                         max_chars=max_chars, anchors=anchors)),
            per.items()))
    papers = []
    for pdf, v in verdicts:
        if v["label"] != "yes":
            continue
        hits = per[pdf]
        idxs = [i for i in v["evidence"] if 1 <= i <= len(hits)] or [1]
        first = hits[idxs[0] - 1]
        papers.append({
            "pdf": pdf,
            "extract": v.get("extract") or "",
            "evidence": idxs,
            "why": v["why"],
            "score": float(first.get("score") or 0.0),
            "snippet": str(first.get("text") or "")[:300],
            "chunk_id": str(first.get("chunk_id") or ""),
            "page": int(first.get("page") or 0),
            "section": str(first.get("section") or ""),
        })
    papers.sort(key=lambda x: -x["score"])
    return {"papers": papers, "n_papers": len(per), "n_yes": len(papers),
            "b": b, "extract": bool(extract),
            "top_papers": int(top_papers or 0),
            # ★ 候选篇的**检索序 + 篇内最高分**：供离线扫 k（改 k 只需重切前缀，
            #   不必重跑判官 —— 窗口只依赖该篇自身，与候选多少无关）
            "candidates": [{"pdf": p, "score": round(float(per[p][0].get("score") or 0.0), 6)}
                           for p in cand],
            "retr": (os.environ.get("PAPERPILOT_SET_RETR") or "multiroute").strip().lower(),
            "dual": bool(verdicts and verdicts[0][1].get("dual")),
            "win_chars": int(max_chars if max_chars is not None
                             else env_int("PAPERPILOT_SET_WIN_CHARS", DEFAULT_WIN_CHARS)),
            "anchors": list(anchors or []),
            "labels": {p: v["label"] for p, v in verdicts},
            # ★ A/B/第三轮 逐篇留档：供"正确性（证据是否支持）"评估做分歧分析
            "verdicts": {p: {"A": v.get("A", ""), "B": v.get("B", ""),
                             "third": v.get("third", ""), "label": v["label"]}
                         for p, v in verdicts},
            "n_dissent": sum(1 for _, v in verdicts
                             if v.get("A") != v.get("B") and v.get("B")),
            "unclear": sorted(p for p, v in verdicts if v["label"] == "unclear")}


def render(question: str, res: dict[str, Any]) -> str:
    """把结果渲染成**答案文本**（篇 + 逐篇一句证据）。

    ⚠️ **不做完备性宣称**（ASReview 的教训：AI 排序 + 人兜底；"这个方向我都看过了"不成立）。
    """
    ps = res["papers"]
    head = (f"在 {res['n_papers']} 篇语料里，判为「{question}」的有 **{len(ps)} 篇**："
            if ps else f"在 {res['n_papers']} 篇语料里，**没有找到**符合「{question}」的论文：")
    lines = [head]
    for i, p in enumerate(ps, 1):
        # `extract` = 该篇在本题上的**具体内容**（复合题必需：只给"哪几篇"答不了"各自是什么"）
        ex = str(p.get("extract") or "").strip()
        body = f"{p['why']}；{ex}" if ex else p["why"]
        lines.append(f"{i}. `{p['pdf']}`：{body}（证据：片段 {p['evidence']}，"
                     f"第 {p['page']} 页 {p['section']}）")
    if res.get("unclear"):
        lines.append(f"⚠️ 另有 {len(res['unclear'])} 篇证据不足、未作判断："
                     f"{', '.join(res['unclear'][:5])}"
                     f"{' …' if len(res['unclear']) > 5 else ''}")
    lines.append("（判定基于各篇被检索到的片段；如需覆盖更多内容可提高 `PAPERPILOT_SET_B`。）")
    return "\n".join(lines)
