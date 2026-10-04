"""**Task 2｜R2 论文语料线：接上 reader（逐篇判定 + 逐篇证据）**

## 为什么是它（`R2_QAMPARI_MAPREDUCE_20260929.md` / `R2_QAMPARI_RERANK_20260929.md`）
R2 线目前**只有排序、没有答案**：`StRecall@10 = 0.851` 是"gold 篇是否落进 top-10"的**检索代理**，
不是"哪几篇做了 X + 逐篇证据"。三条线现在指向同一缺口 = **reader / 逐篇判定**。

## 做法（map → 聚合）
| 步 | 内容 |
|---|---|
| 检索 | 干净查询 `[中文题面] + 子查询`（无 gold 锚点词）→ 篇内取 top-b 块 |
| **map** | **逐篇 1 次调用**：论断 + 该篇的 b 个片段 → `{label: yes/no, evidence:[片段号], why}` |
| 聚合 | 判 yes 的篇 = 交付集合；逐篇附证据片段号 + 理由 |
| 评测 | **篇级 集合 P/R/F1** vs 校准后的最终真值；并与"检索排序 top-k"基线对照 |

判定提示**沿用真值重标的四条纪律**：以字面为准 / 区分"本文做的 vs 引用他人做的" /
"列在任务清单·相关工作里 ≠ 做了" / 中文论断 ↔ 英文证据的同义改写。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_reader.py --limit 6       # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_reader.py --workers 8
"""
from __future__ import annotations

import importlib.util as _iu
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "tmp"))
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))
spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F
mrecall, strecall, setpf = _m.mrecall, _m.strecall, _m.setpf

DEV = HERE / "data" / "r2dev"
CACHE = DEV / "clusters"
SUBQ = DEV / "subqueries.json"
RECAL = DEV / "gold_recalib.csv"                    # legacy（v1 真值）
RICH = DEV / "gold_recalib_rich.csv"
GOLD2 = DEV / "gold_final2.csv"                     # ★ 新真值（三判官多数票）
SEL2 = DEV / "facets_v2_selected.json"              # ★ 新题集（核心/扩展）
CHUNKDIR = DEV / "prodchunk" / "mineru"             # ★ 生产切块（MinerU 口径）
MAP = DEV / "pdf_map.json"                          # ★ PDF 映射（`--corpus 50` 换 corpus50）
OUT = HERE / "results"
CHUNK, OVERLAP = 1000, 100
TOPK = [5, 10]

# 新 facet 定义（操作化题面）；legacy 用 `_r2_std_metrics.F`
_f2spec = _iu.spec_from_file_location("facets2", HERE / "tmp" / "_r2_facets_v2.py")
_f2 = _iu.module_from_spec(_f2spec)
sys.modules["facets2"] = _f2
_f2spec.loader.exec_module(_f2)
F2 = _f2.F2


def load_gold2() -> dict[tuple[int, str], set[str]]:
    """★ 新真值：`gold_final2.csv`（三判官多数票 + 已筛题）。

    ⚠️ 必须**按 `gold` 过滤**：该表对每个组合列出**全部**篇（gold=True/False 都有一行），
    漏过滤会让每题的 gold 变成"全部篇"→ 全部被 `len(g)==len(docs)` 跳过。
    """
    g = pd.read_csv(GOLD2)
    g["gold"] = g["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    return {(int(c), str(f)): set(x[x["gold"]]["docid"])
            for (c, f), x in g.groupby(["cluster", "facet"])}


def set_paths(corpus: str) -> None:
    """★ 语料规模切换：改**全局**（`load_gold2()` 与主体循环都读它们）。

    `20` = 原 3×20 篇（`clusters/`）；`50` = 扩语料 3×50 篇（`corpus50/`，含同领域干扰项）。
    """
    global CACHE, CHUNKDIR, GOLD2, MAP
    if corpus == "50":
        CACHE = DEV / "corpus50"
        CHUNKDIR = DEV / "prodchunk50" / "mineru"
        GOLD2 = DEV / "gold_final3.csv"          # 原 47 篇 + 新增干扰项真值
        MAP = DEV / "corpus50" / "pdf_map_all.json"


def selected_combos() -> set[tuple[int, str]]:
    """★ 新题集：核心集 + 扩展集（来自 `facets_v2_selected.json`）。"""
    s = json.loads(SEL2.read_text(encoding="utf-8"))
    out: set[tuple[int, str]] = set()
    for k in s["per_combo"]:
        c, f = k.split("|")
        out.add((int(c), f))
    return out

SYS = """你要判断**某一篇论文**是否做了某件事。你会看到**该论文自己的**若干原文片段。

只输出三档之一：
- `yes`：该论文**自己做了**这件事（片段里有本文自己的做法/实验/设置）。
- `no`：该论文没有做这件事；或者这件事**只是本文引用的他人工作**；或者只是**任务清单/评价指标/
  相关工作/未来工作**里出现的一个词，本文自己并没有做。
- `unclear`：片段不足以判断（**不要**因为"没看到"就判 no，但也**不要**因为"看起来相关"就判 yes）。

判定纪律：
1. 只看给出的片段，**不要用先验知识补充**。
2. ⚠️ **区分"本文做的"与"引用他人做的"**：若动作的主语是被引用的他文（`X et al. proposed…`、
   `Prior work…`）且**没说明本文也做了** → `no`。这是最常见的错法。
3. **列出/提及 ≠ 做了**：只是清单里出现该词 → `no`。
4. 中文论断 ↔ 英文片段，注意同义改写（"做了消融"↔"we ablate each component"）。

只输出 JSON：{"label":"yes|no|unclear","evidence":[片段序号...],"why":"≤30 字中文理由"}"""


def call(system: str, user: str, max_tokens: int = 300,
         prefix: str = "PAPERPILOT_LLM") -> dict:
    for a in range(3):
        try:
            o = llm.chat_json(system, user, temperature=0.0, max_tokens=max_tokens,
                              prefix=prefix)
            if isinstance(o, dict):
                return o
        except Exception:  # noqa: BLE001
            if a == 2:
                return {}
            time.sleep(1.5 * (a + 1))
    return {}


def load_gold() -> dict[tuple[int, str], set[str]]:
    rec = pd.read_csv(RECAL)
    rec["anchor_gold"] = rec["anchor_gold"].astype(bool)
    rec["new_gold"] = rec["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    rec["final_gold"] = rec["new_gold"]
    if RICH.exists():
        r = pd.read_csv(RICH)
        r["yes"] = r["new_gold_rich"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
        mp = {(int(x.cluster), str(x.facet), str(x.docid)): bool(x.yes) for x in r.itertuples()}
        for i, x in rec.iterrows():
            k = (int(x["cluster"]), str(x["facet"]), str(x["docid"]))
            if k in mp:
                rec.at[i, "final_gold"] = mp[k]
    return {(c, f): set(g[g["final_gold"]]["docid"]) for (c, f), g in
            rec.groupby(["cluster", "facet"])}


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--b", type=int, default=6, help="每篇给几块语义证据")
    ap.add_argument("--anchor-b", type=int, default=3,
                    help="★ 每篇额外注入几块**锚点命中块**（修「信号词面可定位、"
                         "但排名超出语义 top-b」的漏判；0 = 关闭）。"
                         "**默认 3 = 实测膝点**：集合F1 0.724→**0.823**（+0.099）、"
                         "prompt 仅 +5.0%%；`code_release` 0.667→0.982。"
                         "对照：`--b 12 --anchor-b 0`（prompt +79.5%%）只有 0.797")
    ap.add_argument("--bm25-b", type=int, default=0,
                    help="★ C2a：每篇额外注入几块 **BM25 通道**命中的块"
                         "（查询集与 dense 相同 → **生产可用，无需人工正则**；0 = 关闭）")
    ap.add_argument("--bm25-tok", default="cjk", choices=["cjk", "ascii"],
                    help="BM25 分词器：cjk = `_tok_cjk`（**修了中文题面被整段丢弃**，默认）；"
                         "ascii = 旧的 `_tok`（只留 [a-z0-9]+）")
    ap.add_argument("--subq-n", type=int, default=-1,
                    help="★ 诊断用：查询侧只用前 N 条子查询（-1=全部，默认；0=只用中文题面）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个真值（冒烟）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--legacy", action="store_true",
                    help="回退到旧口径（旧真值 gold_recalib / 旧题集 meta.usable / 固定窗 over LitSearch 全文）")
    ap.add_argument("--protocol", default="auto", choices=["auto", "3", "4", "dual"],
                    help="判官协议：3 = reader 旧三档；4 = 真值同款**四档**（entail→yes）；"
                         "dual = 真值**完整协议**（A+B 双判官 + 分歧第三轮）。auto：新口径→4，legacy→3")
    ap.add_argument("--corpus", default="20", choices=["20", "50"],
                    help="语料规模：20 = 原 3×20 篇；50 = 扩语料 3×50 篇"
                         "（`corpus50/` + `prodchunk50/mineru/` + `gold_final3.csv`）")
    args = ap.parse_args()

    if args.corpus == "50":
        set_paths("50")
    # ⚠️ `corpus50/` 没有 meta.json（簇结构与题集在 `clusters/meta.json` + `facets_v2_selected.json`）
    meta_path = (CACHE / "meta.json") if (CACHE / "meta.json").exists() \
        else (DEV / "clusters" / "meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pm_ok = {d for d, v in json.loads(MAP.read_text(encoding="utf-8")).items() if v.get("ok")}
    subq_all = json.loads(SUBQ.read_text(encoding="utf-8"))
    if args.legacy:
        gold_map, combos, F_use, use_prod = load_gold(), None, F, False
    else:
        gold_map, combos, F_use, use_prod = load_gold2(), selected_combos(), F2, True
    # ⚠️ reader 与**真值**的判官协议原本不一致（reader 三档 `yes/no/unclear`，真值四档 `entail/...`）
    #    → `code_release` / `fine_tuning` 上 reader 系统性**偏严**（假阴性）。
    #    这里接上真值那套四档协议（`_r2_gold_recalib.SYS` / `SYS_STRICT`）。
    proto = args.protocol
    if proto == "auto":
        proto = "3" if args.legacy else "4"
    tag = args.tag or (f"r2reader_{proto}_b{args.b}_a{args.anchor_b}_m{args.bm25_b}"
                       f"{'' if args.bm25_tok == 'cjk' else '_ascii'}"
                       if args.legacy else
                       f"r2reader{'50' if args.corpus == '50' else '2'}_{proto}"
                       f"_b{args.b}_a{args.anchor_b}_m{args.bm25_b}"
                       f"{'' if args.bm25_tok == 'cjk' else '_ascii'}")
    gr_spec = _iu.spec_from_file_location("goldrecal", HERE / "tmp" / "_r2_gold_recalib.py")
    _gr = _iu.module_from_spec(gr_spec)
    sys.modules["goldrecal"] = _gr
    gr_spec.loader.exec_module(_gr)
    SYS4, SYS_STRICT = _gr.SYS, _gr.SYS_STRICT
    _SYS_USE = SYS4 if proto in ("4", "dual") else SYS
    print(f"真值 {len(gold_map)} 个 ｜ 每篇证据 b={args.b}"
          f"{f' + 锚点块 a={args.anchor_b}' if args.anchor_b else ''}"
          f"{f' + BM25 块 m={args.bm25_b}' if args.bm25_b else ''}"
          f"{'（无注入）' if not (args.anchor_b or args.bm25_b) else ''}"
          f" ｜ 口径 **{'legacy（固定窗+旧真值）' if args.legacy else '新（生产切块+三判官真值）'}**")
    print(f"  facet 定义：{'v1' if args.legacy else 'v2（操作化）'} ｜ "
          f"切块：{'固定窗 1000/900 over LitSearch 全文' if not use_prod else str(CHUNKDIR)}")
    print(f"  判官协议：**{proto}**"
          f"{'（reader 旧三档 yes/no/unclear）' if proto == '3' else ''}"
          f"{'（真值同款四档 entail/partial/neutral/contradict → entail 记为 yes）' if proto == '4' else ''}"
          f"{'（真值完整协议：A deepseek-chat + B glm-4-flash + 分歧第三轮）' if proto == 'dual' else ''}\n")

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192 if use_prod else 512      # ⚠️ 生产口径 8192（512 是截断坑）
    if dev == "cuda":
        enc.half()

    tasks = []
    t0 = time.time()
    for ci in range(len(meta)):
        bm_obj = None                     # ★ C2a：每簇惰性建一次 BM25（依赖该簇的 chunks）
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
        if use_prod:
            ch = pd.read_parquet(CHUNKDIR / f"c{ci}.parquet")
            # ⚠️ 既要**有切块**，也要在 PDF 映射里 `ok`（`--corpus 50` 时 = 该簇 50 篇）
            docs = [d for d in sub["docid"].tolist()
                    if d in set(ch["docid"]) and d in pm_ok]
            chunks, owner = [], []
            for d in docs:
                cs = ch[ch["docid"] == d]["text"].astype(str).tolist()
                chunks += cs
                owner += [d] * len(cs)
        else:
            docs = sub["docid"].tolist()
            chunks, owner = [], []
            for i, t in enumerate(sub["full_paper"].tolist()):
                cs = _m.chunks_of(t)
                chunks += cs
                owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=16 if use_prod else 32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        idx = {d: np.where(owner == d)[0] for d in docs}
        facets_here = (sorted(f for (c2, f) in combos if c2 == ci + 1) if combos
                       else meta[ci]["usable"])
        for facet in facets_here:
            # ⚠️ 与 `docs` 取交：扩语料下 gold 里可能含**无 PDF** 的旧篇（不在语料内）
            g = (gold_map.get((ci + 1, facet)) or set()) & set(docs)
            if not g or len(g) == len(docs):
                continue
            zh = F_use[facet][1]
            sq = list(subq_all.get(facet, []))
            if args.subq_n >= 0:             # ★ 诊断用：-1=全部（默认）；0=只用中文题面
                sq = sq[: args.subq_n]
            qs = [zh] + sq
            S = enc.encode(qs, normalize_embeddings=True, convert_to_numpy=True
                           ).astype(np.float32) @ C.T
            smax = S.max(axis=0)
            # 检索排序（用于基线对照）
            ranked = sorted(docs, key=lambda d: -smax[idx[d]].max())
            # ★ 修路 #1：证据 = 语义 top-b ∪ **锚点命中块 top-a**
            #   依据 `R2_CR_EVIDENCE_DIAG.csv`：`code_release` 的 13 篇漏判里
            #   **0 篇**是"锚点块进了 top-b 判官仍判 no"，**9 篇**是"链接在篇内但排名 >b"
            #   （中位排名 12）→ 瓶颈是**证据覆盖率**，不是判官。
            #   真值端（`_r2_gold_recalib.py`）早就用"锚点 ∪ 语义"凑证据池，reader 端此前**没做**。
            a_hit = None
            if args.anchor_b:
                a_pat = re.compile(F_use[facet][0], re.I)
                a_hit = np.array([bool(a_pat.search(c)) for c in chunks])
            # ★ C2a：**BM25 通道注入**（生产可用：只需 question 文本，**不需要人工正则**）
            #   查询集与 dense 通道**完全相同**（题面 + 子查询）→ 只换通道，不换查询。
            b_sc = None
            if args.bm25_b:
                if bm_obj is None:
                    bm_obj = _m.BM25(chunks, tok=(_m._tok_cjk if args.bm25_tok == "cjk"
                                                  else _m._tok))
                b_sc = np.max([bm_obj.scores(q) for q in qs], axis=0)
            per = {}
            for d in docs:
                sem = [int(j) for j in idx[d][np.argsort(-smax[idx[d]])[: args.b]]]
                if a_hit is not None:
                    ah = sorted((int(j) for j in idx[d][a_hit[idx[d]]]), key=lambda j: -smax[j])
                    for j in ah[: args.anchor_b]:
                        if j not in sem:
                            sem.append(j)
                if b_sc is not None:
                    bh = sorted((int(j) for j in idx[d][b_sc[idx[d]] > 0]), key=lambda j: -b_sc[j])
                    for j in bh[: args.bm25_b]:
                        if j not in sem:
                            sem.append(j)
                per[d] = sem
            tasks.append(dict(cluster=ci + 1, facet=facet, gold=g, ranked=ranked,
                              per=per, chunks=chunks,
                              claim=(zh if str(zh).startswith("本文") else f"该论文{zh}")))
        print(f"  簇{ci + 1} 就绪（{len(docs)} 篇 / {len(chunks)} 块 / "
              f"{len(facets_here)} facet ｜ {time.time() - t0:.0f}s）", flush=True)

    if args.limit:
        tasks = tasks[: args.limit]
    n_call = sum(len(t["per"]) for t in tasks)
    print(f"待判真值 {len(tasks)} 个 ｜ 逐篇判定合计 **{n_call}** 次 map 调用\n")

    lock = threading.Lock()
    prog = {"n": 0}

    def do_one(t: dict) -> dict:
        q = t["claim"]
        docs = list(t["per"].keys())
        def judge(doc):
            ids = t["per"][doc]
            blocks = "\n\n".join(f"[片段{i + 1}] {t['chunks'][j]}" for i, j in enumerate(ids))
            user = f"论断：**{q}**\n\n该论文的片段：\n{blocks}"
            ev: list[int] = []
            why = ""
            if proto == "dual":                                  # 真值**完整协议**
                oa = call(SYS4, user, prefix="PAPERPILOT_LLM")
                ob = call(SYS4, user, prefix="PAPERPILOT_JUDGE")
                la = str(oa.get("label") or "ERR").lower()
                lb = str(ob.get("label") or "ERR").lower()
                if la == "entail" and lb == "entail":
                    lab = "yes"
                elif "entail" not in (la, lb) and "ERR" not in (la, lb):
                    lab = "no"
                else:
                    ot = call(SYS_STRICT, user, prefix="PAPERPILOT_LLM")
                    lab = "yes" if str(ot.get("answer", "")).upper().startswith("Y") else "no"
                    why = str(ot.get("why") or "")[:60]
                why = why or str(oa.get("reason") or ob.get("reason") or "")[:60]
            else:
                o = call(_SYS_USE, user, max_tokens=400 if proto == "4" else 300)
                raw = str(o.get("label") or "").strip().lower()
                lab = ((raw if raw in ("yes", "no", "unclear") else "") if proto == "3" else
                       {"entail": "yes", "partial": "no", "neutral": "no",
                        "contradict": "no"}.get(raw, ""))
                evr = o.get("evidence") or []
                if isinstance(evr, (int, str)):
                    evr = [evr]
                ev = [int(x) for x in evr if str(x).isdigit()]
                why = str(o.get("why") or o.get("reason") or "")[:60]
            return doc, lab, ev, why
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(judge, docs))
        yes = {d for d, lab, _e, _w in res if lab == "yes"}
        ev_map = {d: e for d, _l, e, _w in res}
        with lock:
            prog["n"] += 1
            if prog["n"] % 5 == 0:
                print(f"  真值 {prog['n']}/{len(tasks)} …", flush=True)
        p, r, f1 = setpf(sorted(yes), t["gold"], len(yes) or 1)
        p10, r10, f10 = setpf(t["ranked"], t["gold"], 10)
        return dict(cluster=t["cluster"], facet=t["facet"], n_gold=len(t["gold"]),
                    n_yes=len(yes), yes=sorted(yes), evidence=ev_map,
                    reader_P=p, reader_R=r, reader_F1=f1,
                    ret_P=p10, ret_R=r10, ret_F1=f10,
                    StRecall10=strecall(t["ranked"], t["gold"], 10),
                    MRecall10=mrecall(t["ranked"], t["gold"], 10))

    r0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:      # 外层并发（内层已并行）
        for x in ex.map(do_one, tasks):
            rows.append(x)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "evidence"} for r in rows])
    df.to_csv(OUT / f"R2_{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\n完成 {time.time() - r0:.0f}s ｜ 调用 {llm.usage_stats()['calls']}"
          f" ｜ prompt {llm.usage_stats()['prompt_tokens']:,}"
          f" / completion {llm.usage_stats()['completion_tokens']:,}")

    print(f"\n{'=' * 108}\n【R2 论文语料线：reader vs 检索基线（k=10）】{len(df)} 个真值")
    print(f"  {'指标':<26}{'reader（判定集合）':>20}{'检索 top-10':>16}")
    for nm, a, b in (("集合 P", df["reader_P"].mean(), df["ret_P"].mean()),
                     ("集合 R", df["reader_R"].mean(), df["ret_R"].mean()),
                     ("集合 F1", df["reader_F1"].mean(), df["ret_F1"].mean())):
        print(f"  {nm:<26}{a:>20.3f}{b:>16.3f}")
    print(f"  {'交付篇数':<26}{df['n_yes'].mean():>20.2f}{10:>16}")
    print(f"  {'gold 篇数（均值）':<26}{df['n_gold'].mean():>20.2f}")
    print(f"\n  检索侧参照：StRecall@10 {df['StRecall10'].mean():.3f} ｜ "
          f"MRecall@10 {df['MRecall10'].mean():.3f}")
    print(f"\n  {'簇':>3}{'facet':<18}{'gold':>5}{'判yes':>6}{'readerF1':>10}{'retF1':>8}")
    for _, x in df.sort_values("reader_F1").iterrows():
        print(f"  {int(x['cluster']):>3}{x['facet']:<18}{int(x['n_gold']):>5}"
              f"{int(x['n_yes']):>6}{x['reader_F1']:>10.3f}{x['ret_F1']:>8.3f}")
    print(f"\n已写 {OUT / f'R2_{tag}.csv'}")
    _emit_report(df, args, proto, GOLD2, tag)
    return 0


def _emit_report(df, args, proto, goldf, tag: str) -> None:
    """把 reader 判定汇总落进统一记录格式（`evals/report.py`）。

    为什么：本脚本原先只写**逐 facet CSV**，汇总只 print —— 无法与 L1/L3 比、
    也无法与历史比（要翻 stdout 考古）。

    ⚠️ **失败绝不影响评测**；且**算不出（NaN）就不 emit 那条**（schema 会拒收 NaN）。
    """
    try:
        sys.path.insert(0, str(HERE.parent))
        from evals import report as R  # noqa: PLC0415

        # ★ 口径必须写全：判官协议 / 证据预算 / facet 定义 / 真值文件 ——
        #   **换任何一个，reader 的 P/R/F1 都不可比**（协议从三档改四档时
        #   `code_release` / `fine_tuning` 上 reader 曾系统性偏严）。
        note = (f"R2 reader 判定（k=10，集合 P/R/F1，{len(df)} 个真值）"
                f" ｜ 语料 **{args.corpus}** ｜ 真值 `{goldf.name}`"
                f" ｜ **判官协议 proto={proto}** ｜ 证据预算 b={args.b}"
                f"/anchor_b={args.anchor_b} ｜ facet 定义 "
                f"{'v1（旧）' if args.legacy else 'v2（操作化）'}"
                f" ｜ 运行 tag={tag}")
        ev = f"retrieval/results/R2_{tag}.csv"
        recs: list[dict] = []
        for col, metric in (("reader_P", "r2.reader_p@10"),
                            ("reader_R", "r2.reader_r@10"),
                            ("reader_F1", "r2.reader_f1@10"),
                            ("ret_F1", "r2.ret_f1@10"),
                            ("n_yes", "r2.delivered_papers"),
                            ("n_gold", "r2.gold_papers")):
            if col not in df.columns:
                continue
            v = float(df[col].mean())
            if not np.isfinite(v):
                continue      # ★ 算不出来就别 emit（不要写 NaN）
            recs.append(dict(layer="L2", name=f"r2reader_{tag}", metric=metric,
                             value=v, n=int(len(df)), note=note,
                             evidence_path=ev))
        if recs:
            p = R.emit(*recs)
            print(f"→ 汇总记录已落 {p.relative_to(HERE.parent)}（{len(recs)} 条）")
    except Exception as e:  # noqa: BLE001  报告失败不能弄挂评测
        print(f"⚠️ 汇总记录落盘失败（不影响本次评测结果）：{type(e).__name__}: {e}",
              file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
