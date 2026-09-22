"""纯 listwise 精排（阶段 ①）：每条查询 1 次调用，把 K_LLM 篇候选交给 LLM 排序。

**要解决的问题**：cross-encoder 是 **pointwise** —— 每篇候选独立打分，
无法表达"这篇相对那篇更好"。实测后果（`LITSEARCH_RERANK_DEPTH.md`）：
池子 100→1000 时上界 83.5%→93.8%（+10.3pt），但精排 R@1 纹丝不动（0.4154→0.4137）。
**listwise 让模型在候选之间交互**，理论上能吃到 pointwise 吃不到的部分。

**两条臂（同一批 20 篇，分别来自两种候选选择）**：

    A. 候选 = 池子（hyb0.5）前 K        → 上界 = 池子 R@K（K=20 时 0.6978）
    B. 候选 = cross-encoder 前 K        → 上界 = 精排 R@K（K=20 时 0.7430）★ 主臂

**臂 B 是干净的对照**：给 LLM 的 20 篇与 cross-encoder 的 top-20 **完全相同**，
唯一变量是"谁排序" → 直接隔离出 listwise 相对 pointwise 的净增益。

工程要点：
  · **打乱呈现顺序**（缓解 LLM 的位置偏差），再用映射还原；
  · 严格校验输出是 1..K 的排列（缺项按原顺序补、重复丢弃），并记录"干净率"；
  · 逐题缓存（`derived/llm_rerank_cache.json`），可中断续跑、重跑不花钱；
  · 记录 token 用量，给出真实成本。

用法：
    python retrieval/scripts/llm_rerank.py --limit 20      # 冒烟
    python retrieval/scripts/llm_rerank.py                 # 全量（597 × 2 臂）
    python retrieval/scripts/llm_rerank.py --arms B        # 只跑主臂
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
RESULTS = HERE / "results"
DERIVED = HERE / "data" / "litsearch" / "derived"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
CORPUS = DERIVED / "corpus_text.parquet"
CACHE = DERIVED / "llm_rerank_cache.json"
TOPK = RESULTS / "topk1000.npz"
CE_SCORES = RESULTS / "rerank1000.npy"

K_LLM_DEFAULT = 20
POOL_DEPTH = 100
ABSTRACT_CHARS = 1000          # 每篇候选喂给 LLM 的摘要上限（控 token）

_SYS = (
    "你是学术文献检索的排序专家。研究者给出一个「文献检索需求」，"
    "下面是若干候选论文的标题与摘要。请按「该论文对满足这个检索需求的有用程度」"
    "**从高到低**排序。\n"
    "判断依据：论文研究的**问题/方法/任务**是否与需求匹配。\n"
    "注意：\n"
    "  · 不要因为出现相同关键词就判为相关 —— 要看它是否真的解决需求描述的那个问题；\n"
    "  · 若需求包含多个条件（如「既用 A 又用 B」），优先选**同时满足更多条件**的论文；\n"
    "  · 你看到的候选顺序是随机的，与相关性无关。\n"
    '只输出 JSON：{"ranking": [编号, 编号, ...]}，'
    "必须包含全部编号、每个恰好出现一次。"
)


# ───────────────────────── 极简 LLM 客户端（纯 stdlib，不依赖 paperpilot）─────────
def load_env() -> None:
    env = HERE.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LLM:
    def __init__(self) -> None:
        self.base = os.environ.get("PAPERPILOT_LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("PAPERPILOT_LLM_API_KEY", "")
        self.model = os.environ.get("PAPERPILOT_LLM_MODEL", "")
        self.calls = 0
        self.ptok = 0
        self.ctok = 0
        self.fail = 0

    def ok(self) -> bool:
        return bool(self.base and self.key and self.model)

    def chat(self, system: str, user: str, temperature: float = 0.0,
             timeout: int = 120) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        u = d.get("usage") or {}
        self.calls += 1
        self.ptok += int(u.get("prompt_tokens") or 0)
        self.ctok += int(u.get("completion_tokens") or 0)
        return d["choices"][0]["message"]["content"]


def parse_ranking(text: str, k: int) -> tuple[list[int], bool]:
    """把 LLM 输出解析成 1..k 的排列。返回 (perm, 是否"干净"= 原本就是完整排列)。"""
    m = re.search(r"\[[\s\d,]*\]", text)
    if m:
        try:
            nums = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
    else:
        nums = [int(x) for x in re.findall(r"\d+", text)]
    clean_out, seen = [], set()
    for x in nums:
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= x <= k and x not in seen:
            seen.add(x)
            clean_out.append(x)
    missing = [i for i in range(1, k + 1) if i not in seen]
    clean = (len(missing) == 0) and (len(nums) == k)
    return clean_out + missing, clean


# ───────────────────────── 文档表示：信息量阶梯 / 紧凑特征 ─────────────────
# 目的：固定 K=20，只改"每篇候选喂多少信息"，看排序质量掉多少。
#   若 title ≈ full  → **信息不是瓶颈** → CoRank 式压缩几乎无损，价值在"装更多篇"；
#   若 title << full → 摘要内容有实质贡献 → 压缩有损，滑动窗口（不丢信息）更稳。
VIEWS = ("title", "short", "kw", "full")
SHORT_CHARS = 100
KW_N = 12


def load_bm25_index():
    """复用已缓存的 BM25 词表与 IDF（本地、免费），用于抽『高区分度关键词』。"""
    d = DERIVED / "bm25"
    st = np.load(d / "stats.npz")
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    return st["idf"], vocab


def keywords_of(text: str, idf: np.ndarray, vocab: dict[str, int], n: int = KW_N) -> str:
    """该文档中出现过的、IDF 最高（最稀有 = 最具区分度）的 n 个词。
    这是 CoRank「关键词特征」的**免费近似**（CoRank 用 LLM 抽 30 个再选 5 个）。"""
    from eval_retrieval import tokenize
    toks = {t for t in tokenize(text) if t in vocab}
    order = sorted(toks, key=lambda t: -float(idf[vocab[t]]))
    return ", ".join(order[:n])


def doc_repr(view: str, title: str, text: str, kw: dict[int, str],
             row: int) -> tuple[str, str]:
    """返回 (标题, 正文)。view 决定喂多少信息 —— 这就是『信息量阶梯』。"""
    if view == "title":
        return title, ""
    if view == "short":
        return title, text[:SHORT_CHARS].replace("\n", " ").strip()
    if view == "kw":
        return title, kw.get(row, "")
    return title, text[:ABSTRACT_CHARS].replace("\n", " ").strip()


def load_citations() -> dict[int, np.ndarray]:
    """语料的 outgoing references（corpusid → 它引用的论文 id）。用于臂 D 的引用标注。"""
    cit: dict[int, np.ndarray] = {}
    for p in sorted((HERE / "data" / "litsearch" / "corpus_clean").glob("*.parquet")):
        d = pd.read_parquet(p, columns=["corpusid", "citations"])
        for cid, cs in zip(d["corpusid"].to_numpy(), d["citations"].to_numpy()):
            cit[int(cid)] = cs if cs is not None else np.array([], dtype=np.int64)
    return cit


def build_user(query: str, docs: list[tuple[str, str]],
               annots: list[str] | None = None) -> str:
    """annots[j] 是第 j 篇的补充标注（臂 D 用来显式给出引用关系）。

    论文（LitSearch one-hop）靠**相邻位置**让 LLM 推断引用关系；这里改为**显式写出**，
    目的是把"排序器看不到的结构信息"直接送到它面前（CE pointwise 读不到该信息）。
    """
    lines = [f"检索需求：{query}", ""]
    if annots is not None:
        # 图例只在本臂出现，保持 _SYS 不变 → C 与 D 只差「标注」这一个变量
        lines += ["（候选后若出现「← 被第 N 篇引用」，表示第 N 篇引用了这一篇；",
                  "  被相关候选引用的论文往往也值得考虑，可作参考线索。）", ""]
    lines.append("候选论文：")
    for j, (title, body) in enumerate(docs, 1):
        a = f"   ← {annots[j - 1]}" if annots and annots[j - 1] else ""
        lines.append(f"{j}. 标题：{title}" + (f"\n   摘要：{body}" if body else "") + a)
    lines += ["", '请输出 JSON：{"ranking": [...]}（编号从高到低，共 '
              f"{len(docs)} 个）"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", default=str(K_LLM_DEFAULT),
                    help="给 LLM 的候选数（= cross-encoder 过滤深度，即 Filter-then-Rerank 的 K）。"
                         "可逗号分隔扫多个：--k 20,30,50,100")
    ap.add_argument("--arms", default="B,A", help="跑哪些臂（B=精排前 k，A=池子前 k）")
    ap.add_argument("--doc-views", default="full",
                    help="每篇候选喂多少信息，逗号分隔。可选 "
                         "title(仅标题) / short(标题+前100字符) / kw(标题+高IDF关键词) / "
                         "full(标题+前1000字符)。这就是『信息量阶梯』")
    ap.add_argument("--cands-file", default="",
                    help="自定义 per-query 候选列表（results/ 下的 .npy，形状 (597, W) 行号）。"
                         "给了它就新增**臂 C**：候选/基线都取自该文件（用于测试"
                         "「引用信号融合」等自定义候选源）")
    ap.add_argument("--out-md", default="LITSEARCH_LLM_RERANK.md",
                    help="报告文件名（results/ 下）。跑新实验时务必换名，避免覆盖旧报告")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260919, help="打乱呈现顺序用")
    args = ap.parse_args()

    load_env()
    q = pd.read_parquet(QFILE)
    ids = np.load(DERIVED / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[g] for g in row if int(g) in id2row} for row in q["corpusids"]]
    dfc = pd.read_parquet(CORPUS, columns=["corpusid", "text", "title"])
    dfc = dfc[dfc["text"] != ""].reset_index(drop=True)
    titles = dfc["title"].astype(str).tolist()
    texts = dfc["text"].astype(str).tolist()

    pool = np.load(TOPK)["hyb0.5"][:, :POOL_DEPTH]
    ce = np.load(CE_SCORES)[:, :POOL_DEPTH]
    n = len(golds)
    if args.limit:
        n = min(n, args.limit)
    ks = [max(2, min(int(x), POOL_DEPTH)) for x in str(args.k).split(",") if str(x).strip()]
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    views = [v.strip() for v in args.doc_views.split(",") if v.strip()]
    bad = [v for v in views if v not in VIEWS]
    if bad:
        print(f"未知 doc-view {bad}；可选 {VIEWS}")
        return 2
    print(f"查询 {n} 条 | K_LLM={ks} | 臂 {arms} | doc-view={views} | "
          f"共 {n * len(arms) * len(views) * len(ks)} 次调用")

    # 精排全序（与 K 无关）→ 各 K 的候选 = 它的前 K
    ce_order = [pool[i][np.argsort(-ce[i])] for i in range(n)]
    base_all = {"B": ce_order, "A": [pool[i] for i in range(n)]}
    cand_by_k = {kk: {"B": [ce_order[i][:kk] for i in range(n)],
                      "A": [pool[i][:kk] for i in range(n)]} for kk in ks}

    # 臂 C：自定义候选源（如「引用信号融合」的 top-20）
    if args.cands_file:
        cf = np.load(RESULTS / args.cands_file)
        if cf.shape[0] < n:
            print(f"{args.cands_file} 题数 {cf.shape[0]} < {n}")
            return 2
        cf = cf[:n]                      # 兼容 --limit 冒烟
        fused_order = [[int(x) for x in cf[i]] for i in range(n)]
        base_all["C"] = fused_order
        # 臂 D = 同一批自定义候选，但 prompt 里**显式标注引用关系**（与 C 严格对照）
        base_all["D"] = fused_order
        cand_by_k = {kk: {**cand_by_k[kk],
                          "C": [fused_order[i][:min(kk, cf.shape[1])] for i in range(n)],
                          "D": [fused_order[i][:min(kk, cf.shape[1])] for i in range(n)]}
                     for kk in ks}
        print(f"臂 C/D（自定义候选）：{args.cands_file} 宽度 {cf.shape[1]}"
              f"（D = 额外在 prompt 中标注引用关系）")
    elif "C" in arms or "D" in arms:
        print("--arms 含 C/D 但未给 --cands-file")
        return 2

    cit_tbl = load_citations() if "D" in arms else {}
    if "D" in arms:
        print(f"臂 D 引用表：{len(cit_tbl)} 篇（用于 prompt 标注）")

    def R(ranked, kk: int) -> float:
        # 兼容 numpy 数组与 Python list（臂 C 的候选来自自定义文件）
        return float(np.mean([len(g & set(list(ranked[i][:kk]))) / len(g) if g else 0
                              for i, g in enumerate(golds[:n])]))

    print("\n【候选集上界 = 该集合里含 gold 的比例，即 LLM 的 R@1 天花板】")
    for kk in ks:
        parts = []
        for a in arms:
            up, b1 = R(cand_by_k[kk][a], kk), R(base_all[a], 1)
            parts.append(f"臂{a}: 上界R@{kk}={up:.4f} 基线R@1={b1:.4f} "
                         f"可用空间={up - b1:.4f}")
        print(f"  K={kk:>3}: " + " | ".join(parts))


    cache: dict[str, dict] = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}
    # 旧键迁移（都要补上 K，否则不同 K 会互相覆盖）：
    #   "B:i"        ← 阶段①，等价于 K=20 / view=full
    #   "view:arm:i" ← 信息量阶梯，等价于 K=20
    migrated: dict[str, dict] = {}
    for key, val in cache.items():
        p = key.split(":")
        if len(p) == 2:
            migrated[f"k{K_LLM_DEFAULT}:full:{p[0]}:{p[1]}"] = val
        elif len(p) == 3:
            migrated[f"k{K_LLM_DEFAULT}:{p[0]}:{p[1]}:{p[2]}"] = val
        else:
            migrated[key] = val
    cache = migrated

    # 关键词特征：本地免费，只需对候选里出现过的文档算一次
    kw: dict[int, str] = {}
    if "kw" in views:
        idf, vocab = load_bm25_index()
        rows_need = sorted({int(r) for kk in ks for a in arms for i in range(n)
                            for r in cand_by_k[kk][a][i]})
        for r in rows_need:
            kw[r] = keywords_of(texts[r], idf, vocab)
        print(f"关键词特征：对 {len(rows_need)} 篇候选抽取完成（本地，无 API 成本）")

    order_res: dict[tuple[int, str, str, int], np.ndarray] = {}
    cleanliness: dict[tuple[int, str, str], list[bool]] = {}
    tok_by_cfg: dict[str, int] = {}

    llm = LLM()
    if not llm.ok():
        print("未配置 PAPERPILOT_LLM_*，退出")
        return 2

    for kk in ks:
        cand = cand_by_k[kk]
        for view in views:
            for a in arms:
                ck = f"k{kk}:{view}:{a}"
                need = [i for i in range(n) if f"{ck}:{i}" not in cache]
                print(f"\n[{ck}] 缓存命中 {n - len(need)}，待调用 {len(need)}")
                t0 = time.time()
                tok0 = llm.ptok
                if need:
                    # 每个配置都从同一 seed 重新生成 → 各配置呈现顺序**完全相同**，
                    # 从而把「信息量 / K」隔离为唯一变量
                    rng = np.random.default_rng(args.seed)
                    disps = {i: rng.permutation(kk) for i in range(n)}
                    jobs = []
                    for i in need:
                        disp = disps[i]
                        rows_here = [int(cand[a][i][d]) for d in disp]
                        docs = [doc_repr(view, titles[r], texts[r], kw, r) for r in rows_here]
                        annots = None
                        if a == "D":
                            # 对每个候选 d，列出"列表中哪些篇引用了 d"（编号用它在本列表中的位置）
                            cids = [int(ids[r]) for r in rows_here]
                            cites = [set(int(x) for x in
                                         cit_tbl.get(c, np.array([], dtype=np.int64)))
                                     for c in cids]
                            annots = []
                            for r in rows_here:
                                who = [j + 1 for j, cs in enumerate(cites) if int(ids[r]) in cs]
                                annots.append(f"被第 {'、'.join(map(str, who))} 篇引用"
                                              if who else "")
                        jobs.append((i, disp,
                                     build_user(str(q.loc[i, "query"]), docs, annots)))

                    def work(job, _kk=kk):
                        i, disp, user = job
                        try:
                            out = llm.chat(_SYS, user)
                        except (urllib.error.URLError, TimeoutError, OSError) as e:  # noqa: BLE001
                            return i, None, False, repr(e)
                        perm, clean = parse_ranking(out, _kk)
                        return i, (disp, perm), clean, None

                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        for t, (i, res, clean, err) in enumerate(ex.map(work, jobs), 1):
                            if res is not None:
                                disp, perm = res
                                cache[f"{ck}:{i}"] = {"disp": disp.tolist(), "perm": perm}
                                cleanliness.setdefault((kk, view, a), []).append(clean)
                            else:
                                llm.fail += 1
                                if llm.fail <= 3:
                                    print(f"      调用失败：{err}")
                            if t % 50 == 0:
                                CACHE.parent.mkdir(parents=True, exist_ok=True)
                                CACHE.write_text(json.dumps(cache, ensure_ascii=False),
                                                 encoding="utf-8")
                                print(f"    {t}/{len(need)}  {time.time() - t0:.0f}s")
                    CACHE.parent.mkdir(parents=True, exist_ok=True)
                    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    print(f"    完成，用时 {time.time() - t0:.0f}s | 失败 {llm.fail} 次")
                tok_by_cfg[ck] = llm.ptok - tok0

                for i in range(n):
                    c = cache.get(f"{ck}:{i}")
                    if c is None:
                        continue
                    disp = np.array(c["disp"])
                    perm = np.array(c["perm"]) - 1
                    order_res[(kk, view, a, i)] = np.asarray(cand[a][i])[disp[perm]]

    # ── 指标 ─────────────────────────────────────────────────
    def RR(ranked, kk, gg):
        return float(np.mean([len(g & set(list(ranked[j][:kk]))) / len(g) if g else 0
                              for j, g in enumerate(gg)]))

    print("\n" + "=" * 104)
    print("listwise 精排 · K 扫描（Filter-then-Rerank：cross-encoder 过滤到 K → LLM 排全部 K）")
    print("=" * 104)
    rows = []
    for kk in ks:
        for view in views:
            for a in arms:
                have = [i for i in range(n) if (kk, view, a, i) in order_res]
                if not have:
                    continue
                rl = [order_res[(kk, view, a, i)] for i in have]
                bl = [base_all[a][i] for i in have]
                gg = [golds[i] for i in have]
                up, b1, r1 = RR(bl, kk, gg), RR(bl, 1, gg), RR(rl, 1, gg)
                space = up - b1
                rows.append({
                    "K": kk, "view": view, "臂": a, "n": len(have),
                    "R@1": r1, "R@5": RR(rl, 5, gg), "R@10": RR(rl, 10, gg),
                    "R@20": RR(rl, 20, gg),
                    "上界R@K": up,
                    "吃掉排序空间%": (100 * (r1 - b1) / space) if space > 1e-9 else float("nan"),
                    "tok/查询": round(tok_by_cfg.get(f"k{kk}:{view}:{a}", 0) / max(len(have), 1)),
                })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n  view：title=仅标题 | short=标题+摘要前100字符 | "
          "kw=标题+高IDF关键词(本地免费) | full=标题+摘要前1000字符")
    print("  '吃掉排序空间%' = (R@1 − 基线R@1) / (上界R@K − 基线R@1)，"
          "衡量 LLM 把可用空间利用了多少")

    # ── 与基线比较 + 显著性 ──────────────────────────────────
    from significance import mcnemar
    print("\n【显著性：LLM listwise vs 同候选集基线（McNemar on hit@1）】")
    for kk in ks:
        for view in views:
            for a in arms:
                have = [i for i in range(n) if (kk, view, a, i) in order_res]
                if not have:
                    continue
                x = np.array([1.0 if (golds[i] & set(list(order_res[(kk, view, a, i)][:1])))
                              else 0.0 for i in have])
                y = np.array([1.0 if (golds[i] & set(list(base_all[a][i][:1]))) else 0.0
                              for i in have])
                only_llm = int(((x == 1) & (y == 0)).sum())
                only_base = int(((x == 0) & (y == 1)).sum())
                pv = mcnemar(only_llm, only_base)
                cl = cleanliness.get((kk, view, a), [])
                print(f"  [K={kk:<3} view={view:<5} 臂={a}] 仅LLM命中 {only_llm:3d} | "
                      f"仅基线命中 {only_base:3d} | 净Δ={only_llm - only_base:+4d} | "
                      f"p={pv:.4g} {'**显著**' if pv < 0.05 else '不显著'}"
                      + (f" | 格式干净率 {100 * np.mean(cl):.0f}%" if cl else ""))

    print(f"\n[ 成本 ] 调用 {llm.calls} 次 | prompt {llm.ptok:,} | completion {llm.ctok:,} "
          f"| 失败 {llm.fail}")

    # ── 落盘 ─────────────────────────────────────────────────
    from eval_retrieval import md_table
    md = RESULTS / args.out_md
    md.write_text(
        f"# LitSearch · listwise 精排（K 扫描 / 信息量阶梯，"
        f"{time.strftime('%Y%m%d_%H%M%S')}）\n\n"
        f"- **Filter-then-Rerank**：cross-encoder 过滤到 K（= `--k`），"
        f"再由 LLM 一次排完全部 K 篇\n"
        f"- 每条查询 **1 次调用**；候选来自 `topk1000.npz` 的 `hyb0.5`\n"
        f"- 臂 B 候选 = cross-encoder 前 K（**与基线同集合，只改排序**）；"
        f"臂 A 候选 = 池子前 K\n"
        f"- doc-view：`title`(仅标题) / `short`(标题+前100字符) / "
        f"`kw`(标题+高IDF关键词) / `full`(标题+前1000字符)\n"
        f"- K={ks}，模型 `{llm.model}`；各配置呈现顺序**完全相同**（seed={args.seed}），"
        f"以隔离「K」与「信息量」两个变量\n"
        f"- `上界R@K` = 未精排时 gold 落在前 K 的比例 = **LLM 的 R@1 理论上界**\n"
        f"- 调用 {llm.calls} 次 | prompt {llm.ptok:,} token | completion {llm.ctok:,}\n\n"
        + md_table(tbl) + "\n", encoding="utf-8")
    print(f"\n已写出：{md.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
