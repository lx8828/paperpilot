"""**修路 #1｜`code_release` 证据侧诊断**

## 问题
三簇 `code_release` 都上不去：gold **9/8/10** 篇，reader 只判 **6/4/5**，F1 卡在 0.667/0.667/0.667。
且**换判官协议（三档→四档）yes 数逐个不动**（见 `R2_RETR_METRICS_20260929.md` §1.1）
→ 说明瓶颈**不在判官**，要往证据侧查。

## 三个互斥的归因（本脚本要分开它们）
对每个 **gold 但被判 no** 的篇：
  · `A 判官放过`  —— 代码链接**在 top-b 块里**，判官仍判 no   → 提示词/证据裁剪问题
  · `B 预算不够`  —— 链接在**该篇内**，但排在 **>b** 名之后   → 提高 `--b` 即可
  · `C 证据缺失`  —— 该篇**全篇没有** own 链接              → 真值可能有争议，或链接在参考文献/脚注被解析器丢了
反向也查：**假阳性**（非 gold 判 yes）该篇有没有 own 链接。

## 两种模式
```json
"own": 作者自己的发布（our code is available / we release our code / available at github…）
"any": 任何代码托管链接（含引用他人的仓库，如 ALCE / FiD —— 这是历史假阳性的来源）
```
用法：`./.venv/Scripts/python.exe -u retrieval/tmp/_r2_cr_diag.py`
"""
from __future__ import annotations

import ast
import importlib.util as _iu
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

DEV = HERE / "data" / "r2dev"
fac = json.loads((DEV / "facets_v2_selected.json").read_text(encoding="utf-8"))
FACET = "code_release"
B = 6                      # reader 的每篇证据预算

fsp = _iu.spec_from_file_location("facets2", HERE / "tmp" / "_r2_facets_v2.py")
_f2m = _iu.module_from_spec(fsp)
sys.modules["facets2"] = _f2m
fsp.loader.exec_module(_f2m)
F2 = _f2m.F2

# 「作者自己的发布」——比锚点严（要求动作主语是本文）
OWN = re.compile(
    r"(?:our|the)\s+(?:code|data|implementation|model|dataset)s?\s+(?:is|are|will be|has been|have been)\s+"
    r"(?:publicly\s+|freely\s+)?(?:available|released|open[- ]sourced?)"
    r"|we\s+(?:will\s+)?(?:release|open[- ]source|publicly\s+release|make\s+(?:our|the|all))"
    r"|code\s+(?:is|will be|has been)\s+(?:publicly\s+)?(?:available|released)"
    r"|(?:implementation|code)\s+(?:is|will be|are)\s+(?:publicly\s+)?available\s+at"
    r"|available\s+at\s+(?:https?://)?(?:www\.)?github"
    r"|(?:我们)?(?:开源|代码已?公开)", re.I)
# 任何代码托管链接（含引用他人）
ANY = re.compile(r"github\.com|gitlab\.com|huggingface\.co|bitbucket|anonymous\.4open\.science|"
                 r"openreview\.net/forum|\bgithub\b", re.I)


def yes_set(s: str) -> set[str]:
    try:
        return set(ast.literal_eval(s)) if str(s).startswith("[") else set()
    except Exception:  # noqa: BLE001
        return set()


def main() -> int:
    gold = pd.read_csv(DEV / "gold_final2.csv")
    gold["gold"] = gold["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    G: dict[tuple[int, str], set[str]] = {}
    for (c, f), x in gold.groupby(["cluster", "facet"]):
        G[(int(c), str(f))] = set(x[x["gold"]]["docid"])
    rd = pd.read_csv(HERE / "results" / "R2_r2reader2_4_b6.csv")
    rd = rd[rd.facet == FACET].set_index("cluster")
    subq = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()

    print("=" * 118)
    print(f"【修路 #1｜`{FACET}` 证据侧诊断】每篇证据预算 b={B} ｜ 生产切块 ｜ 23 篇 gold")
    rows = []
    for ci in range(3):
        ch = pd.read_parquet(DEV / "prodchunk" / "mineru" / f"c{ci}.parquet")
        ls = pd.read_parquet(DEV / "clusters" / f"c{ci}.parquet")
        docs = [d for d in ls["docid"].tolist() if d in set(ch["docid"])]
        chunks, owner = [], []
        for d in docs:
            cs = ch[ch.docid == d]["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=16, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        qs = [F2[FACET][1]] + list(subq.get(FACET, []))
        S = enc.encode(qs, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32) @ C.T
        smax = S.max(axis=0)
        idx = {d: np.where(owner == d)[0] for d in docs}
        gset = G.get((ci + 1, FACET), set())
        yset = yes_set(rd.loc[ci + 1, "yes"])
        for d in docs:
            order = idx[d][np.argsort(-smax[idx[d]])]        # 该篇块按相关度排序
            ranks_own = [i + 1 for i, j in enumerate(order) if OWN.search(chunks[j])]
            ranks_any = [i + 1 for i, j in enumerate(order) if ANY.search(chunks[j])]
            top = order[:B]
            rows.append(dict(
                cluster=ci + 1, docid=d, n_chunks=len(order),
                gold=d in gset, judged=d in yset,
                own_topb=bool(OWN.search(" ".join(chunks[j] for j in top))),
                own_full=bool(ranks_own),
                rank_own=ranks_own[0] if ranks_own else 0,
                n_own=len(ranks_own),
                any_topb=bool(ANY.search(" ".join(chunks[j] for j in top))),
                any_full=bool(ranks_any),
                rank_any=ranks_any[0] if ranks_any else 0,
            ))
        print(f"  簇{ci + 1} 就绪（{len(docs)} 篇 / {len(chunks)} 块 ｜ gold {len(gset)} ｜ 判 yes {len(yset)}）",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "R2_CR_EVIDENCE_DIAG.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 118)
    print("【混淆矩阵 × 证据状态】")
    print(f"  {'象限':<26}{'篇':>4}{'own 在 top-b':>14}{'own 在全篇':>12}{'any 在 top-b':>14}"
          f"{'own 中位排名':>13}")
    quad = [("gold & 判yes（TP）", df.gold & df.judged),
            ("**gold & 判no（FN）**", df.gold & ~df.judged),
            ("非gold & 判yes（FP）", ~df.gold & df.judged),
            ("非gold & 判no（TN）", ~df.gold & ~df.judged)]
    for nm, m in quad:
        s = df[m]
        if not len(s):
            continue
        med = s[s.rank_own > 0].rank_own.median()
        print(f"  {nm:<26}{len(s):>4}{s.own_topb.mean():>14.1%}{s.own_full.mean():>12.1%}"
              f"{s.any_topb.mean():>14.1%}{(med if med == med else float('nan')):>13.1f}")

    fn = df[df.gold & ~df.judged]
    print("\n" + "=" * 118)
    print(f"【★ 归因：{len(fn)} 篇「gold 但判 no」到底缺什么】")
    a = fn[fn.own_topb]
    b_ = fn[~fn.own_topb & fn.own_full]
    c = fn[~fn.own_full]
    d_ = fn[~fn.own_topb & fn.any_topb]          # top-b 里有 any 但没 own → 判官正确地区分了「他人仓库」
    print(f"  A **判官放过**（own 在 top-{B} 块里）      {len(a):>3} 篇  → 提示词/裁剪问题"
          f"{'' if not len(a) else '：' + ', '.join(f'{r.cluster}·{r.docid}(rank{r.rank_own})' for r in a.itertuples())}")
    print(f"  B **预算不够**（own 在全篇，但排名 >{B}）  {len(b_):>3} 篇  → 提高 --b 或加锚点块"
          f"{'' if not len(b_) else '：' + ', '.join(f'{r.cluster}·{r.docid}(rank{r.rank_own})' for r in b_.itertuples())}")
    print(f"  C **证据缺失**（全篇没有 own 链接）        {len(c):>3} 篇  → 真值争议 / 解析器丢了"
          f"{'' if not len(c) else '：' + ', '.join(f'{r.cluster}·{r.docid}' for r in c.itertuples())}")
    print(f"  （其中 {len(d_)} 篇 top-{B} 里有 **他人** 仓库链接 → 判官区分「本文 vs 他人」是**对的**，"
          f"是锚点会错的地方）")

    fp = df[~df.gold & df.judged]
    if len(fp):
        print(f"\n【假阳性 {len(fp)} 篇】own 在 top-{B} 里 {fp.own_topb.mean():.0%} ｜ "
              f"全篇 {fp.own_full.mean():.0%}")
        for r in fp.itertuples():
            print(f"    {r.cluster}·{r.docid}  own_topb={r.own_topb} own_full={r.own_full} "
                  f"rank_own={r.rank_own} any_topb={r.any_topb}")

    print("\n【★ 需要的 b 是多少？】gold 篇里 own 链接的排名分布")
    gd = df[df.gold & (df.rank_own > 0)]
    for k in (1, 3, 6, 10, 15, 20, 30):
        cov = float((gd.rank_own <= k).mean()) if len(gd) else float("nan")
        print(f"    排名 ≤ {k:>2} ：覆盖 {int((gd.rank_own <= k).sum()):>3}/{len(gd)} gold 篇 = **{cov:.1%}**")
    print("\n【该篇的块总数（上限）】gold 篇 n_chunks 中位 "
          f"{gd.n_chunks.median():.0f} ｜ 全篇 {df.n_chunks.median():.0f}")
    print(f"\n→ {HERE / 'results' / 'R2_CR_EVIDENCE_DIAG.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
