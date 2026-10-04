"""**A1｜真值重标 v2（生产切块口径）**

## 与 v1 的唯一差别：**证据块换成生产章节段落切块（MinerU 口径）**
v1（`_r2_gold_recalib.py`）的证据池取自 `chunks_of(full_paper)` = **固定窗 1000/900 over LitSearch 纯文本**。
现在切块已换成生产实现（`prodchunk/mineru/`，19 篇/簇、中位 1,493~1,848 字符），
→ 真值必须在**同一批块**上重标，否则 reader / 指标建在错真值上。

## 协议**一字不改**（从 v1 import，保证可比）
· 证据池 = （facet 正则命中的块）∪（**干净查询** `zh` + 净化子查询的语义 top-k 块）
· 判官 A=`PAPERPILOT_LLM`(deepseek-chat) ／ 判官 B=`PAPERPILOT_JUDGE`(glm-4-flash)，四档 + 逐字 quote
· 采信：双方 `entail` → yes；双方非 `entail` → no；**分歧走第三轮严格对抗提示**
· `max_seq_length`：v1 是 512（截断坑）→ 本版用 **8192**（生产值；新块最长 ~4,000 字符）

产物：`data/r2dev/gold_recalib2.csv` + `gold_recalib2_evidence.json`（抽检用，含全文证据）
      `results/R2_GOLD_RECALIB2_SUMMARY.csv`

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_recalib2.py --limit 12   # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_recalib2.py --workers 8  # 全量
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gr = _load("goldrecal", HERE / "tmp" / "_r2_gold_recalib.py")   # v1（协议来源）
SYS, SYS_STRICT, call, F = gr.SYS, gr.SYS_STRICT, gr.call, gr.F
rx_toks = gr.rx_toks
_m = gr._m

DEV = HERE / "data" / "r2dev"
CACHE = DEV / "clusters"
CHUNKDIR = DEV / "prodchunk" / "mineru"
SUBQ = DEV / "subqueries.json"
OUT = DEV / "gold_recalib2.csv"
EVID = DEV / "gold_recalib2_evidence.json"
EV_TOPK_SEM, EV_MAX_CHUNKS, EV_MAX_CHARS = 3, 4, 4200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--chunktag", default="mineru")
    ap.add_argument("--all-facets", action="store_true",
                    help="跑**全部 19 个 facet × 3 簇**（57 个组合），而不是各簇的 usable 名单（35 个）")
    ap.add_argument("--out", default="gold_recalib2.csv")
    ap.add_argument("--facets-file", default="",
                    help="用另一份 facet 表（如 `_r2_facets_v2.py` 里的 `F2`）")
    # ── 扩语料（150 篇）用：只判**新增篇**，复用已有真值 ──
    ap.add_argument("--corpus-dir", default="clusters",
                    help="语料目录（默认 `clusters`；扩语料用 `corpus50`）")
    ap.add_argument("--chunkdir", default="prodchunk/mineru",
                    help="切块目录（默认 `prodchunk/mineru`；扩语料用 `prodchunk50/mineru`）")
    ap.add_argument("--map", default="pdf_map.json",
                    help="PDF 映射（取 `ok` 篇；扩语料用 `corpus50/pdf_map_all.json`）")
    ap.add_argument("--select", default="",
                    help="题集文件（如 `facets_v2_selected.json`）→ 每簇 facet 列表取自它")
    ap.add_argument("--reuse-gold", default="",
                    help="已有真值 CSV（如 `gold_final2.csv`）→ **跳过已判过的 (簇,facet,篇)**")
    args = ap.parse_args()
    corpus_dir = DEV / args.corpus_dir
    chunkdir = DEV / args.chunkdir
    map_path = DEV / args.map

    F_use = F
    if args.facets_file:
        fs = _load("facets_alt", HERE / "tmp" / args.facets_file)
        F_use = getattr(fs, "F2", None) or getattr(fs, "F")
        print(f"  [facet 表] 用 `{args.facets_file}`（{len(F_use)} 个 facet）")

    base_l, _, model_l = llm.config("PAPERPILOT_LLM")
    base_j, _, model_j = llm.config("PAPERPILOT_JUDGE")
    print(f"判官 A：{model_l}（PAPERPILOT_LLM）｜判官 B：{model_j}（PAPERPILOT_JUDGE）")
    print(f"切块来源：{CHUNKDIR}（生产章节段落·MinerU 口径）\n")

    # ── 每簇的 facet 列表：`--select`（题集）优先，否则 `meta.json`（旧口径）──
    if args.select:
        sel = json.loads((DEV / args.select).read_text(encoding="utf-8"))
        _per: dict[int, list[str]] = {}
        for k in sel["per_combo"]:
            c, f = k.split("|")
            _per.setdefault(int(c), []).append(f)
        n_ci = max(_per) if _per else 0
        facets_of = lambda ci: sorted(set(_per.get(ci + 1, [])))          # noqa: E731
    else:
        meta = json.loads((corpus_dir / "meta.json").read_text(encoding="utf-8"))
        n_ci = len(meta)
        facets_of = (lambda ci: sorted(F_use.keys()) if args.all_facets                # noqa: E731
                     else list(meta[ci]["usable"]))
    subq_all = json.loads(SUBQ.read_text(encoding="utf-8"))
    pmap = json.loads(map_path.read_text(encoding="utf-8"))
    keep = {d for d, v in pmap.items() if v.get("ok")}
    # ── 已有真值（跳过已判过的对）──
    already: set[tuple[int, str, str]] = set()
    if args.reuse_gold:
        rg = pd.read_csv(DEV / args.reuse_gold)
        already = {(int(r["cluster"]), str(r["facet"]), str(r["docid"]))
                   for _, r in rg.iterrows()}
        print(f"  [复用真值] `{args.reuse_gold}` 有 {len(already)} 对 → 本次只判**新增篇**")
    out_path = DEV / args.out
    evid_path = out_path.with_name(out_path.stem + "_evidence.json")
    print(f"  [语料] {corpus_dir.name} ｜ [切块] {args.chunkdir} ｜ [映射] {args.map}"
          f" ｜ [题集] {args.select or 'meta.usable'}\n")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192                     # ⚠️ 生产值（v1 用 512 是截断坑）
    if dev == "cuda":
        enc.half()

    jobs: list[dict] = []
    for ci in range(n_ci):
        ls = pd.read_parquet(corpus_dir / f"c{ci}.parquet")
        titles = {str(r["docid"]): str(r.get("title") or "") for _, r in ls.iterrows()}
        ch = pd.read_parquet(chunkdir / f"c{ci}.parquet")
        ch = ch[ch["docid"].isin(keep)]
        docs = [d for d in ls["docid"].tolist() if d in keep]
        chunks, owner = [], []
        for d in docs:
            g = ch[ch["docid"] == d]
            cs = g["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        if not chunks:
            continue
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=16, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        low = [c.lower() for c in chunks]
        idx = {d: np.where(owner == d)[0] for d in docs}
        print(f"【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块", flush=True)

        for facet in facets_of(ci):
            pat, zh, anchor, para = F_use[facet]
            hit = np.array([bool(_m.judge(c, facet)) for c in low])
            anchor_gold = {d for d in docs if hit[idx[d]].any()}
            # 默认（usable 名单）跳过无分辨率的题；`--all-facets` / `--select` 下**不跳** ——
            # 后者是题集已定（扩语料），必须把每个组合的真值都补齐
            if not (args.all_facets or args.select) and (not anchor_gold
                                                         or len(anchor_gold) == len(docs)):
                print(f"    ⏭️ {facet}：锚点正例 {len(anchor_gold)}/{len(docs)} → 无分辨率，跳过")
                continue
            rx = rx_toks(pat)
            desens = [" ".join(w for w in s.split() if w.lower() not in rx) or s
                      for s in subq_all.get(facet, [])]
            qvec = enc.encode([zh] + desens, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
            sim = qvec @ C.T
            for d in docs:
                if (ci + 1, facet, d) in already:     # ★ 复用已有真值：只补**新增篇**
                    continue
                ids = idx[d]
                sem_top = ids[np.argsort(-sim[:, ids].max(axis=0))[:EV_TOPK_SEM]]
                anch = ids[hit[ids]][:EV_MAX_CHUNKS]
                orderl, seen = [], set()
                for j in list(anch) + list(sem_top):
                    if int(j) not in seen:
                        seen.add(int(j))
                        orderl.append(int(j))
                orderl = orderl[:EV_MAX_CHUNKS]
                ev, evid, tot = [], [], 0
                for j in orderl:
                    t = chunks[j]
                    if tot + len(t) > EV_MAX_CHARS:
                        t = t[: max(0, EV_MAX_CHARS - tot)]
                    if not t:
                        break
                    ev.append(t)
                    evid.append(int(j))
                    tot += len(t)
                jobs.append(dict(cluster=ci + 1, facet=facet, docid=d,
                                 anchor_gold=bool(d in anchor_gold),
                                 title=str(titles.get(d, ""))[:120],
                                 evidence="\n---\n".join(ev),
                                 ev_idx=evid,
                                 claim=(F_use[facet][1] if str(F_use[facet][1]).startswith("本文")
                                        else f"该论文{F_use[facet][1]}"),
                                 facet_def=str(F_use[facet][1])))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"\n待判 (cluster, facet, doc) 对：**{len(jobs)}** ｜ 每对 2 次调用 + 分歧第三轮\n")

    done: dict[tuple, dict] = {}
    if out_path.exists() and not args.fresh:
        for r in pd.read_csv(out_path).to_dict("records"):
            done[(int(r["cluster"]), str(r["facet"]), str(r["docid"]))] = r
        print(f"已有缓存 {len(done)} 条")
    todo = [j for j in jobs if (j["cluster"], j["facet"], j["docid"]) not in done]

    lock = threading.Lock()
    cnt = {"n": 0}

    def work(j: dict) -> dict:
        user = (f"论文标题：{j['title']}\n\n论断：**{j['claim']}**\n\n"
                f"论文原文片段：\n{j['evidence']}")
        a = call("PAPERPILOT_LLM", SYS, user)
        b = call("PAPERPILOT_JUDGE", SYS, user)
        la = str(a.get("label") or "ERR").lower()
        lb = str(b.get("label") or "ERR").lower()
        final, third = "", {}
        if la == "entail" and lb == "entail":
            final = "yes"
        elif "entail" not in (la, lb) and "ERR" not in (la, lb):
            final = "no"
        else:
            third = call("PAPERPILOT_LLM", SYS_STRICT,
                         f"论文标题：{j['title']}\n\n论断：**{j['claim']}**\n\n"
                         f"论文原文片段：\n{j['evidence']}")
            final = "yes" if str(third.get("answer", "")).upper().startswith("Y") else "no"
        with lock:
            cnt["n"] += 1
            if cnt["n"] % 25 == 0:
                print(f"  判了 {cnt['n']}/{len(todo)} …", flush=True)
        return dict(cluster=j["cluster"], facet=j["facet"], docid=j["docid"],
                    anchor_gold=j["anchor_gold"], agent_A=model_l, A_label=la,
                    A_conf=a.get("confidence"), A_quote=str(a.get("quote") or "")[:160],
                    agent_B=model_j, B_label=lb, B_conf=b.get("confidence"),
                    B_quote=str(b.get("quote") or "")[:160],
                    agree=int(la == lb), need_third=int(bool(third)),
                    third=str(third.get("answer") or "")[:12],
                    new_gold=final, A_err=str(a.get("_err") or "")[:60],
                    B_err=str(b.get("_err") or "")[:60], evidence_chars=len(j["evidence"]),
                    ev_idx=",".join(map(str, j["ev_idx"])))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, todo):
            done[(r["cluster"], r["facet"], r["docid"])] = r
    df = pd.DataFrame(list(done.values())).sort_values(["cluster", "facet", "docid"])
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    # 证据全文（抽检用）
    evid_path.write_text(json.dumps({f"{j['cluster']}|{j['facet']}|{j['docid']}":
                                     {"claim": j["claim"], "title": j["title"],
                                      "evidence": j["evidence"], "ev_idx": j["ev_idx"]}
                                     for j in jobs}, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"\n完成 {len(todo)} 对（{time.time() - t0:.0f}s）→ {out_path.name} + {evid_path.name}")

    print("\n" + "=" * 112)
    print("【锚点真值 vs 双 LLM 重标真值（生产切块口径）】")
    df["anchor_gold"] = df["anchor_gold"].astype(bool)
    df["new_gold"] = df["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    rows = []
    for (c, f), g in df.groupby(["cluster", "facet"]):
        A = set(g[g["anchor_gold"]]["docid"])
        N = set(g[g["new_gold"]]["docid"])
        tp = len(A & N)
        rows.append(dict(cluster=c, facet=f, n_paper=len(g), n_anchor=len(A), n_new=len(N), tp=tp,
                         anchor_P=tp / len(A) if A else float("nan"),
                         anchor_R=tp / len(N) if N else float("nan"),
                         added=len(N - A), removed=len(A - N)))
    s = pd.DataFrame(rows)
    print(f"  {'簇':>3}{'facet':<18}{'锚点正例':>9}{'新真值':>8}{'新增':>6}{'删去':>6}"
          f"{'锚点P':>8}{'锚点R':>8}")
    for _, r in s.iterrows():
        print(f"  {int(r['cluster']):>3}{r['facet']:<18}{int(r['n_anchor']):>9}{int(r['n_new']):>8}"
              f"{int(r['added']):>6}{int(r['removed']):>6}{r['anchor_P']:>8.3f}{r['anchor_R']:>8.3f}")
    print(f"\n  **合计**：锚点正例 {int(s['n_anchor'].sum())} ｜ 新真值 {int(s['n_new'].sum())}"
          f" ｜ 锚点 P {df[df['anchor_gold']]['new_gold'].mean():.3f}"
          f" ｜ 锚点 R {df[df['new_gold']]['anchor_gold'].mean():.3f}（= 1 − 漏报率）")
    print(f"  （旧口径标定：P 0.895 / R 0.567，即漏 ~43%）")
    print(f"  两判官一致率 {df['agree'].mean():.3f} ｜ 走第三轮 {df['need_third'].mean():.3f}"
          f" ｜ 错误 A {int((df['A_err'] != '').sum())} / B {int((df['B_err'] != '').sum())}")
    print(f"  新真值篇数/题：均 {s['n_new'].mean():.2f}（锚点 {s['n_anchor'].mean():.2f}）")
    # ⚠️ 汇总文件名跟着 `--out` 走，避免不同 facet 表互相覆盖
    smry = HERE / "results" / f"R2_GOLD_{out_path.stem.upper()}_SUMMARY.csv"
    s.to_csv(smry, index=False, encoding="utf-8-sig")
    print(f"\n→ {smry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
