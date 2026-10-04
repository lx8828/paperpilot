r"""**C3 验证器标定**：NLI 能不能替代 / 分担 LLM 判官？（零新标注）

## 思路
我们手上已有 **1,400 对 (facet, 篇) 的 LLM 判官真值**（`gold_final3.csv`，覆盖每 facet 的全部 50 篇）
→ 用**同一批对子**算 4 个候选验证器的分，看谁能复现 LLM gold。

| 验证器 | 假设 | 成本 | 备注 |
|---|---|---|---|
| `regex` | — | ¥0 | `_m.judge()` 词面锚点（已有，当年"锚点基线"） |
| `rerank` | 题面+子查询 | ¥0（本地） | bge-reranker-v2-m3 —— **相关性**，不是蕴含 |
| `nli_xnli` | **中文论断** | ¥0（本地） | `mDeBERTa-v3-base-mnli-xnli`（多语，跨语言） |
| `nli_fever` | **英文 `para`** | ¥0（本地） | `DeBERTa-v3-base-mnli-fever-anli`（FEVER 调过） |

## 三张表
| 表 | 指标 | 回答什么 |
|---|---|---|
| ① | **AUC**（篇级，50 篇/facet）+ best-F1 | **判别力**：能不能把 gold 篇和干扰篇分开 |
| ② | **同规模 setF1**（取 top-\|gold\| 篇） | 与 LLM gold 的**一致性**（操作点） |
| ③ | **★ 级联可行性曲线** | 阈值 τ 下：**gold 篇留存率** vs **块保留比** → NLI 粗筛能筛掉多少而不漏 |

⚠️ **偏置声明**：LLM gold 是**带判据**（prompt 里给了具体证据条件）做的 → 该比较**天然偏向 LLM**。
本表测的是"**通用蕴含与我们的操作性判据有多相关**"，**不是**"NLI 模型强弱"。

⚠️ **模型不共驻**（6GB 卡教训）：逐个加载 → 卸载 → 再下一个。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_nli_calib.py
"""
from __future__ import annotations

import argparse
import gc
import importlib.util as _iu
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
DEV = HERE / "data" / "r2dev"
XNLI = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
FEVER = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
RERANK = "BAAI/bge-reranker-v2-m3"
# ⚠️ NLI 模型用 `curl` 下到**本地目录**（HF 缓存对 ~1GB 大文件会 TLS 卡死，见 `_nli_curl.py`）
NLI_LOCAL = {XNLI: HERE / "data" / "nli_models" / "mDeBERTa-v3-base-mnli-xnli",
             FEVER: HERE / "data" / "nli_models" / "DeBERTa-v3-base-mnli-fever-anli"}


def src_of(name: str) -> str:
    p = NLI_LOCAL.get(name)
    return str(p) if p and (p / "config.json").exists() else name


TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("stdm", HERE / "tmp" / "_r2_std_metrics.py")
F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """秩统计 AUC（免 sklearn）。"""
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """扫阈值取 best-F1，返回 (f1, τ)。"""
    best, bt = 0.0, 0.0
    for t in np.unique(np.round(s, 4)):
        p = s >= t
        tp = int((p & y).sum())
        if not tp:
            continue
        pr, rc = tp / p.sum(), tp / y.sum()
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best:
            best, bt = f1, float(t)
    return best, bt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="50")
    ap.add_argument("--chunktag", default="mineru")
    ap.add_argument("--batch", type=int, default=64, help="reranker 批量")
    ap.add_argument("--batch-nli", type=int, default=8,
                    help="NLI 批量（DeBERTa-v3 的**解耦注意力**中间张量很大：6GB 卡上 "
                         "batch 64×512 会撑爆显存 → Windows 换页假死）")
    ap.add_argument("--skip", default="", help="跳过的验证器，逗号分隔（如 nli_fever）")
    ap.add_argument("--no-cache", action="store_true", help="忽略已有的逐臂缓存")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}
    if args.corpus == "50":
        CDIR, KDIR = DEV / "corpus50", DEV / "prodchunk50"
        GF = DEV / "gold_final3.csv"
        pmap_f = DEV / "corpus50" / "pdf_map_all.json"
    else:
        CDIR, KDIR = DEV / "clusters", DEV / "prodchunk"
        GF = DEV / "gold_final2.csv"
        pmap_f = DEV / "pdf_map.json"
    pm_ok = {d for d, v in json.loads(pmap_f.read_text(encoding="utf-8")).items() if v.get("ok")}
    gg = pd.read_csv(GF)
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json")
                                  .read_text(encoding="utf-8"))["per_combo"]}
    subq = json.loads((DEV / "subqueries_v2.json").read_text(encoding="utf-8"))

    print("=" * 126)
    print("【C3 验证器标定】对照 1,400 对 LLM 判官真值（覆盖每 facet 全部 50 篇）")
    print(f"  真值 {GF.name} ｜ 语料 {args.corpus} ｜ 语料池是否有 gold 之外的篇？"
          f"{len(pm_ok)} 篇可检索")
    t0 = time.time()

    # ═══════ 阶段 0：准备数据（CPU）═══════
    DATA = []
    for ci in range(3):
        ch = pd.read_parquet(KDIR / args.chunktag / f"c{ci}.parquet")
        ls = pd.read_parquet(CDIR / f"c{ci}.parquet")
        docs = [d for d in ls["docid"].tolist() if d in set(ch["docid"]) and d in pm_ok]
        chunks, owner = [], []
        for d in docs:
            cs = ch[ch.docid == d]["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        owner = np.array(owner)
        didx = {d: np.where(owner == d)[0] for d in docs}
        packs = []
        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            SQ = [str(q) for q in subq.get(facet, []) if str(q).strip()][:3]
            packs.append(dict(
                facet=facet, gold=gold, docs=docs,
                y=np.array([d in gold for d in docs]),
                hz=str(F2[facet][1]),                              # 中文论断
                he=str(F2[facet][3]),                              # 英文 para
                qr=str(F2[facet][1]) + " " + " ".join(SQ),          # 重排用（题面+子查询）
                pat=str(F2[facet][0])))                            # 正则锚点
        DATA.append(dict(ci=ci, chunks=chunks, owner=owner, didx=didx, nb=len(chunks), packs=packs))
        print(f"  簇{ci + 1}：{len(docs)} 篇 / {len(chunks)} 块 / {len(packs)} facet"
              f" ｜ 待判 {len(packs) * len(chunks):,} 对/facet 全库 ｜ {time.time() - t0:.0f}s",
              flush=True)

    V: dict[str, dict] = {}
    for cd in DATA:
        V.setdefault("regex", {})[cd["ci"]] = _regex_scores(cd, pm_ok)
    print(f"  [regex] 完成 {time.time() - t0:.0f}s", flush=True)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    CACHE = HERE / "results" / "nli_cache"
    CACHE.mkdir(parents=True, exist_ok=True)

    def run_ce(model_name: str, key: str, hyp_of, order: str, bs: int) -> None:
        """cross-encoder 路线（重排 / NLI 通用）：逐 cluster 批量 → 块分数组。

        ⚠️ **输入顺序约定不同**（两族模型文档不一致，弄反会显著掉分）：
        · `order="qh"`（cross-encoder / reranker）：`tokenizer(query, passage)`
        · `order="ph"`（NLI，如 MoritzLaurer 系列）：`tokenizer(premise, hypothesis)`
        因 premise（块）远长于 hypothesis（论断），`truncation` 默认 `longest_first`
        → 截的是块，**论断保留完整** ✓
        """
        if key in skip:
            print(f"  [{key}] 已跳过")
            return
        # ★ **逐臂逐簇落盘缓存**：崩一次不再丢整臂（第一版 rerank 跑了 9 分钟，
        #   卡在 NLI 阶段被杀 → 全丢。这里每算完一簇立刻 `npz` 落盘，重跑直接命中。）
        V.setdefault(key, {})
        if not args.no_cache:
            for cd in DATA:
                f2 = CACHE / f"{key}_c{cd['ci']}.npz"
                if f2.exists():
                    with np.load(f2) as z:
                        V[key][cd["ci"]] = {k2: z[k2].astype(np.float32) for k2 in z.files}
            hit = sorted(V[key])
            if len(hit) == len(DATA):
                print(f"  [{key}] ✅ {len(DATA)} 簇全部命中缓存 → 跳过推理")
                return
            print(f"  [{key}] 缓存命中 {[h + 1 for h in hit]}/{len(DATA)} 簇 → 只跑 "
                  f"{[cd['ci'] + 1 for cd in DATA if cd['ci'] not in hit]}")
        src = src_of(model_name)
        tok = AutoTokenizer.from_pretrained(src)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # ★ bf16：RTX 4050（Ada, cc8.9）支持；`dtype=` 是新 API，`torch_dtype=` 是旧 API
        mdl = None
        if dev == "cuda":
            for kw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}, {}):
                try:
                    mdl = AutoModelForSequenceClassification.from_pretrained(src, **kw)
                    break
                except TypeError:
                    continue
        if mdl is None:
            mdl = AutoModelForSequenceClassification.from_pretrained(src)
        mdl.eval().to(dev)
        eidx = next((int(k2) for k2, v2 in (getattr(mdl.config, "id2label", {}) or {}).items()
                     if "entail" in str(v2).lower()), None)
        mode = "nli" if eidx is not None else "regress"
        print(f"  [{key}] {model_name} ｜ device={dev} ｜ mode={mode} ｜ order={order}"
              f" ｜ dtype={next((p.dtype for p in mdl.parameters()), None)}"
              f"{f' ｜ entail_idx={eidx}' if eidx is not None else ''}", flush=True)

        # ★★ **按块长排序再批**：`padding=True` 补到批内最长，乱序时每批都被长块拖满；
        #     排序后批内长度相近（我们的块 1.8k~3.8k 字符，跨度大）→ 实测可省一半以上算力。
        for cd in DATA:
            if cd["ci"] in V.get(key, {}):
                print(f"    簇{cd['ci'] + 1} 跳过（缓存）", flush=True)
                continue
            o = np.argsort([len(c) for c in cd["chunks"]])
            ch_sorted = [cd["chunks"][j] for j in o]
            per: dict[str, np.ndarray] = {}
            for p in cd["packs"]:
                hyp = hyp_of(p)
                a_in, b_in = ([hyp] * cd["nb"], ch_sorted) if order == "qh" \
                    else (ch_sorted, [hyp] * cd["nb"])
                sc = np.empty(cd["nb"], dtype=np.float32)
                with torch.no_grad():
                    for i in range(0, cd["nb"], bs):
                        enc = tok(a_in[i:i + bs], b_in[i:i + bs],
                                  padding=True, truncation=True, max_length=512,
                                  return_tensors="pt").to(dev)
                        lg = mdl(**enc).logits.float()
                        if mode == "nli":
                            sc[i:i + bs] = torch.softmax(lg, -1)[:, eidx].cpu().numpy()
                        else:
                            sc[i:i + bs] = lg.reshape(-1).cpu().numpy()
                out = np.empty_like(sc)
                out[o] = sc                      # 还原到原块顺序
                per[p["facet"]] = out
            V[key][cd["ci"]] = per
            np.savez_compressed(CACHE / f"{key}_c{cd['ci']}.npz", **per)
            f0 = cd["packs"][0]["facet"]
            s0 = per[f0]
            print(f"    簇{cd['ci'] + 1} 完成 ｜ {time.time() - t0:.0f}s ｜ 自检 {f0}："
                  f"均 {s0.mean():.3f} ｜ 范围 [{s0.min():.3f}, {s0.max():.3f}]"
                  f" ｜ NaN {int(np.isnan(s0).sum())}", flush=True)
        del mdl, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ⚠️ 逐个加载 → 卸载（6GB 卡不共驻的教训）；NLI 用**小批量**（DeBERTa 解耦注意力吃显存）
    run_ce(RERANK, "rerank", lambda p: p["qr"], "qh", args.batch)
    run_ce(XNLI, "nli_xnli", lambda p: p["hz"], "ph", args.batch_nli)
    run_ce(FEVER, "nli_fever", lambda p: p["he"], "ph", args.batch_nli)

    # ═══════ 汇总：把块分折成篇分 ═══════
    ARMS = [k for k in ("regex", "rerank", "nli_xnli", "nli_fever") if k in V]
    doc = []          # 篇级
    chk = []          # 块级（级联曲线用）
    for cd in DATA:
        ci = cd["ci"]
        for p in cd["packs"]:
            gold = p["gold"]
            for a in ARMS:
                s = V[a][ci][p["facet"]]
                if a == "regex":
                    s = s.astype(np.float32)
                for d in p["docs"]:
                    idxs = cd["didx"][d]
                    doc.append(dict(cluster=ci + 1, facet=p["facet"], arm=a, docid=d,
                                    gold=int(d in gold), score=float(s[idxs].max())))
                for j in range(cd["nb"]):
                    chk.append(dict(cluster=ci + 1, facet=p["facet"], arm=a,
                                    docid=str(cd["owner"][j]),
                                    gold_chunk=int(cd["owner"][j] in gold), score=float(s[j])))
    D, C = pd.DataFrame(doc), pd.DataFrame(chk)
    D["y"] = D["gold"].astype(bool)          # 表① 的 AUC/best-F1 用布尔真值
    D.to_csv(HERE / "results" / "R2_NLI_CALIB_DOC.csv", index=False, encoding="utf-8-sig")
    C.to_csv(HERE / "results" / "R2_NLI_CALIB_CHUNK.csv", index=False, encoding="utf-8-sig")

    # ── 表① 判别力 ──
    print("\n" + "=" * 126)
    print("【表① 判别力】篇级（每 facet 全部 50 篇，共 1,400 对）｜ 随机 AUC = 0.5")
    print(f"  {'验证器':<12}{'AUC':>8}{'best-F1':>10}{'@τ':>8}{'gold 均分':>11}{'干扰 均分':>11}{'分离度':>9}")
    T1 = []
    for a in ARMS:
        t = D[D["arm"] == a]
        au = np.mean([auc(x["y"].values, x["score"].values)
                      for _, x in t.groupby(["cluster", "facet"])])
        f1s, taus = zip(*[best_f1(x["y"].values, x["score"].values)
                          for _, x in t.groupby(["cluster", "facet"])])
        g, n = t[t.gold == 1]["score"].mean(), t[t.gold == 0]["score"].mean()
        sd = t["score"].std()
        T1.append(dict(arm=a, auc=au, f1=float(np.mean(f1s)), tau=float(np.mean(taus)),
                       g=g, n=n, sep=(g - n) / sd if sd else 0.0))
        print(f"  {a:<12}{au:>8.3f}{np.mean(f1s):>10.3f}{np.mean(taus):>8.2f}"
              f"{g:>11.3f}{n:>11.3f}{(g - n) / sd if sd else 0:>9.2f}")
    pd.DataFrame(T1).to_csv(HERE / "results" / "R2_NLI_CALIB_T1.csv",
                            index=False, encoding="utf-8-sig")

    # ── 表② 同规模操作点 ──
    print("\n" + "=" * 126)
    print("【表② 同规模操作点】每 facet 取验证器 **top-|gold|** 篇 → 与 LLM gold 的 setF1")
    print(f"  {'验证器':<12}{'setP':>8}{'setR':>8}{'setF1':>9}{'StRecall':>10}{'完胜占比':>10}{'全错占比':>10}")
    T2 = []
    for a in ARMS:
        ps, rs, f1s, srs, win, lose = [], [], [], [], 0, 0
        for _, x in D[D["arm"] == a].groupby(["cluster", "facet"]):
            x = x.sort_values("score", ascending=False)
            ng = int(x["gold"].sum())
            pick = set(x.head(ng)["docid"])
            gold = set(x[x.gold == 1]["docid"])
            inter = len(pick & gold)
            ps.append(inter / max(len(pick), 1))
            rs.append(inter / max(len(gold), 1))
            f1s.append(2 * inter / max(len(pick) + len(gold), 1))
            srs.append(inter / max(len(gold), 1))
            win += int(inter == len(gold))
            lose += int(inter == 0)
        n = len(f1s)
        T2.append(dict(arm=a, sp=float(np.mean(ps)), sr=float(np.mean(rs)),
                       f1=float(np.mean(f1s)), win=win / n, lose=lose / n))
        print(f"  {a:<12}{np.mean(ps):>8.3f}{np.mean(rs):>8.3f}{np.mean(f1s):>9.3f}"
              f"{np.mean(srs):>10.3f}{win / n:>10.1%}{lose / n:>10.1%}")
    print("  （LLM gold 自身的 setF1 = 1.000，是**上界**，不是可比项）")
    pd.DataFrame(T2).to_csv(HERE / "results" / "R2_NLI_CALIB_T2.csv",
                            index=False, encoding="utf-8-sig")

    # ── 表③ ★ 级联可行性 ──
    print("\n" + "=" * 126)
    print("【表③ ★ 级联可行性】按 τ 逐块过滤后：**块保留比** vs **gold 篇留存率**")
    print("  （map 阶段要「别漏」→ 看 τ 低时能否既筛掉大量块、又不丢 gold 篇）")
    print(f"  {'验证器':<12}{'τ':>7}{'块保留比':>11}{'gold 篇留存':>13}{'gold 块留存':>13}")
    T3 = []
    for a in ARMS:
        ta = C[C["arm"] == a]
        gd = D[(D["arm"] == a) & (D.gold == 1)]
        for tau in TAUS:
            keep = ta[ta["score"] >= tau]
            # gold 篇留存：gold 篇中**至少有一个块存活**的比例（facet 内先算再平均）
            ret = []
            for (cl, fa), gx in gd.groupby(["cluster", "facet"]):
                kd = set(keep[(keep.cluster == cl) & (keep.facet == fa)]["docid"])
                ret.append(len(set(gx["docid"]) & kd) / max(len(gx), 1))
            ratio = len(keep) / max(len(ta), 1)
            gchunk = (keep[keep.gold_chunk == 1].shape[0]
                      / max(ta[ta.gold_chunk == 1].shape[0], 1))
            T3.append(dict(arm=a, tau=tau, keep=ratio,
                           gold_doc=float(np.mean(ret)), gold_chunk=gchunk))
            print(f"  {a:<12}{tau:>7.2f}{ratio:>11.1%}{np.mean(ret):>13.1%}{gchunk:>13.1%}")
    pd.DataFrame(T3).to_csv(HERE / "results" / "R2_NLI_CALIB_T3.csv",
                            index=False, encoding="utf-8-sig")
    print(f"\n  → 已写 results/R2_NLI_CALIB_{{DOC,CHUNK,T1,T2,T3}}.csv（{time.time() - t0:.0f}s）")
    return 0


def _regex_scores(cd: dict, pm_ok: set) -> dict:
    """正则锚点：块级 0/1。"""
    import re
    out = {}
    for p in cd["packs"]:
        try:
            rx = re.compile(p["pat"], re.I)
        except re.error:
            out[p["facet"]] = np.zeros(cd["nb"], dtype=np.float32)
            continue
        out[p["facet"]] = np.array([1.0 if rx.search(c) else 0.0
                                    for c in cd["chunks"]], dtype=np.float32)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
