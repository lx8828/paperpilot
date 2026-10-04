"""**L3 重算：ALCE 引用指标换 NLI 判定器**（并排词面代理 → 量化"同源偏袒"）。

## 为什么要重算
之前的 `cite_recall=0.33` 用的是**词面锚点**判定，而真值（gold 篇）也是由**同一批锚点词**确定的
→ **判定器与真值同源** → 数字被系统性高估（且结构上无法发现"说反了"）。

## 本脚本做三件事
1. **NLI 判定**（`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`，中文论断 × 英文证据 —— 上一步实测最优路线）
   - `cite_recall` = 逐 claim：被引块**拼接后**是否蕴含该论断（ALCE 定义）
   - `cite_precision` = 逐 citation：该块单独是否蕴含论断
   - `contra_rate` = 逐 claim：是否出现 contradiction（**新能力**：抓"说反了"）
   - 三个阈值都报（绝对 entail 分普遍偏低，0.5 不一定合适）
2. **词面代理并排**（同一批 claim、同一批引用）→ 直接读出**偏袒幅度**
3. **导出人工校准集**（40~60 对分层抽样 + NLI 预测，供人工标 支持/部分/不支持/说反了）

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_l3_nli.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / "data" / "r2dev" / "clusters"
CHUNK, OVERLAP = 1000, 100
BS = [1, 3, 6, 12]
THRS = [0.5, 0.3, 0.2]
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
PREM_MAX = 1800          # ≈450 token，稳妥落在 512 内
N_CALIB = 60             # 人工校准集大小

import importlib.util as _iu  # noqa: E402

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F


def chunks_of(t: str) -> list[str]:
    t = str(t)
    return [t[i:i + CHUNK] for i in range(0, len(t), CHUNK - OVERLAP)] or [""]


def claim_of(facet: str) -> str:
    """论断 = "该论文<facet>"（去掉括号里的注解，避免干扰 NLI）。"""
    return "该论文" + F[facet][1].split("（")[0].strip()


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))

    # ── 1. 编码 + 取每篇 top-b 块（dense 中文+英文词 = 前面实测最优打分）──
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    jobs = []            # (cluster, facet, docid, in_gold, b, [chunk_idx...])
    for ci in range(len(meta)):
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
        docs = sub["docid"].tolist()
        chunks, owner = [], []
        for i, t in enumerate(sub["full_paper"].tolist()):
            cs = chunks_of(t)
            chunks += cs
            owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        idx = {d: np.where(owner == d)[0] for d in docs}
        low = [c.lower() for c in chunks]
        for facet in meta[ci]["usable"]:
            pat, zh, anchor, _ = F[facet]
            rx = re.compile(pat, re.I)
            hit = np.array([bool(rx.search(c)) for c in low])
            gold = {d for d in docs if hit[idx[d]].any()}
            if not gold or len(gold) == len(docs):
                continue
            s = C @ enc.encode([f"{zh} {anchor}"], normalize_embeddings=True,
                               convert_to_numpy=True)[0].astype(np.float32)
            pm = {d: float(s[idx[d]].max()) for d in docs}
            # claim 集：按篇级分取 top-|gold|（**隔离判定误差，只看检索/证据**）
            claims = sorted(docs, key=lambda d: -pm[d])[:len(gold)]
            for d in claims:
                order = idx[d][np.argsort(-s[idx[d]])]
                for b in BS:
                    take = order[:b]
                    jobs.append(dict(cluster=ci + 1, facet=facet, docid=d,
                                     in_gold=bool(d in gold), b=b,
                                     chunk_ids=take.tolist(),
                                     premise="\n".join(chunks[j] for j in take)[:PREM_MAX],
                                     per_chunk=[chunks[j][:PREM_MAX] for j in take],
                                     proxy_hit=bool(hit[take].any()),
                                     proxy_frac=float(hit[take].mean())))
        print(f"簇{ci + 1} 完成", flush=True)
    del enc
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"\nclaim×b 组合 {len(jobs)} 条（claim≈{len(jobs) / len(BS):.0f}）", flush=True)

    # ── 2. NLI 批量判定 ──
    tk = AutoTokenizer.from_pretrained(NLI_MODEL)
    md = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=torch.float16)
    md.eval().to(dev)
    lab = [md.config.id2label[i].lower() for i in range(md.config.num_labels)]
    i_ent = next(i for i, l in enumerate(lab) if "entail" in l)
    i_con = next((i for i, l in enumerate(lab) if "contra" in l), None)
    print(f"NLI labels={lab}", flush=True)

    def infer(pairs: list[tuple[str, str]]) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), 32):
            prem = [p for p, _ in pairs[i:i + 32]]
            hyp = [h for _, h in pairs[i:i + 32]]
            b = tk(prem, hyp, return_tensors="pt", padding=True, truncation=True,
                   max_length=512).to(dev)
            with torch.no_grad():
                pr = torch.softmax(md(**b).logits.float(), dim=-1).cpu().numpy()
            out.append(pr)
        return np.concatenate(out, 0)

    # 2a. cite_recall：整篇论断 × 引用块拼接
    pairs = [(j["premise"], claim_of(j["facet"])) for j in jobs]
    pr = infer(pairs)
    for j, p in zip(jobs, pr):
        j["ent_concat"] = float(p[i_ent])
        j["con_concat"] = float(p[i_con]) if i_con is not None else 0.0
    print("cite_recall 判定完成", flush=True)

    # 2b. cite_precision：逐引用块
    flat = []
    for j in jobs:
        j["_ptr"] = len(flat)
        flat += [(c, claim_of(j["facet"])) for c in j["per_chunk"]]
    pr2 = infer(flat)
    print("cite_precision 判定完成", flush=True)

    rows, calib = [], []
    for j in jobs:
        seg = pr2[j["_ptr"]:j["_ptr"] + len(j["per_chunk"])]
        ent = seg[:, i_ent]
        con = seg[:, i_con] if i_con is not None else np.zeros(len(seg))
        for thr in THRS:
            rows.append(dict(cluster=j["cluster"], facet=j["facet"], docid=j["docid"],
                             in_gold=j["in_gold"], b=j["b"], thr=thr,
                             cite_recall_nli=float(j["ent_concat"] >= thr),
                             cite_precision_nli=float((ent >= thr).mean()),
                             contra_nli=float((con >= 0.5).mean()),
                             cite_recall_proxy=float(j["proxy_hit"]),
                             cite_precision_proxy=float(j["proxy_frac"]),
                             ent_concat=j["ent_concat"], con_concat=j["con_concat"]))
        calib.append(dict(facet=j["facet"], docid=j["docid"], b=j["b"], 在真值内=j["in_gold"],
                          论断=claim_of(j["facet"]), 引用块数=j["b"],
                          证据=" ⏎ ".join(j["per_chunk"])[:1200],
                          构造标签="支持" if j["proxy_hit"] else "不支持",
                          词面命中=j["proxy_hit"],
                          NLI_entail=round(j["ent_concat"], 3),
                          NLI_contra=round(j["con_concat"], 3),
                          人工判定="", 备注=""))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "results" / "R2_L3_NLI.csv", index=False, encoding="utf-8-sig")

    # 人工校准集：分层抽样（NLI 高低 × 构造标签 四象限 + 若干 contra）
    c = pd.DataFrame(calib).drop_duplicates(["facet", "docid", "b"])
    lo, hi = c["NLI_entail"].quantile(0.35), c["NLI_entail"].quantile(0.7)
    strata = [
        c[(c["构造标签"] == "支持") & (c["NLI_entail"] >= hi)],
        c[(c["构造标签"] == "支持") & (c["NLI_entail"] < lo)],
        c[(c["构造标签"] == "不支持") & (c["NLI_entail"] >= hi)],
        c[(c["构造标签"] == "不支持") & (c["NLI_entail"] < lo)],
        c[c["NLI_contra"] >= 0.5],
    ]
    names = ["高entail·构造支持", "低entail·构造支持", "高entail·构造不支持",
             "低entail·构造不支持", "有contradiction"]
    per = max(1, N_CALIB // len(strata))
    parts = [s.head(per).assign(分层=nm) for s, nm in zip(strata, names)]
    pick = pd.concat(parts).drop_duplicates(["facet", "docid", "b"])
    pick.to_csv(HERE / "data" / "r2dev" / "nli_calibration.csv", index=False, encoding="utf-8-sig")

    # ── 3. 打印 ──
    print(f"\n{'=' * 104}\n【NLI vs 词面代理】（{d[d['thr'] == 0.5].groupby(['cluster', 'facet']).ngroups} "
          f"个真值；claim = 按篇级分取 top-|gold|，隔离判定误差）")
    print(f"  {'每篇报b块':>9}{'指标':<20}{'**词面代理**':>14}{'NLI@0.5':>10}{'NLI@0.3':>10}{'NLI@0.2':>10}")
    for b in BS:
        t = d[d["b"] == b]
        for nm, pc, nc in (("cite_recall（论断有支持）", "cite_recall_proxy", "cite_recall_nli"),
                           ("cite_precision（引用块支持率）", "cite_precision_proxy", "cite_precision_nli")):
            g = t.groupby("thr")[nc].mean()
            print(f"  {b:>9}{nm:<20}{t[pc].mean():>14.3f}"
                  f"{g.get(0.5, float('nan')):>10.3f}{g.get(0.3, float('nan')):>10.3f}"
                  f"{g.get(0.2, float('nan')):>10.3f}")
    print(f"\n【同源偏袒幅度】b=1：cite_recall 词面 {d[(d['b'] == 1)]['cite_recall_proxy'].mean():.3f}"
          f" → NLI@0.5 {d[(d['b'] == 1) & (d['thr'] == 0.5)]['cite_recall_nli'].mean():.3f}"
          f"（差 {d[(d['b'] == 1)]['cite_recall_proxy'].mean() - d[(d['b'] == 1) & (d['thr'] == 0.5)]['cite_recall_nli'].mean():+.3f}）")

    print("\n【新能力：contradiction（抓「说反了」）】")
    for b in BS:
        t = d[(d["b"] == b) & (d["thr"] == 0.5)]
        print(f"  b={b:<3} 至少一块被判矛盾的比例 {t['contra_nli'].mean():.3f}"
              f" ｜ 论断整体被判矛盾 {t['con_concat'].mean():.3f}")

    print(f"\n【按是否在真值内】（b=3, thr=0.3）")
    t = d[(d["b"] == 3) & (d["thr"] == 0.3)]
    print(f"  {'':<8}{'cite_recall':>12}{'cite_precision':>15}{'contra':>9}")
    for g, sub in t.groupby("in_gold"):
        print(f"  {'在真值' if g else '不在真值':<8}{sub['cite_recall_nli'].mean():>12.3f}"
              f"{sub['cite_precision_nli'].mean():>15.3f}{sub['contra_nli'].mean():>9.3f}")
    print(f"\n  → 已写 R2_L3_NLI.csv；人工校准集 {len(pick)} 条 → data/r2dev/nli_calibration.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
