"""**重做校准（修正上一轮的三个方法学错误）**

## 上一轮错在哪（必须记录）
1. **截断不一致**：LLM 看 CSV 的 1200 字，NLI 看 1800 字 → **两者看的不是同一段文本**，比较无效。
2. **锚点被截掉**：48 行里 **32 行"构造=支持"的证据里根本搜不到锚点词**（多块拼接后被截断），
   于是 LLM 判 neutral 是**对的**，而"支持"标签是**错的** → 双方都在"对空气打分"。
3. **抽样按 NLI 分数分层**（高/低 entail 各占一半）→ 人为放大分歧，**不能用来量 NLI 精度**。

## 本轮设计
- **证据 = 单块**（b=1，取篇内分最高的一块，1000 字，**不截断**）→ LLM 与 NLI 看到**完全相同的文本**。
- **抽样按类随机**（`构造=支持`（块内含锚点）30 条 + `构造=不支持` 30 条），**不按 NLI 分数分层**。
- 判定器：NLI（本地 279M，中文论断 × 英文证据） + LLM（独立裁判模型 `PAPERPILOT_JUDGE`）。
- 产出：`data/r2dev/calib2.csv`（含原文、双判定、理由）→ 定阈值 / 量锚点精度 / 查 contradiction。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_calib2.py
"""
from __future__ import annotations

import importlib.util as _iu
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
CACHE = HERE / "data" / "r2dev" / "clusters"
OUT = HERE / "data" / "r2dev" / "calib2.csv"
CHUNK, OVERLAP = 1000, 100
N_PER_CLASS = 30
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F

spec2 = _iu.spec_from_file_location("cl", HERE / "tmp" / "_r2_calib_llm.py")
_cl = _iu.module_from_spec(spec2)
sys.modules["cl"] = _cl
spec2.loader.exec_module(_cl)


def chunks_of(t: str) -> list[str]:
    t = str(t)
    return [t[i:i + CHUNK] for i in range(0, len(t), CHUNK - OVERLAP)] or [""]


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))

    # ── 1. 建池：每 (簇,facet) 的 claim 篇 → 取篇内 top-1 块做证据 ──
    pos, neg = [], []
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    for ci in range(len(meta)):
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
        docs = sub["docid"].tolist()
        chunks, owner = [], []
        for i, t in enumerate(sub["full_paper"].astype(str).tolist()):
            cs = chunks_of(t)
            chunks += cs
            owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        idx = {d: np.where(owner == d)[0] for d in docs}
        for facet in meta[ci]["usable"]:
            pat, zh, anchor, _ = F[facet]
            rx = re.compile(pat, re.I)
            hit = np.array([bool(rx.search(c)) for c in chunks])
            gold = {d for d in docs if hit[idx[d]].any()}
            if not gold or len(gold) == len(docs):
                continue
            s = C @ enc.encode([f"{zh} {anchor}"], normalize_embeddings=True,
                               convert_to_numpy=True)[0].astype(np.float32)
            pm = {d: float(s[idx[d]].max()) for d in docs}
            claims = sorted(docs, key=lambda d: -pm[d])[:len(gold)]
            for d in claims:
                j = idx[d][int(np.argmax(s[idx[d]]))]        # 篇内 top-1 块
                rec = dict(cluster=ci + 1, facet=facet, docid=d,
                           在真值=bool(d in gold), 论断="该论文" + zh.split("（")[0].strip(),
                           证据=chunks[j], 块长=len(chunks[j]),
                           构造标签="支持" if hit[j] else "不支持")
                (pos if hit[j] else neg).append(rec)
    del enc
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"池：块内含锚点 {len(pos)} 条 ｜ 不含 {len(neg)} 条（b=1，证据不截断）", flush=True)

    rng = np.random.default_rng(0)
    sel = [pos[i] for i in rng.choice(len(pos), min(N_PER_CLASS, len(pos)), replace=False)]
    sel += [neg[i] for i in rng.choice(len(neg), min(N_PER_CLASS, len(neg)), replace=False)]
    # 校验：构造=支持 的行，锚点必须真的在证据里
    bad = sum(1 for r in sel if r["构造标签"] == "支持"
              and not re.search(F[r["facet"]][0], r["证据"], re.I))
    print(f"抽样 {len(sel)} 条 ｜ 校验失败（支持但证据无锚点）{bad} 条", flush=True)

    # ── 2. NLI 判定（前提=整块，不截断；1000 字符 ≈ 250 token，远小于 512）──
    tk = AutoTokenizer.from_pretrained(NLI_MODEL)
    md = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=torch.float16)
    md.eval().to(dev)
    lab = [md.config.id2label[i].lower() for i in range(md.config.num_labels)]
    pr = []
    for i in range(0, len(sel), 32):
        b = tk([r["证据"] for r in sel[i:i + 32]], [r["论断"] for r in sel[i:i + 32]],
               return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            pr.append(torch.softmax(md(**b).logits.float(), -1).cpu().numpy())
    pr = np.concatenate(pr, 0)
    i_ent = next(i for i, l in enumerate(lab) if "entail" in l)
    i_con = next(i for i, l in enumerate(lab) if "contra" in l)
    for r, p in zip(sel, pr):
        r["NLI_entail"] = round(float(p[i_ent]), 3)
        r["NLI_contra"] = round(float(p[i_con]), 3)
        r["NLI_判定"] = lab[int(p.argmax())]
    del md
    if dev == "cuda":
        torch.cuda.empty_cache()
    print("NLI 判定完成", flush=True)

    # ── 3. LLM 判定（同一段文本）──
    prefix = "PAPERPILOT_JUDGE" if _cl.llm.judge_configured() else "PAPERPILOT_LLM"
    base, key, model = _cl.llm.config(prefix)
    print(f"LLM 判定器：{prefix} → {model}", flush=True)
    for i, r in enumerate(sel):
        r.update(_cl.label_one(r["论断"], r["证据"], prefix))
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(sel)} …", flush=True)

    df = pd.DataFrame(sel)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"完成 → {OUT}")

    ok = df[df["LLM判定"] != "ERROR"]
    print(f"\n【LLM 判定分布】（{len(ok)}/{len(df)} 有效）  " +
          " ｜ ".join(f"{k} {v}" for k, v in ok["LLM判定"].value_counts().items()))
    print("【NLI 判定分布】  " + " ｜ ".join(f"{k} {v}" for k, v in ok["NLI_判定"].value_counts().items()))

    y = (ok["LLM判定"] == "entail").astype(int).values
    print(f"\n【① NLI 阈值校准】正类 = LLM 判 entail（{int(y.sum())}/{len(y)}）")
    print(f"  {'阈值':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>8}{'F1':>7}")
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pred = (ok["NLI_entail"] >= thr).astype(int).values
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        print(f"  {thr:>6.1f}{tp:>5}{fp:>5}{fn:>5}{p:>11.3f}{r_:>8.3f}{f1:>7.3f}")

    print("\n【② 锚点精度：把「块内含锚点」当支持，LLM 同意多少】")
    print(pd.crosstab(ok["构造标签"], ok["LLM判定"]).to_string())
    sup = ok[ok["构造标签"] == "支持"]
    print(f"  → 构造「支持」{len(sup)} 条中 LLM 判 entail：**{(sup['LLM判定'] == 'entail').mean():.3f}**")
    unn = ok[ok["构造标签"] == "不支持"]
    print(f"  → 构造「不支持」{len(unn)} 条中 LLM 判 entail（假阳）：{(unn['LLM判定'] == 'entail').mean():.3f}")

    print("\n【③ contradiction（NLI 报矛盾）是否成立】")
    con = ok[ok["NLI_contra"] >= 0.5]
    print(f"  NLI 报矛盾 {len(con)} 条 → LLM 分布：" +
          (" ｜ ".join(f"{k} {v}" for k, v in con["LLM判定"].value_counts().items()) or "无"))
    if len(con):
        print(f"  → LLM 认同「说反了」：{(con['LLM判定'] == 'contradict').mean():.3f}")
    print("\n【分歧样本（含完整证据，供人工抽查）】")
    dis = ok[((ok["NLI_entail"] >= 0.7) & (ok["LLM判定"] != "entail"))
             | ((ok["NLI_entail"] < 0.3) & (ok["LLM判定"] == "entail"))
             | ((ok["NLI_contra"] >= 0.5) & (ok["LLM判定"] == "entail"))]
    print(f"  {len(dis)} 条")
    for _, r in dis.head(8).iterrows():
        print("=" * 92)
        print(f"[{r['构造标签']}] {r['论断']} ｜ NLI ent={r['NLI_entail']} con={r['NLI_contra']}"
              f" ｜ LLM={r['LLM判定']}（{str(r['LLM理由'])[:50]}）")
        print(f"  证据：{str(r['证据'])[:300]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
