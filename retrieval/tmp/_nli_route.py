"""**论断语言路线实测**：中文论断跨语言 NLI ≠ 英文论断，中间能不能靠翻译补？

## 问题
跨语言 NLI 明显弱于同语言（冒烟测试：中文论断 × 英文证据 多处判成 neutral）。
用户直觉：**加一层翻译，误差应该比跨语言小**。这里把三条路线放在同一批证据上比。

## 设计（控制了变量，所以 500 多对就够下结论）
- **证据**：取自真实论文句。每篇 gold 论文里：
  - `正例` = **含锚点的那一句**（+前一句）→ 真实"支持证据"
  - `负例` = 同一篇里**不含锚点**的块首句 → "拿错了证据"（同一篇，只是没在说这件事）
  → 正/负来自**同一篇论文**，排除"论文主题差异"这个混淆因素。
- **论断路线**（同一批证据，只换论断语言）：
  1. `zh`：中文题面（产品现在会写的）
  2. `zh_mt_en`：机翻成英文（**论断侧加一层翻译**）
  3. `en_para`：**人工英文改写句**（不含锚点原词）→ 英文论断路线，**零翻译误差** = 上界
- **证据路线**：`en` 原文 / `zh_mt`（机翻成中文，**证据侧加一层翻译**）
- **判据**：`entail` 概率把正例和负例分开的 **AUC**（0.5=瞎猜；越高=判定器越有用）

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_nli_route.py
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
MAX_PER_PAPER = 3          # 每篇最多取 3 正 / 3 负
MT_ZH_EN = "Helsinki-NLP/opus-mt-zh-en"
MT_EN_ZH = "Helsinki-NLP/opus-mt-en-zh"

# facet → (锚点, 中文题面, 英文锚点词, 人工英文改写句)
import importlib.util as _iu  # noqa: E402

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F

JUDGES = {
    "xnli2m": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "fever": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "nli_ce": "cross-encoder/nli-deberta-v3-base",
}


def sentences(t: str) -> list[str]:
    t = re.sub(r"\s+", " ", str(t)).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 25]


def chunks_of(t: str) -> list[str]:
    t = str(t)
    return [t[i:i + CHUNK] for i in range(0, len(t), CHUNK - OVERLAP)] or [""]


def evidence_span(chunk: str, pat: str) -> str | None:
    """含锚点的那一句 + 前一句（真实"支持证据"）。"""
    sents = sentences(chunk)
    rx = re.compile(pat, re.I)
    for i, s in enumerate(sents):
        if rx.search(s):
            return (sents[i - 1] + " " + s) if i > 0 else s
    return None


def first_sentences(chunk: str, n: int = 2) -> str:
    return " ".join(sentences(chunk)[:n])[:600]


def main() -> int:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))

    # ── 1. 构造正/负证据对（每条真值 facet 一组） ──
    pairs = []          # (cluster, facet, docid, label, premise_en)
    for ci in range(len(meta)):
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
        docs = sub["docid"].tolist()
        texts = sub["full_paper"].astype(str).tolist()
        for facet in meta[ci]["usable"]:
            pat = F[facet][0]
            rx = re.compile(pat, re.I)
            for d, t in zip(docs, texts):
                cs = chunks_of(t)
                if not any(rx.search(c) for c in cs):
                    continue                                  # 非 gold 篇跳过
                pos, neg = [], []
                for c in cs:
                    span = evidence_span(c, pat)
                    if span:
                        pos.append(span)
                    else:
                        s = first_sentences(c)
                        if s:
                            neg.append(s)
                for sp in pos[:MAX_PER_PAPER]:
                    pairs.append(dict(cluster=ci + 1, facet=facet, docid=d, label=1, premise=sp))
                for sp in neg[:MAX_PER_PAPER]:
                    pairs.append(dict(cluster=ci + 1, facet=facet, docid=d, label=0, premise=sp))
    df = pd.DataFrame(pairs).drop_duplicates(["facet", "docid", "premise"])
    print(f"证据对 {len(df)} 条（正 {int(df['label'].sum())} / 负 {int((1 - df['label']).sum())}）"
          f"｜覆盖 {df['facet'].nunique()} 个 facet", flush=True)

    # ── 2. 论断三路线 + 证据机翻 ──
    from transformers import MarianMTModel

    def load_mt(name):
        # ⚠️ opus-mt 的 tokenizer_config.json 没有 tokenizer_class → AutoTokenizer 会失败，须显式指定
        tk = MarianTokenizer.from_pretrained(name)
        md = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16).eval().to(dev)
        return tk, md

    def translate(tk, md, texts, batch=32) -> list[str]:
        out = []
        for i in range(0, len(texts), batch):
            b = tk([t[:480] for t in texts[i:i + batch]], return_tensors="pt",
                   padding=True, truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                g = md.generate(**b, max_new_tokens=160, num_beams=1)
            out += tk.batch_decode(g, skip_special_tokens=True)
        return out

    # 翻译结果缓存（翻一次很慢，避免重复跑）
    CACHE_TR = HERE / "data" / "r2dev" / "mt_cache.json"
    trc = json.loads(CACHE_TR.read_text(encoding="utf-8")) if CACHE_TR.exists() else {}

    tk_zh_en, md_zh_en = load_mt(MT_ZH_EN)
    claims_en_mt = {}
    for facet in sorted(df["facet"].unique()):
        key = f"claim::{facet}"
        if key not in trc:
            trc[key] = translate(tk_zh_en, md_zh_en, [F[facet][1]])[0]
        claims_en_mt[facet] = trc[key]
    del tk_zh_en, md_zh_en
    if dev == "cuda":
        torch.cuda.empty_cache()
    print("\n【机翻论断抽样】（对照人工英文改写句）")
    for f_, v in list(claims_en_mt.items())[:6]:
        print(f"  中文：{F[f_][1]}")
        print(f"  机翻：{v}")
        print(f"  人工：{F[f_][3]}\n")

    tk_en_zh, md_en_zh = load_mt(MT_EN_ZH)
    todo = [i for i, p in enumerate(df["premise"].tolist()) if f"prem::{i}" not in trc]
    if todo:
        outs = translate(tk_en_zh, md_en_zh, [df["premise"].tolist()[i] for i in todo])
        for i, o in zip(todo, outs):
            trc[f"prem::{i}"] = o
    df["premise_zh_mt"] = [trc[f"prem::{i}"] for i in range(len(df))]
    CACHE_TR.write_text(json.dumps(trc, ensure_ascii=False), encoding="utf-8")
    del tk_en_zh, md_en_zh
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ── 3. 三路线 × judge → 用 entail 概率分开正负（AUC） ──
    from transformers import AutoModelForSequenceClassification

    claim_of = {
        "zh": lambda f_: F[f_][1],
        "zh_mt_en": lambda f_: claims_en_mt[f_],
        "en_para": lambda f_: F[f_][3],
    }
    rows = []
    for jname, mid in JUDGES.items():
        tk = AutoTokenizer.from_pretrained(mid)
        md = AutoModelForSequenceClassification.from_pretrained(mid, torch_dtype=torch.float16)
        md.eval().to(dev)
        lab = [md.config.id2label[i].lower() for i in range(md.config.num_labels)]
        ent = next(i for i, l in enumerate(lab) if "entail" in l)
        probs = {}
        for cname, fn in claim_of.items():
            hyps = [fn(f_) for f_ in df["facet"].tolist()]
            pr = []
            for i in range(0, len(df), 64):
                b = tk(df["premise"].tolist()[i:i + 64], hyps[i:i + 64], return_tensors="pt",
                       padding=True, truncation=True, max_length=512).to(dev)
                with torch.no_grad():
                    p = torch.softmax(md(**b).logits.float(), dim=-1)[:, ent].cpu().tolist()
                pr += p
            probs[(cname, "en")] = pr
        # 证据机翻成中文 → 再走 xnli（跨语言模型本身也支持中文）
        hyps = [claim_of["zh"](f_) for f_ in df["facet"].tolist()]
        pr = []
        for i in range(0, len(df), 64):
            b = tk(df["premise_zh_mt"].tolist()[i:i + 64], hyps[i:i + 64], return_tensors="pt",
                   padding=True, truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                p = torch.softmax(md(**b).logits.float(), dim=-1)[:, ent].cpu().tolist()
            pr += p
        probs[("zh", "zh_mt")] = pr

        y = df["label"].values
        for (cname, ename), p in probs.items():
            p = np.array(p)
            pos, neg = p[y == 1], p[y == 0]
            a = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
            rows.append(dict(judge=jname, claim=cname, evidence=ename, auc=a,
                             mean_pos=float(pos.mean()), mean_neg=float(neg.mean()),
                             acc=float(((p >= 0.5).astype(int) == y).mean()), n=len(p)))
        del md
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"  完成 judge {jname}", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(HERE / "results" / "R2_NLI_ROUTE.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 100}\n【AUC：entail 概率分开「支持的证据」与「拿错的证据」】")
    print(f"  {'judge':<8}{'论断':<10}{'证据':<8}{'AUC':>7}{'正例entail':>11}{'负例entail':>11}{'acc@0.5':>9}")
    for j in JUDGES:
        t = r[r["judge"] == j].sort_values("auc", ascending=False)
        for _, x in t.iterrows():
            star = "  ←" if x["auc"] == r[r["judge"] == j]["auc"].max() else ""
            print(f"  {j:<8}{x['claim']:<10}{x['evidence']:<8}{x['auc']:>7.3f}"
                  f"{x['mean_pos']:>11.3f}{x['mean_neg']:>11.3f}{x['acc']:>9.3f}{star}")
    print("\n【结论性对比｜同一 judge 下换论断语言】")
    for j in JUDGES:
        t = r[(r["judge"] == j) & (r["evidence"] == "en")].set_index("claim")["auc"]
        if {"zh", "zh_mt_en", "en_para"} <= set(t.index):
            print(f"  {j:<8} 中文论断 {t['zh']:.3f} ｜ 机翻英文 {t['zh_mt_en']:.3f} ｜ "
                  f"人工英文 {t['en_para']:.3f}")
    print("\n【证据侧翻译（中文论断 × 中文机翻证据）vs 原文】")
    for j in JUDGES:
        a_zh = r[(r["judge"] == j) & (r["claim"] == "zh")]
        o = a_zh[a_zh["evidence"] == "en"]["auc"]
        m = a_zh[a_zh["evidence"] == "zh_mt"]["auc"]
        if len(o) and len(m):
            print(f"  {j:<8} 英文原文证据 {o.iloc[0]:.3f} ｜ 机翻中文证据 {m.iloc[0]:.3f}")
    print(f"\n  → 已写 {HERE / 'results' / 'R2_NLI_ROUTE.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
