"""**新 RAG-2 · 集合型检索第一版实验**（LitSearch 全文 / 20 篇 / 零 LLM）。

## 任务
题面（中文）：「这 20 篇里，**哪几篇** <做了 X>？列出全部，并各给一条证据。」
真值：`facet_matrix.json`（词面锚点）→ 判据 = **篇级 P / R / F1**（自动，无 LLM judge）。

## 三条臂
| 臂 | 做法 | 它代表什么 |
|---|---|---|
| **A** | 全局 chunk top-k → 涉及哪些篇 | **RAG-1 式**：排序式检索当枚举用（会漏篇） |
| **B** | **逐篇预算**（每篇取 top-b 块，篇级分 = 篇内最高分）→ 跨篇校准 → 阈值判定 | **本次要验证的方案**：覆盖式检索 |
| **C** | 全局 top-k 块**直接映射**到篇（k 大到能覆盖全语料）| 上界参考（= 语料全扫但没篇内预算） |

## 为什么必须做阈值曲线（而不是只报一个 F1）
"哪些篇满足 X" 是**绝对判定**，不是相对排序 → 必须定阈值。这里同时报两种：
- **oracle 阈值**（按真值选最优阈值）= **上界**，说明"信号本身够不够用"
- **固定阈值**（z=0 / cos 阈值，不偷看真值）= **可部署**的水位
两者差距 = "**阈值可不可定**"（这是本任务的真难点）。

## 查询变体（跨语言是关键，见 RAG-2 的 XLING 资产）
- `zh`：只给中文题面（考验跨语言语义）
- `zh+en`：中文题面 + 英文锚点词（= 查询侧扩展）
- `en`：只用英文锚点词（对照：纯词面）

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
CHUNK = 1000
OVERLAP = 100

# facet → 英文锚点词（查询扩展用；**故意不直接用正则**，那是真值来源）
EN_TERMS = {
    "ablation": "ablation study component analysis removing parts of the method",
    "code_release": "our code and data are publicly available at github open source release",
    "human_eval": "human evaluation annotators human ratings",
    "multilingual": "multilingual cross-lingual experiments in many languages",
    "efficiency": "inference latency throughput time cost efficiency",
    "significance": "statistical significance multiple runs standard deviation",
    "new_dataset": "we introduce a new dataset benchmark corpus",
    "case_study": "case study examples",
    "error_analysis": "error analysis failure cases analysis",
    "llm_as_model": "GPT-4 ChatGPT as the model baseline",
}
DROP = {"llm_as_model"}          # 该 facet 在 20 篇里正例 0 → 分辨力为 0，弃用


def chunk_text(t: str) -> list[str]:
    t = str(t)
    out = []
    i = 0
    while i < len(t):
        out.append(t[i:i + CHUNK])
        i += CHUNK - OVERLAP
    return out


def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def best_f1(scores: dict[str, float], gold: set[str]) -> tuple[float, float]:
    """阈值扫描下的最优 F1 与对应阈值（oracle 上界）。"""
    if not scores:
        return 0.0, float("nan")
    vals = np.array(sorted(set(scores.values())))
    cand = np.concatenate([vals - 1e-6, [vals.max() + 1e-6]])
    best, bt = 0.0, float("nan")
    for t in cand:
        pred = {d for d, s in scores.items() if s >= t}
        f = prf(pred, gold)[2]
        if f > best:
            best, bt = f, float(t)
    return best, bt


def main() -> int:
    corpus = pd.read_parquet(DEV / "corpus20.parquet")
    gold_all = json.loads((DEV / "facet_matrix.json").read_text(encoding="utf-8"))
    docs = corpus["docid"].tolist()
    print("=" * 104)
    print(f"语料 {len(docs)} 篇 ｜ {int(corpus['n_chars'].sum()):,} 字符 ≈ "
          f"{int(corpus['n_chars'].sum() / 4 / 1000)}k token（>128k 窗口 → 直读不可行）")

    print("\n切块 + 编码（bge-m3）…", end=" ", flush=True)
    import torch
    from sentence_transformers import SentenceTransformer

    chunks, owner = [], []
    for i, t in enumerate(corpus["full_paper"].tolist()):
        cs = chunk_text(t)
        chunks += cs
        owner += [docs[i]] * len(cs)
    owner = np.array(owner)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                   show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    print(f"{len(chunks)} 块（设备 {dev}）", flush=True)

    rows = []
    for facet, spec in gold_all["facets"].items():
        if facet in DROP:
            continue
        gold = set(spec["pos"])
        en = EN_TERMS.get(facet, facet.replace("_", " "))
        variants = {"zh": spec["zh"], "zh+en": f"{spec['zh']} {en}", "en": en}
        # 逐篇块索引（臂 B 要按篇取预算）
        idx_by_doc = {d: np.where(owner == d)[0] for d in docs}
        for vname, q in variants.items():
            qv = enc.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            s = C @ qv                                    # 每块余弦
            # 臂 B：篇级分 = 篇内最高分；再 z-norm 校准（跨篇可比）
            per_doc_max = {d: float(s[idx_by_doc[d]].max()) for d in docs}
            v = np.array(list(per_doc_max.values()))
            z = {d: (per_doc_max[d] - v.mean()) / (v.std() + 1e-9) for d in docs}
            f_b, t_b = best_f1(z, gold)
            pred_fixed = {d for d in docs if z[d] >= 0.0}          # 固定阈值（不偷看真值）
            p_f, r_f, f_f = prf(pred_fixed, gold)
            # 臂 A：全局 chunk top-k → 涉及篇集合（RAG-1 式）
            a5 = set(owner[np.argsort(-s)[:5]].tolist())
            a20 = set(owner[np.argsort(-s)[:20]].tolist())
            rows.append(dict(facet=facet, variant=vname, n_gold=len(gold),
                             B_oracleF1=f_b, B_thr=round(t_b, 2),
                             B_fix_p=p_f, B_fix_r=r_f, B_fix_f1=f_f,
                             B_npred_fix=len(pred_fixed),
                             A5_f1=prf(a5, gold)[2], A20_f1=prf(a20, gold)[2],
                             A5_rec=prf(a5, gold)[1], A20_rec=prf(a20, gold)[1]))
        print(f"  ✓ {facet:<14}（gold {len(gold)}/20）")

    df = pd.DataFrame(rows)
    out = HERE / "results" / "R2_SETSEL_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 104}\n【臂 B：逐篇预算 + 跨篇校准】按 facet（三种查询变体）")
    print(f"  {'facet':<14}{'gold':>5}｜{'query':>6}{'oracleF1':>10}{'thr':>7}"
          f"{'固定阈F1':>10}{'P':>7}{'R':>7}{'判正数':>8}")
    for facet, sub in df.groupby("facet", sort=False):
        for _, r in sub.iterrows():
            print(f"  {facet:<14}{int(r['n_gold']):>5}｜{r['variant']:>6}{r['B_oracleF1']:>10.3f}"
                  f"{r['B_thr']:>7.1f}{r['B_fix_f1']:>10.3f}{r['B_fix_p']:>7.2f}"
                  f"{r['B_fix_r']:>7.2f}{int(r['B_npred_fix']):>8}")

    print(f"\n【臂 A：全局 chunk top-k（RAG-1 式）】按 facet（zh+en 变体）")
    print(f"  {'facet':<14}{'gold':>5}{'A@5 F1':>9}{'A@5 召':>9}{'A@20 F1':>9}{'A@20 召':>9}")
    for facet, sub in df[df["variant"] == "zh+en"].groupby("facet", sort=False):
        r = sub.iloc[0]
        print(f"  {facet:<14}{int(r['n_gold']):>5}{r['A5_f1']:>9.3f}{r['A5_rec']:>9.2f}"
              f"{r['A20_f1']:>9.3f}{r['A20_rec']:>9.2f}")

    print(f"\n【汇总：三变体均值】")
    for v, sub in df.groupby("variant"):
        print(f"  {v:<6} 臂B oracleF1 {sub['B_oracleF1'].mean():.3f} ｜ "
              f"臂B 固定阈F1 {sub['B_fix_f1'].mean():.3f}（P {sub['B_fix_p'].mean():.2f} "
              f"R {sub['B_fix_r'].mean():.2f}）｜ 臂A@5 F1 {sub['A5_f1'].mean():.3f} "
              f"（召 {sub['A5_rec'].mean():.2f}）｜ 臂A@20 F1 {sub['A20_f1'].mean():.3f}")
    print(f"\n  → 已写 {out}")
    print("""
读法：
  · **B_oracleF1 高、A@k 低** = "覆盖式检索"确实必要（RAG-1 式 top-k 漏篇）
  · **B_oracleF1 与 B_fix_f1 差距** = 阈值可不可定（本任务的真正难点）
  · **zh vs zh+en vs en 差异** = 跨语言语义够不够用（RAG-2 的 XLING 价值）
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
