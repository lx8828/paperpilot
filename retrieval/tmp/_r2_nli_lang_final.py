"""**控制实验**：把论断换成英文（只改语言，语义不变），NLI 会不会变好？

## 问题
两个独立 LLM 一致率 90%、而 NLI 与它们只有 ~40% → 初步指向"NLI 有问题"。
但还需排除一种可能：**是我们喂给 NLI 的方式不对**（中文论断 × 英文证据 = 跨语言）。
控制实验：在同一批 60 条上，把论断换成**人工英文改写句**（`F[facet][3]`，不含锚点原词），
其余全不变 → 若 NLI 明显变好，则问题是"跨语言用法"；若仍然差，则是 **NLI 能力不足**。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_nli_lang_final.py
"""
from __future__ import annotations

import importlib.util as _iu
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

HERE = Path(__file__).resolve().parents[1]
CSV = HERE / "data" / "r2dev" / "calib2_3judge.csv"

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F

MODELS = {
    "xnli2m(多语言)": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "fever(英文)": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
}


def f1_at(score: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float]:
    p = (score >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def mai2(name: str) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(CSV)
    df["claim_en"] = df["facet"].map(lambda f: F[f][3])
    tk = AutoTokenizer.from_pretrained(name)
    md = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16)
    md.eval().to(dev)
    lab = [md.config.id2label[i].lower() for i in range(md.config.num_labels)]
    ie = next(i for i, l in enumerate(lab) if "entail" in l)

    out = {}
    for tag, hyp_col in (("中文论断", "论断"), ("英文改写论断", "claim_en")):
        pr = []
        for i in range(0, len(df), 32):
            b = tk(df["证据"].tolist()[i:i + 32], df[hyp_col].tolist()[i:i + 32],
                   return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                pr += torch.softmax(md(**b).logits.float(), -1)[:, ie].cpu().tolist()
        out[tag] = np.array(pr)
        print(f"\n  【{name}】{tag} 示例：{df[hyp_col].iloc[0][:80]}")
        if tag == "英文改写论断":
            print(f"    ← 中文原文：{df['论断'].iloc[0]}")
    del md
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    results = {}
    for tag, name in MODELS.items():
        print("=" * 100)
        print(f"【{tag}】{name}")
        r = mai2(name)
        results[tag] = r

    df = pd.read_csv(CSV)
    for ref in ("DS判定", "LLM判定"):
        y = (df[ref] == "entail").astype(int).values
        print(f"\n{'=' * 100}\n【对照参考 = {ref}】正类 {int(y.sum())}/{len(y)}")
        print(f"  {'模型':<16}{'论断语言':<14}{'阈值':>6}{'P':>8}{'R':>8}{'F1':>8}")
        for mtag, r in results.items():
            for ltag, score in r.items():
                best = max(((thr,) + f1_at(score, y, thr) for thr in
                            (0.3, 0.5, 0.7, 0.8, 0.9)), key=lambda t: t[3])
                print(f"  {mtag:<16}{ltag:<14}{best[0]:>6.1f}{best[1]:>8.3f}{best[2]:>8.3f}{best[3]:>8.3f}")

    print("\n【结论判据】")
    for mtag, r in results.items():
        y = (df["DS判定"] == "entail").astype(int).values
        bz = max(f1_at(r["中文论断"], y, t)[2] for t in (0.3, 0.5, 0.7, 0.8, 0.9))
        be = max(f1_at(r["英文改写论断"], y, t)[2] for t in (0.3, 0.5, 0.7, 0.8, 0.9))
        print(f"  {mtag:<16} 中文论断 F1 {bz:.3f} → 英文论断 F1 {be:.3f}（Δ {be - bz:+.3f}）")
    print("  → 若 Δ 很小（<0.05）→ **不是跨语言的问题，是 NLI 能力不足**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
