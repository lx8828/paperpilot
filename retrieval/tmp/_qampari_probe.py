"""下载并解剖 `mteb/rag-qampari-32k`：语料规模 / 每题 gold 数 / 字段 / 评测指标。

这决定我们能否把它当作"标准考场"：
- corpus 段落数 & 总 token → **S/C 比值**（检索是否物理必需）
- 每题 gold 段落数 → 是不是"多答案"（覆盖性问题）
- qrels 形态 → 用什么指标（nDCG@k / Recall@k / MRecall@k）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
# ⚠️ `ensure_hf_home()` 会打开离线开关（避免加载模型时联网）。
#    但**下载数据集**必须联网 → 显式覆盖。
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

OUT = Path(__file__).resolve().parents[1] / "data" / "qampari"
REPO = "mteb/rag-qampari-32k"


def main() -> int:
    from huggingface_hub import hf_hub_download

    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in ("corpus/corpus-00000-of-00001.parquet",
              "queries/queries-00000-of-00001.parquet",
              "data/test-00000-of-00001.parquet",
              "README.md"):
        try:
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(OUT))
            files[f] = Path(p)
            print(f"OK  {f}  {Path(p).stat().st_size / 1e6:.1f} MB")
        except Exception as e:  # noqa: BLE001
            print(f"ERR {f}: {type(e).__name__} {str(e)[:120]}")

    print("\n" + "=" * 100)
    for f, p in files.items():
        if p.suffix != ".parquet":
            continue
        d = pd.read_parquet(p)
        print(f"\n【{f}】行数 {len(d):,} ｜ 列 {list(d.columns)}")
        print(d.head(2).to_string()[:900])
        # 关键统计
        for c in d.columns:
            if d[c].dtype == object and len(str(d[c].iloc[0])) > 200:
                n = d[c].astype(str).str.len()
                print(f"  {c}: 字符数 中位 {int(n.median()):,} / 合计 {int(n.sum()):,}")
        if "text" in d.columns:
            pass

    # qrels 形态
    q = files.get("data/test-00000-of-00001.parquet")
    if q:
        d = pd.read_parquet(q)
        print(f"\n【qrels】列 {list(d.columns)}")
        for c in ("query-id", "corpus-id", "score"):
            if c in d.columns:
                print(f"  {c}: nunique {d[c].nunique()} ｜ 取值样例 {list(d[c].head(3))}")
        if {"query-id", "corpus-id"} <= set(d.columns):
            g = d.groupby("query-id").size()
            print(f"  → 每题 gold 段落数：均值 {g.mean():.2f} ｜ 中位 {g.median():.0f} ｜ "
                  f"max {g.max()} ｜ 只有 1 个 gold 的题占比 {(g == 1).mean():.1%}")

    rm = files.get("README.md")
    if rm:
        txt = rm.read_text(encoding="utf-8", errors="replace")
        print("\n【README 关键行】")
        for line in txt.splitlines():
            low = line.lower()
            if any(k in low for k in ("ndcg", "recall", "metric", "task", "corpus",
                                      "qampari", "loft", "token", "evaluat")):
                s = line.strip()
                if s and len(s) < 260:
                    print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
