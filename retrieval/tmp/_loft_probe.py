"""解剖 `f20180301/loft-rag-qampari-32k` 镜像：字段、语料是否内附、规模。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

OUT = Path(__file__).resolve().parents[1] / "data" / "loft"


def main() -> int:
    from huggingface_hub import hf_hub_download

    OUT.mkdir(parents=True, exist_ok=True)
    for repo in ("f20180301/loft-rag-qampari-32k", "mteb/loft-qampari-32k"):
        print("=" * 100)
        print(f"【{repo}】")
        for f in ("data/dev-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"):
            try:
                p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(OUT / repo.split("/")[-1]))
            except Exception as e:  # noqa: BLE001
                print(f"  {f}: {type(e).__name__} {str(e)[:90]}")
                continue
            d = pd.read_parquet(p)
            print(f"  {f}: 行 {len(d):,} ｜ 大小 {Path(p).stat().st_size / 1e6:.1f} MB ｜ 列 {list(d.columns)}")
            for c in d.columns:
                v = str(d[c].iloc[0])
                n = d[c].astype(str).str.len()
                print(f"    {c:<18} 中位 {int(n.median()):>8,} 字 ｜ 合计 {int(n.sum()):>12,} 字 ｜ {v[:110]!r}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
