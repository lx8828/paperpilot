"""看第三判官与新真值的**分歧**（含证据全文）→ 人工判读用。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_dispute_view.py --skip 0 --n 9
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import sys
import textwrap
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F

DEV = HERE / "data" / "r2dev"
adj = pd.read_csv(DEV / "gold_adjudicate.csv")
EVID = json.loads((DEV / "gold_recalib2_evidence.json").read_text(encoding="utf-8"))
adj["third_yes"] = adj["third_label"].astype(str).str.lower().eq("entail")
adj["err_s"] = adj["third_err"].where(adj["third_err"].notna(), "").astype(str)
dis = adj[(adj.err_s == "") & (adj.third_yes != adj.new_gold)].copy()
dis["dir"] = dis.apply(lambda r: ("误删？" if r.new_gold is False else "误收？"), axis=1)
order = {"removed": 0, "agree_neg": 1, "added": 2, "agree_pos": 3}
dis["o"] = dis.kind.map(order).fillna(9)
dis = dis.sort_values(["o", "cluster", "facet"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--evchars", type=int, default=430)
    args = ap.parse_args()

    print(f"分歧共 {len(dis)} 条 ｜ 展示 {args.skip}~{args.skip + args.n}")
    for _, r in dis.iloc[args.skip: args.skip + args.n].iterrows():
        e = EVID.get(str(r.key)) or {}
        print("\n" + "=" * 112)
        print(f"【{r.dir}】{r.kind} ｜ 簇{int(r.cluster)} · {r.facet} · {r.docid}"
              f" ｜ 锚点={'T' if r.anchor_gold else '-'} 新真值={'T' if r.new_gold else '-'}"
              f" 第三判官={'T' if r.third_yes else '-'}（{r.third_label}, conf={r.third_conf}）")
        print(f"  论断：{F[str(r.facet)][1]}")
        print(f"  第三判官理由：{r.third_reason}")
        print(f"  第三判官引用：{str(r.third_quote)[:200]}")
        ev = " ".join(str(e.get("evidence", "")).split())
        print("  ── 证据 ──")
        print(textwrap.fill(ev[: args.evchars], 110, initial_indent="  ", subsequent_indent="  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
