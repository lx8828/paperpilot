"""**子查询静态质量表**（零模型，秒级）：覆盖 / 语言 / 互补性 / 泄露。

独立于 `_r2_subq_diag.py`，便于反复查看（那一段需要 5 分钟编码，这段不需要）。
"""
from __future__ import annotations

import importlib.util as _iu
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
STOP = {"the", "a", "an", "of", "for", "in", "on", "to", "and", "with", "via", "using",
        "towards", "toward", "from", "by", "at", "as", "is", "are", "be", "we", "our",
        "this", "that", "it", "can", "not", "but", "or", "its", "their", "than", "then"}


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2


def toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9\-]{1,}", str(s).lower()) if t not in STOP}


def main() -> int:
    subq = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))
    sel = json.loads((DEV / "facets_v2_selected.json").read_text(encoding="utf-8"))
    in_set = sorted({f for k in sel["per_combo"] for f in [k.split("|")[1]]})
    print("=" * 122)
    print(f"【子查询静态质量】`subqueries.json` {len(subq)} 个 facet ｜ 题集涉及 **{len(in_set)}** 个")
    miss = [f for f in in_set if f not in subq]
    print(f"  ⚠️ 题集里**无子查询**（{len(miss)}）：{miss}")
    rows = []
    for f in in_set:
        for i, s in enumerate(subq.get(f, []), 1):
            pt, at = toks(F2[f][0]), toks(F2[f][2])
            st = toks(s)
            rows.append(dict(facet=f, idx=i, subq=s, n_tok=len(st),
                             has_cjk=int(any("\u4e00" <= c <= "\u9fff" for c in s)),
                             jac_anchor=len(st & at) / max(len(st | at), 1),
                             ov_pat=len(st & pt) / max(len(pt), 1), n_pat=len(pt),
                             n_share=len(st & pt)))
    S = pd.DataFrame(rows)
    S.to_csv(HERE / "results" / "R2_SUBQ_STATIC.csv", index=False, encoding="utf-8-sig")
    print(f"\n  {'facet':<18}{'条数':>5}{'含中文':>7}{'token中位':>10}"
          f"{'↔anchor Jacc':>14}{'↔gold正则重合':>14}{'共享词':>8}  标记")
    for f in in_set:
        t = S[S.facet == f]
        if not len(t):
            print(f"  {f:<18}{0:>5}{'-':>7}{'-':>10}{'-':>14}{'-':>14}{'-':>8}  ❌ 无子查询")
            continue
        ov = t.ov_pat.mean()
        flag = ("❌ 无子查询" if not len(t) else
                "⚠️ 泄露偏高" if ov >= 0.40 else ("· 干净" if ov > 0 else "· 零重合"))
        print(f"  {f:<18}{len(t):>5}{int(t.has_cjk.sum()):>7}{int(t.n_tok.median()):>10}"
              f"{t.jac_anchor.mean():>14.3f}{ov:>14.1%}{t.n_share.mean():>8.1f}  {flag}")
    print(f"\n  【泄露基线】第一版用 `anchor` 做查询时，「↔gold正则重合」= **63%**（判泄露）。")
    print(f"  全体子查询：均 {S.ov_pat.mean():.1%} ｜ 中位 {S.ov_pat.median():.1%}"
          f" ｜ 最大 {S.ov_pat.max():.1%}（{S.loc[S.ov_pat.idxmax(), 'facet']}）")
    print(f"  含中文的子查询：{int(S.has_cjk.sum())} 条（应为 0；>0 则该路 BM25 恒 0）")

    comp = []
    for f in in_set:
        ss = [toks(x) for x in subq.get(f, [])]
        if len(ss) < 2:
            continue
        vs = [len(ss[i] & ss[j]) / max(len(ss[i] | ss[j]), 1)
              for i in range(len(ss)) for j in range(i + 1, len(ss))]
        comp.append(dict(facet=f, n=len(ss), jac_med=float(np.median(vs)),
                         jac_max=float(np.max(vs))))
    C = pd.DataFrame(comp)
    C.to_csv(HERE / "results" / "R2_SUBQ_COMPLEMENT.csv", index=False, encoding="utf-8-sig")
    print(f"\n  【互补性】同 facet 内两两 Jaccard（越低越「措辞不同」；>0.4 视为冗余）")
    print(f"  {'facet':<18}{'中位':>8}{'最大':>8}  标记")
    for _, r in C.sort_values("jac_max", ascending=False).iterrows():
        flag = "⚠️ 冗余（两条几乎同义）" if r.jac_max > 0.4 else "· 互补"
        print(f"  {r['facet']:<18}{r.jac_med:>8.3f}{r.jac_max:>8.3f}  {flag}")
    print(f"\n  全体：中位 {C.jac_med.mean():.3f} ｜ 最大 {C.jac_max.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
