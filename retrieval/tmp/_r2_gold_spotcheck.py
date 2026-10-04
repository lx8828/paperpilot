"""**A1b｜真值抽检**：把"被删去/被新增/低置信/判官出错"的样本连**证据全文 + 判官理由**打出来人工看。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_spotcheck.py --facet deployment
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_spotcheck.py --kind removed --n 3
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_spotcheck.py --kind added --n 3
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_spotcheck.py --kind err --n 4
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_spotcheck.py --cid 1,cdeployment   # 指定
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

spec = _iu.spec_from_file_location("goldrecal", HERE / "tmp" / "_r2_gold_recalib.py")
gr = _iu.module_from_spec(spec)
sys.modules["goldrecal"] = gr
spec.loader.exec_module(gr)
F = gr.F

DEV = HERE / "data" / "r2dev"
df = pd.read_csv(DEV / "gold_recalib2.csv")
df["anchor_gold"] = df["anchor_gold"].astype(bool)
df["new_gold"] = df["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
EVID = json.loads((DEV / "gold_recalib2_evidence.json").read_text(encoding="utf-8"))


def show(r: pd.Series, ev: str, width: int = 108, evchars: int = 700) -> None:
    tag = ("删去" if (r.anchor_gold and not r.new_gold) else
           "新增" if (not r.anchor_gold and r.new_gold) else
           "一致-正" if r.new_gold else "一致-负")
    print("\n" + "=" * width)
    print(f"【{tag}】簇{int(r.cluster)} · {r.facet} · {r.docid}"
          f" ｜ 锚点={bool(r.anchor_gold)} → 新真值={bool(r.new_gold)}"
          f" ｜ 论断：{F[str(r.facet)][1]}")
    print(f"  A {r.agent_A}: **{r.A_label}**（conf={r.A_conf}）"
          + (f"  引用: {str(r.A_quote)[:150]}" if isinstance(r.A_quote, str) and r.A_quote.strip() else ""))
    print(f"  B {r.agent_B}: **{r.B_label}**（conf={r.B_conf}）"
          + (f"  引用: {str(r.B_quote)[:150]}" if isinstance(r.B_quote, str) and r.B_quote.strip() else ""))
    if int(r.need_third):
        print(f"  ⚖️ 第三轮(严格对抗): {r.third}")
    e = pd.notna(r.A_err) and str(r.A_err) not in ("", "nan")
    e2 = pd.notna(r.B_err) and str(r.B_err) not in ("", "nan")
    if e or e2:
        print(f"  ⚠️ 调用错误: A={r.A_err if e else '—'} B={r.B_err if e2 else '—'}")
    ev = " ".join(str(ev).split())
    print(f"  ── 证据（{r.evidence_chars} 字符·块 {r.ev_idx}）──")
    print(textwrap.fill(ev[:evchars], width, initial_indent="  ", subsequent_indent="  "))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="", choices=("", "removed", "added", "lowconf", "err"))
    ap.add_argument("--facet", default="")
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--cid", default="", help="形如 1|deployment|C1P3")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--evchars", type=int, default=700)
    args = ap.parse_args()

    if args.cid:
        c, f, d = args.cid.split("|")
        sub = df[(df.cluster == int(c)) & (df.facet == f) & (df.docid == d)]
    else:
        sub = df
        if args.cluster:
            sub = sub[sub.cluster == args.cluster]
        if args.facet:
            sub = sub[sub.facet == args.facet]
        if args.kind == "removed":
            sub = sub[sub.anchor_gold & ~sub.new_gold]
        elif args.kind == "added":
            sub = sub[~sub.anchor_gold & sub.new_gold]
        elif args.kind == "lowconf":
            sub = sub[sub.need_third == 1]
        elif args.kind == "err":
            sub = sub[(sub.A_err.astype(str) != "") | (sub.B_err.astype(str) != "")]
        elif args.kind == "":
            sub = sub[sub.anchor_gold]        # 默认看锚点正例

    print(f"命中 **{len(sub)}** 条（按簇/facet 排序，展示前 {args.n} 条）")
    if not len(sub):
        return 0
    for _, r in sub.sort_values(["cluster", "facet", "docid"]).head(args.n).iterrows():
        key = f"{int(r.cluster)}|{r.facet}|{r.docid}"
        ev = (EVID.get(key) or {}).get("evidence", "（无证据缓存）")
        show(r, ev, evchars=args.evchars)
    print("\n" + "=" * 108)
    print("人工判读要点：① 判官是否把「本文做了」错判成「引用他人」？")
    print("              ② 删去的那些，证据里到底有没有本文自己的动作？")
    print("              ③ 若大量误删 → 提示词/判官需调；若站得住 → 新真值更严但更准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
