r"""**抽两道题看真实效果**：现场跑生产链路 `set_judge.run` + 原样 `render`。

选的两道（从 `R2_PROD_FINAL.csv` 里挑的**一好一坏**，不是特挑）：
| 题 | 簇·facet | gold | 交付 | setF1 |
|---|---|---|---|---|
| 1 | `c1 · significance`（"报告了显著性检验/多次运行波动"） | 7 | 8 | **0.933** |
| 2 | `c2 · error_analysis`（"做了错误分析"） | 3 | 3 | **0.333** |

⚠️ 两道在**不同簇** → 需各编码一次（约 1~2 分钟/簇）。**若本机有别的程序占显存**
（2026-10-01 实测：游戏占 1.9 GB → 只剩 4 GB，第二个簇编码触发 Windows 换页，
GPU 100% 但 6 分钟无进展）→ 改用 `--picks "1:significance,1:annotation_cost"`（同簇只编码一次）。

顺带把**新旧判官协议**都跑一遍（`PAPERPILOT_SET_DUAL` = 0/1）→ 肉眼确认"单判官 ≈ 双判官"。

跑法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_demo2.py
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_demo2.py --picks "1:significance,3:proof_theory"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ★ 复用评测脚手架：索引契约 / 检索方法 / 模型加载**全是生产实现**，不另写一套
_spec = importlib.util.spec_from_file_location("_PE", HERE / "tmp" / "_r2_prod_eval.py")
PE = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(PE)                         # type: ignore[union-attr]
SJ, DEV, Corpus50Index = PE.SJ, PE.DEV, PE.Corpus50Index
sys.path.insert(0, str(HERE / "tmp"))
from _r2_facets_v2 import F2 as FAC                  # noqa: E402

K_CUT = 30            # 定稿口径：交付 top-30 篇


def load_cluster(ci: int) -> Corpus50Index:
    """建簇索引（编码一次，约 35s）。"""
    pmap = PE.json.loads((DEV / "corpus50" / "pdf_map_all.json").read_text(encoding="utf-8"))
    pm_ok = {d for d, v in pmap.items() if v.get("ok")}
    ch = pd.read_parquet(DEV / "prodchunk50" / "mineru" / f"c{ci - 1}.parquet")
    ls = pd.read_parquet(DEV / "corpus50" / f"c{ci - 1}.parquet")
    docs = [d for d in ls["docid"].tolist() if d in set(ch["docid"]) and d in pm_ok]
    texts, owner = [], []
    for d in docs:
        for t in ch[ch.docid == d]["text"].astype(str).tolist():
            texts.append(t)
            owner.append(d)
    vecs = np.asarray(PE.E.encode_texts(texts), dtype="float32")
    return Corpus50Index(docs, owner, texts, vecs)


def titles() -> dict[str, str]:
    """docid → 标题（有就用，没有就不显示，不猜）。"""
    for p in (DEV / "corpus50" / "pdf_map_all.json", DEV / "arxiv_title_search.json"):
        try:
            o = PE.json.loads(Path(p).read_text(encoding="utf-8"))
            t = {k: str(v.get("title") or v.get("t") or "") for k, v in o.items()
                 if isinstance(v, dict)}
            if any(t.values()):
                return t
        except Exception:  # noqa: BLE001
            continue
    return {}


def run_one(ci: int, facet: str, idx, gold: set[str], tt: dict[str, str],
            *, dual: bool, b: int, workers: int) -> dict:
    claim = str(FAC[facet][1])
    os.environ["PAPERPILOT_SET_DUAL"] = "1" if dual else "0"
    t0 = time.time()
    res = SJ.run(claim, idx, claim=claim, b=b, workers=workers)
    cand = [str(c["pdf"]) for c in res["candidates"]]
    keep = set(cand[:K_CUT])
    pred = {p["pdf"] for p in res["papers"]} & keep
    inter = len(pred & gold)
    P = inter / max(len(pred), 1)
    R = inter / max(len(gold), 1)
    # `render` 会输出**全部判 yes 的篇**；为与上面的 top-K_CUT 口径一致，
    # 先切一份专属结果给渲染用（否则渲染里会多出 rank>30 的篇，与数字对不上）。
    res_k = dict(res)
    res_k["papers"] = [p for p in res["papers"] if p["pdf"] in keep]
    return dict(claim=claim, res=res, res_k=res_k, pred=pred, gold=gold, inter=inter,
                P=P, R=R, F1=2 * P * R / (P + R) if (P + R) else 0.0,
                secs=time.time() - t0, dual=dual)


def show(tag: str, r: dict, tt: dict[str, str]) -> None:
    claim = r["claim"]
    print(f"\n  ── 判官协议【{tag}】── 交付 {len(r['pred'])} 篇 ｜ 对 {r['inter']} ｜ "
          f"setP {r['P']:.3f} ｜ setR {r['R']:.3f} ｜ **setF1 {r['F1']:.3f}**"
          f" ｜ {r['secs']:.0f}s")
    ps = [p for p in r["res"]["papers"] if p["pdf"] in r["pred"]]
    print("     交付（按检索分降序；✓=在 gold 里，✗=误报）：")
    for i, p in enumerate(ps, 1):
        t = (tt.get(p["pdf"]) or "")[:44]
        mark = "✓" if p["pdf"] in r["gold"] else "✗"
        ex = f" ｜抽取：{p['extract']}" if p.get("extract") else ""
        print(f"       {i:>2}. [{mark}] `{p['pdf']}` {t}")
        print(f"           理由：{p['why']}{ex}（证据 片段{p['evidence']}）")
    miss = sorted(r["gold"] - r["pred"])
    if miss:
        print(f"     漏掉 {len(miss)} 篇：")
        for d in miss:
            print(f"       · `{d}` {(tt.get(d) or '')[:44]}"
                  f"  ｜ A={r['res']['verdicts'].get(d, {}).get('A')}"
                  f" B={r['res']['verdicts'].get(d, {}).get('B')}")
    else:
        print("     漏掉 0 篇 ✓")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", default="1:significance,2:error_analysis")
    ap.add_argument("--b", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="忽略子查询缓存（会引入 ±0.04 漂移）")
    args = ap.parse_args()
    if args.fresh:
        os.environ["PAPERPILOT_QUERY_CACHE"] = "0"
    picks = [(int(x.split(":")[0]), x.split(":")[1]) for x in args.picks.split(",")]
    tt = titles()

    print("=" * 118)
    print(f"【抽题演示】{len(picks)} 道 ｜ 生产链路 `set_judge.run` ｜ b={args.b} ｜ 交付 "
          f"top-{K_CUT} ｜ 判官 B 已配置={PE.llm.judge_configured()}")
    print(f"  （口径来自 `R2_PROD_FINAL.csv`：好 = c1|significance F1 0.933，"
          f"坏 = c2|error_analysis F1 0.333）")

    gg = pd.read_csv(DEV / "gold_final3.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}

    IDX: dict[int, Corpus50Index] = {}
    for n, (ci, facet) in enumerate(picks, 1):
        if ci not in IDX:
            t0 = time.time()
            IDX[ci] = load_cluster(ci)
            print(f"  簇{ci} 索引就绪：{len(IDX[ci].pdfs)} 篇 / "
                  f"{len(IDX[ci]._chunks)} 块 ｜ {time.time() - t0:.0f}s", flush=True)
        gold = (NG.get((ci, facet)) or set()) & set(IDX[ci].pdfs)
        print("\n" + "=" * 118)
        print(f"【第 {n} 题】簇{ci} · facet = `{facet}`   ｜ gold {len(gold)} 篇 ｜ "
              f"候选 {len(IDX[ci].pdfs)} 篇")
        print(f"  ★ 论断：{FAC[facet][1]}")

        modes = [(False, "单判官 A（**新默认**）")]
        if PE.llm.judge_configured():
            modes.append((True, "双判官 + 第三轮（旧）"))
        outs = {}
        for dual, tag in modes:
            outs[tag] = run_one(ci, facet, IDX[ci], gold, tt,
                                dual=dual, b=args.b, workers=args.workers)
            show(tag, outs[tag], tt)

        if len(outs) == 2:
            a, b_ = (list(outs.values()))
            same = a["pred"] == b_["pred"]
            print(f"\n  ⚖️ 两协议交付集合**{'完全相同 ✓' if same else '不同'}**"
                  f"（ΔsetF1 {b_['F1'] - a['F1']:+.3f}）"
                  + ("" if same else f"  ｜ 差集 {sorted(a['pred'] ^ b_['pred'])}"))

        first = outs[list(outs)[0]]
        print(f"\n  ── 系统原样渲染的答案（`set_judge.render`，单判官口径、截 top-{K_CUT}）──")
        print(SJ.render(first["claim"], first["res_k"]))
    print("\n" + "=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
