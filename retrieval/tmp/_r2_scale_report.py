"""**语料规模效应报告**：20 篇（47 篇可用）vs 50 篇（150 篇可用）

读三组产物：
· `results/R2_RETR_n{20,50}_gnew_cnew_2arms.csv`  检索臂（B_pdf_fix / C_pdf_prod）
· `results/R2_r2reader*{2,50}_4_b6_a3_m0.csv`      reader 臂
· `data/r2dev/gold_final{2,3}.csv`                 真值（看正例数变化）

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_scale_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
RES = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FMT = "{:<14}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}"


def main() -> int:
    print("=" * 100)
    print("【一】语料与真值规模")
    print(f"  {'语料':<10}{'总篇数':>9}{'簇内篇数':>11}{'真值对':>9}{'正例':>8}"
          f"{'正例/题':>10}{'正例率':>9}")
    for tag, gf, n_doc in (("20", "gold_final2.csv", 47), ("50", "gold_final3.csv", 150)):
        g = pd.read_csv(DEV / gf)
        n_combo = g.groupby(["cluster", "facet"]).ngroups
        print(f"  {tag + ' 篇':<10}{n_doc:>9}{'-':>11}{len(g):>9}{int(g.gold.sum()):>8}"
              f"{g.gold.sum() / n_combo:>10.2f}{g.gold.mean():>9.1%}")

    print("\n【二】检索臂（同真值、同题面、同查询集；sorter = B mq_max）")
    print(f"  {'语料':<6}{'臂':<14}{'块数':>8}{'StRecall@10':>13}{'集合P@10':>10}"
          f"{'集合F1@10':>11}{'αNDCG@10':>11}{'证据@12k':>10}{'均gold':>8}")
    rows = {}
    for tag in ("20", "50"):
        f = RES / f"R2_RETR_n{tag}_gnew_cnew_2arms.csv"
        if not f.exists():
            print(f"  ⚠️ 缺 {f.name}")
            continue
        d = pd.read_csv(f)
        m = d[d.sorter == "B mq_max"]
        rows[tag] = m
        for arm in ("B_pdf_fix", "C_pdf_prod"):
            s = m[m.arm == arm]
            if not len(s):
                continue
            print(f"  {tag:<6}{arm:<14}{s['n_chunks'].mean():>8.0f}"
                  f"{s['StRecall@10'].mean():>13.3f}{s['setP@10'].mean():>10.3f}"
                  f"{s['setF1@10'].mean():>11.3f}{s['aNDCG@10'].mean():>11.3f}"
                  f"{s['ev_recall_c12k'].mean():>10.3f}{s['n_gold'].mean():>8.2f}")

    print("\n  【关键 Δ：换到 50 篇语料后】")
    if "20" in rows and "50" in rows:
        for arm in ("C_pdf_prod", "B_pdf_fix"):
            a, b = rows["20"], rows["50"]
            sa, sb = a[a.arm == arm], b[b.arm == arm]
            if not len(sa) or not len(sb):
                continue
            print(f"    {arm:<14}"
                  f"StRecall@10 {sa['StRecall@10'].mean():.3f}→{sb['StRecall@10'].mean():.3f}"
                  f"（{sb['StRecall@10'].mean() - sa['StRecall@10'].mean():+.3f}）｜ "
                  f"集合F1@10 {sa['setF1@10'].mean():.3f}→{sb['setF1@10'].mean():.3f}"
                  f"（{sb['setF1@10'].mean() - sa['setF1@10'].mean():+.3f}）")
        # 切块臂的**相对优势**（生产 vs 固定窗）在两种规模下是否保持
        print(f"\n  【切块臂优势是否保持】C_pdf_prod − B_pdf_fix")
        for tag in ("20", "50"):
            m = rows[tag]
            c, b = m[m.arm == "C_pdf_prod"], m[m.arm == "B_pdf_fix"]
            if not len(c) or not len(b):
                continue
            print(f"    {tag} 篇：StRecall@10 {c['StRecall@10'].mean() - b['StRecall@10'].mean():+.3f}"
                  f" ｜ 集合F1@10 {c['setF1@10'].mean() - b['setF1@10'].mean():+.3f}"
                  f" ｜ 证据@12k {c['ev_recall_c12k'].mean() - b['ev_recall_c12k'].mean():+.3f}")

    print("\n【三】reader 臂（生产切块 + 锚点注入 a=3，协议 4）")
    print(f"  {'语料':<6}{'题数':>6}{'均gold':>8}{'均YES':>8}{'集合P':>9}{'集合R':>9}"
          f"{'集合F1':>9}{'检索基线F1':>12}{'增益':>9}")
    rrows = {}
    for tag, pat in (("20", "*r2reader2_4_b6_a3_m0.csv"),
                     ("50", "*r2reader50_4_b6_a3_m0.csv")):
        hits = sorted(RES.glob(f"R2_{pat}"))
        if not hits:
            print(f"  ⚠️ 缺 reader({tag})：{pat}")
            continue
        d = pd.read_csv(hits[-1])
        if "reader_F1" not in d.columns:
            print(f"  ⚠️ {hits[-1].name} 无 reader_F1")
            continue
        rrows[tag] = d
        gain = d.reader_F1.mean() - d.ret_F1.mean()
        print(f"  {tag:<6}{len(d):>6}{d.n_gold.mean():>8.2f}{d.n_yes.mean():>8.2f}"
              f"{d.reader_P.mean():>9.3f}{d.reader_R.mean():>9.3f}"
              f"{d.reader_F1.mean():>9.3f}{d.ret_F1.mean():>12.3f}{gain:>+9.3f}")

    if "20" in rrows and "50" in rrows:
        a, b = rrows["20"], rrows["50"]
        print(f"\n  【关键 Δ】reader：集合F1 {a.reader_F1.mean():.3f}→{b.reader_F1.mean():.3f}"
              f"（{b.reader_F1.mean() - a.reader_F1.mean():+.3f}）"
              f" ｜ 相对检索的增益 {a.reader_F1.mean() - a.ret_F1.mean():+.3f}→"
              f"{b.reader_F1.mean() - b.ret_F1.mean():+.3f}")

    print("\n【四】逐 facet（reader：50 篇下 gold vs 判 YES，按 F1 升序）")
    if "50" in rrows:
        d = rrows["50"].copy()
        d["key"] = d.cluster.astype(str) + " " + d.facet
        d = d.sort_values("reader_F1")
        print(f"  {'簇 facet':<22}{'gold':>6}{'YES':>6}{'P':>8}{'R':>8}{'F1':>8}{'retF1':>8}")
        for _, r in d.iterrows():
            print(f"  {r['key']:<22}{int(r['n_gold']):>6}{int(r['n_yes']):>6}"
                  f"{r['reader_P']:>8.3f}{r['reader_R']:>8.3f}{r['reader_F1']:>8.3f}"
                  f"{r['ret_F1']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
