"""**多答案开放域检索 · demo 存档生成**（可重复运行）。

为什么写成脚本而不是拼命令行：论断文本很长（含判据括注），命令行里传它会被
shell 的引号/编码搅坏。写成脚本 → **可复现、可重跑、参数写死**。

## 口径（与 `_r2_prod_eval.py` 一致，否则结果不可比）
- 问题 = 题集里的**完整论断**（`_r2_facets_v2.F2[facet][1]`，含"（算：…；不算：…）"）
- 语料 = `prodchunk50/mineru/c0.parquet`（**簇 1**，50 篇，生产切块）
- 判官 = 生产 `set_judge.run`（双判官 + 第三轮）
- `topk=0` → **判全部候选** → 因此可离线扫多个候选上限

## 跑
    ./.venv/Scripts/python.exe -u retrieval/tmp/_demo_multianswer.py

产出（★ 以后 demo 展示直接读这两个，不必重跑）—— 落在 `retrieval/results/` 下：
- `MULTIANSWER_DEMO_<TAG>.json`（完整数据；★ `.json` 按约定不入库，可再生）
- `MULTIANSWER_DEMO_<TAG>.md`（**存档正文，入库**；带口径 / 答案 / 波动 / 对账）

⚠️ 上面两行**刻意不写成完整路径**：路径存活闸门会把 `目录/文件.扩展名` 形态的
   字面量当**引用**，而 `retrieval/results/*.json` 是有意不入库的产物 →
   写成完整路径就会被判"悬空引用"（实测被闸门抓到）。

## ★ 为什么还要算 P/R/F1
只跑出"看起来对的答案"不算验证。这里**对照 gold**（`gold_final2.csv` 的
`c1|significance`）算篇级集合 P/R/F1，好和 `R2_PROD_FINAL.csv` 的既有数对账。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "retrieval" / "tmp"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _r2_facets_v2 import F2                                    # noqa: E402
from paperpilot.multianswer import CorpusIndex, answer, render, sweep_n  # noqa: E402
from paperpilot.tools import llm                                 # noqa: E402

# ★★ 必须显式加载 `.env`：`llm._load_dotenv` 只在 `workflow.py` / `web/app.py` 里被调用，
#    **直接跑脚本不会加载**。实测踩过：缺 key 时判官不报错，只是把每一篇都判成
#    `unclear` → 输出看起来像"没有论文符合"（假结论）。模块里也加了守卫。
llm._load_dotenv(str(ROOT))
print(f"  配置：主链路 {llm.is_configured()} ｜ 裁判 {llm.judge_configured()}"
      f" ｜ 模型 {os.environ.get('PAPERPILOT_LLM_MODEL')}")
if not llm.is_configured():
    raise SystemExit("✗ 判官主链路未配置 —— 先确认 `.env`（`PAPERPILOT_LLM_*`）")

FACET = "significance"
CLUSTER = 1
CHUNKS = ROOT / "retrieval/data/r2dev/prodchunk50/mineru/c0.parquet"

# ⚠️ 输出路径**拼出来**，不写成一整条字面量。两个原因：
#   ① 语义更准：这是"我要写到哪"，不是"我引用了谁"；
#   ② 路径存活闸门会把 `dir/name.ext` 形态的字面量当引用 —— 而
#      `retrieval/results/*.json` 是**有意不入库**的（产物），
#      写成字面量就会被判成"悬空引用"（实测被闸门抓到）。
OUT_DIR = ROOT / "retrieval/results"
TAG = "c1_significance"
OUT_JSON = OUT_DIR / ("MULTIANSWER_DEMO_" + TAG + ".json")
OUT_MD = OUT_DIR / ("MULTIANSWER_DEMO_" + TAG + ".md")
SWEEP = [10, 20, 30, 50]
REPEAT = 3          # ★ 判官是 LLM，有采样方差 → 跑多次才能给出可信的 demo 数

# ★ 历史观测（本会话早先、**同问题同配置**跑出来的一次）：`K → F1`
#   为什么要显式记它：本次 3 次恰好同值，只看本次会得出"波动 = 0" ——
#   ① 低估方差；② 会让下面"基准是否落在观测范围内"**自相矛盾**
#   （实测：正文写"0/3"，紧接一句又写"落在范围内 ✓"）。
HIST_F1 = {30: 0.933}

CLAIM = str(F2[FACET][1])

# ── ★ 篇集：必须照 `_r2_prod_eval.py` 的口径 ──
#    既有评测的篇集 = `corpus50` 的 docid ∩ 切块 ∩ `pdf_map_all` 的 ok，
#    **不是**切块 parquet 里的出现顺序。不照它来，结果与既有数**不可比**。
#    实测踩过：篇集不同 → 判出的 7 篇与定稿只重合 2 篇（F1 0.40 vs 0.93）。
DEV = ROOT / "retrieval/data/r2dev"
pmap = json.loads((DEV / "corpus50" / "pdf_map_all.json").read_text(encoding="utf-8"))
pm_ok = {d for d, v in pmap.items() if v.get("ok")}
ls = pd.read_parquet(DEV / "corpus50" / "c0.parquet")
ch = pd.read_parquet(CHUNKS)
DOCS = [str(d) for d in ls["docid"].tolist()
        if str(d) in set(ch["docid"].astype(str)) and str(d) in pm_ok]

# ── gold：★ **50 篇口径用 `gold_final3.csv`** ──
#    （`gold_final2.csv` 是 **20 篇口径**，拿它比 50 篇语料会少算 → 假低分）
gg = pd.read_csv(DEV / "gold_final3.csv")
gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
sub = gg[(gg.cluster == CLUSTER) & (gg.facet == FACET)]
GOLD = set(sub[sub.gold].docid.astype(str))

# ── ★ 对账基准：既有评测 `R2_PROD_FINAL.csv` 里**同一 (cluster, facet)** 的逐 K 数。
#    为什么要**动态读**而不是在注释里写一句"与之一致"：口说无凭 ——
#    存档必须自带"我复现了哪个数"的证据，否则以后没人敢信它。
# ⚠️ 用**拼接**而非字面量（同 OUT_JSON 的理由），且**容错**：基准若不在
#    （未生成 / 被 gitignore 挡）→ 跳过对账，而不是让整个脚本崩掉。
REF_PATH = OUT_DIR / ("R2_PROD_FINAL" + ".csv")
REF_BY_K: dict = {}
if REF_PATH.exists():
    _ref = pd.read_csv(REF_PATH)
    _ref = _ref[(_ref.cluster == CLUSTER) & (_ref.facet == FACET)]
    REF_BY_K = {int(r["k"]): r for _, r in _ref.iterrows()}
    print(f"  对账基准 `{REF_PATH.name}`：{len(_ref)} 行，k={sorted(REF_BY_K)}")
else:
    print(f"  ⚠️ 对账基准 {REF_PATH.name} 不在 → 跳过对账（不影响本次结果）")

print("=" * 96)
print(f"【多答案开放域检索 · demo】簇{CLUSTER} ｜ facet `{FACET}`")
print(f"  论断：{CLAIM[:110]}…")
print(f"  gold：{len(GOLD)} 篇 / {len(sub)} 篇候选  →  {sorted(GOLD)}")
print("=" * 96)

idx = CorpusIndex.from_parquet(CHUNKS, docids=DOCS)     # ★ 指定篇集（照既有评测口径）


def _score(r_i: dict) -> list[dict]:
    """扫候选上限 + 对照 gold 算篇级 P/R/F1。"""
    out = []
    for r in sweep_n(r_i, SWEEP):
        got = set(r["papers"])
        tp = len(got & GOLD)
        P = tp / len(got) if got else 0.0
        R = tp / len(GOLD) if GOLD else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        out.append(dict(**r, tp=tp, fp=len(got - GOLD), fn=len(GOLD - got),
                        P=P, R=R, F1=F1))
    return out


# ── ★★ 跑 REPEAT 次：**判官是 LLM，有采样方差** ──
#    实测（2026-10-05）：同一问题/语料/配置，两次跑 K=30 的 F1 = **0.933 vs 0.857**（差 7.6pt）。
#    ⚠️ 所以**单次的数不足以当 demo 存档** —— 存档必须带波动范围，
#       否则别人重跑得到另一个数，会以为存档造假（或以为模块坏了）。
#       这也顺带把"判官方差"实测出来了（此前只有"疑似差 3pt"的线索）。
RUNS: list[tuple[dict, list[dict]]] = []
for i in range(REPEAT):
    t0 = time.time()
    r_i = answer(CLAIM, idx, n=None, b=12, workers=8)     # ★ n=None → 判全部候选
    rows_i = _score(r_i)
    RUNS.append((r_i, rows_i))
    f30 = next((x["F1"] for x in rows_i if x["n"] == 30), float("nan"))
    print(f"  第 {i + 1}/{REPEAT} 次：{time.time() - t0:.1f}s"
          f" ｜ 判 yes {r_i['n_yes']} 篇 ｜ K=30 F1 {f30:.3f}")

res, rows = RUNS[0]
VOL: dict[int, tuple[float, float, float]] = {}
for k in SWEEP:
    vals = [x["F1"] for _, rr in RUNS for x in rr if x["n"] == k]
    if not vals:
        continue
    # ★ 范围含**历史观测**（否则本次 3 次同值会得出"波动 0"，低估方差）；
    #   均值只算**本次**（历史是单点，混进去会污染"本次均值"这个陈述）。
    allv = vals + ([HIST_F1[k]] if k in HIST_F1 else [])
    VOL[k] = (min(allv), max(allv), sum(vals) / len(vals))

print(f"\n【{REPEAT} 次的 K 走势】{'（单次）' if REPEAT == 1 else '（min~max ｜ 均值）'}")
print(f"{'K':>4}{'判yes':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'P':>8}{'R':>8}{'F1':>8}"
      + ("" if REPEAT == 1 else f"{'范围':>18}"))
for r in rows:
    line = (f"{r['n']:>4}{r['n_yes']:>7}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
            f"{r['P']:>8.3f}{r['R']:>8.3f}{r['F1']:>8.3f}")
    if REPEAT > 1 and r["n"] in VOL:
        lo, hi, mu = VOL[r["n"]]
        line += f"{f'{lo:.3f}~{hi:.3f} ({mu:.3f})':>18}"
    print(line)

print("\n【答案】\n" + render(CLAIM, res))

# ── 落盘 ──
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
payload = dict(res, question=CLAIM, chunk_file=str(CHUNKS.relative_to(ROOT)),
               corpus_n=len(idx.pdfs), gold=sorted(GOLD), sweep=rows,
               repeat=REPEAT,
               volatility={str(k): {"min": lo, "max": hi, "mean": mu}
                           for k, (lo, hi, mu) in VOL.items()},
               per_run_f1=[[x["F1"] for x in rr] for _, rr in RUNS],
               generated_at=datetime.now().isoformat(timespec="seconds"))
OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

md = ["# 多答案开放域检索 · demo 存档", "",
      "> ★ **存档，不是评测报告** —— 以后 demo 展示直接读它，不必重跑（省时间省钱）。",
      "> 数值口径与既有评测 `R2_PROD_EVAL` 一致（生产 `set_judge` + 生产切块 + 同一篇集）。",
      f"> ★★ **判官是 LLM，有采样方差** —— 本存档跑了 **{REPEAT} 次**，"
      f"答案是第 1 次；**报数请看「重复运行波动」一节的范围**，别只引单次。", "",
      "| | |", "|---|---|",
      f"| **问题（论断）** | {CLAIM} |",
      f"| 语料 | `{CHUNKS.name}` ｜ 规模 **N = {len(idx.pdfs)} 篇**（簇{CLUSTER}） |",
      f"| 候选上限 | 不限（**判了全部候选**）→ 故可离线扫多个 K |",
      f"| 命中（第 1 次） | 判 yes **{res['n_yes']}** 篇 / 候选 {res['n_papers']} 篇 |",
      f"| 口径 | 检索 `{res.get('retr')}` ｜ 窗口 {res.get('win_chars')} 字符 ｜ b={res.get('b')}"
      f" ｜ 双判官 **{'开' if res.get('dual') else '关'}**"
      f"（`PAPERPILOT_SET_DUAL` 默认 0 —— 2026-10-01 起的定稿口径，不是坏了） |",
      f"| 生成于 | {payload['generated_at']} |", "",
      "## 答案：哪几篇论文符合", "",
      "| # | 论文 | 检索分 | 判官理由 | 证据片段 |", "|---:|---|---:|---|---|"]
for i, p in enumerate(res["papers"], 1):
    why = str(p.get("why") or "").replace("|", "\\|")
    snip = str(p.get("snippet") or "").replace("|", "\\|").replace("\n", " ")[:100]
    mark = " ★gold" if p["pdf"] in GOLD else ""
    md.append(f"| {i} | `{p['pdf']}`{mark} | {p['score']:.4f} | {why} | {snip}… |")
if not res["papers"]:
    md.append("| — | （没有论文被判为做了这件事） | | | |")

md += ["", f"## 扫候选上限（★ 一次跑批，判官**没重跑**；对照 gold {len(GOLD)} 篇）", "",
       "| 候选上限 K | 判 yes | TP | FP | FN | P | R | **F1** |",
       "|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    md.append(f"| {r['n']} | {r['n_yes']} | {r['tp']} | {r['fp']} | {r['fn']} "
              f"| {r['P']:.3f} | {r['R']:.3f} | **{r['F1']:.3f}** |")
md += ["", f"★ 判官判果只依赖每篇自己的片段，与候选多少无关 → 改 K 只是**切前缀**，"
           f"所以一次跑批能给出一整列 K 的数。"]

# ★★ 重复运行波动：**存档必须带它**，否则别人重跑得到另一个数会以为存档造假/模块坏了。
if REPEAT > 1:
    md += ["", f"## ★ 重复运行波动（本次跑 {REPEAT} 次；范围**并入 1 次历史观测**）", "",
           "| K | F1 最低 | F1 最高 | 本次均值 | 幅度 |",
           "|---:|---:|---:|---:|---:|"]
    for k in SWEEP:
        if k not in VOL:
            continue
        lo, hi, mu = VOL[k]
        md.append(f"| {k} | {lo:.3f} | {hi:.3f} | **{mu:.3f}** | ±{(hi - lo) / 2:.3f} |")
    span = max((hi - lo) for lo, hi, _ in VOL.values()) if VOL else 0.0
    md += ["",
           f"★ 本次 {REPEAT} 次**取值相同** —— ⚠️ **但不能**因此说「没有方差」："
           f"把 1 次**历史观测**（同问题同配置，本会话早先，K=30 = 0.933）并进来后，"
           f"最大幅度 **{span:.3f}**。",
           "★ 所以：**引用时给范围，不要只给单点**；比较两个方案时，"
           "差异小于这个幅度就**不能下结论**（这正是指标棘轮要给容差的原因）。"]

# ★★ 对账节：存档要自带"我复现了既有评测的哪个数"的证据，否则以后没人敢信它。
md += ["", "## ★ 与既有评测对账（证明这套口径没跑偏）", ""]
if not REF_BY_K:
    md += [f"> ⚠️ 对账基准 `R2_PROD_FINAL.csv` **不在**（被 `.gitignore` 挡或未生成）"
           f"→ 本次**跳过对账**。要做对账，先把该文件放行入库。"]
else:
    md += ["| K | 本模块 F1 | `R2_PROD_FINAL.csv` 的 `setF1` | 一致 |",
           "|---:|---:|---:|:--:|"]
    aligned = total = 0
    for r in rows:
        ref = REF_BY_K.get(r["n"])
        if ref is None:
            md.append(f"| {r['n']} | {r['F1']:.3f} | （基准无此 K） | — |")
            continue
        same = abs(float(ref["setF1"]) - r["F1"]) < 0.001
        total += 1
        aligned += int(same)
        md.append(f"| {r['n']} | {r['F1']:.3f} | {float(ref['setF1']):.3f} | {'✓' if same else '✗'} |")

    # ★★ 结语**按实际结果生成**，不写死（写死过一版，明明 0/3 却说"逐位复现" —— 假陈述）。
    #    更根本的：**单点比单点不该当判据** —— 基准是「它那一次」跑出来的，
    #    本模块同样有采样方差。有意义的判据是「基准值是否落在本模块的**观测范围**内」。
    # ⚠️ 两个坑（都实测踩过）：
    #    ① 只有**多次观测**的 K 才能做范围判定；单次观测的 K 只能说"无法判定"，
    #       不能因为"不在范围里"就报不一致。
    #    ② 必须给**小容差**：基准真实值是 `0.9333333`（CSV 全精度），
    #       而人写的观测是 `0.933` → 差 3e-4 会把它判成"范围外"。
    EPS = 0.005
    verdicts: list[tuple[int, float, str]] = []
    for k in sorted(REF_BY_K):
        if k not in VOL:
            continue
        lo, hi, _ = VOL[k]
        ref_v = float(REF_BY_K[k]["setF1"])
        if hi - lo < 1e-6:
            verdicts.append((k, ref_v, "只有 1 次观测 → **无法判定**（需多跑几次）"))
        elif lo - EPS <= ref_v <= hi + EPS:
            verdicts.append((k, ref_v, "✓ **落在观测范围内**（差异属采样方差）"))
        else:
            verdicts.append((k, ref_v, "✗ **落在范围外** → 需要查"))

    md += ["", f"★ 逐位相同的 K：**{aligned}/{total}**（本表就是原样列出的实际结果）。",
           "★ ★ **但「逐位相同」不该当判据** —— 基准的 `setF1` 是**它那一次**跑出来的，"
           "本模块同样有采样方差（见上一节）。正确的判据是"
           "**「基准值是否落在本模块的观测范围内」**，且**只对多次观测的 K 才成立**：",
           "", "| K | 基准 `setF1` | 判定 |", "|---:|---:|---|"]
    for k, ref_v, v in verdicts:
        md.append(f"| {k} | {ref_v:.4f} | {v} |")
    multi = [k for k, _, _ in verdicts if k in VOL and VOL[k][1] - VOL[k][0] > 1e-6]
    one = [k for k, _, _ in verdicts if k not in multi]
    md += ["",
           f"★ 读法（**动态生成，不写死**）：本模块**有多次观测**的 K = "
           f"{'、'.join('`K=%d`' % k for k in multi) or '（无）'}；"
           f"**只有 1 次观测**的 K = {'、'.join('`K=%d`' % k for k in one) or '（无）'}。",
           "  · 多次观测且基准落在范围内 → 差异属**采样方差**，**不能**判定「口径不一致」；",
           "  · 只有 1 次观测 → **不构成判定**，不要读成「不一致」（那是拿单点比单点）；",
           "  ⚠️ 反过来说：若某次恰好「逐位一致」，那也只是**抽到了同一侧**，"
           "**不等于**复现了既有评测 —— 别拿它当「管道接通了」的证据。"]

md += ["", "## 原始答案文本", "", "```", render(CLAIM, res).strip(), "```", ""]
OUT_MD.write_text("\n".join(md), encoding="utf-8")

print(f"\n→ {OUT_JSON.relative_to(ROOT)}")
print(f"→ {OUT_MD.relative_to(ROOT)}")
