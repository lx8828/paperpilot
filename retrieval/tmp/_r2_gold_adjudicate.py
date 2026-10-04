"""**A2｜扩大抽检**：用**独立第三判官**（`deepseek-reasoner`，强于 A/B）复核
**全部 dispute（删去 + 新增）+ 分层抽样的"一致"样本** → 量化新真值的错误率。

## 为什么要第三判官
A=`deepseek-chat` / B=`glm-4-flash` 是**同一次重标里**的判官；抽检若只用它们，
等于自己验自己。第三判官**独立模型 + 独立调用**（证据完全相同，来自 `gold_recalib2_evidence.json`），
且用的是**同一套四档提示词 `SYS`**（可比）。

## 检验什么
· `removed`（锚点正 → 新真值负）：若第三判官也说负 → **删得对**（锚点假阳性）
· `added`（锚点负 → 新真值正）：若第三判官也说正 → **捞得对**（锚点漏报）
· `agree-pos/neg` 分层抽样：估计**一致样本里的隐性错误率**（新真值的绝对错误率上界）

用量约 150 次调用（reasoner 慢，workers=8 ≈ 数分钟）。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_adjudicate.py --smoke 3
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_adjudicate.py --per-stratum 25
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gr = _load("goldrecal", HERE / "tmp" / "_r2_gold_recalib.py")
SYS, call, F = gr.SYS, gr.call, gr.F

DEV = HERE / "data" / "r2dev"
OUT = DEV / "gold_adjudicate.csv"
MODEL = os.environ.get("PAPERPILOT_ADJUDGE_MODEL", "deepseek-reasoner")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=25, help="每个分层抽多少条")
    ap.add_argument("--smoke", type=int, default=0, help="只跑前 N 条（冒烟）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--recalib", default="gold_recalib2.csv")
    ap.add_argument("--all-pairs", action="store_true",
                    help="对**全部** (簇,facet,doc) 对送审（供三判官多数票）")
    args = ap.parse_args()

    recalib = DEV / args.recalib
    evidf = recalib.with_name(recalib.stem + "_evidence.json")
    OUT = DEV / f"gold_adjudicate_{recalib.stem}.csv"

    df = pd.read_csv(recalib)
    df["anchor_gold"] = df["anchor_gold"].astype(bool)
    df["new_gold"] = df["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    ev = json.loads(evidf.read_text(encoding="utf-8"))

    strat = {
        "removed": df[df.anchor_gold & ~df.new_gold],           # 锚点正 → 真值负
        "added": df[~df.anchor_gold & df.new_gold],             # 锚点负 → 真值正
        "agree_pos": df[df.anchor_gold & df.new_gold],          # 一致·正
        "agree_neg": df[~df.anchor_gold & ~df.new_gold],        # 一致·负
    }
    print("=" * 110)
    print(f"【A2 扩大抽检】第三判官 = **{MODEL}**（独立于 A/B，证据完全相同）")
    for k, g in strat.items():
        print(f"  {k:<10}{len(g):>5} 条")

    rng = random.Random(20260929)
    if args.all_pairs:
        pick = list(df.itertuples())              # 全量：供"三判官多数票"用
        print(f"\n送审 **{len(pick)}** 条（**全量**，供多数票）\n")
    else:
        pick = list(strat["removed"].itertuples()) + list(strat["added"].itertuples())
        for k in ("agree_pos", "agree_neg"):
            rows = list(strat[k].itertuples())
            pick += rng.sample(rows, min(args.per_stratum, len(rows)))
        print(f"\n送审 **{len(pick)}** 条（全部 dispute + 每层抽样 ≤{args.per_stratum}）\n")
    if args.smoke:
        pick = pick[: args.smoke]

    # 换模型：`llm.config('PAPERPILOT_LLM')` 读 `PAPERPILOT_LLM_MODEL`
    os.environ["PAPERPILOT_LLM_MODEL"] = MODEL
    print(f"  ⚠️ 本轮 `PAPERPILOT_LLM_MODEL={MODEL}`（第三判官走 PAPERPILOT_LLM 通道）")
    print(f"  ✅ 实际解析模型：**{llm.config('PAPERPILOT_LLM')[2]}**"
          f"（判官 A 原为 deepseek-chat）")

    done: dict[str, dict] = {}
    if OUT.exists() and not args.fresh:
        for r in pd.read_csv(OUT).to_dict("records"):
            done[str(r["key"])] = r
    lock = threading.Lock()
    cnt = {"n": 0}

    def work(t) -> dict:
        key = f"{int(t.cluster)}|{t.facet}|{t.docid}"
        e = ev.get(key) or {}
        claim = e.get("claim") or f"该论文{F[str(t.facet)][1]}"
        user = (f"论文标题：{e.get('title', '')}\n\n论断：**{claim}**\n\n"
                f"论文原文片段：\n{e.get('evidence', '')}")
        o = call("PAPERPILOT_LLM", SYS, user, max_tokens=1200)   # reasoner 需要更多预算
        lab = str(o.get("label") or "ERR").lower()
        with lock:
            cnt["n"] += 1
            if cnt["n"] % 20 == 0:
                print(f"  复核 {cnt['n']}/{len(pick)} …", flush=True)
        return dict(key=key, cluster=int(t.cluster), facet=str(t.facet), docid=str(t.docid),
                    kind=("removed" if (t.anchor_gold and not t.new_gold) else
                          "added" if (not t.anchor_gold and t.new_gold) else
                          "agree_pos" if t.new_gold else "agree_neg"),
                    anchor_gold=bool(t.anchor_gold), new_gold=bool(t.new_gold),
                    third_label=lab, third_conf=o.get("confidence"),
                    third_quote=str(o.get("quote") or "")[:180],
                    third_reason=str(o.get("reason") or "")[:120],
                    third_err=str(o.get("_err") or "")[:60], model=MODEL)

    todo = [t for t in pick if f"{int(t.cluster)}|{t.facet}|{t.docid}" not in done]
    t0 = time.time()
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(work, todo):
                done[r["key"]] = r
        pd.DataFrame(list(done.values())).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n完成 {len(todo)} 条（{time.time() - t0:.0f}s）→ {OUT.name}")
    if args.smoke:
        return 0

    d = pd.DataFrame(list(done.values()))
    d["third_yes"] = d["third_label"].astype(str).str.lower().eq("entail")
    d["third_err_s"] = d["third_err"].where(d["third_err"].notna(), "").astype(str)
    d = d[d.third_err_s == ""]
    d["agree3"] = d.third_yes == d.new_gold

    print("\n" + "=" * 110)
    print("【第三判官 vs 新真值】按分层")
    print(f"  {'分层':<12}{'条数':>6}{'一致':>8}{'一致率':>9}   含义")
    meaning = {"removed": "第三判官也说负 → 删得对", "added": "第三判官也说正 → 捞得对",
               "agree_pos": "都说正 → 无异议", "agree_neg": "都说负 → 无异议"}
    for k in ("removed", "added", "agree_pos", "agree_neg"):
        g = d[d.kind == k]
        if not len(g):
            continue
        print(f"  {k:<12}{len(g):>6}{int(g.agree3.sum()):>8}{g.agree3.mean():>9.1%}   {meaning[k]}")

    print(f"\n  【总体一致率】{d.agree3.mean():.1%}（n={len(d)}）")
    print(f"\n  **新真值的估计错误率**（含「一致」样本的隐性错误）：")
    for lab, sub in (("正例（新真值=yes）", d[d.new_gold]), ("负例（新真值=no）", d[~d.new_gold])):
        if len(sub):
            err = 1 - sub.agree3.mean()
            print(f"    {lab:<20}{err:>7.1%}  （{int((~sub.agree3).sum())}/{len(sub)} 条第三判官不同意）")

    dis = d[~d.agree3].sort_values(["kind", "cluster", "facet"])
    print(f"\n  【分歧 {len(dis)} 条】需要人工判读：")
    print(f"  {'kind':<11}{'簇':>3} {'facet':<17}{'docid':<7}{'锚':>4}{'新':>4}{'三':>4}  第三判官引用")
    for _, r in dis.head(40).iterrows():
        print(f"  {r.kind:<11}{int(r.cluster):>3} {r.facet:<17}{r.docid:<7}"
              f"{'T' if r.anchor_gold else '-':>4}{'T' if r.new_gold else '-':>4}"
              f"{'T' if r.third_yes else '-':>4}  {str(r.third_quote)[:56]}")
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
