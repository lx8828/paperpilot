"""QASPER 外部表格通道实验：补上 arXiv 原始 PDF 的表格后，A 桶能回收多少？

背景：QASPER 的 `figures_and_tables` 只给 PNG + caption span，**表格数值无文本形式**，
导致 gold 出自表格的题在纯 QASPER 文本通道下**结构上不可达**（500 题里 36 道 = 8.0pt）。
本实验拉 arXiv 原始 PDF → MinerU 解析 → 把表格/公式文本追加进**检索视图**（chunk_id `xtbl-*`），
重跑这 36 题，看能回收多少。

链路与 `cli/run_qasper_eval.py` **完全一致**（同 `graph.ask`、同裁判 prompt、同 PASS_SCORE），
唯一差别：`PAPERPILOT_QASPER_TABLES=1`。

用法：
    uv run python qa/recall/_qasper_tbl_exp.py \
        --attrib qa/recall/_attrib_500_20260911.json \
        --base-run qa/qasper_run_20260911_072510.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

# ⚠️ 必须在 import graph 之前设：document_cache 的检索视图据此决定是否追加表格块
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"
# 外部块 RRF 权重（α=0.5 即生产原样）；用 PP_ALPHA 做端到端 A/B
if os.environ.get("PP_ALPHA"):
    os.environ["PAPERPILOT_EXT_RRF_ALPHA"] = os.environ["PP_ALPHA"]
# 外部块名额（0=关/现状；2=并集追加表池前 2 名）；用 PP_QUOTA 做端到端 A/B
if os.environ.get("PP_QUOTA"):
    os.environ["PAPERPILOT_EXT_QUOTA"] = os.environ["PP_QUOTA"]

import run_qasper_eval as R  # noqa: E402  评测判据的唯一来源
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path(os.environ.get("PP_OUT") or "qa/recall/qasper_tbl_exp_20260911.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib", default="qa/recall/_attrib_500_20260911.json")
    ap.add_argument("--base-run", default="qa/qasper_run_20260911_072510.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    items = json.loads(Path(args.attrib).read_text(encoding="utf-8"))["A 输入通道缺口"]
    if args.limit:
        items = items[: args.limit]
    base = {r["qid"]: r for r in json.loads(Path(args.base_run).read_text(encoding="utf-8"))}
    papers = load_papers()

    done: dict[str, dict] = {}
    if OUT.exists():
        try:
            done = {r["qid"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))}
        except Exception:  # noqa: BLE001
            done = {}
    recs = list(done.values())
    t0 = time.time()
    for i, it in enumerate(items, 1):
        qid = it["qid"]
        if qid in done:
            continue
        pid = it["pid"].replace("qasper_", "").replace(".qpdf", "")
        paper = papers.get(pid) or {}
        q = next((qq for qq in paper.get("qas") or []
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        name = R._ensure_report(pid)
        dbg: dict = {}
        try:
            r = graph_ask(q["question"], name)
            ans = r.get("answer") or ""
            cites = list(r.get("cites") or [])
            route = list(r.get("route") or [])
            dbg = (r.get("debug") or {}).get("answer") or {}
            level = dbg.get("level", "")
        except Exception as e:  # noqa: BLE001
            ans, cites, route, level = f"ERR {e}", [], [], "err"
        score, reason, passed = R._judge(q, ans, cites)
        b = base.get(qid, {})
        rec = {"qid": qid, "pid": pid, "question": q["question"], "gold": b.get("gold", ""),
               "score_before": b.get("score"), "status_before": b.get("status"),
               "score": score, "pass": passed, "reason": reason, "level": level,
               "answer": ans, "n_cites": len(cites), "route": route,
               "cites": cites,  # ⚠️ 必须存 cites：否则无法核"溯源是否通"
               # 诊断字段（判据 2/3 用）：上下文条目顺序 + facts 锚点 + judge 判定
               "n_entries": dbg.get("n_entries"), "entry_ids": dbg.get("entry_ids") or [],
               "n_facts": dbg.get("n_facts"), "facts_n": dbg.get("facts_n") or [],
               "enough": dbg.get("enough")}
        recs = [x for x in recs if x["qid"] != qid] + [rec]
        OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
        arrow = f"{rec['score_before']}→{score}"
        print(f"[{i}/{len(items)}] {qid[:8]} {pid} {arrow} "
              f"({'PASS' if passed else 'fail'}) {route} {reason[:60]}", flush=True)

    n = len(recs)
    before = sum(1 for r in recs if (r["score_before"] or 0) >= R.PASS_SCORE)
    after = sum(1 for r in recs if r["pass"])
    up = sum(1 for r in recs if r["pass"] and (r["score_before"] or 0) < R.PASS_SCORE)
    down = sum(1 for r in recs if not r["pass"] and (r["score_before"] or 0) >= R.PASS_SCORE)
    L = ["# QASPER 外部表格通道实验（2026-09-11）", "",
         "> 对象：④ 的 A 桶（gold 出自表格、QASPER 文本通道不可达的 36 道题）：",
         "> 拉 arXiv 原始 PDF → MinerU 解析 → 表格/公式文本追加进检索视图（`xtbl-*` 块）后重跑。",
         "> 链路与 `cli/run_qasper_eval.py` 完全一致（同 `graph.ask`/同裁判/同 PASS_SCORE=4），",
         "> 唯一差别：`PAPERPILOT_QASPER_TABLES=1`（默认关，线上行为不变）。", "",
         f"| 指标 | 值 |", "|---|---|",
         f"| 题数 | {n} |",
         f"| 补通道前 pass | {before}/{n} |",
         f"| **补通道后 pass** | **{after}/{n}** |",
         f"| 翻绿（fail→pass） | **{up}** |",
         f"| 翻红（pass→fail） | {down} |", "",
         "## 逐题（按是否翻绿排序）", "",
         "| qid | 论文 | 前 | 后 | level | route |", "|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda x: (x["pass"], -(x["score"] or 0))):
        mark = "**" if r["pass"] and (r["score_before"] or 0) < R.PASS_SCORE else ""
        L.append(f"| {mark}{r['qid'][:8]}{mark} | {r['pid']} | {r['score_before']} | "
                 f"{r['score']} | {r.get('level', '')} | {'→'.join(r['route'][-2:])} |")
    txt = "\n".join(L)
    # 报告路径随 OUT 走（A/B 两臂不再互相覆盖）
    OUT.with_suffix(".md").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\n耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
