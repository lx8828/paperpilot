"""两步法修复的配对回归：新旧代码在**同一 100 题**上 diff。

基线 = `ab_rewrite_result.json` 的 off 臂（同题、同裁判 glm-4-flash ≥4=pass、
同默认配置 nol3j+gate+rewrite off），今天刚跑，可直接做前后对比。
判读：✅→非✅ = 回归（需排查）；非✅→✅ = 改善。

用法：uv run python qa/recall/_facts_fix_regress.py [--limit N]
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))

import _ab_rewrite as AB  # noqa: E402  复用同一套外部裁判（导入时已设 utf-8 stdout）
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = Path("qa/recall/facts_fix_regress_result.json")
PASS = 4


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))

    base = json.load(open("qa/recall/ab_rewrite_result.json", encoding="utf-8"))
    if args.limit:
        base = base[: args.limit]
    papers = load_papers()
    qmap: dict[str, tuple[str, dict]] = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qid = str(q.get("question_id") or "")
            if qid:
                qmap[qid] = (pid, q)

    done: dict[str, dict] = {}
    if OUT.exists():
        try:
            done = {r["qid"]: r for r in json.load(open(OUT, encoding="utf-8"))}
        except Exception:  # noqa: BLE001
            done = {}
    recs = list(done.values())
    t0 = time.time()
    for i, b in enumerate(base, 1):
        qid = b["qid"]
        if qid in done:
            continue
        pid, q = qmap[qid]
        llm.reset_usage()
        try:
            r = graph_ask(q.get("question", ""), f"qasper_{pid}.qpdf")
            ans = r.get("answer") or ""
            cites = list(r.get("cites") or [])
            dbg = r.get("debug") or {}
            adbg = dbg.get("answer") or {}
            level = adbg.get("level", "")
            n_facts = adbg.get("n_facts")
            action = (r.get("validator") or {}).get("action")
        except Exception as e:  # noqa: BLE001
            ans, cites, level, n_facts, action = f"ERR {e}", [], "err", None, None
        score = AB.ext_judge(q, ans, cites)
        rec = {"qid": qid, "grp": b["grp"],
               "base_score": b["off"]["score"], "base_pass": b["off"]["pass"],
               "new_score": score, "new_pass": score >= PASS,
               "level": level, "n_facts": n_facts, "action": action,
               "answer": ans[:400]}
        recs = [x for x in recs if x["qid"] != qid] + [rec]
        OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
        chg = ("=" if rec["base_pass"] == rec["new_pass"]
               else ("GAIN" if rec["new_pass"] else "LOSS"))
        print(f"[{i}/{len(base)}] {qid[:10]} {b['grp']:6} base={b['off']['score']} "
              f"new={score} facts={n_facts} {chg}  t={time.time()-t0:.0f}s", flush=True)

    def agg(sub, key):
        n = len(sub) or 1
        return sum(1 for r in sub if r[key]) / n

    L = ["# 两步法修复配对回归（同 100 题，同裁判 glm-4-flash ≥4=pass）", "",
         "> 基线 = `ab_rewrite_result.json` off 臂（今天早些时候、修复前）", ""]
    L.append("| 分组 | n | 修复前 pass | 修复后 pass | Δ |")
    L.append("|---|---|---|---|---|")
    for nm, f in (("all", lambda r: True), ("hard", lambda r: r["grp"] == "hard"),
                  ("normal", lambda r: r["grp"] == "normal")):
        sub = [r for r in recs if f(r)]
        if not sub:
            continue
        L.append(f"| {nm} | {len(sub)} | {sum(1 for r in sub if r['base_pass'])}/"
                 f"{len(sub)} = {agg(sub,'base_pass')*100:.0f}% | "
                 f"{sum(1 for r in sub if r['new_pass'])}/{len(sub)} = "
                 f"{agg(sub,'new_pass')*100:.0f}% | "
                 f"{sum(1 for r in sub if r['new_pass'])-sum(1 for r in sub if r['base_pass']):+d} |")
    losses = [r for r in recs if r["base_pass"] and not r["new_pass"]]
    gains = [r for r in recs if not r["base_pass"] and r["new_pass"]]
    L += ["", f"> 回归（✅→非✅）：**{len(losses)}** 题｜改善（非✅→✅）：**{len(gains)}** 题", ""]
    L.append("### 回归题（需排查）")
    for r in losses:
        L.append(f"- {r['qid'][:10]} [{r['grp']}] {r['base_score']}→{r['new_score']} "
                 f"(level={r['level']} action={r['action']})")
    L.append("")
    L.append("### 改善题")
    for r in gains:
        L.append(f"- {r['qid'][:10]} [{r['grp']}] {r['base_score']}→{r['new_score']} "
                 f"(level={r['level']} facts={r['n_facts']})")
    txt = "\n".join(L)
    Path("qa/recall/FACTS_FIX_REGRESSION_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
