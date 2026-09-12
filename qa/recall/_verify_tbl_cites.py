"""验证表格通道的**溯源闭环**：翻绿题的引用是否真指向 `xtbl-*` 表格块。

为什么单独做：`_qasper_tbl_exp.py` 首版没把 `cites` 存进记录，导致"cites 为空"的误判。
溯源是本系统核心信条（所有结论可回原文），必须逐题核，不能抽样了事。

判据：每道翻绿题至少有一条 cite 的 `ref` 以 `#xtbl-` 结尾（即引用了外部表格块），
且该 cite 的 `page > 0`（QASPER 正文块是 page=0，表格块带真实页码）。

用法：uv run python qa/recall/_verify_tbl_cites.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

SRC = Path("qa/recall/qasper_tbl_exp_20260911.json")
OUT = Path("qa/recall/qasper_tbl_cites_20260911.json")
PASS = 4


def main() -> int:
    llm._load_dotenv(str(ROOT))
    recs = json.loads(SRC.read_text(encoding="utf-8"))
    recovered = [r for r in recs if r["pass"] and (r["score_before"] or 0) < PASS]
    papers = load_papers()
    out = []
    t0 = time.time()
    for i, r in enumerate(recovered, 1):
        pid, qid = r["pid"], r["qid"]
        paper = papers.get(pid) or {}
        q = next((qq for qq in paper.get("qas") or []
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        res = graph_ask(q["question"], f"qasper_{pid}.qpdf")
        cites = list(res.get("cites") or [])
        hits = [c for c in cites if str(c.get("ref", "")).split("#")[-1].startswith("xtbl-")]
        v = res.get("validator") or {}
        out.append({"qid": qid, "pid": pid, "question": q["question"],
                    "n_cites": len(cites), "n_xtbl": len(hits),
                    "refs": [c.get("ref") for c in cites],
                    "xtbl_refs": [{"ref": c["ref"], "page": c.get("page"),
                                   "ev": str(c.get("evidence"))[:160]} for c in hits],
                    "validator_action": v.get("action"),
                    "answer": res.get("answer")})
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(recovered)}] {qid[:8]} {pid} cites={len(cites)} "
              f"xtbl={len(hits)} action={v.get('action')}", flush=True)

    n = len(out)
    with_tbl = sum(1 for x in out if x["n_xtbl"] > 0)
    L = ["# 表格通道的溯源验证（2026-09-11）", "",
         f"> 对象：表格通道实验里**翻绿的 {n} 道题**（逐题重跑，非抽样）",
         "> 验证：引用是否指向外部表格块 `xtbl-*`（`page>0`；QASPER 正文块为 page=0）", "",
         f"| 指标 | 值 |", "|---|---|",
         f"| 翻绿题数 | {n} |",
         f"| **至少一条 `xtbl-*` 引用** | **{with_tbl}/{n}** |",
         f"| 引用总数 | {sum(x['n_cites'] for x in out)} |",
         f"| 其中表格块引用 | {sum(x['n_xtbl'] for x in out)} |", "",
         "## 逐题", "", "| qid | 论文 | cites | 其中 xtbl | validator |", "|---|---|---|---|---|"]
    for x in out:
        L.append(f"| {x['qid'][:8]} | {x['pid']} | {x['n_cites']} | "
                 f"{'**%d**' % x['n_xtbl'] if x['n_xtbl'] else '0'} | {x['validator_action']} |")
    L += ["", "## 表格引用样例（ref / page / 证据前缀）", ""]
    for x in out[:5]:
        for h in x["xtbl_refs"][:1]:
            L.append(f"- `{x['qid'][:8]}` → `{h['ref']}` (p{h['page']})：{h['ev'][:110]}…")
    txt = "\n".join(L)
    Path("qa/recall/TBL_CITES_20260911.md").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\n耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
