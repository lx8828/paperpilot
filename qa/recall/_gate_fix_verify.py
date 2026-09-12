"""验证闸门数字误杀修复的回收：对那 10 道兜底题按**真实评测口径**重跑问答 + 裁判。

零结构改动，只复用 `cli/run_qasper_eval.py` 的 `run_one`（含新分桶口径）。
对比"修复前记录的分"与"修复后重跑的分"，量化回收。

用法：
    uv run python qa/recall/_gate_fix_verify.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import run_qasper_eval as R  # noqa: E402
from paperpilot.components.validator import FALLBACK_MSG  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

RUN = "qa/qasper_run_20260911_005347.json"


def main() -> int:
    llm._load_dotenv(str(ROOT))
    recs = json.loads((ROOT / RUN).read_text(encoding="utf-8"))
    gated = [r for r in recs if FALLBACK_MSG[:12] in (r.get("answer") or "")]
    papers = load_papers()

    L: list[str] = [f"# 闸门数字误杀修复 · 回收验证（{time.strftime('%Y-%m-%d %H:%M')}）", "",
                    f"> 对象：`{RUN}` 里被闸门兜底的 {len(gated)} 道｜同裁判（glm-4-flash）同口径", "",
                    "| 论文 | 问题 | 修复前 | **修复后** | 变化 |", "|---|---|---|---|---|"]
    n_gain = n_lost = 0
    old_sum = new_sum = 0
    old_pass = sum(1 for x in gated if x.get("score", 0) >= R.PASS_SCORE)
    new_pass = 0
    for i, r in enumerate(gated, 1):
        pid = r["paper"].replace("qasper_", "").replace(".qpdf", "")
        q = next((qq for qq in papers[pid].get("qas") or []
                  if str(qq.get("question_id")) == r["qid"]), None)
        if q is None:
            L.append(f"| {pid} | {r['question'][:36]} | {r['score']} | （题面缺失） | — |")
            continue
        new = R.run_one(q, r["paper"])
        old_sum += r["score"]
        new_sum += new["score"]
        if new["score"] >= R.PASS_SCORE:
            new_pass += 1
        gain = new["score"] - r["score"]
        if gain > 0:
            n_gain += 1
        elif gain < 0:
            n_lost += 1
        L.append(f"| {pid} | {r['question'][:36]} | {r['score']} | **{new['score']}** "
                 f"({new['status']}) | {gain:+d} |")
        print(f"[{i}/{len(gated)}] {pid} {r['score']} → {new['score']} ({new['status']})", flush=True)

    L += ["", "---", "",
          f"- **pass（≥4）**：{old_pass}/{len(gated)} → **{new_pass}/{len(gated)}**（+{new_pass - old_pass}）",
          f"- 合计分：{old_sum} → **{new_sum}**（均 {old_sum / len(gated):.2f} → {new_sum / len(gated):.2f}）",
          f"- 翻正 **{n_gain}** 道｜变差 **{n_lost}** 道",
          "",
          f"> 折算到 QASPER 233 主通过率：**+{new_pass - old_pass}/233 = "
          f"{(new_pass - old_pass) / 233 * 100:.1f}pt**"
          f"（修复前 193/233 = 82.8% → 预计 **{(193 + new_pass - old_pass) / 233 * 100:.1f}%**，"
          f"另需扣除题面缺失/波动项，属**上限估计**）"]
    txt = "\n".join(L)
    (ROOT / "qa/recall/GATE_FIX_VERIFY_20260911.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
