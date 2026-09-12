"""离线重判：用当前口径重算一份 QASPER run 的分桶（零 LLM、不重跑问答）。

用途：**打分口径变更后**，对已存盘的 run 记录重新分桶——记录里已含
`answer / cites / unanswerable / score / level`，所以不需要再调模型。

背景（2026-09-11）：无答案题新增"片段合格式"（`excerpt_ok`）。系统不给结论、
但明确说明"论文未提供该信息"并给出检索到的相关原文片段（带 cites）时，对用户是
**有效交付**（可自行判断），产品口径为合格，不应计 fail（裁判按"回答完整性"
打分会给 3 分，不到 pass 线）。判据的唯一来源是 `cli/run_qasper_eval.py`。

用法：
    uv run python qa/recall/_rescore_unans.py qa/qasper_run_20260911_005347.json
    uv run python qa/recall/_rescore_unans.py <run.json> --write   # 顺手重生成同名 summary
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "src"))

import run_qasper_eval as R  # noqa: E402  评测侧判据的唯一来源，避免判据漂移


def rescore(rec: dict) -> str:
    """按当前口径给出 status（与 `run_qasper_eval.run_one` 同逻辑）。"""
    st = rec.get("status")
    if st in ("error", "skipped-judge"):
        return st
    passed = rec.get("score", 0) >= R.PASS_SCORE
    if not rec.get("unanswerable"):
        if passed:
            return "pass"
        return "unknown_ok" if rec.get("level") == "unknown" else "fail"
    if passed:
        return "pass"
    if rec.get("level") == "unknown":
        return "honest_refuse"
    if R._absence_tone(rec.get("answer") or "") and rec.get("cites"):
        return "excerpt_ok"
    return "fail"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    if not args:
        print(__doc__)
        return 1
    path = Path(args[0])
    recs = json.loads(path.read_text(encoding="utf-8"))
    old = collections.Counter(r.get("status") for r in recs)
    new: collections.Counter[str] = collections.Counter()
    for r in recs:
        r["status_old"] = r.get("status")
        r["status"] = rescore(r)
        new[r["status"]] += 1

    print(f"文件: {path}｜题数 {len(recs)}")
    print(f"旧口径: {dict(old)}")
    print(f"新口径: {dict(new)}")
    un = [r for r in recs if r.get("unanswerable")]
    if un:
        u_pass = sum(1 for r in un if r["status"] == "pass")
        u_exc = sum(1 for r in un if r["status"] == "excerpt_ok")
        print(f"无答案题 {len(un)}：合格 {u_pass + u_exc} = "
              f"{(u_pass + u_exc) / len(un):.1%}（pass {u_pass} / excerpt_ok {u_exc}）")
    an = [r for r in recs if not r.get("unanswerable")]
    if an:
        a_pass = sum(1 for r in an if r["status"] == "pass")
        print(f"有答案题 {len(an)}：pass {a_pass} = {a_pass / len(an):.1%}")
    p = new.get("pass", 0)
    print(f"主通过率 {p}/{len(recs)} = {p / len(recs):.1%}")
    moved = [(r.get("paper"), (r.get("question") or "")[:44], r["status_old"], r["status"])
             for r in recs if r["status_old"] != r["status"]]
    print(f"状态变化 {len(moved)} 条：")
    for pid, q, a, b in moved:
        print(f"    {pid} | {q} | {a} -> {b}")

    if write:
        out = path.with_name(path.name.replace("run", "summary", 1)).with_suffix(".md")
        out.write_text(R.build_summary(recs), encoding="utf-8")
        print(f"已重写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
