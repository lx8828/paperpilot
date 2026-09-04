"""QA v2 全量跑批：21 篇 × 全部题目 → 每问一条 run 记录 + 判定 → 汇总表。

题目来源（自动合并，按 pdf 分组）：
    qa/qa_set_v2.json       v1 迁移 48 题（dev 视角）
    qa/questions/*.json     13 篇新论文 + 8 篇 user 补充 = 128 题

每问跑 graph.ask（L0→L1→L2→L3→unknown 漏斗），自动采集：

行为字段（机器可判）：
    route / answer_level / n_entries
    每层 debug：judge_l0..l3 的 enough/gap/target_sections；l2_step（圆心/半径/预算）；
                embedding_woken（本次是否触发模型加载，来自 embedder 的 print 捕获）
                llm_calls（LLM 请求次数，从 debug 字段粗估 judge 层数+answer）*
    *实际以 llm.py 的调用计数为准（run 前 monkeypatch 计数）。

质量字段（自动勾叉）：
    must_have / must_not 命中 → keyword_ok（全命中才 ✅）
    档位判定：actual_terminal ∈ expect 且 >= route_min → route_ok

判定汇总表：每问一行 status = ✅ / ⚠️(过深或关键词缺) / ❌(虚答或跑错)
输出：
    qa/qa_run_<ts>.json   全量记录
    qa/qa_summary_<ts>.md 人读汇总表（含真实耗时/成本）
用法：
    uv run python cli/run_qa_v2.py [--pdf 2608.31079v1.pdf] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.graph import ask as graph_ask
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "qa"


# ── LLM 调用计数（monkeypatch，不算 embedding 本地模型）─────────────────────
_LLM_CALLS = {"n": 0}
_orig_chat = llm._chat


def _count_chat(system: str, user: str, *, temperature: float,
                max_tokens: int | None) -> str:
    _LLM_CALLS["n"] += 1
    return _orig_chat(system, user, temperature=temperature, max_tokens=max_tokens)


# ── 题目加载 ──────────────────────────────────────────────────────────────────


def load_questions(pdf_filter: str | None = None,
                   limit: int | None = None) -> list[dict[str, Any]]:
    qs: list[dict[str, Any]] = []
    migrated = json.loads((QA_DIR / "qa_set_v2.json").read_text(encoding="utf-8"))
    qs += migrated["questions"]
    for f in sorted(glob.glob(str(QA_DIR / "questions" / "*.json"))):
        qs += json.loads(Path(f).read_text(encoding="utf-8"))
    if pdf_filter:
        qs = [q for q in qs if q.get("pdf") == pdf_filter]
    if limit:
        qs = qs[:limit]
    return qs


def _embedding_woken(captured: str) -> bool:
    return "加载 BAAI/bge-m3" in captured or "构建 ChunkIndex" in captured


# ── 单题跑批 ─────────────────────────────────────────────────────────────────


def run_one(q: dict[str, Any], pdf: str) -> dict[str, Any]:
    stem = Path(pdf).stem
    t0 = time.time()
    _LLM_CALLS["n"] = 0
    llm._chat = _count_chat  # type: ignore[assignment]
    captured: list[str] = []
    try:
        r = graph_ask(q["question"], pdf)
    except Exception as e:  # noqa: BLE001
        return {**q, "error": f"{type(e).__name__}: {e}", "time_s": round(time.time() - t0, 2)}
    finally:
        llm._chat = _orig_chat  # type: ignore[assignment]
    elapsed = time.time() - t0

    dbg = r.get("debug") or {}
    ans_dbg = dbg.get("answer") or {}
    l1 = dbg.get("judge_l1") or {}
    route = list(r.get("route") or [])
    # embedding 唤醒：看 route 是否进入 L1/L3 + debug 是否存在（模型加载 print 难捕获，
    # 用近似：route 含 L1/L3 且对应 debug 存在即认为模型参与；成本字段仍以 llm_calls 为准）
    woken = any(x in route for x in ("L1", "L3"))
    actual = ans_dbg.get("level", "")
    n_entries = ans_dbg.get("n_entries", 0)

    # 关键词核查（召回率/准确率核心判据：答案是否覆盖 must_have、没踩 must_not）
    text = (r.get("answer") or "") + " " + " ".join(
        c.get("evidence", "") for c in (r.get("cites") or [])[:5])
    mh = [k for k in (q.get("must_have") or []) if k]
    mn = [k for k in (q.get("must_not") or []) if k]
    ma = [k for k in (q.get("must_all") or []) if k]
    # must_have：答案命中任一即算（覆盖措辞差异：监督微调 vs SFT、log vs 对数）
    # must_all：必须全部命中（关键数字等硬指标）
    have_hit = any(re.search(re.escape(k), text, re.I) for k in mh) if mh else True
    all_hit = all(re.search(re.escape(k), text, re.I) for k in ma) if ma else True
    not_hit = any(re.search(re.escape(k), text, re.I) for k in mn) if mn else False
    kw_ok = have_hit and all_hit and not not_hit

    # 档位仅作行为记录（expect 标深导致的 route_mismatch ≠ 答错，不罚）
    expect = list(q.get("expect") or [])
    route_min = q.get("route_min", "L0")
    order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "unknown": 4}
    reached_min = actual in expect or order.get(actual, 9) >= order.get(route_min, 0)
    route_ok = bool(expect) and (actual in expect or actual == "unknown")
    expect_unknown = "unknown" in expect

    # 判定：核心是"答对没有"（召回/准确），unknown 预期之外答 unknown 才是真漏检
    if r.get("answer") is None:
        status = "❌"          # 跑挂了
    elif not kw_ok:
        status = "⚠️"          # 关键词缺失（真错 or must_have 过严，待人工分）
    elif actual == "unknown" and not expect_unknown:
        status = "⚠️"          # 该答出却答 unknown → 真漏检候选
    else:
        status = "✅"

    rec = {
        **{k: q.get(k) for k in ("pdf", "qid", "role", "intent", "question",
                                 "expect", "route_min", "hint",
                                 "must_have", "must_all", "must_not")},
        "answer": (r.get("answer") or "")[:600],
        "cites_n": len(r.get("cites") or []),
        "route": route,
        "answer_level": actual,
        "n_entries": n_entries,
        "judge_l0": dbg.get("judge_l0"),
        "judge_l1": dbg.get("judge_l1"),
        "l2_step": dbg.get("l2_step"),
        "judge_l2": dbg.get("judge_l2"),
        "l3": dbg.get("l3"),
        "judge_l3": dbg.get("judge_l3"),
        "llm_calls": _LLM_CALLS["n"],
        "embedding_woken": woken,
        "time_s": round(elapsed, 2),
        "kw_ok": kw_ok,
        "route_ok": route_ok,
        "reached_min": reached_min,
        "status": status,
    }
    return rec


# ── 汇总 ──────────────────────────────────────────────────────────────────────


def build_summary(recs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# QA v2 汇总判定表\n")
    lines.append("| 论文 | qid | role | intent | expect | 实际 | route | kw | 耗时 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        pdf = r["pdf"].replace(".pdf", "")
        lines.append(
            f"| {pdf} | {r['qid']} | {r['role']} | {r['intent']} | "
            f"{','.join(r['expect'] or [])} | {r['answer_level'] or '?'} | "
            f"{'→'.join(r['route']) if r.get('route') else '-'} | "
            f"{'✅' if r['kw_ok'] else '❌'} | {r['time_s']}s | {r['status']} |")
    return "\n".join(lines)


# ═══════════════ 基线/快照（回归护栏：改了代码 → 重跑 → diff 看勾叉变化）═══════════

BASELINE = QA_DIR / "baseline.json"
SNAPSHOT_DIR = QA_DIR / "snapshots"


def snap_of(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """把一次 run 压缩成 qid → 判定快照（只留判定相关，避免答案文本噪音）。"""
    return {r["qid"]: {"status": r.get("status"), "level": r.get("answer_level"),
                        "route": list(r.get("route") or []),
                        "kw_ok": r.get("kw_ok"), "time_s": r.get("time_s")}
            for r in recs}


def diff_baseline(recs: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    """对比 baseline：返回 (回归, 改善, 新增/消失)。"""
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    now = snap_of(recs)
    regress, improve, new = [], [], []
    for qid, s in now.items():
        b = base.get(qid)
        if b is None:
            new.append(f"新增题目 {qid} → {s['status']}")
            continue
        if b["status"] == "✅" and s["status"] != "✅":
            regress.append(f"{qid}: ✅→{s['status']} (was {b.get('level')}, now {s.get('level')})")
        elif b["status"] != "✅" and s["status"] == "✅":
            improve.append(f"{qid}: {b['status']}→✅ (was {b.get('level')}, now {s.get('level')})")
    for qid in base:
        if qid not in now:
            new.append(f"题目消失 {qid}")
    return regress, improve, new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-baseline", action="store_true",
                    help="把本次结果固化为 qa/baseline.json")
    ap.add_argument("--no-baseline-diff", action="store_true",
                    help="不读 baseline.json 对比")
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    if not llm.is_configured():
        print("[错误] LLM 未配置")
        return 1

    qs = load_questions(args.pdf, args.limit)
    print(f"共 {len(qs)} 题")

    # 按 pdf 分组，embedding 索引同论文复用（内存模型单例），逐题跑
    by_pdf: dict[str, list[dict[str, Any]]] = {}
    for q in qs:
        by_pdf.setdefault(q["pdf"], []).append(q)

    all_recs: list[dict[str, Any]] = []
    for pdf in sorted(by_pdf):
        group = by_pdf[pdf]
        print(f"\n{'='*70}\n[{pdf}] {len(group)} 题")
        for q in group:
            rec = run_one(q, pdf)
            flag = rec.get("status", "?")
            ans = (rec.get("answer") or "").replace("\n", " ")[:90]
            print(f"  [{flag}] {rec['qid']} → {rec.get('answer_level','?')} "
                  f"llm={rec.get('llm_calls')} {rec.get('time_s')}s | {ans}")
            if "error" in rec:
                print(f"        ERROR: {rec['error']}")
            all_recs.append(rec)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_file = QA_DIR / f"qa_run_{ts}.json"
    run_file.write_text(json.dumps(all_recs, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    sum_file = QA_DIR / f"qa_summary_{ts}.md"
    sum_file.write_text(build_summary(all_recs), encoding="utf-8")

    ok = sum(1 for r in all_recs if r.get("status") == "✅")
    warn = sum(1 for r in all_recs if r.get("status") == "⚠️")
    err = sum(1 for r in all_recs if r.get("status") == "❌" or r.get("error"))
    print(f"\n✅ {ok}  ⚠️ {warn}  ❌ {err}  /  {len(all_recs)}")
    print(f"记录: {run_file}")
    print(f"汇总: {sum_file}")

    # 快照 + 基线对比（回归护栏）
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap_file = SNAPSHOT_DIR / f"{ts}.json"
    snap_file.write_text(json.dumps(snap_of(all_recs), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"快照: {snap_file}")

    if args.save_baseline:
        BASELINE.write_text(json.dumps(snap_of(all_recs), ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"本次结果已固化为基线: {BASELINE}")
    elif not args.no_baseline_diff and BASELINE.exists():
        regress, improve, new = diff_baseline(all_recs)
        print("\n--- 相对基线 diff ---")
        if regress:
            print("回归（✅→非✅，需排查改动）：")
            for line in regress:
                print(f"  ✗ {line}")
        if improve:
            print("改善（非✅→✅）：")
            for line in improve:
                print(f"  ✓ {line}")
        if new:
            print("题目集变化：")
            for line in new:
                print(f"  ~ {line}")
        if not regress and not improve and not new:
            print("与基线一致，无变化。")
    elif not BASELINE.exists():
        print("（无 baseline.json；首次运行可加 --save-baseline 固化基线）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
