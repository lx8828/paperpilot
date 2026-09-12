"""复现被输出闸门兜底的题，打印闸门完整判定，判断是"真拦住"还是"误杀"。

背景（2026-09-11 复盘）：QASPER 233 的 40 道失败里有 **10 道是 `validator.gate` 直接兜底**
（answer = FALLBACK_MSG、cites 清空），其中 **9 道 gold 明确** → 疑似误杀。
参照 MinerU 的同类发现（§5.7：原文自带引用 `[29]` / 派生数 `30.2`，修两条有界豁免后
26/30 → 30/30），这里做同样的诊断。

判定依据：`validator.issues[]` 每项含 `sev / type / sentence / detail` ——
`sentence` 是触发判定的那句答案，`detail` 是判由，足以区分误杀与真拦。

用法：
    uv run python qa/recall/_gate_probe_qasper.py --run qa/qasper_run_20260911_005347.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from paperpilot.components import validator as _v  # noqa: E402
from paperpilot.components.validator import FALLBACK_MSG  # noqa: E402
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

# ── 草稿拦截器 ────────────────────────────────────────────────────────────────
# gate 会把不过检的答案替换成兜底话术，草稿就丢了。这里在它被替换前把入参记下来
# （纯内存拦截，不额外调用模型、不改变行为）。
_DRAFT: dict[str, Any] = {}
_ORIG_GATE = _v.gate


def _spy_gate(question, answer, cites, **kw):
    _DRAFT.clear()
    _DRAFT["answer"] = answer
    _DRAFT["cites"] = list(cites or [])
    return _ORIG_GATE(question, answer, cites, **kw)


_v.gate = _spy_gate


def _nums(text: str) -> list[float]:
    """validator 内部提取的"显著数字"（与判据同源，便于定位数字从哪来）。"""
    try:
        return sorted(_v._canonical_nums(text or ""))
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="qa/qasper_run_20260911_005347.json")
    ap.add_argument("--out", default="qa/recall/GATE_PROBE_QASPER_20260911.md")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    recs = json.loads((ROOT / args.run).read_text(encoding="utf-8"))
    gated = [r for r in recs if FALLBACK_MSG[:12] in (r.get("answer") or "")]
    if args.limit:
        gated = gated[: args.limit]

    L: list[str] = [f"# 闸门兜底题诊断（{time.strftime('%Y-%m-%d %H:%M')}）", "",
                    f"> 来源 `{args.run}`｜闸门兜底 {len(gated)} 道｜逐题复跑并打印判定", ""]
    by_type: dict[str, int] = {}
    by_action: dict[str, int] = {}
    stats = {"calls": 0, "prompt": 0, "completion": 0}

    for i, r in enumerate(gated, 1):
        llm.reset_usage()
        t0 = time.time()
        pid = r["paper"].replace("qasper_", "").replace(".qpdf", "")
        L.append("=" * 96)
        L.append(f"## {i}. {pid}｜{r['question']}")
        L.append(f"- gold: `{(r.get('gold') or '（空 / 无答案）')[:110]}`")
        L.append(f"- 原判: score={r.get('score')} status={r.get('status')} "
                 f"unanswerable={r.get('unanswerable')}")
        try:
            out = graph_ask(r["question"], r["paper"])
        except Exception as e:  # noqa: BLE001
            L.append(f"- **复跑异常**: {type(e).__name__}: {e}")
            continue
        u = llm.usage_stats()
        for k in stats:
            stats[k] += u.get(k, 0)
        v = out.get("validator") or {}
        ans_dbg = (out.get("debug") or {}).get("answer") or {}
        issues = v.get("issues") or []
        by_action[v.get("action")] = by_action.get(v.get("action"), 0) + 1
        L.append(f"- 复跑: action=**{v.get('action')}** level={ans_dbg.get('level')} "
                 f"n_entries={ans_dbg.get('n_entries')} cites={len(out.get('cites') or [])} "
                 f"t={time.time() - t0:.0f}s")
        if not issues:
            L.append("- **issues: 无**（本次未触发 → 说明是 LLM 波动/一次性误判）")
        for it in issues:
            by_type[it.get("type")] = by_type.get(it.get("type"), 0) + 1
            L.append(f"- 触发: `sev={it.get('sev')} type={it.get('type')}`")
            L.append(f"  - 句: {str(it.get('sentence'))[:220]}")
            L.append(f"  - 由: {str(it.get('detail'))[:320]}")
        L.append(f"- 闸门后答案: {(out.get('answer') or '')[:260]}")
        draft = _DRAFT.get("answer") or ""
        if draft:
            L.append("")
            L.append(f"- **闸门前草稿**（{len(draft)} 字）: {draft[:600].replace(chr(10), ' ')}")
            L.append(f"- 草稿里 validator 提取到的数字: "
                     f"`{[_v._fmt_num(x) for x in _nums(draft)]}`")
        L.append("")

    L += ["", "## 汇总", "", f"- 判定类型统计: `{by_type}`",
          f"- 复跑 action 分布: `{by_action}`",
          f"- 本轮成本: calls={stats['calls']} prompt={stats['prompt']:,} "
          f"completion={stats['completion']:,} "
          f"≈ {stats['prompt'] / 1e6 * 2 + stats['completion'] / 1e6 * 8:.2f} 元（高峰档）"]
    txt = "\n".join(L)
    (ROOT / args.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\n已写 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
