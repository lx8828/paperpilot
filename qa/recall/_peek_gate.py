"""临时：读 gate 修复回路考场结果（fail 10 道 + HyGRAIL qs）。"""
from __future__ import annotations
import json
from collections import Counter


def main() -> int:
    d = json.load(open("qa/recall/ab_gate_fail_result.json", encoding="utf-8"))
    print("=== ab_gate_fail（历史 V fail 题 GATE on，n=%d）===" % len(d))
    print("action:", dict(Counter(r.get("action") for r in d)))
    print("judge pass:", sum(1 for r in d if r.get("pass")), "/", len(d))
    for r in d:
        print(f"  {r['qid'][:10]} act={r.get('action')} score={r.get('score')} "
              f"pass={r.get('pass')} lvl={r.get('level')} issues={r.get('issues')}")
    qs = json.load(open("qa/recall/ab_gate_qs_result.json", encoding="utf-8"))
    print()
    print("=== ab_gate_qs（HyGRAIL 需检索题，GATE on/off，n=%d）===" % len(qs))
    print("action:", dict(Counter(r["on"].get("action") for r in qs)))
    print(f"关键词 pass: off {sum(r['p_off'] for r in qs)}/{len(qs)}  "
          f"on {sum(r['p_on'] for r in qs)}/{len(qs)}")
    for r in qs:
        o, n = r.get("off") or {}, r.get("on") or {}
        print(f"  {r['qid'][:8]} {str(r.get('intent'))[:12]:12} off={o.get('pass')} "
              f"on={n.get('action')}/{n.get('pass')} issues={n.get('issues')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
