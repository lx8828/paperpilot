"""临时：cap=1 A/B 汇总明细。"""
from __future__ import annotations
import json


def main() -> int:
    d = json.load(open("qa/recall/_cap_result.json", encoding="utf-8"))
    d = [r for r in d if not r.get("err")]
    new = sum(1 for r in d if r["pass"]); old = sum(1 for r in d if r["baseV_pass"])
    gain = [r for r in d if r["pass"] and not r["baseV_pass"]]
    loss = [r for r in d if not r["pass"] and r["baseV_pass"]]
    print(f"cap=1 pass {new}/{len(d)}  vs  baseV {old}/{len(d)}  Δ{new-old:+d}  "
          f"(gain {len(gain)} / loss {len(loss)})")
    print()
    print("--- gain（baseV 挂、cap 过）---")
    for r in sorted(gain, key=lambda x: -x["score"]):
        print(f"  {r['qid'][:10]} {r['grp']:6} {r['score']} {r['level']}")
    print("--- loss（baseV 过、cap 挂）---")
    for r in sorted(loss, key=lambda x: x["score"]):
        print(f"  {r['qid'][:10]} {r['grp']:6} {r['score']} {r['level']}")
    print()
    for g in ("hard", "normal"):
        sub = [r for r in d if r["grp"] == g]
        if sub:
            nw = sum(1 for r in sub if r["pass"]); od = sum(1 for r in sub if r["baseV_pass"])
            print(f"{g:6}: cap {nw}/{len(sub)}  baseV {od}/{len(sub)}  Δ{nw-od:+d}")
    print()
    lv = {}
    for r in d:
        lv.setdefault(r["level"], [0, 0])
        lv[r["level"]][0] += r["pass"]; lv[r["level"]][1] += 1
    print("level:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(lv.items())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
