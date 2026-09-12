"""A/B 状态速览（可反复调用）。"""
import json
from pathlib import Path

R = Path("qa/recall")
ROUNDS = {"R1": ("tblfix", "tblfix"), "R2": ("tblfix2", "tblfix2"), "R3": ("tblfix3", "tblfix3"),
          "R4": ("tblfix4", "tblfix4"), "R5": ("tblfix5", "tblfix5")}
pv, pn = [], []
for lab in ("R1", "R2", "R3", "R4", "R5"):
    row = [lab]
    for tag, pref in (("v1", ROUNDS[lab][0]), ("new", ROUNDS[lab][1])):
        p = R / f"{pref}_{tag}_20260912.json"
        if not p.exists():
            row.append(f"{tag}: 未开始")
            continue
        x = json.loads(p.read_text(encoding="utf-8"))
        ok = sum(1 for y in x if y["pass"])
        row.append(f"{tag}: {len(x)}/36 pass {ok}")
        if len(x) == 36:
            (pv if tag == "v1" else pn).append(ok)
    print("  " + " | ".join(row))
if pv and pn:
    print(f"\n  完整轮次 {len(pv)} 轮均值: v1 {sum(pv)/len(pv):.2f} | new {sum(pn)/len(pn):.2f} "
          f"| 差 {sum(pn)/len(pn)-sum(pv)/len(pv):+.2f}")
