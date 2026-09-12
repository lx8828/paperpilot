"""逐题 Δ（跨轮均值）+ 输赢分布：看"new 更差"的题落在哪一组。"""
import json
from pathlib import Path

R = Path("qa/recall")
F = [("R1", "tblfix_v1_20260912.json", "tblfix_new_20260912.json"),
     ("R2", "tblfix2_v1_20260912.json", "tblfix2_new_20260912.json"),
     ("R3", "tblfix3_v1_20260912.json", "tblfix3_new_20260912.json"),
     ("R4", "tblfix4_v1_20260912.json", "tblfix4_new_20260912.json"),
     ("R5", "tblfix5_v1_20260912.json", "tblfix5_new_20260912.json")]
D = []
for lab, fv, fn in F:
    pv, pn = R / fv, R / fn
    if pv.exists() and pn.exists():
        D.append((lab,
                  {x["qid"]: x for x in json.loads(pv.read_text(encoding="utf-8"))},
                  {x["qid"]: x for x in json.loads(pn.read_text(encoding="utf-8"))}))
print("已完成轮次:", [x[0] for x in D])


def has_tbl(rec):
    return any(str(c).startswith("xtbl-") for c in (rec.get("entry_ids") or []))


rows = []
for q in D[0][1]:
    a = sum(1 for _, v, _ in D if v[q]["pass"]) / len(D)
    b = sum(1 for _, _, n in D if n[q]["pass"]) / len(D)
    pids = next(v[q]["pid"] for _, v, _ in D)
    rows.append((b - a, q[:8], pids, a, b, has_tbl(D[0][2][q]),
                 next(n[q].get("question", "") for _, _, n in D)[:44]))
rows.sort()
print(f"\n{'Δ':>5}  {'qid':<9} {'论文':<14} {'v1':>4} {'new':>4}  有表  问题")
for d, q, pid, a, b, t, qq in rows:
    mark = "  <<<" if d < 0 else ("  +++" if d > 0 else "")
    print(f"{d:>+5.2f}  {q:<9} {pid:<14} {a:>4.2f} {b:>4.2f}  {'Y' if t else '-'}   {qq}{mark}")
n = len(rows)
print(f"\n汇总：更好 {sum(1 for r in rows if r[0]>0)} | 相同 {sum(1 for r in rows if r[0]==0)} "
      f"| 更差 {sum(1 for r in rows if r[0]<0)}")
print(f"净胜题里 有表: {sum(1 for r in rows if r[0]>0 and r[5])} | 无表: {sum(1 for r in rows if r[0]>0 and not r[5])}")
print(f"净负题里 有表: {sum(1 for r in rows if r[0]<0 and r[5])} | 无表: {sum(1 for r in rows if r[0]<0 and not r[5])}")
