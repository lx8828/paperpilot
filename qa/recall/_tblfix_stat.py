"""P1+P3 配对 A/B 的统计检验（自动发现轮次；配对 t + 逐题符号检验）。"""
import json
import math
from pathlib import Path

R = Path("qa/recall")
pairs = {}
for f in sorted(R.glob("tblfix*_20260912.json")):
    pre, arm, _ = f.stem.rsplit("_", 2)
    if arm in ("v1", "new"):
        pairs.setdefault(pre, {})[arm] = {x["qid"]: x for x in json.loads(f.read_text(encoding="utf-8"))}
d = [(pre, v["v1"], v["new"]) for pre, v in sorted(pairs.items())
     if "v1" in v and "new" in v and len(v["v1"]) == 36 and len(v["new"]) == 36]

dv = [(sum(1 for x in v.values() if x["pass"]), sum(1 for x in n.values() if x["pass"]))
      for _, v, n in d]
print(f"轮数 {len(dv)} | 逐轮 (v1, new): {dv}")
diffs = [b - a for a, b in dv]
m = sum(diffs) / len(diffs)
sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
se = sd / math.sqrt(len(diffs))
vn = [a for a, _ in dv]
nn = [b for _, b in dv]
print(f"v1 均值 {sum(vn)/len(vn):.2f} (轮间 sd {math.sqrt(sum((x-sum(vn)/len(vn))**2 for x in vn)/(len(vn)-1)):.2f}) "
      f"| new 均值 {sum(nn)/len(nn):.2f}")
print(f"均值差 {m:+.2f} | 差值 sd {sd:.2f} | SE {se:.2f} | **t = {m/se:.2f} (df={len(diffs)-1})** "
      f"→ 双尾 p ≈ {2*(1-_t_cdf(abs(m/se), len(diffs)-1)) if False else ''}")

# 用简单近似给出 p（t 分布双尾，正态近似；df=7 用 t 表临界值提示）
crit = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
df_ = len(diffs) - 1
print(f"  参考：df={df_} 时 t=0.05 双尾临界值 ≈ {crit.get(df_, 2.0)} → "
      f"{'达到显著' if abs(m/se) > crit.get(df_, 2.0) else '未达显著'}")

qids = set(d[0][1]) & set(d[0][2])
w = l = 0
for q in qids:
    a = sum(1 for _, v, _ in d if v[q]["pass"])
    b = sum(1 for _, _, n in d if n[q]["pass"])
    w += b > a
    l += b < a
nd = w + l
p_sign = sum(math.comb(nd, k) for k in range(0, min(w, l) + 1)) * 2 / (2 ** nd) if nd else 1.0
print(f"逐题配对（{len(qids)} 题 × {len(d)} 轮）：new 更好 {w} | 相同 {len(qids)-w-l} | 更差 {l} "
      f"→ 符号检验 p = {min(p_sign, 1.0):.3f}")

for lab, idx in (("v1", 1), ("new", 2)):
    tot = c = 0
    for _, v, n in d:
        for rec in (v if idx == 1 else n).values():
            if any(str(x).startswith("xtbl-") for x in (rec.get("entry_ids") or [])):
                tot += 1
                c += any(str(y.get("ref", "")).split("#")[-1].startswith("xtbl-")
                         for y in (rec.get("cites") or []))
    print(f"表块引用率 {lab}: {c}/{tot} = {c/max(tot,1):.0%}")
