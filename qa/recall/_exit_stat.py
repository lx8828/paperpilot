"""Exit 判据量化：L2 值不值得修（决定"修 L2"还是"砍 L2"）。

口径（离线，零 LLM 成本）：
  样本 = ab_noL2 50 题（hard25 + normal25，同题 A=现状含L2 / B=直接L3）。
  ① L2 净贡献（不可替代）   = A pass ∧ B fail —— 砍 L2 会直接丢的题。
  ② L2 白参与（不可恢复上界）= A fail 且 A 臂 route 经过 L2 的题。
  ③ 按用户三分口径（仅 join 上 diag_l1 的题）：
     不可恢复 = 圆心不准(gold_top∉targets) ∧ judge_l2 未拦截(A最终在 L2 作答=判够即放行)
                ∧ 最终答错(A fail)。
  route 重建：judge_l1 enough → 进 L2（diag_l1 的 enough 字段）；level=='L2' → judge_l2
  判够放行并作答；level=='L3' 且 enough=True → 被 judge_l2 拦截降级（已拦，不算未拦截）。
"""
from __future__ import annotations
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ab = json.load(open("qa/recall/ab_noL2_result.json", encoding="utf-8"))
diag = {d["qid"]: d for d in json.load(open("qa/recall/diag_l1_result.json", encoding="utf-8"))}
items = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))["items"]
qid2grp = {it["qid"]: it["group"] for it in items}


def pass_A(r):
    return r["A"]["pass"]


def pass_B(r):
    return r["B"]["pass"]


rows = []
for r in ab:
    qid = r["qid"]
    A, B = r["A"], r["B"]
    # A 臂 route 是否经过 L2：level==L2 必经过；level==L3 需 enough=True（judge_l1 放行进 L2 后被拦）
    via_l2 = A["level"] == "L2" or (A["level"] == "L3" and qid in diag and diag[qid]["enough"])
    rows.append({**r, "grp": qid2grp.get(qid, "?"), "via_l2": via_l2,
                 "in_diag": qid in diag})

total = len(rows)
l2_contribution = [r for r in rows if pass_A(r) and not pass_B(r)]
l2_waste = [r for r in rows if not pass_A(r) and r["via_l2"]]
l2_never = [r for r in rows if not pass_A(r) and not r["via_l2"]]
_debug = [(r["qid"][:10], r["A"]["level"], repr(r["A"]["level"]),
           pass_A(r), r["A"]["level"] == "L2", r["via_l2"])
          for r in rows if not pass_A(r) and r["A"]["level"] in ("L2", "L3", "L0")]
Apass = sum(1 for r in rows if pass_A(r))
Bpass = sum(1 for r in rows if pass_B(r))

print("=" * 72)
print(f"样本 n={total}  (hard {sum(1 for r in rows if r['grp']=='hard')} / "
      f"normal {sum(1 for r in rows if r['grp']=='normal')})")
print(f"A(现状含L2) pass = {Apass}   B(砍L2) pass = {Bpass}   Δ = {Bpass-Apass}")
print(f"diag_l1 join 覆盖: {sum(1 for r in rows if r['in_diag'])}/{total}")
print("-" * 72)
print("① L2 净贡献（A pass ∧ B fail，砍 L2 会直接丢）:", len(l2_contribution))
for r in l2_contribution:
    print(f"    {r['qid'][:12]} {r['grp']:6} A={r['A']['level']} "
          f"(targets含gold: {diag.get(r['qid'],{}).get('gold_top') in diag.get(r['qid'],{}).get('targets',[])} )")
print("-" * 72)
print("② L2 白参与（A fail 且 route 经过 L2，修 L2 的理论回收面）:", len(l2_waste))
for r in l2_waste:
    d = diag.get(r["qid"])
    tag = ""
    if d:
        tag = f" 圆心准={d['gold_top'] in (d.get('targets') or [])} enough={d['enough']}"
    print(f"    {r['qid'][:12]} {r['grp']:6} A level={r['A']['level']:<3} calls={r['A']['calls']}{tag}")
print("-" * 72)
print("A fail 且未进 L2（与 L2 无关，修 L2 也救不了）:", len(l2_never))
for r in l2_never:
    print(f"    {r['qid'][:12]} {r['grp']:6} A level={r['A']['level']}")
print("-- debug (not pass, level in L0/L2/L3): qid level repr pass isL2 via_l2 --")
for qid, lv, rlv, pa, isl2, via in _debug:
    print(f"    {qid} level={lv} repr={rlv} pass={pa} isL2={isl2} via_l2={via}")
print("-" * 72)
print("③ 三分口径不可恢复（圆心不准 ∧ judge_l2 放行作答 ∧ 答错），仅 join 上的题：")
cnt3 = 0
for r in rows:
    d = diag.get(r["qid"])
    if not d or pass_A(r):
        continue
    if A_pass_cond := (r["A"]["level"] == "L2"):
        if d["gold_top"] not in (d.get("targets") or []):
            cnt3 += 1
            print(f"    {r['qid'][:12]} {r['grp']:6} level=L2 gold_top={d['gold_top'][:10] if d['gold_top'] else '∅'} "
                  f"targets={[t[:8] for t in (d.get('targets') or [])][:4]}")
print(f"    => 满足三分条件共 {cnt3} 题（圆心不准样本内的硬性 L2 漏错）")
print("=" * 72)
print("结论判读：① 小 => 砍 L2 损失有限；② 大且其中圆心准 => 问题在 judge/answer 而非圆心，修 judge 更值；")
print("         ② 大且圆心不准 => 支持策略1/4（修圆心）；② 小 => 修 L2 天花板低，倾向砍。")
