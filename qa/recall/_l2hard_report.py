import json
recs = json.load(open("qa/recall/ab_l2hard_result.json", encoding="utf-8"))
n = len(recs)


def s(recs):
    b = sum(1 for r in recs if r["base"]["pass"])
    h = sum(1 for r in recs if r["hard"]["pass"])
    bp = sum(r["base"]["prompt"] for r in recs) / len(recs)
    hp = sum(r["hard"]["prompt"] for r in recs) / len(recs)
    bc = sum(r["base"]["calls"] for r in recs) / len(recs)
    hc = sum(r["hard"]["calls"] for r in recs) / len(recs)
    return b, h, bp, hp, bc, hc


print(f"总样本 n={n}")
b, h, bp, hp, bc, hc = s(recs)
print(f"全体: base(现状) pass {b}  hard(策略③) pass {h}  "
      f"prompt/题 base={bp:.0f} hard={hp:.0f}  calls/题 base={bc:.1f} hard={hc:.1f}")
ov = sum(1 for r in recs if r["hard"]["hard_override"] == 1)
print(f"hard 模式触发硬拦截题数: {ov}/{n}")

# 按历史 A/B 交叉桶（prevA/prevB 来自 l2target 实验）切片
sub = {
    "净亏(A挂B过,策略目标)": [r for r in recs if not r["prevA"] and r["prevB"]],
    "净赢(A过B挂,别误伤)": [r for r in recs if r["prevA"] and not r["prevB"]],
    "双过(省token面,别误伤)": [r for r in recs if r["prevA"] and r["prevB"]],
    "双挂": [r for r in recs if not r["prevA"] and not r["prevB"]],
}
for name, subr in sub.items():
    if not subr:
        continue
    b, h, bp, hp, bc, hc = s(subr)
    saved = sum(1 for r in subr if not r["base"]["pass"] and r["hard"]["pass"])
    hurt = sum(1 for r in subr if r["base"]["pass"] and not r["hard"]["pass"])
    ovs = sum(1 for r in subr if r["hard"]["hard_override"] == 1)
    print(f"  {name}: n={len(subr)} base pass {b} hard pass {h} "
          f"| 救回 {saved} / 误伤 {hurt} | 硬拦 {ovs} | prompt/题 {bp:.0f}→{hp:.0f}")

print("\n净亏 10 题逐题（历史 vs base vs hard）：")
for r in sub["净亏(A挂B过,策略目标)"]:
    print("  {} {} hist(A={}/B={}) base={}{} hard={}{} 拦={}".format(
        r["qid"][:12], r["grp"], "过" if r["prevA"] else "挂", "过" if r["prevB"] else "挂",
        r["base"]["score"], r["base"]["level"], r["hard"]["score"], r["hard"]["level"],
        r["hard"]["hard_override"]))
print("\n硬拦截触发且 hard 救回的题：")
for r in recs:
    if r["hard"]["hard_override"] == 1 and not r["base"]["pass"] and r["hard"]["pass"]:
        print("  {} histA={} histB={} base={} hard={}".format(
            r["qid"][:12], "过" if r["prevA"] else "挂", "过" if r["prevB"] else "挂",
            r["base"]["score"], r["hard"]["score"]))
