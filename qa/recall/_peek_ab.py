import json

d = json.load(open("qa/recall/ab_answer_unknown_result.json", encoding="utf-8"))
m, r = d["main"], d["risk"]
print("HARD not-enough", m["n_not_enough"], "rescued", m["rescued"],
      "rate", round(m["rescue_rate"], 3))
print("RISK unans", r["n"], "honest pass", r["honest_pass"])
print("risk scores:", [x["score"] for x in r["detail"]])
print("rescue detail:")
for x in m["detail"]:
    print("  ", x["score"], "PASS" if x["pass"] else "fail", "|", x["answer"][:110].replace("\n", " "))
