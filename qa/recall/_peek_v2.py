import json

d = json.load(open("qa/recall/spike_v2_result.json", encoding="utf-8"))
lines = []
for r in d:
    lines.append("=" * 70)
    lines.append(f"{r['qid'][:12]} kind={r['kind']}")
    lines.append(f"  V1 state={r['v1']['state']} score={r['v1']['score']} calls={r['v1']['calls']}")
    lines.append(f"  V2 state={r['v2']['state']} score={r['v2']['score']} calls={r['v2']['calls']} "
                 f"used={r['v2']['used']}")
    lines.append(f"  V2 gap: {r['v2']['gap']}")
open("qa/recall/_peek_v2_out.txt", "w", encoding="utf-8").write("\n".join(lines))
print("ok")
