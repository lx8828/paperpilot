"""Validator 离线校准：在已有 run 的 answer 上跑"漏数/编数"检查，看能否抓出失败题。

数据：ab_multiq_result.json（40 题多方法对比，pass 33/fail 7，量化子类 50% 是已知短板）。
只读文本无需 LLM。指标：fail 题中被 flag 的比例（召回视角） vs pass 题中被误 flag 比例。
"""
from __future__ import annotations
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str("src"))

from paperpilot.components.validator import is_numeric_question, numbers_in  # noqa: E402


def main() -> int:
    recs = json.load(open("qa/recall/ab_multiq_result.json", encoding="utf-8"))
    flagged = []
    for r in recs:
        q, ans = r["question"], (r.get("answer") or "")
        num_q = is_numeric_question(q)
        no_num = num_q and not numbers_in(ans)
        flagged.append({"qid": r["qid"], "pass": r["pass"], "score": r["score"],
                        "num_q": num_q, "no_num": no_num, "q": q[:90]})
    nf = sum(1 for r in flagged if r["no_num"])
    fails = [r for r in flagged if not r["pass"]]
    ps = [r for r in flagged if r["pass"]]
    f_hit = sum(1 for r in fails if r["no_num"])
    p_hit = sum(1 for r in ps if r["no_num"])
    print(f"40 题中: 数值型问题 {sum(1 for r in flagged if r['num_q'])} | 漏数(数值型但答案无数) {nf}")
    print(f"fail 题 {len(fails)} 中漏数 {f_hit}（{f_hit/len(fails):.0%}）")
    print(f"pass 题 {len(ps)} 中漏数(误报) {p_hit}（{p_hit/len(ps):.0%}）")
    print()
    print("漏数题清单：")
    for r in flagged:
        if r["no_num"]:
            print("  {} {} score={} | {}".format(r["qid"][:12],
                  "PASS" if r["pass"] else "FAIL", r["score"], r["q"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
