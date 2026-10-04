"""三类题的**平均耗时 / 平均花费**（数据源：09-27 重跑的 148 题，逐题都记了用量）。

口径：
  · 用时 = 记录里的 `seconds`（`graph.ask` 全图，含 Router/judge/检索/闸门/修复）
  · tokens = `prompt_tokens` / `completion_tokens`（`llm.reset_usage()` 每题独立记账）
  · 花费按 **deepseek-chat 参考价 ¥1/M 输入 + ¥2/M 输出**（改价改 PRICE 即可）
  · 题型按**题集标注**分（`route_min` / `kind`），不按实际路由 —— 实际路由会随机漂移

用法：uv run python retrieval/tmp/_cost_report.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "qa/multi/_runs"
P_IN, P_OUT = 1.0, 2.0          # ¥ / 百万 token


def cost(pt: int, ct: int) -> float:
    return (pt * P_IN + ct * P_OUT) / 1e6


def main() -> int:
    recs: list[dict] = []
    for g in ("group1", "group2"):
        recs += json.loads((R / f"rerun_nom1_{g}.json").read_text(encoding="utf-8"))

    def kind_of(x: dict) -> str:
        g = str(x.get("group"))
        if g == "B":
            return {"M0": "② 多篇·免检索（M0）"}.get(str(x.get("kind")), "④ 组级其他")
        if str(x.get("route_min")) == "L0":
            return "① 单篇·免检索"
        return "③ 单篇·需检索"

    agg: dict[str, list[dict]] = defaultdict(list)
    for x in recs:
        agg[kind_of(x)].append(x)

    print(f"\n########## 三类题的平均耗时 / 平均花费（{len(recs)} 题，09-27 重跑）")
    print(f"  {'类别':<22}{'题数':>5}{'均秒':>7}{'均调用':>7}{'均prompt':>9}"
          f"{'均完成':>7}{'¥/题':>8}{'合计¥':>8}")
    tot = [0, 0.0, 0, 0, 0, 0.0]
    for k in sorted(agg):
        g = agg[k]
        n = len(g)
        sec = sum(float(x.get("seconds") or 0) for x in g) / n
        cal = sum(int(x.get("llm_calls") or 0) for x in g) / n
        pt = sum(int(x.get("prompt_tokens") or 0) for x in g)
        ct = sum(int(x.get("completion_tokens") or 0) for x in g)
        c = cost(pt, ct)
        print(f"  {k:<22}{n:>5}{sec:>7.1f}{cal:>7.1f}{pt / n:>9.0f}{ct / n:>7.0f}"
              f"{c / n:>8.3f}{c:>8.2f}")
        tot[0] += n
        tot[1] += sec * n
        tot[2] += int(cal * n)
        tot[3] += pt
        tot[4] += ct
        tot[5] += c
    print(f"  {'— 合计 / 平均 —':<22}{tot[0]:>5}{tot[1] / tot[0]:>7.1f}"
          f"{tot[2] / tot[0]:>7.1f}{tot[3] / tot[0]:>9.0f}{tot[4] / tot[0]:>7.0f}"
          f"{tot[5] / tot[0]:>8.3f}{tot[5]:>8.2f}")

    print(f"\n########## 按**实际路径**（对照：同一类题走 L0 还是 L3 差多少）")
    print(f"  {'实际路径':<24}{'题数':>5}{'均秒':>7}{'均调用':>7}{'均prompt':>9}"
          f"{'¥/题':>8}")
    byp: dict[str, list[dict]] = defaultdict(list)
    for x in recs:
        byp["L3 检索" if "L3" in (x.get("route") or []) else "L0 直答（不检索）"].append(x)
    for k in sorted(byp):
        g = byp[k]
        n = len(g)
        sec = sum(float(x.get("seconds") or 0) for x in g) / n
        cal = sum(int(x.get("llm_calls") or 0) for x in g) / n
        pt = sum(int(x.get("prompt_tokens") or 0) for x in g)
        ct = sum(int(x.get("completion_tokens") or 0) for x in g)
        print(f"  {k:<24}{n:>5}{sec:>7.1f}{cal:>7.1f}{pt / n:>9.0f}{cost(pt, ct) / n:>8.3f}")

    l3 = byp.get("L3 检索", [])
    l0 = byp.get("L0 直答（不检索）", [])
    if l3 and l0:
        print(f"\n  读法：走 L3 比走 L0 **慢 "
              f"{sum(float(x.get('seconds') or 0) for x in l3) / len(l3) - sum(float(x.get('seconds') or 0) for x in l0) / len(l0):+.1f} 秒**、"
              f"**贵 {cost(sum(int(x.get('prompt_tokens') or 0) for x in l3), 0) / len(l3) - cost(sum(int(x.get('prompt_tokens') or 0) for x in l0), 0) / len(l0):+.3f} 元/题**")
    print("\n  ⚠️ 价格为参考口径（deepseek-chat ¥1/M 输入 + ¥2/M 输出）；"
          "换模型只需改脚本里的 P_IN/P_OUT。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
