"""**实验 2 诊断**：`set` 在 M1 上只拿 6/50，是"判定层不行"还是"题不是集合题"？

三个嫌疑（逐个排除）：
  H1 **题形不匹配**：M1 多是"跨篇对比/枚举"（`各自…是什么`），不是"哪几篇做了 X"的集合筛选题
     → 把整句题面当二值论断 → 判官只能答 no。
  H2 **判定真的错**：给了对的命题也判错。
  H3 **判分口径**：`score_answer` 的 `must_all` 锚点要求数字/专名，`set` 渲染不含 → 系统性低估。

输出：逐题对照表（题形分类 + 两臂得分 + 缺失锚点 + `set` 交付篇）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "qa" / "multi" / "_runs"

fc: dict[str, dict] = {}
for f in sorted(RUNS.glob("fullctx_m1all_group*.json")):
    for r in json.loads(f.read_text(encoding="utf-8")):
        fc[str(r.get("qid"))] = r
st: dict[str, dict] = {}
for f in sorted(RUNS.glob("set_m1all_group*.json")):
    for r in json.loads(f.read_text(encoding="utf-8")):
        st[str(r.get("qid"))] = r

# H1 分类：集合筛选题 vs 跨篇对比/枚举题
SEL = re.compile(r"(哪(几|些|一)篇|哪几篇|哪些篇|有没有哪|有哪几篇|谁).{0,12}(做|用|是|把|报告|提供|设计了|采用|依赖)")
SETQ = re.compile(r"(哪(几|些)篇|哪些篇|有哪几篇|哪几篇)")
ENUM = re.compile(r"(各自|分别|两篇|三篇|对比|不同|谁更|哪一个的|哪一个报告)")


def kind_of(q: str) -> str:
    if SETQ.search(q):
        return "集合筛选" if not ENUM.search(q) else "集合筛选+枚举"
    if ENUM.search(q):
        return "跨篇对比/枚举"
    return "其他"


print("=" * 122)
print(f"【实验 2 诊断】同题 {len(set(fc) & set(st))} 道 M1")
rows = []
for qid in sorted(set(fc) & set(st), key=lambda x: (x[:3], int(re.findall(r'\d+', x)[-1]))):
    a, b = fc[qid], st[qid]
    rows.append(dict(qid=qid, kind=kind_of(str(a.get("question"))),
                     fc=int(bool(a.get("ok_strict"))), st=int(bool(b.get("ok_strict"))),
                     yes=b.get("n_yes"), miss="；".join(map(str, (b.get("miss") or [])[:2]))[:64],
                     q=str(a.get("question"))[:60]))

grp: dict[str, list] = {}
for r in rows:
    grp.setdefault(r["kind"], []).append(r)
print(f"\n  {'题形':<14}{'题数':>5}{'fullctx':>10}{'set':>8}{'set占比':>10}")
for k in sorted(grp, key=lambda x: -len(grp[x])):
    g = grp[k]
    print(f"  {k:<14}{len(g):>5}{sum(x['fc'] for x in g):>10}{sum(x['st'] for x in g):>8}"
          f"{sum(x['st'] for x in g) / len(g):>10.0%}")
print(f"  {'合计':<14}{len(rows):>5}{sum(x['fc'] for x in rows):>10}{sum(x['st'] for x in rows):>8}"
      f"{sum(x['st'] for x in rows) / len(rows):>10.0%}")

print(f"\n{'=' * 122}\n  逐题（按 set 交付篇数升序；`miss` = set 答案缺的锚点）")
print(f"  {'qid':<12}{'题形':<14}{'fc':>4}{'set':>5}{'yes':>5}  {'缺失锚点':<52}题面")
for r in sorted(rows, key=lambda x: x["yes"]):
    print(f"  {r['qid']:<12}{r['kind']:<14}{'✅' if r['fc'] else '·':>3}"
          f"{'✅' if r['st'] else '·':>5}{r['yes']:>5}  {r['miss']:<52}{r['q']}")
