"""**生产链路泄露审计**（用户要求"扩到生产链路"）

三问：
  A. 题面（`question`）里是否已含判分锚点（`must_all`）→ 构造级泄露
  B. 判分是否被"模型自己引用的证据"放水（`ok` vs `ok_strict`）
  C. `hint`（=答案）是否进了 pipeline（静态 + 动态）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(__file__).resolve().parents[2]          # tmp → retrieval → 仓库根


def toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", str(s).lower())} | \
           {t for t in re.findall(r"[\u4e00-\u9fff]{2,}", str(s))}


def norm(s: str) -> str:
    return " ".join(str(s or "").split()).casefold()


# ── A. 题面 vs 锚点 ─────────────────────────────────────────────────────────
print("=" * 112)
print("【A】题面 `question` 是否已含判分锚点 `must_all`（构造级泄露）")
files = sorted(list((R / "qa" / "multi").glob("group*.json")) +
               list((R / "qa" / "questions").glob("*.json")))
rows = []
for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    qs = d.get("questions") if isinstance(d, dict) else d
    for q in qs or []:
        qn = norm(q.get("question"))
        must = [str(x) for x in (q.get("must_all") or [])]
        any_ = [str(x) for x in (q.get("must_any") or q.get("must_have") or [])]
        ev = [norm(e.get("quote")) for e in (q.get("evidence") or []) if e.get("quote")]
        qt = toks(q.get("question"))
        leak = [a for a in must if norm(a) and norm(a) in qn]
        rows.append(dict(
            file=f.name, qid=str(q.get("qid") or ""), kind=str(q.get("kind") or ""),
            n_must=len(must), n_leak=len(leak), leak=leak,
            must_tok_ov=len(toks(" ".join(must)) & qt) / max(1, len(toks(" ".join(must)))) if must else 0.0,
            any_tok_ov=len(toks(" ".join(any_)) & qt) / max(1, len(toks(" ".join(any_)))) if any_ else 0.0,
            ev_tok_ov=float(np.mean([len(toks(e) & qt) / max(1, len(toks(e))) for e in ev])) if ev else 0.0,
            q_len=len(str(q.get("question") or "")),
        ))
a = __import__("pandas").DataFrame(rows)
print(f"  题数 {len(a)} ｜ 题集文件 {a['file'].nunique()}")
print(f"  {'':<38}{'均值':>9}")
print(f"  {'must_all 锚点个数':<38}{a['n_must'].mean():>9.2f}")
print(f"  **{'★ 题面里已含的 must_all 锚点数（泄露）':<34}{a['n_leak'].mean():>9.3f}**")
print(f"  **{'★ 含泄露锚点的题占比':<34}{(a['n_leak'] > 0).mean():>9.1%}**")
print(f"  {'must_all 词与题面的 token 重合率':<36}{a['must_tok_ov'].mean():>9.3f}")
print(f"  {'must_any 词与题面的 token 重合率':<36}{a['any_tok_ov'].mean():>9.3f}")
print(f"  {'gold 引文与题面的 token 重合率':<37}{a['ev_tok_ov'].mean():>9.3f}")
bad = a[a["n_leak"] > 0]
if len(bad):
    print(f"\n  ⚠️ 题面里直接出现锚点的题（{len(bad)} 道）：")
    for _, r in bad.head(12).iterrows():
        print(f"    {r['qid']:<16} {r['file']:<22} 锚点 {r['leak']}")
else:
    print("\n  ✅ **没有一道题的题面里出现 must_all 锚点** → 题面没有直接把答案抄进去")

# ── B. 判分放水：ok vs ok_strict ─────────────────────────────────────────────
print("\n" + "=" * 112)
print("【B】判分放水：`ok`（答案+模型自引证据）vs `ok_strict`（仅答案）")
runs = sorted((R / "qa" / "multi" / "_runs").glob("*.json"))
br = []
for f in runs:
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        continue
    recs = d.get("records") if isinstance(d, dict) else d
    if not isinstance(recs, list) or not recs:
        continue
    q = [r for r in recs if isinstance(r, dict) and "ok" in r]
    if not q:
        continue
    for g, gname in (("A", "单篇题"), ("B", "跨篇 M1/X5")):
        gs = [r for r in q if str(r.get("group") or "") == g]
        if len(gs) < 5:
            continue
        ok = np.mean([bool(r.get("ok")) for r in gs])
        st = np.mean([bool(r.get("ok_strict")) for r in gs]) if any("ok_strict" in r for r in gs) else float("nan")
        br.append(dict(run=f.stem, group=gname, n=len(gs), ok=ok, ok_strict=st, gap=ok - st))
b = __import__("pandas").DataFrame(br)
if len(b):
    grp = b.groupby("group")[["n", "ok", "ok_strict", "gap"]].mean()
    print(f"  {'题类':<12}{'运行数':>7}{'题数':>7}{'ok':>9}{'ok_strict':>11}{'★放水幅度':>11}")
    for g, r in grp.iterrows():
        print(f"  {g:<12}{int(b[b['group'] == g].shape[0]):>7}{r['n']:>7.1f}{r['ok']:>9.3f}"
              f"{r['ok_strict']:>11.3f}{r['gap']:>+11.3f}")
    print(f"\n  → `ok_strict` 是**干净口径**（只看答案）；`ok` 含「模型自引的原文」通道"
          f"→ 平均抬高 **{b['gap'].mean():+.3f}**")
    b.to_csv(R / "retrieval" / "results" / "R2_PROD_JUDGE_GAP.csv", index=False, encoding="utf-8")
else:
    print("  （未找到含 ok/ok_strict 的运行文件）")

# ── C. hint 是否进 pipeline ────────────────────────────────────────────────
print("\n" + "=" * 112)
print("【C】`hint`（=答案）是否进了 pipeline")
print("  静态：`cli/run_group_qa.py` L137 → `graph_ask(question, corpus)`  ← 只传题干 + 语料")
print("  静态：`cli/run_multi_qa.py` L500 → state={'question','pdfs',...}  ← 只传题干")
print("  静态：`run_multi_qa.py::score_answer` 只用 must_all/must_any/must_not（判分侧）")

import subprocess  # noqa: E402
print("\n  动态：用 `python -X importtime` 无法证明；改查运行时读取面 ——")
hits = []
for p in list((R / "cli").glob("*.py")) + list((R / "src" / "paperpilot").rglob("*.py")):
    t = p.read_text(encoding="utf-8", errors="replace")
    for i, ln in enumerate(t.splitlines(), 1):
        st = ln.strip()
        if st.startswith("#") or st.startswith('"') or st.startswith("'"):
            continue
        if re.search(r'(get\(["\']hint["\']\)|\[["\']hint["\']\])', ln):
            hits.append((str(p.relative_to(R)), i, st[:110]))
print(f"  运行时（src/ + cli/ 非注释行）读 `hint` 的位置：{len(hits)} 处")
for h in hits:
    print(f"    {h[0]}:{h[1]}  {h[2]}")
if not hits:
    print("  ✅ **0 处** → `hint`（含答案）**没有进任何运行时路径**；"
          "只在检索/校验脚本里用（`_m1_diag` / `_limitation_dump` / `_anchor_bridge`）")
print("\n  ⚠️ 遗留风险：`hint` 与 `must_all` **一起**存在于 runner 读的题集文件里"
      "（`qa/multi/*.json`、`qa/questions/*.json`）→ 任何新 runner 只要写一句"
      " `q['hint']` 就会瞬间变成满分。建议在题集加载处**显式剔除** `hint` 后再交给 runner。")
a.to_csv(R / "retrieval" / "results" / "R2_PROD_Q_ANCHOR.csv", index=False, encoding="utf-8")
print(f"\n已写 R2_PROD_Q_ANCHOR.csv ／ R2_PROD_JUDGE_GAP.csv")
