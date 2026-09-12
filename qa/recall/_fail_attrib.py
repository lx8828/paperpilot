"""失败题归因：到底是"判分问题 / 题集问题"，还是"我们系统真做得不好"？

对一次 QASPER run 的**有答案题失败**做自动分桶（零 LLM）：

  A 输入通道缺口：gold 的答案内容在论文**正文段落**里找不到（QASPER 把标注挂在
     表格/图对象上 `FLOAT SELECTED: ...`，而我们的 QASPER 输入通道只读 full_text 段落）
     → 结构上做不到，不是答错
  B 送达/检索失败：gold 内容正文里有，但**没进 cites**（检索没把它送到作答层）
  C answer 层漏用：gold 内容**已在 cites 里**，但答案没给出/没采用
  D 疑似判分过严：gold 关键内容**已出现在答案里**，裁判仍判 fail
  E 其他（需人工）

判据：gold 的**数字**为主信号（去千分位后做包含判定），无数字时退化为内容词重合率。
这是**启发式首过**，D 桶必须再用第二裁判复核才能定量"判分问题"。

用法：
    uv run python qa/recall/_fail_attrib.py qa/qasper_run_20260911_072510.json [--show 8]
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

NUM = re.compile(r"\d+(?:[.,]\d+)*")
STOP = set("the a an of and or to in for on with by is are was were be been that this these those "
           "we our they their it its as at from not no can may using used use than then more less".split())


def nums(text: str) -> set[str]:
    out = set()
    for t in NUM.findall(text or ""):
        t = t.replace(",", "")
        if len(t) >= 2:
            out.add(t)
    return out


def toks(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", text or "")
            if w.lower() not in STOP}


def frac(have: set[str], want: set[str]) -> float:
    return (len(have & want) / len(want)) if want else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--dump", default=None, help="把分桶写到 JSON（供下游脚本用，如通道扩容实验）")
    args = ap.parse_args()

    recs = json.loads(Path(args.run).read_text(encoding="utf-8"))
    papers = load_papers()
    bad = [r for r in recs if r.get("status") == "fail" and not r.get("unanswerable")]
    unans = [r for r in recs if r.get("status") == "fail" and r.get("unanswerable")]
    print(f"文件: {args.run}｜题数 {len(recs)}｜有答案题 fail {len(bad)}"
          f"（/{len([r for r in recs if not r.get('unanswerable')])}）｜无答案题 fail {len(unans)}")

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for r in bad:
        pid = r["paper"].replace("qasper_", "").replace(".qpdf", "")
        paper = papers.get(pid) or {}
        q = next((qq for qq in paper.get("qas") or []
                  if str(qq.get("question_id") or "") == r["qid"]), None)
        if q is None:
            buckets["E 其他"].append({"r": r, "why": "题不在数据集里"})
            continue
        gold, evs = gold_answer_full(q)
        g_nums, g_toks = nums(gold), toks(gold)

        # 正文可达性：论文 full_text 段落拼成的文本（= 我们 QASPER 输入通道的来源）
        full = "\n".join(str(p) for sec in (paper.get("full_text") or [])
                         for p in (sec.get("paragraphs") or []))
        cite_txt = "\n".join(str(c.get("evidence") or "") for c in (r.get("cites") or []))
        ans = r.get("answer") or ""

        if g_nums:
            reach = frac(nums(full), g_nums) >= 0.8
            in_cite = frac(nums(cite_txt), g_nums) >= 0.8
            in_ans = frac(nums(ans), g_nums) >= 0.8
            sig = f"数字 {len(g_nums)} 个"
        else:
            reach = frac(toks(full), g_toks) >= 0.6
            in_cite = frac(toks(cite_txt), g_toks) >= 0.6
            in_ans = frac(toks(ans), g_toks) >= 0.6
            sig = f"内容词 {len(g_toks)} 个"
        float_ev = any(str(e).upper().startswith("FLOAT") for e in evs)

        item = {"r": r, "sig": sig, "reach": reach, "in_cite": in_cite,
                "in_ans": in_ans, "float_ev": float_ev, "gold": gold,
                "full_hit": round(frac(nums(full), g_nums), 2) if g_nums else None,
                "cite_hit": round(frac(nums(cite_txt), g_nums), 2) if g_nums else None,
                "ans_hit": round(frac(nums(ans), g_nums), 2) if g_nums else None}
        if in_ans:
            buckets["D 疑似判分过严"].append(item)
        elif "未能通过内部事实校验" in ans:
            buckets["B0 闸门兜底（系统自伤）"].append(item)
        elif not reach:
            buckets["A 输入通道缺口"].append(item)
        elif in_cite:
            buckets["C answer 层漏用"].append(item)
        elif float_ev:
            buckets["A 输入通道缺口"].append(item)
        else:
            buckets["B 送达/检索失败"].append(item)

    print("\n== 分桶 ==")
    for k in sorted(buckets):
        print(f"  {k:<18} {len(buckets[k])}")

    if args.dump:
        payload = {k: [{"pid": it["r"]["paper"], "qid": it["r"]["qid"],
                        "question": it["r"]["question"], "gold": it["gold"],
                        "score": it["r"].get("score"), "answer": it["r"].get("answer")}
                       for it in v] for k, v in buckets.items()}
        Path(args.dump).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"  已写分桶 JSON: {args.dump}")

    # ── 剔除"不可归因于系统"的题（A 输入通道缺口 + D 判分）后重算 ──
    excl = {it["r"]["qid"] for k in ("A 输入通道缺口", "D 疑似判分过严") for it in buckets[k]}
    keep = [r for r in recs if r["qid"] not in excl]
    ok = lambda r: r.get("status") in ("pass", "excerpt_ok")  # noqa: E731
    p = lambda g: sum(1 for r in g if r.get("status") == "pass")  # noqa: E731

    def line(label: str, g: list[dict]) -> str:
        n = len(g)
        if n == 0:
            return f"  {label:<22} —"
        return (f"  {label:<22} 主通过 {p(g)}/{n} = {p(g)/n:6.1%}｜"
                f"合格(含 excerpt_ok) {sum(1 for r in g if ok(r))}/{n} = "
                f"{sum(1 for r in g if ok(r))/n:6.1%}")

    print("\n== 剔除「不可归因于系统」的题后重算 ==")
    print(f"  剔除 {len(excl)} 题：输入通道缺口 "
          f"{len(buckets['A 输入通道缺口'])} + 疑似判分 {len(buckets['D 疑似判分过严'])}")
    print(line("原始（全部）", recs))
    print(line("剔除后（全部）", keep))
    print(line("  剔除后有答案题", [r for r in keep if not r.get("unanswerable")]))
    print(line("  剔除后无答案题", [r for r in keep if r.get("unanswerable")]))

    print("\n== 说明 ==")
    print("  A：gold 内容在正文段落里不可达（QASPER 常挂在 FLOAT 表格/图上）→ 输入通道结构性缺口")
    print("  B：正文里可达但没进 cites → 检索/送达问题（**我们的**）")
    print("  C：cites 里有但答案没用 → answer 层问题（**我们的**）")
    print("  D：答案里已含 gold 内容仍判 fail → 疑似判分问题（需第二裁判复核）")

    for k in sorted(buckets):
        items = buckets[k][: args.show]
        if not items:
            continue
        print(f"\n== {k} 样例（最多 {args.show}） ==")
        for it in items:
            r = it["r"]
            print(f"  - {r['paper'][-14:]} | {r['question'][:58]}")
            print(f"    {it['sig']} full={it['full_hit']} cite={it['cite_hit']} ans={it['ans_hit']}"
                  f" float_ev={it['float_ev']}")
            print(f"    gold: {str(it['gold'])[:110]}")
            print(f"    ans : {str(r.get('answer'))[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
