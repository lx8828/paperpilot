"""把组 gold 题集导出成 `cli/run_qa_v2.py` 能直接跑的 `qa/questions/<stem>.json`。

**为什么要导出而不是手写两份**：gold 题集是**唯一真源**（每题都有逐字原文引文，
由 `_validate_questions.py` 把关）；runner 只认另一套扁平字段。两份手写必漂移。

字段映射（gold → runner）：

    pdf          ← <stem>.pdf
    qid/role/intent/question/hint/expect/route_min/must_not   原样
    must_all     ← gold.must_all（**必现**：全部命中才算过）
    must_have    ← **gold.must_any**（**任一命中**即算）  ← ⚠️ 名字不同、语义对齐
    chunk_kw     ← null（runner 仅作记录）
    evidence     → 丢弃（runner 不用；gold 引文留在 gold 文件里）

⚠️ **这是本管线最容易搞错的一处**：runner 的 `must_have` 语义是"任一命中"（容忍措辞差异），
`must_all` 才是"全部命中"（硬指标）。gold 里的 `must_any` 才是它的 `must_have`。
把 gold 的"必现"清单直接写进 runner 的 `must_have` 会把严格题降级成宽松题。

用法：python retrieval/scripts/_export_questions.py [--group group1]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_groups import (  # noqa: E402
    EXPORT_DIR,
    expected_total,
    export_dir_multi,
    group_expected_total,
    group_gold_file,
    material_dir,
    papers,
)


def split_anchors(q: dict) -> tuple[list[str], list[str]]:
    """gold → (must_all/必现, must_any/备选)。兼容旧 schema（旧 must_have 语义=必现）。"""
    if "must_all" in q or "must_any" in q:
        all_ = list(q.get("must_all") or [])
        any_ = list(q.get("must_any") or [])
    else:                                   # 旧 schema：must_have 是"必现"
        all_, any_ = list(q.get("must_have") or []), []
    return [x for x in all_ if x], [x for x in any_ if x]


def export_group(group: str) -> None:
    """组级题集 → `qa/multi/<group>.json`（`cli/run_multi_qa.py` 的 B 组格式）。

    ⚠️ 与单篇层的关键差别：组级 runner 的判定是
    `must_all`（全部命中）+ `must_any`（任一命中）+ `must_not`，
    所以这里**原样传三个字段**，不做 must_any → must_have 的名字转换。
    同时写一份 `must_have = must_all`，让未升级的旧版 runner 仍按严格口径判。
    """
    gf = group_gold_file(group)
    if not gf.exists():
        print(f"\n（无组级题集 {gf.name}，跳过组级导出）")
        return
    doc = json.loads(gf.read_text(encoding="utf-8"))
    out = []
    for q in doc.get("questions") or []:
        all_ = [str(x) for x in (q.get("must_all") or [])]
        any_ = [str(x) for x in (q.get("must_any") or [])]
        out.append({
            "qid": q["qid"],
            "kind": q.get("kind", ""),
            "role": q.get("role", "dev"),
            "intent": q.get("intent", ""),
            "question": q["question"],
            "hint": q.get("hint", ""),
            "expect": q.get("expect") or [],
            "route_min": q.get("route_min", "L3"),
            "must_all": all_,
            "must_any": any_,
            "must_not": [str(x) for x in (q.get("must_not") or [])],
            "must_have": all_,          # 旧版 runner 回退用（严格口径）
            "corpus": doc.get("corpus") or [],
        })
    dst_dir = export_dir_multi()
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{group}.json"
    old = len(json.loads(dst.read_text(encoding="utf-8"))) if dst.exists() else 0
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ks = Counter(str(q.get("kind") or "?") for q in out)
    print(f"\n组级导出 {len(out)} 题（期望 {group_expected_total(group)}）"
          f" {dict(ks)}  （旧 {old} → 新 {len(out)}）→ {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group1")
    args = ap.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    print(f"{'paper':<16}{'题数':>4}{'L0':>4}{'single':>7}{'table':>6}{'neg':>5}"
          f"{'user':>6}{'expect':>18}")
    for stem in papers(args.group):
        gf = material_dir(args.group) / f"{stem}.questions.json"
        if not gf.exists():
            print(f"⚠️ 缺 {gf.name}")
            continue
        qs = json.loads(gf.read_text(encoding="utf-8"))["questions"]
        out = []
        for q in qs:
            all_, any_ = split_anchors(q)
            out.append({
                "pdf": f"{stem}.pdf",
                "qid": q["qid"],
                "role": q["role"],
                "intent": q["intent"],
                "question": q["question"],
                "expect": q.get("expect") or [],
                "route_min": q.get("route_min", "L0"),
                "hint": q.get("hint", ""),
                "must_have": any_,
                "must_all": all_,
                "must_not": q.get("must_not") or [],
                "chunk_kw": None,
            })
        dst = EXPORT_DIR / f"{stem}.json"
        old = len(json.loads(dst.read_text(encoding="utf-8"))) if dst.exists() else 0
        dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cnt = {c: sum(1 for q in qs if q["category"] == c)
               for c in ("L0", "single", "table", "negative")}
        exp = {}
        for q in qs:
            for lv in q.get("expect") or []:
                exp[lv] = exp.get(lv, 0) + 1
        total += len(out)
        print(f"{stem:<16}{len(out):>4}{cnt['L0']:>4}{cnt['single']:>7}{cnt['table']:>6}"
              f"{cnt['negative']:>5}{sum(1 for q in qs if q['role'] == 'user'):>6}"
              f"{str(exp):>18}   (旧 {old} → 新 {len(out)})")
    print(f"\n合计导出 {total} 题（期望 {expected_total(args.group)}）→ {EXPORT_DIR}")
    export_group(args.group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
