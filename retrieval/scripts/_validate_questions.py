"""题集校验器：JSON 合法性 + 字段完备 + 配额 + 判定锚点 + **gold 引文逐字核对到原文**。

引文核对是本测试集的**可信度底线**：`evidence[].quote` 必须是 PDF 原文里
真实存在的逐字片段（否则 gold 是编的）。比对前把两侧空白折叠为单空格。

另外两处**防回归**（2026-09-22 加，都是踩过的坑）：
    · `expect` / `route_min` 必须落在**现行档位词表** `{L0, L3, unknown}` ——
      旧四层漏斗已下线（L1/L2 在活图里不可达），标 L1/L2 只会让记录里挂一堆假 route_mismatch。
    · 每题必须至少有一个**判定锚点**（`must_all` 或 `must_any`）—— 否则这题不可自动判分。

用法：python retrieval/scripts/_validate_questions.py [--group group1]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_groups import QUOTA, expected_total, material_dir, papers  # noqa: E402

REQ = ("qid", "category", "role", "intent", "question", "hint",
       "evidence", "expect", "route_min")
LEVELS = {"L0", "L3", "unknown"}

# PDF 抽取的两类伪影，两侧须**同口径**归一再比对：
#   ① 行尾连字符折行：`shrink-\nage` → `shrinkage`（还原整词）
#   ② 印刷体标点：`’‘` → `'`，`“”` → `"`，`−`(U+2212) → `-`，不换行空格 → 空格
_TRANS = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2212": "-", "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-",
    "\u00ad": "", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
})

_PAGE_RE = r"={5,}\s*\[\[p\d+\]\]\s*={5,}"


def norm(s: str) -> str:
    """归一化：印刷体标点 → ASCII，空白折叠（**不动连字符**）。"""
    return re.sub(r"\s+", " ", (s or "").translate(_TRANS)).strip()


def anchor_fragile(a: str) -> bool:
    """锚点形状脆弱：**纯 ASCII 且既无数字也无大写字母** → 英文普通词/短语。

    为什么要管（2026-09-23 冒烟实测）：答案多为中文，`candidate metadata` /
    `grounding` / `beam` 这类纯小写英文词**在中文答案里不保证出现**，只能靠
    "引用证据通道"（runner 把答案与前 5 条引用拼起来匹配）兜底 → 不稳定。

      · `category == "L0"`（免检索）：连引用通道都没有 → **必漏**（判 em 错）
      · 其余（检索类）：能兜底但脆弱 → 警告，建议补一个数字/专名锚点

    合规形状：`0.1125` / `88.6%`（数字）、`MIDR` / `Yelp` / `top-K`（含大写专名/术语）。
    纯中文锚点不在此列（它们由 `_check_reachable.py` 的"必须在检索视图里"管）。
    """
    if not re.search(r"[A-Za-z]", a):
        return False
    return not re.search(r"[0-9]", a) and not re.search(r"[A-Z]", a)


def sources_variants(txt: str) -> list[str]:
    """原文的**多种归一化口径** —— 折行连字符天生有二义性，只能多口径比对：

        `shrink-\\nage`      → 连字符是**折行伪影**，该还原成 `shrinkage`
        `personal-\\nfrequency` → 连字符是**真实连字符**，该保留成 `personal-frequency`

    两种在版面上长得一模一样，单一口径必有一种对不上。故生成三个变体，
    引文命中**任一变体**即算通过。
    """
    t = (txt or "").translate(_TRANS)
    t = re.sub(_PAGE_RE, " ", t)
    keep = re.sub(r"\s+", " ", t)                    # ① 保留连字符（拼成 "- " 带空格）
    keep2 = re.sub(r"-\s+", "-", keep)               # ② 把 "- 空格" 压回 "-"
    drop = re.sub(r"-\s*\n\s*", "", t)               # ③ 折行连字符还原成整词
    drop = re.sub(r"\s+", " ", drop)
    out = [keep, keep2, drop]
    return list(dict.fromkeys(out))                  # 去重，保持顺序


def diverge(quote: str, src: str) -> str:
    """返回引文与原文的**首个分歧位置**上下文（诊断用，避免只能看到"不在原文中"）。"""
    lo, hi = 0, min(len(quote), 4000)
    while lo < hi:                            # 二分找最长公共前缀
        mid = (lo + hi + 1) // 2
        if quote[:mid] in src:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        return "(开头就对不上)"
    pos = src.find(quote[:lo])
    return f"前 {lo} 字匹配，分歧处引文={quote[lo:lo + 46]!r} 原文≈{src[pos + lo: pos + lo + 46]!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group1")
    args = ap.parse_args()

    d = material_dir(args.group)
    files = [d / f"{stem}.questions.json" for stem in papers(args.group)]
    files = [f for f in files if f.exists()]
    if not files:
        print(f"⚠️ {d} 下还没有任何 .questions.json")
        return 1
    total = bad = soft = 0
    for f in files:
        stem = f.name[: -len(".questions.json")]
        raw = f.read_text(encoding="utf-8")
        print(f"\n=== {f.name} ===")
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON 非法：{e}")
            print(f"     {raw[max(0, e.pos - 90): e.pos + 90]!r}")
            bad += 1
            continue

        qs = doc.get("questions") or []
        print(f"  题数 {len(qs)}")
        cnt: dict[str, int] = {}
        for q in qs:
            cnt[q.get("category", "?")] = cnt.get(q.get("category", "?"), 0) + 1
        for c, want in QUOTA.items():
            got = cnt.get(c, 0)
            flag = "✓" if got == want else "⚠️"
            print(f"    {flag} {c:<9} {got} / 期望 {want}")

        txt = d / f"{stem}.txt"
        srcs = sources_variants(txt.read_text(encoding="utf-8")) if txt.exists() else []
        if not srcs:
            print(f"  ⚠️ 找不到原文 {txt.name} → 跳过引文核对")
        for q in qs:
            total += 1
            qid = q.get("qid", "?")
            miss = [k for k in REQ if k not in q]
            if miss:
                print(f"  ❌ {qid}: 缺字段 {miss}")
                bad += 1
            if not (q.get("must_all") or q.get("must_any") or q.get("must_have")):
                print(f"  ❌ {qid}: 无判定锚点（must_all / must_any 皆空）")
                bad += 1
            elif q.get("category") != "negative" and not q.get("must_all"):
                # 非拒答题必须有**必现锚点**，否则没法用 `_check_reachable.py` 自检可答性
                print(f"  ❌ {qid}: 缺 must_all（必现锚点）—— 定性题也要给 1 个英文/数字锚点")
                bad += 1
            # 锚点**形状**：纯小写英文词/短语在中文答案里不保证出现（见 anchor_fragile）
            fragile = [a for a in (q.get("must_all") or []) if a and anchor_fragile(a)]
            if fragile:
                if q.get("category") == "L0":
                    print(f"  ❌ {qid}: must_all 锚点 {fragile} 形状脆弱，且 L0 题**没有"
                          f"引用证据通道**兜底 → 中文答案必漏（换数字/专名）")
                    bad += 1
                else:
                    print(f"  ⚠️ {qid}: must_all 含纯小写英文锚点 {fragile}"
                          f" —— 靠引用通道兜底，建议补数字/专名（不计失败）")
                    soft += 1
            for key in ("expect", "route_min"):
                vals = q.get(key) if isinstance(q.get(key), list) else [q.get(key)]
                stale = [v for v in vals if v and v not in LEVELS]
                if stale:
                    print(f"  ❌ {qid}: {key}={stale} 不在现行档位词表 {sorted(LEVELS)}"
                          f"（v2 四层漏斗已下线）")
                    bad += 1
            evs = q.get("evidence") or []
            if not evs:
                print(f"  ❌ {qid}: evidence 为空")
                bad += 1
            for ev in evs:
                quote = norm(ev.get("quote", ""))
                if not quote:
                    print(f"  ❌ {qid}: 空引文")
                    bad += 1
                elif srcs and not any(quote in s for s in srcs):
                    print(f"  ❌ {qid} p{ev.get('page')}: 引文不在原文中（三种口径都未命中）")
                    print(f"       {diverge(quote, srcs[1])}")
                    bad += 1
    print(f"\n{'=' * 62}\n合计 {total} 题（期望 {expected_total(args.group)}），问题 {bad} 处"
          + (f"（另有 {soft} 处**脆弱锚点**警告，不阻塞）" if soft else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
