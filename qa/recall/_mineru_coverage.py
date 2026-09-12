"""MinerU 表格覆盖率自检 v2（QASPER 官方 caption 作 ground truth）。

v1 的三个口径缺陷（已修）：
  1. 官方部分表用**罗马数字**（`TABLE I`）→ 编号提取失败；
  2. **MinerU 表块可能没有 caption** → 被排除，低估覆盖；
  3. **编号整体错位**（1909.00088: 11→13；2001.10179: 2→5）→ 误算成漏抽。

v2 输出四分桶：覆盖且编号一致 / 覆盖但编号错位 / 未覆盖 / **无法判定**（该篇 MinerU 表块无 caption）。
表块判定：xtbl-* 且不是"公式:"开头（表格必然含 markdown 分隔行 `|---|` 或 caption 以 Table 开头）。

用法：uv run python qa/recall/_mineru_coverage.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

from paperpilot.agents.document_cache import MINERU_OUT, retrieval_chunks  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402

NUM_RE = re.compile(r"^\s*(?:Table|Figure)\s+([IVXLC]+|[A-Za-z]?\d+(?:\.\d+)*)", re.I)
STOP = {"table", "figure", "the", "and", "for", "with", "from", "that", "this", "are", "was"}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
         "ix": 9, "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15,
         "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20, "xxi": 21, "xxii": 22}


def cap_num(s: str) -> str:
    """表标题 → 规范化编号（罗马数字转阿拉伯）。"""
    m = NUM_RE.match(str(s))
    if not m:
        return ""
    t = m.group(1).lower()
    if t in ROMAN:
        return str(ROMAN[t])
    return re.sub(r"[^0-9.]", "", t)


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", str(s).lower()) if w not in STOP}


def jac(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if (wa and wb) else 0.0


def main() -> int:
    thr_same = float(os.environ.get("PP_THR_SAME", "0.4"))    # 同号时的标题阈值
    thr_any = float(os.environ.get("PP_THR_ANY", "0.5"))      # 跨号匹配阈值
    P = load_papers()
    pids = sorted(p.name.removesuffix("v1") for p in MINERU_OUT.glob("*v1") if p.is_dir())
    print(f"待检 {len(pids)} 篇 | 阈值 同号 {thr_same} / 跨号 {thr_any}", flush=True)
    rows = []
    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        p = P.get(pid)
        if p is None:
            continue
        q_tbl = [(cap_num(ft.get("caption")), str(ft.get("caption") or ""))
                 for ft in p.get("figures_and_tables") or []
                 if str(ft.get("caption") or "").lstrip().lower().startswith("table")]
        if not q_tbl:
            continue
        # MinerU：全部表块（排除公式块）
        m_all = []
        n_nocap = 0
        for c in retrieval_chunks(f"qasper_{pid}.qpdf"):
            if not str(c.chunk_id).startswith("xtbl"):
                continue
            t = c.text or ""
            if t.lstrip().startswith("公式:"):
                continue
            first = t.strip().splitlines()[0] if t.strip() else ""
            is_tbl = ("|---" in t) or first.lstrip().lower().startswith("table")
            if not is_tbl:
                continue
            num = cap_num(first)
            if not num:
                n_nocap += 1
            m_all.append((c.chunk_id, num, first))
        used: set[int] = set()
        exact, shifted, miss, unknown = [], [], [], []
        for qn, qc in q_tbl:
            # 1) 同号 + 标题像
            hit = -1
            if qn:
                for j, (_cid, mn, mc) in enumerate(m_all):
                    if j in used or mn != qn:
                        continue
                    if jac(qc, mc) >= thr_same:
                        hit = j
                        break
            if hit >= 0:
                used.add(hit)
                exact.append([qn, qc[:70]])
                continue
            # 2) 跨号：标题很像（编号错位）
            best, bi = 0.0, -1
            for j, (_cid, _mn, mc) in enumerate(m_all):
                if j in used:
                    continue
                s = jac(qc, mc)
                if s > best:
                    best, bi = s, j
            if best >= thr_any:
                used.add(bi)
                shifted.append([qn, m_all[bi][1] or "无号", qc[:60], m_all[bi][2][:60]])
                continue
            # 3) 判不了 vs 真漏：该篇有"无 caption 表块"→ 无法判定
            (unknown if n_nocap else miss).append([qn, qc[:80]])
        rows.append({"pid": pid, "n_q": len(q_tbl), "n_m": len(m_all), "n_nocap": n_nocap,
                     "exact": exact, "shifted": shifted, "miss": miss, "unknown": unknown,
                     "extra": len([1 for j in range(len(m_all)) if j not in used])})
        if i % 50 == 0:
            print(f"  [{i}/{len(pids)}] t={time.time()-t0:.0f}s", flush=True)

    n = len(rows)
    tq = sum(r["n_q"] for r in rows)
    tm = sum(r["n_m"] for r in rows)
    nn = sum(r["n_nocap"] for r in rows)
    e = sum(len(r["exact"]) for r in rows)
    sh = sum(len(r["shifted"]) for r in rows)
    ms = sum(len(r["miss"]) for r in rows)
    un = sum(len(r["unknown"]) for r in rows)
    print()
    print(f"=== MinerU 表格覆盖自检 v2（{n} 篇，{time.time()-t0:.0f}s）===")
    print(f"QASPER 官方表 {tq} 张 | MinerU 表块 {tm} 个（其中**无 caption** {nn} 个）")
    print(f"  A 覆盖且编号一致 : {e}/{tq} = {e/tq:.1%}")
    print(f"  B 覆盖但编号错位 : {sh}/{tq} = {sh/tq:.1%}  ← 「按编号匹配目标表」不可靠")
    print(f"  C 未覆盖(可判定) : {ms}/{tq} = {ms/tq:.1%}")
    print(f"  D 无法判定(该篇有表块缺 caption) : {un}/{tq} = {un/tq:.1%}")
    print(f"  A+B = 内容覆盖 {e+sh}/{tq} = {(e+sh)/tq:.1%}")
    print(f"  涉及漏抽(C>0)的论文 {sum(1 for r in rows if r['miss'])}/{n}")
    print(f"  涉及编号错位(B>0)的论文 {sum(1 for r in rows if r['shifted'])}/{n}")
    Path("qa/recall/mineru_coverage_v2_20260912.json").write_text(
        json.dumps({"rows": rows, "thr_same": thr_same, "thr_any": thr_any},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n漏抽最严重的 8 篇（C>0）：")
    for r in sorted([x for x in rows if x["miss"]], key=lambda x: -len(x["miss"]))[:8]:
        print(f"  {r['pid']:<14} 官方 {r['n_q']} 张 / MinerU {r['n_m']} 块(无caption {r['n_nocap']})"
              f" → 一致 {len(r['exact'])}, 错位 {len(r['shifted'])}, 漏 {len(r['miss'])}")
        for qn, qc in r["miss"][:2]:
            print(f"       漏: Table {qn} {qc}")
    print("\n明细已写 qa/recall/mineru_coverage_v2_20260912.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
