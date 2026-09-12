"""无 caption 表块：到底缺什么、有多少能补？

修正 v1 的两个口径 bug：
  1) v1 用"首行必须以 Table N 开头"判定有 caption → 把 `Age Race Gender Table 3: ...`
     这种 **caption 与表体粘连** 的块误判为"无 caption"（虚增）；
  2) v1 没排除公式块（`公式: ...` 里也含 `|`）→ 虚增分母。
指纹改用 **列名集合**（更稳）：同一张表被切成多块时，各块的列名集合相同，
即使表体被横向拼接、列头重复，去重后仍能对上。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"
from paperpilot.agents.document_cache import MINERU_OUT, retrieval_chunks  # noqa: E402

CAP_RE = re.compile(r"(?:Table|Figure)\s+[IVXLC]*\d+", re.I)


def cols_fp(text: str) -> str:
    """列名集合指纹：**只取第一行表头**的单元格名（去重排序）。

    ⚠️ 不能把数据行的值也算进来（v1 就是这么错的）：不同切块的数据行不同，
    会把本可匹配的兄弟块指纹冲散。
    """
    for ln in str(text).splitlines():
        if "|" not in ln or ln.strip().startswith("|--"):
            continue
        if CAP_RE.search(ln) and ln.count("|") < 3:     # caption 行，跳过
            continue
        cells = {c.strip() for c in ln.split("|") if 0 < len(c.strip()) <= 24
                 and not re.fullmatch(r"[\d.\-–—%]+", c.strip())}
        return "|".join(sorted(cells))[:200]
    return ""


def main() -> int:
    pids = sorted(p.name.removesuffix("v1") for p in MINERU_OUT.glob("*v1") if p.is_dir())
    tot = nocap = borrow = cols_only = 0
    samples: list[str] = []
    for pid in pids:
        try:
            cs = [c for c in retrieval_chunks(f"qasper_{pid}.qpdf")
                  if str(c.chunk_id).startswith("xtbl")]
        except Exception:  # noqa: BLE001
            continue
        tabs = []
        for c in cs:
            t = (c.text or "").strip()
            if not t or t.startswith("公式:") or "|" not in t:
                continue
            head = t[:150]
            tabs.append((str(c.chunk_id), t, bool(CAP_RE.search(head)), cols_fp(t)))
        if not tabs:
            continue
        with_cap = [x for x in tabs if x[2]]
        for cid, t, has_cap, fp in tabs:
            tot += 1
            if has_cap:
                continue
            nocap += 1
            sib = next((x for x in with_cap if fp and x[3] == fp), None)
            if sib:
                borrow += 1
                if len(samples) < 3:
                    samples.append(
                        f"-- {pid} {cid}（无 caption）\n"
                        f"   现在块首是纯表格: {t.splitlines()[0][:88]}\n"
                        f"   兄弟块 {sib[0]} 已有 caption: {sib[1].splitlines()[0][:88]}")
            elif fp:
                cols_only += 1
    print(f"=== 表块 {tot} | 无 caption {nocap} = {nocap/max(tot,1):.1%} ===")
    print(f"    其中可从**兄弟块借到 caption**（列名集合相同）: {borrow} = {borrow/max(nocap,1):.0%}")
    print(f"    无兄弟可借、但有列名可拼身份串              : {cols_only} = "
          f"{cols_only/max(nocap,1):.0%}")
    print(f"    两者都无（连列名都取不到）                  : {nocap-borrow-cols_only}")
    print()
    for s in samples:
        print(s)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
