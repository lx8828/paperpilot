"""P1 验证（可复算）：矩阵投影 + 多级表头折叠 + 竖线转义。

- 精确复现**旧算法**（首行当表头、忽略 colspan/rowspan）与新算法逐表比对；
- 统计：变化表数、多级表头数、长度变化、**矩形一致性**（表头列数 vs 数据行列数）；
- 打印那张表（2002.05058 correlation 表，对应 75b69eef 那题）的前后对照。
"""
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import MINERU_OUT  # noqa: E402
from paperpilot.tools.mineru_bridge import (_TableParser, _cell_clean,  # noqa: E402
                                            _find_content_list, table_to_md)


def parse(html):
    p = _TableParser()
    p.feed(html or "")
    p.close()
    return p.rows


def old_md(html: str) -> str:
    """精确复现旧实现：首行当表头，忽略 colspan/rowspan。"""
    lines = []
    for i, row in enumerate(parse(html)):
        cells = [_cell_clean(t) for (t, _cs, _rs, _th) in row]
        if not any(cells):
            continue
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "---|" * len(row))
    return "\n".join(lines)


def cells_count(ln: str) -> int:
    """列数：`\\|` 是转义后的字面竖线，不计入分隔符。"""
    return ln.replace("\\|", "").count("|")


def main() -> int:
    n = changed = multi = empty_hdr = nonrect = 0
    len_old = len_new = 0
    for pid in sorted(p.name.removesuffix("v1") for p in MINERU_OUT.glob("*v1") if p.is_dir()):
        for el in _find_content_list(MINERU_OUT / f"{pid}v1") or []:
            if el.get("type") != "table" or not (el.get("table_body") or ""):
                continue
            html = el["table_body"]
            old, new = old_md(html), table_to_md(html)
            n += 1
            len_old += len(old)
            len_new += len(new)
            changed += old != new
            hdr = new.splitlines()[0] if new.startswith("|") else ""
            multi += "/" in hdr
            empty_hdr += hdr.replace("|", "").strip() == ""
            ls = [ln for ln in new.splitlines() if ln.startswith("|")]
            if len(ls) >= 2 and any(cells_count(x) != cells_count(ls[0]) for x in ls[2:]):
                nonrect += 1

    print(f"全库表块 {n} 张")
    print(f"  输出变化：{changed}/{n} = {changed/n:.1%}")
    print(f"  含多级表头（列名带 /）：{multi}/{n} = {multi/n:.1%}")
    print(f"  空表头：{empty_hdr} | **列数不一致（非矩形）**：{nonrect}（应=0）")
    print(f"  总长度：{len_old} -> {len_new}（{len_new/max(len_old,1)-1:+.1%}）")
    print()
    print("目标表前后对照（2002.05058 / 75b69eef）：")
    for el in _find_content_list(MINERU_OUT / "2002.05058v1") or []:
        if el.get("type") != "table":
            continue
        if "Sample-level correlation" not in " ".join(el.get("table_caption") or []):
            continue
        print("  --- 旧 ---")
        for ln in old_md(el["table_body"]).splitlines()[:3]:
            print("   ", ln)
        print("  --- 新 ---")
        for ln in table_to_md(el["table_body"]).splitlines()[:4]:
            print("   ", ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
