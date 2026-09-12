"""临时：复杂版面对比——双栏阅读顺序 + 页眉/脚/页码剥离（pymupdf vs MinerU）。"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf

PDFS = {
    "2609.02056v1.pdf": "assets/artifacts/out_mineru/2609.02056v1/2609.02056v1/auto/2609.02056v1_content_list.json",
    "2609.08696v1.pdf": "assets/artifacts/out_mineru/2609.08696v1/2609.08696v1/auto/2609.08696v1_content_list.json",
}


def page_cols(pdf_path: str, pno: int) -> tuple[int, int]:
    """返回 (左簇中心x, 右簇中心x)；单栏返回 (0, 0)。"""
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(pno)
    xs = []
    for b in (page.get_text("dict").get("blocks") or []):
        if b["type"] == 0 and len(b.get("lines") or []) > 0:
            xs.append((b["bbox"][0] + b["bbox"][2]) / 2)
    if not xs:
        return 0, 0
    xs.sort()
    n = len(xs)
    left = xs[: n // 2]
    right = xs[n // 2:]
    cl, cr = sum(left) / len(left), sum(right) / len(right)
    # 页宽归一：间隔大则判双栏
    pw = page.rect.width
    return (round(cl), round(cr)) if (cr - cl) > pw * 0.3 else (0, 0)


for pdf, clpath in PDFS.items():
    doc = pymupdf.open(f"src/paperpilot/assets/papers/{pdf}")
    print("=" * 78)
    print(pdf)
    # 找双栏正文页
    two_col_pages = []
    for pno in range(3, min(doc.page_count, 14)):  # 跳过首页/前几页
        cl, cr = page_cols(f"src/paperpilot/assets/papers/{pdf}", pno)
        if cl:
            two_col_pages.append((pno, cl, cr))
    print("双栏候选页:", two_col_pages[:8])
    if not two_col_pages:
        continue
    pno, cl, cr = two_col_pages[0]
    pm = doc.load_page(pno)
    pm_text = pm.get_text().strip()
    cl_ = json.loads(Path(clpath).read_text(encoding="utf-8"))
    page_els = [i for i in cl_ if i.get("page_idx") == pno]
    m_text = "\n".join((i.get("text") or i.get("table_caption") and " ".join(i.get("table_caption")) or "")
                       for i in page_els if i.get("type") in ("text", "aside_text", "page_footnote"))[:2000]
    print(f"\n--- 页 {pno+1}（双栏 左中心{cl} 右中心{cr}） pymupdf 前 700 字 ---")
    print(pm_text[:700])
    print(f"\n--- 页 {pno+1} MinerU 阅读序前 700 字 ---")
    print(m_text[:700])
    types = {t: sum(1 for i in page_els if i.get("type") == t) for t in set(i.get("type") for i in page_els)}
    print(f"\n页元素类型计数: {types}")
    break  # 只展示每篇第一候选页会太长；仅第一篇展开
