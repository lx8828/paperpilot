import os
from typing import Any

import pymupdf


def _is_page_number(text: str, y0: float, y1: float, page_height: float) -> bool:
    """判断 block 是否是页码噪声。

    规则：纯数字 + 长度 1-3 位 + 位于页面顶部或底部。
    （顶部阈值 10%、底部阈值 90%，覆盖页眉里带页码和页脚页码两种版式）
    """
    if not text.isdigit() or len(text) > 3:
        return False
    return y1 < page_height * 0.1 or y0 > page_height * 0.9


def _is_arxiv_watermark(text: str, x0: float, y0: float, page_height: float) -> bool:
    """判断 block 是否是 arXiv 水印。

    arXiv 预印本首页左侧边栏会加水印，格式如：
    "arXiv:2608.28447v1  [cs.AI]  28 Aug 2026"
    特征：以 "arXiv:" 开头 + 位于页面顶部左侧边栏（y0 < 0.3*page_height 且 x0 < 30）。
    加位置约束是为了避免误杀参考文献条目（参考文献也可能以 "arXiv:xxxx.xxxxx" 开头，
    但位置在正文区域，y0 较大、x0 较大）。
    """
    if not text.startswith("arXiv:"):
        return False
    return y0 < page_height * 0.3 and x0 < 30


def _block_lines_from_dict(block: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """从 get_text('dict') 的 block 提取行级信息。

    Returns:
        (lines, text):
          lines: [{text, x0, y0, x1, y1}, ...]（按视觉行序，已去掉行尾空白）
          text:  "\n".join(行文本) 并 strip
    """
    lines = []
    for line in block.get("lines", []):
        x0, y0, x1, y1 = line["bbox"]
        ltext = "".join(span.get("text", "") for span in line.get("spans", []))
        if not ltext.strip():
            continue
        lines.append({
            "text": ltext.rstrip(),
            "x0": float(x0), "y0": float(y0),
            "x1": float(x1), "y1": float(y1),
        })
    text = "\n".join(ln["text"] for ln in lines).strip()
    return lines, text


def parse_pdf(pdf_path: str) -> dict[str, Any]:
    """解析 PDF，返回页级文本、块级带位置与行级信息、全文和元信息。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        dict: {page_count, pages, blocks, raw_text, metadata}
            - pages: 页级纯文本（第一层）
            - blocks: 块级文本（第二层，已过滤页码与 arXiv 水印噪声）。
              每项含 {block_id, page, x0..y1, text, lines}，
              其中 lines 为行级明细（每行 {text, x0, y0, x1, y1}），
              支撑段落检测（首行缩进/行距）与精确溯源。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: PDF 无法解析
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise ValueError(f"无法解析的 PDF: {pdf_path}，原因: {e}") from e

    pages = []
    blocks = []
    full_text = []

    with doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_height = page.rect.height

            # 第一层：页级纯文本
            page_text = page.get_text()
            pages.append({"page": page_num + 1, "text": page_text})
            full_text.append(page_text)

            # 第二层：块级带行级信息（type=0 文本块；type=1 图片块跳过）
            # pymupdf stub 把 get_text("dict") 的返回类型标为 str/list/dict 联合，
            # 运行期实际是 dict，这里先收窄一次，避免后续索引被误报。
            raw_dict = page.get_text("dict")
            page_blocks = raw_dict if isinstance(raw_dict, dict) else {}
            for block in page_blocks.get("blocks", []):
                if block["type"] != 0:
                    continue  # 图片块：不含可提取文本
                x0, y0, x1, y1 = block["bbox"]
                lines, text = _block_lines_from_dict(block)
                if not text:
                    continue
                # 清洗：过滤页码与 arXiv 水印噪声
                if _is_page_number(text, y0, y1, page_height):
                    continue
                if _is_arxiv_watermark(text, x0, y0, page_height):
                    continue
                blocks.append({
                    "block_id": f"p{page_num + 1}_b{block['number']}",
                    "page": page_num + 1,
                    "x0": float(x0), "y0": float(y0),
                    "x1": float(x1), "y1": float(y1),
                    "text": text,
                    "lines": lines,
                })

        metadata = dict(doc.metadata or {})

    return {
        "page_count": len(pages),
        "pages": pages,
        "blocks": blocks,
        "raw_text": "\n".join(full_text),
        "metadata": metadata,
    }
