"""MinerU content_list → 与 chunker 同构的 Chunk[]（表格进检索的解析后端）。

背景（2026-09-09）：pymupdf 只取文本块，PDF 表格被撕成碎片且结构全丢 →
L3 检索永远捞不到表值（Recall ~90% 的那 ~10% 缺口主因之一）。MinerU 输出
带阅读顺序 + 结构化元素（title 已分级、table 为 HTML、equation 为 LaTeX、
chart/image 带 caption），可直接转换为下游 Chunk[]。

本模块只做**格式转换**，产出与 `chunker.chunk_document` 同构的 Chunk
（title_path/page_span/text/spans），下游（ChunkIndex / BM25 / claims 抽取 /
L3 检索 / report 层）零改动复用。表格文本化后进入 chunk.text 即可被检索。

对应 MinerU 3.x（3.4.5 实测）content_list schema：
    通用: {type, bbox[x0,y0,x1,y1], page_idx(0基)}
    text:       {text, text_level:"1"|"2"(标题)|无(正文)}   text 内含 <sup> 等少量 html
    table:      {table_caption[], table_footnote[], table_body(html), img_path}
    equation:   {text: latex("$$...$$"), text_format:"latex", img_path}
    chart:      {content, chart_caption[], chart_footnote[], img_path}
    image:      {image_caption[], image_footnote[], img_path}
    list:       {sub_type("ref_text"=参考文献), list_items[]}
    page_footnote / aside_text / page_number: 噪声/脚注

标题层级重建：MinerU text_level 只有 1(整篇 title)/2(所有节) 两档，不够细。
层级用**标题文本的数字编号**重建（"3.2" → depth 2），无编号节（Abstract/
References）归顶层。标题 label 仿 heading._heading_label 的 "L{d} {no} · {text}"
格式，保证 top_section / report.sections / home_section 归一一致。
"""
from __future__ import annotations

import html as _html
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from paperpilot.models.schema import Chunk, ChunkSpan

# 一个 chunk 的软上限（字符）。正文按元素累积，超限切新 chunk（表格是原子不拆）。
CHUNK_TARGET = 4000

_REF_MARKERS = ("References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY")
_NUM_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)[\s.]+(.*)$", re.S)
_NOISE_TYPES = {"page_number", "aside_text"}


# ── 文本净化 ──────────────────────────────────────────────────────────────


def _strip_sup_sub(s: str) -> str:
    """整体删除 <sup>/<sub> 角标（连同内容：脚注数字/†/作者标记，检索无价值）。"""
    return re.sub(r"<(sup|sub)>.*?</\1>", "", s, flags=re.S)


def _strip_tags(s: str) -> str:
    s = _strip_sup_sub(s or "")
    s = re.sub(r"<[^>]*>", "", s)
    return _html.unescape(s)


def _strip_html(s: str) -> str:
    s = _strip_sup_sub(s or "")
    s = re.sub(r"<[^>]*>", "", s)
    s = _html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ── 表格 HTML → markdown 表 ──────────────────────────────────────────────


# ── 表格 HTML → markdown 表（P1 改造，2026-09-12）─────────────────────────
#
# 业界规范做法（见 qa/recall/TABLE_FEED_20260912.md 的调研）：
#   1) **矩阵投影**：按 rowspan/colspan 物理展开成规整二维网格。
#      必要性：HTML 里被 rowspan 覆盖的后续行**少写 <td>**，逐行直转必然列错位。
#   2) **多级表头折叠成单行**：如 `Story Generation/Pearson`。
#      必要性：旧实现把"首行"当表头；MinerU 的分组表头行（colspan）在首行时，
#      真列名会沦为数据行 → **列归属彻底丢失**（实测 2002.05058 的 Table 1/2，
#      表里有两个 Pearson 列，模型无法判断数值属于哪一列）。
#   3) 表头行数优先 `<thead>`，其次"前导全 <th> 行"，再次"首行含 colspan>1 → 2 行"。


class _TableParser(HTMLParser):
    """把 MinerU table_body(html) 解析成「单元格 + 合并信息」。

    rows: `[[(text, colspan, rowspan, is_th), ...], ...]`
    thead_rows: `<thead>` 内的行数（表头行数最可靠的来源）。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self.thead_rows = 0
        self._row: list[tuple[str, int, int, bool]] | None = None
        self._cell: list[str] | None = None
        self._span: tuple[int, int] = (1, 1)
        self._is_th = False
        self._in_thead = False

    @staticmethod
    def _int_attr(v: str | None) -> int:
        """rowspan/colspan → int；缺失/非法/`0`（HTML4 跨到表尾）一律按 1。"""
        try:
            n = int(str(v or "").strip())
        except (TypeError, ValueError):
            return 1
        return n if n > 0 else 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._is_th = tag == "th"
            self._span = (self._int_attr(a.get("colspan")), self._int_attr(a.get("rowspan")))

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(("".join(self._cell).strip(), self._span[0], self._span[1],
                              self._is_th))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            if self._in_thead:
                self.thead_rows += 1
            self._row = None
        elif tag == "thead":
            self._in_thead = False

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _cell_clean(s: str) -> str:
    """单元格净化：去角标/标签，压缩空白，保留 LaTeX 符号文本。"""
    s = _strip_tags(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _esc_pipe(s: str) -> str:
    """markdown 表格单元格内的 `|` 必须转义为 `\\|`，否则会**串列**。

    实测 MinerU 会产出 `83.3|0.21` 这类含竖线的单元格（2002.08307 等 8 张表），
    不转义时下游按 `|` 切分 → 列数错乱（旧实现同样有此隐患）。
    """
    return s.replace("|", "\\|")


def _grid_project(rows: list[list[tuple[str, int, int, bool]]]) -> list[list[str]]:
    """矩阵投影：按 colspan/rowspan 把合并单元格物理展开成规整二维网格。

    关键点：游标遇到"被上方 rowspan 占走的格子"必须**跳过**，否则所有 rowspan 表必错位。
    用 dict 承载并动态扩张列数，避免"∑colspan 取 max"低估列数而裁掉右侧单元格。
    """
    if not rows:
        return []
    occ: dict[tuple[int, int], str] = {}
    max_c = 0
    for ri, row in enumerate(rows):
        ci = 0
        for txt, cs, rs, _th in row:
            while (ri, ci) in occ:          # 跳过被上方 rowspan 占走的格子
                ci += 1
            for dr in range(rs):
                for dc in range(cs):
                    occ[(ri + dr, ci + dc)] = txt
            max_c = max(max_c, ci + cs - 1)
            ci += cs
    n_rows, n_cols = len(rows), max_c + 1
    return [[occ.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def _header_row_count(rows: list[list[tuple[str, int, int, bool]]], thead_rows: int) -> int:
    """表头行数：`<thead>` → 前导全 `<th>` 行 → 首行含 colspan>1（分组表头）→ 1。"""
    n = len(rows)
    if n == 0:
        return 0
    cap = n - 1 if n > 1 else 1          # 至少留一行给数据
    if thead_rows:
        return max(1, min(thead_rows, cap))
    k = 0
    for row in rows:
        if row and all(c[3] for c in row):
            k += 1
        else:
            break
    if k:
        return max(1, min(k, cap))
    if any(c[1] > 1 for c in rows[0]) and n >= 2:
        return 2                         # 分组表头行（colspan）+ 真列名行
    return 1


def _fold_header(grid: list[list[str]], h: int) -> list[str]:
    """前 h 行表头沿列拼成单行复合列名（相邻重复值去重，避免"电压/电压"）。"""
    n_cols = len(grid[0]) if grid else 0
    out: list[str] = []
    for c in range(n_cols):
        parts: list[str] = []
        for r in range(min(h, len(grid))):
            v = _cell_clean(grid[r][c])
            if v and (not parts or parts[-1] != v):
                parts.append(v)
        out.append("/".join(parts))
    return out


def _table_to_md_v1(body_html: str) -> str:
    """**改造前**的实现（首行当表头、忽略 colspan/rowspan）。

    仅用于 `PAPERPILOT_TABLE_V1=1` 的**同日配对 A/B**（评测用，线上不设此变量）。
    """
    p = _TableParser()
    try:
        p.feed(body_html or "")
        p.close()
    except Exception:  # noqa: BLE001
        return _strip_tags(body_html or "")
    if not p.rows:
        return _strip_tags(body_html or "")
    lines: list[str] = []
    for i, row in enumerate(p.rows):
        cells = [_cell_clean(t) for (t, _cs, _rs, _th) in row]
        if not any(cells):
            continue
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "---|" * len(row))
    return "\n".join(lines)


def table_to_md(body_html: str) -> str:
    """HTML table → markdown 管道表（**矩阵投影 + 多级表头折叠成单行**）。

    首要目标是**保住列语义**（哪个值属于哪一列）；空/异常退化为纯文本。
    `PAPERPILOT_TABLE_V1=1` → 走改造前实现（评测用配对 A/B）。
    """
    if os.environ.get("PAPERPILOT_TABLE_V1") == "1":
        return _table_to_md_v1(body_html)
    p = _TableParser()
    try:
        p.feed(body_html or "")
        p.close()
    except Exception:  # noqa: BLE001  异常 → 退化纯文本
        return _strip_tags(body_html or "")
    if not p.rows:
        return _strip_tags(body_html or "")
    grid = _grid_project(p.rows)
    if not grid:
        return _strip_tags(body_html or "")
    # 去掉整列为空的列（MinerU 常见尾部空列）
    keep = [c for c in range(len(grid[0])) if any((r[c] or "").strip() for r in grid)]
    grid = [[row[c] for c in keep] for row in grid]
    if not grid or not grid[0]:
        return _strip_tags(body_html or "")
    h = _header_row_count(p.rows, p.thead_rows)
    n_cols = len(grid[0])
    header = [_esc_pipe(c) for c in (_fold_header(grid, h) if h else [""] * n_cols)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * n_cols]
    for row in (grid[h:] if h < len(grid) else []):
        cells = [_esc_pipe(_cell_clean(c)) for c in row]
        if not any(cells):
            continue
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── 元素文本化（元素 → 检索友好文本段） ─────────────────────────────────────

# 外部块（由 MinerU 产物补进检索视图的表格/公式）的 chunk_id 前缀，
# 供 document_cache 生成 id、下游（前端/引用解析）识别这类块。
#
# ⚠️ 不要据此在 RRF 里"给外部块屏蔽 BM25 路"：曾因表块 BM25 **位次**偏差尝试过，
# 实测**证否并已回退**——混池 RRF 里外部块恰恰靠 BM25 路挣分（`1/(60+rb)` 仍是正分），
# 屏蔽后目标表块 top-12 命中 14/21→**4/21**、正文召回 R@12 0.972→0.952。
# 详见 `qa/recall/TABLE_POLICY_AB_20260911.md`。
EXT_CHUNK_PREFIX = "xtbl-"

# LaTeX 里的纯结构/排版命令（无检索价值，直接删除）
# 注：**不含 begin/end/array** —— 环境（`\begin{array}{rl}`）必须整段先删，
# 否则 `\begin` 被删后残留 `{array}`，`array` 会作为假词留在文本里（golden case 抓到过）
_LATEX_NOISE = (r"left|right|bigl|bigr|big|Big|bigg|Bigg|cdot|ldots|cdots|dots|quad|qquad|"
                r"hline|nonumber|label|ref|hspace|vspace|displaystyle|"
                r"limits|nolimits|tag|textstyle")
# 命令名本身即内容（只脱壳，保留花括号里的东西）
_LATEX_WRAP = r"mathrm|mathbf|mathit|mathsf|mathtt|textsc|text|operatorname|boldsymbol|mathcal|mathbb"


def latex_to_text(raw: str) -> str:
    """LaTeX（MinerU 形态）→ 可检索文本。

    动机：MinerU 的 equation 元素是 LaTeX，且**在花括号内外逐字母插空格**
    （`\\mathrm { R N N }`），导致 BM25 连 `RNN` 都匹配不上、向量侧语义也弱。

    处理顺序（②③不可颠倒：先脱壳再删噪声，否则 `\\mathrm` 会被当噪声删掉）：
      ① 贴住花括号（MinerU 的空格是主要噪声源）
      ② 包内容的命令脱壳（`\\mathrm{RNN}` → `RNN`）
      ③ 结构/噪声命令删除
      ④ 命令 → 名字本身（`\\alpha` → `alpha`、`\\Phi` → `Phi`）
      ⑤ 上下标 `_{x}`/`^{x}` → ` x `（保留内容，避免粘连出假词）
      ⑥ 残余符号 `{}&\\_^` → 空格
      ⑦ **合并被拆开的单字母序列**（只合有把握的）：`R N N`→`RNN`、`S o f t m a x`→`Softmax`；
         全小写串（`x i t`）**不合并**，否则造出假词。
         （★关键★：漏掉这步会"越规范越差"，见 `qa/recall/_latex_norm_cases.py` 的 golden case）

    ⚠️ 仅对 LaTeX 源使用：⑦ 会把普通文本里的 `a b` 也合并成 `ab`。
    """
    t = raw or ""
    t = t.replace("$$", " ").replace("$", " ")
    t = re.sub(r"\s*\{\s*", "{", t)
    t = re.sub(r"\s*\}\s*", "}", t)
    # 环境整段删除（必须在噪声命令之前）
    t = re.sub(r"\\begin\{[^}]*\}(?:\{[^}]*\})*", " ", t)
    t = re.sub(r"\\end\{[^}]*\}", " ", t)
    t = re.sub(r"\\(?:%s)\s*" % _LATEX_WRAP, "", t)
    t = re.sub(r"\\(?:%s)\b" % _LATEX_NOISE, " ", t)
    t = re.sub(r"\\([A-Za-z]+)", r"\1", t)
    t = re.sub(r"[_^]\s*\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"[{}&\\]", " ", t)
    t = re.sub(r"[_^]", " ", t)
    t = " ".join(t.split())
    # ⑦ 合并被 MinerU 拆开的单字母序列（**只合并有把握的**）：
    #    · 全大写串 `R N N` / `L S T M` → 缩略语，合并
    #    · 首字母大写的 `S o f t m a x`（其余小写、长度≥3）→ 单词，合并
    #    · 全小写的 `x i t`（x 的下标 i、上标 t）→ **不合并**，合并只会造出假词
    def _merge(m: re.Match) -> str:
        parts = m.group(0).split(" ")
        if len(parts) >= 2 and all(x.isupper() for x in parts):
            return "".join(parts)          # R N N → RNN
        # 混合大小写（S o f t m a x）与变量序列（W q y i）靠大小写分不开 →
        # 用**长度门槛 ≥5** 区分：真词（Softmax/Sigmoid）能过，短变量序列不会
        if len(parts) >= 5 and parts[0].isupper() and all(x.islower() for x in parts[1:]):
            return "".join(parts)
        return m.group(0)

    return re.sub(r"\b([A-Za-z])(?: ([A-Za-z]))+\b", _merge, t)



def element_text(el: dict[str, Any]) -> str | None:
    """单个 content_list 元素 → 文本段；噪声/无关（页码/边栏/参考文献图注缺失）→ None。"""
    typ = el.get("type")
    if typ in _NOISE_TYPES:
        return None
    if typ == "text":
        return _strip_html(el.get("text") or "")
    if typ == "table":
        cap = _strip_html(" ".join(el.get("table_caption") or []))
        body = table_to_md(el.get("table_body") or "")
        fn = _strip_html(" ".join(el.get("table_footnote") or []))
        return "\n".join(x for x in (cap, body, fn) if x) or None
    if typ == "equation":
        tex = latex_to_text(el.get("text") or "")
        return f"公式: {tex}" if tex else None
    if typ == "chart":
        cap = _strip_html(" ".join(el.get("chart_caption") or []))
        content = _strip_html(el.get("content") or "")
        return "\n".join(x for x in (content, cap) if x) or None
    if typ == "image":
        cap = _strip_html(" ".join(el.get("image_caption") or []))
        return cap or None  # 无图注的图不进文本（VLM 图表理解前）
    if typ == "list":
        if el.get("sub_type") == "ref_text":
            return None  # 参考文献不进正文（与 analyzer.extractable 一致）
        items = [_strip_html(i) for i in (el.get("list_items") or [])]
        items = [i for i in items if i]
        return "\n".join(items) if items else None
    if typ == "page_footnote":
        return _strip_html(el.get("text") or "") or None
    raw = el.get("text")
    return _strip_html(raw) if isinstance(raw, str) and raw.strip() else None


def _is_heading(el: dict[str, Any]) -> bool:
    """MinerU 标题：type=text 且带 text_level（'1' 整篇题 / '2' 各级节）。"""
    return el.get("type") == "text" and bool(el.get("text_level"))


def _parse_heading(text: str) -> tuple[str, int]:
    """标题文本 → (label, depth)。编号(3.2)→depth 段数；无编号→顶层 depth 1。"""
    text = _strip_html(text)
    m = _NUM_PREFIX.match(text)
    if m:
        no, _rest = m.group(1), m.group(2)
        depth = no.count(".") + 1
        return f"L{depth} {no} · {text}", depth
    return f"L1 · {text}", 1


def _is_ref_heading(label: str) -> bool:
    tail = label.split("·")[-1].strip()
    return tail in _REF_MARKERS


# ── content_list → Chunk[] ───────────────────────────────────────────────


def chunks_from_mineru(content_list: list[dict[str, Any]]) -> list[Chunk]:
    """MinerU content_list（阅读序）→ Chunk[]。

    第一遍：元素归到标题组（标题树由编号重建）；第二遍：每组按 CHUNK_TARGET
    以元素为原子切分。论文整篇标题不进树；参考文献区丢弃；任何标题前正文归
    PREAMBLE（与 chunker.extractable 语义一致）。
    """
    # ── 第一遍：分组 ──
    groups: list[tuple[list[str], list[dict[str, Any]]]] = []  # (title_path, elems)
    stack: list[tuple[int, str]] = []   # (depth, label)
    cur: list[dict[str, Any]] | None = None
    cur_path: list[str] | None = None
    in_refs = False

    def open_group(path: list[str], lead_el: dict[str, Any]) -> None:
        nonlocal cur, cur_path
        groups.append((list(path), [lead_el]))
        cur = groups[-1][1]
        cur_path = groups[-1][0]

    def ensure_group() -> None:
        if cur is None:
            open_group(["(PREAMBLE)"], {})

    for el in content_list:
        if in_refs:
            continue
        if _is_heading(el):
            if str(el.get("text_level")) == "1":
                continue  # 整篇论文标题（元数据），不进检索树
            label, depth = _parse_heading(el.get("text") or "")
            if _is_ref_heading(label):
                in_refs = True
                cur = None
                continue
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, label))
            cur = None  # 强制开新组
            open_group([lab for _, lab in stack], el)
            continue
        txt = element_text(el)
        if txt is None:
            continue
        ensure_group()
        assert cur is not None and cur_path is not None
        cur.append(el)

    # ── 第二遍：分块（标题组内按 CHUNK_TARGET 以元素为原子切分） ──
    chunks: list[Chunk] = []
    seq = 0
    for path, elems in groups:
        elems = [e for e in elems if e]  # 去掉 PREAMBLE 占位空元素
        if not elems:
            continue
        pieces: list[list[dict[str, Any]]] = [[]]
        lens: list[int] = [0]
        for e in elems:
            t = element_text(e) or ""
            if pieces[-1] and lens[-1] + len(t) > CHUNK_TARGET:
                pieces.append([])
                lens.append(0)
            pieces[-1].append(e)
            lens[-1] += len(t)
        for piece in pieces:
            text = "\n".join(filter(None, (element_text(x) or "" for x in piece))).strip()
            if not text:
                continue
            pages = sorted({int(x.get("page_idx", 0)) + 1 for x in piece})
            chunks.append(Chunk(
                chunk_id=f"c{seq}",
                title_path=list(path),
                page_span=(pages[0], pages[-1]) if pages else (0, 0),
                text=text,
                n_blocks=len(piece),
                spans=[ChunkSpan(block_id=f"m{seq}_{i}",
                                 page=int(x.get("page_idx", 0)) + 1,
                                 text=(x.get("text") or "")[:160])
                       for i, x in enumerate(piece, 1)],
            ))
            seq += 1
    return chunks


def _find_content_list(work_dir: str | Path) -> list[dict[str, Any]] | None:
    """读 MinerU 产物目录下的 content_list（找不到/非列表返回 None）。"""
    base = Path(work_dir)
    if not base.is_dir():
        return None
    for p in base.rglob("*_content_list.json"):
        cl = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(cl, list):
            return cl
    return None


def chunks_from_mineru_dir(work_dir: str | Path) -> list[Chunk] | None:
    """从 `mineru -o <dir>` 产物目录读 content_list 转 Chunk[]（找不到返回 None）。"""
    cl = _find_content_list(work_dir)
    return chunks_from_mineru(cl) if cl is not None else None


def title_from_mineru_dir(work_dir: str | Path) -> str:
    """从 MinerU 产物取整篇论文标题（text_level=='1' 元素），找不到返回空串。"""
    cl = _find_content_list(work_dir)
    if cl is None:
        return ""
    for el in cl:
        if el.get("type") == "text" and str(el.get("text_level")) == "1":
            t = _strip_html(el.get("text") or "")
            return t if t else ""
    return ""


def figures_from_mineru_dir(work_dir: str | Path) -> list[dict[str, Any]] | None:
    """从 MinerU 产物目录产 figures（report 图表一览用）；无产物返回 None。"""
    cl = _find_content_list(work_dir)
    return extract_figures_from_mineru(cl) if cl is not None else None


# ── MinerU content_list → figures（report 图表一览 / 读图指南原料） ───────────

_CAP_NUM_RE = re.compile(
    r"^(Table|Tab\.?|Figure|Fig\.?)\s+([A-Za-z]?\d+(?:\.\d+)*[a-z]?)", re.I)


def extract_figures_from_mineru(content_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 content_list 的 table/chart/image 元素产 figures（与 figures.extract_figures 同构）。

    Returns: [{id, kind, num, page, caption, refs}]
      id = caption 自带编号（"Table 1"/"Figure 3a"）；编号缺失 → "Table t1"/"Figure f1" 兜底
      refs = content_list 中引用该编号的正文文本段（图自身后）
    """
    figs: list[dict[str, Any]] = []
    for i, el in enumerate(content_list, 1):
        typ = el.get("type")
        if typ == "table":
            cap = _strip_html(" ".join(el.get("table_caption") or []))
            kind, prefix = "Table", f"Table t{i}"
        elif typ in ("chart", "image"):
            cap = _strip_html(" ".join(el.get("chart_caption") or el.get("image_caption") or []))
            kind, prefix = "Figure", f"Figure f{i}"
        else:
            continue
        m = _CAP_NUM_RE.match(cap) if cap else None
        if m:
            kind = "Table" if m.group(1).lower().startswith("tab") else "Figure"
            fig_id = f"{kind} {m.group(2)}"   # 规范 "Fig. 1" → "Figure 1"
            num = m.group(2)
        else:
            fig_id, num = prefix, "t" + str(i) if kind == "Table" else "f" + str(i)
        figs.append({"id": fig_id, "kind": kind.lower(), "num": num,
                     "page": int(el.get("page_idx", 0)) + 1,
                     "caption": cap or "",
                     "refs": []})

    # 收集正文引用段（与 figures.extract_figures 同策略：引用在其后的正文段落）
    for fig in figs:
        num = re.escape(str(fig["num"]))
        pat = re.compile(rf"\b(?:Figure|Fig\.?|Table|Tab\.?)\s*{num}[a-z]?\b", re.I)
        pref = str(fig["id"])
        for el in content_list:
            if el.get("type") != "text":
                continue
            if int(el.get("page_idx", 0)) + 1 < fig["page"] - 1:
                continue
            t = _strip_html(el.get("text") or "")
            if not pat.search(t):
                continue
            if t.startswith(pref):
                continue
            t = " ".join(t.split())
            if len(t) > 15 and t not in fig["refs"]:
                fig["refs"].append(t)
    return figs
