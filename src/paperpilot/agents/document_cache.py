"""QA 层的正文 chunk 文档缓存（进程级，跨节点复用 parse）。

为什么独立成模块：L3 全文检索、ChunkIndex 向量化都要用到"这篇论文切出的
extractable chunks"——之前各节点各自 parse，浪费。这里统一 lru_cache：
同一 pdf 首次 parse+chunk，之后所有节点复用。

两套视图（2026-09-10 起，**同一批 chunk_id，只是检索文本更厚**）：
    ordered_chunks(pdf)    报告/溯源视图：pymupdf 切块，claims / report / cites 用它
    retrieval_chunks(pdf)  检索视图：在上面的基础上，把 MinerU 的表格/公式文本
                           按页注入到该页首个 chunk → 索引与作答用它
    两者 chunk_id / 页码完全一致 → cites 只有一套 id 空间，溯源闭环不受影响。
    无 MinerU 产物的论文 / 注册的数据源（无额外块时）：retrieval_chunks == ordered_chunks。

约定（与 chunker/analyzer 一致）：
    top_section(chunk)  → chunk 所属顶层节名（title_path[0] 的 "· " 后半段），
    与 report.groups 的 sections 字段 / ClaimHit.home_section 统一。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from paperpilot import sources
from paperpilot.models.schema import Chunk
from paperpilot.tools import analyzer
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.pdf_parser import parse_pdf

# document_cache.py → agents/ → paperpilot/ → src/ → 根
ROOT = Path(__file__).resolve().parents[3]
PAPERS_DIR = ROOT / "assets" / "papers"
MINERU_OUT = ROOT / "assets/artifacts/out_mineru"   # MinerU 解析产物根（env 开启时优先）

MAX_CHUNK_LEN = 4000   # 与 pipeline / run_qa_eval 的二级切分阈值一致

# MinerU 的 `table_caption` 里"真正表标题"的锚点（见 `_clean_table_captions`）。
# ⚠️ 必须同时接受**罗马数字**表号（`TABLE IV`）与带后缀的阿拉伯号（`Table 5a`）：
#    首版写成 `Table\s+[IVXLC]*\d+` 只认"罗马前缀 + 数字"，于是 `TABLE IV COMPARISON`
#    不匹配 → **整段真 caption 被删掉**（自测抓到，语料里确有罗马数字表的论文）。
_TBL_CAP_RE = re.compile(r"Table\s+(?:[IVXLC]+|\d+(?:\.\d+)*[A-Za-z]?)", re.I)


def _clean_table_captions(el: dict) -> list[str]:
    """只保留 `table_caption` 里**真正的表标题**片段。

    MinerU 的 `table_caption` 会混进两类噪声（实测，`qa/recall/MINERU_COVERAGE_20260912.md` §5）：
      1. 上方**图的标题**：`['Figure 6: Attention Distributions...', 'Table 4: Average prec...']`
         → 进块污染 BM25/向量，且会让"按编号定位目标表"的评测口径取到**错的号**；
      2. **表头行文字**：`['Age Race Gender', 'Table 3: Annotator agreement...']`。
    两类都能靠"从首个 `Table <编号>` 处截断"清掉；不含 `Table` 的片段整段丢弃。
    """
    out: list[str] = []
    for c in el.get("table_caption") or []:
        s = str(c).strip()
        m = _TBL_CAP_RE.search(s)
        if m:
            out.append(s[m.start():].strip())
    return out


def _table_header_cells(text: str) -> list[str]:
    """块文本的**表头单元格**（首行 md 表格行）；保序去重。

    给无 caption 的表块补身份时用：同一张表被 MinerU 切成多块时各块表头相同 →
    可互相借 caption；借不到就用列名拼一行身份串（不加长、不加新信息）。
    """
    for ln in str(text).splitlines():
        if "|" not in ln or ln.strip().startswith("|--"):
            continue
        cells: list[str] = []
        for c in ln.split("|"):
            c = c.strip()
            if c and c not in cells:
                cells.append(c)
        if len(cells) >= 2:
            return cells
    return []


def _md_rows(text: str) -> list[list[str]]:
    """md 表格的**数据行**（逐行拆单元格；跳过分隔行/非表格行）。"""
    rows: list[list[str]] = []
    for ln in str(text).splitlines():
        s = ln.strip()
        if not s.startswith("|") or set(s) <= set("|-: "):
            continue
        rows.append([c.strip() for c in s.strip("|").split("|")])
    return rows


def _table_summary(text: str, max_len: int = 600) -> str:
    """表块的**语义摘要**（P2 双写：只喂向量侧，BM25/作答仍用原表）。

    v1（标题 + 列名 + 首列取值）**实测更差**（`qa/recall/P2_DUALWRITE_20260912.md`）：
    同篇表块间平均余弦 0.593→0.616、目标 margin +0.033→+0.018、并集增益 +4→+2。
    机制：**单篇检索靠"同篇内区分度"**，而标题/列名/实体名在同篇的表之间高度重合
    → 向量被平均化、目标失去优势。同源现象：加 150 字通用填充、加 refs 全量都变差。

    v2 因此**保留具体数值**——每个数值在同篇内几乎唯一，正是区分目标表的东西；
    同时保留（压缩的）列名，因为提问会用度量词（accuracy / F1 / AUC）。

    成分：表标题（含 `Table N`） + 列名 + **行标签全集**（实体名，最多 16 个）
    + **前 6 行的数值样例**（`标签: v1 v2 …`）。
    """
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    cap = lines[0] if lines and not lines[0].startswith("|") else ""
    rows = _md_rows(text)
    cols: list[str] = []
    labels: list[str] = []
    samples: list[str] = []
    if rows:
        cols = [c for c in rows[0] if c][:12]
        for r in rows[1:]:
            if not r:
                continue
            label = (r[0] or "").strip()
            vals = [c for c in r[1:] if c][:5]
            if label and not re.fullmatch(r"[\d.\-–—%]+", label) and label not in labels:
                labels.append(label)
            if vals:
                samples.append(f"{label}: {' '.join(vals)}" if label else " ".join(vals))
            if len(labels) >= 16 and len(samples) >= 6:
                break
    parts = [f"表格：{cap}"] if cap else (["表格"] if cols else [])
    if cols:
        parts.append("列：" + "/".join(cols))
    if labels:
        parts.append("行：" + "、".join(labels[:16]))
    if samples:
        parts.append("行样例：" + "；".join(samples[:6]))
    return "。".join(parts)[:max_len]


# ── 数据源（非 PDF 的评测源）──────────────────────────────────────────────────
#
# 生产**不再用文件名前缀猜能力**（2026-09-22 重构）：以前这里有个
# `is_qasper(pdf)`，被用来回答"chunk 从哪来 / 有没有文件 / 标题从哪来 / 产物怎么打标"
# 四件事 → 评测文本源因此穿透了 7 个生产文件。现在统一走 `paperpilot.sources`
# 注册表：**未注册任何源 → 原生 PDF 路径**（下面的 `_chunks_via_mineru` / `parse_pdf`）。


# ── 解析源：MinerU 主 · pymupdf 容灾（2026-09-22 决策）────────────────────────
#
# ⚠️ 这里有两个**不同的**开关，别混：
#     `PAPERPILOT_MINERU`     （默认 "1"）→ MinerU **跑不跑**（ingest / workflow ③）
#     `PAPERPILOT_USE_MINERU` （默认 "1"）→ MinerU 是否作 **chunk 骨架**
#                                          （`=0` 退回 pymupdf，容灾 / A-B 用）
#
# 为什么改用 MinerU 做骨架（同篇双源实测 `1701.00185v1`）：
#     · pymupdf **漏判** `4. Experiments` → 4.1~4.6 六个子节**错挂到 3. Methodology 下**；
#       还把 2 条**参考文献条目误判成 APPENDIX**（`heading.py` 的 `逗号≥2` 防护没兜住
#       "R. Ward, Deep sentence embedding…"）；
#     · MinerU 两个都对（参考文献是 `list/ref_text` 独立类型，标题是版面模型直接给的）；
#     · 产物更全：claims 154 vs 140、骨架边 86 vs 77、图表 16 vs 12；
#     · **溯源回核 MinerU ✓154 ✗0 vs pymupdf ✓135 ✗5**（表格被撕裂 → evidence 对不上）；
#     · 符号保真：MinerU 保住 `STC²` 上标，pymupdf 退化成 `STC2`。
# MinerU 的代价：① 慢（必须先跑完 MinerU 才能建报告链 → worker 转串行）；
#                 ② 会**误标**标题（无防护）→ 待加轻校验。


def mineru_skeleton_enabled() -> bool:
    """MinerU 是否作 **chunk 骨架**（默认**是**；`PAPERPILOT_USE_MINERU=0` 退回 pymupdf）。

    与 `mock_llm.mineru_enabled()`（MinerU **跑不跑**）是两件事：
        `PAPERPILOT_MINERU=0`      → 根本不执行 MinerU（演示/无 GPU）
        `PAPERPILOT_USE_MINERU=0`  → MinerU 照跑，但只做检索视图注入，骨架仍用 pymupdf
    """
    return os.environ.get("PAPERPILOT_USE_MINERU", "1") != "0"


# ── 两条切块路（2026-09-22：默认 MinerU / 备用 pymupdf，**互不相交**）──────────
#
#     pdf ──► 路由 ──┬── **默认路** `_chunks_via_mineru`  ──► Chunk[]
#                    └── **备用路** `_chunks_via_pymupdf` ──► Chunk[]  ← 默认路失败才走
#
# 为什么必须"不相交"：两条路各自**起止明确** ——
#     默认路只读 `assets/artifacts/out_mineru/<stem>/`（不碰 PDF）；
#     备用路只读 `assets/papers/<pdf>`（不碰 out_mineru、不做表格注入）。
# 于是默认路任何环节出问题都只需返回 None，由路由改走备用路，
# **不会出现"半条 MinerU、半条 pymupdf"的混合产物**。


def _chunks_via_mineru(pdf: str) -> list[Chunk] | None:
    """**默认路**：MinerU 产物 → Chunk[]；**不可用一律返回 None**（绝不抛）。

    不可用 = 产物目录不存在 / content_list 损坏 / 解析抛错 / **产不出有效 chunk**。

    ⚠️ 最后一档别漏（2026-09-22 修的潜伏不一致）：旧实现返回空 list，而调用方用
    `is not None` 判断 → **返回空 chunk 而不是降级**，与 `mineru_status()` 判
    `degraded` 的口径自相矛盾（那篇会"有 chunk 但一条都搜不到"）。
    References 已在桥内丢弃；PREAMBLE 用 extractable 过滤，与备用路一致。
    """
    base = MINERU_OUT / Path(pdf).stem
    if not base.is_dir():
        return None
    from paperpilot.tools.mineru_bridge import chunks_from_mineru_dir
    try:
        mc = chunks_from_mineru_dir(base)
    except Exception:  # noqa: BLE001  产物损坏 → 交备用路
        return None
    if not mc:
        return None
    return analyzer.extractable(mc) or None


def _chunks_via_pymupdf(pdf: str) -> list[Chunk]:
    """**备用路（容灾）**：`assets/papers/<pdf>` → pymupdf 版面块 → 标题树切块。

    与默认路**完全不相交**：不读 `out_mineru/`、不做表格/公式注入，
    只依赖 PDF 文件本身 —— 所以 MinerU 环境坏掉/超时/产物损坏时它仍然可用。

    代价（诚实说明）：**表格数值拿不到**（pymupdf 会把表撕成碎片）→ 凡答案出自表格
    的问题会退化。这正是"降级"要**标记出来**的原因（见 `mineru_status`）。
    """
    result = parse_pdf(str(PAPERS_DIR / pdf))
    chunks = chunk_document(result["blocks"], max_len=MAX_CHUNK_LEN)
    return analyzer.extractable(chunks)


@lru_cache(maxsize=32)
def _mineru_status_cached(pdf: str, stamp: str) -> tuple[str, str]:
    """MinerU 骨架的当前状态 → (状态, 详情)。见 `mineru_status`。"""
    if not mineru_skeleton_enabled():
        return "disabled", "PAPERPILOT_USE_MINERU=0：骨架固定 pymupdf"
    from paperpilot.tools.mock_llm import mineru_enabled
    if not mineru_enabled():
        return "disabled", "PAPERPILOT_MINERU=0：不执行 MinerU"
    if sources.resolve(pdf) is not None:
        return "ok", "注册的数据源自带 chunk，与 MinerU 无关"
    base = MINERU_OUT / Path(pdf).stem
    if not base.is_dir():
        # 没有产物目录 → 两种性质完全不同的情况，**必须分开**（2026-09-22 补）：
        #   · MinerU **跑过但失败**（`ingest.json` 记了 status=failed）→ **degraded**
        #     （是故障，用户该知道"本篇表值会缺"）
        #   · **还没跑过** → **pending**（待办，跑一次 ingest 即可）
        # 旧实现只看产物目录 → 把"跑了但失败"也报成 pending（"还没跑"），
        # 于是**真失败被伪装成待办**，降级提示不会出现。
        try:
            from paperpilot import ingest
            m = (ingest.read_meta(Path(pdf).stem) or {}).get("mineru") or {}
        except Exception:  # noqa: BLE001  meta 读不到不影响主判定
            m = {}
        if str(m.get("status")) == "failed":
            return "degraded", (f"MinerU 执行失败："
                                f"{str(m.get('reason') or '未知原因')[:110]}")
        return "pending", "尚未摄取（无 out_mineru/<stem>/）"
    from paperpilot.tools.mineru_bridge import chunks_from_mineru_dir
    try:
        mc = chunks_from_mineru_dir(base)
    except Exception as e:  # noqa: BLE001
        return "degraded", f"产物损坏（{type(e).__name__}: {str(e)[:80]}）"
    if mc is None:
        return "degraded", "content_list 缺失或不可读"
    if not analyzer.extractable(mc):
        return "degraded", "产不出有效 chunk（全被过滤）"
    return "ok", ""


def mineru_status(pdf: str) -> tuple[str, str]:
    """MinerU 骨架状态 → `(状态, 详情)`（按 (文件名, 文件版本) 缓存）。

    四态**必须分清**（2026-09-22：之前把"没跑"错算成"降级"，口径错了）：

        "ok"        产物可用 → MinerU 作骨架（正常态）
        "pending"   **尚未摄取**（没跑过 ingest）→ 这是**待办**，跑一次 ingest 即可；
                    不是故障、不是降级。正常流程（`worker`）本来就会先跑 MinerU。
        "degraded"  **MinerU 试过但失败**（产物损坏 / content_list 不可读 / 产不出 chunk）
                    → 用 pymupdf **容灾**顶上，**必须告警**（否则产物看起来完全正常）。
        "disabled"  显式不用（`PAPERPILOT_USE_MINERU=0` 或 `PAPERPILOT_MINERU=0`）
                    → 配置选择。

    对照组：只有 `degraded` 才写进 `PaperReport.degraded`；`pending` 只提示"该摄取"。
    """
    return _mineru_status_cached(pdf, _pdf_stamp(pdf))


def _pdf_stamp(pdf: str) -> str:
    """进程内缓存键的"文件版本"分量：`size:mtime_ns`（虚拟名/文件不存在 → ""）。

    为什么需要（2026-09-13 审查项 2 的另一半）：`lru_cache` 原本只按**文件名**缓存，
    同一进程里同名文件被替换后仍返回旧解析结果——与"产物按名寻址"是同一个病根。
    """
    from paperpilot import paper_identity
    fp = paper_identity.fingerprint(pdf)
    return f"{fp['size']}:{fp['mtime_ns']}" if fp else ""


@lru_cache(maxsize=16)
def _ordered_chunks_cached(pdf: str, stamp: str) -> tuple[Chunk, ...]:
    """返回该 pdf 的 extractable chunks（按正文顺序，已去 PREAMBLE/References/空）。

    **两条独立的路 + 一个路由**（2026-09-22）：
        ① 注册的数据源（测试侧注入，见 `paperpilot.sources`）自报 chunk；
        ② **默认路** MinerU → `_chunks_via_mineru`（只读 `out_mineru/`）；
        ③ 默认路不可用 → **备用路** pymupdf → `_chunks_via_pymupdf`（只读 PDF）。
    走的是哪条路、为何没走默认路 → 见 `current_source()` / `mineru_status()`。

    `stamp` 只参与缓存键（文件版本），不参与逻辑；见 `_pdf_stamp`。
    """
    src = sources.resolve(pdf)
    if src is not None:
        # 注册的数据源（评测文本集等）自报 chunk；确定性由源自己保证
        return tuple(src.chunks())
    if mineru_skeleton_enabled():
        mc = _chunks_via_mineru(pdf)
        if mc is not None:
            return tuple(mc)                  # 默认路：MinerU 产物
        # 默认路不可用 → 落到备用路（"降级"degraded / "尚未摄取"pending 见 mineru_status）
    return tuple(_chunks_via_pymupdf(pdf))     # 备用路：pymupdf（容灾）


def ordered_chunks(pdf: str) -> list[Chunk]:
    """该 pdf 的正文块（按 (文件名, 文件版本) 缓存；实现见 `_ordered_chunks_cached`）。"""
    return list(_ordered_chunks_cached(pdf, _pdf_stamp(pdf)))


# 兼容老调用方（如 `qa/recall/_ab_chunk_sources.py` 里的 `ordered_chunks.cache_clear()`）
ordered_chunks.cache_clear = _ordered_chunks_cached.cache_clear      # type: ignore[attr-defined]
ordered_chunks.cache_info = _ordered_chunks_cached.cache_info        # type: ignore[attr-defined]


@lru_cache(maxsize=64)
def _mineru_extra_by_page(stem: str) -> dict[int, list[str]]:
    """MinerU content_list 的 table/equation 元素 → {页码(1基): [元素文本]}。

    只取 pymupdf 拿不到（或拿得烂）的两类：table（结构化表值，pymupdf 会撕成碎片）
    与 equation（LaTeX）。**不取 MinerU 的正文 text**——pymupdf 已有，重复注入只会膨胀。
    """
    base = MINERU_OUT / stem
    if not base.is_dir():
        return {}
    from paperpilot.tools.mineru_bridge import _find_content_list, element_text
    try:
        cl = _find_content_list(base)
    except Exception:  # noqa: BLE001 产物损坏 → 不增强
        return {}
    if not cl:
        return {}
    out: dict[int, list[str]] = {}
    for el in cl:
        if el.get("type") not in ("table", "equation"):
            continue
        t = element_text(el)
        if not t:
            continue
        page = int(el.get("page_idx", 0)) + 1
        out.setdefault(page, []).append(t)
    return out


@lru_cache(maxsize=32)
def _retrieval_chunks_cached(pdf: str, stamp: str) -> tuple[Chunk, ...]:
    """**检索视图**：chunk_id/页码与 ordered_chunks 完全一致，仅文本更厚。

    背景（2026-09-10，方案 B3）：报告链无论如何需要 pymupdf（页码/版面/claims），
    但 MinerU 对**检索**更好（表格结构化、公式 LaTeX、图内文字不污染正文）。
    两者不是二选一：
      - chunk 空间只保留 pymupdf 一套 → cites / 前端高亮 / claims 锚点全部不动；
      - 表格/公式文本按页注入检索视图 → 表值能被召回、能被作答读到；
      - claims 抽取读的是 ordered_chunks → **表格只进检索不进 claims**（阶段 O 决策）。

    注入规则：MinerU 第 N 页（page_idx=N-1）的表格/公式，挂到**该页在文档序中
    第一个 chunk** 的末尾；同一页只挂一次（避免跨页 chunk 重复注入）。

    降级：QASPER / 无 MinerU 产物 / MinerU chunks 已成主源（env 遗留路径）→ 原样返回。
    """
    base = ordered_chunks(pdf)
    src = sources.resolve(pdf)
    if src is not None:
        # 数据源自报"检索视图额外块"（默认 []）；追加在正文块之后
        extra = list(src.extra_chunks())
        return list(base) + extra if extra else base
    if not base:
        return base
    if mineru_skeleton_enabled() and _chunks_via_mineru(pdf) is not None:
        return base  # MinerU 骨架：base 已是 MinerU chunks（表格已在内），不再二次注入
    # ⚠️ 走备用路（默认路不可用）时 base 是 pymupdf chunks → **这里必须继续注入**
    # MinerU 的表格/公式（若产物部分可用）：它正是"降级但仍尽量补回表值"的那一环。
    if os.environ.get("PAPERPILOT_MINERU_INJECT", "1") != "1":
        return base  # 评测用开关：关掉表格/公式注入（默认开 → 线上行为不变）
    extra = _mineru_extra_by_page(Path(pdf).stem)
    if not extra:
        return base
    used: set[int] = set()
    out: list[Chunk] = []
    for c in base:
        ps, pe = int(c.page_span[0]), int(c.page_span[1])
        add: list[str] = []
        for p in range(ps, pe + 1):
            if p in extra and p not in used:
                used.add(p)
                add.extend(extra[p])
        if add:
            c = c.model_copy(update={"text": c.text + "\n\n" + "\n".join(add)})
        out.append(c)
    return out


def retrieval_chunks(pdf: str) -> list[Chunk]:
    """检索视图（按 (文件名, 文件版本) 缓存；实现见 `_retrieval_chunks_cached`）。"""
    return list(_retrieval_chunks_cached(pdf, _pdf_stamp(pdf)))


retrieval_chunks.cache_clear = _retrieval_chunks_cached.cache_clear  # type: ignore[attr-defined]
retrieval_chunks.cache_info = _retrieval_chunks_cached.cache_info    # type: ignore[attr-defined]


@lru_cache(maxsize=32)
def external_table_chunks(mineru_dir: str) -> tuple[Chunk, ...]:
    """从 MinerU 产物的 content_list 造**检索视图专用**的表格/公式块（源无关的通用工具）。

    动机（通用）：有些数据源的正文里**表格数值没有任何文本形式**（如某评测集只给
    PNG + caption 的字符 span）→ 凡 gold 出自表格的题，在纯文本通道下**结构上不可达**
    （实测 500 题里 36 道，占 8.0pt）。若该篇的 arXiv 原始 PDF 恰好有 MinerU 产物，
    即可从中补出表/公式块。

    ⚠️ **是否调用、传哪个目录，由数据源自己决定** —— 本函数只做"给定 MinerU 目录 → 块"，
    生产包因此不含任何评测源的策略（2026-09-22 重构）。

    产出为**追加在正文块之后**的独立块（chunk_id `xtbl-*`），只进检索视图，
    不动 `ordered_chunks`（报告/claims 锚点不受影响）。
    """
    base = Path(mineru_dir)
    if not base.is_dir():
        return ()
    from paperpilot.tools.mineru_bridge import (EXT_CHUNK_PREFIX, _find_content_list,
                                                element_text)
    try:
        cl = _find_content_list(base)
    except Exception:  # noqa: BLE001
        return ()
    if not cl:
        return ()
    # 两遍扫：先（清 caption + 取表头指纹），再组装（缺 caption 的借/补）
    items: list[tuple[int, dict, str, list[str], list[str]]] = []
    for i, el in enumerate(cl):
        typ = el.get("type")
        if typ not in ("table", "equation"):
            continue
        caps: list[str] = []
        if typ == "table":
            caps = _clean_table_captions(el)
            # 只在确有噪声被清掉时才复制 el（省一次 dict 拷贝）
            if caps != [str(x).strip() for x in (el.get("table_caption") or []) if str(x).strip()]:
                el = {**el, "table_caption": caps}
        # ⚠️ 不要再拼 caption：`element_text` 对 table 已返回 `cap + body + footnote`
        # （旧代码又拼了一次 `table_caption` → caption 在块内出现 2 次，浪费 ~280 字、
        #  且 caption 词在 BM25 里权重翻倍）。实测去重后目标表进表池 top-1 从 50% → 65%，
        #  位次净改善 4 题（`qa/recall/EXT_REPR_20260911.md`）。
        t = element_text(el)
        if not t:
            continue
        items.append((i, el, t, caps, _table_header_cells(t) if typ == "table" else []))

    # 同篇内"表头相同"的块可互相借 caption：MinerU 会把**一张表切成多块**，
    # 只有其中一块带 caption（实例 `2002.10361`: xtbl-58/59 ← xtbl-60 的 "Table 5: ..."）。
    donor: dict[str, str] = {}
    for _i, _el, _t, caps, cells in items:
        if caps and cells:
            donor.setdefault("|".join(sorted(cells)), " ".join(caps))

    out: list[Chunk] = []
    # P2 双写：`PAPERPILOT_TABLE_EMBED_SUMMARY=1` 时，表块额外给向量侧一行语义摘要
    #（BM25/作答仍读 `text`）。**默认关** —— 它是待验证的实验项，不改变线上行为。
    want_emb = os.environ.get("PAPERPILOT_TABLE_EMBED_SUMMARY") == "1"
    for i, el, t, caps, cells in items:
        if el.get("type") == "table" and not caps:
            # 补身份（**只补缺，不加长**）：借不到就用列名拼一行身份串，
            # 让 BM25 至少能命中 "method / accuracy" 这类词，向量也有语义锚点。
            # 不要给**已有 caption 的好块**加料 —— 实测 refs 全量会让向量净 7 题变差
            # （`qa/recall/EXT_REPR_20260911.md`），情形不同、别学。
            borrowed = donor.get("|".join(sorted(cells))) if cells else None
            if borrowed:
                t = f"{borrowed}\n{t}"
            elif cells:
                t = "表格（列：" + "、".join(cells[:12]) + "）\n" + t
        emb = (_table_summary(t) or None) if (want_emb and el.get("type") == "table") else None
        out.append(Chunk(chunk_id=f"{EXT_CHUNK_PREFIX}{i}", title_path=["(External Tables)"],
                         page_span=(int(el.get("page_idx", 0)) + 1, int(el.get("page_idx", 0)) + 1),
                         text=t[:MAX_CHUNK_LEN], n_blocks=1, embed_text=emb))
    return tuple(out)


@lru_cache(maxsize=32)
def _current_source_cached(pdf: str, stamp: str) -> str:
    """该 pdf 当前解析源：`<注册源名>` | 'mineru' | 'pymupdf'。

    QA/报告链据此决定标题源、figures 源与产物缓存源打标，保证**整链同源**。
    `stamp` 只参与缓存键（文件版本）。

    ⚠️ 判定条件必须与 `_ordered_chunks_cached` **完全一致**（同一个
    `mineru_skeleton_enabled()`）：否则会出现"骨架用了 MinerU、却报 pymupdf"
    → `source` 打标错 → `_display_title` / `stage_figures` 走错分支。
    """
    name = sources.source_name(pdf)
    if name is not None:
        return name
    if mineru_skeleton_enabled() and _chunks_via_mineru(pdf) is not None:
        return "mineru"
    return "pymupdf"      # 默认路不可用 → 备用路（容灾）；原因见 `mineru_status`


def current_source(pdf: str) -> str:
    """见 `_current_source_cached`（按 (文件名, 文件版本) 缓存）。"""
    return _current_source_cached(pdf, _pdf_stamp(pdf))


current_source.cache_clear = _current_source_cached.cache_clear      # type: ignore[attr-defined]
current_source.cache_info = _current_source_cached.cache_info        # type: ignore[attr-defined]


def top_section(chunk: Chunk | None) -> str:
    """chunk 所属顶层节名：title_path[0] 取 '· ' 后半段。

    "L1 1 · 1 Introduction"  → "1 Introduction"
    "(PREAMBLE)"              → "(PREAMBLE)"
    取首个 '· ' 之后的**全部**（保留标题内后续 '·'），保证与 claim 端 normalize 一致。
    """
    if chunk is None:
        return ""
    return section_from_path(list(chunk.title_path))


def section_from_path(title_path: list[str]) -> str:
    """title_path → 顶层节名（与 top_section 同一 normalize，供 claim 端复用）。

    claim.title_path 与 chunk.title_path 同构（chunker 产出），
    ClaimHit.home_section 用它在节点层归一，expand_l2 才能按节匹配。
    """
    if not title_path:
        return ""
    head = title_path[0]
    if " · " in head:
        return head.split(" · ", 1)[1].strip()
    return head.strip()


def section_chunks(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """按顶层节名分组（保序）。L2 取圆心、判定节边界都用它。"""
    groups: dict[str, list[Chunk]] = {}
    for c in chunks:
        groups.setdefault(top_section(c), []).append(c)
    return groups


def section_of_chunk_id(chunks: list[Chunk], cid: str) -> str:
    """按 chunk_id 反查所属顶层节（找不到返回空串）。"""
    for c in chunks:
        if c.chunk_id == cid:
            return top_section(c)
    return ""
