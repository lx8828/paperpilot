"""QASPER 评测数据源（**测试侧**，不属于生产包 `paperpilot`）。

## 为什么住在这儿（2026-09-22 重构）

它原本是 `src/paperpilot/qasper_source.py` —— **评测集适配器却住在生产包里**，
于是"虚拟文件名 `qasper_<pid>.qpdf`"穿透了 7 个生产文件（`pipeline` / `document_cache`
/ `worker` / `ingest` / `paper_identity` / `splitter` / 本模块），每处都得写
`if is_qasper(pdf)`；新增第二个评测源还要再改这 7 处。而**产品目标是 arXiv PDF 解析**，
生产不该知道评测集存在。

现在生产只依赖 `paperpilot.sources` 的接口 + 注册表；**本模块在 import 时把自己注册进去**：

    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "qa" / "sources" / "qasper.py").exists())))  # 测试侧数据源（qa/）
    from qa.sources.qasper import load_papers     # 顺带完成注册
    ...
    ordered_chunks("qasper_2206.12345.qpdf")      # 自动走本源的 chunks()

⚠️ 依赖注入的前提：调用方得让 `qa/` 可 import（脚本里
`sys.path.insert(0, str(ROOT))`，ROOT = 仓库根）。

## 提供

    QasperSource        `PaperSource` 实现（chunks / title / extra_chunks）
    load_papers()       数据集全量（paper_id → paper dict）
    build_chunks()      full_text → Chunk[]（chunk_id 规则与生产 chunk 空间同构）
    gold_answer_full()  该题**全部**人工 evidence 段（检索评测的 gold）
    gold_answer()       兼容旧签名（首段口径）

设计约束（与原 pipeline 一致）：
    - `title_path` 顶层 = section 根名，与分析器 `_skip_chunk` 兼容
    - 超大 section（>MAX_CHUNK_LEN）按段落二级切分（part 编号），与 chunker 同策略
    - `chunk_id` 形如 `c0001`（全局自增），同一 paper 恒定 → 向量索引缓存可复用
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paperpilot import sources
from paperpilot.models.schema import Chunk, ChunkSpan

# qa/sources/qasper.py → qa/sources → qa → 仓库根
ROOT = Path(__file__).resolve().parents[2]
QASPER_JSON = ROOT / "qasper_data" / "json" / "qasper-train-v0.3.json"
MINERU_OUT = ROOT / "assets/artifacts/out_mineru"

MAX_CHUNK_LEN = 4000          # 与 chunker.DEFAULT_MAX_LEN / pipeline.MAX_LEN 一致
REF_MARKERS = ("References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY")

PREFIX = "qasper_"
SUFFIX = ".qpdf"

# S2ORC 引用占位符（"BIBREF0" / "FIGREF1"）→ 归一，避免进 LLM 上下文
_REF_RE = re.compile(r"\b[A-Z]+REF\d+\b")
_FIG_RE = re.compile(r"\bFIGURE\d+\b|\bTABLE\d+\b", re.I)


def _clean_para(p: str) -> str:
    """段落清洗：折叠空白、剥离 S2ORC 占位引用（保留原文主体）。"""
    t = " ".join(p.split())
    t = _REF_RE.sub("", t)
    t = _FIG_RE.sub("", t)
    return " ".join(t.split())


def _split_title(name: str) -> list[str]:
    """'Proposed Method ::: Polarity Function' → ['Proposed Method', 'Polarity Function']。"""
    return [x.strip() for x in name.split(":::") if x.strip()]


def pid_of(pdf_name: str) -> str:
    """`qasper_<pid>.qpdf` → `<pid>`。"""
    return pdf_name[len(PREFIX):-len(SUFFIX)]


def load_papers() -> dict[str, dict[str, Any]]:
    """加载 QASPER train 全量（paper_id → paper dict）。"""
    if not QASPER_JSON.exists():
        raise FileNotFoundError(f"缺 QASPER 数据: {QASPER_JSON}")
    return json.loads(QASPER_JSON.read_text(encoding="utf-8"))


def build_chunks(paper: dict[str, Any]) -> list[Chunk]:
    """从 QASPER full_text 确定性构造 Chunk[]（与 claims/QA 检索共用）。

    - 每 section 至少一个 chunk（含标题行文本）
    - 无 section_name 的前置段归 "(PREAMBLE)"（与分析器过滤规则一致）
    - 超长 section 按段落二级切分，共享 title_path，part="1/2"
    - chunk_id 全局自增 c001..；同一 paper 恒定
    """
    chunks: list[Chunk] = []
    full = paper.get("full_text") or []
    seq = 1
    for sec in full:
        parts = _split_title(sec.get("section_name") or "")
        title_path = parts or ["(PREAMBLE)"]
        paras = [_clean_para(p) for p in (sec.get("paragraphs") or []) if _clean_para(p)]
        if not paras:
            continue
        # 标题 + 段落拼成 raw 文本，按 MAX_CHUNK_LEN 段落累加切分
        raw = "\n".join(paras)
        if len(raw) <= MAX_CHUNK_LEN:
            chunks.append(Chunk(
                chunk_id=f"c{seq:04d}",
                title_path=list(title_path),
                page_span=(0, 0),
                text=raw,
                n_blocks=len(paras),
                spans=[ChunkSpan(block_id=f"c{seq:04d}", page=0, text=raw[:200])],
            ))
            seq += 1
            continue
        # 超长：段落级二级切分（子块共享 title_path）
        cur: list[str] = []
        cur_len = 0
        n_sub = 0
        for para in paras:
            if cur and cur_len + len(para) + 1 > MAX_CHUNK_LEN:
                n_sub += 1
                chunks.append(Chunk(
                    chunk_id=f"c{seq:04d}",
                    title_path=list(title_path),
                    page_span=(0, 0),
                    text="\n".join(cur),
                    n_blocks=len(cur),
                    part=f"{n_sub}/?",
                    spans=[ChunkSpan(block_id=f"c{seq:04d}", page=0,
                                     text="\n".join(cur)[:200])],
                ))
                seq += 1
                cur = []
                cur_len = 0
            cur.append(para)
            cur_len += len(para) + 1
        if cur:
            n_sub += 1
            chunks.append(Chunk(
                chunk_id=f"c{seq:04d}",
                title_path=list(title_path),
                page_span=(0, 0),
                text="\n".join(cur),
                n_blocks=len(cur),
                part=f"{n_sub}/?",
                spans=[ChunkSpan(block_id=f"c{seq:04d}", page=0,
                                 text="\n".join(cur)[:200])],
            ))
            seq += 1
    # 回填 part 总数（"1/?") → 确定性"1/N"
    per_path: dict[tuple[str, ...], list[int]] = {}
    for i, c in enumerate(chunks):
        if c.part:
            per_path.setdefault(tuple(c.title_path), []).append(i)
    for path, idxs in per_path.items():
        for j, i in enumerate(idxs, 1):
            chunks[i].part = f"{j}/{len(idxs)}"
    return chunks


def gold_answer_full(q: dict[str, Any]) -> tuple[str | None, list[str]]:
    """取该题最可信的人工答案 → (答案文本, **全部**证据段落列表)。

    QASPER 单题常有多段人工 evidence（实测 recall_set 约 38% 的题 ≥2 段）。
    检索评测须把全部 evidence 段都算作 gold：只取首段会把"命中第 2/3 段同样正确的
    证据"误判为 miss，系统性压低 Recall/MRR/NDCG。

    优先级：free_form_answer > yes_no > extractive_spans 拼装；多 answerer 取首个非
    unanswerable。unanswerable 返回 (None, [])。
    """
    for a in q.get("answers") or []:
        inner = a.get("answer") or {}
        if inner.get("unanswerable"):
            continue
        evs = [str(e) for e in (inner.get("evidence") or []) if e]
        if inner.get("free_form_answer"):
            return str(inner["free_form_answer"]).strip(), evs
        if inner.get("yes_no") is not None:
            return ("yes" if inner["yes_no"] else "no"), evs
        spans = inner.get("extractive_spans") or []
        if spans:
            return " ".join(str(s) for s in spans).strip(), evs
    return None, []


def gold_answer(q: dict[str, Any]) -> tuple[str | None, str | None]:
    """兼容旧签名：返回 (答案文本, 首条证据)。

    LLM 裁判 prompt 等仍用此首段口径；检索评测请改用 gold_answer_full 取全部证据。
    unanswerable 返回 (None, None)。
    """
    ans, evs = gold_answer_full(q)
    return ans, (evs[0] if evs else None)


# ── PaperSource 实现 ─────────────────────────────────────────────────────────


class QasperSource:
    """无磁盘文件、无版面页码的**文本源**（评测用）。

    能力声明：
      · `name = "qasper"` → 写进产物 `source` 字段（整链同源校验）
      · `has_file = False` → 不做内容指纹、不跑 MinerU、摄取闸门放行
        （这三件事的前提都是"磁盘上有份 PDF"）
    """

    name = "qasper"
    has_file = False

    def __init__(self, pdf_name: str) -> None:
        self.pdf_name = pdf_name
        self.pid = pid_of(pdf_name)

    def _paper(self) -> dict[str, Any]:
        paper = load_papers().get(self.pid)
        if not paper:
            raise FileNotFoundError(f"QASPER 缺论文: {self.pid}")
        return paper

    def title(self) -> str:
        return str(self._paper().get("title", "") or "").strip()

    def chunks(self) -> list[Chunk]:
        """报告视图：full_text → chunks，去 PREAMBLE（与生产的过滤规则一致）。"""
        raw = build_chunks(self._paper())
        return [c for c in raw if c.title_path != ["(PREAMBLE)"]]

    def extra_chunks(self) -> list[Chunk]:
        """**外部表格通道**（实验，默认关）：从 arXiv 原始 PDF 的 MinerU 产物补表/公式。

        为什么需要：QASPER 的 `figures_and_tables` 只给 PNG + caption 的字符 span，
        **表格数值没有任何文本形式** → 凡 gold 出自表格的题，在纯文本通道下
        **结构上不可达**（实测 500 题里 36 道，占 8.0pt）。

        开启条件：`PAPERPILOT_QASPER_TABLES=1` **且** `out_mineru/<pid>v1/` 有 content_list。
        只进检索视图，不动 `ordered_chunks`（报告/claims 锚点不受影响）。

        ⚠️ 这段策略**刻意留在测试侧** —— 生产包不该含评测集的表格实验开关。
        """
        import os

        if os.environ.get("PAPERPILOT_QASPER_TABLES") != "1":
            return []
        from paperpilot.agents.document_cache import external_table_chunks
        return list(external_table_chunks(str(MINERU_OUT / f"{self.pid}v1")))


def _match(name: str) -> bool:
    return name.startswith(PREFIX) and name.endswith(SUFFIX)


# **import 即注册** —— 测试脚本 `from qa.sources.qasper import ...` 时自动生效
sources.register(_match, QasperSource)
