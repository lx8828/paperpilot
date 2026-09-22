"""论文身份（内容指纹）与缓存：**命中 / 失效**必须成对验证。

这两件事是同一个病根（审查项 2）：
- 产物按 `Path(pdf).stem` 寻址 → 同名不同内容会**静默复用别人的报告**；
- 进程内 chunk 缓存 / 磁盘向量缓存同理 → "新文件 + 旧解析结果"。

所以这里既测"该命中时命中"（不然每次都要重算），也测"该失效时失效"
（不然用户拿到别篇论文的结果且**毫无报错**）。
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest

from paperpilot import paper_identity as pid
from paperpilot.agents import document_cache as dc
from paperpilot.agents.embedder import ChunkIndex
from paperpilot.models.schema import Chunk


# ───────────────────────── 指纹 ─────────────────────────


def test_fileless_source_has_no_fingerprint(tmp_assets):
    """**无磁盘文件的数据源**（评测文本集）→ 无内容指纹。

    2026-09-22：以前这里断言 `is_virtual("qasper_*.qpdf")` —— 生产代码不该认识
    评测集的名字。现在改为**注册一个无文件的数据源**：语义等价，且不绑死具体数据集。
    指纹的前提是"磁盘上有份文件"，无文件的源天然不适用。
    """
    from paperpilot import sources

    class _NoFileSource:
        name = "nofile"
        has_file = False

        def title(self) -> str:
            return "t"

        def chunks(self) -> list:
            return []

        def extra_chunks(self) -> list:
            return []

    sources.unregister_all()
    try:
        sources.register(lambda p: p.startswith("nofile:"), lambda p: _NoFileSource())
        assert sources.has_file("nofile:x") is False
        assert pid.fingerprint("nofile:x") is None      # 不走 stat，直接判无文件
    finally:
        sources.unregister_all()                        # 别污染其他用例
    # 未注册 + 文件不存在 → 同样 None（走 stat 失败分支）
    assert pid.fingerprint("missing_paper.pdf") is None
    assert sources.has_file("missing_paper.pdf") is True


def test_fingerprint_is_content_hash(tmp_assets):
    data = b"%PDF-1.4 hello" + b"x" * 100
    p = tmp_assets.papers / "a.pdf"
    p.write_bytes(data)
    fp = pid.fingerprint("a.pdf")
    assert fp and fp["sha256"] == hashlib.sha256(data).hexdigest()
    assert fp["size"] == len(data) and fp["mtime_ns"]


def test_fingerprint_missing_file(tmp_assets):
    assert pid.fingerprint("nope.pdf") is None


# ───────────────────────── check() 四种判定 ─────────────────────────


def _touch(tmp_assets, stem: str) -> list[Path]:
    arts = []
    for n in ("claims", "summary", "report"):
        p = tmp_assets.views / f"{stem}.{n}.json"
        p.write_text("{}", encoding="utf-8")
        arts.append(p)
    return arts


def test_check_ok_when_sha_matches(tmp_assets):
    (tmp_assets.papers / "a.pdf").write_bytes(b"%PDF-1.4 A")
    arts = _touch(tmp_assets, "a")
    fp = pid.fingerprint("a.pdf")
    assert pid.check("a.pdf", fp, arts)[0] == "ok"


def test_check_stale_when_sha_differs(tmp_assets):
    (tmp_assets.papers / "a.pdf").write_bytes(b"%PDF-1.4 A")
    arts = _touch(tmp_assets, "a")
    assert pid.check("a.pdf", {"sha256": "0" * 64}, arts)[0] == "stale"


def test_check_adopt_for_historical_artifacts(tmp_assets):
    """无指纹记录 + 产物比 PDF 新 → adopt（存量库收编，不重建）。"""
    p = tmp_assets.papers / "a.pdf"
    p.write_bytes(b"%PDF-1.4 A")
    os.utime(p, (time.time() - 30, time.time() - 30))
    arts = _touch(tmp_assets, "a")
    now = time.time()
    for a in arts:
        os.utime(a, (now, now))
    assert pid.check("a.pdf", None, arts)[0] == "adopt"


def test_check_stale_when_pdf_newer_than_artifacts(tmp_assets):
    p = tmp_assets.papers / "a.pdf"
    p.write_bytes(b"%PDF-1.4 A")
    arts = _touch(tmp_assets, "a")
    old = time.time() - 30
    for a in arts:
        os.utime(a, (old, old))
    assert pid.check("a.pdf", None, arts)[0] == "stale"


def test_check_unknown_without_artifacts(tmp_assets):
    (tmp_assets.papers / "a.pdf").write_bytes(b"%PDF-1.4 A")
    assert pid.check("a.pdf", None, [])[0] == "unknown"


def test_strict_env_flag(monkeypatch):
    assert pid.strict() is False
    monkeypatch.setenv("PAPERPILOT_PDF_ID_STRICT", "1")
    assert pid.strict() is True


# ───────────────────────── 进程内缓存：按文件版本失效 ─────────────────────────


def test_pdf_stamp_changes_with_file(tmp_assets):
    assert dc._pdf_stamp("does_not_exist.pdf") == ""   # 文件不存在 → 版本分量空
    p = tmp_assets.papers / "a.pdf"
    p.write_bytes(b"%PDF-1.4 A")
    s1 = dc._pdf_stamp("a.pdf")
    p.write_bytes(b"%PDF-1.4 A" + b"y" * 10)
    assert dc._pdf_stamp("a.pdf") != s1


def test_ordered_chunks_cache_hit_and_invalidate(tmp_assets, tiny_pdf):
    """**命中**：同一版本重复解析 → 走缓存（不重复 parse）。
    **失效**：文件内容变了（stamp 变）→ 必须重新解析。
    """
    first = dc.ordered_chunks(tiny_pdf)
    assert first, "小 PDF 应该能切出 chunk"
    hits0 = dc.ordered_chunks.cache_info().hits
    dc.ordered_chunks(tiny_pdf)
    assert dc.ordered_chunks.cache_info().hits > hits0, "同版本应命中缓存"

    before = "".join(c.text for c in first)
    # 替换文件内容（模拟"同名换了 PDF"）→ 文件版本变 → 缓存必须失效
    import fitz

    path = tmp_assets.papers / tiny_pdf
    edited = tmp_assets.papers / "edited.pdf"       # pymupdf 不允许原地覆盖保存
    doc = fitz.open(str(path))
    doc[0].insert_text((72, 300), "MARKER_AFTER_EDIT_12345", fontsize=11)
    doc.save(str(edited))
    doc.close()
    edited.replace(path)                            # 目标名不变、内容变了
    after = "".join(c.text for c in dc.ordered_chunks(tiny_pdf))
    assert after != before and "MARKER_AFTER_EDIT_12345" in after


def test_cache_clear_still_available():
    """老调用方（评测脚本）仍能 `.cache_clear()`。"""
    for fn in (dc.ordered_chunks, dc.retrieval_chunks, dc.current_source):
        assert callable(getattr(fn, "cache_clear", None))


# ───────────────────────── 解析源：MinerU 主 · pymupdf 容灾 ─────────────────────────


def test_current_source_default_is_mineru(tmp_assets, tiny_pdf, fake_mineru):
    """**默认（2026-09-22 起）= MinerU 作骨架**：有产物就用。

    与"跑不跑 MinerU"（`PAPERPILOT_MINERU`）是两件事：后者默认也开，
    前者（`PAPERPILOT_USE_MINERU`）默认由 OFF 翻转成 ON。
    """
    fake_mineru(tiny_pdf)
    assert dc.mineru_skeleton_enabled() is True
    assert dc.current_source(tiny_pdf) == "mineru"


def test_not_ingested_is_pending_not_degraded(tmp_assets, tiny_pdf):
    """**"没跑过 MinerU" ≠ "降级"**（口径必须分清）：

        pending  = 尚未摄取 → **待办**，跑一次 ingest 就好
        degraded = MinerU **试过但失败** → 才用 pymupdf 容灾、才该告警

    两者都导致"当前用 pymupdf"，但性质完全不同；混起来会把待办报成故障。
    """
    assert dc.current_source(tiny_pdf) == "pymupdf"          # 当前确实用 pymupdf
    state, why = dc.mineru_status(tiny_pdf)
    assert state == "pending", f"应为 pending（未摄取），实际 {state}"
    assert "尚未摄取" in why


def test_broken_mineru_products_are_degraded(tmp_assets, tiny_pdf):
    """MinerU **产物损坏** → 这才是 `degraded`（容灾），且原因可查。"""
    d = tmp_assets.mineru / "tiny"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tiny_content_list.json").write_text("{ 不是合法 JSON", encoding="utf-8")
    state, why = dc.mineru_status(tiny_pdf)
    assert state == "degraded", f"应为 degraded，实际 {state}"
    assert why


def test_table_is_in_ordered_chunks_under_default(tmp_assets, tiny_pdf, fake_mineru):
    """默认档：MinerU 作骨架 → **表值直接进 `ordered_chunks`**（不再靠"注入检索视图"）。

    这是新旧默认最大的差别：旧默认报告链读 pymupdf（表格只进检索视图 → **claims/报告
    看不到表值** → L0 对表格类问题系统性无能）；新默认表格就在骨架里，
    于是 claims / report / L0 都能看到表值。
    """
    fake_mineru(tiny_pdf)
    base = dc.ordered_chunks(tiny_pdf)
    assert "Ours" in "".join(c.text for c in base)          # "Ours" 只在假表体里
    # 骨架已是 MinerU → 检索视图无需二次注入，两者同源同 id
    assert [c.chunk_id for c in base] == [c.chunk_id for c in dc.retrieval_chunks(tiny_pdf)]


def test_mineru_table_injected_into_retrieval_view_only(tmp_assets, tiny_pdf, fake_mineru,
                                                        monkeypatch):
    """**容灾档**（`PAPERPILOT_USE_MINERU=0`）：骨架退回 pymupdf，表格只**注入检索视图**。

    这条路径仍在（pymupdf 是容灾与 A/B 对照），语义与 2026-09-10 的设计一致。
    """
    monkeypatch.setenv("PAPERPILOT_USE_MINERU", "0")
    fake_mineru(tiny_pdf)
    base = dc.ordered_chunks(tiny_pdf)
    view = dc.retrieval_chunks(tiny_pdf)
    assert len(base) == len(view)                        # 同一批 chunk_id
    assert [c.chunk_id for c in base] == [c.chunk_id for c in view]
    assert any(len(a.text) != len(b.text) for a, b in zip(base, view))
    # "Ours" 是**只存在于假 MinerU 表体**里的字（PDF 正文里没有）→ 可精确判定"注入生效"
    assert "Ours" in "".join(c.text for c in view)        # 表值可被检索到
    assert "Ours" not in "".join(c.text for c in base)    # 报告视图不含表


# ───────────────────────── 向量缓存：指纹命中 / 失效 ─────────────────────────


def _chunk(cid: str = "c1", text: str = "abc", embed_text: str | None = None) -> Chunk:
    return Chunk(chunk_id=cid, title_path=["1 Introduction"], page_span=(1, 1),
                 text=text, n_blocks=1, embed_text=embed_text)


def test_fingerprint_covers_text_and_embed_text(monkeypatch):
    """缓存键必须包含**实际参与编码的文本**：`P2 双写`下是 `embed_text`。

    反例（曾真实存在的坑）：只 hash `text` → 改了 `embed_text` 却复用旧向量，
    错误完全不可见。
    """
    idx = ChunkIndex("a.pdf")
    monkeypatch.setattr(idx, "_doc_chunks", lambda: [_chunk(text="abc")])
    fp_text = idx._fingerprint()
    monkeypatch.setattr(idx, "_doc_chunks", lambda: [_chunk(text="abd")])
    assert idx._fingerprint()["fp"] != fp_text["fp"]
    monkeypatch.setattr(idx, "_doc_chunks",
                        lambda: [_chunk(text="abc", embed_text="summary")])
    assert idx._fingerprint()["fp"] != fp_text["fp"]


def test_fingerprint_covers_chunk_ids(monkeypatch):
    idx = ChunkIndex("a.pdf")
    monkeypatch.setattr(idx, "_doc_chunks", lambda: [_chunk("c1")])
    a = idx._fingerprint()
    monkeypatch.setattr(idx, "_doc_chunks", lambda: [_chunk("c2")])
    assert idx._fingerprint() != a


def test_cvec_disk_cache_hit_then_invalidate(tmp_assets, fake_embed, monkeypatch):
    """磁盘向量缓存：第一次 encode → 落盘；同内容再问 → **不再 encode**；文本变了 → 重建。"""
    import numpy as np

    from paperpilot.agents import embedder as E

    calls = {"n": 0}

    def _counting(texts):
        calls["n"] += 1
        return np.stack([fake_embed(t) for t in texts]).astype("float32")

    monkeypatch.setattr(E, "encode_texts", _counting)
    chunks = [_chunk(text="alpha beta"), _chunk("c2", "gamma delta")]

    idx1 = ChunkIndex("a.pdf")
    monkeypatch.setattr(idx1, "_doc_chunks", lambda: chunks)
    idx1.vectors()
    assert calls["n"] == 1
    assert idx1._vec_file.exists() and idx1._cid_file.exists()

    idx2 = ChunkIndex("a.pdf")                   # 新实例（模拟新进程）→ 应命中磁盘缓存
    monkeypatch.setattr(idx2, "_doc_chunks", lambda: chunks)
    idx2.vectors()
    assert calls["n"] == 1, "同内容不应重复 encode"

    idx3 = ChunkIndex("a.pdf")                   # 文本变了 → 指纹变 → 必须重建
    monkeypatch.setattr(idx3, "_doc_chunks", lambda: [_chunk(text="changed text")])
    idx3.vectors()
    assert calls["n"] == 2, "内容变了必须重建向量"
