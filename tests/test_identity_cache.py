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


def test_virtual_name_has_no_fingerprint():
    assert pid.fingerprint("qasper_1234.5678.qpdf") is None
    assert pid.is_virtual("qasper_1234.5678.qpdf") is True


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
    assert dc._pdf_stamp("qasper_1.2.qpdf") == ""
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


# ───────────────────────── 检索视图：表格只进检索、不进报告 ─────────────────────────


def test_current_source_default_is_pymupdf(tmp_assets, tiny_pdf, fake_mineru):
    fake_mineru(tiny_pdf)
    assert dc.current_source(tiny_pdf) == "pymupdf"


def test_mineru_table_injected_into_retrieval_view_only(tmp_assets, tiny_pdf, fake_mineru):
    """MinerU 的表格按页注入**检索视图**；报告视图（ordered_chunks）不受影响。"""
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
