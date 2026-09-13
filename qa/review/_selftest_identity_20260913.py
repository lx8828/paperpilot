"""论文身份（A+C 档）自测：内容指纹 / 新鲜度判定 / 迁移规则 / 上传落点。

覆盖审查项 2 的修复：
  A) 内容指纹 + 读取校验（`paper_identity`），历史产物按 mtime 收编；
     进程内缓存按 (文件名, 文件版本) 失效（`document_cache._pdf_stamp`）。
  C) 上传按内容判定落点（`web.plan_upload`）：同名同内容 → 幂等；同名不同内容 → 另存新名。

全程只在 `assets/papers` 与 `assets/artifacts/out_views` 里用**临时文件**（`__idtest__*`），
finally 里删干净；不联网、不调 LLM、不碰真实论文产物。

用法：uv run python qa/review/_selftest_identity_20260913.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paperpilot import paper_identity as pid  # noqa: E402
from paperpilot.agents import document_cache as dc  # noqa: E402

PAPERS = ROOT / "assets/papers"
VIEW = ROOT / "assets/artifacts/out_views"
TMP_STEM = "__idtest__paper"
TMP_PDF = PAPERS / f"{TMP_STEM}.pdf"

FAILS: list[str] = []


def ck(name: str, cond: bool) -> None:
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def touch_artifacts(stem: str, names: list[str]) -> list[Path]:
    out = []
    for n in names:
        p = VIEW / f"{stem}.{n}.json"
        p.write_text("{}", encoding="utf-8")
        out.append(p)
    return out


def cleanup() -> None:
    TMP_PDF.unlink(missing_ok=True)
    for p in PAPERS.glob(f"{TMP_STEM}__*.pdf"):
        p.unlink(missing_ok=True)
    for p in VIEW.glob(f"{TMP_STEM}.*"):
        p.unlink(missing_ok=True)
    for p in (ROOT / "assets/artifacts/out_claims").glob(f"{TMP_STEM}.*"):
        p.unlink(missing_ok=True)
    for p in (ROOT / "assets/artifacts/out_mineru").glob(TMP_STEM):
        if p.is_dir():
            for f in p.rglob("*"):
                f.unlink(missing_ok=True)
            p.rmdir()


def main() -> int:
    cleanup()
    try:
        # ── 1) 指纹基础 ────────────────────────────────────────────────
        ck("虚拟名（QASPER）不做指纹", pid.fingerprint("qasper_1234.5678.qpdf") is None)
        data = b"%PDF-1.4 test-A" + b"x" * 100
        TMP_PDF.write_bytes(data)
        fp = pid.fingerprint(f"{TMP_STEM}.pdf")
        ck("指纹 = sha256(文件字节)",
           (fp or {}).get("sha256") == hashlib.sha256(data).hexdigest())
        ck("指纹含 size/mtime_ns", bool(fp and fp.get("size") == len(data) and fp.get("mtime_ns")))

        # ── 2) check() 四种判定 ────────────────────────────────────────
        arts = touch_artifacts(TMP_STEM, ["claims", "summary", "report"])
        ck("有记录且 sha 一致 → ok",
           pid.check(f"{TMP_STEM}.pdf", fp, arts)[0] == "ok")
        wrong = {"sha256": "0" * 64}
        ck("有记录但 sha 不一致 → stale",
           pid.check(f"{TMP_STEM}.pdf", wrong, arts)[0] == "stale")
        ck("无记录 + 产物比 PDF 新 → adopt（历史产物收编）",
           pid.check(f"{TMP_STEM}.pdf", None, arts)[0] == "adopt")

        time.sleep(1.1)
        new = b"%PDF-1.4 test-B" + b"y" * 100
        TMP_PDF.write_bytes(new)                       # 同名换内容 → PDF 比产物新
        ck("无记录 + PDF 比产物新 → stale（可疑）",
           pid.check(f"{TMP_STEM}.pdf", None, arts)[0] == "stale")
        os.utime(TMP_PDF, (time.time() - 10, time.time() - 10))   # 让 PDF 变旧 → adopt 场景
        # 说明：`check()` 只报**事实**（adopt），"严格模式"是**调用方策略**——所以这里
        # 断言的是 ① check 仍报 adopt、② pid.strict() 生效、③ 调用方（pipeline）真的重建。
        os.environ["PAPERPILOT_PDF_ID_STRICT"] = "1"
        v_strict = pid.check(f"{TMP_STEM}.pdf", None, arts)[0]
        from paperpilot.pipeline import _pdf_identity_stale as _ids
        ck("严格模式：check 报 adopt，但 pid.strict()=True 且 pipeline 判 stale",
           v_strict == "adopt" and pid.strict() is True and _ids(f"{TMP_STEM}.pdf")[0] is True)
        os.environ.pop("PAPERPILOT_PDF_ID_STRICT", None)

        # ── 3) 进程内缓存：按文件版本失效 + 兼容 cache_clear ────────────
        ck("_pdf_stamp 虚拟名为空", dc._pdf_stamp("qasper_1.2.qpdf") == "")
        s1 = dc._pdf_stamp(f"{TMP_STEM}.pdf")
        TMP_PDF.write_bytes(b"%PDF-1.4 test-C" + b"z" * 200)
        ck("文件变了 → stamp 变（缓存自然失效）", dc._pdf_stamp(f"{TMP_STEM}.pdf") != s1)
        ck("ordered_chunks.cache_clear 仍可用（老调用方兼容）",
           callable(getattr(dc.ordered_chunks, "cache_clear", None)))
        ck("retrieval_chunks / current_source 同样兼容",
           callable(getattr(dc.retrieval_chunks, "cache_clear", None))
           and callable(getattr(dc.current_source, "cache_clear", None)))

        # ── 4) pipeline 身份门（mtime 迁移规则）─────────────────────────
        from paperpilot.pipeline import _pdf_identity_stale
        now = time.time()
        os.utime(TMP_PDF, (now - 30, now - 30))        # PDF 旧
        os.utime(arts[0], (now, now))                  # 产物新 → adopt
        stale, why = _pdf_identity_stale(f"{TMP_STEM}.pdf")
        ck("pipeline：产物比 PDF 新 → 不重建（收编）", stale is False)
        time.sleep(1.1)
        TMP_PDF.write_bytes(b"%PDF-1.4 test-D" + b"w" * 50)
        stale2, why2 = _pdf_identity_stale(f"{TMP_STEM}.pdf")
        ck("pipeline：PDF 比产物新 → 判 stale（重建）", stale2 is True)
        print(f"       （原因：{why2[:70]}）")

        # ── 5) web 上传落点（C 档）──────────────────────────────────────
        spec = importlib.util.spec_from_file_location("webapp", "web/app.py")
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)  # type: ignore[union-attr]
        good = b"%PDF-1.4 real" + b"a" * 80
        TMP_PDF.write_bytes(good)
        p_new = app_mod.plan_upload("__idtest__brand_new.pdf", good)
        ck("新文件名 → 落原名（不 reuse）",
           p_new["reuse"] is False and p_new["name"] == "__idtest__brand_new.pdf")
        p_same = app_mod.plan_upload(f"{TMP_STEM}.pdf", good)
        ck("同名同内容 → reuse=True（幂等）",
           p_same["reuse"] is True and p_same["name"] == f"{TMP_STEM}.pdf")
        other = b"%PDF-1.4 other paper" + b"b" * 90
        p_diff = app_mod.plan_upload(f"{TMP_STEM}.pdf", other)
        ck("同名不同内容 → 另存为 <stem>__<sha8>.pdf（不覆盖）",
           p_diff["reuse"] is False
           and p_diff["name"] == f"{TMP_STEM}__{hashlib.sha256(other).hexdigest()[:8]}.pdf"
           and bool(p_diff["note"]))
        ck("原文件未被改动（内容仍是 good）", TMP_PDF.read_bytes() == good)
    finally:
        cleanup()

    print()
    print("SELFTEST-IDENTITY:", "ALL PASS" if not FAILS else f"FAILED ({len(FAILS)})")
    for f in FAILS:
        print("   FAIL:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
