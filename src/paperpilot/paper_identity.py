"""论文**身份** = 内容指纹（sha256），不是文件名。

## 为什么要这个模块（2026-09-13 审查项 2）

全链路产物都以 `Path(pdf).stem` 为键：`claims / summary / skeleton / figures / overview / guide /
report`、`gvec/cvec`（+`gidx/cidx`）、`out_mineru/<stem>/`、`ingest.json` —— 而**没有任何一层记录
PDF 的内容指纹**。后果（已复现，见 `qa/review/_dupname_repro_20260913.py`）：

- **流水线**：把 `A.pdf` 换成 B 的字节（文件名不变）→ **解析层读 B 的文本**、**产物层给 A 的
  claims/报告** → 状态混合，且全程无告警；
- **用户可见**：web `/api/report` 上传"文件名=A、内容=B" → 直接返回 A 的旧报告，**且不写盘**；
- **评测测不到**：QASPER 走虚拟名 `qasper_<pid>.qpdf`（身份由数据集给定，不存在同名不同内容），
  所以问题只存在于**真实上传路径**。

## 本模块提供

`fingerprint(pdf_name)`：`{sha256, size, mtime}`（带进程内缓存，按 mtime/size 失效）。
`check(pdf_name, recorded, artifact_paths)`：判定 → `("ok" | "stale" | "adopt", 原因)`。

## 历史产物迁移规则（没有指纹记录时）

- **产物（最新一件）比 PDF 新** → 认定"由这份 PDF 产出" → **adopt**（收编，不重建）；调用方可顺手
  把指纹写回 meta，之后就走严格比对；
- **PDF 比产物新** → 可疑（很可能文件被换过）→ **stale**，由调用方决定重建还是（非严格模式）告警后收编。

⚠️ 代价：仅仅 `touch` 一下 PDF（内容没变）也会被判 stale → 重建。这是**安全的错误方向**，
且一旦写回指纹就不再依赖 mtime。
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "assets" / "papers"


def is_virtual(pdf_name: str) -> bool:
    """虚拟论文名（QASPER：`qasper_<pid>.qpdf`，无 PDF 文件）→ 不做内容指纹。"""
    return pdf_name.startswith("qasper_") and pdf_name.endswith(".qpdf")


def pdf_path(pdf_name: str) -> Path:
    return PAPERS_DIR / pdf_name


@lru_cache(maxsize=128)
def _fingerprint_cached(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return {"sha256": h.hexdigest(), "size": size, "mtime_ns": mtime_ns}


def fingerprint(pdf_name: str, *, path: Path | None = None) -> dict[str, Any] | None:
    """PDF 的内容指纹；虚拟名 / 文件不存在 → None。

    缓存键含 `mtime_ns + size`：文件被替换或修改后自动重算（同名同尺寸同 mtime 的极端情况除外）。
    """
    if is_virtual(pdf_name):
        return None
    p = path or pdf_path(pdf_name)
    try:
        st = p.stat()
    except OSError:
        return None
    return _fingerprint_cached(str(p), st.st_mtime_ns, st.st_size)


def short_sha(fp: dict[str, Any] | None, n: int = 8) -> str:
    return (fp or {}).get("sha256", "")[:n]


def _newest_mtime(paths: list[Path]) -> float:
    newest = 0.0
    for p in paths:
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    return newest


def check(pdf_name: str, recorded: dict[str, Any] | None,
          artifact_paths: list[Path]) -> tuple[str, str]:
    """产物是否属于当前这份 PDF。

    Returns: (verdict, reason)
        verdict ∈ {"ok", "stale", "adopt", "unknown"}
        · ok      —— 有记录且 sha256 一致
        · stale   —— 有记录但 sha256 不一致，或（无记录时）PDF 比产物新
        · adopt   —— 无记录但产物比 PDF 新（历史产物收编；调用方宜把指纹写回 meta）
        · unknown —— 虚拟名 / 文件不存在 / 无产物可判
    """
    cur = fingerprint(pdf_name)
    if cur is None:
        return "unknown", "无 PDF 文件（虚拟名或不存在）"
    if recorded and recorded.get("sha256"):
        if recorded["sha256"] == cur["sha256"]:
            return "ok", "指纹一致"
        return "stale", (f"内容指纹不一致（记录 {str(recorded['sha256'])[:8]}… "
                         f"≠ 当前 {cur['sha256'][:8]}…）—— PDF 已被替换")
    have = [p for p in artifact_paths if p.exists()]
    if not have:
        return "unknown", "无产物可判"
    newest = _newest_mtime(have)
    if newest >= cur["mtime_ns"] / 1e9 - 1e-6:
        return "adopt", "历史产物（无指纹记录）且比 PDF 新 → 收编"
    return "stale", ("无指纹记录，且 PDF 比全部产物都新（{:.0f}s）→ 可疑，"
                     "很可能文件被换过".format(cur["mtime_ns"] / 1e9 - newest))


def strict() -> bool:
    """`PAPERPILOT_PDF_ID_STRICT=1`：把历史产物（adopt 判定）也当 stale 强制重建。

    默认关：升级后第一次运行**收编**历史产物并写回指纹，之后即走严格比对；
    急着在存量库里立刻排雷时打开它（代价：全库重建）。
    """
    return os.environ.get("PAPERPILOT_PDF_ID_STRICT") == "1"
