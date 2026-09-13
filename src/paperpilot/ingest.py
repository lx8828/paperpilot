"""摄取流水线：PDF → 论文库 → MinerU（供检索）→ pipeline（供报告）→ 产物库。

设计（2026-09-10 定，方案 B3）：
    「报告无论如何需要 pymupdf（页码/版面/claims 锚点），但 MinerU 对检索更好」
    → 二者**不是二选一**：
      - chunk 空间只有一套（pymupdf 的 chunk_id）→ cites / 前端高亮 / claims 锚点不动；
      - MinerU 的表格/公式按页注入**检索视图**（document_cache.retrieval_chunks）
        → 表值能被召回、能被作答读到；claims 抽取仍读 pymupdf 原文（表格不进 claims）。
    「用户直接上传 PDF，没有预生成产物」→ MinerU 在**摄取期**现跑，并把
      状态/版本/时间戳落盘（ingest.json），问答期只读产物。

失败策略（用户定）：
    MinerU 失败或无 GPU → **报告照常生成**（pymupdf）；**图表**回退 pymupdf 抽取（降级）；
    **问答直接失败**，并由 `qa_blocked_reason()` 返回明确原因给用户（不静默降级）。

用法：
    from paperpilot.ingest import ingest
    ingest("2609.01456v1.pdf")          # 幂等：产物齐 + 版本一致 → 跳过
    ingest("x.pdf", force=True)         # 重跑（含 MinerU）
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from paperpilot import paper_identity

ROOT = Path(__file__).resolve().parents[2]           # src/paperpilot/ingest.py → 根
ASSETS = ROOT / "assets"
PAPERS_DIR = ASSETS / "papers"                        # 论文库（上传的原始 PDF）
ARTIFACT_ROOT = ASSETS / "artifacts"                  # 产物库
MINERU_OUT = ARTIFACT_ROOT / "out_mineru"

MINERU_VENV = Path(os.environ.get("PAPERPILOT_MINERU_VENV", ROOT / ".venv-mineru"))
MINERU_TIMEOUT = int(os.environ.get("PAPERPILOT_MINERU_TIMEOUT", "900"))
MINERU_BACKEND = os.environ.get("PAPERPILOT_MINERU_BACKEND", "pipeline")
META_NAME = "ingest.json"


# ── MinerU 可执行与版本 ────────────────────────────────────────────────────────


def mineru_cmd() -> str | None:
    """MinerU 可执行入口（env PAPERPILOT_MINERU_CMD 可覆盖；找不到返回 None）。"""
    env = os.environ.get("PAPERPILOT_MINERU_CMD")
    if env:
        return env
    for cand in (MINERU_VENV / "Scripts" / "mineru.exe",
                 MINERU_VENV / "bin" / "mineru",
                 MINERU_VENV / "Scripts" / "mineru"):
        if cand.exists():
            return str(cand)
    return None


def mineru_available() -> tuple[bool, str]:
    """(可用?, 原因)。不可用原因面向用户可见。"""
    cmd = mineru_cmd()
    if not cmd:
        return False, (f"未找到 MinerU 可执行文件（预期 {MINERU_VENV}/Scripts/mineru.exe）；"
                       f"可用 PAPERPILOT_MINERU_CMD 指定")
    return True, ""


def mineru_version() -> str:
    """读 MinerU 版本（失败返回空串）——用于判断产物是否需要重跑。"""
    py = MINERU_VENV / "Scripts" / "python.exe"
    if not py.exists():
        py = MINERU_VENV / "bin" / "python"
    if not py.exists():
        return ""
    try:
        out = subprocess.run(
            [str(py), "-c",
             "import importlib.metadata as m;print(m.version('mineru'))"],
            capture_output=True, text=True, timeout=60)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# ── ingest.json（摄取元数据）──────────────────────────────────────────────────


def meta_path(stem: str) -> Path:
    return MINERU_OUT / stem / META_NAME


def read_meta(stem: str) -> dict[str, Any] | None:
    p = meta_path(stem)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _write_meta(stem: str, meta: dict[str, Any]) -> None:
    p = meta_path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def stamp_pdf_fingerprint(pdf_name: str) -> bool:
    """把当前 PDF 的内容指纹写进 `ingest.json`（**历史产物收编**用；成功返回 True）。

    场景：存量产物没有指纹记录（升级前生成）。首次运行按 mtime 关系收编后调用本函数补写，
    之后同一篇即走**严格 sha 比对**；下次文件被换掉就会被立刻发现。
    没有 `ingest.json`（从未摄取过的老产物 / 评测语料）→ 返回 False，不做无谓写盘。
    """
    stem = Path(pdf_name).stem
    meta = read_meta(stem)
    fp = paper_identity.fingerprint(pdf_name)
    if meta is None or fp is None:
        return False
    meta["pdf_fingerprint"] = fp
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_meta(stem, meta)
    return True


# ── 问答闸门 ──────────────────────────────────────────────────────────────────


def qa_blocked_reason(pdf_name: str) -> str:
    """问答是否应被拒绝：MinerU **明确失败** → 返回给用户的原因；否则空串（放行）。

    只拦 status=='failed'：未摄取（老产物 / QASPER / 评测语料）与 MinerU 不适用
    的情形一律放行，避免把历史评测集打挂。
    """
    if pdf_name.startswith("qasper_") and pdf_name.endswith(".qpdf"):
        return ""          # QASPER 无 PDF，本就不走 MinerU
    meta = read_meta(Path(pdf_name).stem)
    if not meta:
        return ""
    m = meta.get("mineru") or {}
    if m.get("status") != "failed":
        return ""
    reason = str(m.get("reason") or "未知原因")
    return ("问答暂不可用：本篇的版面解析（MinerU）未成功 —— "
            f"{reason}。\n报告不受影响（基于 PDF 文本层生成，可直接阅读）；"
            "如需问答，请修复 MinerU 环境后重跑摄取。")


# ── MinerU 子进程 ─────────────────────────────────────────────────────────────


def run_mineru(pdf_name: str, *, timeout: int | None = None,
               force: bool = False) -> dict[str, Any]:
    """对 assets/papers/<pdf_name> 跑 MinerU，产物落 out_mineru/<stem>/。

    Returns: {"status": "ok"|"failed"|"skipped", "reason": str, "seconds": float}
    """
    stem = Path(pdf_name).stem
    pdf = PAPERS_DIR / pdf_name
    if not pdf.exists():
        return {"status": "failed", "reason": f"论文库缺少 PDF：{pdf}", "seconds": 0.0}

    ok, why = mineru_available()
    if not ok:
        return {"status": "failed", "reason": why, "seconds": 0.0}

    from paperpilot.tools.mineru_bridge import chunks_from_mineru_dir
    outdir = MINERU_OUT / stem
    old = read_meta(stem) or {}
    old_ver = (old.get("mineru") or {}).get("version")
    cur_ver = mineru_version()
    # 复用规则：产物可解析，且（无版本记录=历史产物直接收编）或（版本一致）
    reusable_ver = (not old_ver) or (old_ver == cur_ver)
    # **身份校验**（2026-09-13 审查项 2）：MinerU 产物是按 `<stem>` 存的，
    # 同名不同内容时必须重跑，否则检索视图会一直来自旧的 PDF。
    fp = paper_identity.fingerprint(pdf_name)
    rec = old.get("pdf_fingerprint")
    same_pdf = (not fp) or (not rec) or (rec.get("sha256") == fp.get("sha256"))
    if not force and same_pdf and outdir.is_dir() and reusable_ver:
        if chunks_from_mineru_dir(outdir) is not None:
            return {"status": "ok", "reason": "复用已有产物", "seconds": 0.0,
                    "reused": True, "version": cur_ver}

    t0 = time.time()
    try:
        proc = subprocess.run(
            [mineru_cmd() or "", "-p", str(pdf), "-o", str(outdir),
             "-b", MINERU_BACKEND],
            capture_output=True, text=True,
            timeout=timeout or MINERU_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"status": "failed",
                "reason": f"MinerU 超时（>{timeout or MINERU_TIMEOUT}s；"
                          f"可调 PAPERPILOT_MINERU_TIMEOUT）",
                "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "reason": f"MinerU 启动失败：{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1)}
    secs = round(time.time() - t0, 1)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return {"status": "failed",
                "reason": f"MinerU 退出码 {proc.returncode}（多见于无 GPU/显存不足）。"
                          f"尾部输出：{tail}",
                "seconds": secs}
    if chunks_from_mineru_dir(outdir) is None:
        return {"status": "failed", "reason": "MinerU 已退出但未找到 content_list 产物",
                "seconds": secs}
    return {"status": "ok", "reason": "", "seconds": secs}


# ── 主入口 ────────────────────────────────────────────────────────────────────


def ingest(pdf_name: str, *, force: bool = False, skip_llm: bool = False,
           mineru: bool = True, verbose: bool = True) -> dict[str, Any]:
    """摄取一篇论文：论文库 → MinerU（检索）→ 报告（pymupdf）。

    Args:
        pdf_name: assets/papers 下的文件名
        force: True 时重跑 MinerU 与报告链
        skip_llm: True 时只装配已有报告产物（不调 LLM）
        mineru: False 时跳过 MinerU（报告仍生成；检索退化为纯 pymupdf）
    """
    stem = Path(pdf_name).stem
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    MINERU_OUT.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = read_meta(stem) or {"pdf": pdf_name, "stem": stem,
                                               "history": []}
    meta["pdf"] = pdf_name
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")

    # 0) **身份校验**（2026-09-13 审查项 2）：产物是否属于**当前这份** PDF。
    #    同名不同内容（用户换了文件 / 覆盖了同名 PDF）→ 旧产物一律不认，强制重建。
    #    历史产物（无指纹记录）按 mtime 关系收编（详见 paper_identity 模块注释）。
    fp = paper_identity.fingerprint(pdf_name)
    if fp is not None:
        arts: list[Path] = []
        try:
            from paperpilot.pipeline import required_files
            req = required_files(pdf_name)
            arts = list(req) + [req[0].with_name(f"{stem}.report.json")]
        except Exception:  # noqa: BLE001  拿不到产物清单不影响主流程
            arts = []
        verdict, why = paper_identity.check(pdf_name, meta.get("pdf_fingerprint"), arts)
        if verdict == "stale" and not force:
            print(f"[ingest] ⚠ {why} → **强制重建**（不采用旧产物）")
            meta.setdefault("history", []).append(
                {"at": meta["updated_at"], "event": "pdf_changed", "detail": why})
            force = True
        elif verdict == "adopt":
            if paper_identity.strict():
                print(f"[ingest] ⚠ {why}（PAPERPILOT_PDF_ID_STRICT=1）→ 重建")
                force = True
            else:
                print(f"[ingest] 注意：{why}；本次收编并写回指纹，之后即严格比对")

    # 1) MinerU（检索用）
    if mineru:
        if verbose:
            print(f"[ingest] {pdf_name} → MinerU（backend={MINERU_BACKEND}）…")
        m = run_mineru(pdf_name, force=force)
        m["version"] = mineru_version()
        m["cmd"] = mineru_cmd() or ""
        meta["mineru"] = m
        if verbose:
            tag = "ok" if m["status"] == "ok" else "FAILED"
            print(f"  [mineru] {tag} {m['seconds']}s {m.get('reason','')[:160]}")
    else:
        meta.setdefault("mineru", {"status": "skipped", "reason": "本次未启用 MinerU"})

    # 2) 报告链（pymupdf；MinerU 失败也照常出报告）
    if verbose:
        print("[ingest] 生成报告（pymupdf 链路）…")
    from paperpilot.pipeline import process_pdf
    report = process_pdf(pdf_name, force=force, skip_llm=skip_llm,
                         verbose=verbose)

    # 3) 检索视图体检（注入了几页/几块，落 meta 供排查）
    diag: dict[str, Any] = {}
    try:
        from paperpilot.agents.document_cache import ordered_chunks, retrieval_chunks
        base = ordered_chunks(pdf_name)
        view = retrieval_chunks(pdf_name)
        n_aug = sum(1 for a, b in zip(base, view) if len(a.text) != len(b.text))
        diag = {"n_chunks": len(base), "n_chunks_augmented": n_aug}
    except Exception as e:  # noqa: BLE001
        diag = {"error": f"{type(e).__name__}: {e}"}
    meta["retrieval_view"] = diag
    # 写回本次的 PDF 内容指纹（下次即走严格比对）
    if fp is not None:
        meta["pdf_fingerprint"] = fp

    _write_meta(stem, meta)
    if verbose:
        print(f"[ingest] 完成：{meta_path(stem)} ｜ 检索视图 {diag}")
    return {"meta": meta, "report": report}
