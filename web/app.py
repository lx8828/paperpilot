"""极简开发服务器：一个对话框 → 上传 PDF → **后台**生成 report → 轮询进度 → 返回 JSON。

用法：
    uv run python web/app.py          # 启动 http://127.0.0.1:8000

接口：
    GET  /                         前端页面（index.html）
    POST /api/report               上传 PDF → **提交后台 job**（MinerU ∥ 报告链 → 索引）
                                   → `202 {job_id, pdf, status}`（已有产物的同内容重传：秒回报告）
    GET  /api/job/{job_id}         任务状态（status/stage/stages 耗时/error）
    GET  /api/jobs/latest?pdf=     该论文最近一次 job（刷新页面后恢复进度）
    POST /api/job/{job_id}/cancel  请求取消（阶段边界生效；MinerU 真终止子进程）
    POST /api/job/{job_id}/retry   重试（已完成的阶段自动复用，很便宜）
    GET  /api/report/{name}        job ready 后取报告 JSON（含 upload_note / mineru_warning）
    POST /api/ask                  提问接口（摄取未完成时由闸门明确拒答）

为什么要异步（2026-09-13）：摄取是**分钟级**任务，原先挂在 HTTP 请求里 → 用户干等，
且阻塞调用占住事件循环（其他请求一起排队）。现在提交即返回，进度可查、可取消、可重试。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from paperpilot import jobs, worker
from paperpilot.graph import ask as graph_ask
from paperpilot.ingest import read_meta
from paperpilot.tools import llm

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent  # web/ → 项目根
PAPERS_DIR = ROOT / "assets" / "papers"
VENDOR_DIR = WEB_DIR / "vendor"   # pdf.js 本地静态资源

llm._load_dotenv(str(ROOT))

app = FastAPI(title="PaperPilot Dev Console")

# 前端静态资源（pdf.js 等），必须在 "/" 路由之前挂载（不覆盖根路由）
if VENDOR_DIR.is_dir():
    app.mount("/vendor", StaticFiles(directory=VENDOR_DIR), name="vendor")


def _safe_pdf_name(name: str) -> str:
    """只保留文件名 + 强制 .pdf 后缀，防路径穿越。"""
    base = Path(name or "paper.pdf").name
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base


def plan_upload(name: str, raw: bytes) -> dict[str, Any]:
    """这次上传**该落到哪个文件**（C 档：按内容判定，同名不同内容不再冒充同一篇）。

    · 目标名不存在          → 落原名（新论文）
    · 目标名存在 + 内容相同  → `reuse=True`：命中 `report.json` 可直接回（幂等，不写盘）
    · 目标名存在 + 内容不同  → **另存为 `<stem>__<sha8>.pdf`**（新论文）并给出提示；
      原文件**一个字节都不动**（既防覆盖别篇，也避开 Windows 文件占用）

    背景（2026-09-13 审查项 2）：旧实现"文件名存在 + 有缓存"就直接回缓存且不写盘
    → 上传另一篇论文但文件名相同时，用户拿到的是**旧论文的报告**。
    """
    dest = PAPERS_DIR / name
    sha = hashlib.sha256(raw).hexdigest()
    if not dest.exists():
        return {"path": dest, "name": name, "sha": sha, "reuse": False, "note": ""}
    try:
        exist_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    except OSError:
        exist_sha = ""
    if exist_sha == sha:
        return {"path": dest, "name": name, "sha": sha, "reuse": True, "note": ""}
    new_name = f"{dest.stem}__{sha[:8]}.pdf"
    return {"path": PAPERS_DIR / new_name, "name": new_name, "sha": sha, "reuse": False,
            "note": (f"文件名「{name}」已属于另一篇论文（内容指纹不同）；"
                     f"本次上传已另存为「{new_name}」作为**新论文**处理。")}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/pdf/{name}")
async def serve_pdf(name: str) -> FileResponse:
    """给前端 pdf.js 渲染用：返回 assets/papers 下的 PDF 文件。"""
    safe = Path(name or "").name
    if not safe.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 .pdf 文件")
    path = PAPERS_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"PDF 不存在: {safe}")
    return FileResponse(path, media_type="application/pdf",
                        filename=safe)


class AskBody(BaseModel):
    question: str
    pdf: str
    history: list[dict[str, Any]] = []   # 之前轮次对话 [{role, content}, ...]，支持追问指代


@app.post("/api/ask")
async def ask_question(body: AskBody) -> JSONResponse:
    """论文问答：v3 两级（L0 总览直答 → L3 全局检索）+ 输出闸门。history 支持多轮追问。

    若该论文摄取时 MinerU 明确失败，graph.ask 会直接返回拒绝话术与原因（不静默降级）。
    """
    if not llm.is_configured():
        raise HTTPException(status_code=500, detail="LLM 未配置：请先填写 .env")
    try:
        r = graph_ask(body.question, body.pdf, history=body.history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"缺论文上下文: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"问答失败: {e}") from e
    # validator：**答案自检结果**（issues 的 sev/type/detail/sentence）——前端据此显示"AI 复核"标注。
    # 只回 action/issues 两键：supplements 里带 chunk 原文（可达数千字），前端用不到。
    # ⚠️ 不要回 `high`：`gate()` 的返回值里**没有**这个键（它在 check() 里），回了会恒为 false。
    v = r.get("validator") or {}
    return JSONResponse({
        "pdf": body.pdf,
        "question": body.question,
        "answer": r.get("answer", ""),
        "cites": r.get("cites", []),
        "debug": r.get("debug", {}),
        "route": r.get("route", []),
        "validator": {"action": v.get("action", ""),
                      "issues": list(v.get("issues") or [])[:8]},
    })


@app.post("/api/report")
async def upload_report(file: UploadFile = File(...)) -> JSONResponse:
    """上传 PDF → **提交后台 job**（`202` 立即返回 job_id）。

    产物就绪后用 `GET /api/report/{name}` 取报告；进度/取消/重试见 `/api/job/*`。

    **按内容判定落点**（2026-09-13 审查项 2，见 `plan_upload`）：
      · 同名同内容 → 幂等复用已有产物（不写盘、不排 job，直接回报告 JSON）；
      · 同名**不同内容** → 另存为 `<stem>__<sha8>.pdf` 当新论文处理（原文件不动）。
    旧实现只按文件名判断（"名字存在 + 有缓存"就直接回缓存），会把**别篇论文**的报告
    返回给用户；现在绝不发生。
    不覆盖已有文件这一点保留：避免 Windows 下文件被 WPS/阅读器占用导致 Permission denied。

    **为什么不再同步跑**（2026-09-13）：摄取是**分钟级**任务（MinerU + 报告链），
    挂在 HTTP 请求里会让用户全程干等，且阻塞调用会占住事件循环（其他请求一起排队）。
    现在提交即返回，worker 后台并行跑，前端轮询进度。
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 .pdf 文件")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")

    # **按内容判定落点**（2026-09-13 审查项 2）：同名 + 同内容 → 幂等复用；
    # 同名 + 不同内容 → 另存为 `<stem>__<sha8>.pdf` 当**新论文**处理（不覆盖、不冒充别篇）。
    plan = plan_upload(_safe_pdf_name(file.filename), raw)
    name = str(plan["name"])
    dest: Path = plan["path"]

    if plan["reuse"]:
        try:                     # 同内容：产物已在 → 秒回（不写盘、不排 job）
            payload = _report_payload(name)
            if payload is not None:
                payload["upload_note"] = "同名同内容：复用已有产物（幂等）"
                return JSONResponse(payload)
        except Exception:  # noqa: BLE001  缓存损坏则走正常流程
            pass

    if not dest.exists():     # 内容不同时 dest 是 `__<sha8>` 新名，不会覆盖别篇
        try:
            dest.write_bytes(raw)
        except PermissionError:
            raise HTTPException(
                status_code=409,
                detail=f"「{name}」已被其他程序（如 WPS/PDF 阅读器）占用，"
                       f"无法写入。请关闭占用它的窗口后重试；"
                       f"或换一个新文件名再上传。",
            ) from None
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"保存 PDF 失败: {e}") from e

    if not llm.is_configured():
        raise HTTPException(
            status_code=500,
            detail="LLM 未配置：请先复制 .env.example 为 .env 并填写。",
        )

    # **立即返回**：同一篇已有未结束 job 时 worker 会复用它（不重复排队）。
    job = worker.submit(name, note=plan["note"])
    return JSONResponse(_job_view(job), status_code=202)


# ───────────── 任务状态 / 取消 / 重试 / 取报告 ─────────────


def _job_view(j: dict[str, Any]) -> dict[str, Any]:
    """job → 前端用的精简视图（含阶段文案与各阶段耗时）。

    ⚠️ **必须是快照**（逐层浅拷贝）：`/api/report` 的 202 路径传入的是 worker
    **正在写**的同一个 dict（job 线程会继续往里加 `stages` 键）→ 直接把引用交给
    JSON 序列化，可能撞上 `RuntimeError: dictionary changed size during iteration`
    （低频但真实）；`/api/job/{id}` 那条读的是磁盘快照，不受影响。
    """
    status = str(j.get("status") or "")
    stages = {str(k): dict(v) for k, v in (j.get("stages") or {}).items()
              if isinstance(v, dict)}
    return {
        "job_id": j.get("job_id", ""),
        "pdf": j.get("pdf", ""),
        "status": status,
        "ready": status == jobs.STATUS_READY,
        "failed": status == jobs.STATUS_FAILED,
        "cancelled": status == jobs.STATUS_CANCELLED,
        "stage": j.get("stage", ""),
        "stage_label": jobs.stage_label(str(j.get("stage") or "")),
        "stages": stages,
        "created_at": j.get("created_at", ""),
        "started_at": j.get("started_at", ""),
        "finished_at": j.get("finished_at", ""),
        "elapsed": j.get("elapsed") or 0.0,
        "attempt": j.get("attempt") or 1,
        "error": j.get("error") or "",
        "note": j.get("note") or "",
    }


def _report_payload(name: str) -> dict[str, Any] | None:
    """读 `<stem>.report.json` 并补上**必须让用户看到**的提示（不静默）。

    - `upload_note`：同名不同内容 → 已另存为新论文（来自最近 job）；
    - `mineru_warning`：版面解析失败 → 问答不可用但报告可用。
    """
    stem = Path(_safe_pdf_name(name)).stem
    p = ROOT / "assets/artifacts/out_views" / f"{stem}.report.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  损坏当没有（前端会提示重试）
        return None
    m = (read_meta(stem) or {}).get("mineru") or {}
    if str(m.get("status")) == "failed":
        payload["mineru_warning"] = (
            "版面解析（MinerU）失败，问答将不可用；报告已正常生成。"
            f"原因：{str(m.get('reason') or '')[:200]}")
    j = jobs.latest(name)
    if j and j.get("note"):
        payload["upload_note"] = j["note"]
    if j:
        payload["job"] = {"job_id": j.get("job_id"), "elapsed": j.get("elapsed") or 0.0,
                          "stages": j.get("stages") or {}}
    return payload


@app.get("/api/meta")
async def api_meta() -> JSONResponse:
    """运行模式（前端据此显示"演示模式"横幅，也方便排查"为什么答案都是示例"）。"""
    from paperpilot.tools.mock_llm import (banner, embed_enabled, llm_enabled,
                                          mineru_enabled)
    mock = llm_enabled()
    return JSONResponse({
        "mock_llm": mock,
        "mock_embed": embed_enabled(),
        "mineru": mineru_enabled(),
        "llm_configured": llm.is_configured(),
        "banner": banner() if mock else "",
        "note": ("演示模式：概述/主张/答案为固定示例；引用锚点仍来自真实检索"
                 if mock else ""),
    })


@app.get("/api/report/{name}")
async def get_report(name: str) -> JSONResponse:
    """job ready 后取报告 JSON（含 upload_note / mineru_warning / 各阶段耗时）。"""
    payload = _report_payload(name)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"报告尚未生成：{_safe_pdf_name(name)}（若正在解析，请轮询 /api/job）")
    return JSONResponse(payload)


@app.get("/api/job/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    """任务状态（前端轮询用）：status / stage / stages（每阶段耗时）/ error。"""
    j = jobs.load(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    return JSONResponse(_job_view(j))


@app.get("/api/jobs/latest")
async def job_latest(pdf: str) -> JSONResponse:
    """该论文最近一次任务（刷新页面后用它恢复进度显示）。"""
    j = jobs.latest(pdf)
    if j is None:
        raise HTTPException(status_code=404, detail=f"该论文暂无摄取任务: {pdf}")
    return JSONResponse(_job_view(j))


@app.post("/api/job/{job_id}/cancel")
async def job_cancel(job_id: str) -> JSONResponse:
    """请求取消：立即置信号，worker 在**阶段边界**响应（MinerU 会真终止子进程）。"""
    if jobs.load(job_id) is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    j = jobs.request_cancel(job_id)
    return JSONResponse(_job_view(j or {}))


@app.post("/api/job/{job_id}/retry")
async def job_retry(job_id: str) -> JSONResponse:
    """重试：重新排一个 job（摄取幂等 + 分阶段缓存 → 已完成的阶段秒过）。"""
    old = jobs.load(job_id)
    if old is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    j = worker.retry(job_id)
    if j is None:
        raise HTTPException(status_code=409, detail="无法重试（任务仍在运行或论文已删除）")
    return JSONResponse(_job_view(j), status_code=202)


def apply_cli_args(argv: list[str] | None = None) -> tuple[str, int]:
    """解析命令行参数（含 `--mock` 演示模式）→ (host, port)。

    **不启动服务**（所以能被测试直接调用）：启动只发生在 `__main__`。
    `--mock` 一次性打开三个开关；若你已显式设过 `PAPERPILOT_MINERU`（如 `=1`）则尊重你的设置。
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(description="PaperPilot 开发服务器")
    ap.add_argument("--mock", action="store_true",
                    help="演示模式：不需要 API Key / 模型 / GPU（内容为固定示例）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    if args.mock:
        os.environ["PAPERPILOT_MOCK_LLM"] = "1"
        os.environ["PAPERPILOT_MOCK_EMBED"] = "1"
        os.environ.setdefault("PAPERPILOT_MINERU", "0")
    return args.host, args.port


if __name__ == "__main__":
    import os

    import uvicorn

    _host, _port = apply_cli_args()
    if os.environ.get("PAPERPILOT_MOCK_LLM") == "1":
        print(f"\n🧪 演示模式：{_host}:{_port}"
              f"（无需 Key / 模型 / GPU；内容为固定示例，引用锚点是真实的）\n")
    uvicorn.run(app, host=_host, port=_port)
