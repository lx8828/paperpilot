"""极简开发服务器：一个对话框 → 上传 PDF → 生成 report → 返回 JSON。

用法：
    uv run python web/app.py          # 启动 http://127.0.0.1:8000

接口：
    GET  /            前端页面（index.html）
    POST /api/report  上传 PDF → process_pdf() → 返回 PaperReport JSON
                      （已有产物缓存时秒回；新论文走全链路，耗时可到分钟级）

预留（下一步问答 RAG 用）：
    POST /api/ask     提问接口
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from paperpilot.graph import ask as graph_ask
from paperpilot.pipeline import process_pdf
from paperpilot.tools import llm

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent  # web/ → 项目根
PAPERS_DIR = ROOT / "src" / "paperpilot" / "storage" / "papers"
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


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/pdf/{name}")
async def serve_pdf(name: str) -> FileResponse:
    """给前端 pdf.js 渲染用：返回 storage/papers 下的 PDF 文件。"""
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
    history: list[dict] = []   # 之前轮次对话 [{role, content}, ...]，支持追问指代


@app.post("/api/ask")
async def ask_question(body: AskBody) -> JSONResponse:
    """论文问答：LangGraph 四层漏斗（L0→L1→L2→L3）。history 支持多轮追问。"""
    if not llm.is_configured():
        raise HTTPException(status_code=500, detail="LLM 未配置：请先填写 .env")
    try:
        r = graph_ask(body.question, body.pdf, history=body.history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"缺论文上下文: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"问答失败: {e}") from e
    return JSONResponse({
        "pdf": body.pdf,
        "question": body.question,
        "answer": r.get("answer", ""),
        "cites": r.get("cites", []),
        "debug": r.get("debug", {}),
        "route": r.get("route", []),
    })


@app.post("/api/report")
async def upload_report(file: UploadFile = File(...)) -> JSONResponse:
    """上传 PDF → process_pdf → report JSON。

    关键：同名论文已存在时【不覆盖写盘】——21 篇论文均已有产物缓存，
    直接复用即可，避免 Windows 下文件被 WPS/阅读器占用导致 Permission denied。
    只有磁盘上没有的同名（新论文）才需要写文件。
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 .pdf 文件")

    name = _safe_pdf_name(file.filename)
    dest = PAPERS_DIR / name

    # 论文已在 storage/papers 且 report.json 已生成 → 直接走缓存装配，不写盘
    if dest.exists():
        stem = dest.stem
        report_cache = ROOT / "out_views" / f"{stem}.report.json"
        if report_cache.exists():
            try:
                return JSONResponse(
                    json.loads(report_cache.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001  缓存损坏则走正常流程
                pass
        # 缓存缺失/损坏：尝试写盘重生成；文件被占用时给出中文提示
        try:
            dest.write_bytes(await file.read())
        except PermissionError:
            raise HTTPException(
                status_code=409,
                detail=f"「{name}」已被其他程序（如 WPS/PDF 阅读器）占用，"
                       f"无法更新。请关闭占用它的窗口后重试；"
                       f"或换一个新文件名再上传。",
            ) from None
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"保存 PDF 失败: {e}") from e
    else:
        try:
            dest.write_bytes(await file.read())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"保存 PDF 失败: {e}") from e

    if not llm.is_configured():
        raise HTTPException(
            status_code=500,
            detail="LLM 未配置：请先复制 .env.example 为 .env 并填写。",
        )
    try:
        report = process_pdf(name, verbose=False)
        payload = json.loads(report.model_dump_json())
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        print("=" * 60)
        print(f"[upload_report] {name} 生成失败，traceback：")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"生成失败: {e}") from e
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
