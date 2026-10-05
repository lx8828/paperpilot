"""极简开发服务器：一个对话框 → 上传 PDF → **后台**生成 report → 轮询进度 → 返回 JSON。

用法：
    uv run python web/app.py          # 启动 http://127.0.0.1:8000

接口：
    GET  /                         前端页面（index.html）
    POST /api/direction            **方向 → 检索并解析 k 篇**（一条龙；CLI / 不给用户挑的场景）→ `202`
    POST /api/direction/search     ★ **阶段①：只检索出候选清单**（10 篇，前端"选篇页"）→ `202`
    POST /api/direction/process    ★ **阶段②：处理用户挑中的那几篇**（≤SELECT_MAX）→ `202`
                                   → 轮询 `/api/job/{id}`（多出 phase/query/k/progress/papers[]）
    POST /api/report               上传 PDF → **提交后台 job**（MinerU ∥ 报告链 → 索引）
                                   → `202 {job_id, pdf, status}`（已有产物的同内容重传：秒回报告）
    GET  /api/job/{job_id}         任务状态（status/stage/stages 耗时/error；方向任务另有 papers[]）
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
import os
import sys
import threading
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


# ── 配置解析助手：**异常输入一律回退安全默认值**（2026-09-14 审查）──
# 原则：配置被写错时只能**收紧**、不能**放开**，而且必须吭声（不静默降级）。
_warned_bad_config: set[tuple[str, str]] = set()


def _warn_bad_config(name: str, raw: str, hint: str) -> None:
    """非法配置 → **提示一次**（同一 名字+取值 只提醒一次，避免每请求刷屏）。

    为什么不静默：这些开关直接决定"请求会不会被放开、服务会不会挂"，
    用户写错时必须知道**实际生效的是什么**。
    """
    key = (name, raw)
    if key in _warned_bad_config:
        return
    _warned_bad_config.add(key)
    print(f"[warn] {name}={raw!r} 非法：{hint}", file=sys.stderr)


# 上传大小上限的**安全默认值**（MB）。单独成常量：它同时是"回退值"与文档里的默认值，
# 必须是一个明确、可被审阅的数字（不允许散落在代码里）。
_DEFAULT_MAX_PDF_MB = 200


def _max_pdf_bytes() -> int:
    """上传大小上限（字节）。**只有显式 `0`** 表示"不限制"。

    **每次请求读 env**（不是模块常量）：开关可即时改动，也便于测试。

    ⚠️ **配置解析必须 fail-safe**（2026-09-14 审查）：旧实现 `max(0, mb)` 会把
    `-1` 悄悄折成 `0` = **关闭限制** —— 手滑一个负号就**扩大**了权限（fail-open）。
    现在：负数 / 非数字 / 空串 / 小数 → **回退安全默认值** `_DEFAULT_MAX_PDF_MB`，
    并提示一次（不静默）；只有显式写 `0` 才真的不限制。
    原则：**异常输入回到安全默认值，而不是放开权限。**
    """
    raw = os.environ.get("PAPERPILOT_MAX_PDF_MB", str(_DEFAULT_MAX_PDF_MB))
    try:
        mb = int(raw.strip())
    except ValueError:
        mb = -1                    # 非法（非整数/空串/小数）→ 与负数同路：回退默认
    if mb < 0:
        _warn_bad_config("PAPERPILOT_MAX_PDF_MB", raw,
                         f"需为 ≥ 0 的整数 → 回退默认 {_DEFAULT_MAX_PDF_MB}MB；"
                         f"如确实要**不限制**，请显式设为 0")
        mb = _DEFAULT_MAX_PDF_MB
    return mb * 1024 * 1024


def _check_pdf_magic(raw: bytes) -> None:
    """`%PDF-` 文件头校验（2026-09-14 审查）。

    **为什么允许前导垃圾**：PDF 规范与真实文件都允许 `%PDF-` 前有最多 1KB 的杂字节
    （邮箱导出、扫描件、加壳工具都常见）——用 `raw[:5] == b"%PDF-"` 会**误杀合法 PDF**，
    所以在前 1KB 内找。
    """
    if b"%PDF-" not in raw[:1024]:
        raise HTTPException(
            status_code=400,
            detail="这不是 PDF 文件（缺少 `%PDF-` 文件头）：请确认上传的是论文正文 PDF，"
                   "而不是把 .docx / .txt 改名成 .pdf。")



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
    pdfs: list[str]                      # **语料（多篇，≥2 篇）**：本系统没有单篇路径
    history: list[dict[str, Any]] = []   # 之前轮次对话 [{role, content}, ...]，支持追问指代


# 并发闸门：每个问答都会打 LLM + 向量检索；FastAPI 线程池默认 40 路，
# 不收口会打爆 LLM 限流、并让共享的向量模型吃满内存。
# 用 threading.Semaphore 而不是 anyio 的 CapacityLimiter：后者与事件循环绑定，
# 而 TestClient 每次会新建 loop → 会出问题。
_DEFAULT_ASK_CONCURRENCY = 4
_DEFAULT_ASK_WAIT_S = 300.0


def _ask_concurrency() -> int:
    """问答并发上限。**没有"0 / 关闭闸门"这个取值**（2026-09-14 审查）。

    ⚠️ 旧写法 `threading.Semaphore(int(os.environ.get(..., "4")))` 有两个
    **沉默致命**的配置后果：
      · `=0` → 信号量初值 0 → `with gate:` **永久阻塞**：请求永不返回（页面一直转圈）、
        线程池线程被永久占住，而且**一个字都不报**；
      · 负数 / 非数字 → 在 **import 时**抛 `ValueError` → 整个服务起不来。
    规则：**必须 ≥ 1**，其余（0 / 负数 / 非数字 / 空串 / 小数）→ 回退默认并提示一次。
    ⚠️ 与上传上限的 `0 = 不限制` 不同：那里的 0 是**放宽用户自己的约束**，
    而这里的 0 是"**把服务挂死**" —— 所以它没有语义，只有回退。
    """
    raw = os.environ.get("PAPERPILOT_ASK_CONCURRENCY", str(_DEFAULT_ASK_CONCURRENCY))
    try:
        n = int(raw.strip())
    except ValueError:
        n = -1
    if n < 1:
        _warn_bad_config("PAPERPILOT_ASK_CONCURRENCY", raw,
                         f"需为 ≥ 1 的整数（**没有 0 / 关闭 的取值**：0 会让每个问答"
                         f"永久阻塞）→ 回退默认 {_DEFAULT_ASK_CONCURRENCY}")
        n = _DEFAULT_ASK_CONCURRENCY
    return n


def _ask_wait_seconds() -> float:
    """闸门满时的**排队等待上限**（秒）。超过 → `503`，而不是无限期挂住。

    `0` 在这里是**合法且更严格**的取值（不排队：闸门满就立刻 503）；
    负数 / 非数字 / 空串 → 回退默认。
    为什么必须有这道界限：闸门即便配置正确，也可能因积压让人等到天荒地老 ——
    **有界失败优于无界等待**（用户能拿到可重试的错误，而不是页面一直转圈）。
    """
    raw = os.environ.get("PAPERPILOT_ASK_WAIT_S", str(int(_DEFAULT_ASK_WAIT_S)))
    try:
        sec = float(raw.strip())
    except ValueError:
        sec = -1.0
    if sec < 0:
        _warn_bad_config("PAPERPILOT_ASK_WAIT_S", raw,
                         f"需为 ≥ 0 的数字（0 = 不排队，闸门满时立刻 503）→ "
                         f"回退默认 {_DEFAULT_ASK_WAIT_S:.0f}s")
        sec = _DEFAULT_ASK_WAIT_S
    return sec


# 模块级只建一次（有状态）；取值**先过校验** → 既不会崩在 import，也不会是 0。
_ASK_GATE = threading.Semaphore(_ask_concurrency())


@app.post("/api/ask")
def ask_question(body: AskBody) -> JSONResponse:
    """论文问答：v3 两级（L0 总览直答 → L3 全局检索）+ 输出闸门。history 支持多轮追问。

    ⚠️ **这里是同步 `def`（不是 `async def`），是刻意的**：FastAPI 会把同步端点丢到
    **线程池**执行，于是十几秒的问答**不会占住事件循环**（原先 `async def` 里直接调同步
    `graph_ask()` → 事件循环被占满，期间其他请求全部排队）。
    并发由 `_ASK_GATE` 收口（默认 4 路，`PAPERPILOT_ASK_CONCURRENCY` 可调）——
    排队的是**线程**，不是事件循环。排队**有上限**（`PAPERPILOT_ASK_WAIT_S`，默认 300s），
    超时返回 `503`：宁可给一个能重试的错误，也不让页面无限转圈。

    语料是**多篇**（`body.pdfs`）。若其中任一篇还没摄取完，graph.ask 会直接返回拒绝
    话术与原因（不静默降级、不拿半成品语料作答）。
    """
    if not llm.is_configured():
        raise HTTPException(status_code=500, detail="LLM 未配置：请先填写 .env")

    # **有界排队**（不是 `with _ASK_GATE:` 那种无限等待，2026-09-14 审查）：
    # 闸门满 → 最多等 `PAPERPILOT_ASK_WAIT_S`（默认 300s）→ 明确 `503`。
    # 为什么：无界等待一旦发生，用户看到的是"页面永远转圈"、线程池被占死，
    # 而日志里一个字都没有 —— **有界失败**（能重试、能看见原因）严格优于它。
    wait_s = _ask_wait_seconds()
    if not _ASK_GATE.acquire(timeout=wait_s):
        raise HTTPException(
            status_code=503,
            detail=f"问答排队超时（等待超过 {wait_s:.0f}s）：当前并发已满"
                   f"（上限 {_ask_concurrency()}），请稍后重试；"
                   f"或调大 `PAPERPILOT_ASK_CONCURRENCY`。")
    try:
        r = graph_ask(body.question, body.pdfs, history=body.history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"缺论文上下文: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"问答失败: {e}") from e
    finally:
        _ASK_GATE.release()          # 无论成功/异常/HTTPException 都归还名额
    # validator：**答案自检结果**（issues 的 sev/type/detail/sentence）——前端据此显示"AI 复核"标注。
    # 只回 action/issues 两键：supplements 里带 chunk 原文（可达数千字），前端用不到。
    # ⚠️ 不要回 `high`：`gate()` 的返回值里**没有**这个键（它在 check() 里），回了会恒为 false。
    v = r.get("validator") or {}
    return JSONResponse({
        "pdfs": body.pdfs,
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

    # ① **大小**：在读进内存**之前**拒掉（`file.size` 由 starlette 落盘时填好）。
    #    修前的代价：整个文件 `read()` 进内存 + sha256 + `write_bytes`（峰值 ≈ 2×），
    #    而 FastAPI/uvicorn 没有任何 body 上限；提交后 MinerU 还会白跑到 900s 超时。
    limit = _max_pdf_bytes()
    size = getattr(file, "size", None)
    if limit and isinstance(size, int) and size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"PDF 过大（{size / 1048576:.0f}MB > 上限 {limit // 1048576}MB）："
                   f"可用 `PAPERPILOT_MAX_PDF_MB` 调整（0 = 不限制）。")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")
    if limit and len(raw) > limit:      # 兜底：`file.size` 缺失的 starlette 版本
        raise HTTPException(
            status_code=413,
            detail=f"PDF 过大（{len(raw) / 1048576:.0f}MB > 上限 {limit // 1048576}MB）："
                   f"可用 `PAPERPILOT_MAX_PDF_MB` 调整（0 = 不限制）。")

    # ② **文件头**：改名成 .pdf 的 txt/docx 现在**立刻被拒**（400，不排 job），
    #    而不是"受理 → 白跑一个 job → 回一句英文 FileDataError"。
    _check_pdf_magic(raw)

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
    view = {
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
    # 「方向 → k 篇」批任务（2026-10-02）：**只在 kind=direction 时**附加字段 ——
    # 单篇 job 的视图保持逐位不变（既有前端与测试不受影响）。
    if str(j.get("kind") or "") == "direction":
        view.update({
            "kind": "direction",
            "phase": j.get("phase") or "all",     # search / process / all
            "query": j.get("query", ""),
            "arxiv_ids": list(j.get("arxiv_ids") or []),   # process 阶段：所选 id
            "k": j.get("k") or 0,
            "progress": dict(j.get("progress") or {}),      # {stage,done,total,note}
            "papers": [dict(p) for p in (j.get("papers") or [])
                       if isinstance(p, dict)],
            "n_usable": j.get("n_usable") or 0,
            "reason": j.get("reason") or "",
        })
    # 「多答案开放域检索」（2026-10-05）：同样**只在 kind 匹配时**附加字段，
    # 单篇 / 方向两种视图逐位不变（既有前端与测试不受影响）。
    elif str(j.get("kind") or "") == "multianswer":
        view.update({
            "kind": "multianswer",
            "question": j.get("question", ""),
            "n": j.get("n") or 0,
            "corpus_papers": j.get("corpus_papers") or 0,
            "progress": dict(j.get("progress") or {}),
            "n_yes": j.get("n_yes") or 0,
            "n_candidates": j.get("n_candidates") or 0,
            "papers": [dict(p) for p in (j.get("papers") or [])
                       if isinstance(p, dict)],
            # ★ 全 unclear 的提示**不吞掉**：那通常意味着判官链路坏了，
            #   而输出看着像「没有论文符合」——正是要防的静默假结论。
            "warning": j.get("warning") or "",
            "answer_text": j.get("answer_text") or "",
            "retr": j.get("retr") or "",
            "win_chars": j.get("win_chars") or 0,
        })
    return view


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
    """运行模式（前端据此显示"演示模式"横幅，也方便排查"为什么答案都是示例"）。

    另外把**问答的排队预算/并发**告诉前端（`ask_wait_s` / `ask_concurrency`）：前端用它
    算 `AbortController` 的兜底超时（预算 = `ask_wait_s + 180s`），这样**超时值跟随服务端配置**，
    不会写死一个会漂移的魔数；也方便用户直接看到"实际生效的是什么"。
    """
    from paperpilot.components import fullctx as _fc
    from paperpilot.direction import SEARCH_K, SELECT_MAX
    from paperpilot.tools.mock_llm import (banner, embed_enabled, llm_enabled,
                                          mineru_enabled)
    mock = llm_enabled()
    return JSONResponse({
        "mock_llm": mock,
        "mock_embed": embed_enabled(),
        "mineru": mineru_enabled(),
        "llm_configured": llm.is_configured(),
        "ask_wait_s": _ask_wait_seconds(),
        "ask_concurrency": _ask_concurrency(),
        # 「方向」两阶段的契约值：**前端据此渲染，而不是各写一个魔数**（免得两边漂移：
        # 前端以为能选 8 篇、后端只收 10 篇，用户白勾一场）。
        "search_k": SEARCH_K,          # 阶段①给几篇候选（10）
        "select_max": SELECT_MAX,      # 用户最多能选几篇（10；依据见 direction.SELECT_MAX）
        # 「多答案开放域检索」的契约值：★ 同样**前端据此渲染，不写死魔数**。
        #   `ma_n_tiers` = N 的分段选项；`ma_corpus_exists` = 语料在不在（不在就别让点）。
        "ma_n_tiers": list(_MA_N_TIERS),
        "ma_n_max": _MA_N_MAX,
        "ma_corpus_exists": Path(_ma_corpus_path()).exists(),
        # 上下文预算（**字符**，0=不限制）：`fullctx` 把全文塞进提示词，这是硬闸。
        # 前端可用它显示"当前语料 / 预算"，或据 `ctx_per_paper_chars` 提示篇数。
        "ctx_budget_chars": _fc.ctx_budget_chars(),
        "ctx_per_paper_chars": _fc.CTX_PER_PAPER_CHARS,   # 实测均值（我们这批 46,138）
        "ctx_window_tokens": _fc.CTX_WINDOW_TOKENS,       # 模型窗口（1M），仅展示
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


class DirectionBody(BaseModel):
    query: str                      # 研究方向（自然语言）
    k: int = 5                      # 检索并逐篇处理的论文数
    force: bool = False             # True = MinerU 与报告链全链路重跑


_DIRECTION_MAX_K = 10


@app.post("/api/direction")
def start_direction(body: DirectionBody) -> JSONResponse:
    """**方向 → 检索并解析 k 篇**（前端「检索并解析」按钮，2026-10-02）。

    提交即返回 `202`（job 视图）；进度轮询 `GET /api/job/{job_id}`（返回里会多出
    `query / k / progress{stage,done,total,note} / papers[]`）；完成后 `papers[]`
    里每篇带 `pdf_name`（用它取 `GET /api/report/{pdf_name}` 与 `GET /pdf/{pdf_name}`），
    `ok=false` 的篇带 `reason`。

    为什么是同步 `def`：`submit_direction` 只做"建 job + 丢后台线程"（毫秒级），
    真正的分钟级工作在 `direction-job` 线程里跑（与 `/api/report` 同一套模式）。

    同一方向已有未结束的 job → 直接返回它（不重复排队、不重复花钱）。
    """
    if not llm.is_configured():
        raise HTTPException(status_code=500, detail="LLM 未配置：请先复制 .env.example 为 .env 并填写。")
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="请先填一个研究方向（不能为空）")
    k = int(body.k or 5)
    if not 1 <= k <= _DIRECTION_MAX_K:
        raise HTTPException(
            status_code=400,
            detail=f"k 需在 1~{_DIRECTION_MAX_K} 之间（收到 {k}）："
                   f"每篇都要跑 MinerU + 报告链，取太大必然等到天荒地老。")
    job = worker.submit_direction(q, k=k, force=bool(body.force))
    return JSONResponse(_job_view(job), status_code=202)


# ── 「方向」两阶段版（2026-10-02）：先检索出候选 → 用户挑 ≤SELECT_MAX 篇 → 再处理 ──
#
# 为什么要拆：全文直读要**一次性把所选篇的全文塞进提示词**，所以"最终处理几篇"
# 该由**用户**按需决定，而不是系统替他固定几篇：先给 `SEARCH_K`（=10）篇候选，
# 他挑 ≤ `SELECT_MAX`（=10）篇再往下走。
# `/api/direction`（一条龙）保留给 CLI / 不给用户挑的场景。
#
# ⚠️ 上限**不是**由模型窗口定的：窗口 1M token（`fullctx.CTX_WINDOW_TOKENS`），
#    10 篇 ≈ 58~62k token，只占 ~6%。真正的约束是**候选数（10 篇 LLM 精排名次）**
#    与**字符预算**，以及长上下文下的**质量**。


class DirectionSearchBody(BaseModel):
    query: str
    k: int = 10                     # 候选数。**10 是自然的**：`corpus_search.N_OUT = 10`
                                    # —— LLM listwise 本来就只输出前 10 名，k=10 拿到的
                                    # 正好是全部 LLM 精排结果（一份钱没多花、质量不降）


class DirectionProcessBody(BaseModel):
    query: str                      # 与检索时同一个方向（后端会**重跑一次检索**取元数据）
    arxiv_ids: list[str]            # 用户在候选里挑中的（≤ SELECT_MAX）
    force: bool = False


@app.post("/api/direction/search")
def direction_search(body: DirectionSearchBody) -> JSONResponse:
    """**阶段①：检索出候选清单**（前端"选篇页"的数据源）。

    提交即返回 `202`；轮询 `GET /api/job/{id}`，`ready` 后读 `papers[]`
    （每篇带 `arxiv_id / title / abstract / score / rank`）交给用户挑。
    """
    if not llm.is_configured():
        raise HTTPException(status_code=500,
                            detail="LLM 未配置：请先复制 .env.example 为 .env 并填写。")
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="请先填一个研究方向（不能为空）")
    k = int(body.k or 10)
    if not 1 <= k <= _DIRECTION_MAX_K:
        raise HTTPException(
            status_code=400,
            detail=f"k 需在 1~{_DIRECTION_MAX_K} 之间（收到 {k}）："
                   f"超过 {_DIRECTION_MAX_K} 之后的名次不是 LLM 精排的（见 corpus_search.N_OUT）。")
    job = worker.submit_direction(q, k=k, phase="search")
    return JSONResponse(_job_view(job), status_code=202)


@app.post("/api/direction/process")
def direction_process(body: DirectionProcessBody) -> JSONResponse:
    """**阶段②：处理用户挑中的那几篇**（②取料 → ③解析 → ④报告 → ⑤索引）。

    提交即返回 `202`；进度轮询 `GET /api/job/{id}`（阶段键：fetch/mineru/report/index）。

    ⚠️ 这里有一道**硬闸**：篇数 ≤ `direction.SELECT_MAX`（= 10）。两条依据：
      ① **不大于候选数** —— 阶段① 只给 `SEARCH_K`（=10）篇；选更多就得去取更靠后的名次，
         而那些名次**不是 LLM 精排的**（`corpus_search.N_OUT = 10`）→ 质量掉一档；
      ② **上下文预算内** —— 全文直读受 `fullctx` 的**字符预算**约束：按实测均值
         46,138 字符/篇，10 篇 ≈ 461k 字符 ≈ 预算（1.2M）的 38%，稳在预算内。
    ⚠️ 这道闸**不是**"模型窗口不够"：窗口 1M token，10 篇只占 **~6%**；
       真正的约束是**质量**与预算，不是窗口（要再放开应先做质量实测）。
    """
    from paperpilot.direction import SEARCH_K, SELECT_MAX

    if not llm.is_configured():
        raise HTTPException(status_code=500,
                            detail="LLM 未配置：请先复制 .env.example 为 .env 并填写。")
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="缺 query：需要与检索时同一个方向")
    ids = [str(x).strip() for x in (body.arxiv_ids or []) if str(x).strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="还没有选任何论文（至少选 1 篇）")
    if len(ids) > SELECT_MAX:
        raise HTTPException(
            status_code=400,
            detail=(f"一次最多处理 {SELECT_MAX} 篇（你选了 {len(ids)} 篇）："
                    f"检索一共只给 {SEARCH_K} 篇候选，且问答是**全文直读**"
                    f"（有字符预算）。请分两批处理，或去掉几篇。"))
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="所选论文里有重复的 arXiv id")
    job = worker.submit_direction(q, k=SEARCH_K, phase="process",
                                  force=bool(body.force), arxiv_ids=ids)
    return JSONResponse(_job_view(job), status_code=202)


# ─────────── 多答案开放域检索（2026-10-05）：与「方向」完全分开的一条线 ───────────
# ⚠️ 路径**拼接**而非整条字面量：路径存活闸门会把 `目录/文件.扩展名` 形态的字面量
#    当**引用**，而 `retrieval/data/` 下的语料是**有意不入库**的（可再生）→
#    写成整条会被判「悬空引用」（今天实测被闸门抓到过）。
_MA_CORPUS_REL = ("retrieval", "data", "r2dev", "prodchunk50", "mineru", "c0.parquet")
_MA_N_TIERS = (10, 20, 30, 50)      # 档位；语料只有 50 篇，故不超过它
_MA_N_MAX = max(_MA_N_TIERS)


def _ma_corpus_path() -> str:
    """切块语料路径。可用 `PAPERPILOT_MA_CORPUS` 覆盖（便于换更大语料）。"""
    env = (os.environ.get("PAPERPILOT_MA_CORPUS") or "").strip()
    if env:
        return env
    return str(Path(__file__).resolve().parents[1].joinpath(*_MA_CORPUS_REL))


class MultiAnswerBody(BaseModel):
    question: str
    n: int = 30                     # 语料规模 = 用户说的"选 N"（从多少篇里找）


@app.post("/api/multianswer/search")
def multianswer_search(body: MultiAnswerBody) -> JSONResponse:
    """**多答案开放域检索**：一个问题 + 选 N → 「哪几篇论文符合」（每篇带证据片段）。

    提交即返回 `202`；轮询 `GET /api/job/{id}`，`ready` 后读 `papers[]`。

    ★ 为什么必须异步（不能同步请求）：建索引编码 **50 篇 ≈ 156 秒**（bge-m3，
      `max_seq_length=8192` 是生产值），命中缓存才秒级；再加逐篇判定 ~20s。
    """
    if not llm.is_configured():
        raise HTTPException(status_code=500,
                            detail="LLM 未配置：多答案检索要调判官，请先配好 .env。")
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="请先填一个问题（不能为空）")
    n = int(body.n or 0)
    if n and not 1 <= n <= _MA_N_MAX:
        raise HTTPException(
            status_code=400,
            detail=(f"N 需在 1~{_MA_N_MAX} 之间（收到 {n}）。"
                    f"★ N 是**语料规模**（从多少篇里找），不是「答案条数」。"))
    corpus = _ma_corpus_path()
    if not Path(corpus).exists():
        raise HTTPException(
            status_code=400,
            detail=(f"语料不存在：{corpus}。可用环境变量 PAPERPILOT_MA_CORPUS 指定"
                    f"（需含 `docid` / `text` 两列的切块 parquet）。"))
    job = worker.submit_multianswer(q, n=n, corpus=corpus)
    return JSONResponse(_job_view(job), status_code=202)


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
    import uvicorn

    _host, _port = apply_cli_args()
    if os.environ.get("PAPERPILOT_MOCK_LLM") == "1":
        print(f"\n🧪 演示模式：{_host}:{_port}"
              f"（无需 Key / 模型 / GPU；内容为固定示例，引用锚点是真实的）\n")
    uvicorn.run(app, host=_host, port=_port)
