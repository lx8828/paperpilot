"""摄取任务（job）状态：**落盘可查、可取消、可重试**（2026-09-13）。

为什么需要：`/api/report` 原先在 HTTP 请求里同步跑完整条摄取链（MinerU + 报告链，
分钟级），用户全程干等，且阻塞调用占住事件循环（别的请求一起排队）。
现在改为「**提交即返回 job_id** → worker 后台跑 → 前端轮询」，
本模块负责这份 job 状态的**唯一真相**。

设计要点：
- **落盘**（`assets/artifacts/out_jobs/<job_id>.json`）：进程重启后前端轮询不会 404，
  也能把卡在 `running` 的 job 标成中断并提示重试（见 `recover_interrupted`）。
- **取消是"标记 + 边界生效"**：`threading.Event` 给 worker 在阶段边界检查；
  MinerU 是子进程，能真 terminate（见 `ingest.run_mineru(cancel=...)`）。
- **重试很便宜**：摄取链是幂等 + 分阶段缓存的（每个 stage 都是"产物在就复用"），
  重跑只会补没完成的那几步。
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
JOB_DIR = ROOT / "assets" / "artifacts" / "out_jobs"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_RUNNING)

# 阶段 → 前端/CLI 显示文案（`report:<名>` 是报告链内部的子阶段）
STAGE_LABEL: dict[str, str] = {
    "": "排队中…",
    "mineru": "版面解析（MinerU · GPU）…",
    "report": "生成报告（pymupdf + LLM）…",
    "report:claims": "抽取主张（claims · LLM）…",
    "report:view": "主张去重 / 打标 / 打分…",
    "report:skeleton": "构建论证骨架…",
    "report:figures": "识别图表 + 生成读图指南…",
    "report:report_text": "生成概述 / 导读 / 报告…",
    "report:assemble": "装配报告…",
    "index": "构建检索索引（向量，供问答）…",
    "done": "完成",
}

_lock = threading.RLock()
_events: dict[str, threading.Event] = {}
_recovered = False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_job_id() -> str:
    """`j-<yyyymmdd>-<6位随机>`；随机段足够避免同一秒内的碰撞。"""
    return f"j-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(3)}"


def job_path(job_id: str) -> Path:
    safe = "".join(ch for ch in str(job_id) if ch.isalnum() or ch in "-_")
    return JOB_DIR / f"{safe}.json"


def save(job: dict[str, Any]) -> None:
    """原子落盘（写临时文件再 replace，避免前端读到半个 JSON）。

    ⚠️ **Windows 必读**：`os.replace` 在**目标文件正被读**时会 `PermissionError`
    （WinError 5）——CPython 打开文件时不带 `FILE_SHARE_DELETE`，而我们自己的
    轮询（`load`）正好在读同一个 job 文件。首轮自测就踩到过：一次写失败 → job 永远
    停在 `queued`（比"报错"更糟，用户只会一直转圈）。故**重试 + 直接写兜底**。
    """
    with _lock:
        p = job_path(job["job_id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        job["updated_at"] = _now()
        blob = json.dumps(job, ensure_ascii=False, indent=2)
        tmp.write_text(blob, encoding="utf-8")
        last: Exception | None = None
        for attempt in range(12):                    # 读者只持有文件几微秒 → 重试即可过
            try:
                os.replace(tmp, p)
                return
            except PermissionError as e:             # noqa: PERF203
                last = e
                time.sleep(0.02 * (attempt + 1))
        try:                                         # 兜底：非原子直写（保证状态能推进）
            p.write_text(blob, encoding="utf-8")
        except Exception:  # noqa: BLE001
            raise last if last is not None else RuntimeError("写入 job 状态失败")
        finally:
            tmp.unlink(missing_ok=True)


def load(job_id: str) -> dict[str, Any] | None:
    p = job_path(job_id)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001  损坏的 job 文件按不存在处理
        return None


def update(job_id: str, **fields: Any) -> dict[str, Any] | None:
    """读-改-写（给 `/cancel` 这类外部调用用；worker 自己持内存 dict 时直接 save）。"""
    with _lock:
        job = load(job_id)
        if job is None:
            return None
        job.update(fields)
        save(job)
        return job


def all_jobs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not JOB_DIR.is_dir():
        return out
    for p in JOB_DIR.glob("j-*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def find_active(pdf: str) -> dict[str, Any] | None:
    """该论文当前是否有**未结束**的 job（问答门控用：解析中 → 请稍候）。"""
    if not pdf:
        return None
    for j in all_jobs():
        if j.get("pdf") == pdf and j.get("status") in ACTIVE_STATUSES:
            return j
    return None


def latest(pdf: str) -> dict[str, Any] | None:
    """该论文最近一次 job（前端刷新页面后据此恢复进度）。"""
    if not pdf:
        return None
    js = [j for j in all_jobs() if j.get("pdf") == pdf]
    if not js:
        return None
    return max(js, key=lambda j: str(j.get("created_at") or ""))


def cancel_event(job_id: str) -> threading.Event:
    """worker 持有的取消信号（进程内）；跨进程取消不支持（单机单进程场景）。"""
    with _lock:
        ev = _events.get(job_id)
        if ev is None:
            ev = threading.Event()
            _events[job_id] = ev
        return ev


def clear_event(job_id: str) -> None:
    with _lock:
        _events.pop(job_id, None)


def request_cancel(job_id: str) -> dict[str, Any] | None:
    """请求取消：置信号 + 落盘标记（worker 会在**阶段边界**响应）。"""
    job = load(job_id)
    if job is None:
        return None
    cancel_event(job_id).set()
    job["cancel_requested"] = True
    job["cancel_requested_at"] = _now()
    save(job)
    return job


def recover_interrupted() -> int:
    """启动时把卡在 `queued`/`running` 的 job 标成中断（进程重启导致）。

    幂等（用模块级 `_recovered` 保证一个进程只扫一次）；不自动重排，
    交给用户点"重试"或重新上传——因为重跑会跳过已有产物，代价很低。
    """
    global _recovered
    with _lock:
        if _recovered:
            return 0
        _recovered = True
        n = 0
        for j in all_jobs():
            if j.get("status") in ACTIVE_STATUSES:
                j["status"] = STATUS_FAILED
                j["stage"] = j.get("stage") or ""
                j["error"] = "服务重启导致任务中断（可点重试；已完成的阶段会自动复用）"
                j.setdefault("stages", {})["interrupted"] = {"status": "failed",
                                                            "seconds": 0.0}
                save(j)
                n += 1
        return n


def stage_label(stage: str) -> str:
    if stage in STAGE_LABEL:
        return STAGE_LABEL[stage]
    if stage.startswith("report:"):
        return STAGE_LABEL.get(stage, f"生成报告（{stage.split(':', 1)[1]}）…")
    return stage or "处理中…"


def new_job(pdf: str, *, force: bool = False) -> dict[str, Any]:
    from pathlib import Path as _P

    return {
        "job_id": new_job_id(),
        "pdf": pdf,
        "stem": _P(pdf).stem,
        "status": STATUS_QUEUED,
        "stage": "",
        "stages": {},              # {名: {"status": ok|running|failed|skipped, "seconds": float}}
        "created_at": _now(),
        "updated_at": _now(),
        "started_at": "",
        "finished_at": "",
        "elapsed": 0.0,            # 总耗时（秒）
        "error": "",
        "attempt": 1,
        "force": bool(force),
        "note": "",                 # 要展示给用户的话（如同名不同内容 → 已另存为…）
        "cancel_requested": False,
    }


__all__ = [
    "JOB_DIR", "ACTIVE_STATUSES",
    "STATUS_QUEUED", "STATUS_RUNNING", "STATUS_READY", "STATUS_FAILED", "STATUS_CANCELLED",
    "all_jobs", "cancel_event", "clear_event", "find_active", "job_path", "latest", "load",
    "new_job", "new_job_id", "recover_interrupted", "request_cancel", "save", "stage_label",
    "update",
]
