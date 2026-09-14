"""API 集成测试（FastAPI TestClient）：上传落点 / 202 早返回 / 任务查询 / 取消重试 / 取报告。

全部离线：`worker.submit` 被替成"只落一个 job 文件"，所以这里**不跑** MinerU/LLM
（真实链路见 `test_e2e_mock.py`）。测的是**契约**：状态码、字段、幂等、404/400、拒答话术。
"""
from __future__ import annotations

import hashlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from paperpilot import jobs, worker

PDF_A = b"%PDF-1.4 content-A" + b"a" * 200
PDF_B = b"%PDF-1.4 content-B" + b"b" * 200


@pytest.fixture
def stub_worker(monkeypatch):
    """把"提交/重试"换成只落 job 文件（不启动真实摄取）。"""
    def _submit(pdf: str, *, force: bool = False, note: str = "") -> dict:
        job = jobs.new_job(pdf, force=force)
        job["note"] = note
        jobs.save(job)
        return job

    def _retry(job_id: str) -> dict | None:
        old = jobs.load(job_id)
        if old is None:
            return None
        job = jobs.new_job(str(old.get("pdf") or ""))
        jobs.save(job)
        return job

    monkeypatch.setattr(worker, "submit", _submit)
    monkeypatch.setattr(worker, "retry", _retry)


@pytest.fixture
def client(webapp_tmp, stub_worker, monkeypatch):
    """TestClient + "LLM 已配置"前提（上传接口会先校验配置）。

    只给**假**配置：`worker.submit` 已被替掉，所以不会有任何真实 LLM 调用。
    """
    monkeypatch.setenv("PAPERPILOT_LLM_BASE_URL", "https://fake.local/v1")
    monkeypatch.setenv("PAPERPILOT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("PAPERPILOT_LLM_MODEL", "fake-model")
    return TestClient(webapp_tmp.app)


def _upload(client, name: str, data: bytes):
    return client.post("/api/report", files={"file": (name, data, "application/pdf")})


# ───────────────────────── 上传落点与早返回 ─────────────────────────


def test_new_upload_returns_202_job(client):
    r = _upload(client, "x.pdf", PDF_A)
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"] and body["pdf"] == "x.pdf"
    assert body["status"] == jobs.STATUS_QUEUED
    assert body["ready"] is False


def test_same_content_is_idempotent_fast_path(client, tmp_assets):
    """同名同内容 + 产物已在 → **不排 job**，直接回报告（幂等）。"""
    (tmp_assets.papers / "x.pdf").write_bytes(PDF_A)
    (tmp_assets.views / "x.report.json").write_text(
        json.dumps({"pdf": "x.pdf", "title": "T", "stats": {}}), encoding="utf-8")
    r = _upload(client, "x.pdf", PDF_A)
    assert r.status_code == 200
    assert "job_id" not in r.json()
    assert "复用已有产物" in r.json()["upload_note"]


def test_same_name_diff_content_saves_as_new_file(client, tmp_assets):
    """同名不同内容 → 另存 `<stem>__<sha8>.pdf` 当新论文；原文件一个字节不动。"""
    _upload(client, "x.pdf", PDF_A)
    r = _upload(client, "x.pdf", PDF_B)
    assert r.status_code == 202
    sha8 = hashlib.sha256(PDF_B).hexdigest()[:8]
    assert r.json()["pdf"] == f"x__{sha8}.pdf"
    assert "另存为" in r.json()["note"]
    assert (tmp_assets.papers / "x.pdf").read_bytes() == PDF_A      # 未被覆盖


@pytest.mark.parametrize("name, data, code", [
    ("x.txt", PDF_A, 400),
    ("x.pdf", b"", 400),
], ids=["非 pdf", "空文件"])
def test_upload_rejects_bad_input(client, name, data, code):
    assert _upload(client, name, data).status_code == code


# ───────────── 上传门加固：`%PDF-` 魔数 + 大小上限（2026-09-14 审查）─────────────
#
# 只做这两条，理由（也是"为什么不做页数/解压炸弹"）：
# · 这两条是**边界上用 3 行省掉一次注定失败的分钟级摄取**（改名 txt → 现在会白跑一个 job，
#   报错还是英文 `FileDataError`；超大文件 → 读进内存 + 无 body 上限 + MinerU 白跑到 900s）；
# · 页数要**先解析一遍**才知道（为了决定"要不要处理贵的东西"先做那个贵的事），且 3000 页的
#   合集往往是合法需求 → 不做；
# · 解压炸弹/恶意 PDF 在"本地单用户 127.0.0.1"下没有攻击者模型（真缓解是子进程沙箱）→ 不做。
#   若哪天变多用户服务，这一套要整体重做，不是这几行能顶的。


def test_upload_rejects_non_pdf_bytes(client, monkeypatch):
    """改名成 .pdf 的 txt/docx → **立刻 400，且不排 job**。

    修前：`202` 受理 → 白跑一个 job → 用户拿到英文
    `FileDataError: Failed to open file '...' as type pdf.`（实测）。
    """
    seen = {"n": 0}
    real = worker.submit

    def _spy(*a, **k):
        seen["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(worker, "submit", _spy)
    r = _upload(client, "x.pdf", b"this is definitely not a pdf" + b"x" * 200)
    assert r.status_code == 400
    assert "%PDF-" in r.json()["detail"]        # 报错要说清"缺什么"
    assert seen["n"] == 0                       # 关键：没有排 job


def test_upload_accepts_leading_junk_before_header(client):
    """`%PDF-` 前有前导垃圾 → **必须放行**（规范允许，邮箱导出/扫描件常见）。

    这条比"拒绝坏文件"更重要：`raw[:5] == b"%PDF-"` 那种写法会**误杀合法 PDF**。
    """
    r = _upload(client, "junk.pdf", b"\xef\xbb\xbfGARBAGE" + b"\x00" * 40 + PDF_A)
    assert r.status_code == 202


def test_upload_rejects_too_large(client, monkeypatch):
    """超上限 → **413**（默认 200MB，`PAPERPILOT_MAX_PDF_MB` 可调）。

    修前：整个文件 `await file.read()` 进内存 + sha256 + 落盘（峰值 ≈ 2×），
    而 FastAPI/uvicorn **没有任何 body 上限** → 误拖大文件还会让 MinerU 白跑到超时。
    """
    monkeypatch.setenv("PAPERPILOT_MAX_PDF_MB", "1")
    big = PDF_A + b"z" * (2 * 1024 * 1024)      # ~2MB > 1MB 上限
    r = _upload(client, "big.pdf", big)
    assert r.status_code == 413
    assert "过大" in r.json()["detail"]


def test_upload_size_limit_can_be_disabled(client, monkeypatch):
    """`PAPERPILOT_MAX_PDF_MB=0` → 不限制（开关真能关，别把大文件用户堵死）。"""
    monkeypatch.setenv("PAPERPILOT_MAX_PDF_MB", "0")
    big = PDF_A + b"z" * (2 * 1024 * 1024)
    assert _upload(client, "big2.pdf", big).status_code == 202


# ───────────── 上限配置必须 fail-safe（2026-09-14 审查）─────────────
#
# 旧实现 `max(0, mb)` 把 `-1` 悄悄变成 0 = **关闭限制** —— 一个手滑的负号就**扩大了权限**。
# 安全配置的原则是"异常输入回到安全默认值"，而不是 fail-open。
# 现在：**只有显式 `0`** 才是不限制；负数/非数字/空串 → 回退默认 200MB。


@pytest.mark.parametrize("raw, want_mb", [
    ("1", 1),
    (" 1 ", 1),          # shell 里带空格的常见写法
    ("0", 0),            # 唯一的"明确不限制"
    ("-1", 200),         # ← bug 回归：旧实现这里是 0（= 不限制）
    ("-100", 200),
    ("abc", 200),
    ("", 200),
    ("2.5", 200),        # 小数不接受（MB 用整数，避免歧义）→ 回退默认
], ids=["1MB", "带空格", "0=不限制", "负数", "大负数", "非数字", "空串", "小数"])
def test_max_pdf_bytes_is_fail_safe(webapp, monkeypatch, raw, want_mb):
    """配置解析语义（单元级，逐值钉住）。"""
    monkeypatch.setenv("PAPERPILOT_MAX_PDF_MB", raw)
    assert webapp._max_pdf_bytes() == want_mb * 1024 * 1024


def test_max_pdf_bytes_default_when_unset(webapp, monkeypatch):
    monkeypatch.delenv("PAPERPILOT_MAX_PDF_MB", raising=False)
    assert webapp._max_pdf_bytes() == 200 * 1024 * 1024


def test_negative_limit_still_enforced(client, webapp, monkeypatch):
    """**bug 回归（端到端）**：`-1` 必须仍然**拦得住**，而不是变成不限制。

    把"默认值"临时降成 1MB，这样一次 2MB 上传就能区分两种行为：
    旧实现（-1 → 0）→ 202 放行；修复后（-1 → 回退默认）→ 413。
    """
    monkeypatch.setattr(webapp, "_DEFAULT_MAX_PDF_MB", 1)
    monkeypatch.setenv("PAPERPILOT_MAX_PDF_MB", "-1")
    big = PDF_A + b"z" * (2 * 1024 * 1024)
    r = _upload(client, "neg.pdf", big)
    assert r.status_code == 413
    assert "1MB" in r.json()["detail"]          # 用的是**回退后的默认值**，不是"不限制"


# ───────────────────────── 任务查询 / 取消 / 重试 ─────────────────────────


def test_job_status_shape_and_404(client):
    job_id = _upload(client, "x.pdf", PDF_A).json()["job_id"]
    r = client.get(f"/api/job/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert {"job_id", "status", "stage", "stage_label", "stages", "elapsed"} <= set(body)
    assert body["stage_label"]                                   # 永远有可显示的文案
    assert client.get("/api/job/j-19700101-ffffff").status_code == 404


def test_jobs_latest(client):
    _upload(client, "x.pdf", PDF_A)
    r = client.get("/api/jobs/latest", params={"pdf": "x.pdf"})
    assert r.status_code == 200 and r.json()["pdf"] == "x.pdf"
    assert client.get("/api/jobs/latest", params={"pdf": "nope.pdf"}).status_code == 404


def test_cancel_endpoint(client):
    job_id = _upload(client, "x.pdf", PDF_A).json()["job_id"]
    r = client.post(f"/api/job/{job_id}/cancel")
    assert r.status_code == 200
    assert jobs.load(job_id)["cancel_requested"] is True
    assert client.post("/api/job/j-19700101-ffffff/cancel").status_code == 404


def test_retry_endpoint_returns_new_job(client):
    job_id = _upload(client, "x.pdf", PDF_A).json()["job_id"]
    jobs.update(job_id, status=jobs.STATUS_FAILED)               # 先结束它才能重试
    r = client.post(f"/api/job/{job_id}/retry")
    assert r.status_code == 202
    assert r.json()["job_id"] != job_id


# ───────────────────────── 取报告 ─────────────────────────


def test_get_report_404_then_200(client, tmp_assets):
    assert client.get("/api/report/x.pdf").status_code == 404
    (tmp_assets.views / "x.report.json").write_text(
        json.dumps({"pdf": "x.pdf", "title": "T", "stats": {}}), encoding="utf-8")
    r = client.get("/api/report/x.pdf")
    assert r.status_code == 200 and r.json()["title"] == "T"


def test_job_view_is_a_snapshot(webapp):
    """`_job_view` 必须返回**快照**。

    202 那条路径把它交给 JSON 序列化的同时，worker 线程还在往同一个 dict 里加
    `stages` 键 → 直接传引用会偶发 `RuntimeError: dictionary changed size during iteration`。
    （读磁盘的 `/api/job/{id}` 没这个问题，只有 202 这条共享内存。）
    """
    live = jobs.new_job("a.pdf")
    live["stages"]["mineru"] = {"status": "running", "seconds": 0.0}
    view = webapp._job_view(live)

    live["stages"]["report"] = {"status": "ok", "seconds": 1.0}   # 模拟 worker 继续写
    live["stage"] = "index"

    assert "report" not in view["stages"]
    assert view["stages"]["mineru"]["status"] == "running"
    assert view["stage"] != "index"


def test_get_report_includes_mineru_warning(client, tmp_assets):
    """MinerU 失败必须**明确告知**（报告可用、问答不可用），不能静默。"""
    (tmp_assets.views / "x.report.json").write_text(
        json.dumps({"pdf": "x.pdf", "stats": {}}), encoding="utf-8")
    stem_dir = tmp_assets.mineru / "x"
    stem_dir.mkdir(parents=True, exist_ok=True)
    (stem_dir / "ingest.json").write_text(json.dumps(
        {"mineru": {"status": "failed", "reason": "无 GPU"}}), encoding="utf-8")
    r = client.get("/api/report/x.pdf")
    assert r.status_code == 200 and "MinerU" in r.json()["mineru_warning"]


# ───────────────────────── 问答门控 ─────────────────────────


def test_ask_blocked_while_job_running(client, fake_llm):
    """后台解析进行中 → 明确告知"请稍候"，而不是失败（也不是硬编码答案）。"""
    job = jobs.new_job("x.pdf")
    job["status"] = jobs.STATUS_RUNNING
    job["stage"] = "mineru"
    jobs.save(job)
    r = client.post("/api/ask", json={"question": "方法是什么？", "pdf": "x.pdf"})
    assert r.status_code == 200
    body = r.json()
    assert "后台解析" in body["answer"]
    assert "ingest_blocked" in body["route"]


def test_ask_missing_paper_404(client, fake_llm):
    assert client.post("/api/ask", json={"question": "q", "pdf": "nope.pdf"}).status_code == 404


def test_ask_requires_llm_config(client, monkeypatch):
    """没配 key 时不能装作能答（明确 500 提示去配置）。"""
    from paperpilot.tools import llm

    monkeypatch.delenv("PAPERPILOT_LLM_API_KEY", raising=False)
    assert llm.is_configured() is False
    r = client.post("/api/ask", json={"question": "q", "pdf": "x.pdf"})
    assert r.status_code == 500 and "LLM 未配置" in r.json()["detail"]


def test_meta_exposes_ask_budget(client, monkeypatch):
    """**前端的兜底超时靠这个契约**：`/api/meta` 必须给出服务端的排队预算与并发。

    前端用它算 `AbortController` 预算（= `ask_wait_s + 180s`）→ 超时值**跟随服务端配置**，
    不会写死一个会漂移的魔数；字段被删掉前端就失去兜底（且不会报错，只会退回默认 300s）。
    """
    monkeypatch.setenv("PAPERPILOT_ASK_WAIT_S", "120")
    monkeypatch.setenv("PAPERPILOT_ASK_CONCURRENCY", "6")
    m = client.get("/api/meta").json()
    assert m["ask_wait_s"] == 120.0
    assert m["ask_concurrency"] == 6
