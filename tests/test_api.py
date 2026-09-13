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
