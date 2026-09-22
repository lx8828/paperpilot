"""端到端（**mock LLM**）：上传 → 后台 job → 报告 → 提问 → 带引用的答案。

这正是审查意见要的那个测试：**不依赖真实 API Key、不依赖 GPU/本地模型**，
但从 HTTP 入口到最终答案走的是**真实代码路径**：

    TestClient → /api/report → job 状态机 → worker（MinerU lane ∥ 报告链 lane）
              → 报告链（claims → 去重/打标/打分 → 骨架 → 图表 → 概述/导读 → 装配）
              → 向量索引 → /api/ask → v3 两级图 → 闸门 → 带 [n] 引用的答案

被替掉的只有三样外部依赖：LLM（`fake_llm`）、embedding（`fake_embed`）、
MinerU 产物（`fake_mineru` + `fake_mineru_env`）。
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from paperpilot import jobs


def _wait_ready(client, job_id: str, limit: float = 120.0) -> dict:
    t0 = time.time()
    last: dict = {}
    while time.time() - t0 < limit:
        r = client.get(f"/api/job/{job_id}")
        assert r.status_code == 200
        last = r.json()
        if last["status"] in (jobs.STATUS_READY, jobs.STATUS_FAILED, jobs.STATUS_CANCELLED):
            return last
        time.sleep(0.1)
    raise AssertionError(f"job 超时未结束：{last}")


@pytest.fixture
def e2e(webapp_tmp, tmp_assets, tiny_pdf, fake_mineru, fake_mineru_env,
        fake_llm, fake_embed):
    """把整条链路搭起来（假 LLM / 假向量 / 假 MinerU + 真 PDF）。"""
    fake_mineru(tiny_pdf)
    return TestClient(webapp_tmp.app)


def test_upload_to_report_to_answer(e2e, tmp_assets, tiny_pdf):
    raw = (tmp_assets.papers / tiny_pdf).read_bytes()

    # ① 上传 → 立即返回 job（不阻塞）
    r = e2e.post("/api/report", files={"file": (tiny_pdf, raw, "application/pdf")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # ② 轮询到 ready，且各阶段都成功、有耗时
    final = _wait_ready(e2e, job_id)
    assert final["status"] == jobs.STATUS_READY, final
    stages = final["stages"]
    assert stages["mineru"]["status"] == "ok"        # 假产物 → 复用分支
    assert stages["report"]["status"] == "ok"
    assert stages["index"]["status"] == "ok"
    assert final["elapsed"] >= 0

    # ③ 报告产物真的落盘了，且**有内容**（claims 来自假 LLM 的固定回答）
    report_file = tmp_assets.views / f"{tmp_assets.papers.joinpath(tiny_pdf).stem}.report.json"
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["stats"]["n_claims"] >= 1
    assert report["overview"]

    # ④ 取报告接口能拿到（带 job 耗时）
    got = e2e.get(f"/api/report/{tiny_pdf}")
    assert got.status_code == 200
    assert got.json()["stats"]["n_claims"] >= 1
    assert got.json()["job"]["elapsed"] >= 0

    # ⑤ 提问 → 答案带 [n] 引用（cites 非空），且经过闸门
    ask = e2e.post("/api/ask", json={"question": "这篇论文提出了什么方法？", "pdfs": [tiny_pdf]})
    assert ask.status_code == 200, ask.text
    ans = ask.json()
    assert ans["answer"]
    assert "ingest_blocked" not in ans["route"]
    assert ans["cites"], "答案带 [1] 引用 → cites 不该为空"
    assert ans["cites"][0]["page"] >= 1
    assert ans["validator"]["action"] in {"pass", "repaired", "fallback"}


def test_job_reuses_cached_products_second_time(e2e, tmp_assets, tiny_pdf):
    """第二次上传同一篇：**幂等秒回报告**（不排 job）—— 产物缓存生效。"""
    raw = (tmp_assets.papers / tiny_pdf).read_bytes()
    first = e2e.post("/api/report", files={"file": (tiny_pdf, raw, "application/pdf")})
    assert first.status_code == 202
    _wait_ready(e2e, first.json()["job_id"])

    again = e2e.post("/api/report", files={"file": (tiny_pdf, raw, "application/pdf")})
    assert again.status_code == 200                    # 直接回报告，不是 202
    assert "job_id" not in again.json()
    assert again.json()["stats"]["n_claims"] >= 1


def test_cancel_running_job_is_honoured(e2e, tmp_assets, tiny_pdf, monkeypatch):
    """取消走**真实** worker 状态机：阶段边界生效，终态是 cancelled。

    刻意**不依赖时序**：报告 lane 被换成"一直等到取消信号才抛 `JobCancelled`"，
    所以不可能出现"任务跑太快、取消来不及生效"的假失败（首版就是这么挂的）。
    """
    from paperpilot import worker

    def _wait_for_cancel(job, ev):
        worker._mark(job, "report", status="running")
        deadline = time.time() + 15
        while time.time() < deadline:
            if ev.is_set():
                raise worker.JobCancelled()
            time.sleep(0.02)
        raise AssertionError("测试超时：取消信号一直没到")

    monkeypatch.setattr(worker, "_lane_report", _wait_for_cancel)
    raw = (tmp_assets.papers / tiny_pdf).read_bytes()
    job_id = e2e.post("/api/report", files={"file": (tiny_pdf, raw, "application/pdf")}
                      ).json()["job_id"]

    # 等它真的进到 report（running）再点取消；然后等终态
    t0 = time.time()
    while time.time() - t0 < 15:
        if e2e.get(f"/api/job/{job_id}").json()["status"] == "running":
            break
        time.sleep(0.05)
    assert e2e.post(f"/api/job/{job_id}/cancel").status_code == 200
    final = _wait_ready(e2e, job_id)
    assert final["status"] == jobs.STATUS_CANCELLED
    assert "取消" in final["error"]


def test_mineru_failed_end_to_end(webapp_tmp, tmp_assets, tiny_pdf,
                                  fake_llm, fake_embed, monkeypatch):
    """**降级与容灾（端到端，2026-09-22 契约变更）**：MinerU 失败 → 报告可用、
    **索引照建**、问答**降级到 pymupdf 备用路仍然可用**，且降级**显性**。

    走的是**真实**代码路径（只把"是否可用"换成 False）：
    真 `_lane_mineru` → 真 `run_mineru`（failed 是**正常返回**，不抛异常）
    → 真 meta 落盘 → 真 `_lane_index` → 真 `qa_blocked_reason()` → 真 graph。

    旧契约是"MinerU 失败 → 不建索引 + 问答被闸门拒绝"。问题：报告链**早就在降级**
    （走 pymupdf），只有问答被拦 → "默认路失败降级到备用路"**永远走不到**。
    """
    from paperpilot import ingest, worker

    monkeypatch.setattr(ingest, "mineru_available", lambda: (False, "无 GPU：CUDA 不可用"))
    seen = {"index": 0}
    real_index = worker._lane_index

    def _spy_index(job, ev):                 # 只观察，不改变行为
        seen["index"] += 1
        return real_index(job, ev)

    monkeypatch.setattr(worker, "_lane_index", _spy_index)

    client = TestClient(webapp_tmp.app)
    raw = (tmp_assets.papers / tiny_pdf).read_bytes()
    r = client.post("/api/report", files={"file": (tiny_pdf, raw, "application/pdf")})
    assert r.status_code == 202
    final = _wait_ready(client, r.json()["job_id"])

    # ① 状态如实：MinerU failed，但**索引照建**（降级后问答要用）
    assert final["status"] == jobs.STATUS_READY, final
    assert final["stages"]["mineru"]["status"] == "failed"
    assert seen["index"] == 1
    assert final["stages"]["index"]["status"] == "ok"

    # ② 报告可读，且带**用户可见**的警告（不静默）
    got = client.get(f"/api/report/{tiny_pdf}")
    assert got.status_code == 200
    assert "mineru_warning" in got.json()

    # ③ **降级而非拒绝**：问答照常可用（代价是表值可能缺）
    ask = client.post("/api/ask", json={"question": "这篇论文提出了什么？", "pdfs": [tiny_pdf]})
    assert ask.status_code == 200
    body = ask.json()
    assert "ingest_blocked" not in (body.get("route") or []), body
    assert body.get("answer"), body
    assert (body.get("validator") or {}).get("action") != "blocked"

    # ④ 降级**必须显性**：判为 `degraded`（不是 `pending`"还没跑"）且原因可查
    from paperpilot.agents.document_cache import mineru_status
    st, why = mineru_status(tiny_pdf)
    assert st == "degraded", f"MinerU 跑过但失败 → 应判 degraded，实际 {st!r}"
    assert why
