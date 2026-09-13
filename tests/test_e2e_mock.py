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
    ask = e2e.post("/api/ask", json={"question": "这篇论文提出了什么方法？", "pdf": tiny_pdf})
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
    """取消走**真实** worker：阶段边界生效，终态是 cancelled（不是 failed）。"""
    from paperpilot import worker

    # 让报告链在阶段边界停一下，保证"取消"来得及插入（真实场景是分钟级）
    real_report = worker._lane_report

    def _slow_report(job, ev):
        time.sleep(0.3)
        return real_report(job, ev)

    monkeypatch.setattr(worker, "_lane_report", _slow_report)
    raw = (tmp_assets.papers / tiny_pdf).read_bytes()
    job_id = e2e.post("/api/report", files={"file": (tiny_pdf, raw, "application/pdf")}
                      ).json()["job_id"]
    e2e.post(f"/api/job/{job_id}/cancel")
    final = _wait_ready(e2e, job_id)
    assert final["status"] == jobs.STATUS_CANCELLED
    assert "取消" in final["error"]
