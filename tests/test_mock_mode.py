"""演示模式（`web/app.py --mock`）：**无 API Key / 无模型 / 无 MinerU** 也要跑通整条链路。

这是"招聘方一条命令看效果"的技术保障，所以必须被测试钉住——
否则哪天 mock 路径悄悄坏掉，README 上的承诺就成了谎言。

与 `test_e2e_mock.py` 的区别：那边用 `fake_llm`/`fake_embed` **fixture**（测试自带替身）；
这里用 `PAPERPILOT_MOCK_*` **env**（走**随代码发布**的那条演示路径）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperpilot import jobs
from paperpilot.tools import mock_llm

DEMO_PDF = Path(__file__).resolve().parents[1] / "demo" / "demo_paper.pdf"


@pytest.fixture
def demo_env(monkeypatch):
    """打开演示模式（等价于 `--mock`）。"""
    monkeypatch.setenv("PAPERPILOT_MOCK_LLM", "1")
    monkeypatch.setenv("PAPERPILOT_MOCK_EMBED", "1")
    monkeypatch.setenv("PAPERPILOT_MINERU", "0")


# ───────────────────────── LLM：无需 key ─────────────────────────


def test_configured_without_any_key(demo_env, monkeypatch):
    from paperpilot.tools import llm

    for k in ("PAPERPILOT_LLM_API_KEY", "PAPERPILOT_JUDGE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert llm.is_configured() is True          # 否则 Web 上传会直接 500
    assert llm.judge_configured() is True


def test_mock_responses_have_expected_shapes(demo_env):
    """每个阶段的固定响应都必须**形状正确**（否则链路会静默降级成空结果）。"""
    from paperpilot.agents.nodes import answer as A
    from paperpilot.agents.nodes import judge as J
    from paperpilot.prompts import analyzer as Pa
    from paperpilot.prompts import report as Pr
    from paperpilot.prompts import viewer as Pv

    claims = json.loads(mock_llm.response(Pa.SYSTEM_PROMPT, "We propose a dual-tower model."))
    assert isinstance(claims, list) and claims
    assert claims[0]["type"] in {"contribution", "method", "result", "limitation"}
    assert claims[0]["text"] and claims[0]["evidence_quote"]
    # 证据句是**从输入里逐字摘的**（这样"证据回核"是真跑通的，不是硬编码假命中）
    assert "dual-tower" in claims[0]["evidence_quote"]

    assert json.loads(mock_llm.response(Pv.DEDUPE_SYSTEM, "x"))[0]["best_id"]
    assert json.loads(mock_llm.response(Pv.LABEL_SYSTEM, "x"))[0]["label"]
    assert "overview" in json.loads(mock_llm.response(Pr.OVERVIEW_SYSTEM, "x"))
    assert "enough" in json.loads(mock_llm.response(J._SYS_L0, "x"))
    assert "[1]" in mock_llm.response(A.SYSTEM, "x")        # 答案带引用号 → cites 非空
    assert mock_llm.response("完全不认识的 prompt", "x") == "[]"   # 未知 → 安全空数组


def test_mock_llm_goes_through_real_client_entry(demo_env):
    """走真实入口（`llm.chat_json`），证明**不是**只在测试里打补丁。"""
    from paperpilot.tools import llm

    got = llm.chat_json("未知 system", "u")
    assert got == []


# ───────────────────────── 向量：无需模型 ─────────────────────────


def test_mock_embed_no_model_loaded(demo_env):
    from paperpilot.agents import embedder

    assert "sentence_transformers" not in sys.modules      # 真的没加载 bge-m3
    v1 = embedder.encode_texts(["dual tower retrieval", "contrastive learning"])
    v2 = embedder.encode_texts(["dual tower retrieval", "contrastive learning"])
    assert v1.shape == (2, embedder.EMBED_DIM)
    assert (v1 == v2).all()                                 # 确定性
    assert abs(float((v1[0] ** 2).sum()) - 1.0) < 1e-5      # 归一化
    # 词重叠 → 相似度更高（演示里"检索到相关块"看得出来）
    q = embedder.encode_query("dual tower retrieval")
    assert float(q @ v1[0]) > float(q @ v1[1])


def test_cli_mock_flag_sets_switches(webapp, monkeypatch):
    """README 的第一条启动命令（`--mock`）必须真的能打开三个开关。"""
    monkeypatch.delenv("PAPERPILOT_MINERU", raising=False)
    host, port = webapp.apply_cli_args(["--mock", "--port", "9001"])
    assert (host, port) == ("127.0.0.1", 9001)
    assert os.environ["PAPERPILOT_MOCK_LLM"] == "1"
    assert os.environ["PAPERPILOT_MOCK_EMBED"] == "1"
    assert os.environ["PAPERPILOT_MINERU"] == "0"


def test_cli_defaults_without_flags(webapp, monkeypatch):
    """不带 --mock → 默认端口 8000，且**不**打开演示开关。"""
    monkeypatch.delenv("PAPERPILOT_MOCK_LLM", raising=False)
    host, port = webapp.apply_cli_args([])
    assert (host, port) == ("127.0.0.1", 8000)
    assert "PAPERPILOT_MOCK_LLM" not in os.environ


def test_mock_vec_dir_is_isolated(tmp_assets, demo_env):
    """演示模式的向量缓存必须**单独一份**：假向量与真向量混用会静默污染检索。"""
    from paperpilot.agents.embedder import ChunkIndex

    idx = ChunkIndex("a.pdf")
    assert idx._vec_file.parent.name.endswith("__mock")
    assert idx._vec_file.parent != tmp_assets.views


# ───────────────────────── MinerU：可关、且不阻塞问答 ─────────────────────────


def test_mineru_disabled_returns_skipped(tmp_assets, tiny_pdf, demo_env):
    from paperpilot.ingest import run_mineru

    got = run_mineru(tiny_pdf)
    assert got["status"] == "skipped" and "PAPERPILOT_MINERU" in got["reason"]


def test_skipped_mineru_does_not_block_qa(tmp_assets, monkeypatch):
    """`skipped`（主动跳过）≠ `failed`（解析失败）：前者问答照常可用。"""
    from paperpilot import ingest, jobs as J

    stem_dir = tmp_assets.mineru / "a"
    stem_dir.mkdir(parents=True, exist_ok=True)
    (stem_dir / "ingest.json").write_text(json.dumps(
        {"mineru": {"status": "skipped", "reason": "PAPERPILOT_MINERU=0"}}), encoding="utf-8")
    assert ingest.qa_blocked_reason("a.pdf") == ""

    job = J.new_job("a.pdf")
    job["status"] = J.STATUS_RUNNING
    J.save(job)
    assert "后台解析" in ingest.qa_blocked_reason("a.pdf")      # 只有"进行中"才提示稍候


# ───────────────────────── 端到端：演示模式跑通"上传 → 报告 → 问答" ─────────────────────────


def test_demo_mode_end_to_end(webapp_tmp, tmp_assets, demo_env):
    """**评审视角**：一条命令启动 → 拖入 demo 论文 → 报告 + 带引用的问答。"""
    assert DEMO_PDF.exists(), "demo 论文应随仓库提供"
    raw = DEMO_PDF.read_bytes()
    client = TestClient(webapp_tmp.app)

    # ① 模式自述（前端横幅用它）
    meta = client.get("/api/meta").json()
    assert meta["mock_llm"] and meta["mock_embed"] and meta["mineru"] is False
    assert "演示模式" in meta["banner"]

    # ② 上传 → 后台 job
    r = client.post("/api/report", files={"file": ("demo_paper.pdf", raw, "application/pdf")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # ③ 轮询到 ready（无 GPU/无模型 → 应当**秒级**）
    import time
    t0 = time.time()
    final = {}
    while time.time() - t0 < 120:
        final = client.get(f"/api/job/{job_id}").json()
        if final["status"] in (jobs.STATUS_READY, jobs.STATUS_FAILED, jobs.STATUS_CANCELLED):
            break
        time.sleep(0.1)
    assert final["status"] == jobs.STATUS_READY, final
    assert final["stages"]["mineru"]["status"] == "skipped"
    assert final["stages"]["report"]["status"] == "ok"
    assert final["stages"]["index"]["status"] == "ok"
    assert final["elapsed"] < 60, "演示模式不该慢（无模型加载）"

    # ④ 报告有内容
    rep = client.get("/api/report/demo_paper.pdf").json()
    assert rep["stats"]["n_claims"] >= 1
    assert "演示内容" in rep["overview"]            # 明确是固定示例
    assert rep["core_points"], "至少要有 1 个核心要点（否则 L0 答不了）"

    # ⑤ 问答：答案是固定文案，但**引用锚点必须真实**（页码 + 原文片段）
    ans = client.post("/api/ask", json={"question": "这篇论文提出了什么方法？",
                                        "pdf": "demo_paper.pdf"}).json()
    assert "演示内容" in ans["answer"]
    assert ans["cites"], "答案里的 [1] 必须解析成真实 cites"
    assert ans["cites"][0]["page"] >= 1
    assert ans["cites"][0]["evidence"], "引用要带原文片段（溯源闭环）"
    assert "ingest_blocked" not in ans["route"]
