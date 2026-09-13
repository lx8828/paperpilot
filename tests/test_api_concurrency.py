"""`/api/ask` **不能阻塞事件循环** —— 回归测试。

背景（外部审查）：原来是 `async def ask_question` 里直接执行同步的 `graph_ask()`，
单题问答十几秒内**整个事件循环被占住**，其他请求（哪怕只是取个状态）全部排队。

修法：端点改成同步 `def`（FastAPI 丢**线程池**执行）+ `_ASK_GATE` 收口并发。
本文件把这两条契约钉住——以后谁把它改回 `async def`，测试立刻红。
"""
from __future__ import annotations

import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient


def _prepare(webapp_tmp, monkeypatch, fake_ask):
    monkeypatch.setattr(webapp_tmp, "graph_ask", fake_ask)
    monkeypatch.setenv("PAPERPILOT_LLM_BASE_URL", "https://fake.local/v1")
    monkeypatch.setenv("PAPERPILOT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("PAPERPILOT_LLM_MODEL", "fake")
    return TestClient(webapp_tmp.app)


def test_ask_endpoint_is_sync(webapp):
    """契约①：必须是**同步**端点（FastAPI 才会把它丢到线程池）。"""
    assert inspect.iscoroutinefunction(webapp.ask_question) is False


def test_ask_does_not_block_event_loop(webapp_tmp, monkeypatch):
    """契约②：问答**进行中**，其他请求仍要立刻返回。

    写法刻意不依赖 sleep 猜测：用 Event 等"问答真的进来了"再发第二个请求，
    并断言第二个请求在问答**结束之前**就返回了（证明两者重叠，而非串行跑完）。
    """
    entered = threading.Event()

    def _slow_ask(question, pdf, history=None):
        entered.set()
        time.sleep(1.2)                       # 十几秒问答的缩短版
        return {"answer": "慢答案 [1]", "cites": [{"n": 1, "page": 1}],
                "route": ["L0"], "validator": {"action": "pass", "issues": []}}

    client = _prepare(webapp_tmp, monkeypatch, _slow_ask)

    with ThreadPoolExecutor(max_workers=2) as ex:
        t_ask0 = time.time()
        ask_fut = ex.submit(client.post, "/api/ask",
                            json={"question": "方法是什么？", "pdf": "a.pdf"})
        assert entered.wait(timeout=5), "问答请求没进来"
        t_meta0 = time.time()
        meta = client.get("/api/meta")
        meta_elapsed = time.time() - t_meta0
        ask = ask_fut.result(timeout=10)
        t_ask_end = time.time()

    assert meta.status_code == 200
    assert ask.status_code == 200 and "慢答案" in ask.json()["answer"]
    assert meta_elapsed < 0.6, f"问答进行中，其他请求被拖慢 {meta_elapsed:.2f}s"
    assert t_meta0 < t_ask_end, "meta 在 ask 之后才返回 → 没能证明并发"
    assert t_ask_end - t_ask0 >= 1.0, "ask 太快，没构成并发场景（测试自身失效）"


def test_ask_gate_serializes_when_set_to_one(webapp_tmp, monkeypatch):
    """契约③：`_ASK_GATE` 真的在限并发（=1 时串行）。"""
    monkeypatch.setattr(webapp_tmp, "_ASK_GATE", threading.Semaphore(1))
    peak = {"n": 0, "max": 0}
    lock = threading.Lock()

    def _ask(question, pdf, history=None):
        with lock:
            peak["n"] += 1
            peak["max"] = max(peak["max"], peak["n"])
        time.sleep(0.25)
        with lock:
            peak["n"] -= 1
        return {"answer": "ok [1]", "cites": [{"n": 1, "page": 1}],
                "route": ["L0"], "validator": {"action": "pass", "issues": []}}

    client = _prepare(webapp_tmp, monkeypatch, _ask)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(client.post, "/api/ask",
                          json={"question": f"q{i}", "pdf": "a.pdf"}) for i in range(3)]
        for f in futs:
            assert f.result(timeout=10).status_code == 200
    elapsed = time.time() - t0

    assert peak["max"] == 1, f"闸门没生效：并发峰值 {peak['max']}"
    assert elapsed >= 0.7, f"3 个 0.25s 的问答被串行执行，总耗时却只有 {elapsed:.2f}s"
