"""浏览器级冒烟（playwright）：**真页面 JS + 真 HTTP + 真 job 状态机**，跑在演示模式。

为什么需要它：其余测试都从 `TestClient` 或函数入口进，**前端 `index.html` 一行都没被执行过** ——
上一轮给 `ask()` 加 `AbortController` 时，就只有 `node --check` 语法门禁、没有行为验证。

走的是**真实用户路径**（页面上真实交互，不调内部函数）：
    打开页面 → file input 上传 → 等报告渲染 → 输入问题 → 点发送 → 答案带 [n] 引用 → 点引用跳原文
另外**捕获未捕获的 JS 异常**（`pageerror`）：前端脚本一旦抛错，这条会红。

跑法（**默认跳过**：CI 不跑，别人 clone 也不会因缺浏览器变红）：
    uv run pytest -m ui
浏览器来源按顺序尝试，**不强制下载 130MB chromium**：
    本机 Edge（channel=msedge）→ 本机 Chrome → playwright 自带 chromium → 都没有则 skip
不需要 key / 模型 / GPU：`live_server` 用演示模式（mock LLM + mock 向量 + 跳过 MinerU）。
"""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui

pytest.importorskip("playwright.sync_api", reason="需要 playwright：uv add --dev playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO_PDF = ROOT / "demo" / "demo_paper.pdf"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live_server(webapp_tmp, monkeypatch):
    """同进程起 uvicorn（后台线程）+ 演示模式 → 浏览器走**真实 HTTP**。

    为什么同进程：`webapp_tmp`/`tmp_assets` 的路径补丁只在当前解释器生效；换子进程就得先把
    assets 拷过去，反而更脆。摄取 worker 线程（后台 job）也在本进程内，照常工作。
    """
    import uvicorn

    monkeypatch.setenv("PAPERPILOT_MOCK_LLM", "1")
    monkeypatch.setenv("PAPERPILOT_MOCK_EMBED", "1")
    monkeypatch.setenv("PAPERPILOT_MINERU", "0")

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(webapp_tmp.app, host="127.0.0.1", port=port,
                                          log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True, name="ui-smoke-uvicorn")
    th.start()
    t0 = time.time()
    while not server.started and time.time() - t0 < 20:
        time.sleep(0.1)
    assert server.started, "uvicorn 未在 20s 内启动"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        th.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    """优先用**本机 Edge/Chrome**（免下载），都没有再退回 playwright 自带的 chromium。"""
    with sync_playwright() as p:
        for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
            try:
                b = p.chromium.launch(headless=True, **kw)
            except Exception:  # noqa: BLE001  该来源不可用 → 试下一个
                continue
            try:
                yield b
            finally:
                b.close()
            return
        pytest.skip("没有可用浏览器：装 Edge/Chrome，或 uv run playwright install chromium")


def test_upload_report_ask_and_jump(live_server, browser):
    """上传 → 报告 → 提问 → 引用可点（覆盖前端全部关键交互）。"""
    assert DEMO_PDF.exists(), "演示论文应随仓库提供"
    page = browser.new_page()
    js_errors: list[str] = []
    page.on("pageerror", lambda e: js_errors.append(str(e)))
    try:
        page.goto(live_server + "/", wait_until="domcontentloaded")
        assert page.locator("#qin").count() == 1, "页面没渲染出来"

        # ① 上传：真实文件 → file input（/api/report → 后台 job → 轮询 → 渲染报告）
        page.set_input_files("#file", str(DEMO_PDF))
        page.wait_for_selector("text=报告已生成", timeout=120_000)
        assert "demo_paper.pdf" in page.inner_text("#docname")
        assert page.locator("#rpv").inner_text().strip(), "报告面板是空的"

        # ② 提问：填输入框 + 点发送（页面上真实交互，不调内部函数）
        page.fill("#qin", "这篇论文提出了什么方法？")
        page.click("#send")
        page.wait_for_selector("#msgs .jump", timeout=30_000)   # 答案里的 [1] 渲染成可点链接
        msgs = page.inner_text("#msgs")
        assert "演示内容" in msgs, "答案没渲染（演示模式固定文案带「演示内容」）"
        assert "[1]" in msgs
        assert page.locator(".cites").count() >= 1, "引用出处块没渲染"

        # ③ 点引用 → 切到 PDF 视图（pdf.js 从 /vendor 加载，已入库）
        page.click("#msgs .jump")
        page.wait_for_selector('.tb[data-view="pdf"].on', timeout=10_000)
        page.wait_for_selector("#pdfv canvas", timeout=30_000)   # pdf.js 真把页面画出来了
    finally:
        page.close()

    assert not js_errors, "前端有未捕获的 JS 异常：" + " | ".join(js_errors[:3])
