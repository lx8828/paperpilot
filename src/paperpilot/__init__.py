"""PaperPilot 包入口。

**这里做一件不起眼但很要紧的事：把标准流切到 UTF-8。**

为什么：Windows 控制台/管道的默认编码是 GBK，而生产代码里有大量中文与符号输出
（`print("✔ report.json 已保存…")`）。在 GBK 流上写 `✔`/emoji 会抛
`UnicodeEncodeError` —— 表现为"跑测试时偶发崩溃"或"CLI 静默报错"，
而且**看起来与业务逻辑毫无关系**，极难定位（本仓库 2026-09-13 的测试偶发失败就是它）。

放在包入口（而不是每个脚本里）的原因：`paperpilot` 一定先于业务模块被导入，
且 pytest 的采集阶段也会走到这里 —— 于是**捕获流的编码**也跟着变成 UTF-8。
失败时静默忽略（例如被 pytest / 重定向接管、或对象不支持 `reconfigure`）。
"""
from __future__ import annotations

import sys


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001  不支持 reconfigure 的流（pytest 捕获等）→ 忽略
            pass


_force_utf8_streams()


def main() -> None:
    """`uv run paperpilot`：给出最短上手路径（面向第一次打开项目的人）。"""
    print(
        "PaperPilot —— 论文帮读器（结构化精读报告 + 带引用溯源的问答）\n"
        "\n"
        "最短上手（详见 README.md）：\n"
        "  1) 装依赖：        uv sync\n"
        "  2) 免 Key 看效果： uv run python web/app.py --mock\n"
        "                     → 打开 http://127.0.0.1:8000 ，拖入 demo/demo_paper.pdf\n"
        "  3) 真跑（填 key）：cp .env.example .env && uv run python web/app.py\n"
        "  4) 跑离线测试：    uv run pytest -q\n"
    )
