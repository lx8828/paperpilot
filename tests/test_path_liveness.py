"""L0 不变量 · 路径存活棘轮接进 CI（只在**新增**悬空引用时红）。

闸门本体在 `evals/checks/path_liveness.py` —— 本文件只做「把它接进默认测试」这一件事，
**不重写扫描逻辑**（判据分两处必然漂移，而"判分散在两处"正是当初那批假测试的成因）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_new_dead_path_references():
    """`evals/checks/path_liveness.py` 必须退出 0（= 无新增悬空引用）。

    为什么是"棘轮"而不是"清零"：全仓 78 条历史悬空引用里，绝大多数在**历史报告**中
    （`qa/**` 57 条）—— 它们记录的是**当时的真实状态**，改写等于篡改历史。
    若一上来就要求清零，闸门必然长期变红 → 被"狼来了"淹没 → **比没有更糟**。
    棘轮把历史债记为基线，只拦新增（且修好的会被 `--update` 移出，只进不退）。

    ⚠️ 不要为了让它变绿而 `--update` 掉真的悬空引用 —— 那等于关掉闸门。
    """
    r = subprocess.run([sys.executable, "evals/checks/path_liveness.py"],
                       cwd=ROOT, capture_output=True, encoding="utf-8",
                       errors="replace", timeout=300)
    assert r.returncode == 0, (
        "发现**新增**的悬空路径引用（基线之外）：\n"
        f"{r.stdout[-2000:]}\n"
        "修：改成真实路径；若确是「举例/占位」写法，改写成不匹配「目录/文件.扩展名」的形态。")
