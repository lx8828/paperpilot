"""L0 不变量 · `retrieval/tmp` 脚本的**分类完整性**。

分类器本体在 `evals/checks/tmp_scripts_audit.py`。本文件**只断言不变量**，
不比对快照 —— 因为分类结果会随"新脚本 / 新报告"自然变化，
钉死数量就会"一加脚本就红"，闸门随即被"狼来了"淹没（比没有更糟）。

守两件事，都是**真会出事**的：

  ① **判定为 `keep` 的脚本必须已入库**
     否则干净 clone 又静默少脚本 —— 与 gold / r2dev 真值 / 评测报告 /
     `_check_export_sync.py` / `cli/eval/` **完全同一类病**（本次已抓到 5 例）。

  ② **已归档的脚本不能被"保留"的脚本 import**
     归档最容易造出的就是**断的 import**：把被引用的库搬走，消费方当场 ImportError。
     实测踩过（`src/` 里 5 处引用指向已搬走的脚本，见 `3ce3548`）。

  ③ 已归档的脚本不能有**外部下游**（src/cli/tests/web 引用它）
     —— 同一条道理，只是引用方在 tmp 之外。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "evals" / "checks" / "tmp_scripts_audit.py"


def _load():
    """按路径加载分类器（`evals/checks/` 不是包，故不能用 import 语句）。"""
    spec = importlib.util.spec_from_file_location("tmp_scripts_audit", AUDIT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tmp_scripts_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def d() -> dict[str, Any]:
    """分类结果（模块级只算一次；`analyze()` 是纯函数，不写文件）。"""
    return _load().analyze()


def test_every_kept_script_is_tracked(d: dict[str, Any]):
    """判定为 `keep` 的脚本，必须已在 git 里 —— 否则干净 clone 会丢掉它们。"""
    tracked = set(subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "retrieval/tmp"],
        capture_output=True, encoding="utf-8", errors="replace").stdout.splitlines())
    still_here = [n for n in d["names"] if d["decision"].get(n) == "keep"]
    missing = [n for n in still_here if f"retrieval/tmp/{n}.py" not in tracked]
    assert not missing, (
        "以下脚本被判定为「保留」，但**未入库**（clone 后会消失）：\n  "
        + "\n  ".join(missing)
        + "\n修：`git add` 它们；若确实该归档，请确认没有下游后再搬。")
    assert still_here, "一个保留脚本都没有 —— 目录结构可能变了，请更新分类器"


def test_no_kept_script_imports_an_archived_one(d: dict[str, Any]):
    """归档不能造成**断的 import**：被保留脚本引用的，不能是已归档的。"""
    bad: list[str] = []
    for target, srcs in d["imported_by"].items():
        if d["decision"].get(target) != "archive":
            continue
        for s in srcs:
            if d["decision"].get(s) == "keep":
                bad.append(f"{s}.py → {target}.py")
    assert not bad, (
        "以下**保留**脚本 import 了**已归档**的脚本（import 会断）：\n  "
        + "\n  ".join(bad)
        + "\n修：把该脚本从 `_archive/` 搬回 `retrieval/tmp/`，或改掉引用。")


def test_no_archived_script_has_external_downstream(d: dict[str, Any]):
    """已归档的脚本不能有 tmp 之外的下游（src/cli/tests/web 引用它）。"""
    bad = [f"{n}.py ← {', '.join(d['external_downstream'][n])}"
           for n in d["external_downstream"]
           if d["decision"].get(n) == "archive"]
    assert not bad, (
        "以下已归档脚本仍被 tmp 之外的代码引用：\n  "
        + "\n  ".join(bad)
        + "\n修：搬回 `retrieval/tmp/`，或更新引用路径。")


def test_decision_covers_every_tmp_script(d: dict[str, Any]):
    """每个 tmp 脚本都必须有明确归属（不许"没分类"= 谁也不管的中间态）。"""
    undecided = sorted(n for n in d["names"] if n not in d["decision"])
    assert not undecided, f"以下脚本没有保留/归档的判定：{undecided}"
