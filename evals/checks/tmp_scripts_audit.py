"""盘点 `retrieval/tmp/*.py`：谁是**库**、谁是**入口**、谁是一次性实验。

**只读**（`main()` 才写基线）。产出 `evals/baselines/tmp_scripts.json`。

## 为什么不能只看 `import`

这批脚本大量用 `importlib.util.spec_from_file_location("x", HERE/"tmp"/"_y.py")`
**动态加载** —— 纯 `ast` 静态分析会**漏掉它们**，而它们恰恰是"库"（被多处共用）。
所以两条路一起走：

  ① **静态**：`ast` 解析 `import X` / `from X import ...`，取本地模块名
  ② **动态**：正则找源码里的字符串常量 `"_xxx.py"` / `'_xxx.py'`

## 分类规则（按"删掉它会怎样"）

    library    被 **≥1 个其它文件**引用（含 retrieval/tmp 之外）→ 不能动
    entrypoint 有 `if __name__ == "__main__"` 或 `argparse` → 可跑
    scratch    两者都不是 → 一次性实验

## 判定（保留 / 归档）

**库 ∪ 有出处 → 保留**；其余（无出处的入口 + scratch）→ 归档。

为什么用「**有出处**」而不是「有 argparse」：报告会写"由 `_xxx.py` 产出"，
那是"真跑过、有结论"的记录；而 argparse 只说明"能跑"，一堆随手写的探针也有。

## 结构：`analyze()` 纯分析 / `main()` 才有副作用

**这样 `tests/test_tmp_scripts_audit.py` 能 import 它做不变量断言，而不会顺手
改掉基线文件**。副作用（写 JSON、打印）全在 `main()`。
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "retrieval" / "tmp"
ARCH = TMP / "_archive"
OUT = ROOT / "evals" / "baselines" / "tmp_scripts.json"

OUTSIDE_DIRS = ("src", "cli", "tests", "web", "evals")
STR_REF = re.compile(r"['\"]\s*([A-Za-z0-9_]+)\.py\s*['\"]")

# 剪枝：**生成物 / 数据集 / 归档**不是"文档出处"的来源，且体量极大。
# （与 `path_liveness.py` 的 PRUNE 同理；那边是"引用源"，这边是"文档来源"。）
PRUNE = (
    "assets/",                 # 18,850 个解析产物
    "retrieval/data/",         # 数据集（8GB）
    "qa/multi/_runs/",         # 跑批产出
    "retrieval/tmp/_archive/", "retrieval/tmp/_gold_history/",
    "evals/baselines/",        # 基线数据（不是文档）
)


def _is_pruned(rel_dir: str) -> bool:
    """相对目录路径（**带尾斜杠**）是否落在剪枝范围。"""
    return rel_dir.startswith(PRUNE)

# `PINNED`：明知「无出处」但**故意保留**的，必须显式钉住 —— 否则下次重跑又会被
# 划成 archive（演示脚本吃过这个亏：它是给人重放的，不该因为报告里没提就归档）。
PINNED: dict[str, str] = {"_salvage_demo": "keep"}


def _scan_file(path: Path, names: set[str]) -> tuple[set[str], bool, bool]:
    """返回 (它引用的本地模块名, 有 __main__, 用 argparse)。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    deps: set[str] = set()

    # ① 静态 import
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    m = a.name.split(".")[0]
                    if m in names:
                        deps.add(m)
            elif isinstance(node, ast.ImportFrom) and node.module:
                m = node.module.split(".")[0]
                if m in names:
                    deps.add(m)
    except SyntaxError:
        pass

    # ② 动态加载（字符串常量）
    for m in STR_REF.finditer(text):
        if m.group(1) in names:
            deps.add(m.group(1))

    return (deps - {path.stem},
            '__name__ == "__main__"' in text or "__name__ == '__main__'" in text,
            "argparse" in text)


def _external_downstream(names: set[str]) -> dict[str, set[str]]:
    """tmp **之外**（src/cli/tests/web/evals/retrieval/scripts）谁引用了这些脚本名。"""
    ext_files: list[Path] = []
    for d in OUTSIDE_DIRS:
        ext_files += [p for p in (ROOT / d).rglob("*.py") if p.is_file()]
    ext_files += [p for p in (ROOT / "retrieval" / "scripts").glob("*.py")]

    out: dict[str, set[str]] = defaultdict(set)
    for p in ext_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        rel = p.relative_to(ROOT).as_posix()
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        for m in STR_REF.finditer(text):
            found.add(m.group(1))
        for n in found & names:
            out[n].add(rel)
    return out


def _mentioned_in_docs(names: set[str]) -> dict[str, set[str]]:
    """脚本名是否被**文档/报告**提到（`_xxx.py` 字面量）—— "有出处"的最强证据。

    ⚠️ 用 `os.walk` + **剪枝**，不用 `ROOT.rglob("*.md")`：后者会走遍
    `assets/artifacts`（**18,850 个文件**）→ 慢到没人愿意跑（棘轮踩过同一个坑）。
    剪的是**生成物/数据**，不是"为了好看"。
    """
    out: dict[str, set[str]] = defaultdict(set)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = Path(dirpath).relative_to(ROOT).as_posix()
        prefix = "" if rel == "." else rel + "/"
        dirnames[:] = [d for d in dirnames
                       if d not in (".venv", ".git", "node_modules", ".pytest-tmp")
                       and not _is_pruned(prefix + d + "/")]
        for fn in filenames:
            if not fn.endswith(".md"):
                continue
            p = Path(dirpath) / fn
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for n in names:
                if f"{n}.py" in text:
                    out[n].add(p.relative_to(ROOT).as_posix())
    return out


def analyze() -> dict[str, Any]:
    """**纯分析**：返回分类与判定（不写文件、不打印）。"""
    tmp_files = sorted(p for p in TMP.glob("*.py") if p.is_file())
    names = {p.stem for p in tmp_files}
    archived_on_disk = sorted(p.stem for p in ARCH.glob("*.py")) if ARCH.exists() else []

    # ★★ 依赖图必须在 **「tmp ∪ 归档」** 上算，不能只在 tmp 上算。
    #    实测踩过：只在 tmp 上算时，把库 `_sweep.py` 搬进归档后，引用它的脚本
    #    **不再记录这条依赖**（因为 `_sweep` 已不在 `names`）→ `imported_by` 里
    #    没有它 → 「归档造成的断 import」这条不变量**静默通过**。
    #    而那恰恰是它唯一要抓的东西。
    known = names | set(archived_on_disk)

    deps_of: dict[str, set[str]] = {}
    is_main: dict[str, bool] = {}
    has_argp: dict[str, bool] = {}
    for p in tmp_files:
        d, m, a = _scan_file(p, known)
        deps_of[p.stem] = d
        is_main[p.stem] = m
        has_argp[p.stem] = a

    imported_by: dict[str, set[str]] = defaultdict(set)
    for src, ds in deps_of.items():
        for d in ds:
            imported_by[d].add(src)

    ext_deps = _external_downstream(known)
    mentioned = _mentioned_in_docs(names)

    cls: dict[str, str] = {}
    for n in names:
        if imported_by.get(n) or ext_deps.get(n):
            cls[n] = "library"
        elif is_main[n] or has_argp[n]:
            cls[n] = "entrypoint"
        else:
            cls[n] = "scratch"

    decision = {n: ("keep" if (cls[n] == "library" or n in mentioned) else "archive")
                for n in names}

    # ⚠️ 重跑只按"当前扫描"重算 → 会把**已搬走的归档记录抹掉**（实测踩过）。
    #    所以**从文件系统重建**归档集，而不是依赖上一版 JSON 的历史：
    #    `_archive/` 里有什么，就是归档了什么（自我描述，重跑也不会丢）。
    decision.update({n: "archive" for n in archived_on_disk if n not in names})
    decision.update(PINNED)

    return {
        "names": names,
        "class": cls,
        "decision": decision,
        "imported_by": {k: sorted(v) for k, v in imported_by.items()},
        "external_downstream": {k: sorted(v) for k, v in ext_deps.items()},
        "mentioned_in_docs": {k: sorted(v) for k, v in mentioned.items()},
        "archived_on_disk": archived_on_disk,
        "counts": dict(Counter(cls.values())),
        "tmp_files": tmp_files,
        "root": ROOT,
    }


def main() -> int:
    d = analyze()
    names: set[str] = d["names"]
    cls: dict[str, str] = d["class"]
    decision: dict[str, str] = d["decision"]
    imported_by: dict[str, list[str]] = d["imported_by"]
    mentioned: dict[str, list[str]] = d["mentioned_in_docs"]
    cnt = Counter(cls.values())

    print(f"retrieval/tmp 下 .py = {len(d['tmp_files'])} 个")
    print(f"\n分类：library {cnt['library']} ｜ entrypoint {cnt['entrypoint']}"
          f" ｜ scratch {cnt['scratch']}")
    print(f"「有出处」（被 .md 报告提到）：{len(mentioned)} 个")
    print(f"「既非库、又无出处」的："
          f"{len([n for n in names if cls[n] == 'entrypoint' and n not in mentioned])}"
          f" 个 entrypoint")

    cited = sorted(n for n in names if cls[n] == "entrypoint" and n in mentioned)
    print(f"\n=== entrypoint 里「有出处」的（报告引用过 → 是真实验，应保留）：{len(cited)} 个")
    for n in cited[:40]:
        print(f"  {n:<30} ← {len(mentioned[n])} 份文档")
    if len(cited) > 40:
        print("  …")

    libs = sorted(n for n in names if cls[n] == "library")
    ext = d["external_downstream"]
    print(f"\n=== 库（{len(libs)} 个）—— 被引用次数 / 是否有外部下游 ===")
    for n in sorted(libs, key=lambda x: -(len(imported_by.get(x, [])) + len(ext.get(x, [])))):
        inside, outside = len(imported_by.get(n, [])), len(ext.get(n, []))
        if inside + outside >= 2 or outside:
            flag = f"  ★外部 {outside} 处" if outside else ""
            print(f"  {n:<28} 被 {inside:>2} 个 tmp 引用{flag}")

    print("\n=== 外部下游明细（tmp 之外谁在用）===")
    if ext:
        for n in sorted(ext, key=lambda x: -len(ext[x])):
            print(f"  {n}  ← {', '.join(ext[n])}")
    else:
        print("  （无：tmp 里没有脚本被 src/cli/tests/web 引用）")

    dc = Counter(decision.values())
    print(f"\n判定：keep {dc['keep']} ｜ archive {dc['archive']}"
          f"（其中**已搬走** {len(d['archived_on_disk'])} 个）")
    print("归档 = 无出处的 entrypoint + scratch（其脚本名**在任何 .md 里都没出现过**）")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": "retrieval/tmp/*.py 分类（库/入口/一次性）+ 保留/归档判定。"
                "生成器：evals/checks/tmp_scripts_audit.py（分析只读，可重跑复核）",
        "counts": d["counts"],
        "decision": dict(sorted(decision.items())),
        "class": dict(sorted(cls.items())),
        "imported_by": d["imported_by"],
        "external_downstream": d["external_downstream"],
        "mentioned_in_docs": d["mentioned_in_docs"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
