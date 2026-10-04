"""盘点 `retrieval/tmp/*.py`（294 个）：谁是**库**、谁是**入口**、谁是一次性实验。

**只读，不改任何东西。** 产出 `evals/baselines/tmp_scripts.json`（分类结果）。

## 为什么不能只看 `import`

这批脚本大量用 `importlib.util.spec_from_file_location("x", HERE/"tmp"/"_y.py")`
**动态加载** —— 纯 `ast` 静态分析会**漏掉它们**，而它们恰恰是"库"（被多处共用）。
所以两条路一起走：

  ① **静态**：`ast` 解析 `import X` / `from X import ...`，取本地模块名
  ② **动态**：正则找源码里的字符串常量 `"_xxx.py"` / `'_xxx.py'`

## 分类规则（按"删掉它会怎样"）

    library   被 **≥1 个其它文件**引用（含 retrieval/tmp 之外）→ 不能动，要迁到 `retrieval_core/`
    entrypoint 有 `if __name__ == "__main__"` 或 `argparse` → 可跑，归 `evals/runners/`
    scratch    两者都不是 → 一次性实验（归档候选）
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "retrieval" / "tmp"
OUT = ROOT / "evals" / "baselines" / "tmp_scripts.json"

# 引用来源：tmp 之外也算（那些是"真正的下游"，迁移时必须一起改）
OUTSIDE_DIRS = ("src", "cli", "tests", "web", "evals")
STR_REF = re.compile(r"['\"]\s*([A-Za-z0-9_]+)\.py\s*['\"]")

tmp_files = sorted(p for p in TMP.glob("*.py") if p.is_file())
names = {p.stem for p in tmp_files}
print(f"retrieval/tmp 下 .py = {len(tmp_files)} 个")


def scan(path: Path) -> tuple[set[str], bool, bool]:
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


deps_of: dict[str, set[str]] = {}
is_main: dict[str, bool] = {}
has_argp: dict[str, bool] = {}
for p in tmp_files:
    d, m, a = scan(p)
    deps_of[p.stem] = d
    is_main[p.stem] = m
    has_argp[p.stem] = a

# 反向：谁被引用
imported_by: dict[str, set[str]] = defaultdict(set)
for src, ds in deps_of.items():
    for d in ds:
        imported_by[d].add(src)

# 外部下游（tmp 之外的目录里，是否出现 `_xxx` 这个模块名）
ext_files = []
for d in OUTSIDE_DIRS:
    ext_files += [p for p in (ROOT / d).rglob("*.py") if p.is_file()]
ext_files += [p for p in (ROOT / "retrieval" / "scripts").glob("*.py")]
ext_deps: dict[str, set[str]] = defaultdict(set)
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
        ext_deps[n].add(rel)

cls: dict[str, str] = {}
for n in names:
    if imported_by.get(n) or ext_deps.get(n):
        cls[n] = "library"
    elif is_main[n] or has_argp[n]:
        cls[n] = "entrypoint"
    else:
        cls[n] = "scratch"

# ③ 「有出处」信号：脚本名是否被**文档/报告**提到。
#    报告一般会写"由 `_xxx.py` 产出" —— 这是"这个脚本是真实验、不是随手写的"的最强证据。
#    只用文件名匹配（`_xxx.py` 字面量），避免把普通词算进来。
mentioned: dict[str, set[str]] = defaultdict(set)
docs = [p for p in ROOT.rglob("*.md")
        if p.is_file() and not any(s in p.parts for s in (".venv", ".git", "node_modules"))]
for p in docs:
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    for n in names:
        if f"{n}.py" in text:
            mentioned[n].add(p.relative_to(ROOT).as_posix())

cnt = Counter(cls.values())
print(f"\n分类：library {cnt['library']} ｜ entrypoint {cnt['entrypoint']} ｜ scratch {cnt['scratch']}")
print(f"「有出处」（被 .md 报告提到）：{len(mentioned)} 个")
print(f"「既非库、又无出处」的："
      f"{len([n for n in names if cls[n] == 'entrypoint' and n not in mentioned])} 个 entrypoint")

print("\n=== entrypoint 里「有出处」的（报告引用过 → 是真实验，应保留）===")
cited = sorted(n for n in names if cls[n] == "entrypoint" and n in mentioned)
for n in cited[:40]:
    print(f"  {n:<30} ← {len(mentioned[n])} 份文档")
if len(cited) > 40:
    print(f"  … 共 {len(cited)} 个")

libs = sorted(n for n in names if cls[n] == "library")
print(f"\n=== 库（{len(libs)} 个）—— 被引用次数 / 是否有外部下游 ===")
for n in sorted(libs, key=lambda x: -(len(imported_by.get(x, set())) + len(ext_deps.get(x, set())))):
    inside = len(imported_by.get(n, set()))
    outside = len(ext_deps.get(n, set()))
    if inside + outside >= 2 or outside:
        flag = f"  ★外部 {outside} 处" if outside else ""
        print(f"  {n:<28} 被 {inside:>2} 个 tmp 引用{flag}")

print("\n=== 外部下游明细（tmp 之外谁在用）===")
if ext_deps:
    for n in sorted(ext_deps, key=lambda x: -len(ext_deps[x])):
        print(f"  {n}  ← {', '.join(sorted(ext_deps[n]))}")
else:
    print("  （无：tmp 里没有脚本被 src/cli/tests/web 引用）")

OUT.parent.mkdir(parents=True, exist_ok=True)
# 判定：库 或 有出处 → 保留；其余（无出处的入口 + scratch）→ 归档。
# 为什么用「有出处」而不是「有 argparse」：报告会写"由 `_xxx.py` 产出"，
# 这是"真跑过、有结论"的记录；而 argparse 只是"能跑"，一堆随手写的探针也有。
decision = {n: ("keep" if (cls[n] == "library" or n in mentioned) else "archive")
            for n in names}

# ⚠️ 重跑只按"当前扫描"重算 → 会把**已搬走的归档记录抹掉**（实测踩过）。
#    所以**从文件系统重建**归档集，而不是依赖上一版 JSON 的历史：
#    `_archive/` 里有什么，就是归档了什么（自我描述，重跑也不会丢）。
ARCH = TMP / "_archive"
archived_on_disk = sorted(p.stem for p in ARCH.glob("*.py")) if ARCH.exists() else []
decision.update({n: "archive" for n in archived_on_disk if n not in names})

# `PINNED`：明知「无出处」但**故意保留**的，必须显式钉住 —— 否则下次重跑又会被划成
# archive（演示脚本就吃过这个亏：它是给人重放的，不该因为报告里没提就被归档）。
PINNED = {"_salvage_demo": "keep"}
decision.update(PINNED)

dc = Counter(decision.values())
archived = sorted(n for n, d in decision.items() if d == "archive" and n not in names)
print(f"\n判定：keep {dc['keep']} ｜ archive {dc['archive']}"
      f"（其中**已搬走** {len(archived)} 个，仍在 tmp 的 {dc['archive'] - len(archived)} 个）")
print("归档 = 无出处的 entrypoint + scratch（其脚本名**在任何 .md 里都没出现过**）")

OUT.write_text(json.dumps({
    "note": "retrieval/tmp/*.py 分类（库/入口/一次性）+ 保留/归档判定。"
            "生成器：evals/checks/tmp_scripts_audit.py（只读，可重跑复核）",
    "counts": dict(cnt),
    "decision": dict(sorted(decision.items())),
    "class": dict(sorted(cls.items())),
    "imported_by": {k: sorted(v) for k, v in sorted(imported_by.items())},
    "external_downstream": {k: sorted(v) for k, v in sorted(ext_deps.items())},
    "mentioned_in_docs": {k: sorted(v) for k, v in sorted(mentioned.items())},
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT.relative_to(ROOT)}")
