"""L0 不变量 · 路径存活**棘轮**：只对**新增**的悬空引用报错。

## 为什么是「棘轮」而不是「清零」

全仓普查（2026-10-05）发现 **492 个路径引用里 79 个是死链**。其中有大量
**历史报告**里的引用（`DEVELOPMENT_LOG.md` / `qa/**` / `archive/**` /
`retrieval/results/*.md`）—— 它们记录的是**当时的真实状态**，改写等于篡改历史。

于是两难：不改 → 闸门一上来就红，很快会被"狼来了"淹没而失效（**比没有更糟**）；
全改 → 篡改历史。

**棘轮**（ratchet）解这个两难：把当前 79 个**记为基线**，闸门只拦**新增**的。
于是历史债不阻塞任何人，而"以后别再写死链"这条纪律**从今天起生效**。
基线里的条目**修一个就少一个**（`--update` 收缩），只进不退。

## 用法

    python evals/checks/path_liveness.py            # 检查（有新死链 → 退出码 1）
    python evals/checks/path_liveness.py --update   # 基线收缩/记录当前状态
    python evals/checks/path_liveness.py --list     # 打印基线全量

## 已知的"假阳性"与白名单

扫的是**文本里的路径字面量**，所以文档里的**举例写法**（`cli/x.py`、`src/.../x.py`、
`a.json/.md`）会被误判为引用。不做白名单的话，闸门会被假阳性淹没 → 必然失效。
故维护 `PLACEHOLDER` 排除这些形态。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "evals" / "baselines" / "path_liveness.json"

# 只认这些顶层目录下的**文件**引用（避免把 URL、包名、相对片段卷进来）
TOPS = ("src", "tests", "cli", "retrieval", "web", "qa", "scripts", "docs", "evals")
REF = re.compile(
    r"(?<![\w./-])"
    r"((?:" + "|".join(TOPS) + r")/[\w./-]+\.(?:py|md|json|csv|html|ps1|js))"
)

SKIP_PARTS = (".venv", "node_modules", ".git", ".pytest-tmp", "site-packages",
              # ⚠️ **归档区不扫**（2026-10-05 加）。理由不是"为了变绿"，而是：
              #    归档 = 冻结的历史脚本，内部路径字符串**天然指向搬走前的布局**。
              #    扫它们会产生一大批"设计使然"的死链，淹没真信号。
              #    闸门守的是**在用的树**；归档的可追溯性由
              #    `evals/baselines/tmp_scripts.json` 负责，不靠路径存活。
              "_archive", "_gold_history")

# ⚠️ **剪枝目录**（按相对路径前缀）：**生成物 / 数据集**，不是"引用源"。
#    为什么必须剪：`assets/artifacts` 有 **18,850 个文件**（MinerU 解析产物），
#    全走一遍要几十秒 —— 闸门跑一次就慢到没人愿意跑，等于失效。
#    这不是"为了变绿而排除"：它们与归档区同理，是**数据/生成物**，
#    里面出现的路径字符串不构成"谁引用了谁"。
PRUNE = (
    "assets/",                 # 18,850 个解析产物
    "retrieval/data/",         # 数据集（8GB）
    "qa/multi/_runs/",         # 跑批产出
    "retrieval/tmp/_archive/", "retrieval/tmp/_gold_history/",
)
SKIP_EXT = (".py", ".md", ".json", ".html", ".ps1", ".js", ".toml", ".cfg", ".ini", ".txt")

# ⚠️ 文档里的**举例/省略**写法，不是引用。不放白名单 → 假阳性淹没闸门。
PLACEHOLDER = re.compile(r"(^|/)x\.py$|xxx|\.\.\.|/x/|(\.\w+){2,}$")


def _is_pruned(rel_dir: str) -> bool:
    """相对目录路径（**带尾斜杠**）是否落在剪枝范围：生成物 / 数据集 / 归档 / 基线数据。"""
    return rel_dir.startswith(PRUNE) or rel_dir.startswith("evals/baselines/")


def _tracked_and_worktree() -> set[Path]:
    """入库文件 ∪ 工作区文本文件（后者含尚未入库的脚本 —— 它们也是"真实存在的引用源"）。

    ⚠️ **用 `os.walk` + 剪枝，不用 `glob("**/*.json")`**：后者会先枚举再筛，
    把 `.venv` 与 `assets/artifacts`（18,850 个文件）全走一遍 → 几十秒（实测踩过）。
    剪枝后亚秒级 —— 闸门慢到没人跑就等于失效。
    """
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                         capture_output=True, encoding="utf-8", errors="replace")
    files = {ROOT / p for p in out.stdout.splitlines() if p.endswith(SKIP_EXT)}

    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"
        # 原地剪枝：把不该进的目录从 dirnames 里删掉，walk 就不会下去（这才是"剪枝"）
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_PARTS and not _is_pruned(prefix + d + "/")]
        for fn in filenames:
            if fn.endswith(SKIP_EXT):
                files.add(Path(dirpath) / fn)

    keep: set[Path] = set()
    for p in files:
        if any(s in p.parts for s in SKIP_PARTS):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(rel.startswith(x) for x in PRUNE):
            continue
        # ⚠️ **基线目录是数据，不是引用源**（2026-10-05 加）。
        #    否则会"自己喂自己"：`path_liveness.json` 里存着死链**作为数据**，
        #    扫描器把它们当引用读 → 基线越跑越胖（实测 78 → 92，新增的正是它自己）。
        #    （**不**排除检查器自身 —— 为变绿而排除自己就是关闸门；它若真含死链，
        #      应该改它的文档，而不是把它从扫描里拿掉。）
        if rel.startswith("evals/baselines/"):
            continue
        keep.add(p)
    return keep


def scan() -> dict[str, list[str]]:
    """返回 `{不存在的被引路径: [引用它的文件, ...]}`（已排占位符）。"""
    refs: dict[str, set[str]] = {}
    for f in sorted(_tracked_and_worktree()):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel_self = f.relative_to(ROOT).as_posix()
        for m in REF.finditer(text):
            t = m.group(1)
            if t == rel_self or PLACEHOLDER.search(t):
                continue
            if not (ROOT / t).exists():
                refs.setdefault(t, set()).add(rel_self)
    return {k: sorted(v) for k, v in refs.items()}


def load_baseline() -> dict[str, list[str]]:
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def save_baseline(d: dict[str, list[str]]) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(
        json.dumps(dict(sorted(d.items())), ensure_ascii=False, indent=1),
        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="路径存活棘轮（只拦新增悬空引用）")
    ap.add_argument("--update", action="store_true",
                    help="把当前状态写回基线（修好的会被移除 → 基线只进不退）")
    ap.add_argument("--list", action="store_true", help="打印基线全量")
    args = ap.parse_args()

    cur = scan()
    base = load_baseline()

    if args.list:
        print(f"基线共 {len(base)} 条：")
        for k in sorted(base):
            print(f"  {k}")
            for s in base[k]:
                print(f"      ← {s}")
        return 0

    if args.update:
        added = sorted(set(cur) - set(base))
        gone = sorted(set(base) - set(cur))
        save_baseline(cur)
        print(f"基线已更新：{len(base)} → {len(cur)}")
        for k in added:
            print(f"  + 新记入  {k}")
        for k in gone:
            print(f"  - 已修复  {k}")
        return 0

    new = sorted(set(cur) - set(base))
    fixed = sorted(set(base) - set(cur))

    print(f"扫描：现 {len(cur)} 个悬空引用 ｜ 基线 {len(base)} 个")
    if fixed:
        print(f"\n✓ 基线里有 {len(fixed)} 条**已修复** —— 跑 `--update` 收缩基线：")
        for k in fixed:
            print(f"    {k}")

    if not new:
        c = Counter(k.split("/")[0] for k in cur)
        print(f"\n结论：**无新增**悬空引用 ✓（基线内 {len(base)} 条为历史债，不阻塞）")
        print("      分布：" + "  ".join(f"{d}×{n}" for d, n in c.most_common()))
        return 0

    print(f"\n✗ 发现 **{len(new)} 个新增**悬空引用（基线里没有）：")
    for k in new:
        print(f"    ✗ {k}")
        for s in cur[k][:4]:
            print(f"          ← {s}")
        if len(cur[k]) > 4:
            print(f"          … 另有 {len(cur[k]) - 4} 处")
    print("\n修法：改成真实路径；若确属「举例/占位」，把它写成不匹配「目录/文件.扩展名」"
          "的形态（如反引号里只写文件名），或确认后 `--update` 记入基线。")
    print("⚠️ **不要**为了让闸门变绿而 `--update` 掉真的悬空引用 —— 那等于关掉闸门。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
