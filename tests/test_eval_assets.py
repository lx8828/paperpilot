"""评测资产完整性：把「干净 clone 到底能不能评测」变成**默认跑的回归防护**。

为什么值得占用 CI 时间：本文件每一条都在防**已经真实发生过的静默失效**。
它们的共同特征是 —— **不报错**，只是让别人 clone 之后「跑不起来」或「拿到错的数」，
所以最容易被忽略、也最难在事后发现：

    ① 出题 gold 没入库   → clone 后 8/15 消失，导出/同步检查直接报「缺 gold」
    ② r2dev 真值没入库   → 论文线评测**完全无从复现**（而 gold 不重标 = 不可再生）
    ③ CLI 迁移没提交     → 旧路径引用全部指向空气；HEAD 里是新家不存在
    ④ 草稿与 gold 混放   → 后来人改错文件（真发生过）

写法上**刻意断言不变量**（"磁盘上每个 gold 都已入库"），而不是今天的快照
（"必须是 15 个"）—— 否则加一组新题就会误报，闸门会被"狼来了"淹没。

两层（照 `conftest.py` 的约定）：
    · 无标记              → 纯 `git` 查询，秒级，CI 默认跑
    · `@pytest.mark.local` → 真起 CLI 入口验证 `parents[2]`，要导 torch，慢，默认跳过
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    """跑 git 并返回 stdout。

    ⚠️ 必须显式 `encoding="utf-8"`：Windows 下 `text=True` 默认按 **gbk** 解码，
    而 git 会吐 UTF-8 中文路径 → reader 线程 `UnicodeDecodeError`（实测踩过）。
    """
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          encoding="utf-8", errors="replace").stdout


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="需要 git 工作区（这几条检查的是「仓库状态」而非代码行为）")


# ─────────────────────── ① 出题 gold 全部入库 ───────────────────────

def test_every_qa_gold_is_tracked():
    """磁盘上**每一个** gold 都必须在 git 里。

    曾发生：15 个 gold 里只 `git add` 了 7 个 → 干净 clone 丢掉 group3/4/5 的
    全部组级题集与 group2 的 4 个单篇题集，而**本地毫无异样**。
    """
    materials = ROOT / "retrieval" / "tmp"
    on_disk = sorted(p for p in materials.rglob("*.questions.json") if p.is_file())
    assert on_disk, "一个 gold 都没找到 —— 目录结构可能变了，请更新本测试的定位"

    tracked = set(_git("ls-files", "retrieval/tmp").splitlines())
    missing = [p.relative_to(ROOT).as_posix() for p in on_disk
               if p.relative_to(ROOT).as_posix() not in tracked]
    assert not missing, (
        "以下 gold **未入库**（干净 clone 会丢掉它们）：\n  "
        + "\n  ".join(missing)
        + "\n修：`git add` 它们，并确认 .gitignore 没有挡住（gold 是出题真源）。")


def test_no_uncommitted_qa_gold_changes():
    """已入库的 gold 不该有未提交改动 —— 否则"仓库里的"和"在用的"是两份。"""
    dirty = [ln for ln in _git("status", "--porcelain", "retrieval/tmp").splitlines()
             if "questions.json" in ln]
    assert not dirty, ("有 gold 存在未提交改动（评测用的可能是另一份）：\n  "
                       + "\n  ".join(dirty))


# ─────────────────────── ② r2dev 评测真值入库 ───────────────────────

# 题集 / 映射：判分的**输入**。缺任何一个都跑不了评测，且都不可再生。
_R2DEV_REQUIRED = (
    "gold_final2.csv",           # 现役真值 · 20 篇口径
    "gold_final3.csv",           # 现役真值 · 50 篇口径
    "facets_v2_selected.json",   # 题集（28 组合）
    "subqueries.json",           # 子查询：查询侧输入
    "pdf_map.json",              # PDF 映射
)

# 体积大 / 可再生：**不该**入库（否则仓库膨胀，且它们能重建）
_R2DEV_FORBIDDEN = ("_evidence.json", ".parquet", "mt_cache",
                    "mineru_progress", "/prodchunk", "/pdftext", "/clusters/")


def test_r2dev_truth_and_inputs_are_tracked():
    """r2dev 的**真值 + 判分输入**必须入库。

    曾发生：整个 `retrieval/data/` 被一句 `retrieval/data/` 挡在 git 之外，
    干净 clone **拿不到任何论文线评测真值**。而既定原则是 **gold 不重标**
    （= 不可再生），所以这是**不可逆的数据风险**，不是"少个文件而已"。
    """
    tracked = _git("ls-files", "retrieval/data").splitlines()
    for name in _R2DEV_REQUIRED:
        assert any(t.endswith(name) for t in tracked), (
            f"r2dev 的 {name} 未入库 —— 干净 clone 跑不了论文线评测。\n"
            f"（{name} 不可再生，见 retrieval/data/r2dev/README.md）")


def test_r2dev_big_regenerable_files_stay_out():
    """反过来也要守住：大件/可再生件**不能**混进 git（仓库体积是公共成本）。"""
    tracked = _git("ls-files", "retrieval/data").splitlines()
    leak = [t for t in tracked if any(b in t for b in _R2DEV_FORBIDDEN)]
    assert not leak, (
        "以下「体积大/可再生」的文件被误入库（应继续被 .gitignore 挡住）：\n  "
        + "\n  ".join(leak))


# ─────────────────────── ③ CLI 路径引用不再悬空 ───────────────────────

# 扫描面：**存活**的代码与入口文档。刻意不含历史报告（DEVELOPMENT_LOG / qa/** /
# archive/** / results/*.md）—— 它们记录的是"当时的真实状态"，改写等于篡改历史。
_SCAN_DIRS = ("src", "tests", "cli/eval")


def _stale_cli_refs() -> list[str]:
    """找出「指向旧 `cli/x.py`、而新家 `cli/eval/x.py` 确实存在」的引用。

    这类引用是**纯噪音**：文件明明搬到了新家，引用却停留在旧地址。
    曾发生：`cli/` → `cli/eval/` 迁移做了，但**既没提交迁移、也没更新引用**，
    导致全仓 79 个"死链"（从 HEAD 看是活的，从工作区看是死的）。
    """
    eval_dir = ROOT / "cli" / "eval"
    names = {p.stem for p in eval_dir.glob("*.py")}
    files: list[Path] = []
    for d in _SCAN_DIRS:
        files += [p for p in (ROOT / d).rglob("*.py") if p.is_file()]
    files += [ROOT / "retrieval" / "README.md"]
    files += [p for p in (ROOT / "retrieval" / "scripts").glob("*.py")]

    out: list[str] = []
    for f in files:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for name in names:
            for m in re.finditer(r"(?<![\w/.-])cli/" + re.escape(name) + r"\.py", text):
                ln = text[:m.start()].count("\n") + 1
                out.append(f"{f.relative_to(ROOT)}:{ln}  {m.group(0)}"
                           f"  → 应写 cli/eval/{name}.py")
    return out


def test_no_stale_cli_paths_in_live_code():
    stale = _stale_cli_refs()
    assert not stale, (
        "以下引用指向旧 CLI 路径（新家已存在，属纯噪音）：\n  "
        + "\n  ".join(stale))


# ─────────────────────── ④ 草稿/备份不再与 gold 混放 ───────────────────────

_DECOY = re.compile(r"\.bak$|draft|\.new[0-9]")


def test_no_draft_or_backup_next_to_gold():
    """gold 目录下不该有草稿/备份 —— 它们同前缀，会让人改错文件（真发生过）。

    归档位置：`retrieval/tmp/_gold_history/<group>/`（保留可追溯，但不进 git status）。
    """
    materials = ROOT / "retrieval" / "tmp"
    decoys = [p.relative_to(ROOT).as_posix()
              for p in materials.glob("group*/*")
              if p.is_file() and _DECOY.search(p.name)]
    assert not decoys, ("gold 目录下仍有草稿/备份（应归档到 _gold_history/）：\n  "
                        + "\n  ".join(decoys))


# ─────────────────────── ⑤ 入口真能跑（慢，默认跳过）───────────────────────

@pytest.mark.local
@pytest.mark.parametrize("rel", [
    "cli/eval/run_group_qa.py",
    "cli/eval/run_multi_qa.py",
])
def test_cli_entrypoint_help_runs(rel: str):
    """`--help` 退出码必须为 0。

    这验证的是迁移时改的 `parents[1]` → `parents[2]`：错了的话 ROOT 会指偏，
    所有依赖 `ROOT` 的路径全废 —— 而 `--help` 是**最便宜**的一次端到端探活。
    """
    r = subprocess.run([sys.executable, rel, "--help"], cwd=ROOT,
                       capture_output=True, encoding="utf-8", errors="replace",
                       timeout=300)
    assert r.returncode == 0, f"{rel} --help 退出码 {r.returncode}\n{r.stderr[-800:]}"
