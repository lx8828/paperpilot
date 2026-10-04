"""L4 · 对外可比：**LoFT 五个任务**，官方口径一条命令重跑与汇总。

## 这一层为什么单独存在

L1~L3 判的是"我们有没有退步"；L4 判的是"**这个数能不能跟别人比**"。
能比的前提有三条，`--check` 把它们**逐条摊开**：

  ① **同一口径** —— 用 **LoFT 官方评测代码**算（`retrieval/data/loft/official/`），
     **不自己重实现指标**（自实现与官方差一点，就是"口径错"）。
  ② **同一任务** —— 五个任务的指标**各不相同**，不可混谈（见下表）。
  ③ **同一前提** —— 我们的数字是"我们的检索 + 某模型"，与论文里别的系统
     **不同源** → 只能作**参照**，不能说"超过了谁"。

## 五个任务的官方指标（逐行读过官方代码，不是猜的）

| `--task` | 官方指标 | ★ 口径要点 |
|---|---|---|
| `multi_value_rag`（QAMPARI） | `em` / `coverage` / `subspan_em` | `coverage` **只有 recall**（无 precision）；**`f1` 官方恒为 0**（多值分支未赋值）→ **别引用** |
| `rag` | `em` / `f1` | SQuAD 风格**单值** |
| `retrieval` | `recall@k` / `mrecall@k` | **Capped**：gold 数 > `k` 时**除以 `k`**（不是 gold 数） |
| `sql` | `execution_accuracy` | **不强制顺序**（数据集已滤掉需要排序的题） |
| `icl` | `em` | 预测**多值被忽略**（只取第一个）；实例是**多轮**的 |

★ **指标不硬编码**：官方输出里有什么数值指标就**透传**什么，只排除已证实无意义的
（见 `EXCLUDE`）。否则官方加个指标就得改这里，而"改漏"会静默少报。

## 用法

    python evals/runners/l4_loft.py --check                    # 前置（离线、秒级）
    python evals/runners/l4_loft.py --collect                  # 汇总已有运行
    python evals/runners/l4_loft.py --official --task retrieval --name my_run
        # 约定路径：retrieval/results/loft_runs/<task>/<name>/{queries,preds}.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
import report as R  # noqa: E402

OFFICIAL = ROOT / "retrieval" / "data" / "loft" / "official"
# 约定：LoFT 各任务的运行产物落这里（与 QAMPARI 历史的 official_runs 并列）
LOFT_RUNS = ROOT / "retrieval" / "results" / "loft_runs"
QAMPARI_RUNS = ROOT / "retrieval" / "results" / "official_runs"
QAMPARI = ROOT / "retrieval" / "data" / "qampari"

# ★ 论文 Table 2（qampari 128k test）的公开数字 —— **随数字一起给参照**，
#   否则"0.700"是个没有坐标的数。各自模型/系统不同，只作参照。
PUBLISHED = {
    "专用 RAG pipeline": 0.55,
    "Gemini 1.5 Pro（直读）": 0.44,
    "GPT-4o": 0.27,
    "Claude": 0.25,
}
CLOSED_BOOK = 0.150   # 审计结论：约这么多来自模型**闭卷知识**

_MV_BOUNDARY = ("★ 边界：**限 128k 档**且该档**保证 gold 存在** → 测不了「空集」；"
                "系统是「我们的检索 + deepseek-flash」，与下列其它系统**不同源**，只作参照："
                + " / ".join(f"{k} {v}" for k, v in PUBLISHED.items())
                + f" ｜ ★ 其中约 {CLOSED_BOOK} 来自模型**闭卷知识**，讲「检索贡献」应报差额")

TASKS: dict[str, dict[str, Any]] = {
    "multi_value_rag": dict(
        prefix="qampari",
        runs=QAMPARI_RUNS,
        extra=("LoFT-RAG / **QAMPARI 128k 档**（755 段 ≈ 105k token，100 题，gold 5~6/题）"
               " ｜ ★ `coverage` **只有 recall**；官方 `f1` 恒为 0（多值分支未赋值）"
               " → **别引用** ｜ " + _MV_BOUNDARY),
    ),
    "rag": dict(prefix="loft_rag", runs=LOFT_RUNS,
                extra="LoFT-RAG **单值**任务（SQuAD 风格 `em`/`f1`）"),
    "retrieval": dict(prefix="loft_retrieval", runs=LOFT_RUNS,
                      extra="LoFT **检索**任务 ｜ ★ `recall@k` 是 **Capped**："
                            "gold 数 > k 时**除以 k**（不是 gold 数），跨榜比数前先确认对方口径"),
    "sql": dict(prefix="loft_sql", runs=LOFT_RUNS,
                extra="LoFT-**SQL** 任务（Spider）｜ 主指标 `execution_accuracy` ｜ "
                      "★ **不强制顺序**（LOFT 建集时已滤掉需要排序的题）"),
    "icl": dict(prefix="loft_icl", runs=LOFT_RUNS,
                extra="LoFT-**ICL** 任务 ｜ 主指标 `em` ｜ ★ 预测**多值被忽略**（只取第一个）、"
                      "实例是**多轮**的"),
}

# ★ 官方输出里**无意义**的指标 → 不进记录（否则会被人当结果引用）。
EXCLUDE: dict[str, frozenset[str]] = {
    # 多值分支没给 f1 赋值，官方 `f1` 恒为 0.0（不是"算出来 0"，是"根本没算"）
    "multi_value_rag": frozenset({"f1"}),
}


def _rel(p: Path) -> str:
    """相对仓库根的路径；不在根下则退回绝对路径。

    ⚠️ `Path.relative_to` 在路径不在根下时会**抛 ValueError**（测试用临时目录就会碰到）。
    不加这层，一次路径意外会**中断整批落记录** —— 与 `report.load_all` 里
    "坏行不该毁整表"同理。
    """
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _load_metrics(p: Path) -> dict[str, Any] | None:
    """读官方 `preds_metrics.json`；**读不出/不合规一律返回 None**（= 拒收，不当 0 分）。

    为什么必须拒收：本仓就躺着一个 **0KB** 的 `preds_metrics.json`
    —— 长得像结果、其实没内容。若把它当"全 0 分"计入，
    **看起来是回退，其实是文件坏了**。

    ⚠️ 如实说明：那条 `size == 0` 判断**不是必需的** —— `json.loads("")` 本来就会
    抛错。显式写出来是为了把「**空文件 = 坏结果**」这条口径摆明；
    真正判"合不合规"的是后面的 `quality` 字段检查。
    """
    try:
        if p.stat().st_size == 0:
            return None
        o = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    q = o.get("quality")
    if not isinstance(q, dict) or not q:
        return None
    return o


def _note(task: str, n: int = 100) -> str:
    t = TASKS[task]
    return (f"{t['extra']} ｜ 指标用 **LoFT 官方评测代码**"
            f"（`retrieval/data/loft/official/`）算，非自实现 ｜ 题数 {n}")


def _emit_run(task: str, name: str, q: dict[str, Any], evidence: str) -> int:
    """把一个运行的官方指标落成记录。★ 指标**透传**，只排除 `EXCLUDE` 里的。"""
    t = TASKS[task]
    skip = EXCLUDE.get(task, frozenset())
    note = _note(task)
    recs: list[dict] = []
    for k, v in q.items():
        if k in skip or k == "num_unanswered_queries":
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        # ★ 非有限值不 emit（schema 会拒收 NaN，而那类值在表里会变成"NaN"、
        #   比大小永远为假 —— 静默失效的典型）
        if v != v or v in (float("inf"), float("-inf")):
            continue
        recs.append(dict(layer="L4", name=f"{t['prefix']}_{name}",
                         metric=f"{t['prefix']}.{k}", value=float(v), n=100,
                         note=note, evidence_path=evidence))
    # 健康度：空预测题数会被官方口径**剔出分母** → 指标虚高，必须一起报（应为 0）
    nu = q.get("num_unanswered_queries")
    if isinstance(nu, int):
        recs.append(dict(layer="L4", name=f"{t['prefix']}_{name}",
                         metric=f"{t['prefix']}.unanswered", value=float(nu), n=100,
                         note="空预测题的条数 ｜ ★ 官方口径会把它们**剔出分母** → "
                              "非 0 时指标**虚高**（**应为 0**）", evidence_path=evidence))
    if recs:
        R.emit(*recs)
    return len(recs)


def check() -> int:
    """前置检查：把"能不能跑"逐条摊开（离线、秒级）。"""
    ok = True
    print("=" * 88)
    print("L4 前置检查（对外可比的三条硬前提）")
    print("=" * 88)

    ev = OFFICIAL / "run_evaluation.py"
    ok_pkg = ev.exists() and (OFFICIAL / "evaluation" / "rag.py").exists()
    ok &= ok_pkg
    print(f"  {'✓' if ok_pkg else '✗'} 官方评测包：{OFFICIAL.relative_to(ROOT)}")
    if not ok_pkg:
        print("      ← 缺它就只能自实现指标，**口径必偏** → 先取 google-deepmind/loft")

    print("\n  各任务（★ 指标各不相同，不可混谈）：")
    for task, t in TASKS.items():
        runs = sorted(d for d in t["runs"].glob("*") if d.is_dir()) \
            if t["runs"].exists() else []
        good = [d for d in runs if _load_metrics(d / "preds_metrics.json")]
        # QAMPARI 的语料/题集是单独下载的，其余任务的数据来自 LoFT 官方发布
        data_ok = QAMPARI.exists() if task == "multi_value_rag" else None
        flag = "✓" if good else ("○" if runs else "—")
        tail = f"运行 {len(good)}/{len(runs)} 可读" if runs else "尚无运行"
        if data_ok is False:
            tail += " ｜ ✗ 缺 QAMPARI 语料/题集"
        print(f"    {flag} {task:<18}{tail}")
        if t["runs"].exists() and runs:
            for d in runs:
                if d not in good:
                    print(f"        ⚠️ {d.name}：preds_metrics.json 缺失 / **0KB** / 缺字段"
                          f"（--collect 会跳过）")

    print(f"\n  约定落盘：{LOFT_RUNS.relative_to(ROOT)}/<task>/<name>/ "
          f"{{queries.jsonl, preds.jsonl}}")
    print(f"\n结论：{'前置齐备' if ok else '**有缺项**（见上）'}"
          f" —— 可 `--collect`；有 preds 时 `--official`")
    return 0 if ok else 1


def collect(task: str) -> int:
    """把某任务已有运行的 `preds_metrics.json` 落成记录。"""
    runs_dir = TASKS[task]["runs"]
    if not runs_dir.exists():
        print(f"（没有 {runs_dir.relative_to(ROOT)} —— 先跑一次 `--official`）")
        return 0
    total = n_skip = 0
    for d in sorted(x for x in runs_dir.glob("*") if x.is_dir()):
        f = d / "preds_metrics.json"
        o = _load_metrics(f)
        if o is None:
            n_skip += 1
            print(f"  ⚠️ 跳过 {d.name}（指标文件缺失 / 0KB / 缺字段）")
            continue
        # ★ per-run 容错：单个运行出问题只跳过它，不中断整批
        try:
            n = _emit_run(task, d.name, o["quality"], _rel(f))
        except Exception as e:  # noqa: BLE001
            n_skip += 1
            print(f"  ⚠️ 跳过 {d.name}（落记录失败：{type(e).__name__}: {e}）")
            continue
        total += n
        print(f"  ✓ {d.name}：落 {n} 条 ｜ {o['quality']}")
    print(f"\n合计落 {total} 条（task={task}），跳过 {n_skip} 个运行")
    print("→ `python evals/report.py --table` 看全部层；"
          "`python evals/checks/metrics_ratchet.py --update` 记入基线")
    return 0


def official(task: str, name: str, preds: Path | None, queries: Path | None) -> int:
    """用**官方 CLI** 现跑一遍，落记录（这才是"重跑官方口径数"）。"""
    ev = OFFICIAL / "run_evaluation.py"
    if not ev.exists():
        print(f"✗ 缺官方评测包：{ev.relative_to(ROOT)}（自实现指标会让口径偏）")
        return 1
    base = TASKS[task]["runs"] / name
    preds = preds or (base / "preds.jsonl")
    queries = queries or (base / "queries.jsonl")
    if not preds.exists():
        print(f"✗ 预测文件不存在：{preds}")
        return 1
    if not queries.exists():
        print(f"✗ 缺 queries.jsonl（官方 CLI 需 `--answer_file_path`）：{queries}")
        return 1

    cmd = [sys.executable, str(ev), f"--answer_file_path={queries}",
           f"--pred_file_path={preds}", f"--task_type={task}"]
    print("跑官方 CLI：\n  " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(OFFICIAL), encoding="utf-8",
                       errors="replace", capture_output=True)
    print(r.stdout[-3000:] or "")
    if r.returncode != 0:
        print(f"✗ 官方 CLI 退出码 {r.returncode}\n{r.stderr[-1500:]}", file=sys.stderr)
        return r.returncode

    m = preds.parent / "preds_metrics.json"
    o = _load_metrics(m)
    if o is None:
        print(f"✗ 官方 CLI 跑完了，但 {m.name} 不可读（0KB / 缺字段）—— 拒收")
        return 1
    n = _emit_run(task, name, o["quality"], _rel(m))
    print(f"✓ 落 {n} 条记录（task={task}, name={name}）｜ {o['quality']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L4：LoFT 五个任务的官方口径重跑/汇总")
    ap.add_argument("--check", action="store_true", help="前置检查（默认）")
    ap.add_argument("--collect", action="store_true", help="汇总已有运行")
    ap.add_argument("--official", action="store_true", help="用官方 CLI 现跑一遍")
    ap.add_argument("--task", default="multi_value_rag", choices=sorted(TASKS),
                    help="LoFT 任务类型（各任务指标不同）")
    ap.add_argument("--name", default="", help="本次运行名（默认用目录名）")
    ap.add_argument("--preds", default="", help="preds.jsonl 路径（默认按约定找）")
    ap.add_argument("--queries", default="", help="queries.jsonl 路径（默认按约定找）")
    args = ap.parse_args()

    if args.official:
        p = Path(args.preds)
        if args.preds and not p.is_absolute():
            p = ROOT / p
        q = Path(args.queries)
        if args.queries and not q.is_absolute():
            q = ROOT / q
        name = args.name or (p.parent.name if args.preds else "")
        if not name:
            print("✗ `--official` 需要 `--name`（或给 `--preds` 让它从目录名推断）")
            return 1
        return official(args.task, name, p or None, q or None)
    if args.collect:
        return collect(args.task)
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
