"""L4 · 对外可比：LoFT-RAG / QAMPARI，**官方口径**一条命令重跑与汇总。

## 这一层为什么单独存在

L1~L3 判的是"我们有没有退步"；L4 判的是"**这个数能不能跟别人比**"。
能比的前提有三条，缺一条数字就没意义 —— 所以本文件的 `--check` 把它们**逐条摊开**：

  ① **同一口径** —— 用 **LoFT 官方评测代码**（`retrieval/data/loft/official/`）算，
     不自己重实现指标（自实现与官方差一点，就是"口径错"）。
  ② **同一档位** —— 只覆盖 **`128k` 档**（755 段落 ≈ 105k token）。
     ★ 该档**保证 gold 存在于语料**，故它**测不了"空集"**情形。
  ③ **同一前提** —— 我们的数字是"deepseek-flash + 我们的检索"，
     与论文里别的模型不是同一套系统 → 只能作**参照**，不能说"超过了谁"。

## 官方口径的坑（逐条核对过，别再踩）

    em          = 集合相等（gold 与 pred 完全一致）
    coverage    = |pred ∩ gold| / |gold|      ← **只有 recall，没有 precision**
    subspan_em  = 双向子串 + 匈牙利分配 + 全对齐才算 1
    f1          = **官方多值分支根本没赋值** → 恒为 **0.0**
                  ★ 所以 **不要引用官方 f1**（它不代表任何东西）

## 用法

    python evals/runners/l4_qampari.py --check        # 前置检查（离线、秒级）
    python evals/runners/l4_qampari.py --collect      # 把已有 official_runs 全落成记录
    python evals/runners/l4_qampari.py --official <preds.jsonl> \\
           --queries <queries.jsonl> --name my_run    # 用**官方 CLI**现跑一遍并落记录
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
RUNS = ROOT / "retrieval" / "results" / "official_runs"
QAMPARI = ROOT / "retrieval" / "data" / "qampari"

# ★ 论文 Table 2（qampari 128k test）的公开数字 —— **随数字一起给参照**，
#   否则"0.700"是个没有坐标的数。各自模型/系统不同，只作参照。
PUBLISHED = {
    "专用 RAG pipeline": 0.55,
    "Gemini 1.5 Pro（直读）": 0.44,
    "GPT-4o": 0.27,
    "Claude": 0.25,
}
# ★ 审计结论：其中约 0.150 来自**模型闭卷知识**；对外讲"检索贡献"应报差额。
CLOSED_BOOK = 0.150

NOTE = (
    "LoFT-RAG / QAMPARI **128k 档**（755 段 ≈ 105k token，100 题，gold 5~6/题）"
    " ｜ 指标用 **LoFT 官方评测代码**（`retrieval/data/loft/official/`）算，非自实现"
    " ｜ ★ `coverage` **只有 recall**；官方 `f1` 恒为 0（多值分支未赋值）→ **别引用**"
    " ｜ ★ 边界：**限 128k 档**且该档**保证 gold 存在** → 测不了「空集」；"
    "系统是「我们的检索 + deepseek-flash」，与下列其它系统**不同源**，只作参照："
    + " / ".join(f"{k} {v}" for k, v in PUBLISHED.items())
    + f" ｜ ★ 其中约 {CLOSED_BOOK} 来自模型**闭卷知识**，讲「检索贡献」应报差额"
)

METRICS = (("subspan_em", "qampari.subspan_em"), ("em", "qampari.em"),
           ("coverage", "qampari.coverage"))


def _load_quality(p: Path) -> dict[str, Any] | None:
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
    if not isinstance(q, dict) or not any(k in q for k, _ in METRICS):
        return None
    return o


def check() -> int:
    """前置检查：把"能不能跑"逐条摊开（离线、秒级）。"""
    ok = True
    print("=" * 84)
    print("L4 前置检查（对外可比的两个硬前提）")
    print("=" * 84)

    ev = OFFICIAL / "run_evaluation.py"
    ok_pkg = ev.exists() and (OFFICIAL / "evaluation" / "rag.py").exists()
    ok &= ok_pkg
    print(f"  {'✓' if ok_pkg else '✗'} 官方评测包：{OFFICIAL.relative_to(ROOT)}")
    if not ok_pkg:
        print("      ← 缺它就只能自实现指标，**口径必偏** → 先取 google-deepmind/loft")

    for name, p, how in (
        ("QAMPARI 128k 语料", QAMPARI / "corpus", "HF: qampari（见 retrieval/data/qampari/README.md）"),
        ("QAMPARI 题集", QAMPARI / "data" / "test-00000-of-00001.parquet", "HF: qampari 的 data/"),
        ("子查询", QAMPARI / "queries" / "queries-00000-of-00001.parquet", "HF: qampari 的 queries/"),
    ):
        e = p.exists()
        ok &= e
        print(f"  {'✓' if e else '✗'} {name}：{p.relative_to(ROOT)}")
        if not e:
            print(f"      ← 取法：{how}")

    runs = sorted(d for d in RUNS.glob("*") if d.is_dir()) if RUNS.exists() else []
    good = [d for d in runs if _load_quality(d / "preds_metrics.json")]
    print(f"\n  已有运行 {len(runs)} 个 ｜ 其中指标**可读** {len(good)} 个")
    for d in runs:
        if d not in good:
            why = "preds_metrics.json 缺失或 **0KB 空文件**" if (d / "preds_metrics.json").exists() \
                else "缺 preds_metrics.json"
            print(f"      ⚠️ {d.name}：{why}（--collect 会跳过它）")

    print(f"\n结论：{'前置齐备，可 `--collect` / `--official`' if ok else '**有缺项**，见上'}")
    return 0 if ok else 1


def _rel(p: Path) -> str:
    """相对仓库根的路径；不在根下则退回绝对路径。

    ⚠️ `Path.relative_to` 在路径不在根下时会**抛 ValueError**（测试用临时目录就会碰到）。
    不加这层，一次路径意外会**中断整批落记录** —— 与 `load_all` 里"坏行不该毁整表"同理。
    """
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _emit_run(name: str, q: dict[str, Any], evidence: str) -> int:
    recs = []
    for k, metric in METRICS:
        v = q.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        recs.append(dict(layer="L4", name=name, metric=metric, value=float(v),
                         n=100, note=NOTE, evidence_path=evidence))
    # 健康度：`num_unanswered_queries`（空预测行会被官方口径**剔出分母** → 虚高陷阱）
    nu = q.get("num_unanswered_queries")
    if isinstance(nu, int):
        recs.append(dict(layer="L4", name=name, metric="qampari.unanswered",
                         value=float(nu), n=100,
                         note="空预测题的条数 ｜ ★ 官方口径会把它们**剔出分母** → "
                              "非 0 时指标**虚高**，必须一起报（**应为 0**）",
                         evidence_path=evidence))
    if recs:
        R.emit(*recs)
    return len(recs)


def collect() -> int:
    """把 `retrieval/results/official_runs/*/preds_metrics.json` 全落成记录。"""
    if not RUNS.exists():
        print(f"（没有 {RUNS.relative_to(ROOT)} —— 先跑一次 `--official`）")
        return 0
    total = n_skip = 0
    for d in sorted(x for x in RUNS.glob("*") if x.is_dir()):
        f = d / "preds_metrics.json"
        o = _load_quality(f)
        if o is None:
            n_skip += 1
            print(f"  ⚠️ 跳过 {d.name}（指标文件缺失 / 0KB / 缺字段）")
            continue
        # ★ per-run 容错：单个运行出问题只跳过它，不中断整批
        try:
            n = _emit_run(d.name, o["quality"], _rel(f))
        except Exception as e:  # noqa: BLE001
            n_skip += 1
            print(f"  ⚠️ 跳过 {d.name}（落记录失败：{type(e).__name__}: {e}）")
            continue
        total += n
        print(f"  ✓ {d.name}：落 {n} 条 ｜ "
              f"subspan_em={o['quality'].get('subspan_em')} em={o['quality'].get('em')}")
    print(f"\n合计落 {total} 条记录，跳过 {n_skip} 个运行")
    print("→ `python evals/report.py --table` 看全部层；"
          "`python evals/checks/metrics_ratchet.py --update` 记入基线")
    return 0


def official(preds: Path, queries: Path | None, name: str) -> int:
    """用**官方 CLI** 现跑一遍，落记录（这才是"重跑官方口径数"）。"""
    ev = OFFICIAL / "run_evaluation.py"
    if not ev.exists():
        print(f"✗ 缺官方评测包：{ev.relative_to(ROOT)}（自实现指标会让口径偏）")
        return 1
    if not preds.exists():
        print(f"✗ 预测文件不存在：{preds}")
        return 1
    if queries is None:
        cand = preds.parent / "queries.jsonl"
        queries = cand if cand.exists() else None
    if queries is None or not queries.exists():
        print("✗ 缺 queries.jsonl（官方 CLI 需 `--answer_file_path`）"
              f"：在 {preds.parent} 下没找到，请显式 `--queries`")
        return 1

    cmd = [sys.executable, str(ev), f"--answer_file_path={queries}",
           f"--pred_file_path={preds}", "--task_type=multi_value_rag"]
    print("跑官方 CLI：\n  " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(OFFICIAL), encoding="utf-8",
                       errors="replace", capture_output=True)
    print(r.stdout[-3000:] or "")
    if r.returncode != 0:
        print(f"✗ 官方 CLI 退出码 {r.returncode}\n{r.stderr[-1500:]}", file=sys.stderr)
        return r.returncode

    m = preds.parent / "preds_metrics.json"
    o = _load_quality(m)
    if o is None:
        print(f"✗ 官方 CLI 跑完了，但 {m.name} 不可读（0KB / 缺字段）—— 拒收")
        return 1
    n = _emit_run(name, o["quality"], _rel(m))
    print(f"✓ 落 {n} 条记录（name={name}）｜ subspan_em={o['quality'].get('subspan_em')}"
          f" em={o['quality'].get('em')} coverage={o['quality'].get('coverage')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L4：QAMPARI 官方口径重跑/汇总")
    ap.add_argument("--check", action="store_true", help="前置检查（默认）")
    ap.add_argument("--collect", action="store_true", help="把已有 official_runs 落成记录")
    ap.add_argument("--official", default="", help="用官方 CLI 跑这个 preds.jsonl")
    ap.add_argument("--queries", default="", help="配套的 queries.jsonl（默认同目录找）")
    ap.add_argument("--name", default="", help="本次运行名（默认用目录名）")
    args = ap.parse_args()

    if args.official:
        p = Path(args.official)
        if not p.is_absolute():
            p = ROOT / p
        q = (ROOT / args.queries) if args.queries else None
        return official(p, q, args.name or p.parent.name)
    if args.collect:
        return collect()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
