"""L0 不变量 · **指标棘轮**：跑批后的数字掉出容差就报错。

## 为什么需要它

统一记录格式（`evals/report.py`）解决了"**能不能比**"；
这一步解决"**有没有人真的去比**"。没有棘轮时，指标掉了只会留在 JSONL 里，
**没人看** —— 直到某天有人翻日志才发现"三周前就退化了"。

## ★ 三条设计要点（都是踩过才写下的）

**① 必须带容差。** M1 的噪声底 ≈ **±16pt**；硬比"掉了 2 个点"会**天天误报**，
闸门随即被"狼来了"淹没 —— 比没有更糟。故：
    默认容差 `0.02`；噪声大的指标**逐个显式放宽**（见 `TOL`，每项写明理由）。

**② 方向不能一律"越高越好"。** `qa.reader_offlabel`（走了非生产读取路径的题数）
**越低越好**；`retrieval.gold_map_failed` 同理。一律按"降了就红"会把
**修好了**也判成回退。故每个指标有 `direction`（默认 `up`，见 `DOWN`）。

**③ 换了口径就不许静默比。** 同样的 `(层, 名, 指标)`，若 `note`（口径）变了，
说明**不是同一个东西** —— 只提示、不自动判回退，要人确认后 `--update`。
（注意：note 里含"分母/失败条数"这类每次都会变的细节，所以此处**只提示不拦**。）

## 用法

    python evals/checks/metrics_ratchet.py            # 与基线比（有回退 → 退出码 1）
    python evals/checks/metrics_ratchet.py --update   # 用最新一次结果**更新基线**
    python evals/checks/metrics_ratchet.py --dir X    # 指定报告目录（测试用）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
import report as _rep  # noqa: E402

BASELINE = ROOT / "evals" / "baselines" / "metrics.json"

# 默认容差：2 个点。低于这个幅度的波动**不值得报警**。
DEFAULT_TOL = 0.02

# ★ 噪声大的指标逐个放宽 —— **每项都必须写明理由**（否则放宽就成了掩盖回退）。
TOL: dict[str, float] = {
    "qa.kind_m1": 0.16,      # M1 噪声底 ≈ ±16pt（5 组 164 题实测）；硬比必误报
    "qa.ok_b": 0.10,         # B 组题量小（每组 13~26 题），单题翻转就动几个点
    "qa.kind_m0": 0.10,
    "qa.kind_x5": 0.10,
    "qa.ok_a": 0.05,
    "r2.reader_f1@10": 0.05,  # reader 判定受判官采样影响
    "qa.avg_prompt_tokens": 0.10,   # 按**相对**幅度比（token 数上万，2 个点没意义）
    "qa.cost_cny": 0.15,
}

# ★ **越低越好**的指标（健康度）。一律按 `up` 处理会把"修好了"判成回退。
DOWN: frozenset[str] = frozenset({
    "qa.reader_offlabel",       # 走了非生产读取路径的题数；应为 0
    "qa.validator_rewritten",   # 被闸门改写/拦下的题数
    "qa.parse_degraded",        # 解析降级题数
    "retrieval.gold_map_failed",  # gold 定位失败条数；应为 0
})

# ★ **后缀**规则：任何 `<前缀>.unanswered` 都"越低越好"（空预测题数）。
#   为什么用后缀而不是逐个列名：LoFT 有五个任务、各自一个前缀
#   （`qampari` / `loft_rag` / `loft_retrieval` / `loft_sql` / `loft_icl`），
#   逐个列名**必然漏**，而漏的后果是"空预测变多"被判成"变好"。
_DOWN_SUFFIX: tuple[str, ...] = (".unanswered",)

# 按**相对**幅度判的指标（绝对值大到 2 个点没意义）
RELATIVE = frozenset({"qa.avg_prompt_tokens", "qa.cost_cny"})


def key_of(rec: dict[str, Any]) -> str:
    return f"{rec.get('layer')}|{rec.get('name')}|{rec.get('metric')}"


def tol_of(metric: str) -> float:
    return TOL.get(metric, DEFAULT_TOL)


def direction_of(metric: str) -> str:
    if metric in DOWN or metric.endswith(_DOWN_SUFFIX):
        return "down"
    return "up"


def load_baseline(p: Path | None = None) -> dict[str, dict[str, Any]]:
    f = p or BASELINE
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def save_baseline(b: dict[str, dict[str, Any]], p: Path | None = None) -> Path:
    f = p or BASELINE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(sorted(b.items())), ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return f


def worse(metric: str, cur: float, base: float) -> float:
    """返回"恶化幅度"（正数 = 变差）；`up` 指标变小、`down` 指标变大都是恶化。"""
    if direction_of(metric) == "down":
        return cur - base
    return base - cur


def exceed(metric: str, gap: float, base: float) -> bool:
    """恶化幅度是否**超出容差**（相对类按比例判）。"""
    if metric in RELATIVE:
        return gap > tol_of(metric) * max(abs(base), 1e-9)
    return gap > tol_of(metric)


def compare(baseline: dict[str, dict[str, Any]], recs: list[dict[str, Any]],
            ) -> tuple[list[str], list[str], list[str]]:
    """返回 `(回退列表, 提示列表, 已通过数)`。"""
    latest = {key_of(r): r for r in _rep._latest(recs)}
    regressions: list[str] = []
    notices: list[str] = []
    ok = 0
    for k, b in sorted(baseline.items()):
        cur = latest.get(k)
        metric = str(b.get("metric") or k.split("|")[-1])
        if cur is None:
            notices.append(f"（无数据）{k} —— 基线有、本次没跑；跳过")
            continue
        v, bv = float(cur["value"]), float(b["value"])
        if b.get("note") and cur.get("note") and b["note"] != cur["note"]:
            notices.append(
                f"⚠️ **口径可能变了**：{k}\n      基线：{b['note']}\n      本次：{cur['note']}")
        gap = worse(metric, v, bv)
        if exceed(metric, gap, bv):
            arrow = "↑" if direction_of(metric) == "down" else "↓"
            regressions.append(
                f"{k}\n      基线 {bv:.4f} → 本次 {v:.4f}"
                f"（{arrow} {abs(gap):.4f}，容差 {tol_of(metric)}"
                f"{'（相对）' if metric in RELATIVE else ''}，"
                f"方向 {'越低越好' if direction_of(metric) == 'down' else '越高越好'}）")
        else:
            ok += 1
    return regressions, notices, ok


def main() -> int:
    ap = argparse.ArgumentParser(description="指标棘轮：掉出容差就报错")
    ap.add_argument("--update", action="store_true", help="用最新结果更新基线")
    ap.add_argument("--dir", default="", help="报告目录（默认 evals/reports）")
    args = ap.parse_args()

    recs = _rep.load_all(Path(args.dir) if args.dir else None)
    latest = _rep._latest(recs)
    if not latest:
        print("（还没有任何记录 —— 先跑一次某个评测层）")
        return 0

    if args.update:
        b = load_baseline()
        n_new = n_upd = 0
        for r in latest:
            k = key_of(r)
            ent = {"value": float(r["value"]), "metric": str(r["metric"]),
                   "tol": tol_of(str(r["metric"])),
                   "direction": direction_of(str(r["metric"])),
                   "note": str(r.get("note", ""))}
            n_new += k not in b
            n_upd += k in b
            b[k] = ent
        p = save_baseline(b)
        print(f"基线已更新：新增 {n_new} 条 / 覆盖 {n_upd} 条 → {p.relative_to(ROOT)}")
        print(f"  容差：默认 {DEFAULT_TOL}；"
              f"{len(TOL)} 个指标显式放宽；{len(DOWN)} 个「越低越好」")
        return 0

    baseline = load_baseline()
    if not baseline:
        print(f"（基线是空的：{BASELINE.relative_to(ROOT)} —— 跑一次 "
              f"`--update` 建立基线后再比较）")
        return 0

    regressions, notices, ok = compare(baseline, recs)
    print(f"比较 {len(baseline)} 条基线 ｜ 通过 {ok} ｜ 回退 {len(regressions)}"
          f" ｜ 提示 {len(notices)}")
    for n in notices:
        print(f"  {n}")
    if not regressions:
        print("\n结论：**无指标回退** ✓")
        return 0
    print(f"\n✗ **{len(regressions)} 个指标掉出容差**：")
    for r in regressions:
        print(f"  ✗ {r}")
    print("\n处置：先查原因（改动 / 数据 / 环境）；确属噪声才 `--update` 放宽基线。")
    print("⚠️ **不要**为了让闸门变绿而 `--update` —— 那等于关掉它。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
