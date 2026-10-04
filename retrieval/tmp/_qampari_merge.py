"""**实体级归并**（零额外召回）：把 `grp10` 输出里"同一实体的不同写法"合并成一个

依据（`R2_QAMPARI_MAPREDUCE_20260929.md §3.5`）：
  `grp10` 的 coverage 在「互为子串」口径下从 0.627 → **0.693（+6.6pt）**
  → 多出来的条目**大多是同一实体的变体**（缺限定词 / 音译差异 / 括号补充），**不是错答案**。
  而官方 `em`/`coverage` 用**规范化后完全相等** → 变体会被判错。

## 关键设计：框成"**合并**"而不是"过滤"
v1 的 `reduce`（要求"剔除不正确的"）**退化成原样返回**；且逐候选判定（`passage_verified`）是**负收益**。
所以这里**只要求合并同实体写法**，并**明确禁止删除"看起来不相关"的条目**（那可能是正确的稀有答案）。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_merge.py --workers 8
"""
from __future__ import annotations

import importlib.util as _iu
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))
spec = _iu.spec_from_file_location("qr", HERE / "tmp" / "_qampari_run.py")
qr = _iu.module_from_spec(spec)
sys.modules["qr"] = qr
spec.loader.exec_module(qr)

OUT = HERE / "results"
OFFICIAL = HERE / "data" / "loft" / "official" / "run_evaluation.py"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SRC = OUT / "official_tier1m_mr20" / "grp10" / "preds.jsonl"

SYS = """给定一个查询，以及一份**候选答案清单**。清单里的条目可能有**同一实体的不同写法**
（例如 `Carousel (A Dance)` 与 `carousel`、`2018 Kentucky Bank Tennis Championships` 与
`… – Men's Singles`、`GL-812 HY` 与 `gourdouleseurre gl812 hy`）。

你的任务**只是合并**：把指向**同一个实体**的写法合并成**一个**最完整、最规范的条目。

严格规则：
1. **不要删除**"看起来与查询无关"的条目——那可能是一个正确的稀有答案，**照原样保留**。
2. **不要新增**清单里没有的实体。
3. **不要改写**你保留的条目（可以用清单中某个更完整的写法替换较短的）。
4. 只合并"确定是同一个实体"的（同一名称的不同书写/限定词/音译）；**不确定就不要合并**。
5. 不要解释。

只输出 JSON：{"answers": ["...", "..."]}"""


def call(user: str, max_tokens: int = 800) -> dict:
    for a in range(3):
        try:
            o = llm.chat_json(SYS, user, temperature=0.0, max_tokens=max_tokens,
                              prefix="PAPERPILOT_LLM")
            if isinstance(o, dict):
                return o
        except Exception:  # noqa: BLE001
            if a == 2:
                return {}
            time.sleep(1.5 * (a + 1))
    return {}


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--tag", default="tier1m_mr20_merged")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--group", default="grp10")
    args = ap.parse_args()

    d = HERE / "data" / "loft" / "qampari" / "1m"
    queries = {json.loads(l)["qid"]: json.loads(l)
               for l in (d / "test_queries.jsonl").read_text(
                   encoding="utf-8").splitlines() if l.strip()}
    preds = {json.loads(l)["qid"]: json.loads(l)["model_outputs"][0]
             for l in Path(args.src).read_text(encoding="utf-8").splitlines() if l.strip()}
    items = list(preds.items())
    if args.limit:
        items = items[: args.limit]
    print(f"输入 {len(items)} 题（来自 {Path(args.src).name}）｜ 平均候选 "
          f"{sum(len(v) for _, v in items) / len(items):.2f}\n")

    def work(it):
        qid, cand = it
        q = queries[qid]
        lst = "\n".join(f"- {x}" for x in cand)
        o = call(f"查询：{q['query_text']}\n\n候选答案清单：\n{lst}")
        v = o.get("answers")
        if isinstance(v, str):
            v = [v]
        out = [str(x).strip() for x in (v or []) if str(x).strip()]
        return qid, (out or list(cand)), len(cand)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        got = list(ex.map(work, items))
    print(f"归并完成 {time.time() - t0:.0f}s")
    n_before = sum(a for _q, _p, a in got)
    n_after = sum(len(p) for _q, p, _a in got)
    print(f"候选数：{n_before / len(got):.2f} → **{n_after / len(got):.2f}**"
          f"（压缩 {n_before / len(got) - n_after / len(got):.2f} 条/题；gold 5.02）")

    od = OUT / f"official_{args.tag}"
    (od / args.group).mkdir(parents=True, exist_ok=True)
    pd_ = od / args.group
    with (pd_ / "preds.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for qid, pr, _a in got:
            fh.write(json.dumps({"qid": qid, "model_outputs": [list(pr)], "num_turns": 1},
                                ensure_ascii=True) + "\n")
    src_q = OUT / "official_tier1m_mr20" / "queries.jsonl"
    shutil.copy(src_q, pd_ / "queries.jsonl")
    subprocess.run([PY, str(OFFICIAL), "--answer_file_path", str(pd_ / "queries.jsonl"),
                    "--pred_file_path", str(pd_ / "preds.jsonl"),
                    "--task_type", "multi_value_rag"],
                   capture_output=True, text=True, encoding="utf-8",
                   cwd=str(OFFICIAL.parent))
    m = json.loads((pd_ / "preds_metrics.json").read_text(encoding="utf-8"))["quality"]
    print(f"\n{'=' * 96}\n【官方指标（1m 档，k=20，重排）】")
    print(f"  {'臂':<22}{'subspan_em':>12}{'em':>9}{'coverage':>11}{'预测数':>9}")
    print(f"  {'single（旧基线）':<22}{0.390:>12.3f}{0.190:>9.3f}{0.5077:>11.4f}{3.75:>9.2f}")
    print(f"  {'grp10（无归并）':<22}{0.390:>12.3f}{0.250:>9.3f}{0.6742:>11.4f}{4.63:>9.2f}")
    print(f"  {'**grp10 + 归并**':<22}{m['subspan_em']:>12.3f}{m['em']:>9.3f}"
          f"{m['coverage']:>11.4f}{n_after / len(got):>9.2f}")
    u = llm.usage_stats()
    print(f"\n  调用 {u['calls']} ｜ prompt {u['prompt_tokens']:,} / completion "
          f"{u['completion_tokens']:,}")
    print(f"  产物：{pd_}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
