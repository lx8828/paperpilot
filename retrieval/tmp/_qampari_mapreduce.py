"""**map → 聚合 → 逐候选判定**（v2）：把"一次看 K 篇"拆开

## 为什么（`R2_QAMPARI_RERANK_20260929.md`）
1m 档：`R@40 = 0.954`、重排后 `R@5 = 0.688`，但端到端 `subspan_em` 稳定在 **0.38**，
且**平均预测数只有 3.6（gold 5.02）→ 系统性少列**。瓶颈在 **reader**。

## v1 冒烟暴露的问题（已修）
`reduce`（一次看候选清单 → 要求过滤）**退化成"照原样返回"**：候选 9.8 → 9.6，几乎不压缩。
→ 改成 **逐候选判定**（每个候选单独一次调用，可证伪，不会退化成"返回同一份清单"）。

## 四个臂
| 臂 | 做法 | 调用/题 | 针对 |
|---|---|---|---|
| `single` | 一次看 K 篇直接列（旧做法，基线） | 1 | — |
| `passage_union` | **逐篇**抽取（每篇 1 次）→ 并集 | K | 提前收手 |
| `passage_verified` | 上一臂 + **逐候选判定**剔除假阳性 | K + 候选数 | 近邻干扰 |
| `grp_union` | 把 K 篇按 **5 篇一组**枚举后并集 | K/5 | 提前收手（更省） |

map 提示要求每个答案附**出处句**（可证伪），判定步据此做二元裁决。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_mapreduce.py --limit 5
  ./.venv/Scripts/python.exe -u retrieval/tmp/_qampari_mapreduce.py --k 20 --workers 8
"""
from __future__ import annotations

import importlib.util as _iu
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "tmp"))
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))
spec = _iu.spec_from_file_location("qr", HERE / "tmp" / "_qampari_run.py")
qr = _iu.module_from_spec(spec)
sys.modules["qr"] = qr
spec.loader.exec_module(qr)

OUT = HERE / "results"
OFFICIAL = HERE / "data" / "loft" / "official" / "run_evaluation.py"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
MAX_CAND_VERIFY = 14          # 每题最多判定多少候选（控成本）

SYS_MAP_SPECIFIC = """你在读**一个**文档片段。给定一个查询，只从**这个片段**里找出它**明确给出**的、能作为该查询答案的具体条目。

严格要求：
1. **只依据片段字面**；不要用先验知识补充、不要推理、不要外推。
2. ⚠️ **必须取"最具体、最完整"的写法**：若片段里该条目带有**限定词 / 后缀 / 副标题**
   （例如赛事名 `… Championships – Men's Singles`、书名含副标题、机构名带所在地），
   **要连同限定词一起抽取**，**不要只取主名**。目标是让这个字符串**唯一确定**一个实体。
3. 每个条目必须给出一句**片段中直接说明它的原文**（≤60 字，照抄，不要改写）。
4. 片段里提到但**并不是在回答该查询**的内容（页脚、无关列表、别的实体）不算。
5. 找不到任何答案 → `items` 为空数组。
6. 答案是**实体/名称/数值**这类短字符串，**照抄片段写法**（不翻译、不补全、不改写）。

只输出 JSON：{"items": [{"answer": "...", "why": "片段中的原文"}, ...]}"""


SYS_MAP = """你在读**一个**文档片段。给定一个查询，只从**这个片段**里找出它**明确给出**的、能作为该查询答案的具体条目。

严格要求：
1. **只依据片段字面**；不要用先验知识补充、不要推理、不要外推。
2. 每个条目必须给出一句**片段中直接说明它的原文**（≤60 字，照抄，不要改写）。
3. 片段里提到但**并不是在回答该查询**的内容（页脚、无关列表、别的实体）不算。
4. 找不到任何答案 → `items` 为空数组。
5. 答案是**实体/名称/数值**这类短字符串，照抄片段写法（不翻译、不补全）。

只输出 JSON：{"items": [{"answer": "...", "why": "片段中的原文"}, ...]}"""

SYS_VERIFY = """给定一个查询、一个**候选答案**、以及它在一段论文/文档原文中的出处句。
请判断：**这个候选是否确实回答了该查询**？（而不是仅仅被提到、或属于别的实体、或名称相近但不匹配。）

只输出 JSON：{"keep": true 或 false, "why": "≤20 字"}"""


def call(system: str, user: str, max_tokens: int = 500) -> dict:
    for a in range(3):
        try:
            o = llm.chat_json(system, user, temperature=0.0, max_tokens=max_tokens,
                              prefix="PAPERPILOT_LLM")
            if isinstance(o, dict):
                return o
        except Exception:  # noqa: BLE001
            if a == 2:
                return {}
            time.sleep(1.5 * (a + 1))
    return {}


def items_of(o: dict) -> list[tuple[str, str]]:
    v = o.get("items") or o.get("answers") or []
    if isinstance(v, str):
        v = [v]
    out = []
    for x in v:
        if isinstance(x, dict):
            a = str(x.get("answer") or "").strip()
            w = str(x.get("why") or "").strip()
        else:
            a, w = str(x).strip(), ""
        if a:
            out.append((a, w))
    return out


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--length", default="1m")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--groups", default="3,5,10", help="分组枚举的组大小（可多个，逐个成臂）")
    ap.add_argument("--tag", default="tier1m_mr20")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rerank-from", default="retrieval/data/loft/qampari/1m/rerank_top40.json")
    ap.add_argument("--arms", default="passage_union,passage_verified,grp_union")
    ap.add_argument("--map-style", choices=["plain", "specific"], default="plain",
                    help="specific = 要求抽取片段中**最具体/最完整**的写法（治『predict ⊂ gold』）")
    args = ap.parse_args()
    MAP_SYS = SYS_MAP_SPECIFIC if args.map_style == "specific" else SYS_MAP
    gsizes = [int(x) for x in str(args.groups).split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    grp_arms = [f"grp{g}" for g in gsizes]

    d = HERE / "data" / "loft" / "qampari" / args.length
    corpus = [json.loads(l) for l in (d / "corpus.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    queries = [json.loads(l) for l in (d / "test_queries.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        queries = queries[: args.limit]
    by_pid = {str(c["pid"]): c for c in corpus}
    rr = json.loads((ROOT / args.rerank_from).read_text(encoding="utf-8"))
    print(f"档位 {args.length} ｜ k={args.k} ｜ 组大小 {gsizes} ｜ 题 {len(queries)}"
          f" ｜ 臂 {arms + grp_arms}\n")

    def ptext(pid: str) -> str:
        c = by_pid[pid]
        return f"TITLE: {c.get('title_text')}\nCONTENT: {c.get('passage_text')}"

    prog, lock = {"n": 0}, threading.Lock()

    def do_query(i: int) -> dict:
        q = queries[i]
        pids = [p for p in rr.get(q["qid"], [])[: args.k] if p in by_pid]
        res = {"qid": q["qid"], "n_passage": len(pids)}
        # ── 逐篇 map ──
        per: list[tuple[str, str, str]] = []          # (norm, shown, why)
        if "passage_union" in arms or "passage_verified" in arms:
            for p in pids:
                for a, w in items_of(call(MAP_SYS,
                                          f"查询：{q['query_text']}\n\n文档片段：\n{ptext(p)}")):
                    per.append((qr.normalize_answer(a), a, w))
        # ── 分组 map（每个组大小一个臂）──
        grp_pairs: dict[str, list[tuple[str, str, str]]] = {nm: [] for nm in grp_arms}
        for nm, g in zip(grp_arms, gsizes):
            for s in range(0, len(pids), g):
                blk = "\n\n".join(ptext(p) for p in pids[s: s + g])
                for a, w in items_of(call(MAP_SYS,
                                          f"查询：{q['query_text']}\n\n文档片段：\n{blk}")):
                    grp_pairs[nm].append((qr.normalize_answer(a), a, w))

        def agg(pairs):
            cnt, shown, why = {}, {}, {}
            for k_, a, w in pairs:
                if not k_:
                    continue
                cnt[k_] = cnt.get(k_, 0) + 1
                if k_ not in shown or len(a) > len(shown[k_]):
                    shown[k_] = a
                if w and k_ not in why:
                    why[k_] = w
            return sorted(cnt.items(), key=lambda t: (-t[1], t[0])), shown, why

        cp, sp, wp = agg(per)
        res["passage_union"] = [sp[k_] for k_, _ in cp]
        res["n_cand_passage"] = len(cp)
        for nm in grp_arms:
            cg, sg, _wg = agg(grp_pairs[nm])
            res[nm] = [sg[k_] for k_, _ in cg]
            res[f"n_cand_{nm}"] = len(cg)

        # ── 逐候选判定（在 passage 并集上做）──
        if "passage_verified" in arms and cp:
            top = cp[:MAX_CAND_VERIFY]
            keep = []
            for k_, c in top:
                v = call(SYS_VERIFY, f"查询：{q['query_text']}\n\n候选答案：{sp[k_]}\n\n"
                                     f"出处句（来自文档原文）：{wp.get(k_, '（无）')}")
                if bool(v.get("keep")):
                    keep.append(sp[k_])
            res["passage_verified"] = keep
            res["n_verified"] = len(keep)
        else:
            res["passage_verified"] = []
        with lock:
            prog["n"] += 1
            if prog["n"] % 10 == 0:
                print(f"  {prog['n']}/{len(queries)} …", flush=True)
        return res

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(do_query, range(len(queries))))
    print(f"\n完成 {time.time() - t0:.0f}s")

    df = pd.DataFrame(recs)
    df.to_csv(OUT / f"R2_QAMPARI_{args.tag}_detail.csv", index=False, encoding="utf-8-sig")

    # ── 官方 CLI 出指标 ──
    od = OUT / f"official_{args.tag}"
    od.mkdir(parents=True, exist_ok=True)
    qs = {q["qid"]: q for q in queries}
    with (od / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps(qs[r["qid"]], ensure_ascii=True) + "\n")
    metrics = {}
    for nm in arms + grp_arms:
        pdir = od / nm
        pdir.mkdir(parents=True, exist_ok=True)
        with (pdir / "preds.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for _, r in df.iterrows():
                fh.write(json.dumps({"qid": r["qid"],
                                     "model_outputs": [list(r[nm])], "num_turns": 1},
                                    ensure_ascii=True) + "\n")
        shutil.copy(od / "queries.jsonl", pdir / "queries.jsonl")
        subprocess.run([PY, str(OFFICIAL), "--answer_file_path", str(pdir / "queries.jsonl"),
                        "--pred_file_path", str(pdir / "preds.jsonl"),
                        "--task_type", "multi_value_rag"],
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=str(OFFICIAL.parent))
        mp = pdir / "preds_metrics.json"
        if mp.exists():
            metrics[nm] = json.loads(mp.read_text(encoding="utf-8"))["quality"]

    print(f"\n{'=' * 108}\n【官方指标（{args.length} 档，k={args.k}）】")
    print(f"  {'臂':<18}{'subspan_em':>12}{'em':>9}{'coverage':>11}{'平均预测数':>12}")
    print(f"  {'single（旧·基线）':<18}{0.390:>12.3f}{0.190:>9.3f}{0.5077:>11.4f}{3.75:>12.2f}")
    for nm in arms + grp_arms:
        if nm not in metrics or nm not in df.columns:
            continue
        npred = float(df[nm].map(len).mean())
        m = metrics[nm]
        print(f"  {nm:<18}{m['subspan_em']:>12.3f}{m['em']:>9.3f}{m['coverage']:>11.4f}"
              f"{npred:>12.2f}")
    print(f"\n  **对照 `single`（一次看 k 篇，同重排序列）**："
          f"subspan 0.390 / em 0.190 / coverage 0.5077 / 预测 3.75")
    print(f"  候选数：逐篇并集 {df['n_cand_passage'].mean():.1f} ｜ "
          + " ｜ ".join(f"{nm} {df[f'n_cand_{nm}'].mean():.1f}" for nm in grp_arms)
          + f" ｜ gold 5.02")
    u = llm.usage_stats()
    print(f"\n  调用 {u['calls']} ｜ prompt {u['prompt_tokens']:,} / completion "
          f"{u['completion_tokens']:,}")
    print(f"  产物：{od}/ ｜ {OUT / f'R2_QAMPARI_{args.tag}_detail.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
