"""**实验 1｜R2 语料整档直读（S/C ≈ 2）vs 检索+逐篇判定**

## 要回答的问题
架构分流的判据是 `S/C = 语料token / 上下文预算`：
  · 48k（生产 5 篇）→ 直读 OK（已实测）
  · 105k（QAMPARI 128k 档）→ 直读 = 检索（0.700 vs 0.700）→ **检索无增益**
  · 844k（QAMPARI 1m 档）→ 直读崩 → **检索必需**
**105k 与 844k 之间是空白**，而 R2 的 **266k** 正落在里面 → 本实验定位阈值。

上下文预算已实测：`_ctx_limit.py` → deepseek-chat **可塞 ≥810k token**（266k 塞得下）。
→ 所以这是**真正的头对头**（不是"塞不下"）：把 20 篇整档塞进去问"哪些篇做了 X"。

## 对照臂
| 臂 | 出处 | 篇级集合 F1 |
|---|---|---|
| **A 整档直读**（本脚本） | 20 篇全文一次调用 | ← 待测 |
| B 逐篇判定（reader） | `R2_r2reader_b6.csv` | **0.846** |
| C 检索 top-10（无判定） | 同上 `ret_F1` | 0.539 |

口径完全一致：真值 = `_r2_reader.load_gold()`（校准后 30 个真值），指标 = `setpf`。

## 注意力代理
长上下文里"塞进去"≠"读到了"。除答案外，另记 **`n_touched`**（答案里显式提到
`【Pn】` 的篇数，0~20）——若只提到前几篇而答案也崩，说明是注意力衰退而非任务不可解。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fullread.py --limit 3    # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fullread.py
"""
from __future__ import annotations

import importlib.util as _iu
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "tmp"))
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_rd = _load("r2reader", HERE / "tmp" / "_r2_reader.py")
_m = _rd._m
F, setpf = _m.F, _m.setpf
load_gold = _rd.load_gold

CACHE = HERE / "data" / "r2dev" / "clusters"
OUT = HERE / "results"

SYS = """你要判断**同一批若干篇论文**里，**哪几篇**满足某个论断。下面按【P1】【P2】…的顺序给出每篇的全文。

判定纪律（与真值校准口径一致，务必遵守）：
1. 只看给定文本，**不要用先验知识补充**。
2. ⚠️ **区分"本文做的"与"引用他人做的"**：若动作的主语是被引用的他文（`X et al. proposed…`、`Prior work…`）
   且**没有说明本文也做了** → 不算。这是最常见的错法。
3. **列出/提及 ≠ 做了**：只是任务清单、相关工作、评价指标、未来工作里出现该词 → 不算。
4. 中文论断 ↔ 英文原文，注意同义改写（"做了消融" ↔ "we ablate each component"）。

输出格式（**必须严格遵守**）：
- 先按顺序逐篇给结论，每篇一行：`【Pn】是 _理由` 或 `【Pn】否 _理由`（理由 ≤20 字）
- **最后一行**单独输出：`结论篇：P3,P7,P12`（没有符合的就写 `结论篇：无`）
- 不要输出其他任何内容。"""


def build_ctx(docs: list[str], texts: list[str]) -> str:
    parts = []
    for n, (d, t) in enumerate(zip(docs, texts), 1):
        parts.append(f"\n\n{'=' * 88}\n【P{n}】{d}\n{'=' * 88}\n{str(t).strip()}")
    return "\n".join(parts)


def parse_yes(answer: str, docs: list[str]) -> tuple[list[str], int]:
    """从末行 `结论篇：P3,P7` 解析篇集合；返回 (docid 列表, 触达篇数)。"""
    nums: list[int] = []
    for line in reversed(answer.strip().splitlines()):
        if "结论篇" in line:
            body = line.split("结论篇", 1)[1]
            nums = [int(x) for x in re.findall(r"P\s*(\d+)", body)]
            break
    yes = [docs[i - 1] for i in dict.fromkeys(nums) if 1 <= i <= len(docs)]
    touched = len({int(x) for x in re.findall(r"【\s*P\s*(\d+)\s*】", answer)})
    return yes, touched


def call(system: str, user: str, max_tokens: int = 2600) -> str:
    for a in range(3):
        try:
            return llm.chat_text(system, user, temperature=0.0, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            if a == 2:
                return f"ERR {type(e).__name__}: {e}"
            time.sleep(2.0 * (a + 1))
    return ""


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个真值（冒烟）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="r2fullread")
    args = ap.parse_args()

    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    gold_map = load_gold()
    print(f"真值 {len(gold_map)} 个 ｜ 整档直读（20 篇全文一次调用）｜ 模型 {llm.config()[2]}\n")

    tasks: list[dict] = []
    for ci in range(len(meta)):
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
        docs = [str(x) for x in sub["docid"].tolist()]
        texts = [str(x) for x in sub["full_paper"].tolist()]
        ctx = build_ctx(docs, texts)
        nch = sum(len(t) for t in texts)
        print(f"  簇{ci + 1}：{len(docs)} 篇 ｜ {nch:,} 字符 ≈ {nch / 4.94:,.0f} token", flush=True)
        for facet in meta[ci]["usable"]:
            g = gold_map.get((ci + 1, facet))
            if not g or len(g) == len(docs):
                continue
            tasks.append(dict(cluster=ci + 1, facet=facet, gold=g, docs=docs, ctx=ctx,
                              question=f"这 {len(docs)} 篇里，哪些篇{F[facet][1]}？"))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"\n待测真值 **{len(tasks)}** 个 × 1 次调用 = {len(tasks)} 次"
          f"（同簇共享 prompt 前缀 → 大部分命中缓存）\n")

    lock, prog = threading.Lock(), {"n": 0}

    def do_one(t: dict) -> dict:
        llm.reset_usage()
        t0 = time.time()
        ans = call(SYS, f"{t['ctx']}\n\n{'=' * 88}\n【论断】{t['question']}")
        u = llm.usage_stats()
        yes, touched = parse_yes(ans, t["docs"])
        p, r, f1 = setpf(yes, t["gold"], len(yes) or 1)
        with lock:
            prog["n"] += 1
            print(f"  [{prog['n']}/{len(tasks)}] 簇{t['cluster']} {t['facet']:<18}"
                  f"gold={len(t['gold'])} 判yes={len(yes)} touched={touched}/20 "
                  f"F1={f1:.2f} {time.time() - t0:>5.0f}s", flush=True)
        return dict(cluster=t["cluster"], facet=t["facet"], n_gold=len(t["gold"]),
                    n_yes=len(yes), yes=",".join(yes), n_touched=touched,
                    gold=",".join(sorted(t["gold"])),
                    P=p, R=r, F1=f1, seconds=round(time.time() - t0, 1),
                    prompt_tokens=u["prompt_tokens"], cache_hit=u["cache_hit_tokens"],
                    answers=ans)

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for x in ex.map(do_one, tasks):
            rows.append(x)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "answers"} for r in rows])
    df.to_csv(OUT / f"R2_{args.tag}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"R2_{args.tag}_raw.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'=' * 108}\n【实验 1｜整档直读 vs 对照】{len(df)} 个真值 ｜ {time.time() - t0:.0f}s")
    print(f"  {'臂':<34}{'P':>9}{'R':>9}{'F1':>9}")
    print(f"  {'A 整档直读（20 篇全文·本实验）':<34}{df['P'].mean():>9.3f}"
          f"{df['R'].mean():>9.3f}{df['F1'].mean():>9.3f}")

    rd = OUT / "R2_r2reader_b6.csv"
    if rd.exists():
        b = pd.read_csv(rd)
        k = b.set_index(["cluster", "facet"])
        m = df.set_index(["cluster", "facet"]).join(k[["reader_P", "reader_R", "reader_F1",
                                                       "ret_P", "ret_R", "ret_F1"]], how="inner")
        if len(m):
            print(f"  {'B 逐篇判定（reader·同真值）':<34}{m['reader_P'].mean():>9.3f}"
                  f"{m['reader_R'].mean():>9.3f}{m['reader_F1'].mean():>9.3f}")
            print(f"  {'C 检索 top-10（无判定）':<34}{m['ret_P'].mean():>9.3f}"
                  f"{m['ret_R'].mean():>9.3f}{m['ret_F1'].mean():>9.3f}")
            print(f"\n  → 对照样本 {len(m)} 个真值（与 reader 交集）")
    print(f"\n  交付篇数：直读 {df['n_yes'].mean():.2f} ｜ gold {df['n_gold'].mean():.2f}"
          f"（比值 {df['n_yes'].mean() / max(df['n_gold'].mean(), 1e-9):.2f}）")
    print(f"  触达篇数：**{df['n_touched'].mean():.1f}/20**（答案里显式提到【Pn】的篇数）"
          f" ← 长上下文注意力代理")
    print(f"  用量：均 prompt {df['prompt_tokens'].mean():,.0f} tok"
          f" ｜ **缓存命中率 {df['cache_hit'].sum() / max(df['prompt_tokens'].sum(), 1):.0%}**")
    print(f"\n已写 {OUT / f'R2_{args.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
