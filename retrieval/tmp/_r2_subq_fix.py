"""**修子查询的 3 个已知问题**（只改这 3 个 facet，其余 13 个原样保留）

依据 `R2_SUBQ_FACET_DIAG_20260930.md` §① 的体检结果：

| # | facet | 问题 | 修法 |
|---|---|---|---|
| 1 | `prompt_eng` | **无子查询**（题集里唯一缺的） | 生成 3 条 |
| 2 | `knowledge_distill` | **↔gold 正则重合 62.5%**（≈ 锚点基线 63%，判泄露） | 重新生成 + **提示词禁止照抄论断里的英文示例词** |
| 3 | `case_study` | 两条子查询 **Jaccard 0.667**（几乎同义） | 重新生成 + **强制三侧面不同** |

## 与旧生成器的两点不同
1. **新增硬约束**：不得使用中文题面里 `反引号` / 括号中的**英文原词原句**（那是判定标准，照抄即泄露）；
2. **生成后自动体检 + 不达标重试**（≤3 轮）：
   · `CJK 纯度`：不能含中文（否则该路 BM25 恒 0）
   · `↔gold 正则重合率` < 0.40
   · `两两 Jaccard` 最大 < 0.40
   三轮仍不达标 → 保留**最好的一轮**并**明确报警**。

## 产物
`data/r2dev/subqueries_v2.json`（= 原 13 个 + 修好的 3 个）；**不动** `subqueries.json`。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_subq_fix.py            # 真跑（需 LLM）
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_subq_fix.py --dry-run  # 只看旧值体检
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
DEV = HERE / "data" / "r2dev"
SUBQ = DEV / "subqueries.json"
SUBQ_V2 = DEV / "subqueries_v2.json"
TO_FIX = ["prompt_eng", "knowledge_distill", "case_study"]

STOP = {"the", "a", "an", "of", "for", "in", "on", "to", "and", "with", "via", "using",
        "towards", "toward", "from", "by", "at", "as", "is", "are", "be", "we", "our",
        "this", "that", "it", "can", "not", "but", "or", "its", "their", "than", "then"}

# ★ 比旧版更严：明确禁止照抄论断中的英文示例词（旧版 3 条要求照旧保留）
SYS_NEW = """你是信息检索助手。给定一条关于**某篇论文**的中文论断，生成 3 条**互为补充、措辞不同**的
**英文**检索式，用于在论文正文中找出支持该论断的段落。

要求：
1. 覆盖论断的**不同侧面/不同表述**（例如"做了什么实验""报告了什么指标""在什么数据上做"）；
2. 不要只是把论断里的词堆在一起，要像论文作者会写的句子；
3. ★★ **禁止照抄论断原文里的英文原词原句**（论断中 `反引号` 内、或括号里的英文，如
   `we introduce X dataset`、`inference time`）—— 那些是判定标准，照抄会构成**答案泄露**。
   请改用**同义的其它英文表述**（换动词、换主语、换语序、换同义词）。
4. ★ 三条之间**措辞必须明显不同**，不得出现两条近似同义；
5. 只输出 JSON：{"queries": ["q1", "q2", "q3"]}"""


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2


def toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9\-]{1,}", str(s).lower()) if t not in STOP}


def diag(facet: str, subs: list[str]) -> dict:
    """体检：CJK 纯度 / ↔gold 正则重合 / 两两 Jaccard。"""
    tt = [toks(x) for x in subs]
    pat = toks(F2[facet][0])
    ov = ([len(t & pat) / max(len(pat), 1) for t in tt] or [0.0])
    jac = ([len(tt[i] & tt[j]) / max(len(tt[i] | tt[j]), 1)
            for i in range(len(tt)) for j in range(i + 1, len(tt))] or [0.0])
    return dict(n=len(subs),
                cjk=int(sum(1 for x in subs if any("\u4e00" <= c <= "\u9fff" for c in str(x)))),
                ov=float(np.mean(ov)), ov_max=float(np.max(ov)),
                jac_max=float(np.max(jac)), n_pat=len(pat),
                ok=bool(len(subs) == 3
                        and not any("\u4e00" <= c <= "\u9fff" for c in "".join(map(str, subs)))
                        and float(np.mean(ov)) < 0.40 and float(np.max(jac)) < 0.40))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只看旧值体检，不调 LLM")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args()

    old = json.loads(SUBQ.read_text(encoding="utf-8"))
    print("=" * 116)
    print(f"【修子查询】待修 {len(TO_FIX)} 个 facet：{TO_FIX}")
    print(f"  {'facet':<20}{'条数':>5}{'含中文':>7}{'↔gold正则':>11}{'两两Jaccard':>13}  状态")
    for f in TO_FIX:
        d = diag(f, old.get(f, []))
        why = ("❌ 无子查询" if d["n"] == 0 else
               f"⚠️ 泄露 {d['ov']:.1%}" if d["ov"] >= 0.40 else
               f"⚠️ 冗余 {d['jac_max']:.2f}" if d["jac_max"] > 0.40 else "✅ 已达标")
        print(f"  {f:<20}{d['n']:>5}{d['cjk']:>7}{d['ov']:>11.1%}{d['jac_max']:>13.2f}  {why}")
    print(f"\n  对照（13 个不修的）：均泄露 "
          f"{np.mean([diag(f, old[f])['ov'] for f in old if f not in TO_FIX]):.1%}")
    if args.dry_run:
        print("\n  --dry-run：不调 LLM，退出。")
        return 0

    from paperpilot.tools import llm
    llm._load_dotenv(str(ROOT))
    cache = dict(old)
    print(f"\n【生成】提示词已加入「禁止照抄论断英文示例词」约束 ｜ 最多 {args.attempts} 轮")
    for f in TO_FIX:
        best, best_key = None, None
        for att in range(1, args.attempts + 1):
            try:
                obj = llm.chat_json(SYS_NEW, f"论断：{F2[f][1]}", temperature=args.temperature,
                                    max_tokens=320, prefix="PAPERPILOT_LLM")
                qs = [str(q).strip() for q in ((obj.get("queries") if isinstance(obj, dict)
                                                else obj) or []) if str(q).strip()][:3]
            except Exception as e:  # noqa: BLE001
                print(f"    {f} 第{att}轮失败：{type(e).__name__}: {str(e)[:70]}")
                qs = []
            if len(qs) < 3:
                print(f"    {f} 第{att}轮只回 {len(qs)} 条 → 重试")
                continue
            d = diag(f, qs)
            # 打分：达标优先 → 泄露越低越好 → 互补越好（Jaccard 越低）
            key = (int(d["ok"]), -d["ov"], -d["jac_max"])
            print(f"    {f} 第{att}轮：泄露 {d['ov']:.1%} ｜ Jaccard {d['jac_max']:.2f}"
                  f" ｜ {'✅达标' if d['ok'] else '未达标'}")
            if best_key is None or key > best_key:
                best, best_key = qs, key
            if d["ok"]:
                break
        if best is None:
            best = old.get(f) or [F2[f][2]]          # 兜底：保留旧值 / 锚点词
            print(f"    {f} ⚠️ 全部轮次无效 → 保留旧值")
        cache[f] = best
        db = diag(f, best)
        print(f"  → {f}：泄露 {db['ov']:.1%} ｜ Jaccard {db['jac_max']:.2f}"
              f" ｜ {'✅' if db['ok'] else '⚠️ 仍不达标（已在报告里记录）'}")

    SUBQ_V2.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n【产物】{SUBQ_V2.name}（{len(cache)} 个 facet）"
          f" ｜ **不动** {SUBQ.name}")

    # ── before / after ──
    print("\n" + "=" * 116)
    print("【before → after】")
    print(f"  {'facet':<20}{'泄露 前→后':>17}{'Jaccard 前→后':>17}{'条数 前→后':>13}")
    for f in TO_FIX:
        a, b = diag(f, old.get(f, [])), diag(f, cache[f])
        print(f"  {f:<20}{a['ov']:>7.1%} → {b['ov']:<7.1%}"
              f"{a['jac_max']:>8.2f} → {b['jac_max']:<7.2f}"
              f"{a['n']:>6} → {b['n']:<6}")
    print(f"\n  ★ 目标：泄露 < 40%（对照锚点基线 63%）｜ 两两 Jaccard < 0.40")
    print(f"  样例（修后）：")
    for f in TO_FIX:
        for q in cache[f]:
            print(f"    [{f}] {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
