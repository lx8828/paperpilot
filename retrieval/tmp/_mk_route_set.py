# -*- coding: utf-8 -*-
"""造题：50（报告层可答）+ 50（必须读全文）—— 用于验证分层设计（2026-10-13）。

## 设计（问答需客观可验，别用 LLM 自证）
· **A 组「只需标题+摘要」**：材料 = **L0 实际所见**（`router.load_l0()` → title/overview/core_points/limitations）
  生成问题 + gold + **evidence 原文片段**；★ 客观校验：evidence 归一化后必须是**该材料**的子串。
· **B 组「必须读全文」**：材料 = MinerU 全文块。挑**不在报告层里**的段落出题；
  ★ 客观校验：evidence 在**全文**里 ✓ 且**不在报告层材料**里 ✓。

只有客观校验通过的题才留下 → **标签不是猜的**。
用法：`python retrieval/tmp/_mk_route_set.py [--limit N]`（N 篇论文；不传=全部 45 篇）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from paperpilot.components import router as R  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path(__file__).resolve().parent / "_route_set.jsonl"
WS = re.compile(r"\s+")
norm = lambda s: WS.sub("", str(s or "")).lower()          # noqa: E731

SYS_A = ("你是出题人。给你一份论文的**摘要级材料**（标题/概述/核心要点/局限）。"
         "请出一题**只靠这份材料就能完整作答**的问题（答案必须能在材料里逐字找到）。"
         "只输出 JSON：{\"question\": \"...\", \"gold\": \"...\", "
         "\"evidence\": \"材料里支持答案的**原文片段，必须逐字照抄**，20~80 字\"}。"
         "问题用中文，要具体（问事实/数值/主张），不要问\"这篇讲什么\"。")
SYS_B = ("你是出题人。给你一段论文**正文**（不在摘要级材料里）。"
         "请出一题**必须读这段正文才能答**的问题（答案只在这段正文里）。"
         "只输出 JSON：{\"question\": \"...\", \"gold\": \"...\", "
         "\"evidence\": \"正文里支持答案的**原文片段，必须逐字照抄**，20~80 字\"}。"
         "问题用中文，要具体（问方法/设置/数值/结论）。"
         "★ 严禁出「摘要里已经写了的同一事实」——换个说法/换个单位也算同一事实。")

# ★ 独立标签校验：**用裁判模型（GLM，与出题的 DeepSeek 不同族）** 判断"只给 L0 材料能否完整回答"
SYS_V = ("你是严格的判据校验员。只依据给定【材料】判断：能否**完整、准确**回答【问题】？"
         "★ 材料里如果已含同一事实（哪怕换成不同说法、不同单位、不同语言），也算「能答」。"
         "只输出 JSON：{\"enough\": true/false, \"why\": \"一句话\"}。")


def validate(mat: str, q: str, want: bool) -> tuple[bool, str]:
    """返回 (是否与构造标签一致, 校验模型给的理由)。"""
    try:
        raw = llm.judge_json(SYS_V, f"【材料】\n{mat}\n\n【问题】\n{q}",
                             temperature=0.0, max_tokens=200)
    except Exception as e:  # noqa: BLE001
        return False, f"校验失败 {type(e).__name__}"
    ok = bool(raw.get("enough")) if isinstance(raw, dict) else False
    why = str(raw.get("why") or "")[:60] if isinstance(raw, dict) else ""
    return (ok == want), f"校验模型判 enough={ok}（期望 {want}）：{why}"


def l0_material(stem: str) -> str:
    """L0 实际所见（与 Router 同源）。"""
    st = R.load_l0([f"{stem}.pdf"])
    cp = st.get("core_points") or []
    lim = st.get("limitations") or []
    txt = [str(st.get("title") or ""), str(st.get("overview") or "")]
    for x in cp + lim:
        txt.append(str(x.get("text") if isinstance(x, dict) else x))
    return "\n".join(t for t in txt if t)


def full_paras(stem: str) -> list[str]:
    d = ROOT / "assets/artifacts/out_mineru" / stem
    cl = next(iter(d.rglob("*content_list*.json")), None)
    if not cl:
        return []
    arr = json.loads(cl.read_text(encoding="utf-8"))
    out = []
    for x in arr:
        t = str(x.get("text") or "").strip()
        if len(t) >= 200 and x.get("type") in (None, "text", "paragraph"):
            out.append(t)
    return out


def ask(sys_p: str, user: str) -> dict | None:
    try:
        raw = llm.chat_json(sys_p, user, temperature=0.4, max_tokens=500)
        return raw if isinstance(raw, dict) else None
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] {type(e).__name__}: {str(e)[:60]}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per", type=int, default=2, help="每篇每组出几个候选")
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))

    rep = {p.name[:-len(".report.json")] for p in (ROOT / "assets/artifacts/out_views").glob("*.report.json")}
    minr = {p.name for p in (ROOT / "assets/artifacts/out_mineru").iterdir() if p.is_dir()}
    stems = sorted(rep & minr)
    if args.limit:
        stems = stems[:args.limit]
    print(f"  论文 = {len(stems)} 篇（报告∩全文）")

    recs = []
    seen: set[str] = set()
    for i, stem in enumerate(stems, 1):
        mat = l0_material(stem)
        paras = full_paras(stem)
        nm = norm(mat)
        # ── A 组：报告层可答（证据∈L0 材料 + 独立模型判「能答」）──
        for k in range(args.per):
            a = ask(SYS_A, f"【摘要级材料】\n{mat}"
                    + ("\n\n（请换一个不同的侧面出题，不要再问数据集规模这类）" if k else ""))
            if not (a and norm(a.get("evidence")) and norm(a["evidence"]) in nm):
                print(f"    [{stem}] A{k + 1} 组校验不过（evidence 不在 L0 材料里），丢弃")
                continue
            q = str(a.get("question") or "")
            if norm(q) in seen:
                continue
            ok, why = validate(mat, q, want=True)
            if not ok:
                print(f"    [{stem}] A{k + 1} 组独立校验不过：{why}")
                continue
            seen.add(norm(q))
            recs.append({"qid": f"{stem}#A{k + 1}", "pdf": f"{stem}.pdf", "stem": stem, "grp": "A",
                         "question": q, "gold": str(a.get("gold") or ""),
                         "evidence": str(a["evidence"]),
                         "verified": "evidence∈L0材料 + 独立校验判能答",
                         "l0_chars": len(mat)})
        # ── B 组：必须读全文（证据∈正文且∉L0 + 独立模型判「不能答」）──
        cand_all = [p for p in paras if norm(p) and norm(p)[:120] not in nm]
        for k in range(args.per):
            cand = cand_all[k * 6:(k + 1) * 6] or cand_all[:6]
            if not cand:
                break
            body = "\n\n".join(cand)
            b = ask(SYS_B, f"【正文片段】\n{body}")
            if not (b and norm(b.get("evidence")) and norm(b["evidence"]) in norm(body)
                    and norm(b["evidence"]) not in nm):
                print(f"    [{stem}] B{k + 1} 组校验不过，丢弃")
                continue
            q = str(b.get("question") or "")
            if norm(q) in seen:
                continue
            ok, why = validate(mat, q, want=False)      # ★ L0 材料必须答不了
            if not ok:
                print(f"    [{stem}] B{k + 1} 组独立校验不过（L0 其实能答）：{why}")
                continue
            seen.add(norm(q))
            recs.append({"qid": f"{stem}#B{k + 1}", "pdf": f"{stem}.pdf", "stem": stem,
                         "grp": "B", "question": q, "gold": str(b.get("gold") or ""),
                         "evidence": str(b["evidence"]),
                         "verified": "evidence∈正文且∉L0材料 + 独立校验判不能答",
                         "l0_chars": len(mat)})

        print(f"  [{i}/{len(stems)}] {stem}  L0 {len(mat)} 字符 ｜ 正文段 {len(paras)} "
              f"｜ 累计 {len(recs)} 题", flush=True)
        OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")

    from collections import Counter
    print(f"\n  完成：{len(recs)} 题 ｜ 分组 {dict(Counter(r['grp'] for r in recs))} → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
