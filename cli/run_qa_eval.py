"""QA 验证：PaperReport 结构化分层是否支撑真实问答（上下文充足性对照实验）。

对 qa/qa_set.json 每个问题跑分档上下文：
  L1 top    仅展示层（overview/core_points/limitations）
  L2 claims L1 + hint_groups 的完整组信息（模拟 claim retrieval 命中）
  L3 chunk  L2 + 相关 chunk 原文（仅 K 类问题跑）

观察：R 类应在 L1 答好；C 类 L1 不足、L2 答好；K 类 L1/L2 不足、L3 答好。
若某类在任何一档都答不好 → 结构化层对该类问题无价值 / 检索或提取有缺口。

用法：
    uv run python cli/run_qa_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.tools import analyzer, llm
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.pdf_parser import parse_pdf

ROOT = Path(__file__).resolve().parents[1]
VIEW_DIR = ROOT / "assets/artifacts/out_views"
PAPERS_DIR = ROOT / "assets" / "papers"
QA_FILE = ROOT / "qa" / "qa_set.json"
OUT_FILE = ROOT / "qa" / "qa_eval_out.json"

SYSTEM = (
    "你是严谨的论文问答助手。只能依据提供的『论文信息』作答；"
    "如果信息不足以准确回答，请明确回答：信息不足：缺少<具体缺什么>。"
    "绝不编造数字、方法或结论。回答 2~5 句，涉及数字时给出出处页码（pX）。"
)

MAX_CHUNK_CHARS = 2600


def _load_report(pdf: str) -> dict[str, Any]:
    return json.loads((VIEW_DIR / f"{Path(pdf).stem}.report.json").read_text(encoding="utf-8"))


def top_ctx(r: dict[str, Any]) -> str:
    lines = [f"标题：{r['title']}", "", "=== 概述 ===", r["overview"], ""]
    lines.append("=== 核心要点 ===")
    for c in r.get("core_points", []):
        lines.append(f"- [{c['label']}] {c['text']}")
    lines.append("")
    if r.get("limitations"):
        lines.append("=== 局限 ===")
        for c in r["limitations"]:
            lines.append(f"- {c['text']}")
    return "\n".join(lines)


def claim_ctx(r: dict[str, Any], gids: list[str]) -> str:
    """模拟 claim retrieval 命中的组（完整信息 + 代表 claim 的原文证据）。"""
    groups = {g["group_id"]: g for g in r.get("groups", [])}
    claims = {c["claim_id"]: c for c in r.get("claims", [])}
    lines = []
    for gid in gids:
        g = groups.get(gid)
        if not g:
            lines.append(f"（未找到组 {gid}）")
            continue
        lines.append(f"- [{g['label']}] imp={g['importance']} {g['rep_text']}")
        if g.get("sections"):
            lines.append(f"  出现章节: {'; '.join(g['sections'][:3])}")
        rep = claims.get(g.get("rep_claim_id", ""))
        if rep and rep.get("evidence_quote"):
            lines.append(f"  原文证据(p{rep.get('page','?')}): {rep['evidence_quote']}")
        lines.append("")
    return "\n".join(lines).strip()


def find_chunk(pdf: str, kw: str) -> str | None:
    """定位含关键词的正文 chunk（返回文本）。"""
    result = parse_pdf(str(PAPERS_DIR / pdf))
    chunks = chunk_document(result["blocks"], max_len=4000)
    for c in analyzer.extractable(chunks):
        if kw and kw.casefold() in c.text.casefold():
            return c.text[:MAX_CHUNK_CHARS]
    return None


def ask(ctx: str, q: str) -> str:
    return llm.chat_text(SYSTEM, f"===== 论文信息 =====\n{ctx}\n\n===== 问题 =====\n{q}",
                         temperature=0.0)


def main() -> int:
    llm._load_dotenv(str(ROOT))
    data = json.loads(QA_FILE.read_text(encoding="utf-8"))
    if not llm.is_configured():
        print("[错误] LLM 未配置")
        return 1

    results = []
    for q in data["questions"]:
        pdf = q["pdf"]
        r = _load_report(pdf)
        qid = q["qid"]
        print("=" * 80)
        print(f"[{qid}] ({q['level']}) {q['question']}")
        print(f"  ground-truth 提示: {q['note']}")

        tctx = top_ctx(r)
        item = {"qid": qid, "level": q["level"], "question": q["question"],
                "note": q["note"], "answers": {}}
        try:
            a1 = ask(tctx, q["question"])
            item["answers"]["L1_top"] = a1
            print(f"\n  ── L1 top ──\n  {' '.join(a1.split())[:220]}")
        except llm.LLMError as e:
            item["answers"]["L1_top"] = f"[LLMError] {e}"
            print(f"  L1 调用失败: {e}")

        cctx = claim_ctx(r, q.get("hint_groups") or [])
        ctx2 = (tctx + "\n\n=== 检索命中的主张（含原文证据） ===\n" + cctx
                if cctx.strip() else tctx + "\n\n（无相关主张命中）")
        try:
            a2 = ask(ctx2, q["question"])
            item["answers"]["L2_claims"] = a2
            print(f"\n  ── L2 claims ──\n  {' '.join(a2.split())[:220]}")
        except llm.LLMError as e:
            item["answers"]["L2_claims"] = f"[LLMError] {e}"
            print(f"  L2 调用失败: {e}")

        if q["level"] == "K" and q.get("chunk_kw"):
            chunk = find_chunk(pdf, q["chunk_kw"])
            if chunk:
                ctx3 = ctx2 + "\n\n=== 正文片段 ===\n" + chunk
                try:
                    a3 = ask(ctx3, q["question"])
                    item["answers"]["L3_chunk"] = a3
                    print(f"\n  ── L3 chunk（kw={q['chunk_kw']}）──\n  {' '.join(a3.split())[:220]}")
                except llm.LLMError as e:
                    item["answers"]["L3_chunk"] = f"[LLMError] {e}"
                    print(f"  L3 调用失败: {e}")
            else:
                item["answers"]["L3_chunk"] = "(未定位到含关键词 chunk)"
                print(f"  L3: 未定位到含 '{q['chunk_kw']}' 的 chunk")
        results.append(item)
        print()

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
