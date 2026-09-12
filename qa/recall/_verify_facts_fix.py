"""验证两步法第一步两处修复（1.1 删负向提示 / 1.2 宽松解析兜底）。"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.nodes import answer as A  # noqa: E402
from paperpilot.agents.nodes.pull_chunk import search_l3  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

ok = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global ok
    ok = ok and cond
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


def main() -> int:
    # ── 1.1 空清单不再注入负向提示 ──
    empty_block = A._fmt_facts_block([])
    check("1.1 空清单 → 空串（无负向提示）", empty_block == "", f"got={empty_block!r}")
    nb = A._fmt_facts_block([{"n": 1, "text": "fact A"}, {"n": 0, "text": "fact B"}])
    check("1.1 非空清单仍生成块", ("fact A" in nb and "fact B" in nb
                                  and "未提取到" not in nb and nb.startswith("\n\n")))

    # ── 1.2 宽松解析兜底（合成：含未转义双引号，严格 json 必失败）──
    broken = ('{"facts": [{"n": 1, "text": "We use the "syntactic" dataset."}, '
              '{"n": 2, "text": "Another fact with 42% gain."}]}')
    strict_failed = False
    try:
        llm._parse_json(broken)
    except Exception:  # noqa: BLE001
        strict_failed = True
    check("1.2 合成样本确实让严格解析失败（前置条件）", strict_failed)
    sal = A._salvage_facts(broken)
    check("1.2 宽松解析救回 2 条", len(sal) == 2, f"got={len(sal)} {sal}")
    if len(sal) == 2:
        check("1.2 文本还原正确",
              sal[0]["text"] == 'We use the "syntactic" dataset.'
              and sal[1]["text"] == "Another fact with 42% gain."
              and sal[0]["n"] == 1 and sal[1]["n"] == 2, f"{sal}")

    # ── 真实失效题端到端（1 次 LLM）──
    llm._load_dotenv(str(ROOT))
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")[:8]] = (pid, q)
    got = qmap.get("3de27c81")
    if got:
        pid, q = got
        st = search_l3({"question": q["question"], "pdf": f"qasper_{pid}.qpdf"})
        parts = ["标题：\n\n=== 全文检索到的正文（独立检索） ==="]
        for j, c in enumerate(st.get("l3_chunks") or [], 1):
            parts.append(A._fmt_chunk_entry(j, c))
        facts = A._extract_facts(q["question"], "\n\n".join(parts))
        check("真实失效题 3de27c81 现在能取到事实", len(facts) > 0, f"facts={len(facts)}")
        for f in facts[:5]:
            print(f"    ({f['n']}) {f['text'][:110]}")

    print("\n==> 全部通过" if ok else "\n==> 有 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
