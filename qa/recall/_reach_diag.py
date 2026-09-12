"""可达性诊断（0 元，纯解析，不调 LLM）：
对 12 道"金标准在表格里"的题，检查 gold 里的数字**在各臂检索文本中是否出现**。

意义：三臂 pass 全 100% 属于天花板，无法分辨 MinerU 的收益。本诊断回答更本质的问题——
**pymupdf 的文本里本来就有这些表值吗？** 若有 → MinerU 对这类题是冗余的（至少在这 10 篇）；
若没有 → MinerU 是必需的，天花板只是"模型绕过去了"。

用法（必须分进程，document_cache 有 lru_cache）：
    $env:PP_ARM='plain';  uv run python qa/recall/_reach_diag.py
    $env:PP_ARM='inject'; uv run python qa/recall/_reach_diag.py
    $env:PP_ARM='mineru'; uv run python qa/recall/_reach_diag.py
"""
from __future__ import annotations
import io
import json
import os
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(".").resolve() / "src"))

ARM = os.environ.get("PP_ARM", "plain")
ENV = {"plain": {"PAPERPILOT_USE_MINERU": "0", "PAPERPILOT_MINERU_INJECT": "0"},
       "inject": {"PAPERPILOT_USE_MINERU": "0", "PAPERPILOT_MINERU_INJECT": "1"},
       "mineru": {"PAPERPILOT_USE_MINERU": "1", "PAPERPILOT_MINERU_INJECT": "1"}}[ARM]
os.environ.update(ENV)

from paperpilot.agents.document_cache import retrieval_chunks, current_source  # noqa: E402

NUM = re.compile(r"\d+(?:[.,]\d+)*%?")


def norm_nums(text: str) -> set[str]:
    out = set()
    for t in NUM.findall(text or ""):
        t = t.rstrip("%").replace(",", "")
        try:
            f = float(t)
        except ValueError:
            continue
        # 整数与小数值分开保留；同时保留原始串以做粗匹配
        out.add(t)
    return out


def main() -> int:
    items = json.loads(Path("qa/recall/mineru_hard_set_v1.json").read_text(encoding="utf-8"))["items"]
    table = [x for x in items if "Table" in (x.get("src") or "")]
    pids = list(dict.fromkeys(x["pid"] for x in table))
    cache: dict[str, str] = {}
    for p in pids:
        try:
            cache[p] = "\n".join(c.text for c in retrieval_chunks(f"{p}.pdf"))
        except Exception as e:  # noqa: BLE001
            cache[p] = f"__ERR__ {type(e).__name__}: {e}"

    tot = hit = 0
    rows = []
    for x in table:
        text = cache.get(x["pid"], "").replace(",", "")  # 去千分位，避免 "283,047" 匹不上 "283047"
        nums = norm_nums(x["gold"])
        nums = {n for n in nums if len(n) >= 2}  # 丢掉 "1""2" 这类无鉴别力的
        if not nums:
            continue
        found = {n for n in nums if n in text}
        tot += len(nums)
        hit += len(found)
        miss = sorted(nums - found)
        rows.append({"qid": x["qid"], "pid": x["pid"], "src": x.get("src", ""),
                     "n_nums": len(nums), "n_hit": len(found), "miss": miss[:6]})

    per_q = sum(1 for r in rows if r["n_hit"] == r["n_nums"])
    print(json.dumps({"arm": ARM, "source": current_source(f"{pids[0]}.pdf"),
                      "chunks_pids": len(pids),
                      "gold_num_hit": f"{hit}/{tot}",
                      "questions_all_nums_present": f"{per_q}/{len(rows)}",
                      "detail": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
