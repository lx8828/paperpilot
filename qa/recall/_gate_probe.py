"""临时：复现失败题，打印 Validator 闸门的完整判定（触发类型/证据），判断是否误杀。"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve() / "src"))
os.environ["PAPERPILOT_USE_MINERU"] = "1"

from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(".")
QIDS = ["mh-29290-q1", "mh-29290-q2", "mh-28447-q1"]
items = {x["qid"]: x for x in
         json.loads(Path("qa/recall/mineru_hard_set_v1.json").read_text(encoding="utf-8"))["items"]}
for qid in QIDS:
    it = items[qid]
    llm.reset_usage()
    r = graph_ask(it["question"], f"{it['pid']}.pdf")
    v = r.get("validator") or {}
    dbg = r.get("debug") or {}
    print("=" * 88)
    print(f"## {qid} ({it['pid']}) level={(dbg.get('answer') or {}).get('level')} "
          f"action={v.get('action')}")
    print(f"Q: {it['question']}")
    print("issues:")
    for i in (v.get("issues") or []):
        print(f"   - sev={i.get('sev')} type={i.get('type')} sent={str(i.get('sentence'))[:90]!r}")
        print(f"     detail={str(i.get('detail'))[:220]}")
    print(f"answer(闸门后): {(r.get('answer') or '')[:300]}")
    pre = v.get("pre_answer") or v.get("draft") or ""
    if pre:
        print(f"answer(闸门前草稿): {str(pre)[:600]}")
