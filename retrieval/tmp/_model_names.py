"""模型名有效性复核（修正上一版的两个 bug：推理模型 token 预算过小、glm 走错端点）。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))

CASES = [
    ("PAPERPILOT_LLM", "deepseek-chat"),
    ("PAPERPILOT_LLM", "deepseek-flash"),
    ("PAPERPILOT_LLM", "deepseek-reasoner"),
    ("PAPERPILOT_LLM", "deepseek-v4-flash"),
    ("PAPERPILOT_LLM", "deepseek-v4-pro"),
    ("PAPERPILOT_JUDGE", "glm-4-flash"),
    ("PAPERPILOT_JUDGE", "glm-4.7-flash"),
    ("PAPERPILOT_JUDGE", "glm-4-flash-250414"),
]

print(f"{'端点':<18}{'模型名':<24}{'结果':<10}说明")
for prefix, model in CASES:
    env = f"{prefix}_MODEL"
    saved = os.environ.get(env)
    os.environ[env] = model
    try:
        # 推理模型需要足够预算（否则思考吃光 → content 为空）
        r = llm.chat_json("只输出 JSON。", '返回 {"ok": 1}', temperature=0,
                          max_tokens=1500, prefix=prefix)
        print(f"{prefix:<18}{model:<24}{'✅ 可用':<10}{r}")
    except Exception as e:  # noqa: BLE001
        print(f"{prefix:<18}{model:<24}{'❌':<10}{type(e).__name__}: {str(e)[:80]}")
    finally:
        if saved is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = saved
    time.sleep(0.4)
