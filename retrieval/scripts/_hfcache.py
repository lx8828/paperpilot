"""HF 缓存定位兜底 + 离线模式开关。

背景一（缓存位置）：模型缓存已从 C 盘迁到 `F:\\hf_cache`（C 盘只有 29 GB，
装不下后续扩容）。`setx HF_HOME` 只对**新登录的会话**生效，已在运行的 shell/IDE
继承的是旧环境，于是脚本会去 C 盘找不到模型 → `_raise_on_head_call_error`。

背景二（网络）：本机 **`huggingface.co` 不可达**（`WinError 10060`：连接方在一段
时间后没有正确答复）。而权重、tokenizer、config **都已在本地缓存里** —— 但
`SentenceTransformer` 加载时仍会对每个模型发 `HEAD` 请求去"核对"
`adapter_config.json` / `processor_config.json`，失败后 **Retry 1/5 … 5/5**，
表现为**卡死几分钟**后报错。这看起来像"连接尝试失败"，其实是 Hub 探测。

所以在**导入 sentence_transformers / transformers 之前**调用 `ensure_hf_home()`：
  1. 按"已设 → F 盘 → 默认位置"挑一个真实存在的缓存根；
  2. 缓存存在 → **顺带切离线**（`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`），
     跳过全部网络探测。

逃生口：如果确实需要联网下载（缓存里没有的新模型），显式设
`HF_HUB_OFFLINE=0` 再跑，本模块不会覆盖它。
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT = Path.home() / ".cache" / "huggingface"
PREFERRED = Path("F:/hf_cache")


def ensure_hf_offline() -> bool:
    """切到离线模式（幂等）。返回是否已设为离线。

    ⚠️ 必须在 `transformers` / `sentence_transformers` 被导入**之前**调用 ——
    `huggingface_hub` 在 import 期就把这些开关读成常量，之后再设无效。
    """
    if os.environ.get("HF_HUB_OFFLINE") == "0":       # 显式要求联网 → 尊重
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    return True


def ensure_hf_home() -> str:
    """挑一个真实存在的缓存根；**只要找到了就顺手切离线**。返回缓存根（可能为空）。"""
    cur = os.environ.get("HF_HOME")
    home = cur if (cur and (Path(cur) / "hub").exists()) else ""
    if not home:
        for cand in (PREFERRED, DEFAULT):
            if (cand / "hub").exists():
                os.environ["HF_HOME"] = str(cand)
                home = str(cand)
                break
        else:
            home = cur or ""
    if home:                       # 本地有缓存 → 没必要联网探测
        ensure_hf_offline()
    return home
