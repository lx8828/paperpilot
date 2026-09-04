"""OpenAI 兼容 Chat Completions 客户端（纯 stdlib，无第三方依赖）。

配置（三选一）：
    1. 环境变量（推荐）：
         PAPERPILOT_LLM_BASE_URL  如 https://api.deepseek.com/v1
         PAPERPILOT_LLM_API_KEY   服务商 API key
         PAPERPILOT_LLM_MODEL     模型名，如 deepseek-chat / glm-4-flash / gpt-4o-mini
    2. 项目根目录 .env 文件（run_claims.py 等入口会自动加载）
    3. 直接设置 os.environ

统一封装 chat_json()：只处理结构化 JSON 输出场景（claims 提取），
失败抛 LLMError 并保留原因，方便上层按 chunk 容错。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

ENV_BASE = "PAPERPILOT_LLM_BASE_URL"
ENV_KEY = "PAPERPILOT_LLM_API_KEY"
ENV_MODEL = "PAPERPILOT_LLM_MODEL"
ENV_TIMEOUT = "PAPERPILOT_LLM_TIMEOUT"


class LLMError(RuntimeError):
    pass


def config() -> tuple[str, str, str]:
    base = os.environ.get(ENV_BASE, "").strip().rstrip("/")
    key = os.environ.get(ENV_KEY, "").strip()
    model = os.environ.get(ENV_MODEL, "").strip()
    return base, key, model


def is_configured() -> bool:
    base, key, model = config()
    return bool(base and key and model)


def _load_dotenv(root: str | None = None) -> None:
    """若存在 {root}/.env 则注入 os.environ（不覆盖已有环境变量）。"""
    from pathlib import Path

    path = Path(root or ".") / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _chat_url(base: str) -> str:
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def _parse_json(content: str):
    """解析模型返回的 JSON：优先 code fence，再直接 loads，最后截取首尾大括号。"""
    if not content or not content.strip():
        raise LLMError("模型返回空内容")
    text = content.strip()
    fences = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    candidates = [f for f in fences if f.strip()]
    if not candidates:
        candidates = [text]
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # 兜底：截取最外层对象 '{...}' 或数组 '[...]'（容忍首尾说明文字）
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"无法解析模型返回的 JSON。内容前 200 字: {text[:200]!r}")


def _chat(system: str, user: str, *, temperature: float,
          max_tokens: int | None) -> str:
    """单轮对话，返回模型原始文本内容。"""
    base, key, model = config()
    if not base or not key or not model:
        raise LLMError(
            f"LLM 未配置：请设置 {ENV_BASE} / {ENV_KEY} / {ENV_MODEL}"
            "（或在工作目录放置 .env 文件）"
        )

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    try:
        body_bin = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            _chat_url(base),
            data=body_bin,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        timeout = float(os.environ.get(ENV_TIMEOUT, "120"))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        raise LLMError(f"HTTP {e.code}: {detail}") from e
    except Exception as e:  # URLError / Timeout / JSON 解析失败等
        raise LLMError(f"请求失败: {e}") from e

    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"响应缺少 choices[0].message.content: {str(data)[:300]}") from e


def chat_json(system: str, user: str, *, temperature: float = 0.2,
              max_tokens: int | None = None) -> object:
    """单轮对话，返回解析后的 JSON（list/dict）。"""
    content = _chat(system, user, temperature=temperature, max_tokens=max_tokens)
    return _parse_json(content)


def chat_text(system: str, user: str, *, temperature: float = 0.2,
              max_tokens: int | None = None) -> str:
    """单轮对话，返回原始文本（问答等非 JSON 场景）。"""
    return _chat(system, user, temperature=temperature, max_tokens=max_tokens)
