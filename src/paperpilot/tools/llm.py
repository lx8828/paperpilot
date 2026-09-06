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
import time
import urllib.error
import urllib.request
from typing import Any

# 主链路（问答/生成）配置前缀
_ENV_PREFIX = "PAPERPILOT_LLM"
# 独立裁判配置前缀（QA 评测打分用；与主链路异源，防同模型自证偏好）
_JUDGE_PREFIX = "PAPERPILOT_JUDGE"


class LLMError(RuntimeError):
    pass


def config(prefix: str = _ENV_PREFIX) -> tuple[str, str, str]:
    base = os.environ.get(f"{prefix}_BASE_URL", "").strip().rstrip("/")
    key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    model = os.environ.get(f"{prefix}_MODEL", "").strip()
    return base, key, model


def is_configured(prefix: str = _ENV_PREFIX) -> bool:
    base, key, model = config(prefix)
    return bool(base and key and model)


def judge_configured() -> bool:
    """裁判模型是否已配置（独立于主链路）。"""
    return is_configured(_JUDGE_PREFIX)


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
          max_tokens: int | None, prefix: str = _ENV_PREFIX) -> str:
    """单轮对话，返回模型原始文本内容。prefix 切换主链路 / 裁判模型。"""
    base, key, model = config(prefix)
    timeout_env = f"{prefix}_TIMEOUT"
    if not base or not key or not model:
        raise LLMError(
            f"LLM 未配置：请设置 {prefix}_BASE_URL / {prefix}_API_KEY / "
            f"{prefix}_MODEL（或在工作目录放置 .env 文件）"
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
    timeout = float(os.environ.get(timeout_env, "120"))
    # 显式不走代理：Windows 上 urllib 默认读系统代理（VPN 全局会劫持 LLM 直连 TLS）
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_err: Exception | None = None
    for attempt in range(3):  # 网络抖动自动重试（最多 3 次），4xx 不重试
        try:
            with opener.open(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                detail = e.read().decode("utf-8", "ignore")[:300]
                raise LLMError(f"HTTP {e.code}: {detail}") from e
            last_err = e  # 5xx / 网关错误 → 重试
        except Exception as e:  # URLError / Timeout / SSL / JSON 解析失败
            last_err = e
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    else:
        raise LLMError(f"请求失败（重试 3 次仍失败）: {last_err}") from last_err

    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"响应缺少 choices[0].message.content: {str(data)[:300]}") from e


def chat_json(system: str, user: str, *, temperature: float = 0.2,
              max_tokens: int | None = None,
              prefix: str = _ENV_PREFIX) -> object:
    """单轮对话，返回解析后的 JSON（list/dict）。"""
    content = _chat(system, user, temperature=temperature, max_tokens=max_tokens,
                    prefix=prefix)
    return _parse_json(content)


def chat_text(system: str, user: str, *, temperature: float = 0.2,
              max_tokens: int | None = None,
              prefix: str = _ENV_PREFIX) -> str:
    """单轮对话，返回原始文本（问答等非 JSON 场景）。"""
    return _chat(system, user, temperature=temperature, max_tokens=max_tokens,
                 prefix=prefix)


# ── 裁判（独立模型打分）─────────────────────────────────────────────


def judge_text(system: str, user: str, *, temperature: float = 0.0,
               max_tokens: int | None = None) -> str:
    """用独立裁判模型对话（原始文本）。"""
    return chat_text(system, user, temperature=temperature,
                     max_tokens=max_tokens, prefix=_JUDGE_PREFIX)


def judge_json(system: str, user: str, *, temperature: float = 0.0,
               max_tokens: int | None = None) -> object:
    """用独立裁判模型对话并解析 JSON。

    裁判模型（GLM 等）偶发返回 reason 内含未转义英文引号/裸换行导致
    json.loads 失败。这里对裁判输出走宽松解析：先按标准 JSON 解析，
    失败则用正则提取 score 与 reason（裁判 schema 固定为这两个字段）。
    """
    content = _chat(system, user, temperature=temperature,
                    max_tokens=max_tokens, prefix=_JUDGE_PREFIX)
    return _parse_judge_content(content)


def _parse_judge_content(content: str) -> object:
    """裁判输出的宽松解析：先严格，失败则提取 score / reason。"""
    if not content or not content.strip():
        raise LLMError("裁判返回空内容")
    try:
        return _parse_json(content)
    except LLMError:
        pass
    m_score = re.search(r'"score"\s*:\s*(\d+)', content)
    if not m_score:
        raise LLMError(f"无法解析裁判返回的 JSON。内容前 200 字: {content[:200]!r}")
    score = int(m_score.group(1))
    reason = ""
    m_r = re.search(r'"reason"\s*:\s*', content)
    if m_r:
        tail = content[m_r.end():].lstrip()
        # reason 值：去掉首尾引号与结尾 "}（容忍内部未转义引号：只剥最外层）
        if tail.startswith('"'):
            tail = tail[1:]
        # 剥到最后一个引号（reason 是最后一个字段，尾部形如 "} 或 "})
        end = tail.rfind('"')
        if end != -1:
            tail = tail[:end]
        tail = tail.rstrip()
        # 若尾部残留反引号/code fence 结尾
        tail = tail.rstrip("`").rstrip()
        reason = tail
    return {"score": score, "reason": reason}
