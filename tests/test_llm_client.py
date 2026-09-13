"""LLM 客户端的**离线**行为：JSON 解析容错、未配置时的报错、超长输入。

这一层是"**不依赖真实 API Key**"的另一半保障：
CI 里永远不联网，但客户端本身的解析/容错逻辑必须被测到——
否则线上遇到"模型返回了带说明文字/被 ``` 包住的 JSON"就会整块失败。
"""
from __future__ import annotations

import json

import pytest

from paperpilot.tools import llm


# ───────────────────────── JSON 解析容错 ─────────────────────────


@pytest.mark.parametrize("content, expect", [
    ('{"a": 1}', {"a": 1}),
    ("```json\n{\"a\": 1}\n```", {"a": 1}),
    ("```\n[1, 2]\n```", [1, 2]),
    ("好的，结果如下：\n{\"a\": 1}\n以上。", {"a": 1}),      # 首尾夹说明文字
    ("[1,2,3]", [1, 2, 3]),
], ids=["纯 JSON", "json 围栏", "裸围栏", "夹说明文字", "数组"])
def test_parse_json_tolerant(content, expect):
    assert llm._parse_json(content) == expect


@pytest.mark.parametrize("content", ["", "   ", "完全不是 JSON", "{{{"],
                         ids=["空", "空白", "纯文本", "坏括号"])
def test_parse_json_raises_on_garbage(content):
    with pytest.raises(llm.LLMError):
        llm._parse_json(content)


def test_parse_json_long_input():
    """超长输入（20 万字符）不崩、不乱截断（模型偶尔吐超长数组）。"""
    payload = [{"text": "x" * 200} for _ in range(1000)]
    assert len(llm._parse_json(json.dumps(payload))) == 1000


# ───────────────────────── 未配置：明确报错，不静默 ─────────────────────────


def test_not_configured_by_default():
    assert llm.is_configured() is False


def test_chat_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv("PAPERPILOT_LLM_API_KEY", raising=False)
    with pytest.raises(llm.LLMError) as e:
        llm.chat_json("sys", "user")
    assert "未配置" in str(e.value)


def test_judge_prefix_is_independent(monkeypatch):
    """主链路配了、裁判没配 → 裁判仍应视为未配置（异源裁判不能被主链路 key 顶替）。"""
    monkeypatch.setenv("PAPERPILOT_LLM_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PAPERPILOT_LLM_API_KEY", "k")
    monkeypatch.setenv("PAPERPILOT_LLM_MODEL", "m")
    assert llm.is_configured() is True
    assert llm.judge_configured() is False


# ───────────────────────── 上层封装（用假 _chat）─────────────────────────


def test_chat_json_and_text_go_through_parse(monkeypatch):
    monkeypatch.setattr(llm, "_chat",
                        lambda *a, **k: '```json\n{"ok": true}\n```')
    assert llm.chat_json("s", "u") == {"ok": True}
    assert llm.chat_text("s", "u") == '```json\n{"ok": true}\n```'


def test_chat_json_raises_on_non_json_reply(monkeypatch):
    """模型返回散文 → 必须抛 LLMError（上层据此逐块容错，而不是静默变 None）。"""
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: "抱歉，我不会")
    with pytest.raises(llm.LLMError):
        llm.chat_json("s", "u")


def test_usage_stats_counted(monkeypatch):
    """用量计数器（评测成本统计）在成功调用后应清零可读。"""
    llm.reset_usage()
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: "x")
    llm.chat_text("s", "u")
    assert llm.usage_stats()["calls"] == 0        # 假 _chat 不动计数器（真实 _chat 才计）
