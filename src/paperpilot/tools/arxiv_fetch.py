"""arXiv PDF 获取（工具 ②）：`arXiv id → assets/papers/<id>.pdf`。

## 为什么单独做

检索只给**元数据**（标题、摘要、分数），而精读链 `paperpilot.ingest.ingest()` 需要一份
**本地 PDF**。本模块只负责中间那一步，且**只管下载** —— 不触发 MinerU、不生成报告，
保持职责单一（那些是 `ingest()` 的事）。

## 四个设计决定

**① 文件名 = arXiv id**（不是标题、不是临时名）

    `2301.12345`      → `2301.12345.pdf`
    `cs/9811009v2`    → `cs_9811009.pdf`      （老式 id 含 `/`，转义为 `_`）

  id 是天然唯一键 ⇒ 天然不会出现"同名不同内容"（不会像用户上传那样把 A 的字节
  塞进 B 的文件名），正好绕开 `paper_identity` 要防的那类静默错配；
  幂等判断也只需看文件在不在。

  ⚠️ **但"同一 id 内容永久不变"只在本地成立**：URL 不带版本号 ⇒ 取的是**最新版**，
  上游出了新版本时重抓会拿到不同内容（缓存里那份不会自己变）。
  这是有意的取舍：**同一篇论文只留一份缓存**（带版本号会出现 v1/v2 两份）。
  真要锁定版本，把 `arxiv_id` 写成 `2301.12345v2` —— 但文件名仍会归一为不带版本，
  此时请用 `force=True` 重抓。

**② 落盘目录 = `assets/papers/`**（与用户上传同一库）

  与 `ingest(pdf_name)` 的输入口径完全一致 → 抓完直接喂，不需要任何胶水。

**③ 原子写**：先落 `<name>.part` 再 `rename`

  否则中断会留下**半截文件**，而幂等检查只看"文件存在" ⇒ 会把坏文件当成好缓存
  永久沿用（静默毒化，且下次不会自愈）。

**④ 校验 `%PDF-` 魔数**

  arXiv 出错时会返回 **HTML 错误页且 HTTP 200**（不是 404）→ 不查魔数就会把 HTML
  当成 PDF 喂进解析链，报错点还会推迟到很远的地方。

## 两种 id 格式都要支持

**现代** `2301.12345`（2023 年至今，我们生产语料全是这种）
**老式** `cs/9811009` / `cmp-lg/9605014`（旧语料里有 494 篇）

版本号 `vN` 一律去掉：`https://arxiv.org/pdf/<id>` 不带版本即取**最新版**。

## 已知边界（实测）

- 语料里 **540,052 篇 arXiv 论文 100% 可取**（抽样 120/120 成功，见
  `retrieval/scripts/probe_pdf_availability.py`）；
- 旧语料侧只有 24.8% 有 arXiv id —— **不在生产语料内**，本模块不处理。

用法：
    from paperpilot.tools.arxiv_fetch import fetch_arxiv
    r = fetch_arxiv("2301.12345")
    if r["ok"]:
        ingest(r["pdf_name"])          # 交给现成的摄取链
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]           # src/paperpilot/tools/x.py → 根
PAPERS_DIR = ROOT / "assets" / "papers"

PDF_URL = "https://arxiv.org/pdf/"
UA = "paperpilot/0.1 (academic reading assistant; contact: local)"
MIN_BYTES = 8 * 1024        # 小于此值几乎肯定不是真论文（arXiv 正常 PDF 都 > 100 KB）
_MAGIC = b"%PDF-"
_VER = re.compile(r"v\d+$", re.I)


class FetchError(RuntimeError):
    """获取失败（原因面向用户可读，放在 `reason` 里返回给调用方）。"""


# ── id 规范化 ────────────────────────────────────────────────────────────────

def normalize_arxiv_id(raw: str) -> str:
    """去掉 `arXiv:` 前缀与版本号。`cs/9811009v2` → `cs/9811009`。"""
    s = str(raw).strip()
    s = re.sub(r"^arxiv:", "", s, flags=re.I).strip()
    return _VER.sub("", s)


def arxiv_pdf_name(arxiv_id: str) -> str:
    """arXiv id → 论文库里的**安全文件名**（老式 id 的 `/` 转义为 `_`）。

    ⚠️ 只做 `/`→`_` 这一种替换：id 里本就只有 `[a-z0-9./-]`，
    其余字符一律不该出现，出现了说明输入有问题（由 `_validate_id` 拦）。
    """
    aid = normalize_arxiv_id(arxiv_id)
    return aid.replace("/", "_") + ".pdf"


def _validate_id(aid: str) -> None:
    """格式校验（现代 / 老式两种）。不合规直接拒，不要拿它去拼 URL。"""
    modern = re.fullmatch(r"\d{4}\.\d{4,5}", aid)
    old = re.fullmatch(r"[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7}", aid)
    if not (modern or old):
        raise FetchError(f"不是合法的 arXiv id：{aid!r}"
                         f"（期望 `2301.12345` 或 `cs/9811009` 这种形式）")


def pdf_url(arxiv_id: str) -> str:
    aid = normalize_arxiv_id(arxiv_id)
    _validate_id(aid)
    return PDF_URL + aid                     # 老式 id 保留 `/`（URL 就该这样）


# ── 缓存判定 ─────────────────────────────────────────────────────────────────

def _is_valid_pdf(p: Path) -> tuple[bool, str]:
    """文件是不是**真 PDF**（大小 + 魔数）。返回 (是否有效, 原因)。"""
    try:
        st = p.stat()
    except OSError as e:
        return False, f"读不到文件：{e}"
    if st.st_size < MIN_BYTES:
        return False, f"文件过小（{st.st_size} 字节），不像是论文 PDF"
    try:
        with open(p, "rb") as f:
            head = f.read(len(_MAGIC))
    except OSError as e:
        return False, f"读不到文件：{e}"
    if head != _MAGIC:
        return False, f"不是 PDF（开头是 {head!r}，常见于 arXiv 返回的 HTML 错误页）"
    return True, ""


# ── 下载 ─────────────────────────────────────────────────────────────────────

_last_request_at = 0.0


def _throttle(min_interval: float) -> None:
    """全局最小请求间隔（arXiv 的 robots 要求礼貌抓取）。"""
    global _last_request_at
    if min_interval <= 0:
        return
    wait = min_interval - (time.time() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


def _download(url: str, dest_part: Path, timeout: int, tries: int = 3) -> int:
    """下到 `dest_part`。网络类错误重试；HTTP 4xx 不重试（重试也没用）。返回字节数。"""
    last: Exception | None = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                ctype = (r.headers.get("Content-Type") or "").lower()
                body = r.read()
            # arXiv 用 200 + HTML 报错很常见 → 这里先查一次，早失败早报错
            if not body.startswith(_MAGIC) and "pdf" not in ctype:
                raise FetchError(
                    f"返回的不是 PDF（Content-Type={ctype!r}，开头 {body[:16]!r}）"
                    f"—— 通常意味着该 arXiv id 已撤回或不可获取")
            dest_part.parent.mkdir(parents=True, exist_ok=True)
            dest_part.write_bytes(body)
            return len(body)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:                       # 404/403… 重试无意义
                raise FetchError(f"HTTP {e.code} {e.reason}") from e
            last = e
        except FetchError:
            raise
        except Exception as e:  # noqa: BLE001  超时 / DNS / 连接重置 → 退避重试
            last = e
        if t < tries - 1:
            time.sleep(2.0 * (t + 1))
    raise FetchError(f"{type(last).__name__}: {last}（已重试 {tries} 次）")


# ── 主入口 ───────────────────────────────────────────────────────────────────

def fetch_arxiv(arxiv_id: str, *, force: bool = False, timeout: int = 60,
                min_interval: float = 1.0) -> dict[str, Any]:
    """把一篇 arXiv 论文的 PDF 落到 `assets/papers/`。

    Args:
        arxiv_id: 现代 / 老式 id 均可，可带 `vN`。
        force: True 时即使本地已有效也重新下载。
        timeout: 单次请求超时（秒）。
        min_interval: 全局最小请求间隔（秒）；批量抓取时由调用方按需调大。

    Returns:
        `{"ok": bool, "pdf_name": str, "path": str, "bytes": int,
          "cached": bool, "reason": str}` —— **失败不抛异常**，靠 `ok`/`reason`，
        方便编排层（agent）统一处理，与 `ingest.run_mineru()` 的返回风格一致。
    """
    try:
        aid = normalize_arxiv_id(arxiv_id)
        _validate_id(aid)
    except FetchError as e:
        return {"ok": False, "pdf_name": "", "path": "", "bytes": 0,
                "cached": False, "reason": str(e)}

    name = arxiv_pdf_name(aid)
    dest = PAPERS_DIR / name

    # ── 缓存：只要文件在、且是**真 PDF** 就直接用 ──────────────────────
    if dest.exists() and not force:
        good, why = _is_valid_pdf(dest)
        if good:
            return {"ok": True, "pdf_name": name, "path": str(dest),
                    "bytes": dest.stat().st_size, "cached": True, "reason": ""}
        # 坏缓存（半截文件 / HTML）：删掉重下，而不是留着毒化后续
        dest.unlink(missing_ok=True)

    part = dest.with_suffix(dest.suffix + ".part")
    try:
        _throttle(min_interval)
        n = _download(pdf_url(aid), part, timeout)
        good, why = _is_valid_pdf(part)
        if not good:
            raise FetchError(why)
        part.replace(dest)                    # 原子替换：此时才认为下载成功
        return {"ok": True, "pdf_name": name, "path": str(dest),
                "bytes": n, "cached": False, "reason": ""}
    except FetchError as e:
        part.unlink(missing_ok=True)
        return {"ok": False, "pdf_name": name, "path": "", "bytes": 0,
                "cached": False, "reason": str(e)}
    except Exception as e:  # noqa: BLE001
        part.unlink(missing_ok=True)
        return {"ok": False, "pdf_name": name, "path": "", "bytes": 0,
                "cached": False, "reason": f"{type(e).__name__}: {e}"}
