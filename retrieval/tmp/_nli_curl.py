r"""用 `curl -L` 把 NLI 模型下到**本地目录**（绕开 HF 缓存的卡死）。

## 为什么不用 `huggingface_hub`
本机 `huggingface.co` 被墙，走 `hf-mirror.com` 时：
· **小文件**（config/tokenizer/spm.model）能下 ✓
· **大文件**（`pytorch_model.bin` ~1.1GB）会 **TLS 断流卡死**（`UNEXPECTED_EOF_WHILE_READING`）
而 `curl -L` 对同一 URL **稳定成功**（镜像用 307 跳到同域 `/api/resolve-cache/...`）。
→ 用 `curl` 下到 `data/nli_models/<name>/`，之后 `from_pretrained(本地目录)` 即可。

文件清单从镜像的 `api/models/<repo>/tree/main` 取（不是硬编码）。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_nli_curl.py
  ./.venv/Scripts/python.exe -u retrieval/tmp/_nli_curl.py --repo MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "data" / "nli_models"
MIRROR = "https://hf-mirror.com"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 只下**推理必需**的文件（跳过 training_args.bin / README / .gitattributes 等）
WANT_EXT = (".json", ".model", ".bin", ".safetensors", ".txt")
SKIP = {"training_args.bin", "README.md", ".gitattributes", "optimizer.pt",
        "scheduler.pt", "trainer_state.json", "rng_state.pth"}


def tree(repo: str) -> list[dict]:
    url = f"{MIRROR}/api/models/{repo}/tree/main"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=30) as r:      # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    args = ap.parse_args()
    name = args.repo.split("/")[-1]
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)

    files = [f for f in tree(args.repo)
             if f.get("type") == "file" and f["path"].split("/")[-1] not in SKIP
             and f["path"].endswith(WANT_EXT)]
    print(f"【{args.repo}】{len(files)} 个文件 → {dest}")
    for f in files:
        p = f["path"]
        mb = f.get("size", 0) / 1e6
        tgt = dest / Path(p).name
        if tgt.exists() and tgt.stat().st_size == f.get("size", -1):
            print(f"  ⏭ {p}（已存在 {mb:.1f} MB）")
            continue
        url = f"{MIRROR}/{args.repo}/resolve/main/{p}"
        r = subprocess.run(                      # noqa: S603
            ["curl.exe", "-sS", "-L", "--retry", "5", "--retry-all-errors",
             "--retry-delay", "3", "--connect-timeout", "30", "--max-time", "1800",
             "-o", str(tgt), url], capture_output=True, text=True)
        ok = r.returncode == 0 and tgt.exists() and tgt.stat().st_size > 0
        print(f"  {'✅' if ok else '❌'} {p}（{mb:.1f} MB）"
              f"{'' if ok else '  ' + (r.stderr or '')[:120]}")
        if not ok:
            return 1
    print(f"  → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
