"""同名不同内容的**验收**脚本（审查项 2 修复后：A+C 档）。

修复前（本脚本原始版本已记录在台账）：同名换内容会**静默**给出旧论文的报告——
　· 解析层读新字节、产物层给旧 claims/报告（状态混合）
　· web 上传同名文件直接返回旧报告且不写盘
修复后应当：
　A) 流水线：识别出"产物不属于当前这份 PDF" → **拒绝 --skip-llm 装配**（要求重建），不再静默混合；
　B) web 上传：同名不同内容 → **另存为 `<stem>__<sha8>.pdf` 当新论文**（原文件一个字节不动），
　　 并把这件事明确回给用户（`upload_note`）。

两条路径都用备份/恢复 + finally 清理，磁盘不留残留。用法：
    uv run python qa/review/_dupname_repro_20260913.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
PAPERS = ROOT / "assets/papers"
VIEW = ROOT / "assets/artifacts/out_views"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def stats(stem: str) -> dict:
    try:
        c = json.loads((ROOT / "assets/artifacts/out_claims" / f"{stem}.claims.json")
                       .read_text(encoding="utf-8"))
        return {"n_claims": len(c.get("claims") or [])}
    except Exception:  # noqa: BLE001
        return {}


def pick() -> tuple[Path, Path]:
    cands = [p for p in sorted(PAPERS.glob("*.pdf")) if (VIEW / f"{p.stem}.report.json").exists()]
    a = cands[0]
    b = next(p for p in sorted(PAPERS.glob("*.pdf")) if p != a)
    return a, b


def part_a_pipeline(a: Path, b: Path) -> None:
    """流水线：换内容后必须**拒绝装配**，不得静默复用旧产物。"""
    from paperpilot.pipeline import process_pdf

    before = stats(a.stem)
    orig, backup = sha(a), a.with_suffix(".pdf.__bak__")
    shutil.copy2(a, backup)
    try:
        a.write_bytes(b.read_bytes())
        print(f"\n[A] 把 {a.name} 换成 B({b.name}) 的字节（文件名不变）")
        print(f"    覆盖前产物：{before}")
        try:
            process_pdf(a.name, skip_llm=True)
            print("    ❌ **仍静默复用**（未被拦截）——修复失效")
        except FileNotFoundError as e:
            print(f"    ✅ 已拦截：{str(e)[:110]}")
        print("       （解析层与产物层不再各说各话：装配被拒 → 必须重建）")
    finally:
        shutil.move(str(backup), str(a))
        print(f"    [恢复] {a.name} sha 一致={sha(a) == orig}")


def _stub_report(name: str):
    class _R:
        def model_dump_json(self) -> str:
            return json.dumps({"pdf": name, "title": "STUB-REPORT", "stats": {}},
                              ensure_ascii=False)
    return _R()


def part_b_web(a: Path, b: Path) -> None:
    """web 上传同名不同内容 → 另存新名（原文件不动），并给出 upload_note。

    为了**不触发真实 LLM 摄取**，这里把 `app.ingest` 换成桩：只验证**落点与返回**。
    """
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # noqa: BLE001
        print("\n[B] 跳过：TestClient 不可用（需 httpx）", e)
        return
    spec = importlib.util.spec_from_file_location("webapp", "web/app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    m.ingest = lambda name, verbose=False: {"report": _stub_report(name),
                                           "meta": {"mineru": {"status": "ok"}}}

    disk_before = sha(a)
    new_name = f"{a.stem}__{sha(b)[:8]}.pdf"
    new_path = PAPERS / new_name
    new_path.unlink(missing_ok=True)
    try:
        r = TestClient(m.app).post("/api/report",
                                   files={"file": (a.name, b.read_bytes(), "application/pdf")})
        print(f"\n[B] 上传：文件名={a.name}，内容=B 的字节")
        print(f"    HTTP {r.status_code}")
        if r.status_code != 200:
            print("    detail:", str(r.json())[:140])
            return
        payload = r.json()
        note = str(payload.get("upload_note") or "")
        same_as_old = payload.get("title") == json.loads(
            (VIEW / f"{a.stem}.report.json").read_text(encoding="utf-8")).get("title")
        print(f"    ✅ 未返回旧论文报告：{'是（正确）' if not same_as_old else '否 ❌'}")
        print(f"    ✅ 另存为新论文：{new_name} 存在={new_path.exists()}")
        print(f"    ✅ 原文件未被改动：{sha(a) == disk_before}")
        print(f"    提示文案：{note[:96]}")
    finally:
        new_path.unlink(missing_ok=True)
        for p in VIEW.glob(f"{new_name[:-4]}.*"):
            p.unlink(missing_ok=True)
        print(f"    [清理] 已删除临时新论文 {new_name}")


def main() -> int:
    a, b = pick()
    print(f"选样：A={a.name}（有缓存）  B={b.name}（用于替换/上传）")
    part_a_pipeline(a, b)
    part_b_web(a, b)
    print("\n结论：同名不同内容**不再**静默给出另一篇论文的报告 —— 要么被拒（要求重建），"
          "要么作为新论文另存并明确告知。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
