"""
PyPI 发布脚本。

用法:
    python publish_pypi.py              # 构建 + 上传（从环境变量读取 Token）
    python publish_pypi.py --token xxx  # 指定 Token
    python publish_pypi.py --test       # 上传到 TestPyPI 先验证

前置:
    pip install build twine

Token 获取:
    https://pypi.org/manage/account/token/  →  创建 API Token
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"


def clean():
    """清理旧构建产物。"""
    # 精确路径
    for d in [DIST, ROOT / "build"]:
        if d.exists() and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            print(f"  清理: {d.name}")
    # 通配 egg-info
    for p in ROOT.glob("*.egg-info"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"  清理: {p.name}")
    print("  [OK] 清理完成\n")


def build():
    """构建 wheel 和 sdist。"""
    print("[..] 构建包 ...")
    result = subprocess.run(
        [sys.executable, "-m", "build", str(ROOT)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT)
    )
    if result.returncode != 0:
        print(f"  [FAIL] 构建失败:\n{result.stderr}")
        return False

    files = list(DIST.glob("*"))
    for f in files:
        print(f"  生成: {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    print(f"  [OK] 构建完成，共 {len(files)} 个文件\n")
    return True


def check_twine():
    """检查 twine 是否安装。"""
    try:
        subprocess.run(
            [sys.executable, "-m", "twine", "--version"],
            capture_output=True, encoding="utf-8", errors="replace", check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[FAIL] 未安装 twine，请运行: pip install twine")
        return False


def upload(token, repository="pypi"):
    """上传到 PyPI。"""
    repo_url = {
        "pypi": "https://upload.pypi.org/legacy/",
        "testpypi": "https://test.pypi.org/legacy/",
    }
    url = repo_url.get(repository, repository)

    dist_files = list(DIST.glob("*"))
    if not dist_files:
        print("[FAIL] dist/ 目录为空，请先构建。")
        return False

    print(f"  [..] 上传到 {repository.upper()} ...")
    result = subprocess.run(
        [sys.executable, "-m", "twine", "upload", "--skip-existing"] +
        [str(f) for f in dist_files],
        env={
            **os.environ,
            "TWINE_USERNAME": "__token__",
            "TWINE_PASSWORD": token,
            "TWINE_REPOSITORY_URL": url,
            "TWINE_NON_INTERACTIVE": "1",
            "PYTHONIOENCODING": "utf-8",
        },
        cwd=str(ROOT)
    )
    if result.returncode == 0:
        print("  [OK] 上传成功!")
        return True
    else:
        print(f"  [FAIL] 上传失败 (退出码 {result.returncode})")
        return False


def verify_upload():
    """验证包可以从 PyPI 安装（不实际安装）。"""
    import urllib.request
    import json
    print("[..] 验证 PyPI 包信息 ...")
    try:
        url = "https://pypi.org/pypi/qsl-quantum/json"
        req = urllib.request.Request(url, headers={"User-Agent": "qsl-publish/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            version = data.get("info", {}).get("version", "unknown")
            print(f"  [OK] PyPI 包存在，版本: {version}")
            return True
    except urllib.request.HTTPError as e:
        if e.code == 404:
            print("  [!] 包尚未索引（新包可能需几分钟）")
        else:
            print(f"  [!] HTTP 错误: {e.code} - {e.reason}")
        return True  # 不算失败
    except Exception as e:
        print(f"  [!] 网络验证跳过: {e}")
        return True  # 不算失败


def main():
    print("=" * 50)
    print("  QSL - PyPI 发布工具")
    print("=" * 50 + "\n")

    # 解析参数
    args = sys.argv[1:]
    token = os.environ.get("PYPI_TOKEN", "")
    repository = "pypi"

    for i, arg in enumerate(args):
        if arg == "--token" and i + 1 < len(args):
            token = args[i + 1]
        elif arg == "--test":
            repository = "testpypi"

    if not token:
        print("[!] 未设置 PyPI Token。")
        print("    方式 1: 设置环境变量 PYPI_TOKEN")
        print("    方式 2: python publish_pypi.py --token pypi-xxxxxxxx")
        print("    获取 Token: https://pypi.org/manage/account/token/")
        print()
        token_input = input("    或直接输入 Token: ").strip()
        if token_input:
            token = token_input
        else:
            sys.exit(1)

    # 1. 清理
    clean()

    # 2. 检查 twine
    if not check_twine():
        sys.exit(1)

    # 3. 构建
    if not build():
        sys.exit(1)

    # 4. 上传
    if not upload(token, repository):
        sys.exit(1)

    # 5. 验证
    if repository == "pypi":
        verify_upload()

    print(f"\n{'=' * 50}")
    print(f"  发布完成!")
    print(f"  pip install qsl")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
