"""Gitee push - whitelist based, clear all files then upload."""
import os, sys, json, base64, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path

TOKEN = "c0c61d686ec249c82569bc4cab994451"
OWNER = "song-jack"
REPO = "qsl"
BRANCH = "master"
ROOT = Path(__file__).parent
API = "https://gitee.com/api/v5"

# ============================================================
# Whitelist: exactly what to upload
# ============================================================
WHITELIST = [
    # Root
    "README.md",
    "LICENSE",
    "pyproject.toml",

    # qsl/ package
    "qsl/__init__.py",
    "qsl/__main__.py",
    "qsl/py.typed",
    "qsl/quantum_gates.py",

    # qsl/core/
    "qsl/core/__init__.py",
    "qsl/core/grover.py",
    "qsl/core/parser.py",
    "qsl/core/state.py",

    # qsl/algorithms/
    "qsl/algorithms/__init__.py",
    "qsl/algorithms/qaoa.py",
    "qsl/algorithms/qft.py",
    "qsl/algorithms/shor.py",
    "qsl/algorithms/vqe.py",

    # qsl/qml/
    "qsl/qml/__init__.py",
    "qsl/qml/kernels.py",
    "qsl/qml/layers.py",
    "qsl/qml/qgan.py",
    "qsl/qml/qnn.py",
    "qsl/qml/qsvm.py",

    # qsl/backends/
    "qsl/backends/__init__.py",
    "qsl/backends/auto_selector.py",
    "qsl/backends/aws_braket.py",
    "qsl/backends/base.py",
    "qsl/backends/ibm.py",
    "qsl/backends/registry.py",
    "qsl/backends/simulator.py",

    # qsl/compiler/
    "qsl/compiler/__init__.py",
    "qsl/compiler/compiler.py",
    "qsl/compiler/dsl.py",
    "qsl/compiler/error_mitigation.py",
    "qsl/compiler/optimizer.py",
    "qsl/compiler/program.py",
    "qsl/compiler/transpiler.py",

    # qsl/ai/
    "qsl/ai/__init__.py",
    "qsl/ai/agent.py",
    "qsl/ai/discovery.py",
    "qsl/ai/explainer.py",
    "qsl/ai/hypotheses.py",
    "qsl/ai/translator.py",

    # qsl/pipelines/
    "qsl/pipelines/__init__.py",
    "qsl/pipelines/crypto_analysis.py",
    "qsl/pipelines/drug_discovery.py",
    "qsl/pipelines/portfolio.py",

    # qsl/meta/
    "qsl/meta/__init__.py",
    "qsl/meta/algorithm_search.py",
    "qsl/meta/quantum_compiler_ai.py",
    "qsl/meta/theory_generator.py",

    # qsl/network/
    "qsl/network/__init__.py",
    "qsl/network/distributed_node.py",
    "qsl/network/quantum_blockchain.py",

    # qsl/utils/
    "qsl/utils/__init__.py",
    "qsl/utils/exceptions.py",
    "qsl/utils/validation.py",

    # tests/
    "tests/__init__.py",
    "tests/test_ai.py",
    "tests/test_algorithms.py",
    "tests/test_backends.py",
    "tests/test_compiler.py",
    "tests/test_compiler_opt.py",
    "tests/test_grover.py",
    "tests/test_integration.py",
    "tests/test_meta.py",
    "tests/test_network.py",
    "tests/test_parser.py",
    "tests/test_pipelines.py",
    "tests/test_qml.py",
    "tests/test_state.py",
]

# ============================================================
# API helpers
# ============================================================

def api(method, endpoint, data=None):
    url = f"{API}{endpoint}?access_token={TOKEN}"
    h = {"User-Agent": "qsl/1.0", "Accept": "application/json"}
    if method == "DELETE":
        if data:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="DELETE", headers=h)
            req.add_header("Content-Type", "application/json; charset=utf-8")
        else:
            req = urllib.request.Request(url, method="DELETE", headers=h)
    elif method == "GET":
        if data:
            for k, v in data.items():
                url += f"&{k}={urllib.parse.quote(str(v))}"
        req = urllib.request.Request(url, method="GET", headers=h)
    else:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method=method, headers=h)
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            if r.status == 204:
                return {}
            raw = r.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")[:200]
        return {"_err": e.code, "_msg": msg}


def list_all_files():
    """List all files in repo via git trees API (recursive)."""
    # Get branch info → tree sha
    branch_info = api("GET", f"/repos/{OWNER}/{REPO}/branches/{BRANCH}")
    if "_err" in branch_info:
        print(f"  get branch failed: {branch_info.get('_msg')}")
        return []

    try:
        tree_sha = branch_info["commit"]["commit"]["tree"]["sha"]
    except (KeyError, TypeError):
        print("  could not extract tree sha from branch info")
        return []

    # Get recursive tree
    tree = api("GET", f"/repos/{OWNER}/{REPO}/git/trees/{tree_sha}", {"recursive": "1"})
    if "_err" in tree:
        print(f"  get tree failed: {tree.get('_msg')}")
        return []

    items = tree.get("tree", [])
    # Handle truncated tree
    if tree.get("truncated"):
        print("  WARNING: tree is truncated, may miss some files")

    files = [(item["path"], item["sha"]) for item in items if item.get("type") == "blob"]
    return files


def clear_repo_files():
    """Delete all files in repo without deleting the repo itself."""
    print("  listing all files ...")
    files = list_all_files()

    if not files:
        print("  no files to delete (repo may already be empty)")
        return True

    print(f"  found {len(files)} files, deleting ...")
    deleted = 0
    for path, sha in files:
        enc_path = urllib.parse.quote(path, safe="/")
        result = api("DELETE", f"/repos/{OWNER}/{REPO}/contents/{enc_path}", {
            "message": f"clear: {path}",
            "sha": sha,
            "branch": BRANCH,
        })
        if "_err" in result:
            print(f"  FAIL delete: {path}  ({result.get('_msg', '')})")
            continue
        deleted += 1
        time.sleep(0.15)

    print(f"  deleted {deleted}/{len(files)} files")
    return deleted == len(files)


def upload_file(rel_path):
    """Upload a single file. Returns True on success."""
    abs_path = ROOT / rel_path
    if not abs_path.exists():
        print(f"  MISSING: {rel_path}")
        return False

    content_b64 = base64.b64encode(abs_path.read_bytes()).decode("ascii")
    enc_path = urllib.parse.quote(rel_path, safe="/")
    msg = f"v0.3.0: {rel_path}"

    # Check if file already exists (get sha)
    sha_result = api("GET", f"/repos/{OWNER}/{REPO}/contents/{enc_path}", {"ref": BRANCH})
    sha = None
    if isinstance(sha_result, dict) and "_err" not in sha_result:
        sha = sha_result.get("sha")

    data = {"content": content_b64, "message": msg, "branch": BRANCH}
    if sha:
        data["sha"] = sha
        result = api("PUT", f"/repos/{OWNER}/{REPO}/contents/{enc_path}", data)
    else:
        result = api("POST", f"/repos/{OWNER}/{REPO}/contents/{enc_path}", data)

    err = result.get("_err") if isinstance(result, dict) else None
    return err is None


# ============================================================
# Main
# ============================================================

print("=" * 50)
print("  QSL -> Gitee push (whitelist, clear then upload)")
print("=" * 50)

# 1. Clear all files in repo
print(f"\n[Step 1] Clear repo {OWNER}/{REPO} ...")
clear_repo_files()

# 2. Upload all whitelisted files
total = len(WHITELIST)
print(f"\n[Step 2] Upload {total} files:")
print()

ok = 0
fail_list = []

for i, path in enumerate(WHITELIST, 1):
    print(f"  [{i:2d}/{total}]", end=" ")
    success = upload_file(path)
    if success:
        ok += 1
        print(f"OK   {path}")
    else:
        fail_list.append(path)
        print(f"FAIL {path}")
    time.sleep(0.25)

print(f"\n{'=' * 50}")
print(f"  Result: {ok}/{total} success")
if fail_list:
    print(f"  Failed ({len(fail_list)}):")
    for f in fail_list:
        print(f"    - {f}")
print(f"  https://gitee.com/{OWNER}/{REPO}")
print(f"{'=' * 50}")
