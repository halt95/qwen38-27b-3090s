#!/usr/bin/env bash
# Build the Qwen3.8-27B serving env: stock vLLM 0.30.0 from PyPI + the patches/vllm-0.30.0 series.
# This script IS the published install route; the release env was built with it verbatim, and a
# clean-room test ran on an earlier three-patch version of it.
#
#   release/install-env.sh <new-venv-path> [python-binary]
#
# Needs: Python 3.13, git, network access to PyPI, an NVIDIA driver for CUDA 13.0 (the PyPI
# torch 2.13.0 wheel is the cu130 build). Every step checks its exit code.
set -euo pipefail

VENV="${1:?usage: install-env.sh <new-venv-path> [python]}"
PY="${2:-python3.13}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$REPO/release/requirements-pinned.txt"
REQ_PIP="$REPO/release/requirements-pip.txt"
PATCHES="$REPO/patches/vllm-0.30.0"

[ -e "$VENV" ] && { echo "refusing: $VENV already exists" >&2; exit 1; }
[ -f "$REQ" ] || { echo "missing $REQ" >&2; exit 1; }
[ -f "$REQ_PIP" ] || { echo "missing $REQ_PIP" >&2; exit 1; }
ls "$PATCHES"/*.patch >/dev/null

"$PY" -m venv "$VENV"
# absolute from here on: the patch step below changes directory into site-packages
VENV="$(cd "$VENV" && pwd -P)"
# pinned pip first, then every package; both files are hash-locked (release/hash_lock.py)
"$VENV/bin/python" -m pip install --require-hashes -r "$REQ_PIP"
"$VENV/bin/python" -m pip install --require-hashes -r "$REQ"

SP="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cd "$SP"
for p in "$PATCHES"/*.patch; do
  git apply --check --directory=vllm "$p" || { echo "does not apply: $p" >&2; exit 1; }
  git apply --directory=vllm "$p"         || { echo "apply failed: $p" >&2; exit 1; }
  echo "applied $(basename "$p")"
done
# drop bytecode compiled before the patches so nothing stale is imported
find "$SP/vllm" -name '__pycache__' -type d -prune -exec rm -rf {} +

"$VENV/bin/python" - <<'PY'
import flashinfer, torch, vllm
import vllm.envs as envs
import vllm.v1.attention.backends.flashinfer as fi
import vllm.v1.worker.gpu.model_runner as mr
assert vllm.__version__ == "0.30.0", vllm.__version__
assert torch.__version__ == "2.13.0+cu130", torch.__version__
assert flashinfer.__version__ == "0.6.18.post1", flashinfer.__version__
assert fi.flashinfer_supports_uniform_multi_token_decode(), "0001 missing"
assert "VLLM_FLASHINFER_PLAN_CACHE" in envs.environment_variables, "0002 missing"
assert hasattr(mr, "_host_embed_tables"), "0003 missing"
import importlib.util
assert importlib.util.find_spec("vllm.model_executor.layers.mamba.gdn.ba_splitk"), "0004 missing"
assert "VLLM_GDN_BA_SPLITK" in envs.environment_variables, "0004 missing"
assert hasattr(fi, "_DRAFT_SEQ_LENS"), "0005 missing"
assert "VLLM_DRAFT_SEQ_LENS_NOSYNC" in envs.environment_variables, "0005 missing"
assert envs.VLLM_HOST_EMBED_TABLE and envs.VLLM_FLASHINFER_PLAN_CACHE, "defaults not on"
assert envs.VLLM_GDN_BA_SPLITK and envs.VLLM_DRAFT_SEQ_LENS_NOSYNC, "defaults not on"
assert not envs.VLLM_DRAFT_SEQ_LENS_VERIFY, "verify must default off"
print("ENV OK: vllm", vllm.__version__, "torch", torch.__version__, "flashinfer", flashinfer.__version__)
PY
"$VENV/bin/python" -m pip check
echo "INSTALL OK $VENV"
