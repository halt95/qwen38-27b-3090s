#!/usr/bin/env bash
# Build the pinned Merlin engine v2.5.2 and apply the Qwen3.8-27B series.
#   release/install-env.sh <new-venv-path> [engine-source-path]
# Requires the v2.5.2 bundle and compiled-artifact tarball beside this repo, or BUNDLE/ARTIFACTS.
set -euo pipefail

VENV="${1:?usage: install-env.sh <new-venv-path> [engine-source-path]}"
ENGINE="${2:-}"
# vLLM's setup.py extracts a prebuilt wheel when any of these is set (even with VLLM_TARGET_DEVICE=empty), after the
# native-library hash checks below, and so could overwrite the verified artifacts or add unverified ones.
for v in VLLM_PRECOMPILED_WHEEL_LOCATION VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST; do
  if [ -n "${!v+x}" ]; then
    echo "refusing: $v is set in the environment (unset it; the engine's native artifacts come only from the pinned tarball)" >&2
    exit 1
  fi
done
PY="python3.13"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$REPO/release/requirements-pinned.txt"
REQ_PIP="$REPO/release/requirements-pip.txt"
REQ_BUILD="$REPO/release/requirements-build-tools.txt"
PATCHES="$REPO/patches/merlin-engine-v2.5.2"
# vllm/ after 0001-0011 on Merlin engine v2.5.2: the gated engine (= Flash-Next v2.5.1 + 0001-0012), checked below.
SERIES_VLLM_TREE=c02142fbffc0d7139a887e2373377ea400573ddf
FLASHINFER_PATCH="$REPO/release/apply-flashinfer-patch.sh"
# shellcheck disable=SC1091
. "$REPO/upstream/PIN"
[ -n "$ENGINE" ] || ENGINE="$REPO/.vllm-v2.5.2"
BUNDLE="${BUNDLE:-$REPO/../$bundle}"
ARTIFACTS="${ARTIFACTS:-$REPO/../$artifacts_tar}"
# Absolute paths before any check or any `git -C "$ENGINE"` call: git resolves a relative bundle path against the
# engine directory, not the caller's. (A path whose directory does not exist is left as given; the checks below report it.)
abspath() {
  local d; d="$(dirname -- "$1")"
  if [ -d "$d" ]; then printf '%s/%s' "$(cd "$d" && pwd -P)" "$(basename -- "$1")"; else printf '%s' "$1"; fi
}
BUNDLE="$(abspath "$BUNDLE")"
ARTIFACTS="$(abspath "$ARTIFACTS")"
ENGINE="$(abspath "$ENGINE")"

[ ! -e "$VENV" ] || { echo "refusing: $VENV already exists" >&2; exit 1; }
[ -f "$REQ" ] || { echo "missing $REQ" >&2; exit 1; }
[ -f "$REQ_PIP" ] || { echo "missing $REQ_PIP" >&2; exit 1; }
[ -f "$REQ_BUILD" ] || { echo "missing $REQ_BUILD" >&2; exit 1; }
[ -x "$FLASHINFER_PATCH" ] || { echo "missing executable $FLASHINFER_PATCH" >&2; exit 1; }
[ -f "$BUNDLE" ] || [ ! -f "$BUNDLE.gz" ] || { gzip -dk "$BUNDLE.gz"; }
[ -f "$BUNDLE" ] || { echo "missing pinned engine bundle: $BUNDLE" >&2; exit 1; }
[ -f "$ARTIFACTS" ] || { echo "missing pinned compiled-artifact tarball: $ARTIFACTS" >&2; exit 1; }
sha_ok() { [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$1" ]; }
sha_ok "$bundle_sha256" "$BUNDLE" || { echo "engine bundle SHA-256 mismatch" >&2; exit 1; }
sha_ok "$artifacts_sha256" "$ARTIFACTS" || { echo "compiled-artifact tarball SHA-256 mismatch" >&2; exit 1; }
ls "$PATCHES"/*.patch >/dev/null

# An engine directory that THIS run creates is removed again on any failure, so a retry starts clean (a half-initialised
# or half-patched tree would otherwise fail the HEAD / modified-files checks forever). A directory that existed before
# this run is never deleted.
ENGINE_CREATED=0
cleanup_engine() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$ENGINE_CREATED" = 1 ] && [ -n "$ENGINE" ] && [ "$ENGINE" != / ] && [ -d "$ENGINE" ]; then
    rm -rf -- "$ENGINE"
    echo "install-env: failed (exit $rc); removed the engine directory this run created: $ENGINE" >&2
  fi
  exit "$rc"
}
trap cleanup_engine EXIT

# Reproduce the Merlin engine v2.5.2 source setup: fetch the exact public base and bundle head,
# then assert both commit and content tree before adding the 27B patches.
if [ ! -d "$ENGINE/.git" ]; then
  [ ! -e "$ENGINE" ] || { echo "refusing existing non-git engine path: $ENGINE" >&2; exit 1; }
  mkdir -p "$ENGINE"
  ENGINE_CREATED=1
  git -C "$ENGINE" init -q
  git -C "$ENGINE" remote add upstream "$repo"
  for c in $prereqs; do git -C "$ENGINE" fetch -q --depth 1 upstream "$c"; done
  git -C "$ENGINE" fetch -q "$BUNDLE" "+$ref:refs/heads/v2.5.2"
  git -C "$ENGINE" checkout -q --detach "$commit"
fi
[ "$(git -C "$ENGINE" rev-parse HEAD)" = "$commit" ] || { echo "engine commit does not match PIN" >&2; exit 1; }
[ "$(git -C "$ENGINE" rev-parse 'HEAD^{tree}')" = "$tree" ] || { echo "engine tree does not match PIN" >&2; exit 1; }
[ -z "$(git -C "$ENGINE" status --porcelain --untracked-files=no)" ] || { echo "engine has modified tracked files" >&2; exit 1; }

"$PY" -m venv "$VENV"
VENV="$(cd "$VENV" && pwd -P)"
"$VENV/bin/python" -m pip install --require-hashes -r "$REQ_PIP"
"$VENV/bin/python" -m pip install --require-hashes -r "$REQ"
# setuptools-rust/-scm: vLLM's setup.py imports them for the editable engine install below (no build isolation).
"$VENV/bin/python" -m pip install --require-hashes --no-deps -r "$REQ_BUILD"

# Build-artifact route: the checked tarball contains stock 0.30.0
# wheel products plus the two v2.2.0 extensions reused unchanged by v2.5.1 and v2.5.2.
tar -xzf "$ARTIFACTS" -C "$ENGINE"
sha_ok "$own_C_sha256" "$ENGINE/vllm/_C_stable_libtorch.abi3.so" || { echo "bad _C_stable_libtorch artifact" >&2; exit 1; }
sha_ok "$own_moe_C_sha256" "$ENGINE/vllm/_moe_C_stable_libtorch.abi3.so" || { echo "bad _moe_C_stable_libtorch artifact" >&2; exit 1; }

# Install the v2.5.2 source as the vLLM 0.30.0 distribution before patching.
(cd "$ENGINE" && env -u VLLM_PRECOMPILED_WHEEL_LOCATION -u VLLM_USE_PRECOMPILED -u VLLM_USE_PRECOMPILED_RUST \
  VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE="$stock_wheel_version" "$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e .)
# The editable install must not have replaced or added native artifacts: re-check the two pinned libraries.
sha_ok "$own_C_sha256" "$ENGINE/vllm/_C_stable_libtorch.abi3.so" || { echo "_C_stable_libtorch artifact changed during the editable install" >&2; exit 1; }
sha_ok "$own_moe_C_sha256" "$ENGINE/vllm/_moe_C_stable_libtorch.abi3.so" || { echo "_moe_C_stable_libtorch artifact changed during the editable install" >&2; exit 1; }
got="$("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
[ "$got" = "$stock_wheel_version" ] || { echo "vllm version $got != $stock_wheel_version" >&2; exit 1; }

# Apply all eleven patches in lexical order (paths are relative to the vllm/ package), checking each before it is applied.
for p in "$PATCHES"/*.patch; do
  git -C "$ENGINE" apply --check --directory=vllm "$p" || { echo "does not apply: $p" >&2; exit 1; }
  git -C "$ENGINE" apply --directory=vllm "$p" || { echo "apply failed: $p" >&2; exit 1; }
  echo "applied $(basename "$p")"
done
# The same patches on a scratch index from HEAD must give the gated vllm/ tree, and the working tree must match it.
IDX="$ENGINE/.git/series-index"; rm -f "$IDX"
GIT_INDEX_FILE="$IDX" git -C "$ENGINE" read-tree HEAD
for p in "$PATCHES"/*.patch; do GIT_INDEX_FILE="$IDX" git -C "$ENGINE" apply --cached --directory=vllm "$p"; done
[ "$(GIT_INDEX_FILE="$IDX" git -C "$ENGINE" write-tree --prefix=vllm/)" = "$SERIES_VLLM_TREE" ] || { echo "patched vllm/ tree is not $SERIES_VLLM_TREE" >&2; exit 1; }
GIT_INDEX_FILE="$IDX" git -C "$ENGINE" diff --quiet -- vllm || { echo "engine working tree differs from the patched tree" >&2; exit 1; }
rm -f "$IDX"
find "$ENGINE/vllm" -name '__pycache__' -type d -prune -exec rm -rf {} +

# Editable source must win over any previously installed package path for the assertions below.
export PYTHONPATH="$ENGINE${PYTHONPATH:+:$PYTHONPATH}"
"$FLASHINFER_PATCH" "$VENV"

"$VENV/bin/python" - <<'PY'
import flashinfer, torch, vllm
import vllm.envs as envs
import vllm.v1.attention.backends.flashinfer as fi
import vllm.v1.qwen35_series as qwen35
assert vllm.__version__ == "0.30.0", vllm.__version__
assert torch.__version__ == "2.13.0+cu130", torch.__version__
assert flashinfer.__version__ == "0.6.18.post1", flashinfer.__version__
assert fi.flashinfer_supports_uniform_multi_token_decode(), "0001 missing"
assert "VLLM_FLASHINFER_PLAN_CACHE" in envs.environment_variables, "0002 missing"
assert hasattr(qwen35, "series_active"), "0007 missing"
assert "VLLM_HOST_EMBED_TABLE" in envs.environment_variables, "0003 missing"
assert "VLLM_GDN_BA_SPLITK" in envs.environment_variables, "0004 missing"
assert "VLLM_DRAFT_SEQ_LENS_NOSYNC" in envs.environment_variables, "0005 missing"
import inspect
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
import vllm.model_executor.models.qwen3_5 as q35
assert 'getattr(self, "_backend", None) == "fa2"' in inspect.getsource(fi), "0006 missing (fa2 plan-cache guard)"
assert "series_active" in inspect.getsource(gdn), "0006/0007 missing (split-K family gate)"
assert hasattr(fi, "_DRAFT_SEQ_LENS"), "0005 missing"
assert hasattr(qwen35, "is_dense_qwen35"), "0007 missing"
assert any(getattr(c, "supports_replayssm", False) for c in (q35.Qwen3_5ForCausalLM, q35.Qwen3_5ForConditionalGeneration)), "0008 missing"
assert "current_device" in inspect.getsource(qwen35), "0009 missing"
assert "device" in inspect.signature(qwen35.series_active).parameters, "0010 missing (explicit execution device)"
import vllm.v1.attention.backends.gdn_recoverssm as gdnr
assert "update_block_table" in vars(gdnr.GDNRecoverSSMMetadataBuilder), "0011 missing (RecoverSSM metadata reuse across GDN groups)"
assert {"VLLM_RECOVERSSM_COMMIT_GRAPH", "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE"} <= set(envs.environment_variables), "0011 missing (commit graph / commit-after-proposal knobs)"
assert "bound" in inspect.signature(gdnr._check_context_binding).parameters, "0011 missing (commit context bound to checkpoint storage, not identity)"
# 0012 (in the engine since v2.5.2): the guard reader idle quiesce is present, on by default, and armed by the worker on
# every entry point that launches guarded kernels (a text check of each decorator block: a missing wrapper fails here).
import importlib.util
import vllm.v1.worker.guard_telemetry as gt
assert "VLLM_E2_GUARD_IDLE" in envs.environment_variables, "0012 missing (VLLM_E2_GUARD_IDLE)"
assert all(hasattr(gt, n) for n in ("engine_call", "enable_idle_gate", "stop_reader", "suspend_reader", "resume_reader", "_off_fence_stream")), "0012 r2 missing (guard idle quiesce)"
assert gt._IDLE_GATE, "0012: guard idle quiesce must default on (is VLLM_E2_GUARD_IDLE=0 set?)"
_gw = open(importlib.util.find_spec("vllm.v1.worker.gpu_worker").origin, encoding="utf-8").read()
for _m in ("def sample_tokens(", "def execute_model(", "def execute_dummy_batch("):
    _head = _gw[: _gw.index(_m)].rsplit("\n\n", 1)[-1]
    assert "@guard_telemetry.engine_call" in _head, f"0012 missing (engine_call on {_m})"
assert all(c in _gw for c in ("guard_telemetry.enable_idle_gate(", "guard_telemetry.stop_reader()", "guard_telemetry.suspend_reader(", "guard_telemetry.resume_reader()")), "0012 r2 missing (worker wiring)"
assert envs.VLLM_HOST_EMBED_TABLE is None, "host-embed knob must be unset (family default)"
assert envs.host_embed_table_enabled_for_model("Qwen3_5ForConditionalGeneration"), "host embedding not on for the 27B"
assert not envs.host_embed_table_enabled_for_model("LlamaForCausalLM"), "host embedding leaks to other families"
assert envs.VLLM_FLASHINFER_PLAN_CACHE, "plan cache default not on"
assert envs.VLLM_GDN_BA_SPLITK, "split-K default not on"
assert envs.VLLM_DRAFT_SEQ_LENS_NOSYNC, "draft seq_lens no-sync default not on"
assert not envs.VLLM_DRAFT_SEQ_LENS_VERIFY, "verify must default off"
print("ENGINE OK: vllm", vllm.__version__, "torch", torch.__version__, "flashinfer", flashinfer.__version__)
PY
"$VENV/bin/python" -m pip check
echo "INSTALL OK $VENV"
