#!/usr/bin/env bash
# Container entrypoint for Qwen3.8-27B v1.0.0: checks the mounted checkpoint, the visible GPUs, the first-serve
# toolchain against the host driver and /dev/shm, then execs serve/serve.sh (TP=2) or serve/serve-tp4.sh (TP=4).
#
#   docker run ... qwen38-27b-3090s:v1.0.0 [/path/inside/container/to/checkpoint]
#
# Serve knobs are the serve scripts' own environment variables (TP, PORT, SERVED, GPUS, KV_BYTES, LPT, SPEC_K,
# CHAT_TEMPLATE, THINKING, VLLM_API_KEY): pass them with -e. The serve scripts take no extra vllm arguments.
#
# Every probe below is guarded: a missing or failing nvidia-smi skips its check with a note, it never ends the
# script silently under `set -e`.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $# -gt 0 ]; then
  case "$1" in
    -*) echo "extra vllm arguments are not supported; set the serve knobs with -e (see README.md, Quick start (container))" >&2; exit 2 ;;
    "") shift ;;
    *) MODEL="$1"; shift ;;
  esac
fi
[ $# -eq 0 ] || { echo "unexpected arguments: $* (the only argument is the checkpoint path)" >&2; exit 2; }
MODEL="${MODEL:-/model}"
export MODEL

TP="${TP:-2}"
case "$TP" in
  2) SERVE="$REPO/serve/serve.sh" ;;
  4) SERVE="$REPO/serve/serve-tp4.sh" ;;
  *) echo "TP must be 2 or 4 (the measured configurations), got '$TP'" >&2; exit 2 ;;
esac

if [ ! -f "$MODEL/config.json" ]; then
  echo "checkpoint not found at $MODEL (no config.json)" >&2
  echo "mount it: -v /path/to/checkpoint:/model:ro" >&2
  exit 1
fi
if ! ls "$MODEL"/*.safetensors >/dev/null 2>&1; then
  echo "no *.safetensors in $MODEL: is the mount the checkpoint directory itself?" >&2
  exit 1
fi

# GPUs. nvidia-smi exists only when the NVIDIA container runtime injected it (docker run --gpus ...).
ngpu=""
if command -v nvidia-smi >/dev/null 2>&1; then
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
fi
if [ -z "$ngpu" ] || [ "$ngpu" = 0 ]; then
  echo "WARNING: no NVIDIA GPU visible in the container. Start it with --gpus (e.g. --gpus '\"device=0,1\"'), with"
  echo "         nvidia-container-toolkit on the host, registered with:"
  echo "         sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
elif [ "$ngpu" -lt "$TP" ]; then
  echo "TP=$TP needs $TP GPUs but the container sees $ngpu (pass them with --gpus)" >&2
  exit 1
fi

# First-serve toolchain: FlashInfer compiles with the nvcc at CUDA_HOME; kernels from an nvcc newer than the driver's
# CUDA version are rejected at load ("Unsupported .version"). Refuse that combination up front.
[ -x "${CUDA_HOME:-}/bin/nvcc" ] || { echo "no nvcc at CUDA_HOME=${CUDA_HOME:-unset}" >&2; exit 1; }
[ -e "$CUDA_HOME/lib64/libcudart.so" ] || { echo "$CUDA_HOME/lib64/libcudart.so missing" >&2; exit 1; }
nv_rel="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' | head -1 || true)"
drv_rel=""
if command -v nvidia-smi >/dev/null 2>&1; then
  drv_rel="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\2/p' | head -1 || true)"
fi
if [ -z "$nv_rel" ] || [ -z "$drv_rel" ]; then
  echo "note: could not read the nvcc (${nv_rel:-?}) or driver (${drv_rel:-?}) CUDA version; the nvcc-versus-driver check is skipped"
elif [ "$(printf '%s\n%s\n' "$nv_rel" "$drv_rel" | sort -V | tail -1)" != "$drv_rel" ]; then
  echo "nvcc at $CUDA_HOME is CUDA $nv_rel but the driver supports CUDA $drv_rel: kernels compiled on the first" >&2
  echo "serve would be rejected. Upgrade the host driver, or mount a CUDA toolkit no newer than $drv_rel and point" >&2
  echo "CUDA_HOME at it (README.md, 'Driver and toolkit')." >&2
  exit 1
fi

# Shared memory and pinned memory: warnings, not failures.
shm_kb="$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2{print $2}' || true)"
case "$shm_kb" in ''|*[!0-9]*) shm_kb="" ;; esac
if [ -n "$shm_kb" ] && [ "$shm_kb" -lt 1048576 ]; then
  echo "WARNING: /dev/shm is $((shm_kb / 1024)) MB; tensor-parallel workers need more (docker run --ipc=host, or --shm-size=8g)"
fi
memlock="$(ulimit -l 2>/dev/null || true)"
if [ -n "$memlock" ] && [ "$memlock" != unlimited ]; then
  echo "WARNING: locked-memory limit is ${memlock} KB; the host-resident embedding table uses pinned memory (docker run --ulimit memlock=-1)"
fi

echo "serving $MODEL with TP=$TP ($(basename "$SERVE")) on ${HOST:-127.0.0.1}:${PORT:-8100}"
exec bash "$SERVE"
