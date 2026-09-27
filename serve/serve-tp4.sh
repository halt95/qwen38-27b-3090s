#!/usr/bin/env bash
# The serve line for Qwen3.8-27B on FOUR RTX 3090s (TP=4, 262K context), vLLM 0.30.0 +
# patches/vllm-0.30.0, installed by release/install-env.sh. This is the command the reference
# host runs, with the host-specific pieces lifted into variables. The defaults (SPEC_K=3, LPT=832, the
# same as serve.sh) and SPEC_K=1 were measured at TP=4 on v1.0.0; see README.md and benchmarks/.
#
#   MODEL=/path/to/checkpoint PORT=8100 bash serve/serve-tp4.sh
#
# Patch knobs are NOT set here on purpose: the series ships them on (VLLM_HOST_EMBED_TABLE=1,
# VLLM_FLASHINFER_PLAN_CACHE=1), so this script runs the shipped defaults. Export either as 0
# to turn it off.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:?path to the checkpoint dir (W4A16 AWQ asym g128 + int4 head/MTP + calibrated E4M3 KV scales)}
PORT=${PORT:-8100}
HOST=${HOST:-127.0.0.1}
SERVED=${SERVED:-qwen38-27b}
VENV=${VENV:-$(cd "$HERE/.." && pwd)/.venv}       # built by release/install-env.sh (README: <repo>/.venv)
GPUS=${GPUS:-0,1,2,3}                             # all four cards (measured with four-way P2P)
KV_BYTES=${KV_BYTES:-17832100834}                 # per-GPU KV pool pin at TP=4: 1,949,844 tokens, ~1 GB free per card (262K-verified)
LPT=${LPT:-832}                                   # --long-prefill-token-threshold: caps one request's prefill chunk per step so
                                                  # short requests aren't held behind a long cold prefill; LPT=0 turns it off
SPEC_K=${SPEC_K:-3}                               # MTP draft tokens per step. At TP=4, 1 is faster for 2-8 concurrent
                                                  # decode-heavy streams and 3 at 32 (measured); 3 is the default
# Chat template: the vendored froggeric v22.5 fix-up of the Qwen template (Apache-2.0, see
# NOTICE). Set CHAT_TEMPLATE= (empty) to use the checkpoint's own template instead.
CHAT_TEMPLATE=${CHAT_TEMPLATE-$HERE/templates/froggeric-qwen-v22.5.jinja}
THINKING=${THINKING:-'{"enable_thinking": true, "reasoning_effort": "medium"}'}

export CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_ATTENTION_BACKEND=FLASHINFER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}              # FlashInfer JIT
export VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS            # P2P host driver; drop if your cards lack P2P
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1} DO_NOT_TRACK=${DO_NOT_TRACK:-1}   # no usage telemetry; set 0 to opt in

extra=()
[ -n "$CHAT_TEMPLATE" ] && extra+=(--chat-template "$CHAT_TEMPLATE")
# KV_BYTES=auto drops the pin and lets vLLM size the pool from --gpu-memory-utilization; use it
# on cards other than 24 GB, or to re-derive a pin.
case "$SPEC_K" in 1|3) ;; *) echo "SPEC_K must be 1 or 3 (the measured values)" >&2; exit 2 ;; esac
if [ "$LPT" = 0 ]; then lpt=(); else lpt=(--long-prefill-token-threshold "$LPT"); fi
if [ "$KV_BYTES" = auto ]; then kv=(); else kv=(--kv-cache-memory-bytes "$KV_BYTES"); fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 4 --attention-backend FLASHINFER \
  --max-model-len 262144 --gpu-memory-utilization 0.96 \
  --kv-cache-dtype fp8_e4m3 "${kv[@]}" --mamba-ssm-cache-dtype float16 \
  --max-num-seqs 32 --max-num-batched-tokens 4096 --enable-prefix-caching "${lpt[@]}" \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":'"$SPEC_K"',"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,12,16,20,24,28,32]}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --default-chat-template-kwargs "$THINKING" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":64,"video":0}' --mm-processor-cache-gb 0 \
  "${extra[@]}" \
  --host "$HOST" --port "$PORT" --shutdown-timeout 60
