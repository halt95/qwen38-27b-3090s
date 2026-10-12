#!/usr/bin/env bash
# The serve line for Qwen3.8-27B on FOUR RTX 3090s (TP=4, 262K context), Merlin engine
# v2.5.2 + patches/merlin-engine-v2.5.2 (27B series 0001-0011), installed by release/install-env.sh.
# This is the command the reference host runs, with the host-specific pieces lifted into variables. The defaults (SPEC_K=4, LPT=832, the
# same as serve.sh) and SPEC_K=3 / SPEC_K=1 were measured at TP=4; see README.md and benchmarks/.
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
KV_BYTES=${KV_BYTES:-18231050240}                 # per-GPU KV pool pin = 2,470 whole blocks (below): 1,974,072 tokens, 7.53x 262K, at K=4
LPT=${LPT:-832}                                   # --long-prefill-token-threshold: caps one request's prefill chunk per step so
                                                  # short requests aren't held behind a long cold prefill; LPT=0 turns it off
SPEC_K=${SPEC_K:-4}                               # MTP draft tokens per step. 4 = default (best single-stream thinking decode);
                                                  # 3 and 1 remain measured options (v1.0.0 at TP=4: 1 was faster for 2-8 concurrent
                                                  # decode-heavy streams and 3 at 32)
MAX_PIXELS=${MAX_PIXELS-4194304}                  # per-image pixel cap: larger images are downscaled to fit before the vision
                                                  # encoder (4,194,304 px = 2048x2048, as Flash-Next). Uncapped, a 4K image ran
                                                  # the encoder out of memory at the TP2 pin (10-07 probe). Empty = no cap.
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
# FlashInfer 32-row verify tile for the MTP verify step (q_len 4|5 = K=3|4, fp8 KV, sm_86 only; other shapes and
# archs take the stock path). Needs FlashInfer patched by release/apply-flashinfer-patch.sh; set it to the empty string (not unset) to opt out.
export FLASHINFER_SM8X_FP8_SMALLQ=${FLASHINFER_SM8X_FP8_SMALLQ-cta32}
# Only the shipped arms are served: "" (off), "0" (off) and cta32. The other arms in the FlashInfer patch (cta16) are benchmark
# experiments that failed their error budget; they are not reachable through the serve profiles.
case "$FLASHINFER_SM8X_FP8_SMALLQ" in "" | 0 | cta32) ;; *) echo "FLASHINFER_SM8X_FP8_SMALLQ must be empty, 0 or cta32 (got '$FLASHINFER_SM8X_FP8_SMALLQ')" >&2; exit 2 ;; esac
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1} DO_NOT_TRACK=${DO_NOT_TRACK:-1}   # no usage telemetry; set 0 to opt in
export VLLM_E2_GUARD_IDLE=${VLLM_E2_GUARD_IDLE:-1}   # engine guard fix (ex-series 0012): the guard reader parks while idle (cards reach P8); 0 = always poll

extra=()
[ -n "$CHAT_TEMPLATE" ] && extra+=(--chat-template "$CHAT_TEMPLATE")
# MAX_PIXELS is interpolated into a JSON number: it must be empty (no cap) or a canonical decimal (no leading zero, digits only)
# inside 65,536 (the shortest_edge below) .. 33,554,432; anything else exits 2 before vllm starts.
case "$MAX_PIXELS" in
  "") ;;
  0* | *[!0-9]*) echo "MAX_PIXELS must be empty or a canonical decimal pixel count (no leading zero, digits only)" >&2; exit 2 ;;
  *) if [ "${#MAX_PIXELS}" -gt 8 ] || [ "$MAX_PIXELS" -lt 65536 ] || [ "$MAX_PIXELS" -gt 33554432 ]; then
       echo "MAX_PIXELS must be between 65536 and 33554432 (got $MAX_PIXELS)" >&2; exit 2; fi ;;
esac
[ -n "$MAX_PIXELS" ] && extra+=(--mm-processor-kwargs "{\"size\":{\"longest_edge\":$MAX_PIXELS,\"shortest_edge\":65536}}")
# 4 is the default; 3 and 1 remain measured options. Anything else is unmeasured.
case "$SPEC_K" in 1|3|4) ;; *) echo "SPEC_K must be 1, 3 or 4 (the measured values; 4 is the default)" >&2; exit 2 ;; esac
# CUDA-graph capture sizes (decode tokens per step = sequences x (K+1)): at K=4, 1, 2, 4 and 8, then multiples of K+1 = 5, so the verify steps land on
# captured sizes. K=4 is the default; the K=3 / K=1 list is the one measured on v1.0.0 and v1.1.0.
if [ "$SPEC_K" = 4 ]; then capture='[1,2,4,5,8,10,15,20,25,30,35,40]'; else capture='[1,2,4,8,12,16,20,24,28,32]'; fi
if [ "$LPT" = 0 ]; then lpt=(); else lpt=(--long-prefill-token-threshold "$LPT"); fi
# KV_BYTES=auto drops the pin and lets vLLM size the pool from --gpu-memory-utilization; use it
# on cards other than 24 GB, or to re-derive a pin.
if [ "$KV_BYTES" = auto ]; then kv=(); else kv=(--kv-cache-memory-bytes "$KV_BYTES"); fi

# KV pin, TP=4, K=4, no RecoverSSM: the attention block is 848 tokens and one pool block is 7,380,992 B per card. 2,470 blocks x 7,380,992 B
# = 18,231,050,240 B. vLLM logs 1,974,072 tokens = floor(2,470 x 262,144 / D), hybrid capacity (not 2,470 x 848 attention slots):
# 7.53x 262K, seven 262K sessions resident. One 262K session needs D = 328 blocks (310 attention +
# 3 GDN groups x 6) plus 1 reserved null block, so 7 sessions need 2,297 blocks. The pin is a K=4 figure (the block size depends on SPEC_K).
# --prefix-match-unit 16: 16 divides the 848-token block (and 832 at K<=3; 64 does not divide 848).
# --enable-mamba-fine-grained-prefix-cache: fine-grained hybrid-state prefix reuse. --max-num-seqs 24 --max-num-batched-tokens 2048.
# RecoverSSM (--use-replayssm) is off at TP=4: it costs decode speed and seven 262K sessions already fit.
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 4 --attention-backend FLASHINFER \
  --max-model-len 262144 --gpu-memory-utilization 0.96 \
  --kv-cache-dtype fp8_e4m3 "${kv[@]}" --mamba-ssm-cache-dtype float16 \
  --max-num-seqs 24 --max-num-batched-tokens 2048 --enable-prefix-caching --enable-prompt-tokens-details "${lpt[@]}" \
  --prefix-match-unit 16 --enable-mamba-fine-grained-prefix-cache \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":'"$SPEC_K"',"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":'"$capture"'}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --default-chat-template-kwargs "$THINKING" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":64,"video":0}' --mm-processor-cache-gb 0 \
  "${extra[@]}" \
  --host "$HOST" --port "$PORT" --shutdown-timeout 60
