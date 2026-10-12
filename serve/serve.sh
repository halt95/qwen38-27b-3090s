#!/usr/bin/env bash
# The serve line for Qwen3.8-27B on two RTX 3090s (TP=2, 262K context), Merlin engine
# v2.5.2 + patches/merlin-engine-v2.5.2 (27B series 0001-0011), installed by release/install-env.sh.
# This is the command the reference host runs, with the host-specific pieces lifted into variables. Every flag was measured; see
# README.md and benchmarks/.
#
#   MODEL=/path/to/checkpoint PORT=8100 bash serve/serve.sh
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
GPUS=${GPUS:-0,1}                                 # the two cards for TP=2
KV_BYTES=${KV_BYTES:-14756413440}                 # per-GPU KV pool pin = 883 whole blocks (below): 826,689 tokens, 3.15x 262K, at K=4
LPT=${LPT:-832}                                   # --long-prefill-token-threshold: caps one request's prefill chunk per step so
                                                  # short requests aren't held behind a long cold prefill; LPT=0 turns it off
SPEC_K=${SPEC_K:-4}                               # MTP draft tokens per step. 4 = default (best single-stream thinking decode);
                                                  # 3 and 1 remain measured options (1 is faster for 8+ concurrent decode-heavy streams on v1.0.0)
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
# RecoverSSM commit (0011): replay the post-sampling GDN commit from CUDA graphs captured at boot, enqueued after the MTP
# proposal. Both only act with --use-replayssm (inert otherwise). Engine defaults stay off; the release runs the measured
# arm (GPU run 2026-10-08: greedy tokens identical to the eager commit; with 0011's metadata reuse +3.6-4.0 % decode vs the
# v1.3.0 RecoverSSM engine). Set VLLM_RECOVERSSM_COMMIT_GRAPH=eager / VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE=0 to opt out.
export VLLM_RECOVERSSM_COMMIT_GRAPH=${VLLM_RECOVERSSM_COMMIT_GRAPH-graph}
export VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE=${VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE-1}

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

# KV pin, TP=2, K=4, RecoverSSM on (--use-replayssm): the attention block is 960 tokens and one pool block is 16,711,680 B per card
# (17,408 B/token x 960). 883 blocks = 14,756,413,440 B. vLLM logs 826,689 tokens = floor(883 x 262,144 / D), hybrid capacity (not
# 883 x 960 attention slots): 3.15x 262K, i.e. three 262K sessions (841 blocks incl. null) OR six 131K sessions (6 x 143 + 1 = 859) resident.
# One 262K session needs D = 280 blocks (274 attention + 3 GDN groups x 2) plus 1 reserved null block. The pin is a whole number of
# blocks so no bytes are wasted; it is a K=4 figure (the block size depends on SPEC_K; K=3 and K=1 are not re-measured with RecoverSSM).
# --prefix-match-unit 16: 16 divides the 960-token block (it also divided 832 and 848, the blocks without RecoverSSM; 64 does not divide 848).
# --enable-mamba-fine-grained-prefix-cache: fine-grained hybrid-state prefix reuse. --use-replayssm: RecoverSSM, the recurrent state is
# recovered by replay, which is what keeps three 262K or six 131K sessions resident. --max-num-seqs 16 --max-num-batched-tokens 2048.
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 2 --attention-backend FLASHINFER \
  --max-model-len 262144 --gpu-memory-utilization 0.96 \
  --kv-cache-dtype fp8_e4m3 "${kv[@]}" --mamba-ssm-cache-dtype float16 \
  --max-num-seqs 16 --max-num-batched-tokens 2048 --enable-prefix-caching --enable-prompt-tokens-details "${lpt[@]}" \
  --prefix-match-unit 16 --enable-mamba-fine-grained-prefix-cache --use-replayssm \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":'"$SPEC_K"',"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":'"$capture"'}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --default-chat-template-kwargs "$THINKING" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":64,"video":0}' --mm-processor-cache-gb 0 \
  "${extra[@]}" \
  --host "$HOST" --port "$PORT" --shutdown-timeout 60
