#!/usr/bin/env bash
# ThinkingCap-Qwen3.8-27B (halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin) on two RTX 3090s (TP=2): serve/serve.sh with one
# default changed, the reasoning effort (xhigh instead of medium). Every other flag, pin and variable is serve/serve.sh's and
# passes through unchanged; a THINKING you set yourself wins.
#
#   MODEL=/path/to/thinkingcap-checkpoint PORT=8100 bash serve/serve-thinkingcap.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export THINKING=${THINKING:-'{"enable_thinking": true, "reasoning_effort": "xhigh"}'}
exec bash "$HERE/serve.sh" "$@"
