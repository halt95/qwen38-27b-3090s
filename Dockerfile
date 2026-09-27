# Qwen3.8-27B v1.0.0 in one image: stock vLLM 0.30.0 from PyPI + the patches/vllm-0.30.0 series, installed by the
# release's own route (release/install-env.sh, hash-locked release/requirements-*.txt), served by serve/serve.sh
# (TP=2) or serve/serve-tp4.sh (TP=4) through serve/docker-entrypoint.sh. The checkpoint is mounted, never baked in.
#
# Build from the repository root (the build context is the repo; .dockerignore keeps it to the files listed below):
#
#   docker build -t qwen38-27b-3090s:v1.0.0 .
#
# Run (TP=2 on two cards; README.md "Quick start (container)" has TP=4 and compose):
#
#   docker run --gpus '"device=0,1"' --ipc=host --ulimit memlock=-1 --stop-timeout 70 -p 8100:8100 \
#     -v /path/to/checkpoint:/model:ro -v qwen38-27b-cache:/cache qwen38-27b-3090s:v1.0.0
#
# Runtime kernel compilation. The build compiles nothing, but the FIRST SERVE does: FlashInfer compiles its kernels
# with nvcc and links -lcudart from $CUDA_HOME/lib64, and Triton compiles its launchers with a C compiler against
# Python.h. So the image carries gcc/g++ and ninja (the python:3.13 base ships Python.h), the venv carries the CUDA
# 13.0 nvcc/crt/nvvm/cccl wheels (release/requirements-container-toolkit.txt: PTX any CUDA >= 13.0 driver accepts; the
# main lock's CUDA 13.4 toolkit would be rejected by an older driver), and the step after that adds the lib64 /
# unversioned library links FlashInfer needs. CUDA_HOME points at that toolkit; serve/docker-entrypoint.sh checks its nvcc against the host
# driver before starting (kernels from an nvcc newer than the driver are rejected with "Unsupported .version").
#
# Host needs: NVIDIA driver for CUDA 13.0 or newer (see README.md "Driver and toolkit" for the nvcc-versus-driver rule), nvidia-container-toolkit
# registered with Docker (nvidia-ctk runtime configure --runtime=docker, then restart docker), 2 or 4 RTX 3090s.

# python 3.13.15 on Debian 12 (glibc 2.36); pinned by digest (multi-arch index; the image is linux/amd64 only)
FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26

LABEL org.opencontainers.image.title="qwen38-27b-3090s" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="Qwen3.8-27B on vLLM 0.30.0 + patch series, 2 or 4 RTX 3090s (checkpoint mounted at /model)" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="docker.io/library/python:3.13-slim-bookworm"

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# git: install-env.sh applies the patch series with `git apply`. gcc/g++/libc6-dev/ninja: first-serve JIT (above).
# procps: ps/pgrep for debugging a running container. curl: the HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl procps gcc g++ libc6-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qwen38-27b
# Only what the install and serve routes read (.dockerignore allow-lists the same set). The scripts get mode 0755
# whatever the build context's checkout did (a 100644 checkout died with "Permission denied" before), and every
# caller below also invokes them through bash. (No COPY --chmod: the legacy builder rejects it.)
COPY LICENSE ./
COPY release/install-env.sh release/requirements-pinned.txt release/requirements-pip.txt \
     release/requirements-container-toolkit.txt release/
COPY patches/vllm-0.30.0/ patches/vllm-0.30.0/
COPY serve/ serve/
RUN chmod 0755 release/install-env.sh serve/*.sh

# The published install route, verbatim: venv at /opt/qwen38-27b/.venv (serve.sh's default VENV), every package hash-checked,
# the patch series applied and asserted, pip check.
RUN bash release/install-env.sh /opt/qwen38-27b/.venv /usr/local/bin/python3.13 \
    && rm -rf /root/.cache

# The first-serve toolkit at CUDA 13.0 (see the top of this file), hash-checked, then the environment re-checked.
RUN /opt/qwen38-27b/.venv/bin/pip install --no-deps --require-hashes -r release/requirements-container-toolkit.txt \
    && /opt/qwen38-27b/.venv/bin/pip check \
    && rm -rf /root/.cache

# Runtime-JIT layout: the NVIDIA CUDA wheels install bin/, include/ and lib/ with versioned libraries only
# (libcudart.so.13, ...); FlashInfer builds with -L$CUDA_HOME/lib64 -lcudart. Idempotent links, then assert the result.
RUN set -eu; \
    CU=/opt/qwen38-27b/.venv/lib/python3.13/site-packages/nvidia/cu13; \
    [ -x "$CU/bin/nvcc" ] || { echo "no nvcc wheel at $CU"; exit 1; }; \
    [ -e "$CU/lib64" ] || ln -s lib "$CU/lib64"; \
    for so in "$CU"/lib/lib*.so.[0-9]*; do \
      [ -e "$so" ] || continue; base="${so%%.so.*}.so"; [ -e "$base" ] || ln -s "$(basename "$so")" "$base"; \
    done; \
    [ -e "$CU/lib64/libcudart.so" ] || { echo "$CU/lib64/libcudart.so missing"; exit 1; }; \
    "$CU/bin/nvcc" --version | tail -1

ENV MODEL=/model HOST=0.0.0.0 PORT=8100 TP=2 \
    VENV=/opt/qwen38-27b/.venv \
    CUDA_HOME=/opt/qwen38-27b/.venv/lib/python3.13/site-packages/nvidia/cu13
ENV PATH="/opt/qwen38-27b/.venv/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH"
# Every compile cache goes to the /cache volume, so a re-created container reuses the first boot's work:
# vLLM's torch.compile / cudagraph cache, FlashInfer's JIT kernels (under $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer)
# and Triton's cache. The checkpoint is local, so the Hub is never contacted; serve.sh already turns usage stats off.
ENV VLLM_CACHE_ROOT=/cache/vllm FLASHINFER_WORKSPACE_BASE=/cache/flashinfer TRITON_CACHE_DIR=/cache/triton \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
VOLUME /cache
EXPOSE 8100

# A real one-token generation, not /v1/models: that endpoint answers 200 as soon as the HTTP server listens and keeps
# answering after the engine has died. The first boot compiles graphs and kernels (~7 min), hence the start period;
# the timeout is generous because a probe can queue behind a long prefill.
HEALTHCHECK --interval=60s --timeout=90s --start-period=30m --retries=3 \
  CMD curl -fs -m 85 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" ${VLLM_API_KEY:+-H "Authorization: Bearer $VLLM_API_KEY"} \
      -d "{\"model\":\"${SERVED:-qwen38-27b}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      | grep -q '"choices"' || exit 1

ENTRYPOINT ["bash", "/opt/qwen38-27b/serve/docker-entrypoint.sh"]
CMD ["/model"]
