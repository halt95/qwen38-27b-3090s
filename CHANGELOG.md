# Changelog

## v1.0.0 (2026-09-27)

First public release. There is no earlier public version and no upgrade path.

- Stock vLLM 0.30.0 from PyPI plus patches 0001-0005 (`patches/vllm-0.30.0/`), all on by default,
  each with an off switch except 0001: uniform multi-token decode on Ampere (a port of part of
  vLLM PR #47979), FlashInfer decode-plan cache (`VLLM_FLASHINFER_PLAN_CACHE`), host-resident
  embedding table (`VLLM_HOST_EMBED_TABLE`), GDN `in_proj_ba` split-K (`VLLM_GDN_BA_SPLITK`),
  no device sync for the second MTP draft step (`VLLM_DRAFT_SEQ_LENS_NOSYNC`).
- Serve profiles: `serve/serve.sh` (TP=2, 262K, KV pool 794,351 tokens) and
  `serve/serve-tp4.sh` (TP=4, 262K, KV pool 1,949,844 tokens), with FP8 E4M3 KV and pinned KV
  bytes.
- `--long-prefill-token-threshold 832` on by default (`LPT=0` turns it off).
- `SPEC_K` = 1 or 3 MTP draft tokens per step, default 3.
- Chat template: froggeric's Qwen-Fixed-Chat-Templates v22.5, vendored unmodified.
- vLLM usage statistics off by default in both serve scripts.
- Install route: `release/install-env.sh`, hash-locked pip and requirements
  (`--require-hashes`), patch markers asserted, `pip check`.
- Checkpoint: `halt95/Qwen3.8-27B-W4A16-Merlin` (11 files, checksums in
  `release/checkpoint.sha256`).
- Container recipe: `Dockerfile`, `docker-compose.yml`, `.dockerignore` and
  `serve/docker-entrypoint.sh` build one image by `release/install-env.sh` and serve
  `serve/serve.sh` (TP=2) or `serve/serve-tp4.sh` (TP=4), with the checkpoint mounted at
  `/model` and compile caches in a `/cache` volume. Its first-serve toolkit is pinned to CUDA 13.0
  (`release/requirements-container-toolkit.txt`), so any CUDA 13.0+ driver accepts the JIT kernels.
  Built and its entrypoint checked on a host without GPUs; not yet run on GPUs inside a container.
