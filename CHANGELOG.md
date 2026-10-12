# Changelog

Release notes, newest first. To install and run the current release, see the [README](README.md); reference detail
(scripts, serve-script variables and knobs, capacity, known behaviours, measurement method) is in
[docs/reference.md](docs/reference.md); v1.0.0 and its numbers are in [docs/history.md](docs/history.md).

## v1.3.0

Released 2026-10-12. Dual and quad card only. The single card profile is parked for this release: no single card serve
script ships. The single card profile was gated as v1.1.0 (below) but not published; it returns in a later release.
Release gate 2026-10-08 to 2026-10-09 (UTC); bench cards: [`benchmarks/2026-10-12/BENCH-CARD.md`](benchmarks/2026-10-12/BENCH-CARD.md)
and, for the ThinkingCap profile, [`benchmarks/2026-10-12/ThinkingCap-BENCH-CARD.md`](benchmarks/2026-10-12/ThinkingCap-BENCH-CARD.md).
To run it: [Quick start](README.md#quick-start), or [Build and serve](docs/reference.md#build-and-serve) for the detail.

### In brief

- **New engine.** Merlin engine v2.5.2 plus the 27B series 0001-0011 (`patches/merlin-engine-v2.5.2/`) replaces stock
  vLLM 0.30.0 plus `patches/vllm-0.30.0/`.
- **Two checkpoints, one repository.** The new ThinkingCap-Qwen3.8-27B checkpoint profile serves BottleCap AI's
  reasoning fine-tune on the same engine and pins, at `xhigh` reasoning effort by default; its weights are under the
  PolyForm Small Business License 1.0.0 plus BottleCap AI's personal-use permission.
- **Faster decode at depth.** The FlashInfer 32-row verify tile and MTP K=4 by default: single-stream thinking decode at
  4K / 32K / 128K / 255K is 206 / 198 / 172 / 144 tok/s on dual card and 258 / 249 / 238 / 226 on quad card (release gate,
  2026-10-08 to 2026-10-09 (UTC)).
- **Sessions stay resident.** Dual card runs RecoverSSM with an 826,689-token pool (three 262K or six 131K sessions, 0
  preemptions); quad card a 1,974,072-token pool (seven 262K sessions, 0 preemptions).
- **Idle cards drop to P8.** The engine's bounds-guard reader parks while the engine is idle (`VLLM_E2_GUARD_IDLE`).
- **Cached tokens in every response** (`--enable-prompt-tokens-details`) and a per-image pixel cap (`MAX_PIXELS`).
- **Combining marks tokenize as upstream.** `tokenizer.json` is the upstream file and transformers is pinned to 5.18.0.
  No weights changed.
- **Bare metal only.** The container route is removed for this release.

### What changed in v1.3.0

- Engine: Merlin engine v2.5.2 (the Flash-Next v2.5.1 base, vLLM 0.30.0 plus the Flash-Next fork commits, plus the guard
  fix below) plus the 27B series 0001-0011 (`patches/merlin-engine-v2.5.2/`), replacing stock vLLM 0.30.0 plus
  `patches/vllm-0.30.0/`. The guard fix was series patch 0012 at the release gate; the final `vllm/` tree is identical:
  - 0001 uniform multi-token decode on Ampere (port of part of vLLM PR #47979);
  - 0002 FlashInfer decode-plan cache;
  - 0003 host-resident embedding table, one `VLLM_HOST_EMBED_TABLE` knob (unset = on for the dense Qwen3.5 family only;
    "Qwen3.5" is the architecture class vLLM loads for Qwen3.8-27B, `Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration`);
  - 0004 GDN `in_proj_ba` split-K;
  - 0005 no device sync for the second MTP draft step;
  - 0006 plan cache only when the resolved FlashInfer backend is `fa2`; split-K only for the dense Qwen3.5 architectures;
  - 0007 family isolation: every patch above applies only to a dense Qwen3.5 target on sm_86, other models take the stock
    path; plus plan-cache, draft-handoff and split-K fixes from the code audit;
  - 0008 RecoverSSM support for the dense 27B (opt-in `--use-replayssm`; on in `serve/serve.sh`, off in `serve/serve-tp4.sh`);
  - 0009 the family gate asks the device the process runs on (matters only on mixed-GPU hosts).
  - 0010 the family gate fails closed when the executing device is unknown and takes the caller's device; the MTP draft seq_lens handoff is published only after a metadata build succeeds.
  - 0011 RecoverSSM commit: the GDN metadata builder reuses per-step metadata, and the post-sampling commit can replay from
    CUDA graphs and run after the MTP proposal (`VLLM_RECOVERSSM_COMMIT_GRAPH`, `VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE`, both
    on in `serve/serve.sh`); output bit-identical, with a lower per-step cost for the commit on dual card.
  - engine guard fix (ex-series 0012): the bounds-guard reader parks with no device work while the engine is idle (idle cards no longer held in P2),
    after a final read of every counter once the last step's GPU work has completed; `VLLM_E2_GUARD_IDLE=0` restores
    the always-polling reader, and both serve scripts set it explicitly. Run-time checks keep every guarded kernel on the
    fenced engine stream (a violation falls back to polling, loudly); fatal sites are armed on the first copy that shows
    the count; sleep mode suspends the reader. Detection while serving stays cadence-bound (about one to three 50 ms
    intervals, then the next step), as before the fix. Thanks to CoolShades for the idle-power report in Flash-Next
    [issue #2](https://github.com/halt95/qwen38-flash-next-3090s/issues/2) and for [PR #3](https://github.com/halt95/qwen38-flash-next-3090s/pull/3),
    which proposed pausing the reader after a fixed idle timeout; the release parks it once the engine has drained
    instead, the same implementation as Flash-Next v2.5.2.
- Both serve scripts pass `--enable-prompt-tokens-details`, so every response reports
  `usage.prompt_tokens_details.cached_tokens` (the prompt tokens served from the prefix cache). Response metadata only:
  output, memory and speed are unchanged.
- Per-image pixel cap `MAX_PIXELS` (default 4,194,304 = 2048×2048; larger images are downscaled before the vision encoder). v1.0.0 had no cap on two or four cards; at the v1.3.0 TP2 pin a large uncapped image can run the vision encoder out of memory.
- FlashInfer 32-row verify tile for the MTP verify step (q_len 4 and 5, fp8 KV, sm_86), on by default through
  `FLASHINFER_SM8X_FP8_SMALLQ=cta32` in both serve scripts; set the variable to the empty string to run stock FlashInfer.
  It needs FlashInfer patched by `release/apply-flashinfer-patch.sh` (whole-file sha256 of the five touched files, pristine
  and patched). Each rank prints the proof line `[flashinfer-sm8x-tile] smallq=cta32 q_len=5 rows=30 cta_tile_q=32 device=<d>`
  when the first verify plan takes the tile; `release/check-tile-proof.sh <log> --tp N` checks devices 0..N-1 once each. The patch's JIT modules carry the suffix
  `_sm8x_smallq_v5`.
- MTP `SPEC_K` default is now 4 (was 3); 3 and 1 remain accepted. Capture sizes follow K: for K=4
  `[1,2,4,5,8,10,15,20,25,30,35,40]` (1, 2, 4 and 8, then multiples of K+1 = 5), the previous list for K=1 and K=3. New `--prefix-match-unit 16` in both scripts: the
  hybrid KV block is 848 tokens at K=4 (960 on TP=2 with RecoverSSM) and 64 does not divide it; 16 divides 832, 848 and 960.
- Serve configs frozen 2026-10-08 (on the evidence of the frozen-config runs of that day):
  - TP=2 (`serve/serve.sh`): `--use-replayssm` and `--enable-mamba-fine-grained-prefix-cache`; `--max-num-seqs 16
    --max-num-batched-tokens 2048` (were 32 / 4096); `KV_BYTES` 14,756,413,440 = 883 whole blocks of 16,711,680 B (block 960
    tokens): 826,689 tokens of hybrid capacity (floor(blocks × 262,144 / D), D = 280 blocks per 262K session), 3.15× 262K;
    three 262K or six 131K sessions resident, 0 preemptions.
  - TP=4 (`serve/serve-tp4.sh`): `--enable-mamba-fine-grained-prefix-cache` (no RecoverSSM); `--max-num-seqs 24
    --max-num-batched-tokens 2048` (were 32 / 4096); `KV_BYTES` 18,231,050,240 = 2,470 whole blocks of 7,380,992 B (block 848
    tokens): 1,974,072 tokens (floor(blocks × 262,144 / D), D = 328), 7.53×; seven 262K sessions resident, 0 preemptions, prefix-cache hit 88.6 % on the 8-conversation load at the release gate.
  - Unchanged: fp16 SSM state (`--mamba-ssm-cache-dtype float16`), `expandable_segments`, `--disable-custom-all-reduce`,
    `LPT=832`, the chat template. The old KV pins (14,534,914,979 and 17,832,100,834) are replaced.
- Single-stream decode, thinking on, K=4 (effort low, 256-token generations), tok/s at 4K / 32K / 128K / 255K (the 255K rung targets 261,120 tokens; 259,587 as sent):
  TP=2 206 / 198 / 172 / 144, TP=4 258 / 249 / 238 / 226 (release gate, 2026-10-08 to 2026-10-09 (UTC)). The gate
  measured the shipped profile (tile on, K=4) only.
- Checkpoint: `tokenizer.json` is replaced by the upstream Qwen3.8-27B file (sha256 `0997f410…`; the previous file lacked
  `\p{M}` in its pre-tokenizer); `release/checkpoint.sha256` lists it. No weights changed. transformers is pinned to 5.18.0
  (5.17.0 and earlier split combining-mark scripts differently from upstream when serving).
- Bare metal only: v1.3.0 ships no container route. `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `serve/docker-entrypoint.sh`
  and `release/requirements-container-toolkit.txt` are removed with the README container quick start; the container route
  returns in a later release (the `v1.0.0` tag still has the old recipe, for the old engine).
- New ThinkingCap-Qwen3.8-27B checkpoint profile (`halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin`, weights under the PolyForm
  Small Business License 1.0.0 plus BottleCap AI's personal-use permission): served via `serve/serve-thinkingcap.sh` (TP=2)
  and `serve/serve-thinkingcap-tp4.sh` (TP=4), `xhigh` reasoning effort by default; checkpoint list
  `release/checkpoint-thinkingcap.sha256`, licence copies in `licenses/`, bench card `benchmarks/2026-10-12/ThinkingCap-BENCH-CARD.md`.
  ThinkingCap has no version of its own: it is the ThinkingCap profile of v1.3.0.

### Upgrading from v1.0.0

- Install into a new venv: `release/install-env.sh <new path>` builds Merlin engine v2.5.2 plus the 27B series
  0001-0011 and the FlashInfer tile patch (it needs the two engine assets named in `upstream/PIN`; see the README Quick
  start). It refuses an existing venv, and the v1.0.0 environment (stock vLLM 0.30.0 plus `patches/vllm-0.30.0/`) does not
  serve the v1.3.0 scripts.
- Re-download `tokenizer.json` from the checkpoint repository: it is now the upstream file (sha256 `0997f410…`). A download
  made before this release still has `f913b92a…` and fails that line of `release/checkpoint.sha256`. No weights changed.
- Breaking or changed defaults: `SPEC_K` 3 → 4; `--max-num-seqs` 32 → 16 (TP=2) and 24 (TP=4); `--max-num-batched-tokens`
  4096 → 2048; new `--prefix-match-unit 16`; new KV pins (14,756,413,440 and 18,231,050,240 B per card, were 14,534,914,979
  and 17,832,100,834); RecoverSSM on in `serve/serve.sh`; FlashInfer verify tile on; `MAX_PIXELS` cap 4,194,304;
  transformers pinned to 5.18.0.
- The container route is removed in this release (`Dockerfile`, `docker-compose.yml`, `.dockerignore`,
  `serve/docker-entrypoint.sh`, `release/requirements-container-toolkit.txt`); the `v1.0.0` tag keeps it for the old engine.

## v1.1.0

Single-card profile: gated, not published.

## v1.0.0

Released 2026-09-27. Numbers: [docs/history.md](docs/history.md#v100-on-stock-vllm-0300-k3); bench card:
[`benchmarks/2026-09-26/BENCH-CARD.md`](benchmarks/2026-09-26/BENCH-CARD.md).

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
