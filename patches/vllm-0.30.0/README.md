# Patch series for vLLM 0.30.0

Five patches against **vLLM 0.30.0 exactly as published on PyPI**
(`vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl`, sha256
`ef52ee58c410ead0b8afb190838fa4cbcb52075596f67862a03859d984966ac4`). Apply them in order
from the `site-packages` directory of the env that holds `vllm`:

```bash
SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cd "$SP"
for p in /path/to/patches/vllm-0.30.0/*.patch; do
  git apply --check --directory=vllm "$p" || { echo "does not apply: $p"; exit 1; }
  git apply --directory=vllm "$p"         || { echo "apply failed: $p"; exit 1; }
done
```

Patch paths are relative to the `vllm` package directory, hence `--directory=vllm`.
Check every exit code: an unchecked `--check` is how a half-applied tree gets served.

| # | what | default | off switch | files | proof line (boot log) |
|---|---|---|---|---|---|
| 0001 | port of part of vLLM PR #47979 (open draft upstream): uniform multi-token (MTP verify) decode on sm_86 | on | none (remove the patch) | `v1/attention/backends/flashinfer.py` | FULL_AND_PIECEWISE capture completes with MTP K=3 and no "not supported" cudagraph warning |
| 0002 | FlashInfer decode-plan cache (single entry; validated on sm_86 / FA2 only, set the off switch elsewhere) | on | `VLLM_FLASHINFER_PLAN_CACHE=0` | `v1/attention/backends/flashinfer.py`, `envs.py` | `flashinfer plan cache: call N, H hits / M misses` |
| 0003 | host-resident embedding table | **on** | `VLLM_HOST_EMBED_TABLE=0` | `envs.py`, `v1/worker/gpu/model_runner.py` | `HOST-RESIDENT EMBED TABLE: freed 1.184 GiB of VRAM on this rank` (TP=2) |
| 0004 | split-K GEMM for the GDN `in_proj_ba` at M <= 16 tokens | on | `VLLM_GDN_BA_SPLITK=0` | `envs.py`, `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, new `model_executor/layers/mamba/gdn/ba_splitk.py` | `GDN in_proj_ba SPLIT-K ACTIVE: M=.. N=.. K=5120 S=16` (once per rank) |
| 0005 | MTP draft step >= 2 takes the previous draft build's CPU `seq_lens` + 1 instead of a device sync | on | `VLLM_DRAFT_SEQ_LENS_NOSYNC=0` | `envs.py`, `v1/attention/backends/flashinfer.py`, `v1/worker/gpu/spec_decode/autoregressive/speculator.py` | `DRAFT SEQ_LENS NOSYNC ACTIVE: draft step 2 ...` (once per rank) |

Every knob is registered in `vllm/envs.py`, so it is part of vLLM's compile cache key: a
cached compile can't silently ignore a changed knob. 0005 also registers a diagnostic,
`VLLM_DRAFT_SEQ_LENS_VERIFY=1` (off by default): it syncs anyway, compares the derived lengths
with the real ones, logs `draft seq_lens verify: N builds verified, M mismatches` and, on a
mismatch, uses the synced value, so outputs are never affected.

## Read before applying

**0001 is only needed on Ampere, and was validated on sm_86 only.** The patch has no
architecture gate: it also activates on other GPUs where trtllm decode is off, and it hasn't
been tested there. 0.30 gates spec-as-decode on a trtllm/XQA decode kernel,
which is `None` on sm_86, so MTP verify batches take the prefill path and FULL cudagraphs
can't capture them. On Blackwell the upstream path is expected (unverified) to cover this and the patch
to be inert (not tested). It needs a FlashInfer whose `fast_decode_plan` takes `q_len_per_req` (0.6.18.post1
does); with an older FlashInfer it falls back to the upstream behaviour.

**0002 was validated on sm_86 with the FlashInfer FA2 backend only. On any other GPU or
FlashInfer backend, set `VLLM_FLASHINFER_PLAN_CACHE=0`.** The patch has no runtime backend
guard: it is on by default wherever it is applied. The key leaves out `kv_lens` for a
structural reason: the fa2 planner re-derives KV length from `kv_indptr` in pages. The SM90
and SM100 planners do read `kv_len_arr`, so on Hopper or Blackwell the key is unsound and
must be re-derived before use. The key includes the wrapper's resolved backend. It leaves out
the `last_page_len` values (it keys only their count) on purpose: the metadata builder copies
`paged_kv_last_page_len` to the GPU every step, and the cudagraph wrapper's buffer is that
same device slice, so a cache hit skips only the schedule, and the schedule is keyed on
`indptr`. The
single-entry design is load-bearing: `_plan_info` holds offsets into a workspace that keeps
only the last plan, so a dict cache would return valid-looking offsets into overwritten
bytes.

**0003 is on by default, and it was measured on PCIe Gen4 x16.** The gather crosses the bus
once per token. A narrower or slower link (Gen3, x4, a riser, an external enclosure) is where
it could cost rather than pay, and it hasn't been measured there. Freeing VRAM doesn't
enlarge the KV pool on its own: `--kv-cache-memory` is a hard pin, so raise it by the freed
bytes (the shipped `serve/serve.sh` already does).

**0004 was validated on sm_86 only.** It dispatches only at the shapes where cuBLAS on sm_86 is pathological
(M <= 16 rows against the 48-row per-rank weight at TP=2, 24 rows at TP=4) and only for an
unquantized, bias-free `in_proj_ba`; everything else takes `F.linear` unchanged. The reduction
order differs from cuBLAS (fp32 accumulate over 16 K-chunks), so outputs are not bit-identical to
a `VLLM_GDN_BA_SPLITK=0` run. On other GPUs it is untested.

**0005 is exact by construction, and verified.** `_update_draft_inputs_kernel` sets draft step 2's
lengths to draft step 1's + 1 on every row, padded rows included (the FULL-graph draft decode
replays an update kernel captured over the padded batch). A first build that bumped only the live
rows mismatched on padded rows at c8/c32; that is why every row is bumped. The target-verify and
draft-step-1 syncs stay: they depend on the rejection count, which lives on the GPU. It only
engages on the FlashInfer backend with the V2 autoregressive speculator.

## Not in this series

- **0.29's 0004** (skip the raw-logits clone when no logprobs are requested) patched the V1
  `v1/sample/rejection_sampler.py`. vLLM 0.30 serves this model on Model Runner V2, which
  never calls that file, so the patch was dead code and is dropped.
- **A reduced MTP draft vocabulary** is not included. It slices a float `lm_head`, and the
  int4-head checkpoint doesn't have one.
- **No repetition detector** is shipped.

## How this series was built and verified

1. 0001-0003 were built one commit per feature on a pristine unpack of the wheel, byte-compiling
   each step before the next. The build differs from the tree they were developed in only in
   enumerated ways: the plan-cache switch is a registered `envs.py` bool instead of an
   `os.environ` string compare; the plan-cache key gains the wrapper's `_backend`;
   `VLLM_HOST_EMBED_TABLE` defaults to 1; comment/log text. The files that other, unshipped
   development changes touched are byte-identical to pristine.
2. 0004 and 0005 were added on top of 0001-0003 from the development patches for the same two
   changes. The build enumerates every divergence from those patches (registered knobs, default
   on, no runtime file toggles, no failing-control hook, proof-line text, and 0005's hand-off
   cleared right after the draft build) and fails on any other difference.
3. The whole series is applied with `git apply --check` then `git apply --directory=vllm` to a
   **fresh** unpack of the wheel, every exit code checked. Re-applying any patch to the patched
   tree fails, as it should (failing control).
4. CPU import smoke on the patched tree: `vllm.envs`, the FlashInfer backend and the V2 model
   runner import; the 0002 and 0003 knobs default to on, parse `0` as off, and appear in
   `envs.compile_factors()`. The 0004 and 0005 knobs are registered in `envs.py` the same way.
5. `release/install-env.sh` applies the series the same way and asserts every patch's marker
   before it prints `INSTALL OK`.
