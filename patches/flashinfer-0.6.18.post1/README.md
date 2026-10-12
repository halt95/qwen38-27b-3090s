# Patch for FlashInfer 0.6.18.post1

One patch against **flashinfer-python 0.6.18.post1 exactly as published on PyPI**
(`flashinfer_python-0.6.18.post1-py3-none-any.whl`, sha256
`adb6d2952b47131138e108bd2999b1fe18b71f1a4c0b2684f168753db9563991`, the hash pinned in
`release/requirements-pinned.txt`). Paths in the patch are relative to the `flashinfer` package
directory in `site-packages`.

| # | what | default | knob | files |
|---|---|---|---|---|
| 0001 | opt-in 32-row CTA for the MTP verify attention plan on sm_86 with FP8 (E4M3) KV | **off** | `FLASHINFER_SM8X_FP8_SMALLQ=cta32` | `data/csrc/batch_prefill.cu`, `data/csrc/batch_prefill_paged_kernel_inst.jinja`, `data/include/flashinfer/attention/prefill.cuh`, `data/include/flashinfer/attention/scheduler.cuh`, `jit/attention/modules.py` |

## What it does

With MTP K=3 or 4, each verify step attends 4 or 5 query tokens per request. At GQA 6 that is 24
or 30 packed query rows per KV head. Stock FlashInfer plans these rows into a 64-row CTA laid out
as 4 Q warps x 1 KV warp. Warps 2 and 3 hold only padding rows, yet they still issue the full QK
and PV tensor-core stream. On sm_86 with FP8 KV that padding work, not DRAM, bounds the kernel.

The patch adds a 32-row CTA laid out as 2 Q warps x 2 KV warps (NUM_MMA_Q 1, FP8 repack path).
All four warps' MMAs are live, so each warp issues half the HMMAs per KV tile (130 static vs 260).
The padding rows (24..31 or 30..31) map to a query index past `qo_len`. They are masked and never
written, the same way as the 40 padding rows in the stock 64-row CTA.

**Scope.** The planner takes the tile only for a plan that matches all of these:
- `uniform_q_len` 4 or 5 (MTP K=3 or 4);
- GQA group 6;
- E4M3 KV (E5M2, NVFP4 and 16-bit KV keep the stock planner);
- head_dim 256 for both QK and VO;
- compute capability exactly 8.6;
- every request's packed length exactly `6 * q_len`.

Any other plan, in any mode, keeps the stock tile and the stock work order. Every kernel-side
change is behind `if constexpr (kFp8SmallQ2x2)`, so other instantiations compile to the stock
kernels: per-kernel SASS of the stock CTA64, CTA16 and merge kernels is identical on desk.

**The knob.** `FLASHINFER_SM8X_FP8_SMALLQ`: unset or `0` = stock, `cta32` = the tile. A third
value, `cta16`, is a discriminator arm from the microbench (q_len 4 only). It failed its error
budget and is **not** for serving. Any other value raises. Set the knob before the server starts
and the same on every rank. The planner reads it on the first plan and freezes it for the
process. A later change raises on the next plan, so a plan cache or a captured CUDA graph can
never mix arms. The knob does not change any build. Patched and stock planners compile the same
kernels.

**Artifact identity.** The patch suffixes the fa2 batch-prefill JIT URIs with
`_sm8x_smallq_v5`. A module built from stock sources, an older patch revision, or an upstream
AOT `.so` can therefore never be loaded for the patched code (`jit/core.py` loads AOT first).
The JIT objects build on first use under
`$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/0.6.18.post1/86/cached_ops/<uri>_sm8x_smallq_v5/`.
The suffix was bumped to `v5` when the patch's identifiers were renamed (same kernel code as v4),
so a JIT cache built from a pre-release revision of this patch is never reused.

**Not with `flashinfer-jit-cache`.** That wheel ships prebuilt kernels. It is not in the release
pins, and `release/apply-flashinfer-patch.sh` refuses a venv that has it.

## Apply

```bash
release/apply-flashinfer-patch.sh <venv>
```

The script does the following, and fails closed at any step:
1. Resolves the venv's purelib (with a clean environment).
2. Asserts that `import flashinfer` resolves to `<purelib>/flashinfer/__init__.py`, twice: in the
   venv's python with a clean environment (`env -i`, isolated mode `-I`: no `PYTHONPATH`, no user
   site, no working directory), and with the caller's environment (`PYTHONPATH` and user site
   kept, `-P`: no working directory, as for the `vllm` entry point). A copy that shadows the
   installed package on either path is refused: it, not the patched files, would be imported.
3. Asserts that flashinfer-python is 0.6.18.post1 and installed there.
4. Asserts that `flashinfer_jit_cache` is neither importable nor installed.
5. Takes an exclusive lock (`flock` on `<purelib>/.flashinfer-sm8x-patch.lock`) and holds it
   through steps 6-9, so two runs on one venv cannot interleave.
6. Classifies the five touched files by whole-file sha256 (pristine, exactly patched, or refused).
7. Runs `patch -p1 --forward --dry-run` in `<purelib>/flashinfer`, then applies the patch with
   `--forward --fuzz=0` (`--forward`: GNU patch never treats an applied hunk as a reversed patch).
8. Removes stale `__pycache__` under `flashinfer`.
9. Re-hashes the five files and asserts the patch markers in every touched file, and that the
   tree reverses cleanly.

The script is idempotent. A tree that already carries exactly this patch exits 0 without
touching it. A tree with another revision of this patch, a half-applied patch, or edits in
the touched files fails (the script compares the whole-file sha256 of all five files). Reinstall flashinfer-python and run
the script again. The script needs GNU `patch` and `flock` (util-linux).

Applying the patch changes nothing at runtime until the knob is set.

## Proof line (check after the first request, not at boot)

When a plan takes the tile, the planner prints one line to stderr, once per (mode, q_len) per
loaded module, so once per TP rank, with the CUDA device the plan was made on:

```
[flashinfer-sm8x-tile] smallq=cta32 q_len=5 rows=30 cta_tile_q=32 device=0
```

`device` is `cudaGetDevice()` of the planning thread, so it is relative to `CUDA_VISIBLE_DEVICES`:
vLLM puts TP rank r on `cuda:r`, and the ranks of a TP=N server print 0..N-1 whichever cards
`GPUS` selects (-1 if the device cannot be read). The line is printed when the **plan** selects
the tile, before any kernel runs: it is planning-time evidence that every rank chose the tile, not
proof that the kernel executed (a later failure of that step would still leave the line). The
serving gates (greedy and GSM8K runs with the knob on) are the execution evidence.

The line appears at the **first verify plan that uses the tile**, not at boot. Release smoke:
boot with the knob set, send one request with MTP on, and wait for it to finish. Then run:

```bash
release/check-tile-proof.sh <server-log> [q_len] --tp N   # N = tensor-parallel size (2 or 4); q_len defaults to 5 (MTP K=4), 4 for K=3
```

The script exits 0 if the current boot's part of the log (after the last "Initializing a V1 LLM engine" line)
holds the complete line (whole-line match: nothing before or after it) exactly once for every device 0..N-1 and
for no other device, and non-zero otherwise: a missing, repeated or out-of-range device fails, and so does a log
without the engine-start line (no boot boundary). FlashInfer's C++ writes the line straight to stderr, so it
carries no vLLM worker prefix; the device field labels the rank. `--tp` is required. When it fails, it lists the
`[flashinfer-sm8x-tile]` lines of the current boot, and names an older patch revision if the lines lack `device=`.
With the knob unset in the FlashInfer module itself, nothing prints.

## Split-KV merge and the -5e4 sentinel

Stock FlashInfer starts every row at m = -5e4 (its finite mask value), d = 1, o = 0, and masks a
key by setting its logit to -5e4. The tile tracks validity explicitly (masked logits are a true
-inf) and reproduces stock's rule that a row whose valid logits all sit at or below -5e4 outputs 0.
Two cases differ in the per-chunk state a split-KV plan hands to the merge:

- **A KV chunk with no visible key for a row** (the chunk starts after the row's causal limit:
  with one-block chunks, a request whose last block holds fewer than q_len keys). The tile writes
  m = -5e4, d = 1, o = 0, so LSE = -5e4 exactly. Stock writes m = -5e4 (unscaled), o = 0 and
  d = 1 + the masked keys of the tiles it read (each masked key scores p = 2^0 = 1 against the
  sentinel), so LSE = -5e4 + log2(d), a few units higher. Both outputs are 0 and both weights are
  zero: every other chunk of that row holds a visible key whose raw logit is above -5e4, so its LSE
  is at least -5e4 * sm_scale_log2, and the masked chunk sits about 5e4 * (1 - sm_scale_log2) below
  it (the merge's flush-to-zero exp2 returns 0 below 2^-126). For the 27B, sm_scale_log2 =
  log2(e) / 16 * k_scale, with calibrated k_scale 0.030-0.063, i.e. 0.0027-0.0057: the gap is
  about 49,700, in both arms.
- **A chunk whose visible keys all score at or below -5e4 raw** while the row's other chunks do
  not (the delta audit's case). The tile keeps d normalised to the chunk's real maximum (up to the
  chunk's key count), stock keeps d = 1, both report m = -5e4. The chunk's weight differs only
  when it is not negligible, i.e. when sm_scale_log2 is close to 1 (at least about 0.997): then the
  merged output differs (the audit's construction at sm_scale_log2 = 1: stock 0.9995, tile 0.80, exact
  1.0). At the 27B's scale the chunk's weight is about 2^-49700 in both arms. In that synthetic cell
  (every logit about -140 in natural units) the tile is exact (6.4e-13 vs fp64 on the box) and stock
  is not: stock's d = 1, m = -5e4 initial state still carries weight there, and its CTA64 output is off
  by up to 0.055 at TP4. No real row has every logit that far below zero.

`tests/test_0001_split_kv_merge.py` checks this on the GPU: tile vs stock (knob off) on the merged
output with split-KV on, q_len 4 and 5, E4M3 KV, the 27B's per-rank heads at TP2 (12/2) and TP4
(6/1) and their attention blocks (960, 848). It covers fully and partially masked last chunks,
sharp logits, 32K/128K CUDA-graph plans, the audit's construction (asserted to differ) and the same
logits at the 27B's scale (the tile gated on its error vs fp64, not on agreement with stock). It reads every fully masked chunk's LSE back from
the workspace and requires it to sit at least 126 below the row's other chunks in both arms.
Every other cell must meet the error budget that accepted the tile (vs an fp64 reference from the
stored fp8 bytes: rel_l2(tile) <= 1.10 x rel_l2(stock) + 1e-5, max_abs(tile) <= 2 x max_abs(stock) + 1e-6).
It also checks the cta32 run's proof line (`device=0`, once per q_len). One GPU, sm_86, the release venv:

```bash
CUDA_VISIBLE_DEVICES=0 <venv>/bin/python patches/flashinfer-0.6.18.post1/tests/test_0001_split_kv_merge.py   # exit 0 pass, 1 fail, 77 skip
```

## Disable

With the release serve scripts, set `FLASHINFER_SM8X_FP8_SMALLQ` to the empty string or to `0` and
restart the server (the serve scripts give an unset variable the default, `cta32`). Outside the serve
scripts, an unset variable also disables the tile. The plan is then the stock heuristic. The patched package can stay installed. To remove the code itself, reinstall
`flashinfer-python==0.6.18.post1` from the pinned hash.

## Evidence (RTX 3090, sm_86)

- **Microbench, patch v4 (2026-10-07, on the release venv):** GO. Kernel speed-up 1.51x at q_len 4
  and at q_len 5. Every cta32 gate passes: sentinel stock-parity (knob off and cta32), racecheck and
  initcheck at q_len 4 and 5. The v3 run of the same day passed racecheck, differential initcheck,
  NaN matrix, code probe, scope/rejection/freeze, error budget, knob-off bit-exact with stock and
  the q1/q3 controls (1.51x at q_len 4, 1.41x at q_len 5); the earlier v2 run measured 1.46-1.55x on
  every cell (TP2 255K: 649 to 424 us).
- **Serving A/B, patch v3 (2026-10-07; MTP K=4, same tree; S = stock FlashInfer, T = patched +
  `cta32`; every T boot showed the proof line and the patched module mapped on all ranks, and no S
  boot did).** Thinking-on decode at 255K:
  - TP2: 125 to **144 tok/s**; step time 35.6 to 31.8 ms.
  - TP4: 200 to **219 tok/s**; step time 23.1 to 21.0 ms.

  Thinking-off at 255K: TP2 77 to 89 tok/s, TP4 124 to 134 tok/s. GSM8K-200: T 195/195/195 vs
  S 194/195/195. The 262K needle passes on TP2 S, TP2 T and TP4 T. These were pre-release serving
  configs (earlier KV pins, no RecoverSSM); the release figures are in the top-level README.
- **Desk** (nvcc 12.8, sm_86): 244 registers, 0 spills, 98,816 B shared memory. The stock
  kernels' SASS is unchanged.
- **Not covered:** the single-card (TP1) arm with K=4 and the tile has not been run, and single
  card is not part of this release. The numerics check is kernel-level (an error budget against
  an fp32 reference). The tile is not bit-exact with the stock CTA64.

## Revisions

- **v4 (2026-10-07, after the release code audit).** The audit found that v3 treated a row whose
  valid scores all sat at or below FlashInfer's finite mask value (-5e4) as empty. Stock
  0.6.18.post1 also outputs 0 for a row whose valid scores all equal that value, so v3 matched
  stock there, but v3 diverged from stock when only one KV-warp half was at or below it. v4 carries
  an explicit per-row valid-key flag on the cta32 path (masked logits are a true -inf there), keeps
  stock's finalisation rule, and matches stock on every sentinel case; per-kernel stock SASS is
  unchanged. The serving A/B above was measured on v3; the v1.3.0 release gate ran on v4.
- **v5 (2026-10-08, naming only).** Identifiers, comments, the proof line and the JIT suffix were
  renamed to neutral `sm8x` names (proof line `[flashinfer-sm8x-tile]`, suffix `_sm8x_smallq_v5`);
  the kernel and planner code is v4 unchanged. The patched files' sha256 in
  `release/apply-flashinfer-patch.sh` are recomputed for this revision.
- **v5, delta fixes (2026-10-08, after the delta audit; JIT suffix still `_sm8x_smallq_v5`).**
  The proof line carries `device=<d>` and is documented as planning-time evidence;
  `check-tile-proof.sh` matches whole lines, requires the engine-start boundary and devices
  0..N-1 once each. `variant.sm_scale_log2` is passed to the KV-warp merge only on the
  kFp8SmallQ2x2 softmax path, so a custom attention variant without that member compiles as on
  stock (desk nvcc 12.8: such a non-softmax variant fails to compile on the v5 release revision
  and compiles with this one at CTA 64 and CTA 32; DefaultAttention SASS is identical to the v5
  release revision at CTA 16, 32 and 64). The apply script checks import resolution, locks, and
  patches with `--forward`. The kernel recurrence is unchanged. Changed headers rebuild an
  existing JIT module under the same URI (FlashInfer's ninja build tracks header dependencies).
