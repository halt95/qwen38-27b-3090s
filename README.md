# Qwen3.8-27B and ThinkingCap on 2× or 4× RTX 3090

**Serve Qwen3.8-27B, or BottleCap AI's ThinkingCap fine-tune of it, at the full 262K context on two or four RTX 3090s,
with vision, tool calls and MTP speculative decoding, behind one OpenAI-compatible endpoint.**

[![GPU](https://img.shields.io/badge/GPU-2x_or_4x_RTX_3090-76B900?logo=nvidia&logoColor=white)](docs/reference.md#hardware)
[![Context](https://img.shields.io/badge/context-262K_per_request-ffb000)](docs/reference.md#the-kv-budget)
[![KV pool](https://img.shields.io/badge/KV_pool-826%2C689_%2F_1%2C974%2C072_tokens-0969da)](docs/reference.md#the-kv-budget)
[![Engine](https://img.shields.io/badge/engine-Merlin_engine_v2.5.2_%2B_27B_series-6f42c1)](docs/reference.md#stack)
[![MTP](https://img.shields.io/badge/MTP-K%3D4-6f42c1)](docs/reference.md#stack)
[![Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97_checkpoint-Qwen3.8--27B--W4A16--Merlin-ffd21e)](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin)
[![ThinkingCap](https://img.shields.io/badge/%F0%9F%A4%97_checkpoint-ThinkingCap--Qwen3.8--27B--W4A16--Merlin-ffd21e)](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin)

This repository is the serving engine for two 4-bit checkpoints of Qwen3.8-27B (a 27B hybrid of Gated DeltaNet linear
attention and full attention, with a vision tower and a one-layer MTP head):
[halt95/Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin) and
[halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin),
BottleCap AI's reasoning fine-tune in the same format. Both run on **Merlin engine v2.5.2** (vLLM 0.30.0 plus the
Flash-Next fork commits and the guard fix) with the 27B patch series `patches/merlin-engine-v2.5.2/` (0001-0011) and a
FlashInfer 32-row verify tile: int4 weights, int4 `lm_head` and drafter, FP8 E4M3 KV, MTP with K=4 draft tokens per
step, fp16 SSM state, and hash-checked install and serve scripts. Bare metal only; validated on sm_86 (Ampere) only.
The current release is **v1.3.0** (2026-10-12; see [What's new](#whats-new-in-v130)).

## Models

| Checkpoint | What it is | Default reasoning effort | Weights and licence | Serve scripts (dual / quad card) | Headline numbers |
|---|---|---|---|---|---|
| **Qwen3.8-27B** | philbert's AWQ int4 build of Qwen3.8-27B with an int4 GPTQ `lm_head`, an int4 MTP drafter head and calibrated FP8 E4M3 KV scales; 11 files, about 17 GB | `medium` | [halt95/Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin), Apache-2.0 | `serve/serve.sh` / `serve/serve-tp4.sh` | Single-stream thinking decode at 32K: 198 tok/s dual card, 249 quad card; GSM8K-200 196 and 194 (release gate, 2026-10-08 to 2026-10-09 (UTC)) |
| **ThinkingCap** | BottleCap AI's reasoning fine-tune of Qwen3.8-27B, which reasons in fewer tokens; our W4A16 quantisation of it in the same format; 11 files, about 17 GB | `xhigh` | [halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin), PolyForm Small Business 1.0.0 plus BottleCap AI's personal-use permission ([License](#license)) | `serve/serve-thinkingcap.sh` / `serve/serve-thinkingcap-tp4.sh` | x0.860 of the base's median thinking tokens at medium effort, at the same GSM8K accuracy (97.27 %, 1,319 items); 172.8 tok/s at 32K on dual card (2026-10-08 gate) and 223.4 on quad card (2026-10-12) |

Both checkpoints run on the same engine, patch series, pins, port and served name (`qwen38-27b`); the ThinkingCap
scripts change only the default reasoning effort. ThinkingCap is an independent quantised derivative: it is **not**
published, reviewed or endorsed by BottleCap AI or Qwen.

## Why use it

- **Long context that actually fits.** Dual card: an 826,689-token FP8 KV pool, **three 262K sessions or six 131K
  sessions resident, 0 preemptions**. Quad card: a 1,974,072-token pool, **seven 262K sessions resident, 0 preemptions**.
- **Built for agents.** In a multi-turn replay on Qwen3.8-27B a follow-up turn started in a median 2.79 s on dual card and
  1.78 s on quad card, against 22.9 s and 15.3 s for the first turn. Eight agents at once on the ThinkingCap checkpoint (2026-10-12):
  215.7 tok/s total output on dual card and 333.5 tok/s on quad card, prefix-cache hit 95.4 %, 0 errors. Every response
  reports its cached prompt tokens.
- **Fast.** Single-stream thinking decode 206 / 198 / 172 / 144 tok/s at 4K / 32K / 128K / 255K on dual card and
  258 / 249 / 238 / 226 on quad card, with MTP K=4 and the FlashInfer 32-row verify tile.
- **Fewer thinking tokens when you want them.** On GSM8K the ThinkingCap checkpoint used x0.860 of the base's median
  thinking tokens at medium effort (same accuracy) and x0.579 at `xhigh`, on the same engine and pins.
- **Quality checked.** Qwen3.8-27B: GSM8K-200 196 on dual card and 194 on quad card (50 of those 200 items were also in
  the KV-scale calibration traffic, activation maxima only), the 262K needle passed on both, 0 genuine repetition
  collapses in 72 generations per profile, and the served tokenizer equal to upstream on a 153-item combining-mark set.
  ThinkingCap at its `xhigh` default (dual card): GSM8K-200 195 with 3 answers cut at the token cap, and 0 genuine
  repetition collapses in 72 generations, 6 of which ended at the length cap.
- **Vision and tools.** Up to 64 images per prompt (each capped at 4 MP), the `qwen3_coder` tool parser and the Qwen3
  reasoning parser.
- **Reproducible.** The install checks the engine bundle and the compiled-artifact tarball against `upstream/PIN`
  (sha256) and the engine's commit and tree, installs every dependency from hash-locked files, and both checkpoints are
  checked file by file.

The Qwen3.8-27B figures are the v1.3.0 release gate (2026-10-08 to 2026-10-09 (UTC)) on the shipped serve scripts with
no knob set, rounded to whole tok/s from the bench card's one-decimal figures, with thinking on at per-request effort low
and 256-token generations ([bench card](benchmarks/2026-10-12/BENCH-CARD.md)). The ThinkingCap figures are dated per row
in their own card ([ThinkingCap bench card](benchmarks/2026-10-12/ThinkingCap-BENCH-CARD.md)). Terms such as MTP, TP,
P2P and ITL are explained in the [Glossary](docs/reference.md#glossary).

## What you need

- 2× (dual card) or 4× (quad card) 24 GB sm_86 cards (RTX 3090), with no other process on them, and working GPU
  peer-to-peer (P2P): the serve scripts skip vLLM's P2P check and set `NCCL_P2P_LEVEL=SYS`. PCIe Gen4 x16 and P2P were
  used for every measurement; without P2P (unmeasured) remove those two settings.
- About 2.4 GiB more host RAM than a standard vLLM serve (the token-embedding table lives in pinned host RAM: 1.18 GiB per
  GPU on dual card, 0.59 on quad card), and disk for one checkpoint (about 17 GB, 15.9 GiB) plus the venv and the engine
  source tree.
- Linux x86_64 with glibc 2.31 or newer, an NVIDIA driver that supports CUDA 13.0, a CUDA toolkit with `nvcc` (for
  FlashInfer's JIT, the tile included), Python 3.13 with `venv`, git, `sha256sum`, GNU `patch` and `flock` (util-linux).

The full table, with the reference host: [Requirements](docs/reference.md#requirements).

## Quick start

Steps 1 and 2 are the same for both checkpoints; step 3 picks one.

```bash
git clone --branch v1.3.0 https://github.com/halt95/qwen38-27b-3090s && cd qwen38-27b-3090s
sha256sum -c release/SHA256SUMS                          # the repository files
# 1. the two engine assets named in upstream/PIN (bundle= and artifacts_tar=): the Merlin engine v2.5.2 bundle and the
#    v2.2.0 compiled-artifact tarball, both public GitHub release assets. Download them into the clone's PARENT directory
#    (the bundle may stay gzipped as v2.5.2-from-upstream-v0.30.0.bundle.gz), then come back: every command from here on
#    runs in the repository root (qwen38-27b-3090s), where upstream/PIN and release/ resolve:
cd ..
curl -fLO https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.5.2/v2.5.2-from-upstream-v0.30.0.bundle.gz
curl -fLO https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.2.0/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz
cd qwen38-27b-3090s
#    which gives exactly these names:
#      v2.5.2-from-upstream-v0.30.0.bundle
#      build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz
#    and verify them against upstream/PIN (install-env.sh checks both again and refuses on a mismatch):
A=..                                                     # the directory that holds them (relative to the repository root)
( set -e; . upstream/PIN
  if [ -f "$A/$bundle" ]; then got=$(sha256sum < "$A/$bundle"); else got=$(gunzip -c "$A/$bundle.gz" | sha256sum); fi
  [ "${got%% *}" = "$bundle_sha256" ] && echo "bundle: ok" || { echo "bundle: SHA-256 MISMATCH" >&2; exit 1; }
  [ "$(sha256sum < "$A/$artifacts_tar" | cut -d' ' -f1)" = "$artifacts_sha256" ] && echo "artifacts: ok" || { echo "artifacts: SHA-256 MISMATCH" >&2; exit 1; } )
# 2. the environment: hash-locked Merlin engine v2.5.2 + the 27B series + the FlashInfer tile patch
release/install-env.sh .venv                             # engine = Merlin engine v2.5.2, bundle branch v2.5.2-public (commit and tree in upstream/PIN)
#    assets somewhere else: BUNDLE=/path/to/<bundle> ARTIFACTS=/path/to/<tarball> release/install-env.sh .venv
# 3. choose a checkpoint (about 17 GB each): int4 weights, int4 lm_head + MTP drafter, calibrated FP8 KV scales
#    Qwen3.8-27B:
HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/hf download halt95/Qwen3.8-27B-W4A16-Merlin --local-dir ./ckpt
(cd ckpt && sha256sum -c ../release/checkpoint.sha256)   # the 11 checkpoint files
#    or ThinkingCap (its weights carry their own licence, see License):
HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/hf download halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin --local-dir ./ckpt-thinkingcap
(cd ckpt-thinkingcap && sha256sum -c ../release/checkpoint-thinkingcap.sha256)   # the 11 checkpoint files
# 4. serve the matching profile (all four need working GPU peer-to-peer, see What you need)
MODEL=$PWD/ckpt serve/serve.sh                                   # Qwen3.8-27B, dual card, TP=2
MODEL=$PWD/ckpt serve/serve-tp4.sh                               # Qwen3.8-27B, quad card, TP=4
MODEL=$PWD/ckpt-thinkingcap serve/serve-thinkingcap.sh           # ThinkingCap, dual card, TP=2
MODEL=$PWD/ckpt-thinkingcap serve/serve-thinkingcap-tp4.sh       # ThinkingCap, quad card, TP=4
```

`install-env.sh` also needs `git`, `gzip` (for a `.bundle.gz`), GNU `patch`, `flock` (util-linux, used by
`release/apply-flashinfer-patch.sh`), Python 3.13 and network access to `github.com/vllm-project/vllm` (it fetches the
pinned base commit from `upstream/PIN` before applying the bundle). It builds the engine source in `.vllm-v2.5.2` inside
the clone and refuses to run if `VLLM_PRECOMPILED_WHEEL_LOCATION`, `VLLM_USE_PRECOMPILED` or
`VLLM_USE_PRECOMPILED_RUST` is set in the environment (those would replace the pinned native libraries). A failed run
removes the engine directory it created; fix the cause and run it again with a new venv path.

The two ThinkingCap wrappers set `THINKING` to `{"enable_thinking": true, "reasoning_effort": "xhigh"}` unless you set it
yourself, then run `serve/serve.sh` / `serve/serve-tp4.sh` with every other variable passed through.

A boot on a fresh compile cache takes several minutes (it compiles FlashInfer kernels, the tile among them; at the
release gate 403 s on dual card and 393 s on quad card); a warm boot reuses the compile cache. The server is ready when
the log shows `Application startup complete` and `GPU KV cache size: 826,689 tokens` (dual card) or `1,974,072 tokens`
(quad card); the pool is pinned in bytes, so any other number means the setup is not the shipped one. It listens on
`127.0.0.1:8100`; `HOST=0.0.0.0` opens it to the network, and then set `VLLM_API_KEY` as well. More checks:
[Check it's working](docs/reference.md#check-its-working); if something goes wrong:
[Troubleshooting](docs/reference.md#troubleshooting). Read [Known behaviours](docs/reference.md#known-behaviours) before
putting it in front of clients.

### Container

Bare metal only in this release; the container route returns in a later release, rebuilt and run on GPUs first. The
`v1.0.0` tag keeps the old recipe, for the old engine.

## Talk to it

Any OpenAI-compatible client works. With curl:

```bash
curl http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"Explain the difference between a stack and a queue."}]}'
```

Or from Python with the `openai` package (set `OPENAI_API_KEY` to your `VLLM_API_KEY`, or to any value if you set none):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8100/v1")
reply = client.chat.completions.create(model="qwen38-27b", messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

- Both checkpoints answer as `qwen38-27b` (`SERVED=` changes the name). Thinking is on by default, at `medium` effort for
  Qwen3.8-27B and `xhigh` for ThinkingCap; a request can choose its own effort with
  `"chat_template_kwargs": {"reasoning_effort": "medium"}`.
- A short `max_tokens` can end inside the reasoning with empty `content` and `finish_reason: "length"`; raise it or send
  `"chat_template_kwargs": {"enable_thinking": false}`.
- Tools use the standard `tools` field (`qwen3_coder` parser, auto tool choice); images go in `image_url` content parts,
  up to 64 per prompt, each downscaled to the 4 MP cap (`MAX_PIXELS`).
- Every response reports `usage.prompt_tokens_details.cached_tokens`, the prompt tokens served from the prefix cache.
- Do not use `/v1/models` as a health check: it keeps answering 200 after the engine has died. A one-token generation
  is the check.

## What's new in v1.3.0

- **Engine.** Stock vLLM 0.30.0 plus five patches is replaced by Merlin engine v2.5.2 plus the 27B series
  `patches/merlin-engine-v2.5.2/0001-0011` ([What is in v1.3.0](docs/reference.md#what-is-in-v130)). The series is
  active only for a dense "Qwen3.5" target on sm_86; every other model takes the stock path. "Qwen3.5" here is the name
  of the architecture class vLLM loads (`Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration`), which Qwen3.8-27B
  uses; it is not a different model, and the patches apply to this checkpoint.
- **Second checkpoint: ThinkingCap.** BottleCap AI's reasoning fine-tune of Qwen3.8-27B, in the same 4-bit format,
  served on this same engine at `xhigh` effort by default (`serve/serve-thinkingcap.sh`,
  `serve/serve-thinkingcap-tp4.sh`, checkpoint list `release/checkpoint-thinkingcap.sha256`). Its weights carry their own
  licence ([License](#license)); details in [ThinkingCap checkpoint profile](docs/reference.md#thinkingcap-checkpoint-profile).
- **FlashInfer 32-row verify tile, on by default** (`FLASHINFER_SM8X_FP8_SMALLQ=cta32`). The MTP verify step at K=4 has
  5 query rows per sequence; the tile runs that attention with 32-row tiles instead of stock FlashInfer's smaller ones,
  and because it acts on attention its effect grows with context depth. The release gate measured the shipped profiles
  with the tile on; tile-off decode speed was not re-measured (bench card section 6 has a tile-off boot and its
  GSM8K-200). `release/install-env.sh` patches FlashInfer with
  `release/apply-flashinfer-patch.sh`; each rank prints a proof line when the tile is in use
  ([Check it's working](docs/reference.md#check-its-working)).
- **MTP K=4 by default** (was K=3): the drafter proposes four tokens per step instead of three. The release gate measured
  K=4 only; `SPEC_K=3` and `SPEC_K=1` remain accepted ([SPEC_K](docs/reference.md#spec_k)).
- **`--prefix-match-unit 16`.** The hybrid KV block is 848 tokens at K=4 (832 at K≤3) and 960 on dual card with
  RecoverSSM; 64 does not divide 848, 16 divides all three.
- **Dual card runs RecoverSSM** (`--use-replayssm`) with `--enable-mamba-fine-grained-prefix-cache`, so three 262K or six
  131K sessions stay fully resident; the pin is 883 whole blocks. Patch 0011 runs the RecoverSSM commit from CUDA graphs
  after the MTP proposal to keep its per-step cost down; the release-gate decode figures in
  [numbers](docs/reference.md#v130-numbers) are with RecoverSSM on.
- **Quad card** adds `--enable-mamba-fine-grained-prefix-cache` (no RecoverSSM) and holds seven 262K sessions.
- **Scheduler limits tightened:** dual card `--max-num-seqs 16 --max-num-batched-tokens 2048` (was 32 / 4096), quad card
  `--max-num-seqs 24 --max-num-batched-tokens 2048`.
- **fp16 SSM state, unchanged.**
- **Idle cards drop to P8.** The engine's bounds-guard reader parks with no device work while the server is idle, so
  idle cards can drop to P8 instead of holding P2 (`VLLM_E2_GUARD_IDLE=0` restores the always-polling reader). Thanks to
  CoolShades for the idle-power report in [Flash-Next issue #2](https://github.com/halt95/qwen38-flash-next-3090s/issues/2)
  and [PR #3](https://github.com/halt95/qwen38-flash-next-3090s/pull/3), which led to this fix.
- **Cached tokens in every response** (`--enable-prompt-tokens-details`), and a per-image pixel cap, `MAX_PIXELS`
  (default 4,194,304 = 2048×2048).
- **Combining marks tokenize as upstream.** transformers is pinned to 5.18.0: with 5.17.0 and earlier the served
  tokenizer split combining-mark scripts (Hindi, Thai, Arabic) differently from upstream; with 5.18.0 it matches upstream
  on a 153-item combining-mark set. The checkpoint's `tokenizer.json` is also replaced by the upstream Qwen3.8-27B file
  (sha256 `0997f410c57a1f4e…`; the file shipped before lacked `\p{M}` in its pre-tokenizer), for tools that read
  `tokenizer.json` directly; `release/checkpoint.sha256` lists the upstream file
  ([Checkpoint](docs/reference.md#checkpoint)). No weights changed.
- **Single card is parked.** One RTX 3090 returns in a later release; a single card profile was gated as v1.1.0 but not
  published. The container route is removed for this release.

Upgrading from v1.0.0, and every change: [CHANGELOG.md](CHANGELOG.md#v130).

## What's new in v1.0.0

The first public release (2026-09-27): stock vLLM 0.30.0 from PyPI plus patches 0001-0005, the dual card and quad card
serve profiles at 262K with MTP K=3, and a container recipe. Its numbers: [docs/history.md](docs/history.md); its notes:
[CHANGELOG.md](CHANGELOG.md#v100).

## Documentation

- [CHANGELOG.md](CHANGELOG.md): release notes, newest first.
- [docs/reference.md](docs/reference.md): what the build and serve scripts do, serve-script variables and knobs,
  requirements, health checks, troubleshooting, the ThinkingCap profile, what is in v1.3.0, how the model sits on the
  cards, capacity, benchmarks, known behaviours and a glossary.
- Bench cards: [v1.3.0](benchmarks/2026-10-12/BENCH-CARD.md) and
  [ThinkingCap](benchmarks/2026-10-12/ThinkingCap-BENCH-CARD.md) (2026-10-12), [v1.0.0](benchmarks/2026-09-26/BENCH-CARD.md)
  (2026-09-26), [the earlier three-patch build](benchmarks/2026-09-25/BENCH-CARD.md) (2026-09-25).
- [docs/loop-check.md](docs/loop-check.md): the repetition-loop check.
- Patch notes: [patches/merlin-engine-v2.5.2/README.md](patches/merlin-engine-v2.5.2/README.md) (the 27B series) and
  [patches/flashinfer-0.6.18.post1/README.md](patches/flashinfer-0.6.18.post1/README.md) (the verify tile).
- [docs/history.md](docs/history.md): v1.0.0 on stock vLLM 0.30.0, with its numbers.

## Known limitations

- Tested on one setup: 4× RTX 3090 at 220 W, PCIe Gen4 x16, with peer-to-peer. sm_86 only; no other GPU, distribution,
  driver or non-P2P host has been run.
- Dual and quad card only: the single card profile and the container route return in a later release.
- Switching the verify tile (`FLASHINFER_SM8X_FP8_SMALLQ`) or `SPEC_K`, or reusing prefix-cache blocks, can change output
  at near-tied tokens. Greedy (T=0) output repeats within one compile cache, not across fresh compiles.
- Changing `--limit-mm-per-prompt` needs a fresh compile cache (`VLLM_CACHE_ROOT`).
- JSON-schema structured output was not tested at the release gate, and no long soak is part of this release.
- The ThinkingCap token saving was measured on arithmetic word problems only.

The full list, with causes: [Known behaviours](docs/reference.md#known-behaviours).

## Credits

Qwen ([Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)), philbert
([AWQ int4 base checkpoint](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ)), BottleCap AI
([ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B); the ThinkingCap profile serves a
quantised derivative of it, not affiliated with or endorsed by BottleCap AI), [vLLM](https://github.com/vllm-project/vllm)
and lucifer1004 ([PR #47979](https://github.com/vllm-project/vllm/pull/47979), which patch 0001 ports),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) (which the verify tile patches),
[compressed-tensors](https://github.com/vllm-project/compressed-tensors) (the checkpoint format; `quant/` packs with it),
froggeric ([Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) v22.5),
[aikitoria's open-kernel-module patch](https://github.com/aikitoria/open-gpu-kernel-modules) (P2P on consumer Ampere),
and [CoolShades](https://github.com/CoolShades) (idle-power report, Flash-Next
[issue #2](https://github.com/halt95/qwen38-flash-next-3090s/issues/2), and
[PR #3](https://github.com/halt95/qwen38-flash-next-3090s/pull/3), behind the guard fix in Merlin engine v2.5.2).
Third-party notices are in [NOTICE](NOTICE); the list with links per item: [Credit](docs/reference.md#credit).

## License

Code in this repository (patches, install and serve scripts, quantisation tools, checksums): Apache-2.0
([LICENSE](LICENSE)); the patches modify the Apache-2.0 engine (vLLM and Merlin engine v2.5.2) and FlashInfer, see [NOTICE](NOTICE).
Model weights are not part of this repository. The checkpoint `halt95/Qwen3.8-27B-W4A16-Merlin` is Apache-2.0 like its
bases. The ThinkingCap checkpoint `halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin` is not:

**Licence of the ThinkingCap weights.** ThinkingCap is licensed by BottleCap AI under the
[PolyForm Small Business License 1.0.0](https://polyformproject.org/licenses/small-business/1.0.0), plus BottleCap AI's
additional permission for personal, non-commercial use by an individual, free of charge. A business may use it only if it
qualifies as a "Small Business" under the PolyForm text (fewer than 100 people and under US$1 million in prior-year
revenue, inflation adjusted); any other commercial use needs a separate licence from BottleCap AI. **There is no
unrestricted commercial use.** Required Notice: Copyright 2026 BottleCap AI (https://bottlecapai.com). These terms cover
the ThinkingCap weights on Hugging Face, which carry the licence files; they do not cover this repository's code, which
stays Apache-2.0. Copies of BottleCap AI's `LICENSE` and `NOTICE`, the PolyForm text and Qwen's Apache-2.0 text are in
[`licenses/`](licenses/); the changes made to the checkpoint are listed in [NOTICE](NOTICE). This is a summary, not legal
advice.
