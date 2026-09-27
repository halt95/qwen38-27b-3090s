# Repetition-loop check (2026-09-26)

**Question:** does the vLLM 0.30 build loop more than the previous build?

**Arms** (TP=2, each build's serve line of the time; only the environment/checkpoint differ):
- **A1:** the earlier three-patch vLLM 0.30.0 build (patches 0001-0003, int4-head checkpoint);
- **A2:** the previous build (vLLM 0.29.0 + its series, same checkpoint);
- **A3:** the A1 build with the BF16-head version of the checkpoint.

**Protocol:**
- 12 prompts × 6 seeds per arm, run on both card pairs of the reference host (counterbalanced): 144 generations per arm.
- Sampling as served (T 1.0, top-p 0.95, top-k 20, seed sent).
- Output caps of 8K tokens, and 16K with reasoning effort xhigh on the 4 thinking prompts.
- 6 concurrent requests. Full reasoning and content text saved for every generation.
- A generation is a **genuine collapse** if its tail diversity drops below 0.15 or a 12-token window repeats ≥ 20 times. Every flag, loose or genuine, was read by hand.

| arm | generations | genuine collapses | loose flags (all read: false positives or long reasoning) | length-capped | engine errors |
|---|---:|---:|---:|---:|---:|
| A1 (three-patch 0.30 build) | 144 | **0** | 2 | 12 | 0 |
| A2 (previous build) | 144 | 0 | 1 | 12 | 0 |
| A3 (BF16 head) | 144 | 0 | 3 | 12 | 0 |

One-sided Fisher exact test, A1 worse than A2 or A3: p = 1.0 for both.

**Concurrency** (A1, thinking forced on, 3 seeds): c3 had 0 genuine and 0 flagged of 36; c8 had 0 genuine and 0 flagged of 36.

**Length-capped answers at 32 concurrent** (48 requests of the same shape per arm, 2,048-token cap, paired):

| arm | stopped | length-capped | still reasoning at the cap | genuine loops |
|---|---:|---:|---:|---:|
| A1 | 32 | 16 | 16 | **0** |
| A2 | 36 | 12 | 12 | 0 |

The loosely flagged capped answers were the model enumerating a periodic column in the synthetic prompt, not collapsing.

**Limit:** zero collapses in 432 main generations shows no regression, but a sample this size can't rule out a rare loop. The base model has been seen to loop occasionally in earlier checks at a different sampling temperature.

## Pre-release five-patch build addendum (2026-09-26)

The pre-release five-patch build (five patches + `--long-prefill-token-threshold 832`, TP=2, shipped serve line)
was run through the same 12 prompts × 6 seeds at 6 concurrent, with the same caps, detector and
thresholds as above, on one card pair.

| build | generations | genuine collapses | loose flags | length-capped | engine errors |
|---|---:|---:|---:|---:|---:|
| pre-release five-patch | 72 | **0** | 1 | 6 | 0 |

The one loose flag was read by hand: it hit the 16K reasoning cap with tail diversity 0.696 and a
maximum 12-token-window repeat count of 5 (genuine thresholds: < 0.15 or ≥ 20). The model was
re-deriving a cycle-detection algorithm and hand-simulating cases, and the repeated code blocks
inflated the loose score; there is no collapse. All 6 length-capped answers are that same prompt
(all 6 seeds) and all have tail diversity of 0.70 or just under (minimum 0.696). That is the same capped rate as the three-patch arm
A1 above (12 of 144 across two card pairs, 6 per pair).

With the 432 generations above, that is 0 genuine collapses in 504. The same limit applies: this
shows no regression, not the absence of a rare loop.

## v1.0.0 (2026-09-26)

v1.0.0 as shipped (`serve/serve.sh`, TP=2, knobs unset) ran the same 12 prompts × 6 seeds at 6
concurrent, with the same caps, detector and thresholds, plus a cell of 12 prompts × 3 seeds at 8
concurrent with thinking forced on.

| cell | generations | genuine collapses | loose flags | length-capped | engine errors |
|---|---:|---:|---:|---:|---:|
| 12 × 6 at 6 concurrent | 72 | **0** | 1 | 7 | 0 |
| 12 × 3 at 8 concurrent | 36 | **0** | 0 | 10 | 0 |

The one loose flag was read by hand. It is the same cycle-detection prompt (seed 42): it hit the
16K reasoning cap with tail diversity 0.764 and a maximum 12-token-window repeat count of 5
(genuine thresholds: < 0.15 or ≥ 20). The model was deriving and hand-testing Brent's algorithm;
there is no collapse.

That is 0 genuine collapses in 108 generations on v1.0.0, and 0 in 504 on the earlier builds
above. The same limit applies: this shows no regression, not the absence of a rare loop.
