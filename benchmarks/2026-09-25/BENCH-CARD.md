# Bench card, earlier three-patch build: Qwen3.8-27B on vLLM 0.30.0 + patches 0001-0003 (TP=2 and TP=4, 262K)

Measured 2026-09-25 18:27–18:51 UTC (the TP=2 rows; the TP=4 section below was measured 2026-09-26) on the reference host (2× RTX 3090, PCIe Gen4 x16, P2P).
Subject: `serve/serve.sh` (as shipped, apart from the KV pin; see the pin note) on an env built by
`release/install-env.sh`, with the int4-head checkpoint (`halt95/Qwen3.8-27B-W4A16-Merlin`). KV pin
14,803,350,467 B/card gives a pool of 809,399 tokens. Served as shipped: thinking on (medium), with
the server's sampling override (T 1.0 / top-p 0.95 / top-k 20); the client sends no sampling
fields. **Natural EOS on every row** (92 of 92 `stop`, `ignore_eos` false on all). Each prompt
opens with a nonce, so every single-stream request is a cold prefill. Measured with the maintainer's
OpenAI-API streaming harness (not included).

> **Scope note.** Everything on this card was measured on an earlier three-patch build
> (0001–0003, no long-prefill threshold; not published). v1.0.0 ships five patches and
> `--long-prefill-token-threshold 832` by default; its numbers are in
> [`2026-09-26/BENCH-CARD.md`](../2026-09-26/BENCH-CARD.md), which compares against this card.

> **Pin note (2026-09-26).** The TP=2 rows on this card were measured at the earlier pin of 14,803,350,467 B/card (pool 809,399 tokens). The shipped `serve.sh` default is now **14,534,914,979 B/card, a pool of 794,351 tokens** (B+0.25 GiB instead of B+0.5, because the +0.5 GiB edge proved noisy at TP=4). A smaller pool doesn't change the single-stream or c8/c32 speed numbers: none of these cells comes near the pool limit.

## Single stream (3 prompts per depth, 512-token budget)

Aggregation, per column: **median ITL** = the median of the 3 per-request median inter-chunk
latencies; **tokens/step** = 1 + accepted draft tokens / drafts, from the engine's spec-decode
counters over the 3 requests; **decode tok/s** = the median of the 3 per-request rates
(completion tokens / decode wall); **TTFT** = the median of 3. With n=3 each median is one of
the three samples.

**Cold check.** The API's `cached_tokens` isn't in these rows: the shipped line doesn't set
`--enable-prompt-tokens-details`. The server log shows it instead: every one of the 34
engine-stats lines across the card run reads `Prefix cache hit rate: 0.0%`.
It is also cold by construction, because a nonce opens each prompt, so no block hash chain can
match. Later cards record a per-request cached-token count from the
`vllm:prefix_cache_hits_total` delta.

| depth (prompt tokens) | median ITL | tokens/step (MTP K=3) | decode tok/s (label: completion tokens / decode wall) | TTFT | prefill tok/s |
|---|---|---|---|---|---|
| 4K (3,937) | **21.2 ms** | 3.58 | 166.7 | 2.1 s | ~1,870 |
| 32K (32,602) | **23.2 ms** | 3.46 | 153.3 | 19.5 s | ~1,670 |
| 131K (130,903) | **28.7 ms** | 3.62 | 127.9 | 109.1 s | ~1,200 |
| 250K (249,826) | **35.0 ms** | 3.49 | 103.0 | 276.8 s | ~902 |

Output lengths, natural EOS: 4K [178, 286, 179]; 32K [196, 373, 220]; 131K [235, 165, 311]; 250K [297, 216, 169].

## Concurrency (2,048-token prompts, 512-token budget, 2 reps)

| load | aggregate tok/s (wall, including prefill) | tokens/step |
|---|---|---|
| c8 | 166.1 | 3.47 |
| c32 | 175.0 | 3.46 |

## Against the 09-23 card (0.29.0 + 0.29 series, same checkpoint): an unpaired cross-card-pair reference, not a measured effect

| depth | ITL 09-23 → now | decode tok/s 09-23 → now | TTFT 09-23 → now |
|---|---|---|---|
| 4K | 22.0 → 21.2 ms (−3.6 %) | 156.5 → 166.7 | 2.1 → 2.1 s |
| 32K | 23.8 → 23.2 ms (−2.5 %) | 142.6 → 153.3 | 19.6 → 19.5 s |
| 131K | 29.2 → 28.7 ms (−1.7 %) | 117.5 → 127.9 | 109.4 → 109.1 s |
| 250K | 36.1 → 35.0 ms (−3.0 %) | 95.2 → 103.0 | 279.2 → 276.8 s |
| c8 / c32 | | 160.9 / 175.4 → 166.1 / 175.0 | |

**Read this as "no regression", not as a speed-up.** The 09-23 card ran on the other TP=2 card
pair (the other two cards of the host), and the two pairs differ by about 5.4 %, which covers the whole ITL difference. That
card also didn't record tokens/step, and with MTP decode tok/s is set by content as much as by
the engine. Median ITL moves −1.7 to −3.6 % and TTFT is flat, both within the card-pair gap. The
0.29→0.30 move is flat on speed, which matches an earlier paired screen. What it gains is capacity: pool
809,399 vs 703,274 tokens (+15.1 %) from the int4-head re-pin, and a standalone env.

Notes:
- vLLM 0.30's fused multi-step draft decode is unsupported on the FlashInfer backend. The engine
  logs a fallback (it rebuilds attention metadata between draft steps). A possible lever, not
  measured here.
- On this shape a 250K cold prefill takes ~277 s. A request decoding alongside it on the same
  engine gets one step per 4,096-token prefill chunk (~4.5 s), so it runs at ~1 tok/s until the
  prefill finishes. (Without the long-prefill threshold, as measured here; the v1.0.0 default
  changes this, see `../2026-09-26/BENCH-CARD.md`.)

---

# TP=4 section: the same model on four RTX 3090s

Measured 2026-09-26 on the same host, **all four cards (four-way P2P, patched driver, all Gen4 x16)**, with `serve/serve-tp4.sh` as it then stood (the same script as shipped apart from the KV pin, see the pin note). That is `serve.sh` with `--tensor-parallel-size 4`, `GPUS=0,1,2,3` and a KV pin of 18,381,091,779 B/card, which gives a pool of **2,010,034 tokens** (2.53× the shipped TP=2 pin's 794,351; 2.48× the 809,399 pin the TP=2 rows were measured at). The protocol is identical to the TP=2 card above: as served, natural EOS on every row (92/92 `stop`), cold nonce prompts, with per-request cached tokens recorded (all 0).

| depth | median ITL TP=4 | TP=2 (above) | **decode ratio (TP2 ITL / TP4 ITL)** | tokens/step TP=4 | TTFT TP=4 / TP=2 | **prefill ratio** | decode tok/s (label) |
|---|---|---|---|---|---|---|---|
| 4K | **16.6 ms** | 21.2 ms | **1.28×** | 3.38 | 1.3 / 2.1 s | 1.64× | 202.8 |
| 32K | **17.6 ms** | 23.2 ms | **1.32×** | 3.42 | 11.4 / 19.5 s | 1.71× | 190.6 |
| 131K | **20.2 ms** | 28.7 ms | **1.42×** | 3.48 | 61.5 / 109.1 s | 1.77× | 177.7 |
| 250K | **23.3 ms** | 35.0 ms | **1.50×** | 3.44 | 153.1 / 276.8 s | **1.81×** | 154.0 |

Quality and gates: GSM8K-200, TP=4 **196/200** vs the TP=2 arm's 192/200 on identical questions (discordant 0/4, exact McNemar p = 0.125). Prefix-cache residue sweep: **PASS by mechanism on the valid cells** (same-prompt 32/32 byte-identical; the multi-turn near-tie flips reproduce the known decode-vs-prefill class 6/6). One cell is untested (it got no prefix-cache hit, so it compares nothing). That sweep's instrument boot alone added `--enable-prompt-tokens-details`.

Concurrency (2,048-token prompts): c8 **231.9 tok/s**, c32 **264.3 tok/s** aggregate (wall, including prefill). That's 1.40× and 1.51× TP=2. Prefill runs at 3,090 / 2,850 / 2,130 / 1,632 tok/s at 4K / 32K / 131K / 250K.

**Against the maintainer's earlier TP=4 projections:**
- *Decode 0.93× shallow:* **refuted**. TP=4 is **1.28× faster at 4K**. The 0.93× figure predates FULL graphs + MTP K=3. With those, shallow decode on this profile is no longer all-reduce-dominated.
- *Decode 1.5–1.67× at 128–250K:* **holds at the low end**. 1.42× at 131K is just below the band, and 1.50× at 250K is on its lower edge.
- *Prefill ~1.8×:* **holds at depth**. 1.77× at 131K and 1.81× at 250K; 1.64–1.71× shallow.

Caveats:
- tokens/step differs a little between the cards (content-driven with MTP). The ratios use median ITL, not decode tok/s.
- The TP=2 rows used one pair of the host's cards; TP=4 uses all four.
- **Measured with four-way P2P only. Without P2P, TP=4 is unmeasured.**
