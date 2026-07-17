# M11 — Decode performance anatomy (Qwen3-1.7B int8, rbm21, 2026-07-17)

Why is CPU decode of a 1.7B int8 model stuck at ~13 tok/s, and what actually
limits it? A systematic teardown — each hypothesis tested, most falsified.

## The number
Direct curl (no pi), 2k ctx: **decode 13–14 tok/s, prefill 33 tok/s** on rbm21
(i5-13400T, AVX-VNNI). This is the honest idle-box ceiling (`vmstat` confirmed the
box 99% idle at steady state; earlier load 6–12 was the decaying average of our own
benchmarks/warm, not contention).

## What does NOT limit decode (all measured, all falsified)
| Hypothesis | Test | Result |
|---|---|---|
| Thread count | sweep THREADS 1..16 | 2→4→6 scales (6.5→10.7→12.3), then **flat** 6–16 (~13). More threads do nothing. |
| int4 weights | parallel `dot_q4` vs `dot_q8` kernel bench | **1.08×** — the scalar nibble-unpack is compute-bound, never cashes in the halved bytes |
| VNNI `dot_q8` | (reasoned from below) | Wouldn't help — decode isn't q8-compute-bound |
| Channel/barrier overhead | round-trip microbench | **3.9 µs/round-trip** → ~4.5 ms/token (~6%) — not it |
| Matmul fusion | fused Q\|K\|V + gate\|up (one quantize, one barrier, big region) | **13.1 → 13.1 tok/s, 0% gain** |

## What DOES limit it: memory *latency* on read-once weight streaming
The tell was a false "56 GB/s peak." A parallel-process streaming bench that
**reused one 128 MB buffer 10×** hit 56–59 GB/s aggregate — but that keeps the
prefetcher warm. Decode reads each of the model's **1.83 GB of int8 weights exactly
once per token, scattered across matrices, with zero reuse** (autoregressive: one
token at a time). Re-running the bench **read-once (ITER=1)** gave **20 GB/s** — which
matches decode's effective **24 GB/s**. So:

> Decode is **memory-latency-bound on read-once weight streaming, already at the
> box's ceiling** (~20–24 GB/s). There is no bandwidth gap to capture — which is
> exactly why fusion, threads, and barrier-reduction all did nothing.

13 tok/s × 1.83 GB = 24 GB/s ≈ the no-reuse ceiling. The engine is not leaving
throughput on the table.

## The only levers that can actually move decode
1. **Fewer bytes per token → int4, but with a kernel that keeps up.** int4 halves
   weights to 0.93 GB → ~2× *if* a vectorized (AVX-VNNI `vpdpbusd` on unpacked
   nibbles) `dot_q4` reaches the latency ceiling. The current scalar `dot_q4` does
   not (1.08×). Core SIMD-kernel work + int4 model + accuracy validation. Unproven.
2. **Reuse weights across tokens → speculative decoding.** A tiny draft model
   proposes K tokens; the 1.7B verifies all K in one batched pass that reads each
   weight once for K tokens. Directly attacks the no-reuse root cause → potential
   2–3×. The architecturally-correct fix for latency-bound autoregressive decode.
3. **Smaller model** (0.6–0.8B) → fewer bytes → ~2–3×, quality cost, trivial effort.

VNNI-on-q8, more threads, barrier/fusion refactors, and int4-with-the-current-kernel
are all dead ends for decode — proven here, so we don't re-litigate them.

## Note: decode was never the pi-latency problem
A short reply ("Hello!", ~6 tokens) is <1 s of decode. pi's "slow" is **prefill** —
its ~3200-token system prompt costs ~97 s cold at 33 tok/s — addressed separately by
the warm (M10) + date normalization. Decode speed only matters for long generations
and multi-step agentic loops.
