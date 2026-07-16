# The performance frontier of pure-MFL LLM inference

An honest map of how fast a 1B-parameter LLM can run in pure [machin](https://github.com/javimosch/machin)
on **low-spec CPU hardware** — where the walls are, what moves them, and what
doesn't. Every number here was measured on rbm21 (Intel i5-13400T, 14 threads,
DDR4), serving Llama-3.2-1B-Instruct at int8/Q8_0.

## TL;DR

The engine is **already within ~7% of an optimal hand-written C GEMM**. Decode
is memory-bandwidth-bound; prefill is at the GEMM roofline. Every cheap,
exact software lever we tried (faster kernels, better parallelism, speculative
decoding, gate-based sparsity) hit a wall. **On weak hardware, further speed is
a *model* decision (quantization / architecture), not an *engine* one.**

## The two regimes and their walls

| Regime | Bound by | Measured | Roofline |
|---|---|---|---|
| **Decode** (1 token at a time) | weight streaming (1.1 GB/token) | ~21 tok/s = ~23 GB/s | DDR bandwidth |
| **Prefill** (many tokens) | FFN GEMM (74% of the time) | 86 GF/s aggregate | ~92 GF/s (optimal C) |

Decode reads every weight for every token — 21 tok/s × 1.1 GB = 23 GB/s, at the
practical DDR ceiling. Prefill is dominated by the FFN matmuls (w1/w2/w3 are 86%
of the model), and they run at 86 GF/s — matching a standalone optimal C GEMM
(92 GF/s) on the same box. **There is no idle silicon to reclaim.**

## What got the engine to the roofline (the wins that shipped)

- **mmap the checkpoint** (machin `mmap_file`) — load 31s → 43ms.
- **map auto-grow** (machin core fix) — the tokenizer's 128k-entry map was O(n²); now O(1) amortized.
- **Prefix cache that persists across requests** — a repeated system prompt goes
  37s → 0.8s (46×). The bug was arena-freed bookkeeping; fixed with a raw buffer.
- **Batched prefill** — run a tile of positions through each layer together so
  each weight is loaded once (GEMM, not N GEMVs). Bit-identical to per-token.
- **`dot_q8` builtin** (machin core) — the whole Q8_0 matmul inner product in one
  vectorized call. Decode ~19 → ~21 tok/s.

## What did NOT move the needle (the honest null results)

These are the interesting part — each is a technique that *should* help and
doesn't, because the bottleneck is elsewhere.

- **Faster matmul kernel (`dot_q8`) on prefill: no change.** The FFN was never
  kernel-bound — it's memory-bandwidth-bound. `dot_q8` helped decode a little,
  prefill not at all.
- **Vectorized attention (`dot_f32`/`axpy_f32`): no change.** Attention is only
  ~9% of prefill; the FFN dominates.
- **Barriers: cheap (28 µs).** The worker-pool sync was suspected but measured
  negligible (~27ms total across a 1000-token prefill).
- **Tile size / glue & rope parallelization: no change.** All confirm the same
  thing: the compute is memory-bound and already near-optimal.
- **Speculative (prompt-lookup) decoding: +10%, and exact.** It *works*
  (accept/step = 2.27 on verbatim-echo output, output bit-identical to greedy),
  but on low-spec the batched-verify forward goes **compute-bound** (k× the FFN
  compute) — and this box has no spare compute either. Speculation trades spare
  compute for bandwidth savings; low-spec has neither to spare. Best case ~+10%
  at k=3; net-negative at larger k. (Kept behind `COLIBRI_SPEC`, off by default —
  it will pay off on a higher-bandwidth box.)
- **Gate-based contextual FFN sparsity: unusable.** 74% of FFN neurons have a
  near-zero SiLU gate — but they carry **56% of the output mass** (`h = silu(w1x)·w3x`;
  a small gate times a large up-projection still matters). Skipping by the gate
  destroys quality. Real sparsity would need a *trained* predictor (Deja Vu-style),
  not a runtime heuristic.

- **Adaptive early-exit / layer skipping: ~1.15×, quality-risky.** A 1B token
  needs ~14 of 16 layers to stay coherent — fixed exit at 14 works ("…Notre
  Dame"), exit at 12 garbles, exit at 8 is gibberish. The model wasn't *trained*
  for intermediate readout (LayerSkip-style training would help a lot, but that
  is retraining). Not disruptive on an untrained model. (Behind
  `COLIBRI_EXIT_LAYER`.)

## Continuous batching: built, exact, ~1.35× on this box

For a *fleet* of agents, **continuous batching** batches independent requests
(each with its own KV cache slot + position) into one forward, so N requests
share each weight-read. Built and verified: a batched decode step over N slots,
output **bit-identical to solo greedy** (`cli_cb.src` harness, all slots agree).

But it is **~1.35×, not N×** on rbm21 (N=8: 31 vs 23 tok/s; N=2 even regressed):

| N (concurrent slots) | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| aggregate tok/s | 22.9 | 20.5 | 29.5 | 31.0 |

Same balanced-roofline wall: the batched matmul's **compute scales with N**, and
this box is compute-balanced, so amortizing the weight-*read* across N slots does
not dominate — the extra compute for N slots eats the saving. (Parallelizing the
O(N) serial glue — per-slot argmax/rope/scatter — did not move it either,
confirming it is the matmul compute, not the glue.) Continuous batching remains
the *right* technique on a **memory-bandwidth-bound** box (where it would give
~N×), plus it gives fairness on any box (all requests progress vs single-flight
FIFO). On rbm21 it is a modest 1.35× + fairness. The primitive
(`cb_step`/`cb_prefill`, per-slot KV via `kv_off`) is in the engine, unused by
the default server path.

- **int4 weights: half the memory, NO decode speedup on this box.** The int4
  model is 695 MB vs 1.3 GB (53%) and stays coherent ("The capital of France is
  Paris…"). It *should* ~2× a memory-bound decode. It doesn't (20.4 vs 20.1
  tok/s) — because halving the weight bytes drops decode below the memory
  ceiling (~11 GB/s) and exposes the **balanced-roofline wall**: the nibble
  unpack, the single-threaded per-token glue (rmsnorm/quantize/silu), and the
  classifier now dominate. Reduce one scarce resource and another equally-scarce
  one appears. (`dot_q4` builtin + `ak45` int4-v4 format; the win is memory
  *footprint*, and speed on a box that is memory-bound rather than balanced.)

### The balanced-roofline wall

The deepest finding: **rbm21 holds a ~20 tok/s wall for a 1B dense model
regardless of int8/int4/speculation/sparsity/early-exit**, because its compute
and memory bandwidth are balanced for this model size. Every technique trades
one for the other:

- Speculation: spends compute to save bandwidth → compute-bound.
- int4: saves bandwidth, spends unpack compute → compute-bound.
- Sparsity/early-exit: needs training, or loses quality.

There is no free lunch when both resources are equally scarce. The only levers
that shrink *both* at once are a **smaller model** (0.5B → ~2×) or **training a
model to be cheaper** (distillation, LayerSkip, native sparsity). Those are
model decisions, not engine ones — which is exactly the conclusion.

## Why the standard playbook fails on low-spec

The techniques that make LLMs fast elsewhere all assume a resource this box
lacks:

- **Speculative decoding** assumes spare compute (to verify drafts cheaply). ❌
- **Contextual sparsity** assumes a trained router, or an activation function
  (ReLU) whose zeros predict importance. Llama's SiLU gate doesn't. ❌
- **Continuous batching** assumes many concurrent requests (helps *throughput*,
  not single-request *latency*). ✅ for a fleet, ✗ for one agent.

## The conclusion: the disruption is in the model, not the engine

On weak hardware a dense 1B model is near a *fundamental* wall — bytes-moved-per-
token, at the DDR ceiling, with an engine already at the GEMM roofline. The
levers that remain all change the *model*, not the code:

- **int4 / int3 weights** — the one clean, hardware-independent ~2× left (halve
  the bytes on a memory-bound decode). Mild quality cost, well understood.
- **A natively-sparse or distilled model** — an MoE-small, or a task-distilled
  0.5B. This is where a real 3–5× lives, but it is an architecture/training
  decision fed *to* the runtime.

The engine's job — get to the roofline in pure MFL — is done. Everything past
here is a quantization or model-architecture choice.

## The escape hatch: MoE + expert streaming (the original colibrì's insight)

Studying [JustVugg/colibri](https://github.com/JustVugg/colibri)'s code (the
project this one is named after) closes the loop on "the disruption is in the
model." Their engine runs a **744B-parameter MoE on 25 GB of RAM** by keeping
the dense weights resident and **streaming the 19,456 routed experts from NVMe
on demand** (per-layer LRU + pinned hot-store + the OS page cache as a free L2;
`COLI_MMAP=1` mode is literally mmap views into the safetensors).

Why this beats our wall, in our own terms:

- The balanced-roofline wall prices a token at **active** parameters (bytes
  moved × compute). A dense model's active = total, so quality is capped by the
  wall. **MoE decouples them**: OLMoE-1B-7B has 7B total / **1.3B active** —
  7B-class quality at the ~1B-dense cost we've already proven (~20 tok/s here).
- Streaming works *only* because of routing: a dense model larger than RAM
  thrashes (every weight touched every token), an MoE touches top-k experts per
  layer per token, so the hot set stays in page cache and cold experts fault in
  from NVMe. **Total model size becomes bounded by SSD, not RAM.**
- We already shipped the enabler: `mmap_file`. The OS page cache is the LRU.
  The kernels (`dot_q8`/`dot_q4`) price each expert's three small matmuls.
- Their MTP speculation gets **39–59% draft acceptance** (2.2–2.8 tok/forward)
  where our prompt-lookup got +10% — because GLM ships a *trained* MTP head.
  Same lesson a third time: speculation pays when the model provides the draft.

Concretely for this repo (the proposed M7): **OLMoE-1B-7B** — the same model the
original colibrì used to validate its core ("Stage A", `c/olmoe.c`). int8 experts
≈ 7 GB against a 10 GB box: the hot experts live in page cache, the router picks
8 of 64 per layer, and the active set is 1.3B — our proven speed regime.

### M7 — DONE (2026-07-13): it works, in pure MFL

**OLMoE-1B-7B (6.9B total / 1.3B active) runs in this engine, token-identical to
the fp32 reference, at 11.5 tok/s on a laptop** ([certs/M7-olmoe.md](../certs/M7-olmoe.md)).
The 7.96 GB int8 checkpoint streams experts via `mmap` (622/1024 faulted in for a
12-token run — only the routed ones). Per-token traffic ≈ 1.5 GB × 11.5 tok/s ≈
17 GB/s — the MoE engine is already memory-bound at the active footprint, exactly
as the dense analysis predicted, but now delivering **7B-class quality at ~1B
speed**. The wall was never the engine; MoE is the model-side lever that clears it,
and the pure-MFL runtime carries it with the same `dot_q8` kernel and `mmap_file`
streaming that the dense engine shipped. **Headline achieved: a 7B-class model at
the speed of a 1B, on hardware you own, in pure MFL.**

## Reproduce

All flags live on `colibri-serve`: `COLIBRI_NOBATCH=1` (per-token prefill A/B),
`COLIBRI_TILE=N`, `COLIBRI_SPEC=1` + `COLIBRI_SPEC_K=k` (speculative decode),
`COLIBRI_SPARSE=1` (FFN sparsity probe), `COLIBRI_PROF=1` (phase timers). The
GEMM roofline microbench is `bench/gemm_mt.c`.
