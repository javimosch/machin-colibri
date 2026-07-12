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

## The one exact disruption we did NOT chase: continuous batching

For a *fleet* of agents (rbm21 serving many roam/tau workers), **continuous
batching** is exact, needs no spare compute (every batched token is real work),
and the GEMM scaling data proves the kernel supports it (B=256 at 92 GF/s). One
batched forward advances N requests for ~one weight-read → ~N× *throughput*. It
does nothing for single-request *latency*, and it is a substantial build
(per-request state + a batched step scheduler), so it is logged here as the
right lever for the *serving-many-agents* scenario, distinct from the
single-request work above.

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

## Reproduce

All flags live on `colibri-serve`: `COLIBRI_NOBATCH=1` (per-token prefill A/B),
`COLIBRI_TILE=N`, `COLIBRI_SPEC=1` + `COLIBRI_SPEC_K=k` (speculative decode),
`COLIBRI_SPARSE=1` (FFN sparsity probe), `COLIBRI_PROF=1` (phase timers). The
GEMM roofline microbench is `bench/gemm_mt.c`.
