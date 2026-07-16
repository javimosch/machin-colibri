# M7 spike certification — pure-MFL MoE + expert streaming (2026-07-13)

**Claim proven:** the disruptive mechanism from the original colibrì — a
Mixture-of-Experts forward with **experts streamed on demand** — works in pure
MFL, token-for-token exact, with **zero model download**.

## What was built
- `tools_gen_moe_spike.py` — generates a tiny synthetic MoE transformer
  (D=32, 4 heads, 8 experts, top-2, 2 layers, vocab 48) and a **numpy reference**
  greedy token stream for prompt `[1,2,3,4,5]`.
- `moe_spike.src` — the same model, forward pass in **pure MFL**: RMSNorm, RoPE,
  attention, and the MoE FFN (`gate → softmax → top-k → SwiGLU expert → weighted
  sum`). Experts live in an **`mmap`'d** region and are read **only when
  routed** — the OS page cache is the per-layer LRU; cold experts never fault in.

## Result
```
numpy ref : [2, 29, 2, 29, 2, 44, 18, 32, 32, 32, 32, 32]
MFL       : [2, 29, 2, 29, 2, 44, 18, 32, 32, 32, 32, 32]   ← token-identical
experts touched (streamed in): 14 / 16
```
Run (CPU-capped): `taskset -c 0-3 nice -n 15 ./moe_spike models/moe-spike.bin`.

## Why it matters
This closes the last open question from `docs/PERFORMANCE-FRONTIER.md`: the escape
from the balanced-roofline wall is MoE (quality = *total* params, speed = *active*
params), and streaming is what makes an over-RAM model viable — a dense model
bigger than RAM thrashes, an MoE touches only top-k experts so the hot set stays
cached. The full mechanism is now proven correct in pure MFL. The routing math
matches the reference (top-k by softmax prob, `norm_topk` renormalization, SwiGLU),
and streaming reads exactly the routed experts.

## Next (M7 proper)
Scale to **OLMoE-1B-7B-0924-Instruct** (7B total / 1.3B active, 64 experts, top-8,
the same model the original colibrì used for its "Stage A" validation):
- int8/int4 expert quantization (reuse `dot_q8`/`dot_q4`; the dense trunk stays resident).
- OLMoE tokenizer + qk-norm (OLMoE has `q_norm`/`k_norm`) + HF→bin converter.
- Verify vs a HF reference stream, then bench decode tok/s.
- **Runs LOCALLY** (rbm21 has only 4 GB free disk; local box has 42 GB + 16 GB RAM),
  CPU-capped, int4 (~3.5 GB).
- Headline target: a **7B-class model at ~1B speed on a 10 GB box, in pure MFL.**
