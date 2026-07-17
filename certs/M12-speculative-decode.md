# M12 — Speculative decoding (prompt-lookup, greedy) (2026-07-17)

The disruption for latency-bound CPU decode: get more than one token per weight
sweep. A cheap drafter proposes K tokens; the target verifies all K in **one
batched forward** that reads each weight once (the prefill path). Because colibri
decodes **greedily (argmax)**, greedy speculative decode is **byte-identical** to
plain decode — a pure, correctness-preserving speedup (verified: spec-on output
diffs clean against spec-off on every test).

## Why prompt-lookup drafting (no draft model)
On a memory-latency-bound target (M11), the batched verify of K positions costs
~K/prefill-rate, yielding up to K tokens. A 0.6B draft *model* would add its own
decode cost that eats the gain. **Prompt-lookup** (n-gram: find the last occurrence
of the current n-gram in the context, propose what followed) has **zero draft cost**
and hits well on the repetitive output agentic clients produce (code edits, echoed
context, structured lists). No match → K=0 → normal decode.

## Design
- `spec_draft(cur,pos)`: n-gram key = last `spec_ng` tokens; scan `spec_seq`
  (prompt+generated) backwards for the most recent match; propose the following
  up-to-`spec_k` tokens.
- `spec_verify(B,pos)`: embed [cur, draft…] at positions [pos..pos+B-1], `batch_layers`
  (causal, reuses KV[0..pos-1], fills KV[pos..]), then a **batched tied lm_head**
  (transposed output `[i*B+b]` so writes stay contiguous) + parallel per-position
  argmax → `spec_pred`.
- Accept the longest prefix where `spec_pred[j] == draft[j]`; the first mismatch's
  `spec_pred` is a free correct token. KV stays valid to the accepted length; the
  next step continues from there.
- Env: `COLIBRI_SPEC=<K>` (0=off, deployed at 8), `COLIBRI_SPEC_NG`, `COLIBRI_SPEC_STATS`.

## Measured (rbm21, Qwen3-1.7B int8, 12 threads, 2k ctx)
| Workload | baseline | spec=8 | acceptance |
|---|---|---|---|
| Code echo (150 tok) | 8.3 tok/s | (~15 tok/s spec steps) | 41/72 = 57% |
| Repeat-sentence-10× (199 tok) | **13.4 tok/s** | **16.8 tok/s (1.25×)** | 171/176 = **97%** |

Profile of a spec step (B=9): `fwd=358ms lm=62ms arg≈0`. The spec **decode** runs
~21.6 tok/s at high acceptance (9 tokens / 419 ms) vs ~13 plain → **~1.6× decode**;
end-to-end ~1.25× (prompt prefill is shared and unaffected). On novel/chat text,
no n-gram matches → falls back to plain decode (neutral). Partial-match-but-diverge
text can be slightly slower (a wasted B=K batch) — the one non-free case.

## Ceiling and the compounding path
The win is capped by the **batched-forward rate at small B**: prefill B=8 = 24 tok/s,
only **1.85×** decode, because the int8 dot kernel is compute-bound (1 byte = 1 MAC),
so batching only reclaims memory-stall time, not compute. A **vectorized AVX-VNNI
`dot_q8`** (`vpdpbusd`) would raise the batched-forward ceiling and **compound** with
speculation — that's the next lever if a bigger decode win is wanted. Speculation is
correct and shipped; the kernel is where the remaining 2× hides.
