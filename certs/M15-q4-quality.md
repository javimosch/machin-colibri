# M15 — Q4 quality investigation (toward production default) (2026-07-17)

Goal: make Q4 good enough to replace Q8 as the production default. Finding: Q4_0 is
already close, the cheap precision fix doesn't help, and the real gap is layer
accumulation.

## Weight error (relative Frobenius, GS=64)
| tensor | Q8 | Q4_0 | Q4_1 (asym) |
|---|---|---|---|
| embed_tokens | 0.006 | 0.108 | 0.090 |
| q_proj (L0) | 0.006 | 0.115 | 0.093 |
| down_proj (L0) | 0.006 | 0.112 | 0.092 |
| gate_proj (L14) | 0.006 | 0.114 | 0.093 |

Q4 is ~18× the error of Q8; Q4_1 (asymmetric min+scale) trims it ~18% but stays far
above Q8. **Weight error, though, does not track output quality** — transformers
absorb it.

## End-to-end greedy agreement (Q8 vs Q4_0, spec off, NOTHINK)
Word-for-word identical on: "0.9 vs 0.11", gold=Au, opposite-of-hot=cold, 15%-of-240
(same approach), train-speed (identical), factorial. **~6/7 prompts identical.**
The one miss: *"one-line Python sum of squares 1..10"* — Q8 gave the clean
`sum(i**2 for i in range(1,11))`; **Q4 emitted a buggy intermediate**
`sum(range(11)**2 for range in range(1,11))` then self-corrected to the right answer.
A close-logit token flip on a precision-sensitive (code) task.

## What did NOT fix it: Q8 embed/lm_head (`qwen3_q4e8_engine.src`)
Built a mixed variant — **Q8 tied embed/lm_head + Q4 layers** (1.1 GB). On the exact
failing prompt it produced the **identical** buggy output. So the flip does **not**
come from the output projection; it comes from **28 Q4 layers accumulating error in
the hidden state**. Mixed-precision-embed is not worth the +155 MB. (Variant kept for
reference; not adopted.)

## Where that leaves Q4 as default
- Q4_0: half size (924 MB), 1.5× decode, ~6/7 identical to Q8, occasional
  precision flips on code that often self-correct. Genuinely good, not Q8-equal.
- The real quality lever is **better layer quant**: Q4_1 (needs a new kernel with the
  per-group min term — `acc·wsc·xsc + min·xsc·Σx`; ~18% less error, uncertain flip
  fix) or calibration-based GPTQ/AWQ (bigger, needs data, closes more of the gap).
- int4 has a floor below Q8; no free lunch to full Q8 quality.

Decision pending: ship Q4_0 as default accepting the occasional flip, invest in Q4_1,
or keep Q8 default for the coding agent with Q4 as an opt-in fast mode.
