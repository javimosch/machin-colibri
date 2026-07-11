# Skills reliability eval — an honest negative result (2026-07-11)

**Question:** can a "skills" layer (curated instructions + few-shot examples
injected into the system prompt) make Llama-3.2-1B more reliable at tool
calling — "finetuning" the cheap model via context, no weight training?

**Answer for this model + these tasks: no. Skills hurt.** Recorded here so the
mechanism ships honestly and nobody re-discovers this the hard way.

## Setup

- Model: Llama-3.2-1B-Instruct (Q8_0), served by colibri-serve on rbm21 (idle).
- temp 0.7, seeds varied. `bench/eval_reliability.py`.
- **toolval**: over tool-requiring prompts, fraction that return a valid
  `tool_calls` (right tool, arguments parse as an object). N=21.
- **faithful**: after a tool result listing exactly 3 known files, fraction of
  answers that cite only those files (no hallucinated extras). N=14.

## Results

| Condition                              | faithful | toolval |
|----------------------------------------|----------|---------|
| **Baseline (M6 prompt + tolerant parser, no skills)** | 100%   | **90%** |
| Elaborate tool-use skill (with "do NOT" negative examples) | 78% | 52% |
| Minimal tool-use skill (positive-only, 2 lines)       | 100%   | 71%   |

## What we learned

1. **Negative examples poison small models.** The elaborate skill's "do NOT do
   this: `{"tool_call": "ls", ...}`" section *taught* the 1B model to sometimes
   emit exactly those wrong shapes — 90% → 52%. Removing them recovered most of
   the loss (52% → 71%). Never show a small model a labelled-wrong pattern.
2. **A saturated baseline has no headroom.** Even the clean minimal skill (71%)
   stayed below the no-skills baseline (90%). The M6 work (one clean example in
   the tool instruction + a tolerant parser) already made tool calling reliable;
   extra prompt material only gives a 1B model more ways to drift.
3. **Wrong-context injection is a footgun.** An early always-inject version put
   tool-call syntax into plain chat (no tools present), collapsing faithfulness
   to 0%. Fixed with `when: tools` / `when: tool_result` gating — but even
   correctly gated, skills didn't beat baseline.

## Disposition

The skills mechanism (loader, trigger match, context gating) is kept — it is
sound and would plausibly help a *weaker* baseline or a *bigger* model on
harder, non-saturated tasks. It is **off by default**; enable with
`COLIBRI_SKILLS=<dir>`. The experimental skills that produced these numbers live
in `bench/skills-experimental/`. Reproduce: `python3 bench/eval_reliability.py`.
