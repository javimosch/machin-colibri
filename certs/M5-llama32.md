# M5 certification — Llama-3.2-1B-Instruct in pure MFL (2026-07-11)

Model: unsloth/Llama-3.2-1B-Instruct (ungated mirror of meta-llama), converted
by `tools_export_l3.py` to the v4 format (ak44 = Q8_0 + RoPE header: theta,
llama3 scaling factors, original ctx; seq clamped to 4096). Tied embeddings,
GQA 32/8, vocab 128256. Max Q8_0 group error: 0.00468.

## Tokenizer (pure-MFL tiktoken)

Hand-rolled pretokenizer state machine for the llama3 regex (contractions,
letter/digit/punct/whitespace classes; \p{L}/\p{N} exact for
ASCII/Latin/Greek/Cyrillic/Hebrew/Arabic/kana/CJK/Hangul, rare scripts may
split differently) + lowest-rank BPE merges over hex-string symbols (tokens
may contain NUL bytes; hex keys keep the map exact).

**Exact match vs HF AutoTokenizer on the whole test corpus** (English, French
accents, contractions, code, digit triples, multi-space runs): 10/10 lines,
id-for-id. Harness: `tok_test` + `certs/` corpus.

## Forward pass vs HF transformers fp32 (greedy, plain prompts)

- "The capital of France is" → **32/32 tokens identical** to fp32 greedy
  ("Paris. The Eiffel Tower is located in Paris. The Louvre Museum ... Claude Mon…")
- "Once upon a time" → identical for 11 tokens, then a near-tie flip
  ("Tuscany" → "the countryside"); continuations stay semantically parallel
  ("there lived a young girl named Sophia. Sophia was a curious and adventurous…").

Divergence cause is Q8_0-vs-fp32 rounding on near-ties (industry-standard for
quantized inference; TinyLlama matched runq.c exactly because both were int8).
A wrong RoPE/scaling/tokenizer implementation garbles within a few tokens —
it does not match 32/32.

## Serving + glue (rbm21, i5-13400T, niced alongside a training job)

- `serve_l3` (same serve.src; template switches to llama3 header-id tokens):
  `{"content":"The capital of France is Paris.","usage":{...}}`
- ~15 tok/s generation under contention (nice -19, training job on 8 cores)
- tau (TAU_ENDPOINT → rbm21): streamed chat end-to-end
- roam (--api-base → rbm21): job done, goal "State the capital of Japan" →
  "[llm] Tokyo." finish_reason=stop, 129 tokens accounted

## Known limitations (disclosed)

- `tools`/function-calling not implemented in the server (chat only); tau's
  --tools loop needs server-side tool_calls formatting — future work.
- Chat template omits HF's date preamble ("Cutting Knowledge Date…") — a
  template choice, not a correctness issue.
- Model load is ~24s (byte-copy of 1.3GB through bytes_to_buf) — known gap,
  candidate for an mmap-style builtin.
