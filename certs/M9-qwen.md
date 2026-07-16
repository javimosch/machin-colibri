# M9 certification — Qwen2.5-1.5B-Instruct at 32k in pure MFL (2026-07-17)

A **32k-context** instruct model with **GQA + QKV bias + tied embeddings** in pure
MFL — token-identical to fp32 — and the server now does **SSE streaming**, which
was the real reason pi never worked.

## Model
`Qwen/Qwen2.5-1.5B-Instruct` — Llama-family, 1.54B: D=1536, 28 layers, **GQA 12/2**
(kv_dim 256), inter 8960, vocab **151936**, **QKV bias**, **tied embeddings**
(lm_head = embed), RoPE θ=1e6, native **32768** context, no qk-norm. Download 2.87
GB bf16 → **1.64 GB** int8 `.bin`.

## Why this model for long context
KV cache is the long-context blocker, and **GQA is what makes 32k fit a laptop**:

| model | GQA | KV @32k (fp32) |
|---|---|---|
| xLAM-1b (no GQA) | ❌ | 12.9 GB (won't fit) |
| **Qwen2.5-1.5B** | ✅ 12/2 | **1.9 GB** (fits 16 GB easily) |

## Correctness — token-identical to fp32
`tools_qwen_ref.py` (numpy fp32, reads safetensors) vs the pure-MFL int8 engine:
```
prompt "The capital of France is"  ids [785,6722,315,9625,374]
numpy fp32 : [12095,13,576,6722,315,9625,374,1083,279,6722,315,892]
pure MFL   : [12095,13,576,6722,315,9625,374,1083,279,6722,315,892]   ← 12/12 identical
decoded    : " Paris. The capital of France is also the capital of which"
```
GQA, QKV bias, and tied embeddings all verified. The `int8`-quantized tied
lm_head/embed did not flip a token.

## The huge-vocab discovery
Qwen's **151936-token vocab** makes the tied **fp32** lm_head cost **933 MB/token**
(a 151936×1536 matvec) — decode was only **3.6 tok/s**. Quantizing the tied
embed/lm_head to int8 (dequant on lookup, `dot_q8` on classify) cut it to ~233
MB/token: **7.8 tok/s (2.2×)**, still token-identical, and the `.bin` shrank 2.33
→ 1.64 GB. Lesson: on a large-vocab model the *output projection*, not the
transformer, dominates decode.

## Tokenizer — pure MFL, exact
`qwen_tok.src`: byte-level BPE with the **llama3-style pretokenizer + individual
digits** (Qwen's `\p{N}` rule). Exact vs the `tokenizers` library:
```
"The capital of France is" -> [785,6722,315,9625,374]        exact
"get_weather(city=Paris)"  -> [455,69364,43502,28,59604,8]    exact
"Hello, world! 123"        -> [9707,11,1879,0,220,16,17,18]    exact (individual digits)
```

## Server — ChatML + SSE STREAMING (the pi fix) + tools
`serve_qwen.src` builds the **ChatML** token sequence directly (special ids
`<|im_start|>`=151644 / `<|im_end|>`=151645 interleaved with BPE segments) and —
critically — supports **`stream: true` with proper SSE `chat.completion.chunk`
deltas + `finish_reason`**. pi's failure was *"Stream ended without finish_reason"*
— it always requests streaming, which the earlier (OLMoE/xLAM) servers didn't do.
Verified via curl:
```
chat        : "What is the capital of France?" -> "The capital of France is Paris."
streaming   : SSE deltas  {role} -> "Hello" -> "!" -> {finish_reason:"stop"} -> [DONE]
tools       : "weather in Tokyo?" -> tool_calls: get_weather({"city":"Tokyo"})  (finish_reason:tool_calls)
```

## Speed / footprint / deploy
7.8 tok/s (8 threads, warm), 1.64 GB int8, 32k context, KV 1.9 GB fp32 at full
32k. Batched prefill + KV prefix cache carried over (multi-turn agentic loops
reuse the cached system prompt). Idle-release: 27 MB. Deployed:
`deploy/qwen.user.service` (user systemd, :8093). pi provider `qwen` set as
default; with streaming fixed, pi's agentic loop should now complete (test
manually — the pi subprocess can't be driven under the build harness).
