# M10 certification — Qwen3-1.7B in pure MFL (2026-07-17)

The newer Qwen generation in pure MFL: **GQA + per-head qk-norm + 40k context +
thinking**, on the same engine as Qwen2.5, plus the merged `matmul_q8_batch`
builtin for 2× prefill.

## Model
`Qwen/Qwen3-1.7B` — 1.72B: D=2048, 28 layers, **GQA 16/8** (kv_dim 1024), head_dim
128, **per-head qk-norm** (RMSNorm over head_dim=128), **no QKV bias**, tied
embeddings, RoPE θ=1e6, native **40960** context, thinking-capable. Sharded
safetensors (2 files). Download 3.4 GB bf16 → **1.83 GB** int8.

vs Qwen2.5-1.5B, the engine deltas: **add** per-head qk-norm (reused from the
OLMoE engine), **remove** the QKV bias. Everything else — GQA, tied int8 embed,
batched prefill, prefix cache, `matmul_q8_batch` — carries over.

## Correctness
`tools_qwen3_ref.py` (numpy fp32, sharded reader) vs the pure-MFL int8 engine:
```
prompt "The capital of France is"  ids [785,6722,315,9625,374]
numpy fp32 : [12095,13,576,6722,315,15344,374,21718,13,576,6722,315]  "Paris. The capital of Italy is Rome. The capital of"
pure MFL   : [12095,13,576,6722,315,  279,3639,4180,374,6515,  11,422]  "Paris. The capital of the United States is Washington, D"
```
First 5 tokens identical, both coherent + correct; token 6 diverges on an int8
near-tie (the model has many valid continuations of "…Paris. The capital of ___").
The 5-token exact prefix confirms **qk-norm is correct** (it affects every layer's
attention). q_norm weight shape `(128,)` confirmed per-head.

## Benchmarks (rbm21, i5-13400T, AVX-VNNI, 14 threads, cool)
- Decode: **12.7 tok/s** (thinking mode; correct — "2 plus 2 equals 4").
- Prefill uses the `matmul_q8_batch` builtin (machin core, merged): **2× over
  per-token** (256 tok 15.2→7.5 s, 1024 tok 64.8→31.4 s).
- KV cache: at 40k, GQA kv_dim=1024 → 9.4 GB fp32 (too big for rbm21's 9 GB), so
  the engine caps context via **`COLIBRI_CTX`** (deployed at 8k → 1.9 GB KV).

## Thinking + pi
Qwen3 answers pi's *"Current model does not support thinking"* warning — it emits
a reasoning trace by default (controllable via the chat template's
`enable_thinking`). Deployed on rbm21 `qwen3.service` :8094; pi's default provider
now points there. The streaming server (SSE), ChatML, and function-calling all
carry over from the Qwen2.5 server.

## Deploy
`serve_qwen3` on rbm21 (`qwen3.service`, :8094, `COLIBRI_CTX=8192`,
`COLIBRI_IDLE=600`). Build: `machin encode $MACHIN/framework/machweb.src
qwen3_engine.src qwen_tok.src serve_qwen3.src` (needs post-#474 machin for the
builtin). Converter `tools_qwen3_convert.py`, tokenizer `tools_qwen_tok.py`
(handles list-format merges).
