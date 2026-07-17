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

## Disruption: warm the fixed system prompt once (COLIBRI_WARM)

An agentic client (pi) resends the **same** multi-thousand-token system prompt on
every turn. The prefix cache already skips it *within* a process, but the **first**
turn of every server lifetime still pays the full cold prefill (~1–2 min). The
fix: **preload the system prompt's KV at boot** so *every* request — including the
first — only prefills the small user delta.

- `warm(prompt, np)` (engine) — batched-prefill a prefix into the KV cache and set
  `cache_buf`/`cache_len`, without generating.
- `COLIBRI_WARM=<file>` (server) — a file of comma-separated prompt token ids;
  preloaded at boot.
- `COLIBRI_LOGPROMPT=1` — `println`s each request's prompt ids, to capture pi's
  actual system prompt once (then feed it to `COLIBRI_WARM`).

Measured (local, throttled — relative comparison holds):
```
cold gen (203-token prefill)        : 12,873 ms
warm the prefix once (200 tok, boot):  12,422 ms   <- one-time
gen after warm (only 3-token delta) :     623 ms   <- EVERY request now pays this
```
So the fixed system-prompt cost is amortized to a **one-time boot prefill**, and
every pi turn's first token drops from ~13 s to ~0.6 s (only the user delta). This
is the disruption for agentic-client-on-CPU: decouple the constant prompt from
per-request latency.

### Persisting the warmed KV to disk (COLIBRI_KVSTORE)

The boot warm is a one-time ~2.4 min cost — but a *restart* re-pays it. So the
warmed KV is now **serialized to disk** and reloaded on the next boot:

- `save_kv(base)` / `load_kv(base, ids, n)` (engine) — dump/restore the
  `kcache`/`vcache` buffers + a meta (`cache_len` + prefix ids + dims) via the
  machin-core `write_file_raw`/`read_file_raw` builtins (machin PR #478).
- `COLIBRI_KVSTORE=<base>` (server) — on boot: if a snapshot exists and its ids +
  dims match the current `COLIBRI_WARM` prefix, **load it (ms)** and skip the
  prefill; else warm (once) and save the snapshot.

Validated across processes: run 1 warmed 200 tok + saved; run 2 (fresh process)
**`load_kv` = 80 ms** (vs ~12 s to re-warm), then delta-only gen 578 ms. On the
8k-ctx rbm21 deploy the snapshot is ~1.9 GB (`k`+`v`), loaded in a few seconds vs
the 144 s boot warm. So a redeploy/reboot no longer re-pays the prefill.

### Normalizing pi's per-request date (the real per-turn fix)

Warming the system prompt made the *first* turn fast, but pi still timed out on
*every* turn. Root cause, found in pi's `system-prompt.js`:

```js
// Add date and working directory last
prompt += `\nCurrent date: ${date}`;
prompt += `\nCurrent working directory: ${promptCwd}`;
```

pi appends a **fresh timestamp** to the end of the system prompt on every request.
That timestamp sits *right before the conversation* (the stable instructions end at
~2358 tokens, then the date, then the chat). Because the date changes each turn, the
prefix cache diverges at token ~2358 **every turn** — so the entire growing
conversation re-prefills each turn (852-token prefill measured on a mid-length chat →
exceeds pi's client timeout → "Error: Request timed out" then the retry answers off
the now-populated cache). The boot warm can't help: it only covers the stable head.

Fix: **normalize the date server-side.** `build_prompt_ids` rewrites
`Current date: <timestamp>` → a fixed `Current date: 2026-01-01T00:00:00.000Z`
before tokenizing (`norm_date`). The prompt is now byte-stable across turns, so the
full conversation caches and each turn prefills **only the new user message**. The
cwd line is left as-is (stable per session). Trade-off: the model sees a fixed date
(irrelevant for a coding agent). Deployed on `qwen3.service`.

Diagnostic recipe (reusable for any agentic client): `COLIBRI_LOGPROMPT=1`, capture
two consecutive real requests, diff their token ids — the **common-prefix length** is
exactly where the client injects per-request dynamic content. Warm up to there and, if
that dynamic field precedes the conversation, normalize it.

## Deploy
`serve_qwen3` on rbm21 (`qwen3.service`, :8094, `COLIBRI_CTX=8192`,
`COLIBRI_IDLE=600`). Build: `machin encode $MACHIN/framework/machweb.src
qwen3_engine.src qwen_tok.src serve_qwen3.src` (needs post-#474 machin for the
builtin). Converter `tools_qwen3_convert.py`, tokenizer `tools_qwen_tok.py`
(handles list-format merges).
