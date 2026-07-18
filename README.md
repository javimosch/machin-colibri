# ⚒️ machin-anvil

**A real LLM — and the agent that runs on it — forged from scratch, in a language you've never heard of.** Run **Llama-3.2-1B-Instruct** (and Qwen2.5/3, xLAM, a 7B-class MoE) on a CPU you already own at **20+ tokens/sec** — written in **pure [machin](https://github.com/javimosch/machin)** (MFL, a machine-first language), with **zero dependencies**: no BLAS, no llama.cpp, no PyTorch, no Python at runtime. One static binary. It decodes **4× faster than the reference C** (`llama2.c`'s `runq`) it is verified token-for-token against — and it's fast enough to **back a real coding agent, self-hosted, with no cloud.**

> **Renamed from `machin-colibri` (2026-07).** This project began inspired by [JustVugg/colibri](https://github.com/JustVugg/colibri) — *tiny engine, immense model* — and borrowed its philosophy: one small engine, quantized weights, brutally honest numbers. It has since grown into its own thing: a from-scratch, zero-dependency runtime for a whole **family** of small models that now **serves agentic/coding workloads on your own hardware**. Hence **anvil** — forged in MFL, no libraries, and the anvil under the coding models it serves. The colibri lineage is kept, gratefully; the name reflects what it became.

```
$ anvil chat models/llama32-1b-q80.bin
  ⚒️ anvil — Llama-3.2-1B-Instruct · int8 · pure MFL · 0 deps
  ✓ loaded in 43ms (mmap)  ·  10 threads
  › what is the capital of France?
  ◆ Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also there.
  · 21.8 tok/s
```

## Why this exists

Everyone runs LLMs in C++/CUDA/Python. This one is written **entirely in a language most people have never heard of**, to answer a concrete question: *how fast can a real transformer go with a from-scratch engine and no numeric libraries at all — and what can you build on it?* The answer, on a weak CPU: **right up against the hardware roofline** (within ~7% of an optimal hand-tuned C GEMM, 4× faster than the canonical reference C) — fast enough to serve a self-hosted coding agent.

No `libtorch`. No `ggml`. No `-lopenblas`. The int8/int4 matmul kernels, the KV cache, the tokenizer (both SentencePiece **and** a from-scratch tiktoken/BPE), the RoPE tables, the worker pool, the OpenAI-compatible HTTP server — all of it is MFL, compiled to a single native binary.

## Highlights

- **20+ tok/s** decoding Llama-3.2-1B-Instruct (int8) on a low-power desktop CPU (i5-13400T).
- **4× faster than the reference C** (`runq`), single-threaded, **token-identical** output (verified every step).
- **43 ms cold load** — the checkpoint is `mmap`'d, not read (was 31 s).
- **Prefix cache across requests** — a repeated system prompt goes **37 s → 0.8 s** (46×).
- **int4 support** — halves the model to 695 MB, stays coherent.
- **OpenAI-compatible server** — `/v1/chat/completions`, streaming (SSE), tool/function-calling, seeds. Drop-in for any OpenAI client.
- **A whole family of models** — Llama-3.2, Qwen2.5-1.5B (32k), Qwen3-1.7B (thinking), xLAM-1b (function-calling), OLMoE-1B-7B (MoE) — all pure MFL.
- **Backs a self-hosted coding agent** — a measured, zero-cloud edit→run→verify loop (below).
- **Verified, honest, reproducible** — every speedup has a certification doc; every dead end is written down.

> **Config note:** runtime env knobs (thread count, idle release, context cap, …) currently use the `COLIBRI_*` prefix for historical reasons; `ANVIL_*` aliases are planned. The examples below use the working names.

## Forged for agents on your own hardware

The point of a fast, zero-dependency, OpenAI-compatible runtime is what it lets you *self-host*. anvil serves tool-calling models (Qwen2.5-Coder- / xLAM-class) that drive real agentic loops with **no cloud, no API key, no data leaving the box**.

In a measured spike, the **[tau](https://github.com/javimosch/tau)** coding agent — itself a from-scratch OpenAI-compatible client — drove a **Hammer (Qwen2.5-Coder-1.5B)** model *served by anvil* through a full **edit → run → verify** loop: given a bounded coding task, the model emitted a `write` tool call, tau executed it, ran the result with `bash`, and confirmed the output. **3/3 completions, ~17 s each, entirely on a 6-core CPU box** — pure MFL from the compiler, through the tokenizer and int8 inference, to the tool loop. The engine is done; the interesting frontier now is the *agent* it enables.

## The idea

A dense transformer's decode speed is set by one number: **bytes moved per token**. Every weight is read for every token, so a 1B int8 model moves ~1.1 GB/token, and on this box's DDR4 that caps decode at ~21 tok/s — no matter how clever the code. The job of the engine, then, is to *reach* that ceiling with nothing but the language:

- **Quantized weights, dequant-on-use** — Q8_0 (int8 + per-block scale) and an int4 variant. The whole quantized inner product is a single vectorized MFL builtin (`dot_q8` / `dot_q4`), contributed upstream to machin itself.
- **`mmap` the checkpoint** — the model is a memory-mapped file; load is a page-table setup, not a 1.3 GB read.
- **A worker pool over MFL channels** — rows of each matmul fan out across threads with a sliding-window dispatcher; barriers measured at 28 µs.
- **Batched prefill** — a tile of positions goes through each layer together, so each weight is loaded once (a GEMM, not N GEMVs). Bit-identical to per-token.
- **A prefix cache that survives across requests** — the expensive part of a long, repeated system prompt is computed once.

The result is an engine that is **already at the roofline**. What that means, and where it can't go further, is written up honestly in **[docs/PERFORMANCE-FRONTIER.md](docs/PERFORMANCE-FRONTIER.md)** — including every technique that *should* have helped and didn't (speculative decoding, gate sparsity, early-exit, continuous batching), each with the measurement that killed it.

**Docs site** (GitHub Pages): project home, roadmap to a stable OSS pair of agentic models at >20 tok/s on cheap hardware, and monthly changelog — [javimosch.github.io/machin-anvil](https://javimosch.github.io/machin-anvil/). Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Honest numbers (rbm21 — Intel i5-13400T, DDR4)

| metric | value |
|---|---|
| model on disk (int8 / Q8_0) | 1.3 GB |
| model on disk (int4) | 695 MB |
| cold load (mmap) | **43 ms** (was 31 s) |
| decode, int8 | **~21 tok/s** (≈ 23 GB/s, at the DDR ceiling) |
| decode vs reference C (`runq`, 1 thread) | **4× faster**, token-identical |
| prefill | 86 GF/s aggregate (optimal C GEMM on this box: 92) |
| repeated-system-prompt latency | 37 s → **0.8 s** (46×, prefix cache) |

This is not a frontier model. It is a **real instruction-tuned LLM answering correctly, fast, with an engine that has no right to be this small** — and a map of exactly where the hardware wall is.

## Benchmarks (measured — rbm21: Intel i5-13400T, AVX-VNNI, DDR4)

All pure MFL, int8, no dependencies. Numbers are real, on a cool box (not thermally throttled).

**Decode throughput**

| model | params (active) | context | tok/s |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | 32k | **11.7** |
| Qwen3-1.7B (thinking) | 1.7B | 40k | **12.7** |
| OLMoE-1B-7B (MoE) | 6.9B (1.3B) | 4k | 14.5¹ |

**Prefill** — the [`matmul_q8_batch`](https://github.com/javimosch/machin) builtin (contributed to machin core) vs a per-token loop:

| prompt tokens | per-token | **batched** | speedup |
|---|---|---|---|
| 256 | 15.2 s | **7.5 s** | 2.0× |
| 512 | 30.8 s | **15.2 s** | 2.0× |
| 1024 | 64.8 s | **31.4 s** | 2.1× |

**Prefix cache** — a repeated system prompt (the agentic-client pattern), end-to-end from a laptop over the network to rbm21, ~600-token system prompt:

| | latency |
|---|---|
| cold (first turn) | 9.5 s |
| warm (prefix cache) | **0.7 s** (13× — only the delta re-prefills) |

Every model above is **token-identical to its fp32 numpy reference** (int8 quantization did not change the greedy output). ¹OLMoE measured on the laptop.

## Quick start

```bash
# 1) build the machin compiler (once) — see github.com/javimosch/machin
# 2) convert a model to the anvil int8 format (HF -> ak44):
python3 tools_export_l3.py unsloth/Llama-3.2-1B-Instruct models/llama32-1b-q80.bin

# 3) compile the CLI (pure MFL -> C -> native binary):
machin encode engine.src cli.src > anvil.mfl
machin build anvil.mfl -o anvil

# 4) chat:
COLIBRI_THREADS=10 ./anvil chat models/llama32-1b-q80.bin
```

### Run it as an OpenAI-compatible server

```bash
# machweb is machin's std HTTP framework (ships with the machin repo)
machin encode $MACHIN/framework/machweb.src engine.src serve.src > serve.mfl
machin build serve.mfl -o anvil-serve
COLIBRI_THREADS=10 ./anvil-serve models/llama32-1b-q80.bin 8090

# then point any OpenAI client at http://localhost:8090/v1
curl -s localhost:8090/v1/chat/completions -d '{
  "model": "llama-3.2-1b-instruct-q80",
  "messages": [{"role":"user","content":"say hi in one word"}]
}'
```

Streaming (`"stream": true`), tool/function-calling, `seed`, `temperature`, and `top_p` all work. See **[certs/M6-tools.md](certs/M6-tools.md)**.

## What's implemented

- **Forward pass**: RMSNorm, RoPE (with llama3 frequency scaling), GQA attention, SwiGLU FFN, tied embeddings — all pure MFL.
- **Quantization**: Q8_0 (int8) and an int4 variant, dequant-on-use in the matmul.
- **Tokenizers**: SentencePiece (TinyLlama) **and** a from-scratch tiktoken/BPE (Llama-3.2), NUL-safe.
- **Server**: OpenAI-compatible `/v1/chat/completions` + `/v1/models` + `/health`, SSE streaming, prompt-based tool calling, prefix cache.
- **Speed**: mmap load, worker-pool matmul, batched prefill, `dot_q8`/`dot_q4` builtins (contributed to machin core).
- **Continuous batching primitive** — per-slot KV caches, one batched forward advances N requests (exact; see the frontier doc for why it's a fleet-throughput lever, not a latency one).

## A 7B-class MoE, in pure MFL — streamed onto a laptop ✅

The [performance frontier doc](docs/PERFORMANCE-FRONTIER.md) reaches a clean conclusion: on weak hardware a **dense** 1B model is at a fundamental wall, and the only lever that shrinks *both* compute and bandwidth at once lives in the **model architecture**, not the engine. So we followed the original colibrì's insight to its conclusion.

**OLMoE-1B-7B now runs in this engine, in pure MFL** ([cert](certs/M7-olmoe.md)). A Mixture-of-Experts model decouples quality (total parameters) from speed (the few *active* per token): OLMoE-1B-7B is **6.9B total but only 1.3B active** (64 experts/layer, top-8 routed). Only the routed experts are read each token, so the cold ones are **`mmap`-streamed on demand** — the OS page cache is a free per-layer LRU, and total model size is bounded by *disk*, not RAM.

```
$ COLIBRI_THREADS=8 ./olmoe models/olmoe-q8-lm8.bin
  OLMoE loaded: 64 experts · top-8 · 16 layers · pure MFL
  "The capital of France is Paris. The capital of the United States is Washington"
  14.5 tok/s · experts streamed in: 622/1024 · token-identical to fp32 reference
```

- **Token-identical to the numpy fp32 ground truth** across *every* quantization — int8, int4 experts, and int8 lm_head all reproduce fp32 12/12.
- **14.5 tok/s** on a laptop (8 cores), near the DDR ceiling for the active-param footprint.
- **Pick your tradeoff** — same engine, four measured configs ([cert](certs/M7-olmoe.md)):

  | experts | lm_head | size | tok/s |
  |---|---|---|---|
  | int8 | int8 | 7.65 GB | **14.5** (fastest) |
  | int4 | int8 | **4.43 GB** | 11.2 (smallest) |
  | int8 | fp32 | 7.96 GB | 9.4 |
  | int4 | fp32 | 4.74 GB | 7.4 |

- Same pure-MFL engine, same `dot_q8`/`dot_q4` kernels, same `mmap_file` streaming.
- **OpenAI-compatible server** — including the OLMoE byte-level BPE tokenizer, written in pure MFL. Verified with the official `openai` client:

  ```bash
  COLIBRI_THREADS=8 ./serve_olmoe models/olmoe-q8-lm8.bin 8091 models/olmoe-tok.bin
  # POST /v1/chat/completions -> "The capital of France is Paris. The capital of the United States is Washington, D.C."
  ```

**7B-class quality at ~1B speed, on hardware you own — in a language you'd never heard of.**

## Also: a 32k-context instruct model (Qwen2.5-1.5B) with streaming

**[Qwen2.5-1.5B-Instruct](certs/M9-qwen.md)** runs in pure MFL at **32k context** —
token-identical to fp32, **1.64 GB** int8. It adds **GQA** (which is what makes 32k
fit a 16 GB laptop: the KV cache is 1.9 GB at full context vs 12.9 GB without),
**QKV bias**, and **tied embeddings**. Its server does **real SSE streaming** +
ChatML + function-calling — so agentic clients that require streaming work.
Discovery: on a 152k-token vocab, the tied **fp32** lm_head cost 933 MB/token
(decode 3.6 tok/s); quantizing it to int8 → **7.8 tok/s** (2.2×), still
token-identical.

And **Qwen3-1.7B** — the newer generation, **40k context**, **thinking-capable**,
adds **per-head qk-norm** (reused from the OLMoE engine) to the same GQA + tied-embed
engine. 12.7 tok/s on rbm21. See [certs/M10-qwen3.md](certs/M10-qwen3.md).

## Also: function-calling agent models (xLAM-1b-fc-r, Qwen2.5-Coder / "Hammer")

The same engine family runs **[xLAM-1b-fc-r](certs/M8-xlam.md)** — a 1B
function-calling-specialized model (deepseek/Llama arch) in pure MFL,
token-identical to fp32, **1.62 GB** on disk — and **Hammer** (Qwen2.5-Coder-1.5B,
a tool-calling specialist), which is the model behind the self-hosted coding-agent
loop above. Their OpenAI servers do real tool calls (`"weather in Tokyo?"` →
`get_weather({"city":"Tokyo"})`), with byte-level BPE tokenizers written in MFL too.
**Batched prefill + a single-slot KV prefix cache** make multi-turn agentic clients
practical: a repeated system prompt goes **20.7 s → 2.0 s** (91× on the cache).
Honest note: batched prefill itself is only ~1.3× on this compute-balanced CPU (the
[balanced-roofline wall](docs/PERFORMANCE-FRONTIER.md) applies to prefill too) — the
prefix cache is what carries the multi-turn win.

## Related

- **[machin](https://github.com/javimosch/machin)** — the machine-first language this is written in. The int8/int4 kernel builtins (`dot_q8`, `dot_q4`, `dot_f32`, `axpy_f32`, `mmap_file`) were contributed upstream from this project.
- **[tau](https://github.com/javimosch/tau)** — the from-scratch OpenAI-compatible coding agent that anvil serves in the self-hosted loop above.
- **[JustVugg/colibri](https://github.com/JustVugg/colibri)** — the streaming-MoE engine that inspired this project's origin and its name-until-2026.
- **[docs/PERFORMANCE-FRONTIER.md](docs/PERFORMANCE-FRONTIER.md)** — the honest map of the wall.
- Certifications: **[M5 (Llama-3.2)](certs/M5-llama32.md)** · **[M6 (tools + systemd)](certs/M6-tools.md)**

---

*Pure MFL. Zero dependencies. One binary. Forged from scratch, verified against the reference, token for token.*
