# 🐦 machin-colibri

**A real LLM, in a language you've never heard of.** Run **Llama-3.2-1B-Instruct** on a laptop CPU at **20+ tokens/sec** — written in **pure [machin](https://github.com/javimosch/machin)** (MFL, a machine-first language), with **zero dependencies**: no BLAS, no llama.cpp, no PyTorch, no Python at runtime. One static binary. And it decodes **4× faster than the reference C** (`llama2.c`'s `runq`) it is verified token-for-token against.

Inspired by [JustVugg/colibri](https://github.com/JustVugg/colibri) — *tiny engine, immense model*, a 744B MoE streamed onto 25 GB of RAM. Same philosophy (one small engine, quantized weights, brutally honest numbers), aimed at the **opposite corner** of the design space: **small models, as fast as the silicon allows, on hardware you already own.**

```
$ colibri chat models/llama32-1b-q80.bin
  🐦 colibri — Llama-3.2-1B-Instruct · int8 · pure MFL · 0 deps
  ✓ loaded in 43ms (mmap)  ·  10 threads
  › what is the capital of France?
  ◆ Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also there.
  · 21.8 tok/s
```

## Why this exists

Everyone runs LLMs in C++/CUDA/Python. This one is written **entirely in a language most people have never heard of**, to answer a concrete question: *how fast can a real transformer go with a from-scratch engine and no numeric libraries at all?* The answer, on a weak CPU: **right up against the hardware roofline** — within ~7% of an optimal hand-tuned C GEMM, and 4× faster than the canonical reference C implementation.

No `libtorch`. No `ggml`. No `-lopenblas`. The int8/int4 matmul kernels, the KV cache, the tokenizer (both SentencePiece **and** a from-scratch tiktoken/BPE), the RoPE tables, the worker pool, the OpenAI-compatible HTTP server — all of it is MFL, compiled to a single native binary.

## Highlights

- **20+ tok/s** decoding Llama-3.2-1B-Instruct (int8) on a low-power desktop CPU (i5-13400T).
- **4× faster than the reference C** (`runq`), single-threaded, **token-identical** output (verified every step).
- **43 ms cold load** — the checkpoint is `mmap`'d, not read (was 31 s).
- **Prefix cache across requests** — a repeated system prompt goes **37 s → 0.8 s** (46×).
- **int4 support** — halves the model to 695 MB, stays coherent.
- **OpenAI-compatible server** — `/v1/chat/completions`, streaming (SSE), tool/function-calling, seeds. Drop-in for any OpenAI client.
- **Llama-3.2** (tiktoken BPE) **and** TinyLlama (SentencePiece), both in pure MFL.
- **Verified, honest, reproducible** — every speedup has a certification doc; every dead end is written down.

## The idea

A dense transformer's decode speed is set by one number: **bytes moved per token**. Every weight is read for every token, so a 1B int8 model moves ~1.1 GB/token, and on this box's DDR4 that caps decode at ~21 tok/s — no matter how clever the code. The job of the engine, then, is to *reach* that ceiling with nothing but the language:

- **Quantized weights, dequant-on-use** — Q8_0 (int8 + per-block scale) and an int4 variant. The whole quantized inner product is a single vectorized MFL builtin (`dot_q8` / `dot_q4`), contributed upstream to machin itself.
- **`mmap` the checkpoint** — the model is a memory-mapped file; load is a page-table setup, not a 1.3 GB read.
- **A worker pool over MFL channels** — rows of each matmul fan out across threads with a sliding-window dispatcher; barriers measured at 28 µs.
- **Batched prefill** — a tile of positions goes through each layer together, so each weight is loaded once (a GEMM, not N GEMVs). Bit-identical to per-token.
- **A prefix cache that survives across requests** — the expensive part of a long, repeated system prompt is computed once.

The result is an engine that is **already at the roofline**. What that means, and where it can't go further, is written up honestly in **[docs/PERFORMANCE-FRONTIER.md](docs/PERFORMANCE-FRONTIER.md)** — including every technique that *should* have helped and didn't (speculative decoding, gate sparsity, early-exit, continuous batching), each with the measurement that killed it.

## Honest numbers (rbm21 — Intel i5-13400T, 14 threads, DDR4)

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

## Quick start

```bash
# 1) build the machin compiler (once) — see github.com/javimosch/machin
# 2) convert a model to the colibri int8 format (HF -> ak44):
python3 tools_export_l3.py unsloth/Llama-3.2-1B-Instruct models/llama32-1b-q80.bin

# 3) compile the CLI (pure MFL -> C -> native binary):
machin encode engine.src cli.src > colibri.mfl
machin build colibri.mfl -o colibri

# 4) chat:
COLIBRI_THREADS=10 ./colibri chat models/llama32-1b-q80.bin
```

### Run it as an OpenAI-compatible server

```bash
# machweb is machin's std HTTP framework (ships with the machin repo)
machin encode $MACHIN/framework/machweb.src engine.src serve.src > serve.mfl
machin build serve.mfl -o colibri-serve
COLIBRI_THREADS=10 ./colibri-serve models/llama32-1b-q80.bin 8090

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

## The road ahead: streaming a 7B-class model onto a 10 GB box

The [performance frontier doc](docs/PERFORMANCE-FRONTIER.md) reaches a clean conclusion: on weak hardware a **dense** 1B model is at a fundamental wall, and the only lever that shrinks *both* compute and bandwidth at once lives in the **model architecture**, not the engine.

That is exactly what the original colibrì proved with its 744B MoE — and where this project is headed next. A **Mixture-of-Experts** model decouples quality (total parameters) from speed (the few *active* per token). **OLMoE-1B-7B** is 7B total but only **1.3B active** — 7B-class quality at the ~1B-dense cost this engine already hits at 20 tok/s. And because only a handful of experts fire per token, the cold ones can be **streamed from NVMe on demand** (the OS page cache is a free LRU), so total model size becomes bounded by *disk*, not RAM. The enabler — `mmap_file` — already shipped.

**Goal: a 7B-class model, on a 10 GB box, at the speed of a 1B — in pure MFL.**

## Related

- **[machin](https://github.com/javimosch/machin)** — the machine-first language this is written in. The int8/int4 kernel builtins (`dot_q8`, `dot_q4`, `dot_f32`, `axpy_f32`, `mmap_file`) were contributed upstream from this project.
- **[JustVugg/colibri](https://github.com/JustVugg/colibri)** — the streaming-MoE engine that inspired the name and the next milestone.
- **[docs/PERFORMANCE-FRONTIER.md](docs/PERFORMANCE-FRONTIER.md)** — the honest map of the wall.
- Certifications: **[M5 (Llama-3.2)](certs/M5-llama32.md)** · **[M6 (tools + systemd)](certs/M6-tools.md)**

---

*Pure MFL. Zero dependencies. One binary. Verified against the reference, token for token.*
